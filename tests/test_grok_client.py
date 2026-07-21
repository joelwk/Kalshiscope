import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from config import SearchConfig, Settings
from grok_client import GrokClient, _category_research_hint, _extract_json, _is_timeout_class_error, _is_retriable_grok_error
from models import Market, MarketOutcome, TradeDecision


class DummyChatSession:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = []

    def append(self, message):
        self.messages.append(message)

    def stream(self):
        yield None, SimpleNamespace(content=self.content)


class DummyChatClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.create_kwargs = None

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return DummyChatSession(self.content)


class DummyClient:
    def __init__(self, content: str) -> None:
        self.chat = DummyChatClient(content)


class FailingChatSession:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def append(self, message):
        return None

    def stream(self):
        raise self.error


class FailingChatClient:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.create_calls = 0

    def create(self, **kwargs):
        self.create_calls += 1
        return FailingChatSession(self.error)


class FailingClient:
    def __init__(self, error: Exception) -> None:
        self.chat = FailingChatClient(error)


class SequencedChatClient:
    def __init__(self, responses: list[Exception | str]) -> None:
        self.responses = responses
        self.create_calls = 0
        self.create_kwargs: list[dict] = []

    def create(self, **kwargs):
        self.create_calls += 1
        self.create_kwargs.append(kwargs)
        response = self.responses[min(self.create_calls - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            return FailingChatSession(response)
        return DummyChatSession(response)


class SequencedClient:
    def __init__(self, responses: list[Exception | str]) -> None:
        self.chat = SequencedChatClient(responses)


class TestGrokClient(unittest.TestCase):
    def test_extract_json(self) -> None:
        payload = _extract_json("prefix {\"foo\": 1} suffix")
        self.assertEqual(payload, {"foo": 1})

    def test_extract_json_invalid(self) -> None:
        with self.assertRaises(ValueError):
            _extract_json("no-json")

    def test_extract_json_recovers_truncated_object(self) -> None:
        """Brent deep-failure pattern: opens `{` but truncates mid-string without `}`."""
        truncated = (
            '{ "should_trade": true, "outcome": "YES", "confidence": 0.70, '
            '"reasoning": "Buffer supports edge from WSJ quote'
        )
        payload = _extract_json(truncated)
        self.assertTrue(payload["should_trade"])
        self.assertEqual(payload["outcome"], "YES")
        self.assertAlmostEqual(float(payload["confidence"]), 0.70)

    def test_is_retriable_grok_error_classifies_fast_internal(self) -> None:
        err = RuntimeError("StatusCode.INTERNAL: internal server error")
        self.assertTrue(_is_retriable_grok_error(err, duration_ms=350.0))
        self.assertFalse(_is_retriable_grok_error(err, duration_ms=20_000.0))

    def test_fast_empty_response_from_grok_is_retriable(self) -> None:
        """Empty responses under _EMPTY_RESPONSE_RETRY_MAX_MS are upstream blips."""
        err = ValueError("Empty response from Grok")
        self.assertTrue(_is_retriable_grok_error(err, duration_ms=350.0))
        self.assertTrue(_is_retriable_grok_error(err, duration_ms=4707.46))
        self.assertTrue(_is_retriable_grok_error(err, duration_ms=14_999.0))
        self.assertTrue(_is_retriable_grok_error(err, duration_ms=21_519.0))
        self.assertTrue(_is_retriable_grok_error(err, duration_ms=29_999.0))

    def test_slow_empty_response_from_grok_stays_non_retriable(self) -> None:
        """Empty responses at/above _EMPTY_RESPONSE_RETRY_MAX_MS stay non-retriable."""
        err = ValueError("Empty response from Grok")
        self.assertFalse(_is_retriable_grok_error(err, duration_ms=30_000.0))
        self.assertFalse(_is_retriable_grok_error(err, duration_ms=60_000.0))

    def test_grpc_stream_removed_is_retriable_even_when_slow(self) -> None:
        err = RuntimeError(
            '<_MultiThreadedRendezvous of RPC that terminated with:\n'
            '\tstatus = StatusCode.UNKNOWN\n'
            '\tdetails = "Stream removed"\n'
            '\tdebug_error_string = "UNKNOWN:Error received from peer  '
            '{grpc_message:"Stream removed", grpc_status:2}"\n>'
        )
        self.assertTrue(_is_retriable_grok_error(err, duration_ms=391.0))
        self.assertTrue(_is_retriable_grok_error(err, duration_ms=60_000.0))

    def test_grpc_deadline_exceeded_is_retriable_regardless_of_duration(self) -> None:
        err = RuntimeError(
            'StatusCode.DEADLINE_EXCEEDED\ndetails = "Deadline Exceeded"'
        )
        self.assertTrue(_is_timeout_class_error(err))
        self.assertTrue(_is_retriable_grok_error(err, duration_ms=60_000.0))
        self.assertTrue(_is_retriable_grok_error(err, duration_ms=120_000.0))

    def test_custom_stream_timeout_is_retriable(self) -> None:
        err = TimeoutError("Grok stream exceeded 60s for market KXTEST-123")
        self.assertTrue(_is_timeout_class_error(err))
        self.assertTrue(_is_retriable_grok_error(err, duration_ms=60_000.0))

    def test_non_deadline_slow_failure_is_not_retriable(self) -> None:
        err = RuntimeError("some random error")
        self.assertFalse(_is_timeout_class_error(err))
        self.assertFalse(_is_retriable_grok_error(err, duration_ms=60_000.0))

    def test_grpc_rst_stream_is_retriable_even_when_slow(self) -> None:
        err = RuntimeError(
            '_MultiThreadedRendezvous ... status = StatusCode.INTERNAL '
            'details = "Received RST_STREAM with error code 2"'
        )
        self.assertFalse(_is_timeout_class_error(err))
        self.assertTrue(_is_retriable_grok_error(err, duration_ms=19_000.0))
        self.assertTrue(_is_retriable_grok_error(err, duration_ms=60_000.0))

    def test_grpc_unavailable_is_retriable_even_when_slow(self) -> None:
        err = RuntimeError(
            'StatusCode.UNAVAILABLE details = "connection reset by peer"'
        )
        self.assertTrue(_is_retriable_grok_error(err, duration_ms=30_000.0))

    def test_stream_deadline_clamps_to_remaining_budget(self) -> None:
        client = GrokClient(api_key="x")
        client.stream_timeout_seconds = 90
        full = client._resolve_stream_deadline_seconds(None)
        clamped = client._resolve_stream_deadline_seconds(20_000.0)
        uncapped = client._resolve_stream_deadline_seconds(200_000.0)
        self.assertEqual(full, 90.0)
        self.assertLess(clamped, 20.0)
        self.assertGreater(clamped, 18.0)
        self.assertEqual(uncapped, 90.0)

    def test_stream_deadline_uses_weather_profile_override(self) -> None:
        """Cycle 1 review: weather analyses timed out at the legacy 100s
        ceiling; the new ``GROK_STREAM_TIMEOUT_SECONDS_WEATHER`` setting
        raises the per-attempt cap to 120s for the weather profile."""
        settings = Settings(
            GROK_STREAM_TIMEOUT_SECONDS=100,
            GROK_STREAM_TIMEOUT_SECONDS_WEATHER=120,
            GROK_STREAM_TIMEOUT_SECONDS_CRYPTO=120,
        )
        client = GrokClient(api_key="x", settings=settings)
        client.stream_timeout_seconds = 100

        weather_deadline = client._resolve_stream_deadline_seconds(
            None, search_profile="weather"
        )
        crypto_deadline = client._resolve_stream_deadline_seconds(
            None, search_profile="crypto"
        )
        generic_deadline = client._resolve_stream_deadline_seconds(
            None, search_profile=None
        )
        self.assertEqual(weather_deadline, 120.0)
        self.assertEqual(crypto_deadline, 120.0)
        self.assertEqual(generic_deadline, 100.0)

    def test_stream_deadline_weather_override_clamped_to_budget(self) -> None:
        """Even with a weather override, the remaining analysis budget
        still clamps the per-attempt deadline so a near-exhausted budget
        doesn't pretend it has 120s left."""
        settings = Settings(
            GROK_STREAM_TIMEOUT_SECONDS=100,
            GROK_STREAM_TIMEOUT_SECONDS_WEATHER=120,
        )
        client = GrokClient(api_key="x", settings=settings)
        client.stream_timeout_seconds = 100

        deadline = client._resolve_stream_deadline_seconds(
            30_000.0, search_profile="weather"
        )
        self.assertLess(deadline, 30.0)
        self.assertGreater(deadline, 28.0)

    def test_stream_deadline_weather_override_disabled_when_zero(self) -> None:
        """Setting GROK_STREAM_TIMEOUT_SECONDS_WEATHER=0 falls back to the
        global stream timeout."""
        settings = Settings(
            GROK_STREAM_TIMEOUT_SECONDS=100,
            GROK_STREAM_TIMEOUT_SECONDS_WEATHER=0,
        )
        client = GrokClient(api_key="x", settings=settings)
        client.stream_timeout_seconds = 100

        weather_deadline = client._resolve_stream_deadline_seconds(
            None, search_profile="weather"
        )
        self.assertEqual(weather_deadline, 100.0)

    def test_rpc_timeout_is_capped_to_stream_deadline(self) -> None:
        client = GrokClient(
            api_key="x",
            settings=Settings(
                XAI_CLIENT_TIMEOUT_SECONDS=120,
                GROK_STREAM_TIMEOUT_SECONDS=100,
            ),
        )

        self.assertEqual(client._resolve_rpc_timeout_seconds(100.0), 100.0)
        self.assertEqual(client._resolve_rpc_timeout_seconds(150.0), 120.0)
        self.assertEqual(client._resolve_rpc_timeout_seconds(0.25), 1.0)

    def test_recovered_retriable_analysis_attempt_does_not_log_error(self) -> None:
        market = Market(
            id="m-recovered-timeout",
            question="Will the minimum temperature be above 50F?",
            outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
        )
        content = (
            '{"should_trade": false, "outcome": "YES", "confidence": 0.5, '
            '"bet_size_pct": 0.0, "reasoning": "No durable edge.", '
            '"evidence_quality": 0.5}'
        )
        client = GrokClient(api_key="x")
        client.client = SequencedClient(
            [
                RuntimeError('StatusCode.DEADLINE_EXCEEDED details = "Deadline Exceeded"'),
                content,
            ]
        )

        with patch("grok_client.logger.error") as error_mock, patch(
            "grok_client.logger.warning"
        ) as warning_mock:
            decision = client.analyze_market(market)

        self.assertFalse(decision.should_trade)
        error_mock.assert_not_called()
        self.assertTrue(
            any(
                call.kwargs.get("data", {}).get("will_retry") is True
                for call in warning_mock.call_args_list
            )
        )

    def test_deep_analysis_retries_retriable_failure_and_attempts_fast_fallback(self) -> None:
        market = Market(
            id="m-deep-noretry",
            question="Will it rain?",
            outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
        )
        client = GrokClient(api_key="x")
        failing = FailingClient(
            RuntimeError('StatusCode.INTERNAL details = "Received RST_STREAM with error code 2"')
        )
        client.client = failing
        with self.assertRaises(RuntimeError):
            client.analyze_market_deep(market)
        self.assertEqual(failing.chat.create_calls, 3)

    def test_deep_analysis_fast_fallback_can_recover_without_previous_analysis(self) -> None:
        market = Market(
            id="m-deep-fast-fallback",
            question="Will it rain?",
            outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
        )
        content = (
            '{"should_trade": false, "outcome": "YES", "confidence": 0.5, '
            '"bet_size_pct": 0.0, "reasoning": "Fast fallback found no edge.", '
            '"evidence_quality": 0.5}'
        )
        client = GrokClient(api_key="x", settings=Settings(GROK_DEEP_ANALYSIS_MAX_ATTEMPTS=2))
        sequenced = SequencedClient(
            [
                RuntimeError('StatusCode.INTERNAL details = "Received RST_STREAM with error code 2"'),
                RuntimeError('StatusCode.INTERNAL details = "Received RST_STREAM with error code 2"'),
                content,
            ]
        )
        client.client = sequenced

        decision = client.analyze_market_deep(market)

        self.assertFalse(decision.should_trade)
        self.assertIn("retriable_fast_fallback", decision.reasoning)
        self.assertEqual(sequenced.chat.create_calls, 3)

    def test_deep_analysis_preserves_previous_on_retriable_retry_exhaustion(self) -> None:
        market = Market(
            id="m-deep-fallback",
            question="Will it rain?",
            outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
        )
        previous = TradeDecision(
            should_trade=False,
            outcome="YES",
            confidence=0.52,
            bet_size_pct=0.0,
            reasoning="First pass found no executable edge.",
            evidence_quality=0.8,
        )
        client = GrokClient(api_key="x", settings=Settings(GROK_DEEP_ANALYSIS_MAX_ATTEMPTS=2))
        failing = FailingClient(
            RuntimeError('StatusCode.INTERNAL details = "Received RST_STREAM with error code 2"')
        )
        client.client = failing

        decision = client.analyze_market_deep(market, previous_analysis=previous)

        self.assertFalse(decision.should_trade)
        self.assertIn("DeepAnalysisFallback", decision.reasoning)
        self.assertEqual(failing.chat.create_calls, 2)

    def test_unimplemented_deep_model_falls_back_to_fast_reasoning_model(self) -> None:
        market = Market(
            id="m-deep-unimplemented",
            question="Will it rain?",
            outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
        )
        content = (
            '{"should_trade": false, "outcome": "YES", "confidence": 0.5, '
            '"bet_size_pct": 0.0, "reasoning": "No durable edge.", '
            '"evidence_quality": 0.5}'
        )
        client = GrokClient(
            api_key="x",
            model="grok-4-1-fast-reasoning",
            model_deep="grok-4.3-latest",
        )
        sequenced = SequencedClient(
            [
                RuntimeError("StatusCode.UNIMPLEMENTED 404 model unavailable"),
                content,
            ]
        )
        client.client = sequenced

        decision = client.analyze_market_deep(market)

        self.assertFalse(decision.should_trade)
        self.assertEqual(sequenced.chat.create_calls, 2)
        self.assertEqual(sequenced.chat.create_kwargs[0]["model"], "grok-4.3-latest")
        self.assertEqual(
            sequenced.chat.create_kwargs[1]["model"],
            "grok-4-1-fast-reasoning",
        )

    def test_initial_analysis_still_retries_on_retriable_error(self) -> None:
        market = Market(
            id="m-initial-retry",
            question="Will it rain?",
            outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
        )
        client = GrokClient(api_key="x")
        failing = FailingClient(TimeoutError("Grok stream exceeded 90.0s for market m-initial-retry"))
        client.client = failing
        with self.assertRaises(TimeoutError):
            client.analyze_market(market)
        self.assertGreaterEqual(failing.chat.create_calls, 2)

    def test_analyze_market_stops_when_budget_exhausted(self) -> None:
        market = Market(
            id="m-budget",
            question="Will it rain?",
            outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
        )
        client = GrokClient(api_key="x")
        client.client = FailingClient(
            RuntimeError("StatusCode.INTERNAL: internal server error")
        )
        with patch("grok_client._DEFAULT_MAX_ANALYSIS_BUDGET_SECONDS", 0.0):
            with self.assertRaises(RuntimeError):
                client.analyze_market(market)

    def test_category_research_hint_weather_profile(self) -> None:
        hint = _category_research_hint("weather")
        self.assertIn("weather.gov", hint)
        self.assertIn("METAR/ASOS", hint)
        self.assertIn("GFS vs ECMWF", hint)
        self.assertIn("~72 hours", hint)

    def test_category_research_hint_commodities_market(self) -> None:
        market = Market(id="g1", question="Will gold close above 4600?", category="business")
        hint = _category_research_hint("generic", market=market)
        self.assertIn("Commodities guidance", hint)

    def test_category_research_hint_speech_profile(self) -> None:
        hint = _category_research_hint("speech")
        self.assertIn("Speech/event guidance", hint)
        self.assertIn("transcripts", hint)

    def test_category_research_hint_speech_mention_market(self) -> None:
        market = Market(
            id="KXGOVERNORMENTION-26APR09-OIL",
            question="Will the governor mention oil today?",
            category="politics",
        )
        hint = _category_research_hint("speech", market=market)
        self.assertIn("Word-mention guidance", hint)
        self.assertIn("scheduled events", hint)
        self.assertIn("vocabulary base rates", hint)

    def test_category_research_hint_music_profile(self) -> None:
        hint = _category_research_hint("music")
        self.assertIn("Music/streaming guidance", hint)
        self.assertIn("Spotify charts", hint)
        self.assertIn("Billboard", hint)

    def test_analyze_market_parses_markdown_fenced_json(self) -> None:
        market = Market(
            id="m1a",
            question="Will it rain?",
            outcomes=[
                MarketOutcome(name="YES", price=0.55),
                MarketOutcome(name="NO", price=0.45),
            ],
            liquidity_usdc=150.0,
        )
        content = """
        ```json
        {"should_trade": false, "outcome": "NO", "confidence": 0.56, "bet_size_pct": 0.0, "reasoning": "Implied prob: 55%, My prob: 44%, Edge: -11%"}
        ```
        """
        client = GrokClient(api_key="x")
        client.client = DummyClient(content)

        decision = client.analyze_market(market)
        self.assertFalse(decision.should_trade)
        self.assertEqual(decision.outcome, "NO")
        self.assertAlmostEqual(decision.confidence, 0.56)

    def test_analyze_market_repairs_single_quoted_keys(self) -> None:
        market = Market(
            id="m1b",
            question="Will it rain?",
            outcomes=[
                MarketOutcome(name="YES", price=0.55),
                MarketOutcome(name="NO", price=0.45),
            ],
            liquidity_usdc=150.0,
        )
        content = """
        {'should_trade': false, 'outcome': "NO", 'confidence': 0.52, 'bet_size_pct': 0.0, 'reasoning': "Implied prob: 55%, My prob: 50%, Edge: -5%"}
        """
        client = GrokClient(api_key="x")
        client.client = DummyClient(content)

        decision = client.analyze_market(market)
        self.assertFalse(decision.should_trade)
        self.assertEqual(decision.outcome, "NO")
        self.assertAlmostEqual(decision.confidence, 0.52)

    def test_analyze_market(self) -> None:
        market = Market(
            id="m1",
            question="Will it rain?",
            outcomes=[
                MarketOutcome(name="YES", price=0.55),
                MarketOutcome(name="NO", price=0.45),
            ],
            liquidity_usdc=150.0,
        )
        content = """
        {"should_trade": true, "outcome": "YES", "confidence": 0.8, "bet_size_pct": 0.5, "reasoning": "Implied prob: 55%, My prob: 70%, Edge: 15%", "implied_prob_external": 0.55, "my_prob": 0.70, "edge_external": 0.15, "evidence_quality": 0.8}
        """
        client = GrokClient(api_key="x")
        client.client = DummyClient(content)

        decision = client.analyze_market(market)
        self.assertTrue(decision.should_trade)
        self.assertEqual(decision.outcome, "YES")
        self.assertAlmostEqual(decision.confidence, 0.8)

        last_kwargs = client.client.chat.create_kwargs
        self.assertEqual(last_kwargs["model"], client.model)
        self.assertEqual(len(last_kwargs["tools"]), 2)
        self.assertIs(last_kwargs["response_format"], TradeDecision)
        self.assertEqual(last_kwargs["temperature"], 0.7)

    def test_self_consistency_runs_second_pass_and_averages_yes_probability(self) -> None:
        market = Market(
            id="m-self-consistency",
            question="Will it rain?",
            outcomes=[
                MarketOutcome(name="YES", price=0.55),
                MarketOutcome(name="NO", price=0.45),
            ],
            liquidity_usdc=500.0,
        )
        first = (
            '{"should_trade": true, "outcome": "YES", "confidence": 0.75, '
            '"probability_yes": 0.75, "bet_size_pct": 0.5, '
            '"reasoning": "Implied prob: 55%, My prob: 75%, Edge: 20%", '
            '"evidence_quality": 0.8, "key_sources": ["source A"], '
            '"base_rate_used": true}'
        )
        second = (
            '{"should_trade": true, "outcome": "YES", "confidence": 0.65, '
            '"probability_yes": 0.65, "bet_size_pct": 0.5, '
            '"reasoning": "Implied prob: 55%, My prob: 65%, Edge: 10%. Counter-evidence lowers this.", '
            '"evidence_quality": 0.8, "key_sources": ["source B"], '
            '"self_critique": "Recent base rate lowers probability."}'
        )
        client = GrokClient(api_key="x")
        sequenced = SequencedClient([first, second])
        client.client = sequenced

        decision = client.analyze_market(market)

        self.assertEqual(sequenced.chat.create_calls, 2)
        self.assertEqual(sequenced.chat.create_kwargs[0]["temperature"], 0.3)
        self.assertEqual(sequenced.chat.create_kwargs[1]["temperature"], 0.7)
        self.assertAlmostEqual(decision.probability_yes or 0.0, 0.70)
        self.assertLessEqual(decision.confidence, 0.70)
        self.assertIn("source A", decision.key_sources)
        self.assertIn("source B", decision.key_sources)
        self.assertIn("self_consistency_agreement", decision.reasoning)
        self.assertIn("Recent base rate", decision.self_critique or "")

    def test_self_consistency_disagreement_marks_deep_repair_required(self) -> None:
        market = Market(
            id="m-self-consistency-disagree",
            question="Will it rain?",
            outcomes=[
                MarketOutcome(name="YES", price=0.55),
                MarketOutcome(name="NO", price=0.45),
            ],
            liquidity_usdc=500.0,
        )
        first = (
            '{"should_trade": true, "outcome": "YES", "confidence": 0.78, '
            '"probability_yes": 0.78, "bet_size_pct": 0.5, '
            '"reasoning": "YES edge", "evidence_quality": 0.8}'
        )
        second = (
            '{"should_trade": false, "outcome": "NO", "confidence": 0.62, '
            '"probability_yes": 0.38, "bet_size_pct": 0.0, '
            '"reasoning": "Counter-evidence favors NO.", '
            '"evidence_quality": 0.8, "self_critique": "Sources conflict."}'
        )
        client = GrokClient(api_key="x")
        sequenced = SequencedClient([first, second])
        client.client = sequenced

        decision = client.analyze_market(market)

        self.assertEqual(sequenced.chat.create_calls, 2)
        self.assertFalse(decision.should_trade)
        self.assertTrue(decision.abstain)
        self.assertIn("self_consistency_disagreement", decision.reasoning)
        self.assertIn("deep repair required", decision.self_critique or "")
        # Disagreement merge must still persist YES-side polarity fields.
        self.assertAlmostEqual(decision.probability_yes or 0.0, 0.58, places=2)
        self.assertAlmostEqual(decision.my_prob or 0.0, 0.58, places=2)
        self.assertAlmostEqual(decision.implied_prob_external or 0.0, 0.55, places=2)

    def test_self_consistency_no_agreement_stores_yes_side_my_prob(self) -> None:
        market = Market(
            id="m-self-consistency-no",
            question="Will the high be 85-86?",
            outcomes=[
                MarketOutcome(name="YES", price=0.47),
                MarketOutcome(name="NO", price=0.53),
            ],
            liquidity_usdc=500.0,
        )
        first = (
            '{"should_trade": true, "outcome": "NO", "confidence": 0.78, '
            '"probability_yes": 0.22, "my_prob": 0.22, "bet_size_pct": 0.5, '
            '"reasoning": "NWS favors below bin. Implied YES 47%, my YES 22%.", '
            '"evidence_quality": 0.9, "primary_source_url": '
            '"https://forecast.weather.gov/MapClick.php?lat=33.76&lon=-84.43", '
            '"evidence_basis": "direct", "edge_source": "computed"}'
        )
        second = (
            '{"should_trade": true, "outcome": "NO", "confidence": 0.82, '
            '"probability_yes": 0.18, "my_prob": 0.18, "bet_size_pct": 0.5, '
            '"reasoning": "NWS still favors below bin after counter-check.", '
            '"evidence_quality": 0.9, "primary_source_url": '
            '"https://forecast.weather.gov/MapClick.php?lat=33.76&lon=-84.43", '
            '"evidence_basis": "direct", "edge_source": "computed", '
            '"self_critique": "Humidity residual uncertainty."}'
        )
        client = GrokClient(api_key="x")
        sequenced = SequencedClient([first, second])
        client.client = sequenced

        decision = client.analyze_market(market)

        self.assertEqual(sequenced.chat.create_calls, 2)
        self.assertEqual(decision.outcome, "NO")
        self.assertAlmostEqual(decision.probability_yes or 0.0, 0.20, places=2)
        # Must store P(YES), not chosen-side confidence (~0.80).
        self.assertAlmostEqual(decision.my_prob or 0.0, 0.20, places=2)
        self.assertAlmostEqual(decision.implied_prob_external or 0.0, 0.47, places=2)
        self.assertLess(decision.edge_external or 0.0, 0.0)

    def test_validate_and_enrich_normalizes_chosen_side_no_to_yes_side(self) -> None:
        market = Market(
            id="KXHIGHTATL-26JUL14-B85.5",
            question="Will the maximum temperature be 85-86 on Jul 14, 2026?",
            category="climate",
            outcomes=[
                MarketOutcome(name="YES", price=0.47),
                MarketOutcome(name="NO", price=0.53),
            ],
        )
        # ATL-shaped bug: model stored chosen-side my_prob=confidence with
        # positive edge_external on a NO call.
        decision = TradeDecision(
            should_trade=True,
            outcome="NO",
            confidence=0.82,
            raw_confidence=0.82,
            bet_size_pct=0.4,
            reasoning=(
                "NWS forecast.weather.gov as of Jul 14 favors high below 85. "
                "Implied NO 53%, my NO 82%, edge 29%."
            ),
            my_prob=0.82,
            implied_prob_external=0.53,
            edge_external=0.29,
            edge_source="computed",
            evidence_basis="direct",
            evidence_quality=1.0,
            primary_source_url="https://forecast.weather.gov/MapClick.php?lat=33.76&lon=-84.43",
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="weather",
        )
        self.assertAlmostEqual(validated.my_prob or 0.0, 0.18, places=2)
        self.assertAlmostEqual(validated.implied_prob_external or 0.0, 0.47, places=2)
        self.assertAlmostEqual(validated.edge_external or 0.0, -0.29, places=2)
        self.assertTrue(validated.should_trade)

    def test_self_consistency_skips_second_pass_below_thresholds(self) -> None:
        market = Market(
            id="m-no-self-consistency",
            question="Will it rain?",
            outcomes=[
                MarketOutcome(name="YES", price=0.55),
                MarketOutcome(name="NO", price=0.45),
            ],
            liquidity_usdc=100.0,
        )
        content = (
            '{"should_trade": false, "outcome": "YES", "confidence": 0.60, '
            '"probability_yes": 0.60, "bet_size_pct": 0.0, '
            '"reasoning": "Implied prob: 55%, My prob: 60%, Edge: 5%", '
            '"evidence_quality": 0.7}'
        )
        client = GrokClient(api_key="x")
        sequenced = SequencedClient([content])
        client.client = sequenced

        client.analyze_market(market)

        self.assertEqual(sequenced.chat.create_calls, 1)
        self.assertEqual(sequenced.chat.create_kwargs[0]["temperature"], 0.3)

    def test_self_consistency_second_pass_timeout_is_not_logged_as_error(self) -> None:
        market = Market(
            id="m-self-consistency-timeout",
            question="Will it rain?",
            outcomes=[
                MarketOutcome(name="YES", price=0.55),
                MarketOutcome(name="NO", price=0.45),
            ],
            liquidity_usdc=500.0,
        )
        first = (
            '{"should_trade": true, "outcome": "YES", "confidence": 0.75, '
            '"probability_yes": 0.75, "bet_size_pct": 0.5, '
            '"reasoning": "Implied prob: 55%, My prob: 75%, Edge: 20%", '
            '"evidence_quality": 0.8}'
        )
        timeout = RuntimeError('StatusCode.DEADLINE_EXCEEDED details = "Deadline Exceeded"')
        client = GrokClient(api_key="x")
        sequenced = SequencedClient([first, timeout])
        client.client = sequenced

        with patch("grok_client.logger.error") as error_mock, patch(
            "grok_client.logger.warning"
        ) as warning_mock:
            decision = client.analyze_market(market)

        self.assertEqual(sequenced.chat.create_calls, 2)
        self.assertTrue(decision.should_trade)
        error_mock.assert_not_called()
        self.assertTrue(
            any(
                call.kwargs.get("data", {}).get("self_consistency_variant") is True
                and call.kwargs.get("data", {}).get("will_retry") is False
                for call in warning_mock.call_args_list
            )
        )

    def test_tools_use_search_config(self) -> None:
        market = Market(
            id="m2",
            question="Will BTC be above $50k tomorrow?",
            outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
            liquidity_usdc=200.0,
        )
        content = """
        {"should_trade": false, "outcome": "NO", "confidence": 0.6, "bet_size_pct": 0.0, "reasoning": "test"}
        """
        search_config = SearchConfig(
            from_date=datetime(2026, 1, 13, 0, 0, tzinfo=timezone.utc),
            to_date=datetime(2026, 1, 13, 12, 0, tzinfo=timezone.utc),
            allowed_domains=["example.com"],
            allowed_x_handles=["Foo"],
        )
        client = GrokClient(api_key="x", search_config=search_config)
        client.client = DummyClient(content)

        captured = {}

        def fake_web_search(*args, **kwargs):
            captured["web"] = kwargs
            return {"tool": "web"}

        def fake_x_search(*args, **kwargs):
            captured["x"] = kwargs
            return {"tool": "x"}

        with patch("xai_provider.web_search", side_effect=fake_web_search), patch(
            "xai_provider.x_search", side_effect=fake_x_search
        ):
            client.analyze_market(market)

        self.assertEqual(captured["web"]["allowed_domains"], ["example.com"])
        self.assertEqual(captured["x"]["from_date"], datetime(2026, 1, 13, 0, 0, tzinfo=timezone.utc))
        self.assertEqual(captured["x"]["to_date"], datetime(2026, 1, 13, 12, 0, tzinfo=timezone.utc))
        self.assertEqual(captured["x"]["allowed_x_handles"], ["Foo"])
        self.assertFalse(captured["x"]["enable_image_understanding"])
        self.assertFalse(captured["x"]["enable_video_understanding"])

    def test_tools_respect_search_config_source_caps(self) -> None:
        market = Market(
            id="m2-cap",
            question="Will BTC be above $50k tomorrow?",
            outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
            liquidity_usdc=200.0,
        )
        content = """
        {"should_trade": false, "outcome": "NO", "confidence": 0.6, "bet_size_pct": 0.0, "reasoning": "test"}
        """
        search_config = SearchConfig(
            from_date=datetime(2026, 1, 13, 0, 0, tzinfo=timezone.utc),
            to_date=datetime(2026, 1, 13, 12, 0, tzinfo=timezone.utc),
            allowed_domains=["a.com", "b.com", "c.com", "d.com"],
            allowed_x_handles=["A", "B", "C", "D", "E"],
            max_allowed_domains=3,
            max_allowed_x_handles=4,
        )
        client = GrokClient(api_key="x", search_config=search_config)
        client.client = DummyClient(content)

        captured = {}

        def fake_web_search(*args, **kwargs):
            captured["web"] = kwargs
            return {"tool": "web"}

        def fake_x_search(*args, **kwargs):
            captured["x"] = kwargs
            return {"tool": "x"}

        with patch("xai_provider.web_search", side_effect=fake_web_search), patch(
            "xai_provider.x_search", side_effect=fake_x_search
        ):
            client.analyze_market(market)

        self.assertEqual(
            captured["web"]["allowed_domains"],
            ["a.com", "b.com", "c.com"],
        )
        self.assertEqual(captured["x"]["allowed_x_handles"], ["A", "B", "C", "D"])

    def test_tools_clamp_search_sources_to_xai_caps(self) -> None:
        market = Market(
            id="m2-provider-cap",
            question="Will BTC be above $50k tomorrow?",
            outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
            liquidity_usdc=200.0,
        )
        content = """
        {"should_trade": false, "outcome": "NO", "confidence": 0.6, "bet_size_pct": 0.0, "reasoning": "test"}
        """
        search_config = SearchConfig(
            from_date=datetime(2026, 1, 13, 0, 0, tzinfo=timezone.utc),
            to_date=datetime(2026, 1, 13, 12, 0, tzinfo=timezone.utc),
            allowed_domains=["a.com", "b.com", "c.com", "d.com", "e.com", "f.com"],
            allowed_x_handles=[
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "K",
            ],
            max_allowed_domains=8,
            max_allowed_x_handles=14,
        )
        client = GrokClient(api_key="x", search_config=search_config)
        client.client = DummyClient(content)

        captured = {}

        def fake_web_search(*args, **kwargs):
            captured["web"] = kwargs
            return {"tool": "web"}

        def fake_x_search(*args, **kwargs):
            captured["x"] = kwargs
            return {"tool": "x"}

        with patch("xai_provider.web_search", side_effect=fake_web_search), patch(
            "xai_provider.x_search", side_effect=fake_x_search
        ):
            client.analyze_market(market)

        self.assertEqual(
            captured["web"]["allowed_domains"],
            ["a.com", "b.com", "c.com", "d.com", "e.com"],
        )
        self.assertEqual(
            captured["x"]["allowed_x_handles"],
            ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        )

    def test_analyze_market_deep_enables_multimedia_for_borderline(self) -> None:
        market = Market(
            id="m3",
            question="Will ETH break $3k?",
            outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
            liquidity_usdc=300.0,
        )
        previous = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.6,
            bet_size_pct=0.3,
            reasoning="prev",
        )
        content = """
        {"should_trade": true, "outcome": "YES", "confidence": 0.7, "bet_size_pct": 0.4, "reasoning": "test"}
        """
        search_config = SearchConfig(
            from_date=datetime(2026, 1, 13, 0, 0, tzinfo=timezone.utc),
            to_date=datetime(2026, 1, 13, 12, 0, tzinfo=timezone.utc),
            allowed_domains=["example.com"],
            allowed_x_handles=["Foo"],
            multimedia_confidence_range=(0.55, 0.75),
        )
        client = GrokClient(api_key="x", search_config=search_config)
        client.client = DummyClient(content)

        captured = {}

        def fake_web_search(*args, **kwargs):
            captured["web"] = kwargs
            return {"tool": "web"}

        def fake_x_search(*args, **kwargs):
            captured["x"] = kwargs
            return {"tool": "x"}

        with patch("xai_provider.web_search", side_effect=fake_web_search), patch(
            "xai_provider.x_search", side_effect=fake_x_search
        ):
            client.analyze_market_deep(market, previous_analysis=previous)

        self.assertTrue(captured["x"]["enable_image_understanding"])
        self.assertTrue(captured["x"]["enable_video_understanding"])

    def test_analyze_market_deep_merges_partial_payload(self) -> None:
        market = Market(
            id="m6",
            question="Will home team win?",
            outcomes=[
                MarketOutcome(name="YES", price=0.60),
                MarketOutcome(name="NO", price=0.40),
            ],
            liquidity_usdc=120.0,
        )
        previous = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.65,
            bet_size_pct=0.4,
            reasoning="Prior reasoning",
            implied_prob_external=0.58,
            my_prob=0.65,
            edge_external=0.07,
            evidence_quality=0.7,
        )
        content = """
        {"implied_prob_external": 0.60, "my_prob": 0.63, "edge_external": 0.03, "evidence_quality": 0.9}
        """
        client = GrokClient(api_key="x")
        client.client = DummyClient(content)

        decision = client.analyze_market_deep(market, previous_analysis=previous)

        self.assertFalse(decision.should_trade)
        self.assertEqual(decision.outcome, "YES")
        self.assertAlmostEqual(decision.confidence, 0.63)
        self.assertEqual(decision.bet_size_pct, 0.0)
        self.assertAlmostEqual(decision.edge_external, 0.03)
        self.assertIn("Validated", decision.reasoning)

    def test_analyze_market_deep_normalizes_percent_like_fields(self) -> None:
        market = Market(
            id="m12",
            question="Will event happen?",
            outcomes=[MarketOutcome(name="YES", price=0.50), MarketOutcome(name="NO", price=0.50)],
            liquidity_usdc=120.0,
        )
        content = """
        {"should_trade": false, "outcome": "YES", "confidence": 74, "bet_size_pct": 0.0, "reasoning": "Edge: -160%", "edge_external": -1.6}
        """
        client = GrokClient(api_key="x")
        client.client = DummyClient(content)

        decision = client.analyze_market_deep(market)
        self.assertAlmostEqual(decision.confidence, 0.74, places=6)
        self.assertAlmostEqual(decision.edge_external or 0.0, 0.0, places=6)

    def test_normalize_preserves_probability_boundary_one_point_zero(self) -> None:
        client = GrokClient(api_key="x")
        payload = {"confidence": 1.0, "my_prob": 1.0}
        with patch("grok_client.logger.warning") as warning_mock:
            normalized = client._normalize_numeric_fields(payload, market_id="m-boundary-prob")
        self.assertEqual(normalized["confidence"], 1.0)
        self.assertEqual(normalized["my_prob"], 1.0)
        warning_mock.assert_not_called()

    def test_normalize_preserves_edge_boundary_negative_one_point_zero(self) -> None:
        client = GrokClient(api_key="x")
        payload = {"edge_external": -1.0}
        normalized = client._normalize_numeric_fields(payload, market_id="m-boundary-edge")
        self.assertEqual(normalized["edge_external"], -1.0)

    def test_normalize_still_converts_percentages_above_one(self) -> None:
        client = GrokClient(api_key="x")
        payload = {"my_prob": 50, "edge_external": -1.07}
        normalized = client._normalize_numeric_fields(payload, market_id="m-percent-convert")
        self.assertAlmostEqual(float(normalized["my_prob"]), 0.5, places=6)
        self.assertAlmostEqual(float(normalized["edge_external"]), -0.0107, places=6)

    def test_normalize_edge_near_boundary_logs_debug(self) -> None:
        client = GrokClient(api_key="x")
        payload = {"edge_external": -1.02}
        with patch("grok_client.logger.debug") as debug_mock, patch(
            "grok_client.logger.warning"
        ) as warning_mock:
            normalized = client._normalize_numeric_fields(
                payload,
                market_id="m-edge-debug-boundary",
            )
        self.assertAlmostEqual(float(normalized["edge_external"]), -0.0102, places=6)
        debug_mock.assert_called_once()
        warning_mock.assert_not_called()

    def test_normalize_edge_above_one_point_five_still_warns(self) -> None:
        client = GrokClient(api_key="x")
        payload = {"edge_external": -1.8}
        with patch("grok_client.logger.debug") as debug_mock, patch(
            "grok_client.logger.warning"
        ) as warning_mock:
            normalized = client._normalize_numeric_fields(
                payload,
                market_id="m-edge-warning",
            )
        self.assertAlmostEqual(float(normalized["edge_external"]), -0.018, places=6)
        warning_mock.assert_called_once()
        debug_mock.assert_not_called()

    def test_analyze_market_deep_normalizes_edge_at_negative_one_boundary(self) -> None:
        market = Market(
            id="m12-boundary",
            question="Will event happen?",
            outcomes=[MarketOutcome(name="YES", price=0.50), MarketOutcome(name="NO", price=0.50)],
            liquidity_usdc=120.0,
        )
        content = """
        {"should_trade": false, "outcome": "YES", "confidence": 0.60, "bet_size_pct": 0.0, "reasoning": "Edge: -1%", "edge_external": -1.0}
        """
        client = GrokClient(api_key="x")
        client.client = DummyClient(content)

        decision = client.analyze_market_deep(market)
        self.assertAlmostEqual(decision.edge_external or 0.0, 0.0, places=6)

    def test_analyze_market_deep_retains_previous_likelihood_ratio_when_missing(self) -> None:
        market = Market(
            id="m16",
            question="Will event happen?",
            outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
            liquidity_usdc=120.0,
        )
        previous = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.66,
            bet_size_pct=0.3,
            reasoning="Previous analysis",
            likelihood_ratio=1.8,
        )
        content = """
        {"should_trade": true, "outcome": "YES", "confidence": 0.68, "bet_size_pct": 0.35, "reasoning": "updated", "likelihood_ratio": null}
        """
        client = GrokClient(api_key="x")
        client.client = DummyClient(content)

        decision = client.analyze_market_deep(market, previous_analysis=previous)
        self.assertAlmostEqual(decision.likelihood_ratio or 0.0, 1.8, places=6)

    def test_analyze_market_clamps_overscaled_confidence(self) -> None:
        market = Market(
            id="m13",
            question="Will event happen?",
            outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
            liquidity_usdc=150.0,
        )
        content = """
        {"should_trade": false, "outcome": "YES", "confidence": 340, "bet_size_pct": 0.0, "reasoning": "Implied prob: 55%, My prob: 340%, Edge: 285%"}
        """
        client = GrokClient(api_key="x")
        client.client = DummyClient(content)

        decision = client.analyze_market(market)
        self.assertAlmostEqual(decision.confidence, 1.0, places=6)

    def test_should_enable_multimedia_urgent_market(self) -> None:
        market = Market(
            id="m4",
            question="Will event happen?",
            outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
            close_time=datetime.now(timezone.utc) + timedelta(hours=12),
        )
        client = GrokClient(api_key="x")
        self.assertTrue(
            client._should_enable_multimedia(
                market,
                decision=None,
                config=SearchConfig(),
            )
        )

    def test_should_enable_multimedia_for_speech_profile(self) -> None:
        market = Market(
            id="KXCARNEYMENTION-26APR08-ROCK",
            question="Will Carney say rocket?",
            outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
            close_time=datetime.now(timezone.utc) + timedelta(days=3),
        )
        client = GrokClient(api_key="x")
        self.assertTrue(
            client._should_enable_multimedia(
                market,
                decision=None,
                config=SearchConfig(profile_name="speech"),
            )
        )

    def test_validate_and_enrich_decision_downgrades_bad_evidence(self) -> None:
        market = Market(
            id="m5",
            question="Will event happen?",
            outcomes=[MarketOutcome(name="YES", price=0.62), MarketOutcome(name="NO", price=0.38)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.65,
            bet_size_pct=0.5,
            reasoning="base",
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="generic",
        )
        self.assertFalse(validated.should_trade)
        self.assertLess(validated.evidence_quality, 0.45)
        self.assertFalse(validated.abstain)

    def test_validate_and_enrich_sets_abstain_for_very_low_evidence(self) -> None:
        market = Market(
            id="m5a",
            question="Will event happen?",
            outcomes=[MarketOutcome(name="YES", price=0.62), MarketOutcome(name="NO", price=0.38)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.65,
            bet_size_pct=0.5,
            reasoning="No search results. No evidence found.",
            evidence_quality=0.0,
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="generic",
        )
        self.assertFalse(validated.should_trade)
        self.assertFalse(validated.abstain)

    def test_validate_and_enrich_sets_abstain_for_double_blind_information_gap(self) -> None:
        market = Market(
            id="m5b",
            question="Will event happen?",
            outcomes=[MarketOutcome(name="YES", price=0.62), MarketOutcome(name="NO", price=0.38)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.66,
            bet_size_pct=0.4,
            reasoning=(
                "No external odds found. Implied probability: unknown. "
                "No search results. No evidence found."
            ),
            evidence_quality=0.0,
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="generic",
        )
        self.assertTrue(validated.abstain)
        self.assertFalse(validated.should_trade)
        self.assertIn("abstain_double_blind_information_gap", validated.reasoning)

    def test_validate_and_enrich_caps_evidence_when_no_external_odds_found(self) -> None:
        market = Market(
            id="m9",
            question="Will event happen?",
            outcomes=[MarketOutcome(name="YES", price=0.50), MarketOutcome(name="NO", price=0.50)],
        )
        decision = TradeDecision(
            should_trade=False,
            outcome="YES",
            confidence=0.62,
            bet_size_pct=0.0,
            reasoning="No external odds found. Implied prob: unknown. My prob: 62%.",
            implied_prob_external=0.50,
            my_prob=0.62,
            edge_external=0.12,
            evidence_quality=0.9,
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="generic",
        )
        self.assertLessEqual(validated.evidence_quality, 0.5)

    def test_validate_and_enrich_treats_wire_preview_as_proxy_without_floor(self) -> None:
        market = Market(
            id="m9-preview",
            question="Will Team A win?",
            outcomes=[MarketOutcome(name="YES", price=0.50), MarketOutcome(name="NO", price=0.50)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.62,
            bet_size_pct=0.3,
            reasoning=(
                "Reuters preview notes the matchup and probable starters. "
                "No external odds found. Implied prob: unknown. My prob: 62%. Edge: 12%."
            ),
            implied_prob_external=None,
            my_prob=0.62,
            edge_external=0.12,
            edge_source="fallback",
            evidence_quality=0.10,
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="sports",
        )
        self.assertEqual(validated.source_match_class, "preview_or_proxy")
        self.assertEqual(validated.evidence_basis, "proxy")
        self.assertIsNone(validated.evidence_quality_floor_applied)
        self.assertEqual(
            validated.evidence_floor_suppressed_reason,
            "preview_or_proxy_source",
        )
        self.assertLessEqual(validated.evidence_quality, 0.50)
        # Edge (0.12) is below the high-edge proxy participation floor (0.15),
        # so a no-direct-source preview remains blocked.
        self.assertFalse(validated.should_trade)

    def test_validate_and_enrich_allows_high_edge_proxy_via_override(self) -> None:
        market = Market(
            id="m9-highedge",
            question="Will Team A win?",
            outcomes=[MarketOutcome(name="YES", price=0.50), MarketOutcome(name="NO", price=0.50)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.70,
            bet_size_pct=0.3,
            reasoning=(
                "Reuters preview notes the matchup and probable starters. "
                "No external odds found. Implied prob: unknown. My prob: 70%. Edge: 20%."
            ),
            implied_prob_external=None,
            my_prob=0.70,
            edge_external=0.20,
            edge_source="fallback",
            evidence_quality=0.9,
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="sports",
        )
        # Proxy + fallback edge would normally be blocked, but a market edge at
        # or above the participation floor (0.15) lets it pass validation so the
        # downstream edge gate and family sizing can size it.
        self.assertEqual(validated.source_match_class, "preview_or_proxy")
        self.assertEqual(validated.evidence_basis, "proxy")
        self.assertTrue(validated.should_trade)

    def test_validate_and_enrich_blocks_high_edge_proxy_below_floor(self) -> None:
        market = Market(
            id="m9-lowedge",
            question="Will Team A win?",
            outcomes=[MarketOutcome(name="YES", price=0.50), MarketOutcome(name="NO", price=0.50)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.60,
            bet_size_pct=0.3,
            reasoning=(
                "Reuters preview notes the matchup and probable starters. "
                "No external odds found. Implied prob: unknown. My prob: 60%. Edge: 10%."
            ),
            implied_prob_external=None,
            my_prob=0.60,
            edge_external=0.10,
            edge_source="fallback",
            evidence_quality=0.9,
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="sports",
        )
        # Edge (0.10) is below the participation floor (0.15), so the proxy block
        # still applies even though evidence quality clears the cap.
        self.assertEqual(validated.source_match_class, "preview_or_proxy")
        self.assertFalse(validated.should_trade)

    def test_validate_and_enrich_generic_requires_higher_proxy_edge(self) -> None:
        market = Market(
            id="KXGENERIC-TEST",
            question="Will the Fed cut rates?",
            category="economics",
            outcomes=[MarketOutcome(name="YES", price=0.50), MarketOutcome(name="NO", price=0.50)],
        )
        # Mirror the sports high-edge proxy fixture so source_match_class is
        # preview_or_proxy and the strong_proxy_edge_override path is exercised.
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.66,
            bet_size_pct=0.3,
            reasoning=(
                "Reuters preview notes the meeting outlook and probable path. "
                "No external odds found. Implied prob: unknown. My prob: 66%. Edge: 16%."
            ),
            implied_prob_external=None,
            my_prob=0.66,
            edge_external=0.16,
            edge_source="fallback",
            evidence_quality=0.9,
        )
        client = GrokClient(api_key="x")
        # 0.16 clears the global 0.15 floor but not GENERIC_PROXY_HIGH_EDGE_MIN=0.18.
        blocked = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="generic",
        )
        self.assertEqual(blocked.source_match_class, "preview_or_proxy")
        self.assertFalse(blocked.should_trade)
        allowed = client._validate_and_enrich_decision(
            market,
            decision.model_copy(
                update={
                    "confidence": 0.70,
                    "my_prob": 0.70,
                    "edge_external": 0.20,
                    "reasoning": (
                        "Reuters preview notes the meeting outlook and probable path. "
                        "No external odds found. Implied prob: unknown. My prob: 70%. Edge: 20%."
                    ),
                }
            ),
            profile_name="generic",
        )
        self.assertTrue(allowed.should_trade)

    def test_validate_and_enrich_prefers_computed_edge_over_reasoning_text(self) -> None:
        market = Market(
            id="m10",
            question="WTA: Jones vs Stearns",
            outcomes=[MarketOutcome(name="Jones", price=0.278), MarketOutcome(name="Stearns", price=0.726)],
        )
        decision = TradeDecision(
            should_trade=False,
            outcome="Stearns",
            confidence=0.72,
            bet_size_pct=0.0,
            reasoning="Implied prob: 0.726, My prob: 0.72, Edge: 0.72 - 0.726 = -0.006",
            implied_prob_external=0.726,
            my_prob=0.72,
            edge_external=None,
            evidence_quality=0.0,
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="sports",
        )
        self.assertAlmostEqual(validated.edge_external or 0.0, -0.006, places=6)

    def test_validate_and_enrich_uses_market_edge_for_trade_gate(self) -> None:
        market = Market(
            id="m14",
            question="Will Team A win?",
            outcomes=[MarketOutcome(name="Team A", price=0.58), MarketOutcome(name="Team B", price=0.42)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="Team A",
            confidence=0.66,
            bet_size_pct=0.4,
            reasoning="Implied prob: 58%, My prob: 66%, Edge: 8%",
            implied_prob_external=0.64,
            my_prob=0.66,
            edge_external=0.02,
            evidence_quality=0.9,
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="sports",
        )
        self.assertTrue(validated.should_trade)
        self.assertGreater(validated.bet_size_pct, 0.0)

    def test_validate_and_enrich_disables_trade_when_market_implied_missing(self) -> None:
        market = Market(
            id="m15",
            question="Will Team A win?",
            outcomes=[MarketOutcome(name="Team A"), MarketOutcome(name="Team B")],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="Team A",
            confidence=0.72,
            bet_size_pct=0.5,
            reasoning="Implied prob: unknown, My prob: 72%, Edge: 12%",
            implied_prob_external=0.60,
            my_prob=0.72,
            edge_external=0.12,
            evidence_quality=0.9,
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="sports",
        )
        self.assertFalse(validated.should_trade)
        self.assertEqual(validated.bet_size_pct, 0.0)

    def test_validate_and_enrich_uses_fallback_edge_when_probabilities_missing(self) -> None:
        market = Market(
            id="m11",
            question="Will event happen?",
            outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.67,
            bet_size_pct=0.2,
            reasoning="Edge: 8%",
            implied_prob_external=None,
            my_prob=None,
            edge_external=0.08,
            evidence_quality=0.0,
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="generic",
        )
        self.assertAlmostEqual(validated.edge_external or 0.0, 0.0, places=6)

    def test_validate_and_enrich_uses_abs_market_gap_for_fallback_edge(self) -> None:
        market = Market(
            id="m11c",
            question="Will event happen?",
            outcomes=[MarketOutcome(name="YES", price=0.70), MarketOutcome(name="NO", price=0.30)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.64,
            bet_size_pct=0.2,
            reasoning="Edge: 8%",
            implied_prob_external=None,
            my_prob=0.64,
            edge_external=0.08,
            edge_source="fallback",
            evidence_quality=0.5,
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="generic",
        )
        self.assertAlmostEqual(validated.edge_external or 0.0, 0.06, places=6)

    def test_validate_and_enrich_applies_configured_fallback_caps(self) -> None:
        market = Market(
            id="m-fallback-caps",
            question="Will event happen?",
            outcomes=[MarketOutcome(name="YES", price=0.70), MarketOutcome(name="NO", price=0.30)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.90,
            bet_size_pct=0.3,
            reasoning="No external odds found. Implied prob: unknown. No data available.",
            implied_prob_external=None,
            my_prob=0.80,
            edge_external=0.15,
            edge_source="fallback",
            evidence_quality=0.1,
        )
        client = GrokClient(
            api_key="x",
            settings=Settings(
                GROK_PROXY_CONFIDENCE_CAP=0.67,
                GROK_LOW_INFO_CONFIDENCE_CAP=0.59,
                GROK_FALLBACK_MIN_EVIDENCE_QUALITY=0.72,
            ),
        )
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="generic",
        )
        self.assertLessEqual(validated.confidence, 0.59)
        self.assertFalse(validated.should_trade)
        self.assertIn("fallback_edge_without_verifiable_signal", validated.reasoning)

    def test_validate_and_enrich_applies_definitive_outcome_evidence_floor(self) -> None:
        market = Market(
            id="m-definitive-floor",
            question="Player prop post-game market",
            outcomes=[MarketOutcome(name="YES", price=0.40), MarketOutcome(name="NO", price=0.60)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.85,
            bet_size_pct=0.3,
            reasoning=(
                "Final score in official recap from Reuters confirms settlement outcome."
            ),
            implied_prob_external=None,
            my_prob=0.97,
            edge_external=0.35,
            edge_source="fallback",
            likelihood_ratio=25.0,
            evidence_quality=0.80,
            primary_source_url="https://www.reuters.com/sports/example",
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="generic",
        )
        self.assertTrue(validated.definitive_outcome_detected)
        self.assertGreaterEqual(validated.evidence_quality, 0.72)
        self.assertEqual(validated.evidence_quality_floor_applied, "definitive_outcome_floor")
        self.assertEqual(validated.source_match_class, "settlement_aligned")
        self.assertEqual(validated.evidence_basis, "direct")

    def test_validate_and_enrich_definitive_outcome_requires_structured_probability(self) -> None:
        market = Market(
            id="m-definitive-no-my-prob",
            question="Player prop post-game market",
            outcomes=[MarketOutcome(name="YES", price=0.40), MarketOutcome(name="NO", price=0.60)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.85,
            bet_size_pct=0.3,
            reasoning=(
                "Final score in official recap from Reuters confirms settlement outcome. "
                "My probability: 97%."
            ),
            implied_prob_external=None,
            my_prob=None,
            edge_external=0.35,
            edge_source="fallback",
            likelihood_ratio=25.0,
            evidence_quality=0.90,
            primary_source_url="https://www.reuters.com/sports/example",
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="generic",
        )
        self.assertFalse(validated.definitive_outcome_detected)
        self.assertNotEqual(validated.evidence_quality_floor_applied, "definitive_outcome_floor")
        self.assertEqual(validated.source_match_class, "settlement_aligned")

    def test_validate_and_enrich_suppresses_non_sports_direct_floor_without_primary_url(self) -> None:
        market = Market(
            id="m-definitive-missing-source",
            question="Player prop post-game market",
            outcomes=[MarketOutcome(name="YES", price=0.40), MarketOutcome(name="NO", price=0.60)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.85,
            bet_size_pct=0.3,
            reasoning=(
                "Final score in official recap from Reuters confirms settlement outcome."
            ),
            implied_prob_external=None,
            my_prob=0.85,
            edge_external=0.35,
            edge_source="fallback",
            likelihood_ratio=25.0,
            evidence_quality=0.10,
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="generic",
        )
        self.assertFalse(validated.definitive_outcome_detected)
        self.assertNotEqual(validated.evidence_quality_floor_applied, "definitive_outcome_floor")
        self.assertEqual(validated.evidence_floor_suppressed_reason, "missing_primary_source_url")
        self.assertEqual(validated.evidence_basis, "proxy")

    def test_weather_profile_exempt_from_primary_source_url_requirement(self) -> None:
        # Same decision under two profiles isolates the exemption. The reasoning
        # is reused from the generic suppression test above (it classifies as
        # direct), so the only difference is profile_name.
        market = Market(
            id="KXHIGHDEN-26JUN20-B89.5",
            question="Will the Denver daily high settle in the 89-90F bin?",
            category="weather",
            outcomes=[MarketOutcome(name="YES", price=0.40), MarketOutcome(name="NO", price=0.60)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.85,
            bet_size_pct=0.3,
            reasoning="Final score in official recap from Reuters confirms settlement outcome.",
            implied_prob_external=None,
            my_prob=0.85,
            edge_external=0.35,
            edge_source="fallback",
            likelihood_ratio=25.0,
            evidence_quality=0.10,
        )
        client = GrokClient(api_key="x")
        weather = client._validate_and_enrich_decision(market, decision, profile_name="weather")
        generic = client._validate_and_enrich_decision(market, decision, profile_name="generic")
        # Weather has a universal NWS settlement source -> exempt: stays direct,
        # not suppressed for a missing URL. Generic still requires the URL.
        self.assertEqual(weather.evidence_basis, "direct")
        self.assertNotEqual(
            weather.evidence_floor_suppressed_reason, "missing_primary_source_url"
        )
        self.assertEqual(generic.evidence_basis, "proxy")

    def test_weather_missing_current_source_is_not_upgraded_by_source_keywords(self) -> None:
        market = Market(
            id="KXLOWTSATX-26JUL17-T77",
            question="Will the minimum temperature be above 77F in San Antonio?",
            category="weather",
            outcomes=[
                MarketOutcome(name="YES", price=0.74),
                MarketOutcome(name="NO", price=0.26),
            ],
        )
        decision = TradeDecision(
            should_trade=False,
            outcome="NO",
            confidence=0.52,
            probability_yes=0.48,
            bet_size_pct=0.0,
            reasoning=(
                "No NWS point forecast or station observation was found for the "
                "current settlement day. The official climatological report is "
                "pending. evidence_basis=absence_only; missing primary source."
            ),
            implied_prob_external=0.74,
            my_prob=0.48,
            edge_external=-0.26,
            edge_source="computed",
            evidence_basis="absence_only",
            evidence_quality=0.0,
        )

        validated = GrokClient(api_key="x")._validate_and_enrich_decision(
            market,
            decision,
            profile_name="weather",
        )

        self.assertEqual(validated.source_match_class, "missing_or_absence_only")
        self.assertEqual(validated.evidence_basis, "absence_only")
        self.assertLessEqual(validated.evidence_quality, 0.50)
        self.assertFalse(validated.should_trade)

    def test_weather_outdated_observation_url_does_not_override_current_data_gap(self) -> None:
        market = Market(
            id="KXHIGHTSEA-26JUL17-B74.5",
            question="Will Seattle's maximum temperature be 74-75F?",
            category="weather",
            outcomes=[
                MarketOutcome(name="YES", price=0.33),
                MarketOutcome(name="NO", price=0.67),
            ],
        )
        decision = TradeDecision(
            should_trade=False,
            outcome="YES",
            confidence=0.53,
            probability_yes=0.53,
            bet_size_pct=0.0,
            reasoning=(
                "No current NWS daily summary or observed maximum is available "
                "yet. The cited station page only contains prior-day observations, "
                "so the settlement-aligned source is still missing."
            ),
            key_sources=[
                "https://tgftp.nws.noaa.gov/weather/current/KSEA.html (prior day)"
            ],
            implied_prob_external=0.33,
            my_prob=0.53,
            edge_external=0.20,
            edge_source="computed",
            evidence_basis="absence_only",
            primary_source_url=(
                "https://tgftp.nws.noaa.gov/weather/current/KSEA.html"
            ),
            evidence_quality=0.0,
        )

        validated = GrokClient(api_key="x")._validate_and_enrich_decision(
            market,
            decision,
            profile_name="weather",
        )

        self.assertEqual(validated.source_match_class, "missing_or_absence_only")
        self.assertEqual(validated.evidence_basis, "absence_only")
        self.assertLessEqual(validated.evidence_quality, 0.50)
        self.assertIsNone(validated.evidence_quality_floor_applied)

    def test_weather_current_observation_with_url_remains_direct(self) -> None:
        market = Market(
            id="KXLOWTDEN-26JUL17-B64.5",
            question="Will Denver's minimum temperature be 64-65F?",
            category="weather",
            outcomes=[
                MarketOutcome(name="YES", price=0.39),
                MarketOutcome(name="NO", price=0.61),
            ],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="NO",
            confidence=0.81,
            probability_yes=0.10,
            bet_size_pct=0.2,
            reasoning=(
                "Current NWS KDEN ASOS observations show today's daily minimum "
                "was 66F at 05:53 MDT, above the bin. The observed value and "
                "timestamp directly resolve the settlement criterion."
            ),
            implied_prob_external=0.39,
            my_prob=0.10,
            edge_external=-0.29,
            edge_source="computed",
            evidence_basis="direct",
            primary_source_url=(
                "https://forecast.weather.gov/data/obhistory/KDEN.html"
            ),
            evidence_quality=0.90,
        )

        validated = GrokClient(api_key="x")._validate_and_enrich_decision(
            market,
            decision,
            profile_name="weather",
        )

        self.assertEqual(validated.source_match_class, "settlement_aligned")
        self.assertEqual(validated.evidence_basis, "direct")
        self.assertGreaterEqual(validated.evidence_quality, 0.75)

    def test_weather_forecast_explicit_proxy_is_not_upgraded_to_direct(self) -> None:
        market = Market(
            id="KXHIGHTSEA-26JUL18-B74.5",
            question="Will Seattle's maximum temperature be 74-75F?",
            category="weather",
            outcomes=[
                MarketOutcome(name="YES", price=0.43),
                MarketOutcome(name="NO", price=0.57),
            ],
        )
        decision = TradeDecision(
            should_trade=False,
            outcome="NO",
            confidence=0.68,
            probability_yes=0.32,
            bet_size_pct=0.0,
            reasoning=(
                "NWS forecasts show a high near 75-76F. Proxy evidence only; "
                "the maximum has not been observed and the final climate report "
                "is missing."
            ),
            implied_prob_external=0.43,
            my_prob=0.32,
            edge_external=-0.11,
            edge_source="computed",
            evidence_basis="proxy",
            primary_source_url=(
                "https://forecast.weather.gov/MapClick.php?lat=47.6062&lon=-122.3321"
            ),
            evidence_quality=0.65,
        )

        validated = GrokClient(api_key="x")._validate_and_enrich_decision(
            market,
            decision,
            profile_name="weather",
        )

        self.assertEqual(validated.evidence_basis, "proxy")
        self.assertLessEqual(validated.evidence_quality, 0.75)

    def test_sports_pregame_props_remain_proxy(self) -> None:
        market = Market(
            id="KXMLBKS-26JUL181420MINCHC-MINTBRADLEY26-8",
            question="Taj Bradley: 8+ strikeouts?",
            category="sports",
            outcomes=[
                MarketOutcome(name="YES", price=0.53),
                MarketOutcome(name="NO", price=0.47),
            ],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="NO",
            confidence=0.61,
            probability_yes=0.39,
            bet_size_pct=0.1,
            reasoning=(
                "Covers pregame player props and recent-start statistics imply "
                "P(8+) near 39%. The game is scheduled but has not been played."
            ),
            implied_prob_external=0.53,
            my_prob=0.39,
            edge_external=-0.14,
            edge_source="computed",
            evidence_basis="proxy",
            primary_source_url=(
                "https://www.covers.com/sport/baseball/mlb/players/12617/taj-bradley"
            ),
            evidence_quality=0.60,
        )

        validated = GrokClient(api_key="x")._validate_and_enrich_decision(
            market,
            decision,
            profile_name="sports",
        )

        self.assertEqual(validated.source_match_class, "preview_or_proxy")
        self.assertEqual(validated.evidence_basis, "proxy")
        self.assertLessEqual(validated.evidence_quality, 0.80)

    def test_unpublished_settlement_chart_overrides_direct_keyword_match(self) -> None:
        market = Market(
            id="KXNETFLIXTOPVIEWSMOVIE-26JUL20-12",
            question="Will the top Netflix movie exceed 12M views?",
            category="entertainment",
            outcomes=[
                MarketOutcome(name="YES", price=0.55),
                MarketOutcome(name="NO", price=0.45),
            ],
        )
        decision = TradeDecision(
            should_trade=False,
            outcome="YES",
            confidence=0.50,
            bet_size_pct=0.0,
            reasoning=(
                "No official Netflix Top 10 chart for the settlement period has "
                "been published yet. The current chart covers the prior week, so "
                "direct settlement data is unavailable."
            ),
            implied_prob_external=0.55,
            my_prob=0.50,
            edge_external=-0.05,
            edge_source="computed",
            evidence_basis="direct",
            primary_source_url="https://www.netflix.com/tudum/top10",
            evidence_quality=0.0,
        )

        validated = GrokClient(api_key="x")._validate_and_enrich_decision(
            market,
            decision,
            profile_name="entertainment",
        )

        self.assertEqual(validated.source_match_class, "missing_or_absence_only")
        self.assertEqual(validated.evidence_basis, "absence_only")
        self.assertLessEqual(validated.evidence_quality, 0.50)

    def test_commodity_with_allowlisted_settlement_url_counts_as_direct(self) -> None:
        # Core unblock: once a commodity market cites a reachable settlement-grade
        # URL (cmegroup.com), settlement-aligned evidence counts as direct and the
        # floor is not suppressed -- the same path weather already enjoys.
        market = Market(
            id="KXSILVERD-26JUN2217-T66.25",
            question="Will the silver close price be above 66.25 on Jun 26?",
            category="commodities",
            outcomes=[MarketOutcome(name="YES", price=0.45), MarketOutcome(name="NO", price=0.55)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.80,
            bet_size_pct=0.2,
            reasoning=(
                "Official CME front-month settlement price shows silver spot price "
                "at 67.10 as of the 1:30 PM ET COMEX close (observed value with "
                "timestamp), above the 66.25 threshold."
            ),
            implied_prob_external=0.45,
            my_prob=0.80,
            edge_external=0.35,
            edge_source="computed",
            likelihood_ratio=20.0,
            evidence_quality=0.10,
            primary_source_url="https://www.cmegroup.com/markets/metals/precious/silver.quotes.html",
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market, decision, profile_name="commodity"
        )
        self.assertEqual(validated.source_match_class, "settlement_aligned")
        self.assertEqual(validated.evidence_basis, "direct")
        self.assertNotEqual(
            validated.evidence_floor_suppressed_reason, "missing_primary_source_url"
        )

    def test_commodity_without_allowlisted_url_still_suppressed(self) -> None:
        # Safety preserved: an aggregator URL (not settlement-grade) is still
        # rejected, so the fix does not loosen the evidence gate.
        market = Market(
            id="KXSILVERD-26JUN2217-T66.25",
            question="Will the silver close price be above 66.25 on Jun 26?",
            category="commodities",
            outcomes=[MarketOutcome(name="YES", price=0.45), MarketOutcome(name="NO", price=0.55)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.80,
            bet_size_pct=0.2,
            reasoning=(
                "Official CME front-month settlement price shows silver spot price "
                "at 67.10 as of the 1:30 PM ET COMEX close (observed value with "
                "timestamp), above the 66.25 threshold."
            ),
            implied_prob_external=0.45,
            my_prob=0.80,
            edge_external=0.35,
            edge_source="computed",
            likelihood_ratio=20.0,
            evidence_quality=0.10,
            primary_source_url="https://www.investing.com/commodities/silver",
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market, decision, profile_name="commodity"
        )
        self.assertEqual(
            validated.evidence_floor_suppressed_reason, "missing_primary_source_url"
        )
        self.assertEqual(validated.evidence_basis, "proxy")

    def test_crypto_with_allowlisted_exchange_url_counts_as_direct(self) -> None:
        market = Market(
            id="KXETHD-26JUN2217-T1739.99",
            question="Will ETH close above 1739.99 on Jun 22 17:00?",
            category="crypto",
            outcomes=[MarketOutcome(name="YES", price=0.45), MarketOutcome(name="NO", price=0.55)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.80,
            bet_size_pct=0.2,
            reasoning=(
                "Coinbase exchange spot price shows ETH at 1750 as of 17:00 UTC "
                "(observed value with timestamp); settlement confirms above threshold."
            ),
            implied_prob_external=0.45,
            my_prob=0.80,
            edge_external=0.35,
            edge_source="computed",
            likelihood_ratio=20.0,
            evidence_quality=0.10,
            primary_source_url="https://www.coinbase.com/price/ethereum",
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market, decision, profile_name="crypto"
        )
        self.assertEqual(validated.source_match_class, "settlement_aligned")
        self.assertEqual(validated.evidence_basis, "direct")
        self.assertNotEqual(
            validated.evidence_floor_suppressed_reason, "missing_primary_source_url"
        )

    def test_validate_and_enrich_caps_proxy_evidence_quality(self) -> None:
        market = Market(
            id="m-proxy-cap",
            question="Will the index close above the threshold?",
            outcomes=[MarketOutcome(name="YES", price=0.40), MarketOutcome(name="NO", price=0.60)],
        )
        decision = TradeDecision(
            should_trade=False,
            outcome="YES",
            confidence=0.65,
            bet_size_pct=0.0,
            reasoning=(
                "Implied probability: 40%. My probability: 65%. "
                "Edge from momentum and sentiment trends."
            ),
            implied_prob_external=0.40,
            my_prob=0.65,
            edge_external=0.25,
            edge_source="computed",
            evidence_quality=0.95,
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="generic",
        )
        self.assertEqual(validated.evidence_basis, "proxy")
        self.assertLessEqual(validated.evidence_quality, 0.75)
        self.assertEqual(validated.evidence_quality_floor_applied, "proxy_evidence_cap")

    def test_validate_and_enrich_allows_higher_proxy_cap_for_sports_with_odds(self) -> None:
        market = Market(
            id="m-sports-odds-cap",
            question="Will the favorite win the game?",
            outcomes=[MarketOutcome(name="YES", price=0.40), MarketOutcome(name="NO", price=0.60)],
        )
        decision = TradeDecision(
            should_trade=False,
            outcome="YES",
            confidence=0.65,
            bet_size_pct=0.0,
            reasoning=(
                "Implied probability: 40%. My probability: 65%. "
                "Edge from momentum and sentiment trends."
            ),
            implied_prob_external=0.40,
            my_prob=0.65,
            edge_external=0.25,
            edge_source="computed",
            evidence_quality=0.95,
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="sports",
        )
        self.assertEqual(validated.evidence_basis, "proxy")
        self.assertEqual(validated.evidence_quality_floor_applied, "proxy_evidence_cap")
        self.assertAlmostEqual(validated.evidence_quality, 0.80)

    def test_validate_and_enrich_sports_proxy_cap_is_sports_specific(self) -> None:
        # The identical proxy decision under a non-sports profile keeps the
        # standard 0.75 ceiling, proving the 0.80 ceiling is sports-only.
        market = Market(
            id="m-sports-odds-cap",
            question="Will the index close above the threshold?",
            outcomes=[MarketOutcome(name="YES", price=0.40), MarketOutcome(name="NO", price=0.60)],
        )
        decision = TradeDecision(
            should_trade=False,
            outcome="YES",
            confidence=0.65,
            bet_size_pct=0.0,
            reasoning=(
                "Implied probability: 40%. My probability: 65%. "
                "Edge from momentum and sentiment trends."
            ),
            implied_prob_external=0.40,
            my_prob=0.65,
            edge_external=0.25,
            edge_source="computed",
            evidence_quality=0.95,
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="generic",
        )
        self.assertEqual(validated.evidence_basis, "proxy")
        self.assertEqual(validated.evidence_quality_floor_applied, "proxy_evidence_cap")
        self.assertAlmostEqual(validated.evidence_quality, 0.75)

    def test_validate_and_enrich_treats_aggregator_url_as_proxy(self) -> None:
        market = Market(
            id="m-aggregator-url",
            question="Will gold close above the threshold?",
            outcomes=[MarketOutcome(name="YES", price=0.40), MarketOutcome(name="NO", price=0.60)],
        )
        decision = TradeDecision(
            should_trade=False,
            outcome="NO",
            confidence=0.65,
            bet_size_pct=0.0,
            reasoning=(
                "Implied probability: 40%. My probability: 65%. "
                "Live quote shows spot price 4374 as of today; threshold 4439."
            ),
            implied_prob_external=0.40,
            my_prob=0.65,
            edge_external=0.25,
            edge_source="computed",
            likelihood_ratio=1.0,
            evidence_quality=0.95,
            primary_source_url="https://tradingeconomics.com/commodity/gold",
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="commodities",
        )
        self.assertEqual(validated.evidence_basis, "proxy")
        self.assertLessEqual(validated.evidence_quality, 0.75)
        self.assertIn("tradingeconomics.com", validated.primary_source_url or "")

    def test_validate_and_enrich_allows_settlement_grade_url_direct(self) -> None:
        market = Market(
            id="m-allowlisted-url",
            question="Will WTI close above the threshold?",
            outcomes=[MarketOutcome(name="YES", price=0.40), MarketOutcome(name="NO", price=0.60)],
        )
        decision = TradeDecision(
            should_trade=False,
            outcome="NO",
            confidence=0.85,
            bet_size_pct=0.0,
            reasoning=(
                "Implied probability: 40%. My probability: 85%. "
                "EIA weekly report confirms official settlement; observed value 87.10 as of today."
            ),
            implied_prob_external=0.40,
            my_prob=0.85,
            edge_external=0.45,
            edge_source="computed",
            likelihood_ratio=1.0,
            evidence_quality=0.95,
            primary_source_url="https://www.eia.gov/petroleum/supply/weekly/",
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="commodities",
        )
        self.assertEqual(validated.evidence_basis, "direct")
        self.assertGreater(validated.evidence_quality, 0.75)

    def test_validate_and_enrich_extracts_primary_url_from_key_sources(self) -> None:
        market = Market(
            id="m-definitive-key-source",
            question="Player prop post-game market",
            outcomes=[MarketOutcome(name="YES", price=0.40), MarketOutcome(name="NO", price=0.60)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.85,
            bet_size_pct=0.3,
            reasoning=(
                "Final score in official recap confirms settlement outcome."
            ),
            key_sources=["Reuters recap https://www.reuters.com/sports/example."],
            implied_prob_external=None,
            my_prob=0.97,
            edge_external=0.35,
            edge_source="fallback",
            likelihood_ratio=25.0,
            evidence_quality=0.80,
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="generic",
        )

        self.assertEqual(
            validated.primary_source_url,
            "https://www.reuters.com/sports/example",
        )
        self.assertTrue(validated.definitive_outcome_detected)
        self.assertEqual(validated.evidence_basis, "direct")

    def test_validate_and_enrich_direct_fallback_bypasses_min_evidence_gate(self) -> None:
        market = Market(
            id="m-definitive-bypass",
            question="Player prop post-game market",
            outcomes=[MarketOutcome(name="YES", price=0.40), MarketOutcome(name="NO", price=0.60)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.85,
            bet_size_pct=0.3,
            reasoning=(
                "Final score in official recap from Reuters confirms settlement outcome."
            ),
            implied_prob_external=None,
            my_prob=0.97,
            edge_external=0.35,
            edge_source="fallback",
            likelihood_ratio=25.0,
            evidence_quality=0.80,
            primary_source_url="https://www.reuters.com/sports/example",
        )
        client = GrokClient(
            api_key="x",
            settings=Settings(
                GROK_FALLBACK_MIN_EVIDENCE_QUALITY=0.90,
            ),
        )
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="generic",
        )
        self.assertTrue(validated.should_trade)
        self.assertNotIn("fallback_edge_without_verifiable_signal", validated.reasoning)

    def test_validate_and_enrich_verifiable_fallback_not_capped_to_half(self) -> None:
        market = Market(
            id="m11b",
            question="Will minimum temp stay above 40F?",
            outcomes=[MarketOutcome(name="YES", price=0.30), MarketOutcome(name="NO", price=0.70)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="NO",
            confidence=0.92,
            bet_size_pct=0.3,
            reasoning=(
                "No external odds found. Official NWS CLI and ASOS observation confirmed "
                "settlement conditions."
            ),
            implied_prob_external=None,
            my_prob=None,
            edge_external=0.22,
            evidence_quality=0.0,
            primary_source_url="https://forecast.weather.gov/MapClick.php?lat=39.1&lon=-94.6",
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="weather",
        )
        self.assertGreaterEqual(validated.evidence_quality, 0.55)

    def test_validate_and_enrich_applies_high_confidence_evidence_override(self) -> None:
        market = Market(
            id="m-override-eq",
            question="Will max temp stay below 70F?",
            outcomes=[MarketOutcome(name="YES", price=0.72), MarketOutcome(name="NO", price=0.28)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.95,
            bet_size_pct=0.2,
            reasoning=(
                "NWS observation update with settlement context from exchange bulletin. "
                "My prob: 95%. Edge: 23%."
            ),
            implied_prob_external=None,
            my_prob=None,
            edge_external=0.23,
            evidence_quality=0.0,
            primary_source_url="https://forecast.weather.gov/MapClick.php?lat=39.1&lon=-94.6",
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="weather",
        )
        self.assertGreaterEqual(validated.evidence_quality, 0.60)

    def test_validate_and_enrich_applies_weather_observed_evidence_floor(self) -> None:
        market = Market(
            id="m-weather-obs-floor",
            question="Will max temp stay below 70F?",
            outcomes=[MarketOutcome(name="YES", price=0.72), MarketOutcome(name="NO", price=0.28)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.90,
            raw_confidence=0.90,
            bet_size_pct=0.2,
            reasoning=(
                "Observed station data shows threshold already exceeded and physically impossible to reverse."
            ),
            implied_prob_external=None,
            my_prob=0.90,
            edge_external=0.20,
            edge_source="fallback",
            evidence_quality=0.1,
            primary_source_url="https://forecast.weather.gov/MapClick.php?lat=39.1&lon=-94.6",
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="weather",
        )
        self.assertGreaterEqual(validated.evidence_quality, 0.75)

    def test_validate_and_enrich_skips_weather_observed_floor_for_lowt_without_daily_low(self) -> None:
        market = Market(
            id="KXLOWTOKC-26MAY24-T63",
            question="Lowest temperature in Oklahoma City today?",
            outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.90,
            raw_confidence=0.90,
            bet_size_pct=0.2,
            reasoning=(
                "Current METAR reading is 63F, already above the 63F threshold and threshold already exceeded."
            ),
            implied_prob_external=None,
            my_prob=0.90,
            edge_external=0.20,
            edge_source="fallback",
            evidence_quality=0.1,
            primary_source_url="https://forecast.weather.gov/MapClick.php?lat=35.4&lon=-97.6",
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="weather",
        )
        self.assertLess(validated.evidence_quality, 0.75)

    def test_validate_and_enrich_applies_weather_observed_floor_for_lowt_with_daily_low(self) -> None:
        market = Market(
            id="KXLOWTOKC-26MAY24-T63",
            question="Lowest temperature in Oklahoma City today?",
            outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.90,
            raw_confidence=0.90,
            bet_size_pct=0.2,
            reasoning=(
                "NWS reports today's observed daily low was 64F at 6:12 AM, threshold already exceeded."
            ),
            implied_prob_external=None,
            my_prob=0.90,
            edge_external=0.20,
            edge_source="fallback",
            evidence_quality=0.1,
            primary_source_url="https://forecast.weather.gov/MapClick.php?lat=35.4&lon=-97.6",
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="weather",
        )
        self.assertGreaterEqual(validated.evidence_quality, 0.75)

    def test_validate_and_enrich_normalizes_outcome_label(self) -> None:
        market = Market(
            id="m7",
            question="Who wins?",
            outcomes=[MarketOutcome(name="Team A", price=0.45), MarketOutcome(name="Team B", price=0.55)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome=" team   a ",
            confidence=0.72,
            bet_size_pct=0.5,
            reasoning="Implied prob: 45%, My prob: 72%, Edge: 27% as of now",
            implied_prob_external=0.45,
            my_prob=0.72,
            edge_external=0.27,
            evidence_quality=0.8,
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="generic",
        )
        self.assertEqual(validated.outcome, "Team A")
        self.assertTrue(validated.should_trade)

    def test_validate_and_enrich_blocks_unresolvable_outcome(self) -> None:
        market = Market(
            id="m8",
            question="Will event happen?",
            outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="NOT_LISTED",
            confidence=0.8,
            bet_size_pct=0.6,
            reasoning="test",
            evidence_quality=0.9,
        )
        client = GrokClient(api_key="x")
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="generic",
        )
        self.assertFalse(validated.should_trade)
        self.assertEqual(validated.bet_size_pct, 0.0)
        self.assertIn("[Outcome mismatch]", validated.reasoning)


class TestQuotaExhaustedClassification(unittest.TestCase):
    """Verify _is_quota_exhausted_grok_error detects xAI spending-limit errors."""

    def test_resource_exhausted_detected(self) -> None:
        from grok_client import _is_quota_exhausted_grok_error

        exc = RuntimeError(
            "RESOURCE_EXHAUSTED: Your team has either used all available credits "
            "or reached its monthly spending limit."
        )
        self.assertTrue(_is_quota_exhausted_grok_error(exc))

    def test_monthly_spending_limit_detected(self) -> None:
        from grok_client import _is_quota_exhausted_grok_error

        exc = RuntimeError("reached its monthly spending limit")
        self.assertTrue(_is_quota_exhausted_grok_error(exc))

    def test_transient_internal_not_quota(self) -> None:
        from grok_client import _is_quota_exhausted_grok_error

        exc = RuntimeError("StatusCode.INTERNAL: internal server error")
        self.assertFalse(_is_quota_exhausted_grok_error(exc))

    def test_retriable_excludes_quota(self) -> None:
        exc = RuntimeError(
            "RESOURCE_EXHAUSTED: monthly spending limit"
        )
        result = _is_retriable_grok_error(exc, 100.0)
        self.assertFalse(result)

    def test_validate_and_enrich_applies_convergent_evidence_floor(self) -> None:
        from config import Settings

        market = Market(
            id="m-convergent",
            question="Will event happen?",
            outcomes=[
                MarketOutcome(name="YES", price=0.50),
                MarketOutcome(name="NO", price=0.50),
            ],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.70,
            bet_size_pct=0.2,
            reasoning="No external odds found. Implied prob: unknown. My prob: 70%.",
            implied_prob_external=0.50,
            my_prob=0.70,
            edge_external=0.20,
            evidence_quality=0.9,
        )
        settings = Settings(
            EVIDENCE_QUALITY_CONVERGENT_FLOOR_ENABLED=True,
            EVIDENCE_QUALITY_CONVERGENT_FLOOR_VALUE=0.60,
        )
        client = GrokClient(api_key="x", settings=settings)
        validated = client._validate_and_enrich_decision(
            market,
            decision,
            profile_name="generic",
            self_consistency_passed=True,
            family_is_profitable=True,
        )
        self.assertGreaterEqual(validated.evidence_quality, 0.60)
        self.assertEqual(
            validated.evidence_quality_floor_applied,
            "convergent_evidence_floor",
        )


class SelfConsistencyShouldRunTest(unittest.TestCase):
    def _market(self) -> Market:
        return Market(
            id="m-sc",
            question="Will event happen?",
            outcomes=[
                MarketOutcome(name="YES", price=0.55),
                MarketOutcome(name="NO", price=0.45),
            ],
            liquidity_usdc=5000.0,
        )

    def _decision(self) -> TradeDecision:
        return TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.70,
            bet_size_pct=0.3,
            reasoning="x",
            implied_prob_external=0.45,
            my_prob=0.70,
            edge_external=0.25,
        )

    def test_allow_flag_gates_self_consistency(self) -> None:
        client = GrokClient(api_key="x")
        market = self._market()
        decision = self._decision()
        # High liquidity clears the threshold, so it runs when allowed.
        self.assertTrue(
            client._should_run_self_consistency(
                market, decision, deep=False, allow_self_consistency=True
            )
        )
        # Gated out for non-top candidates regardless of thresholds.
        self.assertFalse(
            client._should_run_self_consistency(
                market, decision, deep=False, allow_self_consistency=False
            )
        )


if __name__ == "__main__":
    unittest.main()
