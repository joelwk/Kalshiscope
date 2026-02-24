import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from config import SearchConfig
from grok_client import GrokClient, _extract_json
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


class TestGrokClient(unittest.TestCase):
    def test_extract_json(self) -> None:
        payload = _extract_json("prefix {\"foo\": 1} suffix")
        self.assertEqual(payload, {"foo": 1})

    def test_extract_json_invalid(self) -> None:
        with self.assertRaises(ValueError):
            _extract_json("no-json")

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

        with patch("grok_client.web_search", side_effect=fake_web_search), patch(
            "grok_client.x_search", side_effect=fake_x_search
        ):
            client.analyze_market(market)

        self.assertEqual(captured["web"]["allowed_domains"], ["example.com"])
        self.assertEqual(captured["x"]["from_date"], datetime(2026, 1, 13, 0, 0, tzinfo=timezone.utc))
        self.assertEqual(captured["x"]["to_date"], datetime(2026, 1, 13, 12, 0, tzinfo=timezone.utc))
        self.assertEqual(captured["x"]["allowed_x_handles"], ["Foo"])
        self.assertFalse(captured["x"]["enable_image_understanding"])
        self.assertFalse(captured["x"]["enable_video_understanding"])

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

        with patch("grok_client.web_search", side_effect=fake_web_search), patch(
            "grok_client.x_search", side_effect=fake_x_search
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
        self.assertAlmostEqual(validated.edge_external or 0.0, 0.08, places=6)

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


if __name__ == "__main__":
    unittest.main()
