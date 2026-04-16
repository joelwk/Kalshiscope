import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from config import Settings
from main import (
    _available_orderbook_sell_quantity,
    _analysis_result_rank,
    _analyze_market_candidate,
    _best_orderbook_sell_price,
    _build_order_request_from_market,
    _build_kalshi_market_fetch_window,
    _build_execution_audit,
    _build_reasoning_hash,
    _can_use_lenient_stale_refresh_fallback,
    _cap_analysis_candidates,
    _cap_effective_confidence_for_market,
    _calculate_bet,
    _collapse_event_ladders,
    _confidence_gate_override_metrics,
    _compute_next_wakeup_seconds,
    _daily_balance_delta_usdc,
    _daily_drawdown_cap_reached,
    _daily_trade_cap_reached,
    _edge_threshold_for_market,
    _event_concentration_blocked,
    _event_ticker_prefix,
    _effective_position_override_threshold,
    _extract_order_cancel_reason,
    _extract_order_fill_count,
    _fetch_markets_with_optional_server_filters,
    _filter_markets,
    _kelly_fraction_for_market_horizon,
    _log_settings_summary,
    _max_confidence_for_market,
    _min_evidence_quality_for_market,
    _pre_analysis_hard_rejection,
    _pre_analysis_opportunity_score,
    _passes_edge_threshold,
    _passes_refreshed_edge_guard,
    _requires_market_refresh,
    _resolve_dynamic_analysis_candidate_cap,
    _should_skip_for_balance,
    _ticker_resolution_date,
    _should_adjust_position,
    _is_likely_resolved_by_ticker_date,
    _is_coinflip_signal,
    _is_crypto_bin_market,
    _is_weather_bin_market,
    _is_weather_market_by_ticker,
    _parse_exchange_position_row,
)
from models import Market, MarketOutcome, MarketState, Position, TradeDecision


class DummyStateManager:
    def __init__(self, mapping: dict[str, MarketState | None]) -> None:
        self.mapping = mapping

    def get_market_state(self, market_id: str) -> MarketState | None:
        return self.mapping.get(market_id)


class DummyGrokClient:
    def __init__(self, decision: TradeDecision) -> None:
        self.decision = decision

    def analyze_market(self, market, search_config=None, previous_analysis=None):
        return self.decision


class FailingGrokClient:
    def analyze_market(self, market, search_config=None, previous_analysis=None):
        raise RuntimeError("StatusCode.INTERNAL: internal server error")


class TestMainUtils(unittest.TestCase):
    class _DummyKalshiClient:
        def __init__(self, responses):
            self._responses = list(responses)
            self.calls = []
            self.reset_calls = 0

        def get_markets(self, close_time_start=None, close_time_end=None):
            self.calls.append((close_time_start, close_time_end))
            response = self._responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        def reset_session(self):
            self.reset_calls += 1

    def test_analysis_result_rank_prioritizes_tradeable_high_quality(self) -> None:
        tradeable = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.82,
            bet_size_pct=0.2,
            reasoning="tradeable",
            evidence_quality=0.9,
        )
        non_tradeable = tradeable.model_copy(
            update={"should_trade": False, "evidence_quality": 1.0}
        )
        self.assertGreater(
            _analysis_result_rank({"decision": tradeable}),
            _analysis_result_rank({"decision": non_tradeable}),
        )

    def test_should_skip_for_balance_when_below_min_bet(self) -> None:
        self.assertTrue(
            _should_skip_for_balance(
                available_balance=1.5,
                min_bet_usdc=2.0,
            )
        )
        self.assertFalse(
            _should_skip_for_balance(
                available_balance=2.0,
                min_bet_usdc=2.0,
            )
        )
        self.assertFalse(
            _should_skip_for_balance(
                available_balance=None,
                min_bet_usdc=2.0,
            )
        )

    def test_analysis_result_rank_prefers_higher_pre_execution_final_score(self) -> None:
        tradeable = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.75,
            bet_size_pct=0.2,
            reasoning="tradeable",
            evidence_quality=0.8,
        )
        lower = {"decision": tradeable, "pre_execution_final_score": 0.10}
        higher = {"decision": tradeable, "pre_execution_final_score": 0.30}
        self.assertGreater(_analysis_result_rank(higher), _analysis_result_rank(lower))

    def test_event_ticker_prefix_prefers_event_ticker_field(self) -> None:
        market = Market(
            id="KXMLBGAME-26APR121610TEXLAD-LAD",
            event_ticker="KXMLBGAME-26APR121610TEXLAD",
            question="Test",
        )
        self.assertEqual(
            _event_ticker_prefix(market),
            "KXMLBGAME-26APR121610TEXLAD",
        )

    def test_event_ticker_prefix_falls_back_to_market_id_prefix(self) -> None:
        market = Market(
            id="KXINXU-26APR13H1600-T6774.9999",
            question="Test",
        )
        self.assertEqual(
            _event_ticker_prefix(market),
            "KXINXU-26APR13H1600",
        )

    def test_event_concentration_blocked_when_event_cap_reached(self) -> None:
        self.assertTrue(
            _event_concentration_blocked(
                max_bets_per_event=2,
                open_other_positions_count=1,
                cycle_other_attempts_count=1,
            )
        )
        self.assertFalse(
            _event_concentration_blocked(
                max_bets_per_event=2,
                open_other_positions_count=1,
                cycle_other_attempts_count=0,
            )
        )

    def test_daily_trade_and_drawdown_caps(self) -> None:
        self.assertTrue(
            _daily_trade_cap_reached(daily_trade_count=15, max_trades_per_day=15)
        )
        self.assertFalse(
            _daily_trade_cap_reached(daily_trade_count=14, max_trades_per_day=15)
        )
        self.assertEqual(
            _daily_balance_delta_usdc(day_start_balance=100.0, current_balance=87.5),
            -12.5,
        )
        self.assertTrue(
            _daily_drawdown_cap_reached(
                daily_balance_delta=-30.0,
                max_daily_drawdown_usdc=30.0,
            )
        )
        self.assertFalse(
            _daily_drawdown_cap_reached(
                daily_balance_delta=-29.9,
                max_daily_drawdown_usdc=30.0,
            )
        )

    def test_market_model_exposes_volume_24h_field(self) -> None:
        market = Market(id="m-volume", question="Volume test", volume_24h=123.0)
        self.assertEqual(market.volume_24h, 123.0)

    def test_build_execution_audit_omits_none_values(self) -> None:
        payload = _build_execution_audit(
            decision_phase="order_submission",
            decision_terminal=True,
            final_action="skip",
            final_reason="test_reason",
            nullable_value=None,
            kept_value=3,
        )
        self.assertEqual(payload["decision_phase"], "order_submission")
        self.assertTrue(payload["decision_terminal"])
        self.assertEqual(payload["final_action"], "skip")
        self.assertEqual(payload["final_reason"], "test_reason")
        self.assertEqual(payload["kept_value"], 3)
        self.assertNotIn("nullable_value", payload)

    def test_build_execution_audit_normalizes_legacy_alias_keys(self) -> None:
        payload = _build_execution_audit(
            final_reason="test",
            amount_usdc=5.0,
            score_value=0.42,
            confidence_gate_override_edge=0.09,
            confidence_gate_override_market_edge=0.07,
            implied_prob=0.51,
            edge=0.11,
        )
        self.assertEqual(payload["bet_amount_usdc"], 5.0)
        self.assertEqual(payload["score_final"], 0.42)
        self.assertEqual(payload["override_edge"], 0.09)
        self.assertEqual(payload["market_edge"], 0.07)
        self.assertEqual(payload["implied_prob_market"], 0.51)
        self.assertEqual(payload["edge_market"], 0.11)
        self.assertNotIn("amount_usdc", payload)
        self.assertNotIn("score_value", payload)
        self.assertNotIn("confidence_gate_override_edge", payload)
        self.assertNotIn("confidence_gate_override_market_edge", payload)
        self.assertNotIn("implied_prob", payload)
        self.assertNotIn("edge", payload)

    def test_build_execution_audit_infers_rejection_stage(self) -> None:
        payload = _build_execution_audit(
            decision_terminal=True,
            final_action="skip",
            final_reason="score_gate_blocked",
        )
        self.assertEqual(payload.get("rejection_stage"), "score_gate")

    def test_build_execution_audit_keeps_explicit_score_breakdown(self) -> None:
        payload = _build_execution_audit(
            decision_terminal=True,
            final_action="skip",
            final_reason="score_gate_blocked",
            score_breakdown={"final_score": 0.28, "score_threshold": 0.38},
            score_final=0.28,
        )
        self.assertEqual(
            payload.get("score_breakdown"),
            {"final_score": 0.28, "score_threshold": 0.38},
        )
        self.assertEqual(payload.get("score_final"), 0.28)

    def test_parse_exchange_position_row_extracts_signed_position(self) -> None:
        parsed = _parse_exchange_position_row(
            {
                "ticker": "KXTEST-1",
                "position": -4,
                "market_exposure_dollars": 2.75,
            }
        )
        self.assertEqual(parsed, ("KXTEST-1", "NO", 2.75, 4))

    def test_pre_analysis_opportunity_score_penalizes_churned_speech_market(self) -> None:
        market = Market(
            id="KXPERSONMENTION-26APR09-TERM",
            question="Will candidate mention term?",
            category="politics",
            liquidity_usdc=600.0,
            outcomes=[MarketOutcome(name="YES", price=0.50), MarketOutcome(name="NO", price=0.50)],
            close_time=datetime.now(timezone.utc) + timedelta(hours=12),
            resolution_criteria="Official transcript source",
        )
        clean_state = MarketState(market_id=market.id, analysis_count=1, non_actionable_streak=0)
        churned_state = MarketState(
            market_id=market.id,
            analysis_count=15,
            non_actionable_streak=9,
            last_terminal_outcome="no_trade_recommended",
        )
        settings = Settings()
        clean_score, _ = _pre_analysis_opportunity_score(
            market,
            clean_state,
            settings,
            traded_before=True,
        )
        churned_score, breakdown = _pre_analysis_opportunity_score(
            market,
            churned_state,
            settings,
            traded_before=False,
        )
        self.assertLess(churned_score, clean_score)
        self.assertGreater(breakdown["pre_score_family_penalty"], 0.0)
        self.assertGreater(breakdown["pre_score_non_actionable_penalty"], 0.0)

    def test_pre_analysis_hard_rejection_blocks_repeated_non_actionable_family(self) -> None:
        market = Market(
            id="KXPERSONMENTION-26APR09-TERM",
            question="Will candidate mention term?",
            category="politics",
            outcomes=[MarketOutcome(name="YES", price=0.50), MarketOutcome(name="NO", price=0.50)],
            liquidity_usdc=400.0,
        )
        state = MarketState(
            market_id=market.id,
            analysis_count=12,
            non_actionable_streak=7,
            last_terminal_outcome="evidence_quality_below_min",
        )
        rejected, reason, metadata = _pre_analysis_hard_rejection(
            market=market,
            state=state,
            settings=Settings(),
            traded_before=False,
        )
        self.assertTrue(rejected)
        self.assertEqual(reason, "pre_analysis_repeated_non_actionable_market")
        self.assertEqual(metadata["pre_analysis_hard_reject_family"], "speech")

    def test_pre_analysis_hard_rejection_blocks_repeated_non_actionable_generic_bin(self) -> None:
        market = Market(
            id="KXNASDAQ100-26APR10H1600-B25250",
            question="Will the Nasdaq-100 be between 25200 and 25299.99?",
            category="finance",
            outcomes=[MarketOutcome(name="YES", price=0.50), MarketOutcome(name="NO", price=0.50)],
            liquidity_usdc=400.0,
        )
        state = MarketState(
            market_id=market.id,
            analysis_count=12,
            non_actionable_streak=7,
            last_terminal_outcome="no_trade_recommended",
        )
        rejected, reason, metadata = _pre_analysis_hard_rejection(
            market=market,
            state=state,
            settings=Settings(),
            traded_before=False,
        )
        self.assertTrue(rejected)
        self.assertEqual(reason, "pre_analysis_repeated_non_actionable_bin_market")
        self.assertEqual(metadata["pre_analysis_hard_reject_family"], "generic")

    def test_pre_analysis_hard_rejection_blocks_zero_action_family(self) -> None:
        market = Market(
            id="KXPERSONMENTION-26APR09-TERM",
            question="Will candidate mention term?",
            category="politics",
            outcomes=[MarketOutcome(name="YES", price=0.50), MarketOutcome(name="NO", price=0.50)],
            close_time=datetime.now(timezone.utc) + timedelta(hours=8),
            resolution_criteria="Official transcript source",
        )
        state = MarketState(
            market_id=market.id,
            analysis_count=6,
            non_actionable_streak=4,
            last_terminal_outcome="no_trade_recommended",
        )
        settings = Settings(
            PRE_ANALYSIS_HARD_REJECTION_ENABLED=True,
            PRE_ANALYSIS_ZERO_ACTION_FAMILY_BLOCK_ENABLED=True,
            PRE_ANALYSIS_ZERO_ACTION_FAMILY_MIN_SAMPLES=20,
            PRE_ANALYSIS_HARD_REJECTION_FAMILIES=("speech", "mention"),
        )
        rejected, reason, metadata = _pre_analysis_hard_rejection(
            market=market,
            state=state,
            settings=settings,
            traded_before=False,
            family_action_stats={"sample_size": 25, "action_rate": 0.0},
        )
        self.assertTrue(rejected)
        self.assertEqual(reason, "pre_analysis_zero_action_family")
        self.assertEqual(metadata["pre_analysis_hard_reject_family"], "speech")

    def test_pre_analysis_hard_rejection_blocks_fallback_edge_high_churn(self) -> None:
        market = Market(
            id="KXBTCD-26APR0917-T70499.99",
            question="Bitcoin threshold",
            category="crypto",
            outcomes=[MarketOutcome(name="YES", price=0.60), MarketOutcome(name="NO", price=0.40)],
            liquidity_usdc=500.0,
        )
        state = MarketState(
            market_id=market.id,
            analysis_count=4,
            non_actionable_streak=3,
            last_terminal_outcome="no_trade_recommended",
        )
        rejected, reason, metadata = _pre_analysis_hard_rejection(
            market=market,
            state=state,
            settings=Settings(),
            traded_before=False,
            had_recent_fallback_edge=True,
        )
        self.assertTrue(rejected)
        self.assertEqual(reason, "pre_analysis_fallback_edge_high_churn")
        self.assertTrue(metadata["pre_analysis_hard_reject_had_recent_fallback_edge"])

    def test_pre_analysis_hard_rejection_blocks_repeated_churn_market(self) -> None:
        market = Market(
            id="KXWTI-26APR14-T96.99",
            question="WTI settlement threshold",
            category="commodities",
            outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
            liquidity_usdc=1200.0,
        )
        state = MarketState(
            market_id=market.id,
            analysis_count=5,
            non_actionable_streak=3,
            last_terminal_outcome="manual_skip",
        )
        rejected, reason, metadata = _pre_analysis_hard_rejection(
            market=market,
            state=state,
            settings=Settings(),
            traded_before=False,
        )
        self.assertTrue(rejected)
        self.assertEqual(reason, "pre_analysis_repeated_churn_market")
        self.assertEqual(metadata["pre_analysis_hard_reject_analysis_count"], 5)

    def test_pre_analysis_hard_rejection_blocks_unprofitable_crypto_fallback_family(self) -> None:
        market = Market(
            id="KXBTCD-26APR1217-T70999.99",
            question="Bitcoin threshold",
            category="crypto",
            outcomes=[MarketOutcome(name="YES", price=0.60), MarketOutcome(name="NO", price=0.40)],
            liquidity_usdc=500.0,
        )
        state = MarketState(
            market_id=market.id,
            analysis_count=4,
            non_actionable_streak=1,
            last_terminal_outcome="no_trade_recommended",
        )
        rejected, reason, metadata = _pre_analysis_hard_rejection(
            market=market,
            state=state,
            settings=Settings(
                PRE_ANALYSIS_CRYPTO_NEGATIVE_PNL_BLOCK_ENABLED=True,
                PRE_ANALYSIS_CRYPTO_NEGATIVE_PNL_THRESHOLD=0.0,
                PRE_ANALYSIS_CRYPTO_FALLBACK_RATE_BLOCK_THRESHOLD=0.55,
                PRE_ANALYSIS_CRYPTO_MIN_SAMPLES=20,
            ),
            traded_before=False,
            historical_family_stats={
                "sample_size": 25,
                "pnl_total": -0.4,
            },
            fallback_family_edge_rate=0.70,
            fallback_family_sample_size=30,
        )
        self.assertTrue(rejected)
        self.assertEqual(reason, "pre_analysis_crypto_historically_unprofitable")
        self.assertEqual(metadata["pre_analysis_hard_reject_family"], "crypto")

    def test_pre_analysis_opportunity_score_penalizes_generic_bin_churn(self) -> None:
        market = Market(
            id="KXNASDAQ100-26APR10H1600-B25350",
            question="Will the Nasdaq-100 be between 25300 and 25399.99?",
            category="finance",
            liquidity_usdc=600.0,
            outcomes=[MarketOutcome(name="YES", price=0.50), MarketOutcome(name="NO", price=0.50)],
            close_time=datetime.now(timezone.utc) + timedelta(hours=12),
            resolution_criteria="Official close print",
        )
        state = MarketState(
            market_id=market.id,
            analysis_count=10,
            non_actionable_streak=5,
            last_terminal_outcome="no_trade_recommended",
        )
        score, breakdown = _pre_analysis_opportunity_score(
            market,
            state,
            Settings(),
            traded_before=False,
        )
        self.assertLess(score, 0.8)
        self.assertGreater(breakdown["pre_score_generic_bin_penalty"], 0.0)
        self.assertGreater(breakdown["pre_score_churn_penalty"], 0.0)

    def test_pre_analysis_opportunity_score_penalizes_high_fallback_family_rate(self) -> None:
        market = Market(
            id="KXBTCD-26APR0917-T70499.99",
            question="Bitcoin threshold",
            category="crypto",
            liquidity_usdc=800.0,
            outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
            close_time=datetime.now(timezone.utc) + timedelta(hours=10),
            resolution_criteria="Official settlement source",
        )
        settings = Settings()
        clean_score, _ = _pre_analysis_opportunity_score(
            market,
            None,
            settings,
            traded_before=False,
            fallback_family_edge_rate=0.20,
            fallback_family_sample_size=120,
        )
        penalized_score, breakdown = _pre_analysis_opportunity_score(
            market,
            None,
            settings,
            traded_before=False,
            fallback_family_edge_rate=0.92,
            fallback_family_sample_size=120,
        )
        self.assertLess(penalized_score, clean_score)
        self.assertGreater(breakdown["pre_score_fallback_family_penalty"], 0.0)

    def test_pre_analysis_opportunity_score_scales_fallback_penalty_for_profitable_family(self) -> None:
        market = Market(
            id="KXBTCD-26APR0917-T70499.99",
            question="Bitcoin threshold",
            category="crypto",
            liquidity_usdc=800.0,
            outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
            close_time=datetime.now(timezone.utc) + timedelta(hours=10),
            resolution_criteria="Official settlement source",
        )
        settings = Settings()
        _, baseline_breakdown = _pre_analysis_opportunity_score(
            market,
            None,
            settings,
            traded_before=False,
            fallback_family_edge_rate=0.92,
            fallback_family_sample_size=120,
            historical_family_stats={"sample_size": 40, "win_rate": 0.45, "pnl_total": -5.0},
        )
        _, profitable_breakdown = _pre_analysis_opportunity_score(
            market,
            None,
            settings,
            traded_before=False,
            fallback_family_edge_rate=0.92,
            fallback_family_sample_size=120,
            historical_family_stats={"sample_size": 40, "win_rate": 0.60, "pnl_total": 12.0},
        )
        self.assertGreater(baseline_breakdown["pre_score_fallback_family_penalty"], 0.0)
        self.assertEqual(profitable_breakdown["pre_score_fallback_family_penalty_scale"], 0.5)
        self.assertAlmostEqual(
            profitable_breakdown["pre_score_fallback_family_penalty"],
            baseline_breakdown["pre_score_fallback_family_penalty"] * 0.5,
            places=6,
        )

    def test_pre_analysis_opportunity_score_adds_post_event_bonus(self) -> None:
        settings = Settings()
        market_past = Market(
            id="KXMLBGAME-PAST",
            question="Post-event market",
            category="sports",
            liquidity_usdc=800.0,
            outcomes=[MarketOutcome(name="YES", price=0.52), MarketOutcome(name="NO", price=0.48)],
            close_time=datetime.now(timezone.utc) - timedelta(hours=3),
            resolution_criteria="Official box score",
        )
        market_future = market_past.model_copy(
            update={"id": "KXMLBGAME-FUTURE", "close_time": datetime.now(timezone.utc) + timedelta(hours=3)}
        )
        past_score, past_breakdown = _pre_analysis_opportunity_score(
            market_past,
            None,
            settings,
            traded_before=False,
        )
        future_score, future_breakdown = _pre_analysis_opportunity_score(
            market_future,
            None,
            settings,
            traded_before=False,
        )
        self.assertEqual(past_breakdown["pre_score_post_event_bonus"], 0.10)
        self.assertEqual(future_breakdown["pre_score_post_event_bonus"], 0.0)
        self.assertGreater(past_score, future_score)

    def test_resolve_dynamic_analysis_candidate_cap_reduces_when_best_score_low(self) -> None:
        settings = Settings(
            MAX_MARKETS_PER_CYCLE=6,
            PRE_ANALYSIS_MUST_ANALYZE_THRESHOLD=0.50,
            PRE_ANALYSIS_REDUCED_MAX_CANDIDATES=3,
        )
        cap, applied = _resolve_dynamic_analysis_candidate_cap(
            settings=settings,
            best_pre_analysis_score=0.42,
        )
        self.assertEqual(cap, 3)
        self.assertTrue(applied)

    def test_resolve_dynamic_analysis_candidate_cap_keeps_default_when_score_high(self) -> None:
        settings = Settings(
            MAX_MARKETS_PER_CYCLE=6,
            PRE_ANALYSIS_MUST_ANALYZE_THRESHOLD=0.50,
            PRE_ANALYSIS_REDUCED_MAX_CANDIDATES=3,
        )
        cap, applied = _resolve_dynamic_analysis_candidate_cap(
            settings=settings,
            best_pre_analysis_score=0.75,
        )
        self.assertEqual(cap, 6)
        self.assertFalse(applied)

    def test_pre_analysis_opportunity_score_penalizes_weak_historical_family_performance(self) -> None:
        market = Market(
            id="KXBTCD-26APR0917-T70499.99",
            question="Bitcoin threshold",
            category="crypto",
            liquidity_usdc=800.0,
            outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
            close_time=datetime.now(timezone.utc) + timedelta(hours=10),
            resolution_criteria="Official settlement source",
        )
        settings = Settings(
            PRE_ANALYSIS_HISTORICAL_FAMILY_MIN_SAMPLES=10,
            PRE_ANALYSIS_HISTORICAL_FAMILY_WIN_RATE_THRESHOLD=0.45,
            PRE_ANALYSIS_HISTORICAL_FAMILY_PENALTY=0.12,
        )
        clean_score, _ = _pre_analysis_opportunity_score(
            market,
            None,
            settings,
            traded_before=False,
            historical_family_stats={"sample_size": 30, "win_rate": 0.58, "pnl_total": 8.0},
        )
        penalized_score, breakdown = _pre_analysis_opportunity_score(
            market,
            None,
            settings,
            traded_before=False,
            historical_family_stats={"sample_size": 30, "win_rate": 0.35, "pnl_total": -20.0},
        )
        self.assertLess(penalized_score, clean_score)
        self.assertGreater(breakdown["pre_score_historical_family_penalty"], 0.0)

    def test_pre_analysis_opportunity_score_applies_severe_negative_pnl_penalty(self) -> None:
        market = Market(
            id="KXWTI-26APR13-T100.99",
            question="Will WTI settle above threshold?",
            category="commodities",
            liquidity_usdc=800.0,
            outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
            close_time=datetime.now(timezone.utc) + timedelta(hours=10),
            resolution_criteria="Official settlement source",
        )
        settings = Settings(
            PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_THRESHOLD=-10.0,
            PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_PENALTY=0.10,
            PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_SEVERE_THRESHOLD=-15.0,
            PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_SEVERE_PENALTY=0.15,
        )
        _, breakdown = _pre_analysis_opportunity_score(
            market,
            None,
            settings,
            traded_before=False,
            historical_family_stats={
                "sample_size": 40,
                "win_rate": 0.45,
                "pnl_total": -21.0,
            },
        )
        self.assertEqual(breakdown["pre_score_historical_family_pnl_penalty"], 0.15)

    def test_is_coinflip_signal_detects_low_information_decision(self) -> None:
        weak_decision = TradeDecision(
            should_trade=False,
            outcome="YES",
            confidence=0.55,
            bet_size_pct=0.0,
            reasoning="uncertain",
            evidence_quality=0.59,
        )
        strong_decision = weak_decision.model_copy(
            update={"confidence": 0.62, "evidence_quality": 0.75}
        )
        self.assertTrue(_is_coinflip_signal(weak_decision))
        self.assertFalse(_is_coinflip_signal(strong_decision))

    def test_passes_refreshed_edge_guard_blocks_eroded_edge(self) -> None:
        market = Market(
            id="m-refresh",
            question="Will team win?",
            outcomes=[MarketOutcome(name="YES", price=0.70), MarketOutcome(name="NO", price=0.30)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.72,
            bet_size_pct=0.4,
            reasoning="test",
        )
        ok, implied_prob, edge, reason = _passes_refreshed_edge_guard(
            market,
            decision,
            Settings(MIN_EDGE=0.05),
        )
        self.assertFalse(ok)
        self.assertEqual(implied_prob, 0.70)
        self.assertAlmostEqual(edge or 0.0, 0.02, places=6)
        self.assertIn("below min", reason)

    def test_filter_markets(self) -> None:
        markets = [
            Market(id="1", question="Q1", liquidity_usdc=50, category="sports"),
            Market(id="2", question="Q2", liquidity_usdc=150, category="sports"),
            Market(id="3", question="Q3", liquidity_usdc=200, category="politics"),
        ]
        filtered = _filter_markets(
            markets,
            min_liquidity=100,
            allowlist=("sports",),
            blocklist=("politics",),
        )
        self.assertEqual([m.id for m in filtered], ["2"])

    def test_filter_markets_respects_family_blocklist(self) -> None:
        markets = [
            Market(id="KXHIGHCHI-26APR11-T58", question="Will high temp exceed 58?", category="weather"),
            Market(id="KXBTCD-26APR1117-T73249.99", question="BTC threshold", category="crypto"),
            Market(id="KXKBOGAME-26APR090530SAMKIA-KIA", question="Baseball winner", category="sports"),
        ]
        filtered = _filter_markets(
            markets,
            min_liquidity=0,
            allowlist=(),
            blocklist=(),
            family_blocklist=("weather", "crypto"),
        )
        self.assertEqual([m.id for m in filtered], ["KXKBOGAME-26APR090530SAMKIA-KIA"])

    def test_filter_markets_by_close_date(self) -> None:
        now = datetime.now(timezone.utc)
        markets = [
            Market(id="1", question="Q1", close_time=now + timedelta(hours=6)),
            Market(id="2", question="Q2", close_time=now + timedelta(days=3)),
            Market(id="3", question="Q3", close_time=now + timedelta(days=10)),
            Market(id="4", question="Q4", close_time=None),
        ]
        # Filter: only markets closing between 1 and 7 days from now
        filtered = _filter_markets(
            markets,
            min_liquidity=0,
            allowlist=(),
            blocklist=(),
            min_close_days=1,
            max_close_days=7,
        )
        # Market 1 closes too soon (<1 day), Market 3 closes too far (>7 days)
        # Market 4 has no close_time, so it passes (no filter applied)
        self.assertEqual([m.id for m in filtered], ["2", "4"])

    def test_filter_markets_with_zero_min_close_days_applies_lower_bound(self) -> None:
        now = datetime.now(timezone.utc)
        markets = [
            Market(id="closed", question="Closed", close_time=now - timedelta(seconds=1)),
            Market(id="future", question="Future", close_time=now + timedelta(hours=1)),
        ]
        filtered = _filter_markets(
            markets,
            min_liquidity=0,
            allowlist=(),
            blocklist=(),
            min_close_days=0,
            max_close_days=1,
        )
        self.assertEqual([m.id for m in filtered], ["future"])

    def test_build_kalshi_market_fetch_window_preserves_zero_day_start(self) -> None:
        start, end = _build_kalshi_market_fetch_window(0, 1)
        self.assertIsNotNone(start)
        self.assertIsNotNone(end)
        self.assertLess(start, end)

    def test_filter_markets_max_close_days_only(self) -> None:
        now = datetime.now(timezone.utc)
        markets = [
            Market(id="1", question="Q1", close_time=now + timedelta(hours=12)),
            Market(id="2", question="Q2", close_time=now + timedelta(days=5)),
        ]
        # Only set max_close_days (markets closing within 3 days)
        filtered = _filter_markets(
            markets,
            min_liquidity=0,
            allowlist=(),
            blocklist=(),
            max_close_days=3,
        )
        self.assertEqual([m.id for m in filtered], ["1"])

    def test_calculate_bet(self) -> None:
        self.assertEqual(_calculate_bet(100, 0.5), 50)
        self.assertEqual(_calculate_bet(100, -1), 0)
        self.assertEqual(_calculate_bet(100, 2), 100)

    def test_filter_markets_populates_skip_counters(self) -> None:
        now = datetime.now(timezone.utc)
        markets = [
            Market(id="open", question="Open market", category="sports", liquidity_usdc=200, close_time=now + timedelta(days=2)),
            Market(id="low", question="Low liquidity", category="sports", liquidity_usdc=10, close_time=now + timedelta(days=2)),
            Market(id="blocked", question="Blocked category", category="politics", liquidity_usdc=200, close_time=now + timedelta(days=2)),
            Market(id="soon", question="Closing soon", category="sports", liquidity_usdc=200, close_time=now + timedelta(hours=4)),
        ]
        stats: dict[str, int] = {}
        filtered = _filter_markets(
            markets,
            min_liquidity=100,
            allowlist=(),
            blocklist=("politics",),
            min_close_days=1,
            stats=stats,
        )
        self.assertEqual([m.id for m in filtered], ["open"])
        self.assertEqual(stats["kept"], 1)
        self.assertEqual(stats["skipped_liquidity"], 1)
        self.assertEqual(stats["skipped_blocklist"], 1)
        self.assertEqual(stats["skipped_close_too_soon"], 1)

    def test_filter_markets_applies_ticker_prefix_blocklist(self) -> None:
        markets = [
            Market(id="KXBTC15M-26APR061800-00", question="15m market", liquidity_usdc=200),
            Market(id="KXBTCD-26APR0717-T70000", question="Daily market", liquidity_usdc=200),
        ]
        stats: dict[str, int] = {}
        filtered = _filter_markets(
            markets,
            min_liquidity=100,
            allowlist=(),
            blocklist=(),
            ticker_prefix_blocklist=("KXBTC15M-",),
            stats=stats,
        )
        self.assertEqual([m.id for m in filtered], ["KXBTCD-26APR0717-T70000"])
        self.assertEqual(stats["skipped_ticker_prefix_blocklist"], 1)

    def test_filter_markets_blocks_survivor_mention_prefix(self) -> None:
        markets = [
            Market(id="KXSURVIVORMENTION-26APR09-SHEL", question="Mention market", liquidity_usdc=200),
            Market(id="KXBTCD-26APR0717-T70000", question="Daily market", liquidity_usdc=200),
        ]
        stats: dict[str, int] = {}
        filtered = _filter_markets(
            markets,
            min_liquidity=100,
            allowlist=(),
            blocklist=(),
            ticker_prefix_blocklist=("KXSURVIVORMENTION-",),
            stats=stats,
        )
        self.assertEqual([m.id for m in filtered], ["KXBTCD-26APR0717-T70000"])
        self.assertEqual(stats["skipped_ticker_prefix_blocklist"], 1)

    def test_filter_markets_skips_weather_bin_markets_when_enabled(self) -> None:
        markets = [
            Market(id="KXLOWTCHI-99DEC31-B33.5", question="Bin market", liquidity_usdc=200),
            Market(id="KXHIGHMIA-99DEC31-B76.5", question="Miami bin market", liquidity_usdc=200),
            Market(id="KXLOWTCHI-99DEC31-T33", question="Threshold market", liquidity_usdc=200),
        ]
        stats: dict[str, int] = {}
        filtered = _filter_markets(
            markets,
            min_liquidity=100,
            allowlist=(),
            blocklist=(),
            skip_weather_bin_markets=True,
            stats=stats,
        )
        self.assertEqual([m.id for m in filtered], ["KXLOWTCHI-99DEC31-T33"])
        self.assertEqual(stats["skipped_weather_bin_markets"], 2)

    def test_filter_markets_blocks_weather_markets_when_enabled(self) -> None:
        markets = [
            Market(
                id="KXLOWTCHI-99DEC31-T33",
                question="Will the low temp in Chicago be >33°?",
                category="weather",
                liquidity_usdc=200,
            ),
            Market(
                id="KXBTCD-26APR0810-T71699.99",
                question="Bitcoin above threshold?",
                category="crypto",
                liquidity_usdc=200,
            ),
        ]
        stats: dict[str, int] = {}
        filtered = _filter_markets(
            markets,
            min_liquidity=100,
            allowlist=(),
            blocklist=("weather",),
            stats=stats,
        )
        self.assertEqual([m.id for m in filtered], ["KXBTCD-26APR0810-T71699.99"])
        self.assertEqual(stats["skipped_blocklist"], 1)

    def test_filter_markets_blocks_weather_family_when_category_missing(self) -> None:
        markets = [
            Market(
                id="weather-uncategorized",
                question="Will rainfall exceed 2 inches in Miami tomorrow?",
                category=None,
                liquidity_usdc=200,
            ),
            Market(
                id="crypto-kept",
                question="Will BTC close above 70k?",
                category=None,
                liquidity_usdc=200,
            ),
        ]
        stats: dict[str, int] = {}
        filtered = _filter_markets(
            markets,
            min_liquidity=100,
            allowlist=(),
            blocklist=("weather",),
            stats=stats,
        )
        self.assertEqual([m.id for m in filtered], ["crypto-kept"])
        self.assertEqual(stats["skipped_blocklist"], 1)

    def test_weather_ticker_detection_helpers_match_expected_patterns(self) -> None:
        self.assertTrue(_is_weather_bin_market("KXHIGHMIA-99DEC31-B76.5"))
        self.assertTrue(_is_weather_bin_market("KXLOWTPHX-99DEC31-B67"))
        self.assertFalse(_is_weather_bin_market("KXLOWTPHX-99DEC31-T67"))
        self.assertTrue(_is_weather_market_by_ticker("KXHIGHCHI-99DEC31-T70"))
        self.assertTrue(_is_weather_market_by_ticker("KXLOWTLV-99DEC31-B64"))
        self.assertFalse(_is_weather_market_by_ticker("KXBTCD-26APR0810-T71699.99"))

    def test_crypto_bin_ticker_detection_helper_matches_expected_patterns(self) -> None:
        self.assertTrue(_is_crypto_bin_market("KXBTC-26APR0814-B71650"))
        self.assertTrue(_is_crypto_bin_market("KXETHD-26APR08-B2000"))
        self.assertFalse(_is_crypto_bin_market("KXBTCD-26APR0814-T71650"))

    def test_filter_markets_skips_crypto_bin_markets_when_enabled(self) -> None:
        markets = [
            Market(id="KXBTC-26APR0814-B71650", question="BTC bin", liquidity_usdc=200),
            Market(id="KXETHD-26APR08-B2000", question="ETH bin", liquidity_usdc=200),
            Market(id="KXBTCD-26APR0814-T71650", question="BTC threshold", liquidity_usdc=200),
        ]
        stats: dict[str, int] = {}
        filtered = _filter_markets(
            markets,
            min_liquidity=100,
            allowlist=(),
            blocklist=(),
            skip_crypto_bin_markets=True,
            stats=stats,
        )
        self.assertEqual([m.id for m in filtered], ["KXBTCD-26APR0814-T71650"])
        self.assertEqual(stats["skipped_crypto_bin_markets"], 2)

    def test_extract_order_cancel_reason_prefers_explicit_reason_keys(self) -> None:
        payload = {"status": "canceled", "cancel_reason": "price moved"}
        self.assertEqual(_extract_order_cancel_reason(payload), "price moved")

    def test_extract_order_fill_count_reads_nested_order_field(self) -> None:
        payload = {"order": {"status": "canceled", "fill_count_fp": "0.00"}}
        self.assertEqual(_extract_order_fill_count(payload), 0.0)

    def test_filter_markets_treats_null_liquidity_as_zero(self) -> None:
        markets = [
            Market(id="null-liq", question="Null liquidity", liquidity_usdc=None),
            Market(id="ok-liq", question="Sufficient liquidity", liquidity_usdc=150),
        ]
        filtered = _filter_markets(
            markets,
            min_liquidity=100,
            allowlist=(),
            blocklist=(),
        )
        self.assertEqual([m.id for m in filtered], ["ok-liq"])

    def test_filter_markets_applies_volume_and_extreme_price_filters(self) -> None:
        markets = [
            Market(
                id="low-volume",
                question="Low volume",
                outcomes=[MarketOutcome(name="YES", price=0.50), MarketOutcome(name="NO", price=0.50)],
                liquidity_usdc=200,
                volume_24h=5,
            ),
            Market(
                id="extreme-price",
                question="Extreme price",
                outcomes=[MarketOutcome(name="YES", price=0.99), MarketOutcome(name="NO", price=0.01)],
                liquidity_usdc=200,
                volume_24h=100,
            ),
            Market(
                id="kept",
                question="Kept market",
                outcomes=[MarketOutcome(name="YES", price=0.52), MarketOutcome(name="NO", price=0.48)],
                liquidity_usdc=200,
                volume_24h=100,
            ),
        ]
        stats: dict[str, int] = {}
        filtered = _filter_markets(
            markets,
            min_liquidity=100,
            allowlist=(),
            blocklist=(),
            stats=stats,
            min_volume_24h=10,
            extreme_yes_price_lower=0.05,
            extreme_yes_price_upper=0.95,
        )
        self.assertEqual([m.id for m in filtered], ["kept"])
        self.assertEqual(stats["skipped_volume_24h"], 1)
        self.assertEqual(stats["skipped_extreme_price"], 1)

    def test_filter_markets_allows_open_interest_override_when_volume_is_low(self) -> None:
        markets = [
            Market(
                id="oi-pass",
                question="Open interest fallback pass",
                outcomes=[MarketOutcome(name="YES", price=0.51), MarketOutcome(name="NO", price=0.49)],
                liquidity_usdc=120,
                volume_24h=2,
                open_interest=120,
            ),
            Market(
                id="oi-fail",
                question="Fails both activity thresholds",
                outcomes=[MarketOutcome(name="YES", price=0.51), MarketOutcome(name="NO", price=0.49)],
                liquidity_usdc=120,
                volume_24h=2,
                open_interest=8,
            ),
        ]
        stats: dict[str, int] = {}
        filtered = _filter_markets(
            markets,
            min_liquidity=100,
            allowlist=(),
            blocklist=(),
            stats=stats,
            min_volume_24h=10,
            min_open_interest=50,
        )
        self.assertEqual([m.id for m in filtered], ["oi-pass"])
        self.assertEqual(stats["activity_passed_by_open_interest"], 1)
        self.assertEqual(stats["skipped_open_interest"], 1)

    def test_filter_markets_applies_tradeable_price_band(self) -> None:
        markets = [
            Market(
                id="too-cheap",
                question="Too cheap market",
                outcomes=[MarketOutcome(name="YES", price=0.02), MarketOutcome(name="NO", price=0.98)],
                liquidity_usdc=200,
            ),
            Market(
                id="too-expensive",
                question="Too expensive market",
                outcomes=[MarketOutcome(name="YES", price=0.98), MarketOutcome(name="NO", price=0.02)],
                liquidity_usdc=200,
            ),
            Market(
                id="tradeable",
                question="Tradeable market",
                outcomes=[MarketOutcome(name="YES", price=0.52), MarketOutcome(name="NO", price=0.48)],
                liquidity_usdc=200,
            ),
        ]
        stats: dict[str, int] = {}
        filtered = _filter_markets(
            markets,
            min_liquidity=100,
            allowlist=(),
            blocklist=(),
            stats=stats,
            min_tradeable_yes_price=0.05,
            max_tradeable_yes_price=0.95,
        )
        self.assertEqual([m.id for m in filtered], ["tradeable"])
        self.assertEqual(stats["skipped_untradeable_price"], 2)

    def test_ticker_resolution_date_parses_kalshi_style_token(self) -> None:
        parsed = _ticker_resolution_date("KXLOWTDC-26APR07-T44")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.year, 2026)
        self.assertEqual(parsed.month, 4)
        self.assertEqual(parsed.day, 7)

    def test_likely_resolved_by_ticker_date_flags_past_day(self) -> None:
        market = Market(id="KXLOWTDC-26APR07-T44", question="Q")
        now = datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)
        self.assertTrue(_is_likely_resolved_by_ticker_date(market, now))

    def test_collapse_event_ladders_keeps_most_informative_brackets(self) -> None:
        event_markets = [
            Market(id="m1", event_ticker="E1", question="Q", outcomes=[MarketOutcome(name="YES", price=0.10)]),
            Market(id="m2", event_ticker="E1", question="Q", outcomes=[MarketOutcome(name="YES", price=0.49)]),
            Market(id="m3", event_ticker="E1", question="Q", outcomes=[MarketOutcome(name="YES", price=0.51)]),
            Market(id="m4", event_ticker="E1", question="Q", outcomes=[MarketOutcome(name="YES", price=0.90)]),
            Market(id="m5", event_ticker="E1", question="Q", outcomes=[MarketOutcome(name="YES", price=0.70)]),
            Market(id="m6", event_ticker="E2", question="Q", outcomes=[MarketOutcome(name="YES", price=0.30)]),
            Market(id="no-event", question="Q", outcomes=[MarketOutcome(name="YES", price=0.60)]),
        ]
        collapsed = _collapse_event_ladders(
            event_markets,
            ladder_collapse_threshold=4,
            max_brackets_per_event=3,
        )
        collapsed_ids = {market.id for market in collapsed}
        self.assertEqual(len([m for m in collapsed if m.event_ticker == "E1"]), 3)
        self.assertIn("m2", collapsed_ids)
        self.assertIn("m3", collapsed_ids)
        self.assertIn("m5", collapsed_ids)
        self.assertIn("m6", collapsed_ids)
        self.assertIn("no-event", collapsed_ids)

    def test_cap_analysis_candidates_limits_list_size(self) -> None:
        candidates = [{"market": f"m{i}"} for i in range(6)]
        capped = _cap_analysis_candidates(candidates, max_markets_per_cycle=3)
        self.assertEqual(len(capped), 3)
        self.assertEqual(capped[0]["market"], "m0")
        self.assertEqual(capped[-1]["market"], "m2")

    def test_cap_analysis_candidates_uses_family_round_robin(self) -> None:
        candidates = [
            {
                "market": Market(
                    id="c1",
                    question="Will Bitcoin close above $70k?",
                    category="crypto",
                )
            },
            {
                "market": Market(
                    id="c2",
                    question="Will Ethereum close above $4k?",
                    category="crypto",
                )
            },
            {
                "market": Market(
                    id="c3",
                    question="Will Solana close above $200?",
                    category="crypto",
                )
            },
            {
                "market": Market(
                    id="s1",
                    question="Will the Lakers win tonight?",
                    category="sports",
                )
            },
            {
                "market": Market(
                    id="s2",
                    question="Will the Celtics win tonight?",
                    category="sports",
                )
            },
            {
                "market": Market(
                    id="p1",
                    question="Will candidate X win the election?",
                    category="politics",
                )
            },
        ]
        capped = _cap_analysis_candidates(candidates, max_markets_per_cycle=4)
        self.assertEqual([item["market"].id for item in capped], ["c1", "s1", "p1", "c2"])

    def test_cap_analysis_candidates_limits_weather_candidates(self) -> None:
        candidates = [
            {"market": Market(id="w1", question="Weather 1", category="weather")},
            {"market": Market(id="c1", question="Crypto 1", category="crypto")},
            {"market": Market(id="w2", question="Weather 2", category="weather")},
            {"market": Market(id="s1", question="Sports 1", category="sports")},
            {"market": Market(id="w3", question="Weather 3", category="weather")},
            {"market": Market(id="p1", question="Politics 1", category="politics")},
        ]
        capped = _cap_analysis_candidates(
            candidates,
            max_markets_per_cycle=5,
            max_weather_candidates_per_cycle=1,
        )
        self.assertEqual([item["market"].id for item in capped], ["w1", "c1", "p1", "s1"])

    def test_cap_analysis_candidates_limits_crypto_candidates(self) -> None:
        candidates = [
            {"market": Market(id="c1", question="BTC 1", category="crypto")},
            {"market": Market(id="w1", question="Weather 1", category="weather")},
            {"market": Market(id="c2", question="BTC 2", category="crypto")},
            {"market": Market(id="s1", question="Sports 1", category="sports")},
            {"market": Market(id="c3", question="BTC 3", category="crypto")},
            {"market": Market(id="p1", question="Politics 1", category="politics")},
            {"market": Market(id="c4", question="BTC 4", category="crypto")},
            {"market": Market(id="g1", question="Generic 1", category="business")},
        ]
        capped = _cap_analysis_candidates(
            candidates,
            max_markets_per_cycle=7,
            max_weather_candidates_per_cycle=1,
            max_crypto_candidates_per_cycle=3,
        )
        capped_ids = [item["market"].id for item in capped]
        self.assertEqual(len([market_id for market_id in capped_ids if market_id.startswith("c")]), 3)
        self.assertNotIn("c4", capped_ids)

    def test_cap_analysis_candidates_limits_music_and_speech_candidates(self) -> None:
        candidates = [
            {"market": Market(id="m1", question="Album streams question 1", category="music")},
            {"market": Market(id="sp1", question="Will person say phrase?", category="speech")},
            {"market": Market(id="m2", question="Album streams question 2", category="music")},
            {"market": Market(id="sp2", question="Will person mention topic?", category="speech")},
            {"market": Market(id="c1", question="BTC threshold", category="crypto")},
            {"market": Market(id="w1", question="Weather threshold", category="weather")},
            {"market": Market(id="p1", question="Politics", category="politics")},
        ]
        capped = _cap_analysis_candidates(
            candidates,
            max_markets_per_cycle=6,
            max_music_candidates_per_cycle=1,
            max_speech_candidates_per_cycle=1,
        )
        capped_ids = [item["market"].id for item in capped]
        self.assertIn("m1", capped_ids)
        self.assertNotIn("m2", capped_ids)
        self.assertIn("sp1", capped_ids)
        self.assertNotIn("sp2", capped_ids)

    def test_cap_analysis_candidates_prefers_lower_non_actionable_streak(self) -> None:
        candidates = [
            {
                "market": Market(id="w-high-streak", question="Weather A", category="weather"),
                "non_actionable_streak": 5,
            },
            {
                "market": Market(id="w-low-streak", question="Weather B", category="weather"),
                "non_actionable_streak": 1,
            },
            {
                "market": Market(id="c1", question="Crypto", category="crypto"),
                "non_actionable_streak": 0,
            },
        ]
        capped = _cap_analysis_candidates(candidates, max_markets_per_cycle=2)
        self.assertEqual([item["market"].id for item in capped], ["w-low-streak", "c1"])

    def test_cap_analysis_candidates_prefers_higher_pre_analysis_score_within_family(self) -> None:
        candidates = [
            {
                "market": Market(id="w-high-score", question="Weather A", category="weather"),
                "non_actionable_streak": 0,
                "pre_analysis_score": 0.90,
            },
            {
                "market": Market(id="w-low-score", question="Weather B", category="weather"),
                "non_actionable_streak": 0,
                "pre_analysis_score": 0.10,
            },
            {
                "market": Market(id="c1", question="Crypto", category="crypto"),
                "non_actionable_streak": 0,
                "pre_analysis_score": 0.70,
            },
        ]
        capped = _cap_analysis_candidates(candidates, max_markets_per_cycle=2)
        self.assertEqual([item["market"].id for item in capped], ["w-high-score", "c1"])

    def test_best_orderbook_sell_price(self) -> None:
        orderbook = {
            "sells": [
                {"optionIndex": 0, "price": 0.62},
                {"optionIndex": 1, "price": 0.44},
                {"optionIndex": 0, "price": 0.60},
            ]
        }
        self.assertAlmostEqual(_best_orderbook_sell_price(orderbook, 0) or 0.0, 0.60)
        self.assertAlmostEqual(_best_orderbook_sell_price(orderbook, 1) or 0.0, 0.44)
        self.assertIsNone(_best_orderbook_sell_price(orderbook, 2))

    def test_available_orderbook_sell_quantity_respects_price_limit(self) -> None:
        orderbook = {
            "sells": [
                {"optionIndex": 0, "price": 0.45, "quantity": 2},
                {"optionIndex": 0, "price": 0.50, "count": 3},
                {"optionIndex": 0, "price": 0.60, "size": 7},
                {"optionIndex": 1, "price": 0.44, "quantity": 5},
            ]
        }
        self.assertEqual(
            _available_orderbook_sell_quantity(orderbook, option_index=0, max_price=0.50),
            5.0,
        )
        self.assertEqual(
            _available_orderbook_sell_quantity(orderbook, option_index=0, max_price=None),
            12.0,
        )

    def test_log_settings_summary_includes_phase1_flags(self) -> None:
        settings = Settings(
            BAYESIAN_ENABLED=False,
            LMSR_ENABLED=False,
            KELLY_SIZING_ENABLED=True,
            KELLY_FRACTION_DEFAULT=0.2,
            KELLY_FRACTION_SHORT_HORIZON_HOURS=1,
            KELLY_FRACTION_SHORT_HORIZON=0.1,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        with patch("main.logger.info") as info_mock:
            _log_settings_summary(settings)

        self.assertTrue(info_mock.called)
        summary_data = {}
        strict_hint_data = {}
        for call in info_mock.call_args_list:
            data = call.kwargs.get("data") or {}
            if "dry_run" in data:
                summary_data = data
            if "effective_min_bet_pct" in data:
                strict_hint_data = data
        data = summary_data
        self.assertEqual(data.get("bayesian_enabled"), False)
        self.assertEqual(data.get("lmsr_enabled"), False)
        self.assertEqual(data.get("kelly_sizing_enabled"), True)
        self.assertEqual(data.get("kelly_fraction_default"), 0.2)
        self.assertEqual(data.get("kelly_fraction_short_horizon_hours"), 1)
        self.assertEqual(data.get("kelly_fraction_short_horizon"), 0.1)
        self.assertEqual(data.get("kelly_min_bet_policy"), "fallback_edge_scaling")
        self.assertGreater(strict_hint_data.get("effective_min_bet_pct", 0.0), 0.0)

    def test_compute_next_wakeup_seconds_uses_action_aware_cooldown(self) -> None:
        now = datetime.now(timezone.utc)
        market = Market(
            id="m-cooldown",
            question="Cooldown test",
            outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
            close_time=now + timedelta(days=2),
        )
        state = MarketState(
            market_id="m-cooldown",
            last_analysis=now - timedelta(minutes=20),
            analysis_count=1,
            last_confidence=0.55,
            confidence_trend=[0.55],
            last_terminal_outcome="no_trade_recommended",
        )
        state_manager = DummyStateManager({"m-cooldown": state})
        settings = Settings(
            REANALYSIS_COOLDOWN_HOURS=6,
            URGENT_REANALYSIS_DAYS_BEFORE_CLOSE=1,
            URGENT_REANALYSIS_COOLDOWN_HOURS=1,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        wakeup_seconds = _compute_next_wakeup_seconds(
            [market],
            state_manager,
            settings,
            now=now,
        )
        self.assertEqual(wakeup_seconds, 1)

    def test_fetch_markets_with_optional_server_filters_retries_filtered_then_unfiltered(self) -> None:
        now = datetime.now(timezone.utc)
        expected = [Market(id="m", question="Q")]
        client = self._DummyKalshiClient(
            [
                RuntimeError("first filtered failure"),
                RuntimeError("second filtered failure"),
                expected,
            ]
        )
        markets = _fetch_markets_with_optional_server_filters(
            client,
            use_server_side_filters=True,
            fetch_window_start=now,
            fetch_window_end=now + timedelta(days=1),
        )
        self.assertEqual(markets, expected)
        self.assertEqual(client.reset_calls, 1)
        self.assertEqual(len(client.calls), 3)
        # last call should be unfiltered fallback
        self.assertEqual(client.calls[-1], (None, None))

    def test_cap_effective_confidence_for_market_respects_category_caps(self) -> None:
        settings = Settings(
            MAX_GLOBAL_CONFIDENCE=0.85,
            MAX_SPORTS_CONFIDENCE=0.80,
            MAX_ESPORTS_CONFIDENCE=0.75,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        sports_market = Market(id="s1", question="NBA: A vs B", category="sports")
        esports_market = Market(id="e1", question="Esports: A vs B", category="esports")
        politics_market = Market(id="p1", question="Election", category="politics")

        self.assertEqual(
            _cap_effective_confidence_for_market(0.99, sports_market, settings),
            0.80,
        )
        self.assertEqual(
            _cap_effective_confidence_for_market(0.99, esports_market, settings),
            0.75,
        )
        self.assertEqual(
            _cap_effective_confidence_for_market(0.99, politics_market, settings),
            0.85,
        )

    def test_edge_threshold_applies_fallback_and_coinflip_guards(self) -> None:
        settings = Settings(
            MIN_EDGE=0.05,
            VERY_LOW_PRICE_THRESHOLD=0.25,
            VERY_LOW_PRICE_MIN_EDGE=0.25,
            LOW_PRICE_MIN_EDGE=0.08,
            LOW_PRICE_THRESHOLD=0.50,
            COINFLIP_PRICE_LOWER=0.45,
            COINFLIP_PRICE_UPPER=0.55,
            FALLBACK_EDGE_MIN_EDGE=0.08,
            WEATHER_MIN_EDGE=0.10,
            WEATHER_FALLBACK_EDGE_MIN_EDGE=0.15,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        self.assertEqual(_edge_threshold_for_market(0.60, settings, "computed"), 0.05)
        self.assertEqual(_edge_threshold_for_market(0.20, settings, "computed"), 0.25)
        self.assertEqual(_edge_threshold_for_market(0.52, settings, "computed"), 0.08)
        self.assertEqual(_edge_threshold_for_market(0.60, settings, "fallback"), 0.08)
        weather_market = Market(
            id="w-edge",
            question="Will rainfall exceed 1 inch in Miami tomorrow?",
            category="weather",
        )
        self.assertEqual(
            _edge_threshold_for_market(0.60, settings, "computed", market=weather_market),
            0.10,
        )
        self.assertEqual(
            _edge_threshold_for_market(0.60, settings, "fallback", market=weather_market),
            0.15,
        )

    def test_passes_edge_threshold_blocks_very_low_price_without_extreme_edge(self) -> None:
        settings = Settings(
            MIN_EDGE=0.05,
            VERY_LOW_PRICE_THRESHOLD=0.25,
            VERY_LOW_PRICE_MIN_EDGE=0.25,
            LOW_PRICE_THRESHOLD=0.50,
            LOW_PRICE_MIN_EDGE=0.10,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.34,
            raw_confidence=0.34,
            bet_size_pct=0.2,
            reasoning="test",
            evidence_quality=0.8,
        )
        passed, edge, reason = _passes_edge_threshold(0.14, decision, settings)
        self.assertFalse(passed)
        self.assertAlmostEqual(edge or 0.0, 0.20, places=6)
        self.assertIn("below min 0.25", reason)

    def test_min_evidence_quality_for_weather_market_uses_weather_floor(self) -> None:
        settings = Settings(
            MIN_EVIDENCE_QUALITY_FOR_TRADE=0.5,
            WEATHER_MIN_EVIDENCE_QUALITY=0.7,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        weather_market = Market(
            id="w-eq",
            question="Will rainfall exceed 2 inches in Miami?",
            category="weather",
        )
        generic_market = Market(
            id="g-eq",
            question="Will BTC close above threshold?",
            category="crypto",
        )
        self.assertEqual(_min_evidence_quality_for_market(weather_market, settings), 0.7)
        self.assertEqual(_min_evidence_quality_for_market(generic_market, settings), 0.5)

    def test_max_confidence_for_weather_market_uses_weather_cap(self) -> None:
        settings = Settings(
            MAX_WEATHER_CONFIDENCE=0.79,
            MAX_GLOBAL_CONFIDENCE=0.90,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        market = Market(
            id="w-cap",
            question="Will a tropical storm form in the Gulf?",
            category="weather",
        )
        self.assertEqual(_max_confidence_for_market(market, settings), 0.79)

    def test_max_confidence_for_index_market_uses_index_cap(self) -> None:
        settings = Settings(
            MAX_GLOBAL_CONFIDENCE=0.90,
            MAX_INDEX_CONFIDENCE=0.67,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        market = Market(
            id="KXINXU-26APR10H1200-T6839.9999",
            question="Will the S&P 500 be above threshold?",
            category="finance",
        )
        self.assertEqual(_max_confidence_for_market(market, settings), 0.67)

    def test_max_confidence_for_generic_market_uses_global_cap(self) -> None:
        settings = Settings(
            MAX_GLOBAL_CONFIDENCE=0.83,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        market = Market(
            id="GENERIC-1",
            question="Generic market",
            category="business",
        )
        self.assertEqual(_max_confidence_for_market(market, settings), 0.83)

    def test_analyze_market_candidate_applies_confidence_calibration(self) -> None:
        market = Market(
            id="KXBTCD-26APR1013-T72699.99",
            question="Bitcoin threshold",
            category="crypto",
            outcomes=[MarketOutcome(name="YES", price=0.60), MarketOutcome(name="NO", price=0.40)],
            liquidity_usdc=500.0,
            resolution_criteria="Exchange settlement",
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.90,
            bet_size_pct=0.8,
            reasoning="high confidence",
            evidence_quality=0.8,
        )
        settings = Settings(
            CONFIDENCE_SHRINKAGE_FLOOR=0.50,
            CONFIDENCE_SHRINKAGE_FACTOR=0.40,
            MAX_GLOBAL_CONFIDENCE=1.0,
            MAX_CRYPTO_CONFIDENCE=1.0,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        result = _analyze_market_candidate(
            market=market,
            state=None,
            anchor_analysis=None,
            settings=settings,
            grok_client=DummyGrokClient(decision),
        )
        calibrated = result["decision"]
        self.assertTrue(result["confidence_calibration_applied"])
        self.assertAlmostEqual(result["confidence_before_calibration"], 0.90)
        self.assertAlmostEqual(result["confidence_after_calibration"], 0.66)
        self.assertAlmostEqual(result["raw_vs_calibrated_delta"], 0.24)
        self.assertAlmostEqual(calibrated.confidence, 0.66)
        self.assertLess(calibrated.bet_size_pct, decision.bet_size_pct)
        self.assertIn("Confidence calibrated", calibrated.reasoning)

    def test_kelly_fraction_weather_multiplier_applies(self) -> None:
        now = datetime.now(timezone.utc)
        settings = Settings(
            KELLY_FRACTION_DEFAULT=0.25,
            KELLY_FRACTION_SHORT_HORIZON_HOURS=2,
            KELLY_FRACTION_SHORT_HORIZON=0.10,
            KELLY_FRACTION_WEATHER=0.50,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        weather_market = Market(
            id="w-kelly",
            question="Will it snow in Denver tonight?",
            close_time=now + timedelta(hours=1),
            category="weather",
        )
        generic_market = Market(
            id="g-kelly",
            question="Will earnings beat estimates?",
            close_time=now + timedelta(hours=1),
            category="business",
        )
        self.assertEqual(_kelly_fraction_for_market_horizon(generic_market, settings), 0.10)
        self.assertEqual(_kelly_fraction_for_market_horizon(weather_market, settings), 0.05)

    def test_should_adjust_position_uses_bankroll_relative_cap(self) -> None:
        settings = Settings(
            MAX_POSITION_PER_MARKET_USDC=200.0,
            MAX_POSITION_PCT_OF_BANKROLL=0.15,
            MAX_BET_USDC=50.0,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.70,
            bet_size_pct=1.0,
            reasoning="test",
        )
        existing_position = Position(
            market_id="m-bankroll",
            outcome="YES",
            total_amount_usdc=2.5,
            avg_confidence=0.60,
            trade_count=1,
            first_trade=datetime.now(timezone.utc),
            last_trade=datetime.now(timezone.utc),
        )
        allowed, bet_pct, reason = _should_adjust_position(
            decision=decision,
            market=Market(id="m-bankroll", question="Q", category="sports"),
            existing_position=existing_position,
            state=None,
            settings=settings,
            cycle_bankroll=20.0,
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "confidence_increase_threshold_met")
        self.assertAlmostEqual(bet_pct, 0.01, places=4)

    def test_build_reasoning_hash_ignores_validated_prefix_variation(self) -> None:
        decision_a = TradeDecision(
            should_trade=False,
            outcome="Yes",
            confidence=0.70,
            bet_size_pct=0.0,
            reasoning=(
                "[Validated eq=1.00 gate=allow reason=ok edge_market=0.041 "
                "edge_source=computed] Core thesis unchanged"
            ),
        )
        decision_b = TradeDecision(
            should_trade=False,
            outcome="Yes",
            confidence=0.70,
            bet_size_pct=0.0,
            reasoning=(
                "[Validated eq=0.95 gate=allow reason=ok edge_market=0.038 "
                "edge_source=computed] Core thesis unchanged"
            ),
        )
        self.assertEqual(_build_reasoning_hash(decision_a), _build_reasoning_hash(decision_b))

    def test_effective_position_override_threshold_not_capped_by_category(self) -> None:
        settings = Settings(
            HIGH_CONFIDENCE_POSITION_OVERRIDE=0.85,
            MAX_SPORTS_CONFIDENCE=0.80,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        sports_market = Market(id="s2", question="NBA: A vs B", category="sports")
        threshold = _effective_position_override_threshold(sports_market, settings)
        self.assertEqual(threshold, 0.85)
        self.assertFalse(0.80 >= threshold)

    def test_requires_market_refresh_enforces_staleness_threshold(self) -> None:
        self.assertTrue(
            _requires_market_refresh(
                pre_order_market_refresh=True,
                market_data_age_seconds=None,
                max_market_data_age_seconds=120,
            )
        )
        self.assertFalse(
            _requires_market_refresh(
                pre_order_market_refresh=False,
                market_data_age_seconds=60.0,
                max_market_data_age_seconds=120,
            )
        )
        self.assertTrue(
            _requires_market_refresh(
                pre_order_market_refresh=False,
                market_data_age_seconds=121.0,
                max_market_data_age_seconds=120,
            )
        )

    def test_can_use_lenient_stale_refresh_fallback_requires_direct_high_score(self) -> None:
        settings = Settings(
            MAX_MARKET_DATA_AGE_SECONDS=120,
            SCORE_GATE_THRESHOLD=0.38,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        self.assertTrue(
            _can_use_lenient_stale_refresh_fallback(
                evidence_basis_class="direct",
                pre_execution_final_score=0.45,
                market_data_age_seconds=180.0,
                settings=settings,
            )
        )
        self.assertFalse(
            _can_use_lenient_stale_refresh_fallback(
                evidence_basis_class="proxy",
                pre_execution_final_score=0.45,
                market_data_age_seconds=180.0,
                settings=settings,
            )
        )
        self.assertFalse(
            _can_use_lenient_stale_refresh_fallback(
                evidence_basis_class="direct",
                pre_execution_final_score=0.30,
                market_data_age_seconds=180.0,
                settings=settings,
            )
        )
        self.assertFalse(
            _can_use_lenient_stale_refresh_fallback(
                evidence_basis_class="direct",
                pre_execution_final_score=0.45,
                market_data_age_seconds=360.0,
                settings=settings,
            )
        )

    def test_passes_edge_threshold_blocks_missing_implied_when_required(self) -> None:
        settings = Settings(
            REQUIRE_IMPLIED_PRICE=True,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.75,
            bet_size_pct=0.3,
            reasoning="test",
        )
        passed, edge, reason = _passes_edge_threshold(None, decision, settings)
        self.assertFalse(passed)
        self.assertIsNone(edge)
        self.assertIn("missing implied", reason)

    def test_passes_edge_threshold_uses_raw_confidence_when_available(self) -> None:
        settings = Settings(
            MIN_EDGE=0.05,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="NO",
            confidence=0.80,
            raw_confidence=1.00,
            bet_size_pct=0.4,
            reasoning="test",
        )
        passed, edge, reason = _passes_edge_threshold(0.85, decision, settings)
        self.assertTrue(passed)
        self.assertAlmostEqual(edge or 0.0, 0.15, places=6)
        self.assertEqual(reason, "")

    def test_analyze_market_candidate_returns_decision_payload(self) -> None:
        market = Market(
            id="m-candidate",
            question="Will Team A win?",
            outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
            liquidity_usdc=200.0,
            category="sports",
        )
        settings = Settings(
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.84,
            bet_size_pct=0.4,
            reasoning="Implied prob: 55%, My prob: 72%, Edge: 17%",
            implied_prob_external=0.55,
            my_prob=0.72,
            edge_external=0.17,
            evidence_quality=0.7,
        )
        result = _analyze_market_candidate(
            market=market,
            state=None,
            anchor_analysis=None,
            settings=settings,
            grok_client=DummyGrokClient(decision),
        )
        self.assertIn("decision", result)
        self.assertIn("was_refined", result)
        self.assertFalse(result["was_refined"])
        self.assertEqual(result["decision"].outcome, "YES")

    def test_analyze_market_candidate_returns_failure_payload_on_initial_error(self) -> None:
        market = Market(
            id="m-candidate-fail",
            question="Will Team A win?",
            outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
            liquidity_usdc=200.0,
            category="sports",
        )
        settings = Settings(
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        result = _analyze_market_candidate(
            market=market,
            state=None,
            anchor_analysis=None,
            settings=settings,
            grok_client=FailingGrokClient(),
        )
        self.assertTrue(result["analysis_failed"])
        self.assertIn("internal server error", result["analysis_error"].lower())
        self.assertTrue(result["analysis_error_retriable_xai"])
        self.assertFalse(result["was_refined"])

    def test_build_order_request_from_market_uses_current_market_price(self) -> None:
        market = Market(
            id="m-order",
            question="Will value be above threshold?",
            outcomes=[
                MarketOutcome(name="YES", price=0.67),
                MarketOutcome(name="NO", price=0.33),
            ],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.81,
            bet_size_pct=0.3,
            reasoning="test",
        )
        order = _build_order_request_from_market(
            market=market,
            decision=decision,
            amount_usdc=5.0,
        )
        self.assertEqual(order.market_id, "m-order")
        self.assertEqual(order.outcome, "YES")
        self.assertEqual(order.yes_price, 67)

    def test_confidence_gate_override_metrics_prefers_stronger_edge(self) -> None:
        market = Market(
            id="m-override",
            question="Will Team A win?",
            outcomes=[MarketOutcome(name="YES", price=0.10), MarketOutcome(name="NO", price=0.90)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.40,
            bet_size_pct=0.5,
            reasoning="test",
            edge_external=0.12,
        )
        override_edge, market_edge = _confidence_gate_override_metrics(market, decision)
        self.assertAlmostEqual(market_edge or 0.0, 0.30)
        self.assertAlmostEqual(override_edge or 0.0, 0.30)

if __name__ == "__main__":
    unittest.main()
