import ast
import unittest
import inspect
import json
import re
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import main as main_module
from calibration_gates import PerformanceStats
from config import Settings
from participation import ParticipationTier
from main import (
    _available_orderbook_sell_quantity,
    _analysis_result_rank,
    _analysis_candidate_attempt_limit,
    _apply_participation_audit_fields,
    _analyze_market_candidate,
    _apply_runtime_score_receipt,
    _best_orderbook_sell_price,
    _build_order_request_from_market,
    _build_kalshi_market_fetch_window,
    _build_counterfactual_audit_fields,
    _build_execution_audit,
    _build_previous_analysis,
    _build_reasoning_hash,
    _format_tier_breakdown_for_log,
    _can_use_lenient_stale_refresh_fallback,
    _cap_analysis_candidates,
    _cap_effective_confidence_for_market,
    _calculate_bet,
    _classify_no_trade_routing,
    _collapse_event_ladders,
    _confidence_gate_override_metrics,
    _compute_next_wakeup_seconds,
    _conviction_repair_reason,
    _daily_balance_delta_usdc,
    _daily_drawdown_basis_usdc,
    _daily_drawdown_cap_reached,
    _daily_expectancy_ev_block_reason,
    _posterior_for_lmsr_signal,
    _satellite_recap_bet,
    _daily_trade_cap_reached,
    _daily_expectancy_role,
    _dry_streak_sleep_seconds,
    _edge_band_label,
    _effective_research_queue_drain_quota,
    _edge_threshold_for_market,
    _event_concentration_blocked,
    _event_side_conflict_blocked,
    _expected_value_usdc,
    _should_apply_definitive_side_override,
    _event_ticker_prefix,
    _effective_position_override_threshold,
    _extract_order_cancel_reason,
    _extract_order_fill_count,
    _fetch_markets_with_optional_server_filters,
    _filter_markets,
    _kelly_fraction_for_market_horizon,
    _load_execution_market_snapshot,
    _log_settings_summary,
    _max_confidence_for_market,
    _min_evidence_quality_for_market,
    _non_definitive_confidence_ceiling,
    _pre_analysis_participation_hold,
    _pre_analysis_opportunity_score,
    _passes_edge_threshold,
    _passes_refreshed_edge_guard,
    _requires_market_refresh,
    _research_queue_last_decision_json,
    _research_queue_drain_sort_key,
    _research_queue_effective_drain_priority,
    _research_queue_priority_below_drain_floor,
    _research_queue_recent_drain_attempt,
    _research_queue_zero_yield_sort_key,
    _resolve_dynamic_analysis_candidate_cap,
    _resolve_min_bet_floor,
    _score_breakdown_from_execution_audit,
    _score_gate_critical_rejection_reasons,
    _should_queue_research_for_blocked_trade,
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
from research_profiles import build_market_search_config
from score_engine import compute_final_score


class DummyStateManager:
    def __init__(self, mapping: dict[str, MarketState | None]) -> None:
        self.mapping = mapping

    def get_market_state(self, market_id: str) -> MarketState | None:
        return self.mapping.get(market_id)


class DummyGrokClient:
    def __init__(self, decision: TradeDecision, deep_decision: TradeDecision | None = None) -> None:
        self.decision = decision
        self.deep_decision = deep_decision or decision
        self.deep_calls = 0

    def analyze_market(self, market, search_config=None, previous_analysis=None, **kwargs):
        return self.decision

    def analyze_market_deep(self, market, previous_analysis=None, search_config=None, **kwargs):
        self.deep_calls += 1
        return self.deep_decision


class FailingGrokClient:
    def analyze_market(self, market, search_config=None, previous_analysis=None, **kwargs):
        raise RuntimeError("StatusCode.INTERNAL: internal server error")


class RecordingGrokClient:
    def __init__(self, decision: TradeDecision) -> None:
        self.decision = decision
        self.last_search_config = None

    def analyze_market(self, market, search_config=None, previous_analysis=None, **kwargs):
        self.last_search_config = search_config
        return self.decision


class TestMainUtils(unittest.TestCase):
    class _DummyKalshiClient:
        def __init__(self, responses):
            self._responses = list(responses)
            self.calls = []
            self.reset_calls = 0
            self.last_fetch_pages = 0
            self.last_fetch_cap_hit = False
            self.last_fetch_mve_filter = None

        def get_markets(
            self,
            close_time_start=None,
            close_time_end=None,
            mve_filter=None,
        ):
            self.calls.append((close_time_start, close_time_end, mve_filter))
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

    def test_analysis_result_rank_uses_chosen_side_external_edge_for_no(self) -> None:
        stronger_no = TradeDecision(
            should_trade=True,
            outcome="NO",
            confidence=0.65,
            bet_size_pct=0.2,
            reasoning="Stronger NO edge",
            edge_external=-0.15,
            evidence_quality=0.7,
        )
        weaker_no = stronger_no.model_copy(
            update={
                "reasoning": "Weaker NO edge",
                "edge_external": -0.05,
            }
        )

        self.assertGreater(
            _analysis_result_rank({"decision": stronger_no}),
            _analysis_result_rank({"decision": weaker_no}),
        )

    def test_analysis_result_rank_demotes_critical_score_rejections(self) -> None:
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.75,
            bet_size_pct=0.2,
            reasoning="tradeable",
            evidence_quality=0.9,
            edge_source="computed",
        )
        clean = {
            "decision": decision,
            "pre_execution_final_score": 0.30,
            "pre_execution_rejection_reasons": (),
        }
        critical = {
            "decision": decision,
            "pre_execution_final_score": 0.90,
            "pre_execution_rejection_reasons": ("non_positive_market_edge",),
        }
        self.assertGreater(_analysis_result_rank(clean), _analysis_result_rank(critical))

    def test_runtime_score_receipt_overwrites_pre_execution_score_fields(self) -> None:
        market = Market(
            id="KXSAMPLEGAME-26APR121610TEAMA-TEAMA",
            question="Will Team A beat Team B?",
            liquidity_usdc=1000.0,
            outcomes=[
                MarketOutcome(name="YES", price=0.55),
                MarketOutcome(name="NO", price=0.45),
            ],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.78,
            bet_size_pct=0.2,
            reasoning="Direct source says final score is settled.",
            evidence_quality=0.86,
            edge_source="computed",
            evidence_basis="direct",
            primary_source_url="https://www.example.com/game/test",
        )
        runtime_score = compute_final_score(
            market,
            decision,
            implied_prob_market=0.55,
            evidence_basis_class="direct",
            edge_source="computed",
        )
        audit_context = {
            "score_final": -0.10,
            "score_breakdown": {"score_final": -0.10},
        }

        score_fields = _apply_runtime_score_receipt(
            audit_context,
            score_result=runtime_score,
            score_threshold_effective=0.30,
            pre_execution_final_score=-0.10,
            score_gate_score_source="runtime_recomputed",
            score_gate_critical_reasons=(),
        )

        self.assertEqual(score_fields["score_final"], runtime_score.final_score)
        self.assertIn("score_source_confirmed_edge", score_fields)
        self.assertEqual(audit_context["score_final"], runtime_score.final_score)
        self.assertEqual(
            audit_context["score_breakdown"]["score_final"],
            runtime_score.final_score,
        )
        self.assertEqual(audit_context["execution_score_final"], runtime_score.final_score)
        self.assertEqual(audit_context["execution_score_threshold"], 0.30)
        self.assertEqual(audit_context["score_gate_score_source"], "runtime_recomputed")
        self.assertAlmostEqual(
            audit_context["pre_vs_runtime_score_delta"],
            runtime_score.final_score - (-0.10),
        )

    def test_research_queued_audit_preserves_learning_fields(self) -> None:
        audit = _build_execution_audit(
            decision_terminal=False,
            final_action="research_queued",
            final_reason="historical_prefix_small_sample_negative",
            historical_prefix_action="research_queued",
            learning_hold_reason="historical_prefix_small_sample_negative",
            what_to_learn_next="Review settled prefix outcomes before execution.",
            score_gate_score_source="runtime_recomputed",
        )

        self.assertEqual(audit["final_action"], "research_queued")
        self.assertEqual(
            audit["learning_hold_reason"],
            "historical_prefix_small_sample_negative",
        )
        self.assertEqual(audit["historical_prefix_action"], "research_queued")
        self.assertEqual(
            audit["what_to_learn_next"],
            "Review settled prefix outcomes before execution.",
        )
        self.assertEqual(audit["score_gate_score_source"], "runtime_recomputed")

    def test_score_gate_critical_rejection_blocks_fallback_source_failures(self) -> None:
        reasons = _score_gate_critical_rejection_reasons(
            rejection_reasons=("fallback_edge_penalty", "no_external_odds_penalty"),
            evidence_basis_class="proxy",
            edge_source="fallback",
        )
        self.assertEqual(
            reasons,
            ("fallback_edge_penalty", "no_external_odds_penalty"),
        )

    def test_analysis_result_rank_prefers_profitable_family_when_scores_equal(self) -> None:
        tradeable = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.74,
            bet_size_pct=0.2,
            reasoning="tradeable",
            evidence_quality=0.8,
        )
        result = {"decision": tradeable, "pre_execution_final_score": 0.25}
        profitable_family_rank = _analysis_result_rank(
            result,
            historical_family_pnl_total=12.0,
            historical_family_sample_size=10,
        )
        weak_family_rank = _analysis_result_rank(
            result,
            historical_family_pnl_total=-4.0,
            historical_family_sample_size=10,
        )
        self.assertGreater(profitable_family_rank, weak_family_rank)

    def test_event_ticker_prefix_prefers_event_ticker_field(self) -> None:
        market = Market(
            id="KXSAMPLEGAME-26APR121610TEAMA-TEAMA",
            event_ticker="KXSAMPLEGAME-26APR121610TEAMA",
            question="Test",
        )
        self.assertEqual(
            _event_ticker_prefix(market),
            "KXSAMPLEGAME-26APR121610TEAMA",
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

    def test_event_side_conflict_blocks_opposite_outcome(self) -> None:
        blocked, outcomes = _event_side_conflict_blocked(
            proposed_outcome="YES",
            open_event_outcomes={"no"},
            cycle_event_outcomes=set(),
        )
        self.assertTrue(blocked)
        self.assertEqual(outcomes, ["no"])

    def test_event_side_conflict_allows_same_outcome(self) -> None:
        blocked, outcomes = _event_side_conflict_blocked(
            proposed_outcome="NO",
            open_event_outcomes={"no"},
            cycle_event_outcomes={"no"},
        )
        self.assertFalse(blocked)
        self.assertEqual(outcomes, ["no"])

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
        self.assertEqual(payload.get("rejection_reason"), "score_gate_blocked")
        self.assertEqual(payload.get("skip_reasons"), ["score_gate_blocked"])

    def test_build_execution_audit_marks_research_queued_skip_reason(self) -> None:
        payload = _build_execution_audit(
            decision_terminal=False,
            final_action="research_queued",
            final_reason="edge_gate_blocked",
        )
        self.assertEqual(payload.get("final_action"), "research_queued")
        self.assertEqual(payload.get("skip_reasons"), ["edge_gate_blocked"])
        self.assertEqual(payload.get("rejection_reason"), "edge_gate_blocked")

    def test_should_queue_research_for_blocked_trade_accepts_edge_above_reasonable_max(self) -> None:
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.92,
            bet_size_pct=0.3,
            reasoning="direct source signal",
            evidence_basis="direct",
        )
        should_queue = _should_queue_research_for_blocked_trade(
            settings=Settings(RESEARCH_QUEUE_ENABLED=True),
            decision=decision,
            evidence_basis="direct",
            gate_name="edge",
            threshold_gap=0.0,
            edge_reason="edge_above_reasonable_max",
        )
        self.assertTrue(should_queue)

    def test_should_queue_research_for_blocked_trade_accepts_hallucinated_edge(self) -> None:
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.84,
            bet_size_pct=0.2,
            reasoning="Direct evidence but edge anomaly.",
            evidence_basis="direct",
        )
        should_queue = _should_queue_research_for_blocked_trade(
            settings=Settings(RESEARCH_QUEUE_ENABLED=True),
            decision=decision,
            evidence_basis="direct",
            gate_name="hallucinated_edge",
            threshold_gap=0.05,
        )
        self.assertTrue(should_queue)

    def test_should_queue_research_for_blocked_trade_accepts_extreme_market_edge(self) -> None:
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.86,
            bet_size_pct=0.2,
            reasoning="Direct evidence with oversized market edge.",
            evidence_basis="direct",
        )
        should_queue = _should_queue_research_for_blocked_trade(
            settings=Settings(RESEARCH_QUEUE_ENABLED=True),
            decision=decision,
            evidence_basis="direct",
            gate_name="extreme_market_edge",
            threshold_gap=0.10,
        )
        self.assertTrue(should_queue)

    def test_should_queue_research_for_settlement_aligned_proxy_edge_near_miss(self) -> None:
        """Brent-shaped: proxy + settlement_aligned + tiny edge gap must queue."""
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.63,
            bet_size_pct=0.13,
            reasoning="Settlement-aligned Brent quote near commodity floor.",
            evidence_basis="proxy",
            evidence_quality=0.75,
            edge_source="computed",
            source_match_class="settlement_aligned",
            primary_source_url="https://www.bloomberg.com/energy",
        )
        settings = Settings(RESEARCH_QUEUE_ENABLED=True)
        self.assertTrue(
            _should_queue_research_for_blocked_trade(
                settings=settings,
                decision=decision,
                evidence_basis="proxy",
                gate_name="edge",
                threshold_gap=0.0035,
                edge_reason="edge 0.2165 below min 0.2200",
            )
        )
        # Unverified / low-EQ proxy edge blocks stay terminal (not queued).
        unverified = decision.model_copy(
            update={"source_match_class": "unverified", "evidence_quality": 0.55}
        )
        self.assertFalse(
            _should_queue_research_for_blocked_trade(
                settings=settings,
                decision=unverified,
                evidence_basis="proxy",
                gate_name="edge",
                threshold_gap=0.0035,
                edge_reason="edge 0.2165 below min 0.2200",
            )
        )
        # Absence-only edge blocks are not edge-queue eligible.
        absence = decision.model_copy(
            update={
                "evidence_basis": "absence_only",
                "source_match_class": "missing_or_absence_only",
                "evidence_quality": 0.45,
                "edge_source": "none",
            }
        )
        self.assertFalse(
            _should_queue_research_for_blocked_trade(
                settings=settings,
                decision=absence,
                evidence_basis="absence_only",
                gate_name="edge",
                threshold_gap=0.0035,
            )
        )

    def test_should_queue_research_for_kelly_sub_floor_even_when_proxy(self) -> None:
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.81,
            bet_size_pct=0.08,
            reasoning="High-quality settled weather below MIN_BET.",
            evidence_basis="proxy",
            evidence_quality=0.90,
            edge_source="computed",
            source_match_class="settlement_aligned",
        )
        self.assertTrue(
            _should_queue_research_for_blocked_trade(
                settings=Settings(RESEARCH_QUEUE_ENABLED=True),
                decision=decision,
                evidence_basis="proxy",
                gate_name="kelly_sub_floor_skip",
                threshold_gap=0.25,
            )
        )

    def test_analysis_result_rank_prefers_lower_overconfidence_gap(self) -> None:
        safer = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.74,
            bet_size_pct=0.2,
            reasoning="Safer calibrated setup",
            evidence_quality=0.72,
        )
        overconfident = safer.model_copy(update={"confidence": 0.88, "evidence_quality": 0.50})
        safer_rank = _analysis_result_rank({"decision": safer, "pre_execution_final_score": 0.25})
        overconfident_rank = _analysis_result_rank(
            {"decision": overconfident, "pre_execution_final_score": 0.25}
        )
        self.assertGreater(safer_rank, overconfident_rank)

    def test_analysis_result_rank_uses_historical_family_win_rate_tie_breaker(self) -> None:
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.78,
            bet_size_pct=0.2,
            reasoning="Tie-breaker check",
            evidence_quality=0.74,
        )
        high_win_rate_rank = _analysis_result_rank(
            {"decision": decision, "pre_execution_final_score": 0.20},
            historical_family_pnl_total=12.0,
            historical_family_sample_size=10,
            historical_family_win_rate=0.72,
        )
        low_win_rate_rank = _analysis_result_rank(
            {"decision": decision, "pre_execution_final_score": 0.20},
            historical_family_pnl_total=12.0,
            historical_family_sample_size=10,
            historical_family_win_rate=0.41,
        )
        self.assertGreater(high_win_rate_rank, low_win_rate_rank)

    def test_build_execution_audit_non_sports_primary_source_skip_no_duplicate_market_family(self) -> None:
        audit_context = {
            "market_family": "generic",
            "pre_analysis_score": 0.7,
            "edge_market": 0.05,
        }
        payload = _build_execution_audit(
            decision_terminal=True,
            final_action="skip",
            final_reason="non_sports_missing_primary_source",
            primary_source_url=None,
            **audit_context,
        )
        self.assertEqual(payload.get("market_family"), "generic")
        self.assertEqual(payload.get("final_reason"), "non_sports_missing_primary_source")
        self.assertIsNone(payload.get("primary_source_url"))

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

    def test_build_execution_audit_accepts_audit_context_with_edge_market(self) -> None:
        payload = _build_execution_audit(
            decision_phase="post_sizing",
            decision_terminal=True,
            final_action="skip",
            final_reason="zero_bet_after_sizing",
            **{"edge_market": 0.12, "edge_external": 0.08},
        )
        self.assertEqual(payload.get("edge_market"), 0.12)
        self.assertEqual(payload.get("edge_external"), 0.08)
        self.assertEqual(payload.get("rejection_reason"), "zero_bet_after_sizing")

    def test_order_audits_do_not_duplicate_market_age_from_context(self) -> None:
        tree = ast.parse(inspect.getsource(main_module.main))
        duplicate_calls: list[int] = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_build_execution_audit"
            ):
                continue
            explicit_keys = {keyword.arg for keyword in node.keywords if keyword.arg}
            expanded_contexts = {
                keyword.value.id
                for keyword in node.keywords
                if keyword.arg is None and isinstance(keyword.value, ast.Name)
            }
            if (
                "market_data_age_seconds" in explicit_keys
                and expanded_contexts
                & {"audit_context", "order_audit_context", "final_order_audit_context"}
            ):
                duplicate_calls.append(node.lineno)

        self.assertEqual(duplicate_calls, [])

    def test_order_failure_audit_preserves_market_age_from_context(self) -> None:
        order_audit_context = {
            "market_data_age_seconds": 270.84,
            "bet_amount_usdc": 3.43,
        }
        try:
            raise RuntimeError("simulated order rejection")
        except RuntimeError:
            payload = _build_execution_audit(
                decision_phase="order_submission",
                decision_terminal=True,
                final_action="order_attempt",
                final_reason="jurisdiction_sports_blocked",
                order_error="403 forbidden",
                **order_audit_context,
            )

        self.assertEqual(payload.get("market_data_age_seconds"), 270.84)
        self.assertEqual(payload.get("final_reason"), "jurisdiction_sports_blocked")

    def test_build_execution_audit_prefers_canonical_over_alias_when_both_present(self) -> None:
        payload = _build_execution_audit(
            final_reason="test",
            edge=0.10,
            edge_market=0.12,
            implied_prob=0.51,
            implied_prob_market=0.54,
        )
        self.assertEqual(payload.get("edge_market"), 0.12)
        self.assertEqual(payload.get("implied_prob_market"), 0.54)
        self.assertNotIn("edge", payload)
        self.assertNotIn("implied_prob", payload)

    def test_build_execution_audit_replays_post_sizing_signature_without_collision(self) -> None:
        audit_context = {
            "market_family": "generic",
            "pre_execution_final_score": 0.11,
            "edge_market": 0.12,
            "edge_external": 0.05,
        }
        payload = _build_execution_audit(
            decision_phase="post_sizing",
            decision_terminal=True,
            final_action="skip",
            final_reason="zero_bet_after_sizing",
            sizing_mode="kelly",
            adjusted_bet_pct=0.0,
            bet_amount_usdc=0.0,
            kelly_raw=0.02,
            kelly_fraction_value=0.25,
            posterior_for_kelly=0.57,
            bayesian_posterior_raw=0.58,
            bayesian_posterior_applied=0.57,
            bayesian_applied=True,
            bayesian_update_count=2,
            bayesian_min_updates=1,
            likelihood_ratio=1.4,
            implied_prob_market=0.45,
            min_edge_for_kelly=0.10,
            lmsr_execution_price=0.46,
            lmsr_inefficiency_signal=0.03,
            lmsr_liquidity_param_b=100000.0,
            **audit_context,
        )
        self.assertEqual(payload.get("edge_market"), 0.12)
        self.assertEqual(payload.get("edge_external"), 0.05)
        self.assertEqual(payload.get("final_reason"), "zero_bet_after_sizing")
        self.assertIsNone(payload.get("rejection_stage"))

    def test_kelly_zero_routed_through_fallback_edge_scaling_floors_at_min_bet(self) -> None:
        (
            bet_amount,
            bet_pct,
            min_floor_applied,
            kelly_sub_floor_skipped,
            policy_applied,
        ) = _resolve_min_bet_floor(
            bet_amount=0.0,
            min_bet_usdc=8.0,
            max_bet_usdc=16.0,
            kelly_path_active=True,
            min_bet_policy="fallback_edge_scaling",
            edge_scaling_bet_pct=0.20,
        )
        self.assertEqual(policy_applied, "fallback_edge_scaling")
        self.assertEqual(bet_amount, 8.0)
        self.assertEqual(bet_pct, 0.5)
        self.assertTrue(min_floor_applied)
        self.assertFalse(kelly_sub_floor_skipped)

    def test_kelly_zero_with_skip_policy_still_hard_skips(self) -> None:
        (
            bet_amount,
            bet_pct,
            min_floor_applied,
            kelly_sub_floor_skipped,
            policy_applied,
        ) = _resolve_min_bet_floor(
            bet_amount=0.0,
            min_bet_usdc=8.0,
            max_bet_usdc=16.0,
            kelly_path_active=True,
            min_bet_policy="skip",
            edge_scaling_bet_pct=0.40,
        )
        self.assertEqual(policy_applied, "skip")
        self.assertEqual(bet_amount, 0.0)
        self.assertEqual(bet_pct, 0.0)
        self.assertFalse(min_floor_applied)
        self.assertTrue(kelly_sub_floor_skipped)

    def test_audit_contains_sizing_zero_reason(self) -> None:
        payload = _build_execution_audit(
            decision_phase="post_sizing",
            decision_terminal=True,
            final_action="skip",
            final_reason="zero_bet_after_sizing",
            sizing_mode="kelly",
            sizing_zero_reason="kelly_posterior_edge_below_min",
        )
        self.assertEqual(payload.get("sizing_zero_reason"), "kelly_posterior_edge_below_min")

    def test_score_breakdown_from_execution_audit_infers_score_fields(self) -> None:
        score_breakdown = _score_breakdown_from_execution_audit(
            execution_audit={
                "score_final": 0.42,
                "score_edge_market": 0.11,
                "final_reason": "score_gate_blocked",
            },
            explicit_score_breakdown=None,
        )
        self.assertIsNotNone(score_breakdown)
        assert score_breakdown is not None
        self.assertEqual(score_breakdown["score_final"], 0.42)
        self.assertEqual(score_breakdown["score_edge_market"], 0.11)

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

    def test_pre_analysis_opportunity_score_records_actionability_penalties(self) -> None:
        market = Market(
            id="KXWHCDATTEND-26-MRUB",
            question="Will Marco Rubio attend the dinner?",
            category="politics",
            liquidity_usdc=800.0,
            outcomes=[MarketOutcome(name="YES", price=0.45), MarketOutcome(name="NO", price=0.55)],
            close_time=datetime.now(timezone.utc) + timedelta(hours=12),
            resolution_criteria="",
        )
        score, breakdown = _pre_analysis_opportunity_score(
            market,
            MarketState(market_id=market.id),
            Settings(),
            traded_before=False,
        )
        self.assertLess(score, 0.80)
        self.assertGreater(breakdown["pre_score_source_difficulty_penalty"], 0.0)
        self.assertGreater(breakdown["pre_score_ambiguous_market_penalty"], 0.0)
        self.assertGreater(breakdown["pre_score_ambiguous_resolution_penalty"], 0.0)

    def test_pre_analysis_opportunity_score_does_not_favor_coinflip_prices(self) -> None:
        """The flat tradeable-price band must not deprioritize a market priced
        away from 0.50 (where directional edge is easier to find) relative to a
        coinflip-priced market; coinflips are still trimmed by the penalty."""
        settings = Settings()

        def _sports(price_yes: float, suffix: str) -> Market:
            return Market(
                id=f"KXMLBGAME-26JUN06NYYBOS-{suffix}",
                question="Will the New York Yankees win the game vs Boston?",
                category="sports",
                liquidity_usdc=1000.0,
                outcomes=[
                    MarketOutcome(name="YES", price=price_yes),
                    MarketOutcome(name="NO", price=round(1.0 - price_yes, 2)),
                ],
                close_time=datetime.now(timezone.utc) + timedelta(hours=6),
                resolution_criteria="Per official MLB result",
            )

        coinflip_score, coinflip_bd = _pre_analysis_opportunity_score(
            _sports(0.50, "A"),
            MarketState(market_id="a", analysis_count=0, non_actionable_streak=0),
            settings,
            traded_before=False,
        )
        directional_score, directional_bd = _pre_analysis_opportunity_score(
            _sports(0.72, "B"),
            MarketState(market_id="b", analysis_count=0, non_actionable_streak=0),
            settings,
            traded_before=False,
        )
        self.assertGreaterEqual(directional_score, coinflip_score)
        self.assertEqual(directional_bd["pre_score_tradeable_price"], 1.0)
        self.assertGreater(coinflip_bd["pre_score_coinflip_penalty"], 0.0)

    def test_pre_analysis_opportunity_score_rewards_direct_evidence_family(self) -> None:
        """Families with reliable direct settlement evidence (weather, via
        NWS/NOAA) get a selection affinity nudge so analysis slots favor
        tradeable edge."""
        settings = Settings()
        market = Market(
            id="KXHIGHDEN-26JUN20-B89.5",
            question="Will the high temperature in Denver be 89-90F on Jun 20?",
            category="weather",
            liquidity_usdc=800.0,
            outcomes=[MarketOutcome(name="YES", price=0.40), MarketOutcome(name="NO", price=0.60)],
            close_time=datetime.now(timezone.utc) + timedelta(hours=20),
            resolution_criteria="Per NWS/NOAA daily high temperature",
        )
        _, breakdown = _pre_analysis_opportunity_score(
            market,
            MarketState(market_id=market.id, analysis_count=0, non_actionable_streak=0),
            settings,
            traded_before=False,
        )
        self.assertGreater(
            breakdown["pre_score_direct_evidence_family_affinity"], 0.0
        )

    def test_pre_analysis_demotion_for_repeated_non_actionable_family(self) -> None:
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
        rejected, reason, metadata = _pre_analysis_participation_hold(
            market=market,
            state=state,
            settings=Settings(),
            traded_before=False,
        )
        self.assertTrue(rejected)
        self.assertEqual(reason, "pre_analysis_repeated_non_actionable_market")
        self.assertEqual(metadata["participation_demotion_family"], "speech")

    def test_pre_analysis_demotion_for_repeated_non_actionable_generic_bin(self) -> None:
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
        rejected, reason, metadata = _pre_analysis_participation_hold(
            market=market,
            state=state,
            settings=Settings(),
            traded_before=False,
        )
        self.assertTrue(rejected)
        self.assertEqual(reason, "pre_analysis_repeated_non_actionable_bin_market")
        self.assertEqual(metadata["participation_demotion_family"], "generic")

    def test_pre_analysis_demotion_for_fallback_edge_high_churn(self) -> None:
        market = Market(
            id="KXBTCD-26APR0917-T70499.99",
            question="Bitcoin threshold",
            category="crypto",
            outcomes=[MarketOutcome(name="YES", price=0.60), MarketOutcome(name="NO", price=0.40)],
            liquidity_usdc=500.0,
        )
        state = MarketState(
            market_id=market.id,
            analysis_count=6,
            # Streak must reach _PRE_ANALYSIS_FALLBACK_CHURN_MIN_STREAK (raised
            # 3 -> 5) before a never-traded fallback-edge market is benched as
            # high-churn, giving such markets more analysis runway now that the
            # calibration/scoring fixes let them clear the gates.
            non_actionable_streak=5,
            last_terminal_outcome="no_trade_recommended",
        )
        rejected, reason, metadata = _pre_analysis_participation_hold(
            market=market,
            state=state,
            settings=Settings(),
            traded_before=False,
            had_recent_fallback_edge=True,
        )
        self.assertTrue(rejected)
        self.assertEqual(reason, "pre_analysis_fallback_edge_high_churn")
        self.assertTrue(metadata["participation_demotion_had_recent_fallback_edge"])

    def test_pre_analysis_demotion_for_repeated_churn_market(self) -> None:
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
        rejected, reason, metadata = _pre_analysis_participation_hold(
            market=market,
            state=state,
            settings=Settings(),
            traded_before=False,
        )
        self.assertTrue(rejected)
        self.assertEqual(reason, "pre_analysis_repeated_churn_market")
        self.assertEqual(metadata["participation_demotion_analysis_count"], 5)

    def test_pre_analysis_crypto_history_is_signal_not_demotion(self) -> None:
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
        rejected, reason, metadata = _pre_analysis_participation_hold(
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
        self.assertFalse(rejected)
        self.assertIsNone(reason)
        self.assertEqual(metadata["participation_demotion_family"], "crypto")

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

    def test_pre_analysis_opportunity_score_adds_small_positive_family_volume_bonus(self) -> None:
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
        baseline_score, baseline_breakdown = _pre_analysis_opportunity_score(
            market,
            None,
            settings,
            traded_before=False,
            historical_family_stats={"sample_size": 7, "win_rate": 0.50, "pnl_total": 2.0},
        )
        boosted_score, boosted_breakdown = _pre_analysis_opportunity_score(
            market,
            None,
            settings,
            traded_before=False,
            historical_family_stats={"sample_size": 8, "win_rate": 0.50, "pnl_total": 2.0},
        )

        self.assertEqual(baseline_breakdown["pre_score_historical_family_volume_bonus"], 0.0)
        self.assertGreater(boosted_breakdown["pre_score_historical_family_volume_bonus"], 0.0)
        self.assertEqual(boosted_breakdown["pre_score_positive_family_pnl_bonus"], 0.02)
        self.assertGreater(boosted_score, baseline_score)

    def test_pre_analysis_opportunity_score_boosts_profitable_family_pnl(self) -> None:
        market = Market(
            id="KXMLBF5-26APR0917-TEST",
            question="MLB First 5 Innings threshold",
            category="sports",
            liquidity_usdc=800.0,
            outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
            close_time=datetime.now(timezone.utc) + timedelta(hours=10),
            resolution_criteria="Official settlement source",
        )
        settings = Settings(PRE_ANALYSIS_ADAPTIVE_BOOST=0.03)
        baseline_score, baseline_breakdown = _pre_analysis_opportunity_score(
            market,
            None,
            settings,
            traded_before=False,
            historical_family_stats={"sample_size": 0, "win_rate": 0.0, "pnl_total": 0.0},
        )
        boosted_score, boosted_breakdown = _pre_analysis_opportunity_score(
            market,
            None,
            settings,
            traded_before=False,
            historical_family_stats={"sample_size": 1, "win_rate": 1.0, "pnl_total": 1.0},
        )

        self.assertEqual(baseline_breakdown["pre_score_positive_family_pnl_bonus"], 0.0)
        self.assertEqual(boosted_breakdown["pre_score_positive_family_pnl_bonus"], 0.02)
        self.assertAlmostEqual(boosted_score - baseline_score, 0.02, places=6)

    def test_pre_analysis_opportunity_score_adds_post_event_bonus(self) -> None:
        settings = Settings()
        market_past = Market(
            id="KXSAMPLEGAME-PAST",
            question="Post-event market",
            category="sports",
            liquidity_usdc=800.0,
            outcomes=[MarketOutcome(name="YES", price=0.52), MarketOutcome(name="NO", price=0.48)],
            close_time=datetime.now(timezone.utc) - timedelta(hours=3),
            resolution_criteria="Official box score",
        )
        market_future = market_past.model_copy(
            update={"id": "KXSAMPLEGAME-FUTURE", "close_time": datetime.now(timezone.utc) + timedelta(hours=3)}
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

    def test_pre_analysis_opportunity_score_applies_zero_trade_rate_penalty_from_history(self) -> None:
        settings = Settings(
            PRE_ANALYSIS_ZERO_TRADE_RATE_PENALTY=0.04,
            HISTORICAL_TICKER_PREFIX_LEN=12,
            HISTORICAL_TICKER_PREFIX_MIN_SAMPLES=3,
        )
        market = Market(
            id="KXGENERIC-26APR201335-T1",
            question="Generic threshold contract",
            category="finance",
            liquidity_usdc=700.0,
            outcomes=[MarketOutcome(name="YES", price=0.50), MarketOutcome(name="NO", price=0.50)],
            close_time=datetime.now(timezone.utc) + timedelta(hours=8),
            resolution_criteria="Official settlement source",
        )
        market_prefix = market.id[: settings.HISTORICAL_TICKER_PREFIX_LEN]
        historical_prefix_stats = {
            market_prefix: PerformanceStats(
                sample_size=6,
                wins=0,
                win_rate=0.0,
                pnl_total=-4.0,
            )
        }
        penalized_score, penalized_breakdown = _pre_analysis_opportunity_score(
            market,
            None,
            settings,
            traded_before=False,
            historical_prefix_stats=historical_prefix_stats,
        )
        disabled_settings = Settings(
            PRE_ANALYSIS_ZERO_TRADE_RATE_PENALTY=0.0,
            HISTORICAL_TICKER_PREFIX_LEN=12,
            HISTORICAL_TICKER_PREFIX_MIN_SAMPLES=3,
        )
        unpenalized_score, unpenalized_breakdown = _pre_analysis_opportunity_score(
            market,
            None,
            disabled_settings,
            traded_before=False,
            historical_prefix_stats=historical_prefix_stats,
        )
        self.assertEqual(penalized_breakdown["pre_score_zero_trade_rate_penalty"], 0.04)
        self.assertEqual(unpenalized_breakdown["pre_score_zero_trade_rate_penalty"], 0.0)
        self.assertLess(penalized_score, unpenalized_score)

    def test_pre_analysis_historical_prefix_loss_gate_is_signal_not_demotion(self) -> None:
        market = Market(
            id="KXTESTMARKET-26APR20-T50",
            question="Test market",
            category="generic",
            outcomes=[MarketOutcome(name="YES", price=0.5), MarketOutcome(name="NO", price=0.5)],
        )
        state = MarketState(
            market_id=market.id,
            analysis_count=2,
            non_actionable_streak=1,
            last_terminal_outcome="no_trade_recommended",
        )
        rejected, reason, metadata = _pre_analysis_participation_hold(
            market=market,
            state=state,
            settings=Settings(),
            traded_before=False,
            historical_gate_allowed=False,
            historical_gate_reason="historical_prefix_pnl_block",
            historical_gate_metrics={
                "historical_gate_prefix_sample_size": 6,
                "historical_gate_prefix_win_rate": 0.25,
                "historical_gate_prefix_pnl_total": -12.0,
            },
        )
        self.assertFalse(rejected)
        self.assertIsNone(reason)
        self.assertEqual(metadata["historical_gate_prefix_sample_size"], 6)

    def test_resolve_dynamic_analysis_candidate_cap_reduces_when_best_score_low(self) -> None:
        settings = Settings(
            MAX_MARKETS_PER_CYCLE=6,
            PRE_ANALYSIS_MUST_ANALYZE_THRESHOLD=0.50,
            PRE_ANALYSIS_REDUCED_MAX_CANDIDATES=3,
        )
        cap, applied, neg_floor = _resolve_dynamic_analysis_candidate_cap(
            settings=settings,
            best_pre_analysis_score=0.42,
        )
        self.assertEqual(cap, 3)
        self.assertTrue(applied)
        self.assertFalse(neg_floor)

    def test_resolve_dynamic_analysis_candidate_cap_keeps_default_when_score_high(self) -> None:
        settings = Settings(
            MAX_MARKETS_PER_CYCLE=6,
            PRE_ANALYSIS_MUST_ANALYZE_THRESHOLD=0.50,
            PRE_ANALYSIS_REDUCED_MAX_CANDIDATES=3,
        )
        cap, applied, neg_floor = _resolve_dynamic_analysis_candidate_cap(
            settings=settings,
            best_pre_analysis_score=0.75,
        )
        self.assertEqual(cap, 6)
        self.assertFalse(applied)
        self.assertFalse(neg_floor)

    def test_resolve_dynamic_analysis_candidate_cap_reduces_after_zero_yield_streak(self) -> None:
        settings = Settings(
            MAX_MARKETS_PER_CYCLE=20,
            PRE_ANALYSIS_MUST_ANALYZE_THRESHOLD=0.50,
            PRE_ANALYSIS_REDUCED_MAX_CANDIDATES=8,
            CYCLE_YIELD_ALERT_ESCALATE_AFTER=4,
        )
        cap, applied, neg_floor = _resolve_dynamic_analysis_candidate_cap(
            settings=settings,
            best_pre_analysis_score=0.90,
            consecutive_zero_execution_yield_cycles=4,
        )
        self.assertEqual(cap, 8)
        self.assertTrue(applied)
        self.assertFalse(neg_floor)

    def test_resolve_dynamic_analysis_candidate_cap_negative_floor_applied(self) -> None:
        settings = Settings(
            MAX_MARKETS_PER_CYCLE=6,
            PRE_ANALYSIS_MUST_ANALYZE_THRESHOLD=0.50,
            PRE_ANALYSIS_REDUCED_MAX_CANDIDATES=3,
            NEGATIVE_BEST_SCORE_DEEP_ANALYSIS_FLOOR=0.05,
        )
        cap, applied, neg_floor = _resolve_dynamic_analysis_candidate_cap(
            settings=settings,
            best_pre_analysis_score=-0.10,
        )
        self.assertEqual(cap, 1)
        self.assertTrue(applied)
        self.assertTrue(neg_floor)

    def test_research_queue_drain_sort_key_prioritizes_small_threshold_gap(self) -> None:
        entries = [
            {"market_id": "KXWIDE", "threshold_gap": 0.11, "queued_at": "2026-05-13T00:01:00+00:00"},
            {"market_id": "KXCLOSE", "threshold_gap": 0.01, "queued_at": "2026-05-13T00:02:00+00:00"},
            {"market_id": "KXUNKNOWN", "threshold_gap": None, "queued_at": "2026-05-13T00:00:00+00:00"},
        ]

        ordered = sorted(entries, key=_research_queue_drain_sort_key)

        self.assertEqual([entry["market_id"] for entry in ordered], ["KXCLOSE", "KXWIDE", "KXUNKNOWN"])

    def test_research_queue_drain_demotes_chronic_research_gap(self) -> None:
        """High-attempt research_gap loses to fresh edge/kelly near-misses."""
        chronic_gap = {
            "market_id": "KXT20CHRONIC",
            "reason": "no_trade_research_gap",
            "research_priority": 0.55,
            "threshold_gap": 0.11,
            "queued_at": "2026-05-13T00:01:00+00:00",
            "last_decision_json": json.dumps(
                {
                    "audit": {
                        "research_queue_drain_attempts": 7,
                        "final_reason": "no_trade_research_gap",
                    }
                }
            ),
        }
        fresh_edge = {
            "market_id": "KXEDGEFRESH",
            "reason": "edge_gate_blocked",
            "research_priority": 0.90,
            "threshold_gap": 0.01,
            "queued_at": "2026-05-13T00:02:00+00:00",
            "last_decision_json": json.dumps(
                {
                    "audit": {
                        "research_queue_drain_attempts": 0,
                        "final_reason": "edge_gate_blocked",
                    }
                }
            ),
        }
        fresh_kelly = {
            "market_id": "KXKELLYFRESH",
            "reason": "kelly_sub_floor_skip",
            "research_priority": 0.92,
            "threshold_gap": 0.0,
            "queued_at": "2026-05-13T00:03:00+00:00",
            "last_decision_json": json.dumps(
                {
                    "audit": {
                        "research_queue_drain_attempts": 1,
                        "final_reason": "kelly_sub_floor_skip",
                    }
                }
            ),
        }

        demoted = _research_queue_effective_drain_priority(chronic_gap)
        self.assertAlmostEqual(demoted, 0.35, places=4)
        self.assertTrue(
            _research_queue_priority_below_drain_floor(
                chronic_gap,
                min_priority=0.40,
            )
        )
        self.assertFalse(
            _research_queue_priority_below_drain_floor(
                fresh_edge,
                min_priority=0.40,
            )
        )

        ordered = sorted(
            [chronic_gap, fresh_edge, fresh_kelly],
            key=_research_queue_drain_sort_key,
        )
        self.assertEqual(
            [entry["market_id"] for entry in ordered],
            ["KXKELLYFRESH", "KXEDGEFRESH", "KXT20CHRONIC"],
        )

        zero_yield_ordered = sorted(
            [chronic_gap, fresh_edge, fresh_kelly],
            key=_research_queue_zero_yield_sort_key,
        )
        self.assertEqual(
            [entry["market_id"] for entry in zero_yield_ordered],
            ["KXKELLYFRESH", "KXEDGEFRESH", "KXT20CHRONIC"],
        )

    def test_research_queue_drain_does_not_demote_fresh_research_gap(self) -> None:
        fresh_gap = {
            "market_id": "KXGAPFRESH",
            "reason": "no_trade_research_gap",
            "research_priority": 0.55,
            "threshold_gap": 0.11,
            "queued_at": "2026-05-13T00:01:00+00:00",
            "last_decision_json": json.dumps(
                {
                    "audit": {
                        "research_queue_drain_attempts": 2,
                        "final_reason": "no_trade_research_gap",
                    }
                }
            ),
        }
        self.assertAlmostEqual(
            _research_queue_effective_drain_priority(fresh_gap),
            0.55,
            places=4,
        )
        self.assertFalse(
            _research_queue_priority_below_drain_floor(
                fresh_gap,
                min_priority=0.40,
            )
        )

    def test_zero_yield_queue_sort_uses_gap_then_times_seen(self) -> None:
        entries = [
            {"market_id": "KXONCE", "threshold_gap": 0.04, "times_seen": 1, "queued_at": "2026-05-13T00:02:00+00:00"},
            {"market_id": "KXMANY", "threshold_gap": 0.04, "times_seen": 7, "queued_at": "2026-05-13T00:03:00+00:00"},
            {"market_id": "KXCLOSE", "threshold_gap": 0.01, "times_seen": 1, "queued_at": "2026-05-13T00:04:00+00:00"},
        ]

        ordered = sorted(entries, key=_research_queue_zero_yield_sort_key)

        self.assertEqual(
            [entry["market_id"] for entry in ordered],
            ["KXCLOSE", "KXMANY", "KXONCE"],
        )

    def test_zero_yield_drought_limits_queue_drain_to_diagnostic_probe(self) -> None:
        self.assertEqual(
            _effective_research_queue_drain_quota(
                configured_quota=8,
                sustained_zero_yield=True,
            ),
            2,
        )
        self.assertEqual(
            _effective_research_queue_drain_quota(
                configured_quota=8,
                sustained_zero_yield=False,
            ),
            8,
        )

    def test_research_queue_recent_drain_attempt_respects_cooldown(self) -> None:
        now = datetime(2026, 5, 16, 20, 0, tzinfo=timezone.utc)
        recent = {
            "last_decision_json": json.dumps(
                {
                    "audit": {
                        "research_queue_drain_attempts": 1,
                        "research_queue_last_drain_attempt_at": (
                            now - timedelta(minutes=20)
                        ).isoformat(),
                    }
                }
            )
        }
        old = {
            "last_decision_json": json.dumps(
                {
                    "audit": {
                        "research_queue_drain_attempts": 1,
                        "research_queue_last_drain_attempt_at": (
                            now - timedelta(minutes=90)
                        ).isoformat(),
                    }
                }
            )
        }

        self.assertTrue(
            _research_queue_recent_drain_attempt(
                recent,
                cooldown_minutes=45,
                now=now,
            )
        )
        self.assertFalse(
            _research_queue_recent_drain_attempt(
                old,
                cooldown_minutes=45,
                now=now,
            )
        )

    def test_conviction_repair_triggers_on_high_edge_high_evidence_no_trade(self) -> None:
        market = Market(
            id="KXSPORTS-26MAY161900TEAMATEAMB",
            question="Will Team A win?",
            outcomes=[
                MarketOutcome(name="YES", price=0.48),
                MarketOutcome(name="NO", price=0.52),
            ],
            liquidity_usdc=800.0,
            resolution_criteria="Official score",
        )
        decision = TradeDecision(
            should_trade=False,
            outcome="YES",
            confidence=0.74,
            bet_size_pct=0.0,
            reasoning="Strong edge but no trade.",
            edge_external=0.24,
            edge_source="computed",
            evidence_basis="proxy",
            evidence_quality=0.93,
            primary_source_url="https://espn.com/game",
            source_match_class="settlement_aligned",
        )

        reason = _conviction_repair_reason(
            decision=decision,
            market=market,
            settings=Settings(),
            score_result=None,
            score_threshold=None,
        )

        self.assertEqual(reason, "conviction_repair_no_trade_contradiction")

    def test_conviction_repair_triggers_on_calibrated_confidence_block(self) -> None:
        market = Market(
            id="KXSPORTS-26MAY161900TEAMATEAMB",
            question="Will Team A win?",
            outcomes=[
                MarketOutcome(name="YES", price=0.44),
                MarketOutcome(name="NO", price=0.56),
            ],
            liquidity_usdc=800.0,
            resolution_criteria="Official score",
        )
        decision = TradeDecision(
            should_trade=True,
            raw_should_trade=True,
            outcome="YES",
            raw_confidence=0.74,
            confidence=0.50,
            bet_size_pct=0.05,
            reasoning="Raw edge was strong before calibration.",
            edge_external=0.24,
            edge_source="computed",
            evidence_basis="direct",
            evidence_quality=0.95,
            primary_source_url="https://espn.com/game",
            source_match_class="settlement_aligned",
        )
        diagnostics: dict[str, object] = {}
        score_result = type("ScoreResultStub", (), {"final_score": 0.04})()

        reason = _conviction_repair_reason(
            decision=decision,
            market=market,
            settings=Settings(MIN_CONFIDENCE=0.62),
            score_result=score_result,
            score_threshold=0.40,
            diagnostics=diagnostics,
        )

        self.assertEqual(reason, "conviction_repair_confidence_calibration_block")
        self.assertTrue(diagnostics["conviction_repair_triggerable"])
        self.assertEqual(
            diagnostics["conviction_repair_reason"],
            "conviction_repair_confidence_calibration_block",
        )

    def test_conviction_repair_rejects_absence_only_or_missing_source(self) -> None:
        market = Market(
            id="KXGENERIC-B1",
            question="Will a generic outcome happen?",
            outcomes=[
                MarketOutcome(name="YES", price=0.48),
                MarketOutcome(name="NO", price=0.52),
            ],
            liquidity_usdc=800.0,
            resolution_criteria="Official source",
        )
        absence_decision = TradeDecision(
            should_trade=False,
            outcome="YES",
            confidence=0.74,
            bet_size_pct=0.0,
            reasoning="Absence-only edge.",
            edge_external=0.24,
            edge_source="computed",
            evidence_basis="absence_only",
            evidence_quality=0.95,
            source_match_class="settlement_aligned",
        )
        missing_source_decision = absence_decision.model_copy(
            update={"evidence_basis": "proxy"}
        )
        diagnostics: dict[str, object] = {}

        self.assertIsNone(
            _conviction_repair_reason(
                decision=absence_decision,
                market=market,
                settings=Settings(),
                diagnostics=diagnostics,
            )
        )
        self.assertEqual(
            diagnostics["conviction_repair_missed_reason"],
            "absence_only_evidence",
        )
        self.assertIsNone(
            _conviction_repair_reason(
                decision=missing_source_decision,
                market=market,
                settings=Settings(),
            )
        )

    def test_conviction_repair_exempts_weather_from_missing_primary_source(self) -> None:
        decision = TradeDecision(
            should_trade=False,
            outcome="YES",
            confidence=0.74,
            bet_size_pct=0.0,
            reasoning="Direct NWS observed daily high; no URL attached.",
            edge_external=0.24,
            edge_source="computed",
            evidence_basis="direct",
            evidence_quality=0.93,
            source_match_class="settlement_aligned",
        )
        weather_market = Market(
            id="KXHIGHDEN-26JUN20-B89.5",
            question="Will the Denver daily high settle in the 89-90F bin?",
            category="weather",
            outcomes=[MarketOutcome(name="YES", price=0.48), MarketOutcome(name="NO", price=0.52)],
            liquidity_usdc=800.0,
        )
        generic_market = Market(
            id="KXGENERIC-B1",
            question="Will a generic outcome happen?",
            outcomes=[MarketOutcome(name="YES", price=0.48), MarketOutcome(name="NO", price=0.52)],
            liquidity_usdc=800.0,
        )
        # Weather is exempt: it is not rejected for a missing primary_source_url
        # and proceeds to the no-trade-contradiction repair path.
        weather_reason = _conviction_repair_reason(
            decision=decision,
            market=weather_market,
            settings=Settings(),
            score_result=None,
            score_threshold=None,
        )
        self.assertEqual(weather_reason, "conviction_repair_no_trade_contradiction")
        # Generic still requires the URL.
        generic_diag: dict[str, object] = {}
        generic_reason = _conviction_repair_reason(
            decision=decision,
            market=generic_market,
            settings=Settings(),
            score_result=None,
            score_threshold=None,
            diagnostics=generic_diag,
        )
        self.assertIsNone(generic_reason)
        self.assertEqual(
            generic_diag.get("conviction_repair_missed_reason"),
            "non_sports_missing_primary_source",
        )

    def test_parallel_attempt_limit_does_not_add_failure_buffer(self) -> None:
        settings = Settings(MAX_MARKETS_PER_CYCLE=6, XAI_CIRCUIT_BREAKER_MAX_FAILURES=3)
        self.assertEqual(
            _analysis_candidate_attempt_limit(
                settings,
                dynamic_max_markets_per_cycle=6,
                parallel_analysis_enabled=True,
            ),
            6,
        )

    def test_sequential_attempt_limit_keeps_failure_buffer(self) -> None:
        settings = Settings(MAX_MARKETS_PER_CYCLE=6, XAI_CIRCUIT_BREAKER_MAX_FAILURES=3)
        self.assertEqual(
            _analysis_candidate_attempt_limit(
                settings,
                dynamic_max_markets_per_cycle=6,
                parallel_analysis_enabled=False,
            ),
            9,
        )

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
        self.assertAlmostEqual(
            breakdown["pre_score_historical_family_pnl_penalty"],
            0.15,
            places=6,
        )
        self.assertAlmostEqual(
            breakdown["pre_score_historical_family_pnl_ratio"],
            0.525,
            places=6,
        )

    def test_generic_family_negative_pnl_penalty_is_capped(self) -> None:
        market = Market(
            id="KXJOBLESSCLAIMS-26MAY07-200000",
            question="Will initial jobless claims be above 200,000?",
            category="economic",
            liquidity_usdc=800.0,
            outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
            close_time=datetime.now(timezone.utc) + timedelta(hours=10),
            resolution_criteria="Official Department of Labor release",
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
                "win_rate": 0.55,
                "pnl_total": -40.0,
            },
        )
        self.assertEqual(breakdown["pre_score_market_subfamily"], "generic_macro_release")
        self.assertAlmostEqual(
            breakdown["pre_score_historical_family_pnl_penalty"],
            0.04,
            places=6,
        )

    def test_pre_analysis_opportunity_score_caps_stacked_historical_penalties(
        self,
    ) -> None:
        """Stacked historical/family penalties from overlapping data sources
        (fallback rate + family PnL + severe family PnL + zero-trade-rate
        + negative-prefix + historical-gate score-penalty) must not be allowed
        to compound into a 0.5pp+ score collapse. The cap credits any excess
        back into the score and surfaces the credit in the breakdown so the
        receipts stay auditable.
        """

        from market_state import MarketState
        from calibration_gates import PerformanceStats, GateTier

        market = Market(
            id="KXBADCRYPTO-1234-T100",
            question="Will BTC settle above threshold?",
            category="crypto",
            liquidity_usdc=800.0,
            outcomes=[
                MarketOutcome(name="YES", price=0.50),
                MarketOutcome(name="NO", price=0.50),
            ],
            close_time=datetime.now(timezone.utc) + timedelta(hours=4),
            resolution_criteria="Official settlement source",
        )
        cap = 0.20
        settings = Settings(
            PRE_ANALYSIS_FALLBACK_FAMILY_RATE_THRESHOLD=0.50,
            PRE_ANALYSIS_FALLBACK_FAMILY_MIN_SAMPLES=5,
            PRE_ANALYSIS_FALLBACK_FAMILY_PENALTY=0.20,
            PRE_ANALYSIS_HISTORICAL_FAMILY_MIN_SAMPLES=10,
            PRE_ANALYSIS_HISTORICAL_FAMILY_WIN_RATE_THRESHOLD=0.45,
            PRE_ANALYSIS_HISTORICAL_FAMILY_PENALTY=0.18,
            PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_MIN_SAMPLES=10,
            PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_THRESHOLD=-5.0,
            PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_PENALTY=0.12,
            PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_SEVERE_THRESHOLD=-15.0,
            PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_SEVERE_PENALTY=0.20,
            PRE_ANALYSIS_ZERO_TRADE_RATE_PENALTY=0.04,
            HISTORICAL_TICKER_PREFIX_LEN=12,
            HISTORICAL_TICKER_PREFIX_MIN_SAMPLES=3,
            HISTORICAL_TICKER_PREFIX_PNL_CUTOFF=-2.0,
            HISTORICAL_TICKER_PREFIX_SOFT_DEMOTE_SCORE_PENALTY=0.10,
            PRE_ANALYSIS_STACKED_HISTORICAL_PENALTY_CAP=cap,
        )
        # Build penalty stack from every overlapping source so we can verify
        # the cap's credit-back behavior.
        prefix_stats = {
            "KXBADCRYPTO-": PerformanceStats(
                sample_size=8,
                wins=0,
                win_rate=0.0,
                pnl_total=-20.0,
            )
        }
        historical_gate_metrics = {
            "historical_gate_tier": GateTier.SOFT_DEMOTE,
            "historical_gate_sample_weight": 0.8,
            "historical_gate_score_penalty": 0.10,
        }
        score, breakdown = _pre_analysis_opportunity_score(
            market,
            MarketState(market_id=market.id, analysis_count=0),
            settings,
            traded_before=False,
            fallback_family_edge_rate=0.95,
            fallback_family_sample_size=80,
            historical_family_stats={
                "sample_size": 30,
                "win_rate": 0.30,
                "pnl_total": -25.0,
            },
            historical_prefix_stats=prefix_stats,
            historical_gate_metrics=historical_gate_metrics,
        )
        raw_stack = breakdown["pre_score_stacked_historical_penalty_raw"]
        excess = breakdown["pre_score_stacked_historical_excess_credited"]
        cap_emitted = breakdown["pre_score_stacked_historical_penalty_cap"]
        self.assertGreater(
            raw_stack,
            cap,
            "Test setup should produce a raw stacked penalty exceeding the cap",
        )
        self.assertAlmostEqual(excess, raw_stack - cap, places=6)
        self.assertAlmostEqual(cap_emitted, cap, places=6)

        # Recompute the same market with the cap effectively disabled and
        # verify the score recovers exactly the credited excess. This proves
        # the cap purely adjusts the score arithmetic without altering the
        # individual penalty fields the receipts depend on.
        settings_uncapped = Settings(
            PRE_ANALYSIS_FALLBACK_FAMILY_RATE_THRESHOLD=settings.PRE_ANALYSIS_FALLBACK_FAMILY_RATE_THRESHOLD,
            PRE_ANALYSIS_FALLBACK_FAMILY_MIN_SAMPLES=settings.PRE_ANALYSIS_FALLBACK_FAMILY_MIN_SAMPLES,
            PRE_ANALYSIS_FALLBACK_FAMILY_PENALTY=settings.PRE_ANALYSIS_FALLBACK_FAMILY_PENALTY,
            PRE_ANALYSIS_HISTORICAL_FAMILY_MIN_SAMPLES=settings.PRE_ANALYSIS_HISTORICAL_FAMILY_MIN_SAMPLES,
            PRE_ANALYSIS_HISTORICAL_FAMILY_WIN_RATE_THRESHOLD=settings.PRE_ANALYSIS_HISTORICAL_FAMILY_WIN_RATE_THRESHOLD,
            PRE_ANALYSIS_HISTORICAL_FAMILY_PENALTY=settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PENALTY,
            PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_MIN_SAMPLES=settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_MIN_SAMPLES,
            PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_THRESHOLD=settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_THRESHOLD,
            PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_PENALTY=settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_PENALTY,
            PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_SEVERE_THRESHOLD=settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_SEVERE_THRESHOLD,
            PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_SEVERE_PENALTY=settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_SEVERE_PENALTY,
            PRE_ANALYSIS_ZERO_TRADE_RATE_PENALTY=settings.PRE_ANALYSIS_ZERO_TRADE_RATE_PENALTY,
            HISTORICAL_TICKER_PREFIX_LEN=settings.HISTORICAL_TICKER_PREFIX_LEN,
            HISTORICAL_TICKER_PREFIX_MIN_SAMPLES=settings.HISTORICAL_TICKER_PREFIX_MIN_SAMPLES,
            HISTORICAL_TICKER_PREFIX_PNL_CUTOFF=settings.HISTORICAL_TICKER_PREFIX_PNL_CUTOFF,
            HISTORICAL_TICKER_PREFIX_SOFT_DEMOTE_SCORE_PENALTY=settings.HISTORICAL_TICKER_PREFIX_SOFT_DEMOTE_SCORE_PENALTY,
            PRE_ANALYSIS_STACKED_HISTORICAL_PENALTY_CAP=0.0,
        )
        score_uncapped, breakdown_uncapped = _pre_analysis_opportunity_score(
            market,
            MarketState(market_id=market.id, analysis_count=0),
            settings_uncapped,
            traded_before=False,
            fallback_family_edge_rate=0.95,
            fallback_family_sample_size=80,
            historical_family_stats={
                "sample_size": 30,
                "win_rate": 0.30,
                "pnl_total": -25.0,
            },
            historical_prefix_stats=prefix_stats,
            historical_gate_metrics=historical_gate_metrics,
        )
        self.assertAlmostEqual(score_uncapped + excess, score, places=6)
        self.assertEqual(
            breakdown_uncapped["pre_score_stacked_historical_excess_credited"],
            0.0,
        )

    def test_pre_analysis_opportunity_score_does_not_credit_when_below_cap(
        self,
    ) -> None:
        """When stacked penalties stay under the cap, the cap must be a no-op."""
        market = Market(
            id="KXSTABLE-1234-T100",
            question="Will the index settle above threshold?",
            category="generic",
            liquidity_usdc=400.0,
            outcomes=[
                MarketOutcome(name="YES", price=0.55),
                MarketOutcome(name="NO", price=0.45),
            ],
            close_time=datetime.now(timezone.utc) + timedelta(hours=20),
            resolution_criteria="Official settlement source",
        )
        settings = Settings(
            PRE_ANALYSIS_HISTORICAL_FAMILY_MIN_SAMPLES=10,
            PRE_ANALYSIS_HISTORICAL_FAMILY_WIN_RATE_THRESHOLD=0.45,
            PRE_ANALYSIS_HISTORICAL_FAMILY_PENALTY=0.10,
            PRE_ANALYSIS_STACKED_HISTORICAL_PENALTY_CAP=0.40,
        )
        _, breakdown = _pre_analysis_opportunity_score(
            market,
            None,
            settings,
            traded_before=False,
            historical_family_stats={
                "sample_size": 20,
                "win_rate": 0.40,
                "pnl_total": -3.0,
            },
        )
        self.assertEqual(
            breakdown["pre_score_stacked_historical_excess_credited"],
            0.0,
        )

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
            category="sports",
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
            Settings(MIN_EDGE=0.05, NON_SPORTS_REQUIRES_DIRECT_EVIDENCE=False),
        )
        self.assertFalse(ok)
        self.assertEqual(implied_prob, 0.70)
        self.assertAlmostEqual(edge or 0.0, 0.02, places=6)
        self.assertIn("below min", reason)

    def test_passes_refreshed_edge_guard_uses_confidence_override(self) -> None:
        market = Market(
            id="m-refresh-floor",
            question="Will high temp exceed 93?",
            category="weather",
            outcomes=[
                MarketOutcome(name="YES", price=0.60),
                MarketOutcome(name="NO", price=0.40),
            ],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.545,
            bet_size_pct=0.4,
            reasoning="calibrated down",
            edge_source="computed",
            edge_external=0.13,
            evidence_basis="direct",
            evidence_quality=0.90,
            primary_source_url="https://forecast.weather.gov/MapClick.php?lat=30.3&lon=-97.7",
        )
        settings = Settings(
            MIN_EDGE=0.05,
            WEATHER_MIN_EDGE=0.05,
            NON_SPORTS_REQUIRES_DIRECT_EVIDENCE=False,
            MAX_REASONABLE_EDGE=0.40,
            DEFINITIVE_OUTCOME_EDGE_REASONABLE_MAX=0.50,
            DIRECT_POSTERIOR_FLOOR_ENABLED=True,
            DIRECT_POSTERIOR_FLOOR_MIN_EVIDENCE_QUALITY=0.80,
        )
        # Without an explicit override the guard recomputes the posterior floor
        # (parity with the primary edge gate) and clears the threshold.
        ok_auto, _, edge_auto, reason_auto = _passes_refreshed_edge_guard(
            market, decision, settings
        )
        self.assertTrue(ok_auto, reason_auto)
        self.assertAlmostEqual(edge_auto or 0.0, 0.13, places=6)
        # An explicit low override that ignores the floor still fails.
        ok_low, _, edge_low, reason_low = _passes_refreshed_edge_guard(
            market,
            decision,
            settings,
            effective_confidence_override=0.545,
        )
        self.assertFalse(ok_low)
        self.assertIn("below min", reason_low)
        self.assertAlmostEqual(edge_low or 0.0, -0.055, places=6)
        # Explicit floored override matches the auto path.
        ok_floored, _, edge_floored, reason_floored = _passes_refreshed_edge_guard(
            market,
            decision,
            settings,
            effective_confidence_override=0.73,
        )
        self.assertTrue(ok_floored, reason_floored)
        self.assertAlmostEqual(edge_floored or 0.0, 0.13, places=6)

    def test_research_queue_context_text_includes_prior_fields_and_repair_action(self) -> None:
        from main import _research_queue_context_text

        text = _research_queue_context_text(
            {
                "reason": "edge_gate_blocked",
                "what_to_learn_next": "Find NWS URL",
                "gate_name": "edge_gate",
                "last_decision": {
                    "confidence": 0.72,
                    "edge_market": 0.08,
                    "evidence_basis": "proxy",
                    "edge_source": "computed",
                    "evidence_quality": 0.70,
                    "primary_source_url": "",
                },
            }
        )
        self.assertIsNotNone(text)
        assert text is not None
        self.assertIn("prior_confidence=0.72", text)
        self.assertIn("prior_edge_market=0.08", text)
        self.assertIn("prior_evidence_basis=proxy", text)
        self.assertIn("prior_primary_source_url=missing", text)
        self.assertIn("repair_action=", text)
        self.assertIn("should_trade=true", text)

    def test_is_michigan_sports_jurisdiction_error(self) -> None:
        from main import _is_michigan_sports_jurisdiction_error

        self.assertTrue(
            _is_michigan_sports_jurisdiction_error(
                "403 body=Michigan_residents_are_not_currently_allowed_to_open_positions_in_Sports"
            )
        )
        self.assertFalse(_is_michigan_sports_jurisdiction_error("insufficient balance"))

    def test_edge_repair_skips_sports_computed_odds_near_binary(self) -> None:
        from main import _edge_repair_reason

        settings = Settings(
            EDGE_REPAIR_ENABLED=True,
            EDGE_BAND_CALIBRATION_ENABLED=True,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        market = Market(
            id="KXMLBGAME-26JUL10LADSF-LAD",
            question="Will the Dodgers win?",
            category="sports",
            outcomes=[
                MarketOutcome(name="YES", price=0.55),
                MarketOutcome(name="NO", price=0.45),
            ],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.96,
            my_prob=0.96,
            probability_yes=0.96,
            bet_size_pct=0.2,
            reasoning="sportsbook consensus edge",
            edge_source="computed",
            edge_external=0.41,
            evidence_basis="proxy",
            evidence_quality=0.72,
        )
        self.assertIsNone(
            _edge_repair_reason(
                decision=decision,
                market=market,
                settings=settings,
                implied_prob=0.55,
            )
        )

    def test_edge_repair_still_flags_non_sports_high_edge(self) -> None:
        from main import _edge_repair_reason

        settings = Settings(
            EDGE_REPAIR_ENABLED=True,
            EDGE_BAND_CALIBRATION_ENABLED=True,
            COMMODITY_HIGH_EQ_MIN_EVIDENCE_QUALITY=0.75,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        market = Market(
            id="KXWTI-26JUL13-T70",
            question="Will WTI be above $70?",
            category="economics",
            outcomes=[
                MarketOutcome(name="YES", price=0.40),
                MarketOutcome(name="NO", price=0.60),
            ],
        )
        # Low-EQ / non-settlement-aligned proxy still triggers repair.
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.80,
            bet_size_pct=0.2,
            reasoning="commodity proxy",
            edge_source="computed",
            edge_external=0.40,
            evidence_basis="proxy",
            evidence_quality=0.70,
            source_match_class="verifiable_unmatched",
        )
        self.assertEqual(
            _edge_repair_reason(
                decision=decision,
                market=market,
                settings=settings,
                implied_prob=0.40,
            ),
            "high_edge_without_definitive_evidence",
        )

    def test_edge_repair_skips_settlement_aligned_high_eq_computed(self) -> None:
        from main import (
            _edge_repair_reason,
            _is_settlement_aligned_high_eq_computed,
            _should_force_abstain_on_edge_repair_unresolved,
        )

        settings = Settings(
            EDGE_REPAIR_ENABLED=True,
            EDGE_BAND_CALIBRATION_ENABLED=True,
            COMMODITY_HIGH_EQ_MIN_EVIDENCE_QUALITY=0.75,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        market = Market(
            id="KXSOLD-26JUL2017-T80.9999",
            question="Will SOL be above 80.9999?",
            category="crypto",
            outcomes=[
                MarketOutcome(name="YES", price=0.40),
                MarketOutcome(name="NO", price=0.60),
            ],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.82,
            raw_confidence=0.82,
            bet_size_pct=0.2,
            reasoning="Binance live quote vs strike",
            edge_source="computed",
            edge_external=0.42,
            evidence_basis="proxy",
            evidence_quality=0.75,
            source_match_class="settlement_aligned",
            primary_source_url="https://www.binance.com/en/trade/SOL_USDT",
        )
        self.assertTrue(_is_settlement_aligned_high_eq_computed(decision, settings))
        self.assertIsNone(
            _edge_repair_reason(
                decision=decision,
                market=market,
                settings=settings,
                implied_prob=0.40,
            )
        )
        self.assertFalse(
            _should_force_abstain_on_edge_repair_unresolved(
                unresolved_reason="high_edge_without_definitive_evidence",
                decision=decision,
                settings=settings,
            )
        )

    def test_order_exception_error_text_includes_kalshi_body(self) -> None:
        import requests
        from main import _order_exception_error_text

        response = requests.models.Response()
        response.status_code = 403
        response._content = (
            b'{"error":{"code":"michigan_residents_are_not_currently_'
            b'allowed_to_open_positions_in_Sports"}}'
        )
        exc = requests.exceptions.HTTPError(
            "403 Client Error: Forbidden for url: https://api.example/orders",
            response=response,
        )
        setattr(
            exc,
            "_kalshi_response_body",
            "michigan_residents_are_not_currently_allowed_to_open_positions_in_Sports",
        )
        text = _order_exception_error_text(exc)
        self.assertIn("403 Client Error", text)
        self.assertIn("michigan_residents_are_not_currently_allowed_to_open_positions_in_Sports", text)
        from main import _is_michigan_sports_jurisdiction_error

        self.assertTrue(_is_michigan_sports_jurisdiction_error(text))

    def test_kelly_fraction_shrinks_on_weather_calibration_gap(self) -> None:
        from kelly import kelly_bet_pct, kelly_fraction
        from main import (
            _dynamic_kelly_floor_allowed,
            _kelly_fraction_for_decision,
        )

        settings = Settings(
            KELLY_FRACTION_DEFAULT=0.30,
            KELLY_FRACTION_WEATHER=0.50,
            KELLY_FRACTION_SHORT_HORIZON_HOURS=0,
            WEATHER_CALIBRATION_GAP_FOR_KELLY_SHRINK=0.20,
            WEATHER_CALIBRATION_GAP_KELLY_MULTIPLIER=0.50,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        market = Market(
            id="KXHIGHNY-26JUL12-T88",
            question="Will the high temperature in NYC be above 88°F?",
            category="weather",
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.55,
            raw_confidence=0.90,
            bet_size_pct=0.3,
            reasoning="calibration gap",
        )
        # Base weather Kelly = 0.30 * 0.50 = 0.15; gap 0.35 >= 0.20 => * 0.50 = 0.075
        final_fraction = _kelly_fraction_for_decision(
            market,
            settings,
            decision,
            0.55,
        )
        self.assertAlmostEqual(final_fraction, 0.075)
        self.assertFalse(
            _dynamic_kelly_floor_allowed(
                final_fraction=final_fraction,
                settings=settings,
                reference_fraction=0.15,
            )
        )
        # Weather at its intended horizon fraction should arm dynamic floor
        # even when that fraction is below KELLY_FRACTION_DEFAULT.
        self.assertTrue(
            _dynamic_kelly_floor_allowed(
                final_fraction=0.15,
                settings=settings,
                reference_fraction=0.15,
            )
        )
        raw_kelly = kelly_fraction(0.81, 0.69)
        sized_pct = kelly_bet_pct(
            posterior=0.81,
            market_price=0.69,
            fraction=final_fraction,
            min_edge=0.10,
            edge=0.12,
            dynamic_enabled=False,
        )
        self.assertAlmostEqual(sized_pct / raw_kelly, 0.075)
        small_gap = decision.model_copy(update={"raw_confidence": 0.60})
        self.assertAlmostEqual(
            _kelly_fraction_for_decision(market, settings, small_gap, 0.55),
            0.15,
        )

    def test_kelly_fraction_skips_weather_shrink_for_high_quality_settled(self) -> None:
        from main import (
            _dynamic_kelly_floor_allowed,
            _kelly_fraction_for_decision,
        )

        settings = Settings(
            KELLY_FRACTION_DEFAULT=0.30,
            KELLY_FRACTION_WEATHER=0.50,
            KELLY_FRACTION_SHORT_HORIZON_HOURS=0,
            WEATHER_CALIBRATION_GAP_FOR_KELLY_SHRINK=0.20,
            WEATHER_CALIBRATION_GAP_KELLY_MULTIPLIER=0.50,
            HIGH_QUALITY_SETTLED_EVIDENCE_MIN_EQ=0.95,
            DIRECT_SOURCE_WHITELIST=("weather.gov", "nws.noaa.gov"),
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        market = Market(
            id="KXLOWTMIA-26JUL19-B81.5",
            question="Will the low temp in Miami be below 81.5°F?",
            category="weather",
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.55,
            raw_confidence=0.90,
            bet_size_pct=0.3,
            reasoning="NWS observation confirms settlement criterion.",
            evidence_basis="direct",
            evidence_quality=1.0,
            raw_evidence_quality=0.95,
            primary_source_url="https://www.weather.gov/mfl/",
            source_match_class="settlement_aligned",
            edge_source="computed",
        )
        # High-quality settled weather uses DEFAULT (0.30), not 0.075 shrink.
        final_fraction = _kelly_fraction_for_decision(
            market,
            settings,
            decision,
            0.55,
        )
        self.assertAlmostEqual(final_fraction, 0.30)
        self.assertTrue(
            _dynamic_kelly_floor_allowed(
                final_fraction=final_fraction,
                settings=settings,
                reference_fraction=final_fraction,
            )
        )

    def test_no_trade_routing_distinguishes_validation_gate_from_model_choice(
        self,
    ) -> None:
        gated = TradeDecision(
            should_trade=False,
            outcome="YES",
            confidence=0.60,
            bet_size_pct=0.0,
            reasoning=(
                "[Validated eq=0.60 gate=block reason=market_edge_below_min "
                "basis=proxy] Model initially recommended a trade."
            ),
        )
        model_no_trade = gated.model_copy(
            update={"reasoning": "Model recommends waiting for more evidence."}
        )

        gated_routing = _classify_no_trade_routing(gated)
        model_routing = _classify_no_trade_routing(model_no_trade)

        self.assertEqual(gated_routing.reason, "edge_gate_blocked")
        self.assertEqual(gated_routing.gate_name, "edge")
        self.assertTrue(gated_routing.research_eligible)
        self.assertEqual(model_routing.reason, "no_trade_recommended")
        self.assertFalse(model_routing.research_eligible)

    def test_no_trade_with_material_edge_and_research_gap_is_queued(self) -> None:
        decision = TradeDecision(
            should_trade=False,
            outcome="NO",
            confidence=0.60,
            probability_yes=0.40,
            bet_size_pct=0.0,
            reasoning=(
                "[Validated eq=0.20 gate=allow reason=ok basis=absence_only] "
                "No settlement-aligned source was found."
            ),
            evidence_quality=0.20,
            evidence_basis="absence_only",
            edge_source="none",
        )

        routed = _classify_no_trade_routing(
            decision,
            market_edge=0.12,
            research_edge_floor=0.08,
        )
        below_floor = _classify_no_trade_routing(
            decision,
            market_edge=0.04,
            research_edge_floor=0.08,
        )

        self.assertEqual(routed.reason, "no_trade_research_gap")
        self.assertEqual(routed.gate_name, "evidence")
        self.assertTrue(routed.research_eligible)
        self.assertEqual(below_floor.reason, "no_trade_recommended")
        self.assertFalse(below_floor.research_eligible)

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

    def test_expected_value_usdc_uses_entry_price_and_probability(self) -> None:
        ev = _expected_value_usdc(probability=0.65, entry_price=0.50, amount_usdc=10.0)
        self.assertAlmostEqual(ev or 0.0, 3.0, places=6)
        self.assertEqual(_edge_band_label(0.30), "25-35pp")

    def test_daily_expectancy_role_selects_primary_then_satellite_cap(self) -> None:
        settings = Settings(
            DAILY_EXPECTANCY_ENABLED=True,
            DAILY_EXPECTANCY_PRIMARY_TARGETS=2,
            DAILY_EXPECTANCY_SATELLITE_MAX_BET_PCT=0.25,
        )

        self.assertEqual(_daily_expectancy_role(settings=settings, daily_exposure_count=0), ("primary_target", None))
        self.assertEqual(_daily_expectancy_role(settings=settings, daily_exposure_count=1), ("primary_target", None))
        self.assertEqual(_daily_expectancy_role(settings=settings, daily_exposure_count=2), ("satellite", 0.25))

    def test_satellite_cap_recap_respects_min_bet_floor(self) -> None:
        # A satellite bet that only exceeds the cap because the min-bet floor
        # raised it to MIN_BET must NOT be resized (it executes at min bet);
        # genuinely over-cap bets are re-clamped to the cap instead of being
        # terminally blocked.
        self.assertIsNone(
            _satellite_recap_bet(
                bet_pct=0.417,
                satellite_cap_pct=0.25,
                min_bet_floor_applied=True,
                max_bet_usdc=50.0,
                min_bet_usdc=5.0,
            )
        )
        recap = _satellite_recap_bet(
            bet_pct=0.417,
            satellite_cap_pct=0.25,
            min_bet_floor_applied=False,
            max_bet_usdc=50.0,
            min_bet_usdc=5.0,
        )
        self.assertIsNotNone(recap)
        _, clamped_amount = recap
        self.assertAlmostEqual(clamped_amount, 12.5)

    def test_satellite_cap_executable_above_min_bet_when_configured(self) -> None:
        # Invariant: the recommended satellite cap (raised to 0.45) x MAX_BET
        # must be >= MIN_BET so satellite trades are not structurally blocked.
        settings = Settings(
            MAX_BET_USDC=12.0,
            MIN_BET_USDC=5.0,
            DAILY_EXPECTANCY_SATELLITE_MAX_BET_PCT=0.45,
        )
        _, cap = _daily_expectancy_role(settings=settings, daily_exposure_count=2)
        self.assertIsNotNone(cap)
        self.assertGreaterEqual(cap * settings.MAX_BET_USDC, settings.MIN_BET_USDC)

    def test_daily_expectancy_ev_blocks_non_positive_primary_and_unfunded_satellite(self) -> None:
        self.assertEqual(
            _daily_expectancy_ev_block_reason(
                opportunity_role="primary_target",
                expected_value_usdc=-0.01,
                projected_daily_ev_after_usdc=-0.01,
            ),
            "daily_expectancy_primary_ev_blocked",
        )
        self.assertEqual(
            _daily_expectancy_ev_block_reason(
                opportunity_role="satellite",
                expected_value_usdc=-0.25,
                projected_daily_ev_after_usdc=0.0,
            ),
            "daily_expectancy_satellite_ev_blocked",
        )
        self.assertIsNone(
            _daily_expectancy_ev_block_reason(
                opportunity_role="satellite",
                expected_value_usdc=-0.25,
                projected_daily_ev_after_usdc=0.10,
            )
        )

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

    def test_cap_analysis_candidates_uses_global_risk_adjusted_rank(self) -> None:
        candidates = [
            {
                "market": Market(
                    id="c1",
                    question="Will Bitcoin close above $70k?",
                    category="crypto",
                ),
                "pre_analysis_score": 0.90,
            },
            {
                "market": Market(
                    id="c2",
                    question="Will Ethereum close above $4k?",
                    category="crypto",
                ),
                "pre_analysis_score": 0.95,
            },
            {
                "market": Market(
                    id="c3",
                    question="Will Solana close above $200?",
                    category="crypto",
                ),
                "pre_analysis_score": 0.10,
            },
            {
                "market": Market(
                    id="s1",
                    question="Will the Lakers win tonight?",
                    category="sports",
                ),
                "pre_analysis_score": 0.80,
            },
            {
                "market": Market(
                    id="s2",
                    question="Will the Celtics win tonight?",
                    category="sports",
                ),
                "pre_analysis_score": 0.20,
            },
            {
                "market": Market(
                    id="p1",
                    question="Will candidate X win the election?",
                    category="politics",
                ),
                "pre_analysis_score": 0.70,
            },
        ]
        capped = _cap_analysis_candidates(candidates, max_markets_per_cycle=4)
        self.assertEqual([item["market"].id for item in capped], ["c2", "c1", "s1", "p1"])
        self.assertIn("selection_rank_components", capped[0])

    def test_cap_analysis_candidates_score_promotions_do_not_displace_stronger_candidates(self) -> None:
        candidates = [
            {
                "market": Market(id="normal-high", question="Normal high", category="sports"),
                "pre_analysis_score": 0.95,
            },
            {
                "market": Market(id="score-promo", question="Queued near miss", category="generic"),
                "pre_analysis_score": 0.30,
                "is_research_queue_score_promotion": True,
            },
            {
                "market": Market(id="normal-mid", question="Normal mid", category="sports"),
                "pre_analysis_score": 0.80,
            },
        ]

        capped = _cap_analysis_candidates(candidates, max_markets_per_cycle=2)

        self.assertEqual([item["market"].id for item in capped], ["normal-high", "normal-mid"])

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
        capped_ids = [item["market"].id for item in capped]
        self.assertEqual(len([market_id for market_id in capped_ids if market_id.startswith("w")]), 1)
        self.assertIn("w1", capped_ids)
        self.assertIn("c1", capped_ids)
        self.assertIn("s1", capped_ids)
        self.assertIn("p1", capped_ids)

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

    def test_cap_analysis_candidates_limits_generic_candidates(self) -> None:
        candidates = [
            {"market": Market(id="g1", question="Quarterly metric question one", category="business")},
            {"market": Market(id="w1", question="High temperature in Denver", category="weather")},
            {"market": Market(id="g2", question="Quarterly metric question two", category="business")},
            {"market": Market(id="c1", question="Bitcoin price above 50000", category="crypto")},
            {"market": Market(id="g3", question="Quarterly metric question three", category="business")},
            {"market": Market(id="g4", question="Quarterly metric question four", category="business")},
        ]
        capped = _cap_analysis_candidates(
            candidates,
            max_markets_per_cycle=5,
            max_generic_candidates_per_cycle=2,
        )
        capped_ids = [item["market"].id for item in capped]
        generic_selected = [mid for mid in capped_ids if mid.startswith("g")]
        # Generic is capped at 2; the freed slots stay with direct-evidence
        # families (weather/crypto), which are not generic-capped.
        self.assertEqual(len(generic_selected), 2)
        self.assertIn("w1", capped_ids)
        self.assertIn("c1", capped_ids)

    def test_cap_analysis_candidates_generic_cap_none_keeps_legacy_behavior(self) -> None:
        candidates = [
            {"market": Market(id="g1", question="Quarterly metric question one", category="business")},
            {"market": Market(id="g2", question="Quarterly metric question two", category="business")},
            {"market": Market(id="g3", question="Quarterly metric question three", category="business")},
            {"market": Market(id="w1", question="High temperature in Denver", category="weather")},
        ]
        capped = _cap_analysis_candidates(
            candidates,
            max_markets_per_cycle=3,
            max_generic_candidates_per_cycle=None,
        )
        # No generic cap: the top 3 by rank are kept regardless of family.
        self.assertEqual(len(capped), 3)

    def test_cap_analysis_candidates_prefers_lower_non_actionable_streak(self) -> None:
        candidates = [
            {
                "market": Market(id="w-high-streak", question="Weather A", category="weather"),
                "non_actionable_streak": 5,
                "pre_analysis_score": 0.90,
            },
            {
                "market": Market(id="w-low-streak", question="Weather B", category="weather"),
                "non_actionable_streak": 1,
                "pre_analysis_score": 0.90,
            },
            {
                "market": Market(id="c1", question="Crypto", category="crypto"),
                "non_actionable_streak": 0,
                "pre_analysis_score": 0.80,
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

    def test_cap_analysis_candidates_does_not_double_count_historical_gate_penalty(self) -> None:
        """Fix 5d: when historical_gate_metrics is present (gate ran and
        surfaced its score penalty into _pre_analysis_opportunity_score),
        _cap_analysis_candidates must NOT also deduct the legacy 0.12 flat
        penalty. The score penalty would otherwise be double-counted.

        Markets here use NBA-keyword questions so family_from_text resolves
        them to sports family (zero source_difficulty_penalty), isolating
        the historical_gate behavior under test.
        """
        market = Market(
            id="KXSPORTS-METRICS-ATTACHED",
            question="Will the NBA Lakers win the playoffs?",
            category="sports",
        )
        candidates = [
            {
                "market": market,
                "pre_analysis_score": 0.65,
                "historical_gate_allowed": False,
                "historical_gate_metrics": {
                    "historical_gate_score_penalty": 0.05,
                },
            },
            {
                "market": Market(
                    id="KXSPORTS-CLEAN",
                    question="Will the NBA Celtics win tonight?",
                    category="sports",
                ),
                "pre_analysis_score": 0.60,
                "historical_gate_allowed": True,
            },
            {
                "market": Market(
                    id="KXSPORTS-FILLER",
                    question="Will the NBA Heat win the conference?",
                    category="sports",
                ),
                "pre_analysis_score": 0.40,
                "historical_gate_allowed": True,
            },
        ]
        capped = _cap_analysis_candidates(candidates, max_markets_per_cycle=2)
        with_metrics = next(
            item for item in capped if item["market"].id == "KXSPORTS-METRICS-ATTACHED"
        )
        components = with_metrics["selection_rank_components"]
        # Penalty must be exactly the metric-supplied value, NOT 0.12.
        assert components["historical_gate_penalty"] == 0.05
        assert components["risk_adjusted_score"] == 0.60  # 0.65 - 0.05

    def test_cap_analysis_candidates_falls_back_to_flat_penalty_without_metrics(self) -> None:
        """Backward-compat: when historical_gate_allowed is False but metrics
        are missing entirely, the legacy flat 0.12 penalty still applies so
        old code paths don't suddenly become more permissive.

        The legacy market is given a high enough base score (0.85) that even
        with the flat 0.12 penalty it still ranks ahead of the third
        candidate, so the cap selection actually exercises the legacy path.
        """
        candidates = [
            {
                "market": Market(
                    id="KXSPORTS-LEGACY-NO-METRICS",
                    question="Will the NBA Lakers win the playoffs?",
                    category="sports",
                ),
                "pre_analysis_score": 0.85,
                "historical_gate_allowed": False,
            },
            {
                "market": Market(
                    id="KXSPORTS-CLEAN",
                    question="Will the NBA Celtics win tonight?",
                    category="sports",
                ),
                "pre_analysis_score": 0.60,
                "historical_gate_allowed": True,
            },
            {
                "market": Market(
                    id="KXSPORTS-FILLER",
                    question="Will the NBA Heat win the conference?",
                    category="sports",
                ),
                "pre_analysis_score": 0.40,
                "historical_gate_allowed": True,
            },
        ]
        capped = _cap_analysis_candidates(candidates, max_markets_per_cycle=2)
        legacy = next(
            item for item in capped if item["market"].id == "KXSPORTS-LEGACY-NO-METRICS"
        )
        components = legacy["selection_rank_components"]
        assert components["historical_gate_penalty"] == 0.12
        assert components["risk_adjusted_score"] == 0.73  # 0.85 - 0.12

    def test_cap_analysis_candidates_limits_sports_candidates(self) -> None:
        """Cycle 4 recovery: sports props were monopolizing all analysis
        slots even when other families had eligible candidates. The sports
        cap must reserve room for non-sports markets so the cycle still
        evaluates direct-evidence opportunities elsewhere."""
        candidates = [
            {
                "market": Market(
                    id="KXMLBHRR-1",
                    question="Will the Yankees hit a home run?",
                    category="sports",
                ),
                "pre_analysis_score": 0.90,
            },
            {
                "market": Market(
                    id="KXMLBHRR-2",
                    question="Will the Red Sox hit a home run?",
                    category="sports",
                ),
                "pre_analysis_score": 0.88,
            },
            {
                "market": Market(
                    id="KXMLBHRR-3",
                    question="Will the Dodgers hit a home run?",
                    category="sports",
                ),
                "pre_analysis_score": 0.86,
            },
            {
                "market": Market(
                    id="KXMLBHRR-4",
                    question="Will the Braves hit a home run?",
                    category="sports",
                ),
                "pre_analysis_score": 0.84,
            },
            {
                "market": Market(
                    id="KXBTCD-T70K",
                    question="Will Bitcoin close above $70k?",
                    category="crypto",
                ),
                "pre_analysis_score": 0.70,
            },
            {
                "market": Market(
                    id="KXHIGHCHI-T50",
                    question="Will Chicago high be below 50F?",
                    category="weather",
                ),
                "pre_analysis_score": 0.65,
            },
        ]
        capped = _cap_analysis_candidates(
            candidates,
            max_markets_per_cycle=4,
            max_sports_candidates_per_cycle=2,
            max_weather_candidates_per_cycle=1,
            max_crypto_candidates_per_cycle=1,
        )
        capped_ids = [item["market"].id for item in capped]
        sports_ids = [mid for mid in capped_ids if mid.startswith("KXMLBHRR-")]
        self.assertEqual(len(sports_ids), 2)
        self.assertIn("KXBTCD-T70K", capped_ids)
        self.assertIn("KXHIGHCHI-T50", capped_ids)

    def test_cap_analysis_candidates_sports_cap_none_keeps_legacy_behavior(self) -> None:
        """Backward-compat: with max_sports_candidates_per_cycle=None the
        previous behavior (no sports-specific cap) must remain so existing
        operators are not surprised by silent diversification."""
        candidates = [
            {
                "market": Market(
                    id="KXMLB-1",
                    question="Will Yankees win?",
                    category="sports",
                ),
                "pre_analysis_score": 0.90,
            },
            {
                "market": Market(
                    id="KXMLB-2",
                    question="Will Red Sox win?",
                    category="sports",
                ),
                "pre_analysis_score": 0.85,
            },
            {
                "market": Market(
                    id="KXMLB-3",
                    question="Will Dodgers win?",
                    category="sports",
                ),
                "pre_analysis_score": 0.80,
            },
            {
                "market": Market(
                    id="KXBTCD-T70K",
                    question="Will Bitcoin close above $70k?",
                    category="crypto",
                ),
                "pre_analysis_score": 0.50,
            },
        ]
        capped = _cap_analysis_candidates(
            candidates,
            max_markets_per_cycle=3,
            max_sports_candidates_per_cycle=None,
        )
        capped_ids = [item["market"].id for item in capped]
        self.assertEqual(
            len([mid for mid in capped_ids if mid.startswith("KXMLB-")]),
            3,
        )

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

    def test_execution_snapshot_promotes_orderbook_price_before_scoring(self) -> None:
        scheduled = Market(
            id="KX-CANONICAL",
            question="Will the event happen?",
            outcomes=[
                MarketOutcome(name="YES", price=0.45),
                MarketOutcome(name="NO", price=0.55),
            ],
        )
        refreshed = scheduled.model_copy(
            update={
                "outcomes": [
                    MarketOutcome(name="YES", price=0.50),
                    MarketOutcome(name="NO", price=0.50),
                ]
            },
            deep=True,
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.80,
            bet_size_pct=0.2,
            reasoning="Direct evidence",
        )

        class Client:
            @staticmethod
            def get_market(market_id: str) -> Market:
                assert market_id == scheduled.id
                return refreshed

            @staticmethod
            def get_market_orderbook(market_id: str) -> dict:
                assert market_id == scheduled.id
                return {
                    "sells": [
                        {"optionIndex": 0, "price": 0.62, "quantity": 10},
                        {"optionIndex": 1, "price": 0.40, "quantity": 10},
                    ]
                }

        snapshot = _load_execution_market_snapshot(
            market=scheduled,
            decision=decision,
            kalshi_client=Client(),
            settings=Settings(
                DRY_RUN=False,
                PRE_ORDER_MARKET_REFRESH=True,
                ORDERBOOK_PRECHECK_ENABLED=True,
                ORDERBOOK_PRECHECK_MIN_CONFIDENCE=0.75,
            ),
            market_snapshot_monotonic=None,
        )

        assert snapshot.source == "orderbook_best_sell"
        assert snapshot.scheduled_entry_price == 0.45
        assert snapshot.refreshed_entry_price == 0.50
        assert snapshot.execution_entry_price == 0.62
        assert snapshot.market.outcomes[0].price == 0.62

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
        self.assertEqual(data.get("kelly_min_bet_policy"), "skip")
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
        self.assertEqual(client.calls[-1], (None, None, None))

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
        self.assertAlmostEqual(_edge_threshold_for_market(0.20, settings, "computed"), 0.2125)
        self.assertAlmostEqual(_edge_threshold_for_market(0.52, settings, "computed"), 0.068)
        self.assertAlmostEqual(_edge_threshold_for_market(0.60, settings, "fallback"), 0.072)
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
            0.135,
        )
        liquid_market = Market(
            id="liq-edge",
            question="Liquid sports market",
            category="sports",
            liquidity_usdc=200.0,
        )
        very_liquid_market = liquid_market.model_copy(update={"liquidity_usdc": 501.0})
        self.assertAlmostEqual(
            _edge_threshold_for_market(0.60, settings, "computed", market=liquid_market),
            0.05 * settings.MIN_EDGE_MEDIUM_LIQUIDITY_MULTIPLIER,
        )
        self.assertAlmostEqual(
            _edge_threshold_for_market(0.60, settings, "computed", market=very_liquid_market),
            0.05 * settings.MIN_EDGE_HIGH_LIQUIDITY_MULTIPLIER,
        )
        self.assertEqual(
            _edge_threshold_for_market(
                0.20,
                settings,
                "fallback",
                market=very_liquid_market,
                definitive_outcome_eligible=True,
            ),
            0.05,
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
        self.assertIn("below min 0.2125", reason)

    def test_min_evidence_quality_for_weather_market_uses_weather_floor(self) -> None:
        settings = Settings(
            MIN_EVIDENCE_QUALITY_FOR_TRADE=0.5,
            WEATHER_MIN_EVIDENCE_QUALITY=0.7,
            SPORTS_MIN_EVIDENCE_QUALITY=0.6,
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
        sports_market = Market(
            id="KXIPLSIX-26MAY11DCPBKS-20",
            question="Will there be over 19.5 total match sixes?",
            category=None,
        )
        self.assertEqual(_min_evidence_quality_for_market(weather_market, settings), 0.7)
        self.assertEqual(_min_evidence_quality_for_market(sports_market, settings), 0.6)
        self.assertEqual(_min_evidence_quality_for_market(generic_market, settings), 0.5)

    def test_min_evidence_quality_relaxes_for_whitelisted_direct_sources(self) -> None:
        settings = Settings(
            MIN_EVIDENCE_QUALITY_FOR_TRADE=0.75,
            WEATHER_MIN_EVIDENCE_QUALITY=0.80,
            SPORTS_MIN_EVIDENCE_QUALITY=0.70,
            DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_DEFAULT=0.68,
            DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_WEATHER=0.72,
            DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_SPORTS=0.60,
            DIRECT_SOURCE_WHITELIST=("weather.gov", "coindesk.com", "espncricinfo.com"),
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        weather_market = Market(
            id="KXLOWTNOLA-26APR20-T60",
            question="Will low temperature be above threshold in NOLA?",
            category="weather",
        )
        weather_decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.65,
            bet_size_pct=0.4,
            reasoning="direct weather source",
            evidence_basis="direct",
            primary_source_url="https://forecast.weather.gov/MapClick.php?lat=29.95&lon=-90.07",
        )
        generic_market = Market(
            id="KXBTCD-26APR2016-T76299.99",
            question="Will BTC close above threshold?",
            category="crypto",
        )
        generic_decision = weather_decision.model_copy(
            update={"primary_source_url": "https://www.coindesk.com/price/bitcoin"}
        )
        sports_market = Market(
            id="KXT20MATCH-26MAY10COOPNG-COO",
            question="Will the T20 cricket match winner be COO?",
            category=None,
        )
        sports_decision = weather_decision.model_copy(
            update={"primary_source_url": "https://www.espncricinfo.com/series/test"}
        )
        proxy_decision = weather_decision.model_copy(update={"evidence_basis": "proxy"})
        self.assertEqual(
            _min_evidence_quality_for_market(weather_market, settings, weather_decision),
            0.72,
        )
        self.assertEqual(
            _min_evidence_quality_for_market(generic_market, settings, generic_decision),
            0.68,
        )
        self.assertEqual(
            _min_evidence_quality_for_market(sports_market, settings, sports_decision),
            0.60,
        )
        self.assertEqual(
            _min_evidence_quality_for_market(generic_market, settings, proxy_decision),
            0.75,
        )

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

    def test_max_confidence_for_heating_oil_market_uses_subcategory_cap(self) -> None:
        settings = Settings(
            MAX_GLOBAL_CONFIDENCE=0.90,
            MAX_COMMODITY_CONFIDENCE=0.80,
            MAX_HEATING_OIL_CONFIDENCE=0.70,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        market = Market(
            id="KXHOILW-26APR1717-T3.649",
            question="Heating oil threshold",
            category="commodities",
        )
        self.assertEqual(_max_confidence_for_market(market, settings), 0.70)

    def test_max_confidence_for_livestock_market_uses_subcategory_cap(self) -> None:
        settings = Settings(
            MAX_GLOBAL_CONFIDENCE=0.90,
            MAX_COMMODITY_CONFIDENCE=0.80,
            MAX_LIVESTOCK_CONFIDENCE=0.65,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        market = Market(
            id="KXLCATTLEW-26APR1717-T249.99",
            question="Live cattle threshold",
            category="commodities",
        )
        self.assertEqual(_max_confidence_for_market(market, settings), 0.65)

    def test_max_confidence_for_corn_market_uses_subcategory_cap(self) -> None:
        settings = Settings(
            MAX_GLOBAL_CONFIDENCE=0.90,
            MAX_COMMODITY_CONFIDENCE=0.80,
            MAX_CORN_CONFIDENCE=0.70,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        market = Market(
            id="KXCORNW-26APR1717-T448.99",
            question="Corn threshold",
            category="commodities",
        )
        self.assertEqual(_max_confidence_for_market(market, settings), 0.70)

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
        self.assertAlmostEqual(result["confidence_after_calibration"], 0.612)
        self.assertAlmostEqual(result["raw_vs_calibrated_delta"], 0.288)
        self.assertAlmostEqual(calibrated.confidence, 0.612)
        self.assertLess(calibrated.bet_size_pct, decision.bet_size_pct)
        self.assertIn("Confidence calibrated", calibrated.reasoning)

    def test_dry_streak_sleep_seconds_applies_after_three_zero_order_cycles(self) -> None:
        self.assertIsNone(
            _dry_streak_sleep_seconds(
                base_poll_interval_sec=300,
                consecutive_zero_order_cycles=2,
            )
        )
        self.assertEqual(
            _dry_streak_sleep_seconds(
                base_poll_interval_sec=300,
                consecutive_zero_order_cycles=3,
            ),
            600,
        )

    def test_dry_streak_sleep_seconds_disabled_returns_none(self) -> None:
        self.assertIsNone(
            _dry_streak_sleep_seconds(
                base_poll_interval_sec=300,
                consecutive_zero_order_cycles=5,
                enabled=False,
            )
        )

    def test_entry_price_too_low_skip_uses_decision_payload(self) -> None:
        import main as main_module

        source = inspect.getsource(main_module)
        pattern = re.compile(
            r"entry_price_too_low[\s\S]{0,260}decision=decision\.model_dump\(\)",
        )
        self.assertRegex(source, pattern)

    def test_entry_price_floor_skip_guarded_by_edge_override(self) -> None:
        # The hard entry-price floor must be bypassable for high-edge, direct,
        # settlement-aligned trades (the ENTRY_PRICE_FLOOR_EDGE_OVERRIDE path).
        import main as main_module

        source = inspect.getsource(main_module)
        self.assertRegex(
            source,
            re.compile(
                r"entry_price_floor_override\s*=\s*\([\s\S]{0,400}"
                r"ENTRY_PRICE_FLOOR_EDGE_OVERRIDE_ENABLED"
            ),
        )
        self.assertRegex(
            source,
            re.compile(
                r"entry_price\s*<\s*settings\.VERY_LOW_PRICE_THRESHOLD\s*"
                r"and\s*not\s*entry_price_floor_override"
            ),
        )

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

    def test_passes_edge_threshold_override_uses_effective_confidence(self) -> None:
        # Coherence fix: when the execution path supplies the effective
        # (post-calibration/post-Bayesian) confidence, the edge gate must use it
        # instead of raw_confidence, so a market whose calibrated conviction is a
        # coinflip fails the edge gate even though raw conviction looked strong.
        settings = Settings(
            MIN_EDGE=0.05,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.50,
            raw_confidence=0.82,
            bet_size_pct=0.4,
            reasoning="test",
        )
        # Legacy raw path: edge 0.82 - 0.57 = 0.25 -> passes.
        passed_raw, edge_raw, _ = _passes_edge_threshold(0.57, decision, settings)
        self.assertTrue(passed_raw)
        self.assertAlmostEqual(edge_raw or 0.0, 0.25, places=6)
        # Effective-confidence override (0.50): edge -0.07 -> fails (coherent with
        # the score gate and Kelly which also see the 0.50 calibrated conviction).
        passed_eff, edge_eff, _ = _passes_edge_threshold(
            0.57, decision, settings, effective_confidence_override=0.50
        )
        self.assertFalse(passed_eff)
        self.assertAlmostEqual(edge_eff or 0.0, -0.07, places=6)

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

    def test_analyze_market_candidate_repairs_edge_source_none_before_execution(self) -> None:
        market = Market(
            id="m-edge-repair",
            question="Will Team A win?",
            outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
            liquidity_usdc=500.0,
            category="sports",
        )
        initial = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.74,
            bet_size_pct=0.4,
            reasoning="Trade but edge fields are missing.",
            edge_source="none",
            evidence_quality=0.8,
        )
        repaired = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.72,
            probability_yes=0.72,
            bet_size_pct=0.35,
            reasoning="Computed implied=0.55, probability_yes=0.72, edge=0.17 with source.",
            implied_prob_external=0.55,
            my_prob=0.72,
            edge_external=0.17,
            edge_source="computed",
            evidence_basis="direct",
            evidence_quality=0.8,
        )
        client = DummyGrokClient(initial, deep_decision=repaired)
        settings = Settings(
            EDGE_REPAIR_ENABLED=True,
            MAX_GLOBAL_CONFIDENCE=1.0,
            MAX_SPORTS_CONFIDENCE=1.0,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )

        result = _analyze_market_candidate(
            market=market,
            state=None,
            anchor_analysis=None,
            settings=settings,
            grok_client=client,
        )

        self.assertEqual(client.deep_calls, 1)
        self.assertTrue(result["edge_repair_attempted"])
        self.assertEqual(result["edge_repair_reason"], "edge_source_none")
        self.assertIsNone(result["edge_repair_unresolved_reason"])
        self.assertEqual(result["decision"].edge_source, "computed")

    def test_analyze_market_candidate_blocks_unresolved_edge_repair(self) -> None:
        market = Market(
            id="m-edge-repair-unresolved",
            question="Will Team A win?",
            outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
            liquidity_usdc=500.0,
            category="sports",
        )
        initial = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.74,
            bet_size_pct=0.4,
            reasoning="Trade but edge fields are missing.",
            edge_source="none",
            evidence_quality=0.8,
        )
        client = DummyGrokClient(initial, deep_decision=initial)
        settings = Settings(
            EDGE_REPAIR_ENABLED=True,
            MAX_GLOBAL_CONFIDENCE=1.0,
            MAX_SPORTS_CONFIDENCE=1.0,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )

        result = _analyze_market_candidate(
            market=market,
            state=None,
            anchor_analysis=None,
            settings=settings,
            grok_client=client,
        )

        self.assertEqual(client.deep_calls, 1)
        self.assertTrue(result["edge_repair_attempted"])
        self.assertEqual(result["edge_repair_unresolved_reason"], "edge_source_none")
        self.assertFalse(result["decision"].should_trade)
        self.assertTrue(result["decision"].abstain)

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

    def test_analyze_market_candidate_uses_extended_research_profile(self) -> None:
        market = Market(
            id="m-extended-research",
            question="NBA: Will Team A win?",
            outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
            liquidity_usdc=200.0,
            category="nba",
            close_time=datetime.now(timezone.utc) + timedelta(hours=12),
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
        client = RecordingGrokClient(decision)
        baseline_config = build_market_search_config(settings, market)
        result = _analyze_market_candidate(
            market=market,
            state=None,
            anchor_analysis=None,
            settings=settings,
            grok_client=client,
            force_extended_research=True,
        )
        self.assertTrue(result["used_extended_research"])
        self.assertIsNotNone(client.last_search_config)
        self.assertGreater(
            int(client.last_search_config.lookback_hours or 0),
            int(baseline_config.lookback_hours or 0),
        )
        self.assertNotEqual(
            client.last_search_config.allowed_domains,
            baseline_config.allowed_domains,
        )
        self.assertEqual(
            client.last_search_config.allowed_domains[0],
            baseline_config.source_domains_pool[settings.EXTENDED_RESEARCH_SOURCE_OFFSET],
        )

    def test_analyze_market_candidate_uses_high_confidence_shrinkage_factor(self) -> None:
        market = Market(
            id="m-high-shrinkage",
            question="Will Team A win?",
            outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
            category="sports",
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.89,
            bet_size_pct=0.5,
            reasoning="High confidence baseline",
            evidence_quality=0.8,
        )
        settings = Settings(
            CONFIDENCE_SHRINKAGE_FLOOR=0.50,
            CONFIDENCE_SHRINKAGE_FACTOR=0.40,
            CONFIDENCE_SHRINKAGE_FACTOR_HIGH=0.20,
            MAX_GLOBAL_CONFIDENCE=1.0,
            MAX_SPORTS_CONFIDENCE=1.0,
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
        self.assertAlmostEqual(result["confidence_after_calibration"], 0.578)

class TestDefinitiveSideOverride(unittest.TestCase):
    def _make_decision(self, **kw):
        defaults = dict(
            should_trade=True,
            outcome="YES",
            confidence=0.85,
            bet_size_pct=0.5,
            reasoning="AP confirmed game over",
            evidence_quality=0.90,
            evidence_basis="direct",
            definitive_outcome_detected=True,
            likelihood_ratio=15.0,
            raw_confidence=0.90,
        )
        defaults.update(kw)
        return TradeDecision(**defaults)

    def test_definitive_direct_ap_source_grants_override(self) -> None:
        decision = self._make_decision()
        result = _should_apply_definitive_side_override(
            decision=decision,
            evidence_basis="direct",
            primary_source_whitelisted=True,
            cycle_overrides_applied=0,
            max_overrides_per_cycle=2,
        )
        self.assertTrue(result)

    def test_definitive_proxy_evidence_blocked(self) -> None:
        decision = self._make_decision(evidence_basis="proxy")
        result = _should_apply_definitive_side_override(
            decision=decision,
            evidence_basis="proxy",
            primary_source_whitelisted=True,
            cycle_overrides_applied=0,
            max_overrides_per_cycle=2,
        )
        self.assertFalse(result)

    def test_non_definitive_blocked(self) -> None:
        decision = self._make_decision(definitive_outcome_detected=False)
        result = _should_apply_definitive_side_override(
            decision=decision,
            evidence_basis="direct",
            primary_source_whitelisted=True,
            cycle_overrides_applied=0,
            max_overrides_per_cycle=2,
        )
        self.assertFalse(result)

    def test_cap_exceeded_blocks_override(self) -> None:
        decision = self._make_decision()
        result = _should_apply_definitive_side_override(
            decision=decision,
            evidence_basis="direct",
            primary_source_whitelisted=True,
            cycle_overrides_applied=2,
            max_overrides_per_cycle=2,
        )
        self.assertFalse(result)

    def test_low_likelihood_ratio_blocked(self) -> None:
        decision = self._make_decision(likelihood_ratio=5.0)
        result = _should_apply_definitive_side_override(
            decision=decision,
            evidence_basis="direct",
            primary_source_whitelisted=True,
            cycle_overrides_applied=0,
            max_overrides_per_cycle=2,
        )
        self.assertFalse(result)

    def test_low_raw_confidence_blocked(self) -> None:
        decision = self._make_decision(raw_confidence=0.70, confidence=0.70)
        result = _should_apply_definitive_side_override(
            decision=decision,
            evidence_basis="direct",
            primary_source_whitelisted=True,
            cycle_overrides_applied=0,
            max_overrides_per_cycle=2,
        )
        self.assertFalse(result)

    def test_not_whitelisted_source_blocked(self) -> None:
        decision = self._make_decision()
        result = _should_apply_definitive_side_override(
            decision=decision,
            evidence_basis="direct",
            primary_source_whitelisted=False,
            cycle_overrides_applied=0,
            max_overrides_per_cycle=2,
        )
        self.assertFalse(result)


class TestAuditPayloadResearchProfile(unittest.TestCase):
    def test_audit_payload_includes_research_profile(self) -> None:
        payload = _build_execution_audit(
            final_action="order_attempt",
            final_reason="all_gates_passed",
            research_profile="commodity",
        )
        self.assertEqual(payload.get("research_profile"), "commodity")

    def test_audit_payload_research_profile_none_when_absent(self) -> None:
        payload = _build_execution_audit(
            final_action="order_attempt",
            final_reason="all_gates_passed",
        )
        self.assertNotIn("research_profile", payload)


class TestNonDefinitiveConfidenceCeiling(unittest.TestCase):
    def _make_decision(self, **kwargs) -> TradeDecision:
        defaults = {
            "should_trade": True,
            "outcome": "YES",
            "confidence": 0.92,
            "bet_size_pct": 0.5,
            "reasoning": "test",
            "evidence_quality": 0.80,
        }
        defaults.update(kwargs)
        return TradeDecision(**defaults)

    def test_non_definitive_caps_at_089(self) -> None:
        decision = self._make_decision(
            confidence=0.95,
            evidence_basis="proxy",
        )
        settings = Settings()
        ceiling = _non_definitive_confidence_ceiling(decision, settings)
        self.assertLessEqual(ceiling, 0.89)

    def test_definitive_allows_above_089(self) -> None:
        decision = self._make_decision(
            confidence=0.95,
            evidence_basis="direct",
            evidence_quality=0.90,
            raw_evidence_quality=0.90,
            definitive_outcome_detected=True,
            primary_source_url="https://apnews.com/article/example",
            source_match_class="settlement_aligned",
            my_prob=0.97,
        )
        settings = Settings()
        ceiling = _non_definitive_confidence_ceiling(decision, settings)
        self.assertGreater(ceiling, 0.89)

    def test_direct_evidence_uses_direct_cap(self) -> None:
        decision = self._make_decision(
            confidence=0.92,
            evidence_basis="direct",
        )
        settings = Settings(MAX_GLOBAL_CONFIDENCE_DIRECT=0.89)
        ceiling = _non_definitive_confidence_ceiling(decision, settings)
        self.assertEqual(ceiling, 0.89)


class TestPreAnalysisOpportunityResearchBand(unittest.TestCase):
    def test_score_in_research_band_gets_soft_research_tag(self) -> None:
        """Score in [min - band, min) should produce soft_research rejection tag."""
        settings = Settings(
            PRE_ANALYSIS_OPPORTUNITY_ENABLED=True,
            PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE=0.60,
            PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND=0.20,
            PRE_ANALYSIS_OPPORTUNITY_RESEARCH_ENABLED=True,
        )
        market = Market(
            id="KXTEST-RESEARCH-BAND",
            question="Test market for research band",
            outcomes=[MarketOutcome(name="YES", price=0.50), MarketOutcome(name="NO", price=0.50)],
            liquidity_usdc=200.0,
            close_time=datetime.now(timezone.utc) + timedelta(days=1),
        )
        score, breakdown = _pre_analysis_opportunity_score(
            market,
            None,
            settings,
            traded_before=False,
        )
        research_floor = settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE - settings.PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND
        self.assertAlmostEqual(research_floor, 0.40, places=6)
        self.assertLess(research_floor + 1e-9, settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE)

    def test_config_research_band_defaults(self) -> None:
        settings = Settings()
        self.assertEqual(settings.PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND, 0.20)
        self.assertTrue(settings.PRE_ANALYSIS_OPPORTUNITY_RESEARCH_ENABLED)


class TestLifetimeAnalysisCap(unittest.TestCase):
    def test_config_defaults(self) -> None:
        settings = Settings()
        self.assertEqual(settings.MAX_LIFETIME_ANALYSES_PER_MARKET, 8)

    def test_cap_zero_disables_check(self) -> None:
        settings = Settings(MAX_LIFETIME_ANALYSES_PER_MARKET=0)
        self.assertEqual(settings.MAX_LIFETIME_ANALYSES_PER_MARKET, 0)


class TestSyntheticDecisionMarker(unittest.TestCase):
    def test_synthetic_research_queue_origin_marks_audit(self) -> None:
        audit = _build_execution_audit(
            final_action="research_queued",
            final_reason="pre_analysis_score_soft_research",
            decision_origin="synthetic_research_queue",
        )
        self.assertTrue(audit.get("synthetic_decision"))

    def test_synthetic_operational_hold_origin_marks_audit(self) -> None:
        audit = _build_execution_audit(
            final_action="research_queued",
            final_reason="grok_stream_timeout",
            decision_origin="synthetic_operational_hold",
        )
        self.assertTrue(audit.get("synthetic_decision"))

    def test_real_decision_origin_is_not_synthetic(self) -> None:
        audit = _build_execution_audit(
            final_action="skip",
            final_reason="confidence_below_min",
            decision_origin="grok_initial",
        )
        self.assertFalse(audit.get("synthetic_decision"))

    def test_missing_decision_origin_defaults_to_not_synthetic(self) -> None:
        audit = _build_execution_audit(
            final_action="order_submitted",
        )
        self.assertFalse(audit.get("synthetic_decision"))


class TestParticipationAuditStamping(unittest.TestCase):
    def test_should_trade_confidence_skip_gets_canonical_participation_fields(self) -> None:
        audit = _build_execution_audit(
            final_action="skip",
            final_reason="confidence_below_min",
            counterfactual_required_confidence=0.62,
            evidence_basis_class="direct",
            edge_source="computed",
        )
        inferred = _apply_participation_audit_fields(
            audit,
            decision={
                "should_trade": True,
                "confidence": 0.58,
                "evidence_quality": 0.90,
                "evidence_basis": "direct",
                "edge_source": "computed",
            },
            settings=Settings(MIN_CONFIDENCE=0.62),
        )
        self.assertTrue(inferred)
        self.assertEqual(
            audit["participation_tier"],
            str(ParticipationTier.SKIP_FOR_NOW_WITH_REASON),
        )
        self.assertEqual(audit["participation_decision"], "confidence_below_min")
        self.assertTrue(audit["blocked_conviction"])
        self.assertEqual(audit["skip_due_to"], "weak_edge")

    def test_order_attempt_gets_execution_eligible_tier(self) -> None:
        audit = _build_execution_audit(
            final_action="order_attempt",
            final_reason="order_submitted",
        )
        inferred = _apply_participation_audit_fields(
            audit,
            decision={"should_trade": True, "confidence": 0.80},
            settings=Settings(),
        )
        self.assertTrue(inferred)
        self.assertEqual(
            audit["participation_tier"],
            str(ParticipationTier.EXECUTION_ELIGIBLE),
        )

    def test_no_trade_placeholder_is_research_gap(self) -> None:
        audit = _build_execution_audit(
            final_action="skip",
            final_reason="no_trade_recommended",
        )
        inferred = _apply_participation_audit_fields(
            audit,
            decision={
                "should_trade": False,
                "confidence": 0.50,
                "evidence_quality": 0.0,
                "evidence_basis": "absence_only",
                "edge_source": "none",
            },
            settings=Settings(),
        )
        self.assertTrue(inferred)
        self.assertEqual(
            audit["participation_tier"],
            str(ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE),
        )
        self.assertEqual(audit["participation_decision"], "no_trade_research_gap")
        self.assertEqual(audit["skip_due_to"], "lack_of_evidence")


class TestHistoricalFamilyFlattening(unittest.TestCase):
    def test_breakdown_fields_lift_to_top_level(self) -> None:
        audit = _build_execution_audit(
            final_action="research_queued",
            pre_analysis_breakdown={
                "pre_score_historical_family_samples": 138.0,
                "pre_score_historical_family_pnl_total": -16.64,
                "pre_score_historical_family_win_rate": 0.536,
                "pre_score_historical_family_pnl_ratio": -0.121,
            },
        )
        self.assertEqual(audit.get("historical_family_samples"), 138.0)
        self.assertAlmostEqual(audit.get("historical_family_pnl_total"), -16.64)
        self.assertAlmostEqual(audit.get("historical_family_win_rate"), 0.536)
        self.assertAlmostEqual(audit.get("historical_family_pnl_ratio"), -0.121)

    def test_top_level_value_wins_over_breakdown(self) -> None:
        audit = _build_execution_audit(
            final_action="research_queued",
            historical_family_samples=99,
            pre_analysis_breakdown={
                "pre_score_historical_family_samples": 138.0,
            },
        )
        self.assertEqual(audit.get("historical_family_samples"), 99)


class TestHistoricalFamilyStatsRuntimeLoad(unittest.TestCase):
    def test_confidence_shrink_block_does_not_reset_recent_family_stats(self) -> None:
        src = inspect.getsource(main_module.main)
        confidence_block = src.split(
            "if settings.HISTORICAL_CONFIDENCE_SHRINK_ENABLED:",
            1,
        )[1].split("recent_research_entries", 1)[0]
        self.assertNotIn("historical_family_stats_recent = {}", confidence_block)

    def test_emergency_research_drain_keeps_current_market_guard(self) -> None:
        src = inspect.getsource(main_module.main)
        emergency_block = src.split(
            "Emergency second-pass drain:",
            1,
        )[1].split("if drainable_research_entries:", 1)[0]
        self.assertIn("if mid not in current_market_ids:", emergency_block)
        self.assertNotIn("if mid in current_market_ids:", emergency_block)


class TestCounterfactualAuditFields(unittest.TestCase):
    def test_helper_emits_all_universal_thresholds(self) -> None:
        settings = Settings()
        fields = _build_counterfactual_audit_fields(
            reason="pre_analysis_score_soft_research",
            settings=settings,
            pre_analysis_score=0.55,
        )
        self.assertEqual(fields["counterfactual_required_confidence"], settings.MIN_CONFIDENCE)
        self.assertEqual(
            fields["counterfactual_required_evidence_quality"],
            settings.MIN_EVIDENCE_QUALITY_FOR_TRADE,
        )
        self.assertEqual(fields["counterfactual_required_edge_min"], settings.MIN_EDGE)
        self.assertAlmostEqual(
            fields["counterfactual_required_pre_analysis_score"],
            settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE,
        )

    def test_helper_emits_threshold_gap_when_score_provided(self) -> None:
        settings = Settings(PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE=0.60)
        fields = _build_counterfactual_audit_fields(
            reason="pre_analysis_score_soft_research",
            settings=settings,
            pre_analysis_score=0.45,
        )
        self.assertAlmostEqual(fields["pre_analysis_threshold_gap"], 0.15)

    def test_helper_emits_prefix_sample_size_for_historical_prefix_reason(self) -> None:
        settings = Settings()
        fields = _build_counterfactual_audit_fields(
            reason="pre_analysis_historical_prefix_pnl_block",
            settings=settings,
        )
        self.assertEqual(
            fields["counterfactual_required_prefix_sample_size"],
            settings.HISTORICAL_TICKER_PREFIX_HARD_BLOCK_MIN_SAMPLES,
        )

    def test_helper_emits_family_sample_size_for_family_reason(self) -> None:
        settings = Settings()
        fields = _build_counterfactual_audit_fields(
            reason="pre_analysis_historical_family_pnl_block",
            settings=settings,
        )
        self.assertEqual(
            fields["counterfactual_required_family_sample_size"],
            settings.HISTORICAL_FAMILY_MIN_SAMPLES,
        )

    def test_helper_quantifies_prefix_sample_shortfall(self) -> None:
        settings = Settings(HISTORICAL_TICKER_PREFIX_HARD_BLOCK_MIN_SAMPLES=20)
        fields = _build_counterfactual_audit_fields(
            reason="pre_analysis_historical_prefix_pnl_block",
            settings=settings,
            historical_metrics={"historical_gate_prefix_sample_size": 7},
        )
        self.assertEqual(fields["counterfactual_prefix_samples_short_by"], 13)

    def test_helper_emits_drawdown_counterfactual_for_drawdown_reason(self) -> None:
        settings = Settings(MAX_DAILY_DRAWDOWN_USDC=25.0)
        fields = _build_counterfactual_audit_fields(
            reason="pre_analysis_daily_drawdown_blocked",
            settings=settings,
        )
        self.assertEqual(
            fields["counterfactual_required_for_drawdown_block"],
            "drawdown_reset_or_position_close",
        )
        self.assertEqual(fields["counterfactual_max_daily_drawdown_usdc"], 25.0)


class TestSkipDueToForReason(unittest.TestCase):
    """Regression tests for the skip-reason categorizer used by audit/receipts."""

    def setUp(self) -> None:
        from main import _skip_due_to_for_reason
        self._fn = _skip_due_to_for_reason

    def test_pre_analysis_score_soft_research_returns_weak_pre_analysis_score(self) -> None:
        """Score-band soft-research must be tagged as a pre-analysis-score
        weakness, not as 'weak_edge'. Distinguishing these in receipts lets
        analytics tell apart 'low pre-analysis opportunity score' from 'low
        runtime edge'."""
        self.assertEqual(
            self._fn("pre_analysis_score_soft_research"),
            "weak_pre_analysis_score",
        )
        self.assertEqual(
            self._fn("pre_analysis_score_below_min"),
            "weak_pre_analysis_score",
        )
        self.assertEqual(
            self._fn("pre_analysis_score_far_below_min"),
            "weak_pre_analysis_score",
        )

    def test_daily_drawdown_blocked_returns_risk_cap(self) -> None:
        self.assertEqual(
            self._fn("pre_analysis_daily_drawdown_blocked"),
            "risk_cap",
        )
        self.assertEqual(self._fn("daily_drawdown_limit"), "risk_cap")

    def test_fallback_edge_high_churn_returns_repeated_churn(self) -> None:
        """High-churn fallback-edge reasons describe repeated non-actionable
        cycles, not weak-edge market-judgment failures."""
        self.assertEqual(
            self._fn("pre_analysis_fallback_edge_high_churn"),
            "repeated_churn",
        )

    def test_evidence_reason_returns_lack_of_evidence(self) -> None:
        self.assertEqual(self._fn("evidence_quality_below_min"), "lack_of_evidence")

    def test_timeout_reason_returns_timeout(self) -> None:
        self.assertEqual(self._fn("grok_stream_timeout"), "timeout")


class TestSyntheticDecisionAuditFields(unittest.TestCase):
    """Receipts for markets that were never actually analyzed by Grok must
    explicitly mark themselves so analytics can partition real findings from
    placeholder triples (eq=0.0/edge_source=none/basis=absence_only)."""

    def test_constant_carries_expected_flags(self) -> None:
        from main import _SYNTHETIC_DECISION_AUDIT_FIELDS as fields
        self.assertTrue(fields["analysis_skipped"])
        self.assertTrue(fields["evidence_quality_unevaluated"])
        self.assertTrue(fields["edge_source_unevaluated"])
        self.assertFalse(fields["pre_analysis_hard_reject"])

    def test_synthetic_audit_resolves_hard_reject_naming(self) -> None:
        """final_action=research_queued must NOT be conflated with hard reject;
        the explicit pre_analysis_hard_reject=False resolves the legacy
        naming confusion in receipts."""
        from main import _SYNTHETIC_DECISION_AUDIT_FIELDS, _build_execution_audit
        audit = _build_execution_audit(
            decision_terminal=False,
            final_action="research_queued",
            final_reason="pre_analysis_score_soft_research",
            decision_origin="synthetic_research_queue",
            **_SYNTHETIC_DECISION_AUDIT_FIELDS,
        )
        self.assertEqual(audit["final_action"], "research_queued")
        self.assertFalse(audit["pre_analysis_hard_reject"])
        self.assertTrue(audit["analysis_skipped"])
        self.assertTrue(audit["synthetic_decision"])

    def test_research_queue_payload_embeds_audit_context(self) -> None:
        decision = TradeDecision(
            should_trade=False,
            outcome="YES",
            confidence=0.50,
            bet_size_pct=0.0,
            reasoning="queued",
            edge_source="none",
            evidence_basis="absence_only",
            evidence_quality=0.0,
            abstain=True,
        )
        audit = _build_execution_audit(
            final_action="research_queued",
            final_reason="pre_analysis_score_soft_research",
            decision_origin="synthetic_research_queue",
            participation_tier=str(ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE),
            skip_due_to="weak_pre_analysis_score",
            pre_analysis_score=0.41,
            pre_analysis_breakdown={"pre_score_market_subfamily": "generic_macro_release"},
            edge_market=0.12,
            edge_required=0.08,
            score_final=0.24,
            score_kelly_raw=0.18,
            score_lmsr_price=0.48,
            why_not_execution_eligible="score below threshold",
            counterfactual_required_pre_analysis_score=0.55,
        )
        payload = json.loads(_research_queue_last_decision_json(decision, audit))
        self.assertEqual(payload["audit"]["pre_analysis_score"], 0.41)
        self.assertEqual(payload["participation_tier"], audit["participation_tier"])
        self.assertEqual(payload["skip_due_to"], "weak_pre_analysis_score")
        self.assertEqual(payload["edge_market"], 0.12)
        self.assertEqual(payload["edge_required"], 0.08)
        self.assertEqual(payload["score_final"], 0.24)
        self.assertEqual(payload["score_kelly_raw"], 0.18)
        self.assertEqual(payload["score_lmsr_price"], 0.48)
        self.assertEqual(
            payload["counterfactual_required_pre_analysis_score"],
            0.55,
        )

    def test_no_research_queued_audit_marks_hard_reject(self) -> None:
        """Invariant: every production path that sets final_action=
        research_queued routes through the synthetic-audit constant which
        explicitly stamps pre_analysis_hard_reject=False. This regression
        test exercises every reason string the audit currently emits with
        research_queued and asserts the invariant holds."""
        from main import _SYNTHETIC_DECISION_AUDIT_FIELDS, _build_execution_audit
        research_queued_reasons = (
            "pre_analysis_score_soft_research",
            "pre_analysis_score_far_below_min",
            "pre_analysis_historical_prefix_pnl_block",
            "pre_analysis_historical_prefix_small_sample_negative",
            "pre_analysis_historical_family_pnl_block",
            "pre_analysis_crypto_historically_unprofitable",
            "pre_analysis_repeated_non_actionable_market",
            "pre_analysis_repeated_non_actionable_bin_market",
            "pre_analysis_repeated_churn_market",
            "pre_analysis_fallback_edge_high_churn",
            "repeated_non_actionable_research_only",
            "evidence_quality_below_min",
            "confidence_below_min",
            "edge_above_reasonable_max",
            "score_gate_blocked",
            "daily_drawdown_blocked",
            "grok_stream_timeout",
        )
        for reason in research_queued_reasons:
            audit = _build_execution_audit(
                decision_terminal=False,
                final_action="research_queued",
                final_reason=reason,
                decision_origin="synthetic_research_queue",
                **_SYNTHETIC_DECISION_AUDIT_FIELDS,
            )
            self.assertEqual(audit["final_action"], "research_queued")
            self.assertFalse(
                audit.get("pre_analysis_hard_reject", False),
                f"Invariant violated for final_reason={reason}: "
                f"pre_analysis_hard_reject must be False when "
                f"final_action='research_queued'.",
            )

    def test_production_source_has_no_hard_reject_true(self) -> None:
        """Static safety net: scan main.py source for any literal that sets
        pre_analysis_hard_reject=True. Catches regressions that the runtime
        invariant test would miss because they wouldn't go through the
        _SYNTHETIC_DECISION_AUDIT_FIELDS constant."""
        from pathlib import Path
        import re
        main_path = Path(__file__).resolve().parent.parent / "main.py"
        src = main_path.read_text(encoding="utf-8")
        # Strip lines that document the invariant (string literals / comments
        # that mention the field name with True for pedagogical reasons).
        # Match assignments only: optional quote, key, optional quote, then
        # `=` or `:` separator, then `True`.
        offending_lines: list[str] = []
        pattern = re.compile(
            r'["\']?pre_analysis_hard_reject["\']?\s*[:=]\s*True\b'
        )
        for line_no, line in enumerate(src.splitlines(), start=1):
            stripped = line.lstrip()
            # Skip comment-only lines.
            if stripped.startswith("#"):
                continue
            if pattern.search(line):
                offending_lines.append(f"{line_no}: {line.rstrip()}")
        self.assertFalse(
            offending_lines,
            "Production code must never set pre_analysis_hard_reject=True; "
            "found:\n" + "\n".join(offending_lines),
        )


class TestPreviousAnalysisAnchorEvidence(unittest.TestCase):
    def test_anchor_evidence_fields_preserved(self) -> None:
        anchor = {
            "outcome": "YES",
            "confidence": 0.74,
            "reasoning": "Found settlement-aligned source.",
            "evidence_quality": 0.82,
            "edge_source": "computed",
            "evidence_basis": "direct",
            "implied_prob_external": 0.55,
            "edge_external": 0.12,
            "my_prob": 0.67,
        }
        prev = _build_previous_analysis(anchor)
        assert prev is not None
        self.assertAlmostEqual(prev.evidence_quality, 0.82)
        self.assertEqual(prev.edge_source, "computed")
        self.assertEqual(prev.evidence_basis, "direct")
        self.assertAlmostEqual(prev.implied_prob_external, 0.55)
        self.assertAlmostEqual(prev.edge_external, 0.12)
        self.assertAlmostEqual(prev.my_prob, 0.67)

    def test_direct_evidence_anchor_appends_material_change_hint(self) -> None:
        anchor = {
            "outcome": "YES",
            "confidence": 0.74,
            "reasoning": "Found settlement-aligned source.",
            "evidence_quality": 0.82,
            "edge_source": "computed",
            "evidence_basis": "direct",
        }
        prev = _build_previous_analysis(anchor)
        assert prev is not None
        self.assertIn("material change", prev.reasoning.lower())

    def test_proxy_evidence_anchor_does_not_append_hint(self) -> None:
        anchor = {
            "outcome": "YES",
            "confidence": 0.50,
            "reasoning": "Looked at proxy data only.",
            "evidence_quality": 0.40,
            "edge_source": "fallback",
            "evidence_basis": "proxy",
        }
        prev = _build_previous_analysis(anchor)
        assert prev is not None
        self.assertNotIn("material change", prev.reasoning.lower())

    def test_missing_evidence_fields_default_safely(self) -> None:
        anchor = {
            "outcome": "YES",
            "confidence": 0.50,
            "reasoning": "Minimal anchor.",
        }
        prev = _build_previous_analysis(anchor)
        assert prev is not None
        self.assertEqual(prev.evidence_quality, 0.0)
        self.assertIsNone(prev.edge_source)
        self.assertIsNone(prev.evidence_basis)


class TestTierBreakdownFormat(unittest.TestCase):
    def test_empty_breakdown_renders_empty_braces(self) -> None:
        self.assertEqual(_format_tier_breakdown_for_log(None), "{}")
        self.assertEqual(_format_tier_breakdown_for_log({}), "{}")

    def test_single_tier_renders_label(self) -> None:
        result = _format_tier_breakdown_for_log({"research_only_learning_queue": 142})
        self.assertEqual(result, "{research:142}")

    def test_multiple_tiers_render_alphabetically_by_key(self) -> None:
        result = _format_tier_breakdown_for_log(
            {
                "research_only_learning_queue": 100,
                "skip_for_now_with_reason": 5,
                "monitor_only": 2,
            }
        )
        self.assertIn("research:100", result)
        self.assertIn("skip:5", result)
        self.assertIn("monitor:2", result)

    def test_zero_count_tiers_omitted(self) -> None:
        result = _format_tier_breakdown_for_log(
            {"research_only_learning_queue": 0, "monitor_only": 3}
        )
        self.assertNotIn("research:0", result)
        self.assertIn("monitor:3", result)

    def test_unknown_tier_falls_back_to_raw_label(self) -> None:
        result = _format_tier_breakdown_for_log({"some_future_tier": 1})
        self.assertIn("some_future_tier:1", result)


class TestSummarizeDistribution(unittest.TestCase):
    """Score-distribution helper used by cycle receipts (5f)."""

    def _fn(self, samples):
        from main import _summarize_distribution
        return _summarize_distribution(samples)

    def test_empty_samples_return_count_only(self) -> None:
        result = self._fn([])
        self.assertEqual(result, {"count": 0})

    def test_single_sample_returns_collapsed_distribution(self) -> None:
        result = self._fn([0.42])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["min"], 0.42)
        self.assertEqual(result["max"], 0.42)
        self.assertEqual(result["p50"], 0.42)

    def test_distribution_percentiles_match_linear_interpolation(self) -> None:
        result = self._fn([0.10, 0.20, 0.30, 0.40, 0.50])
        self.assertEqual(result["count"], 5)
        self.assertEqual(result["min"], 0.10)
        self.assertEqual(result["max"], 0.50)
        self.assertEqual(result["p50"], 0.30)
        self.assertAlmostEqual(result["p25"], 0.20)
        self.assertAlmostEqual(result["p75"], 0.40)


class TestResearchQueueCycleLogMaxlenSetting(unittest.TestCase):
    """Configurable per-cycle research-queue capture log (5g)."""

    def test_default_maxlen_is_200(self) -> None:
        from config import Settings
        self.assertEqual(Settings().RESEARCH_QUEUE_CYCLE_LOG_MAXLEN, 200)

    def test_setting_can_be_overridden(self) -> None:
        from config import Settings
        custom = Settings(RESEARCH_QUEUE_CYCLE_LOG_MAXLEN=50)
        self.assertEqual(custom.RESEARCH_QUEUE_CYCLE_LOG_MAXLEN, 50)


class TestAdaptiveResearchBandSettings(unittest.TestCase):
    """Adaptive widening of the soft-research routing band (5e)."""

    def test_default_settings_enable_adaptive_band(self) -> None:
        from config import Settings
        s = Settings()
        self.assertTrue(s.PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND_ADAPTIVE_ENABLED)
        self.assertEqual(s.PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND_MAX, 0.30)
        self.assertGreaterEqual(
            s.PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND_MAX,
            s.PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND,
        )

    def test_adaptive_band_can_be_disabled(self) -> None:
        from config import Settings
        s = Settings(PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND_ADAPTIVE_ENABLED=False)
        self.assertFalse(s.PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND_ADAPTIVE_ENABLED)


def test_family_profitable_uses_positive_short_window() -> None:
    context = {
        "historical_family_pnl_total": 15.0,
        "historical_family_sample_size": 25,
    }
    assert main_module._family_is_profitable_from_context(context) is True


def test_family_profitable_blend_recognizes_lifetime_through_minor_drawdown() -> None:
    context = {
        "historical_family_pnl_total": -20.0,
        "historical_family_sample_size": 25,
        "lifetime_family_pnl_total": 120.0,
        "lifetime_family_sample_size": 80,
    }
    assert main_module._family_is_profitable_from_context(context) is True


def test_family_profitable_blend_rejects_severe_recent_drawdown() -> None:
    context = {
        "historical_family_pnl_total": -130.0,
        "historical_family_sample_size": 25,
        "lifetime_family_pnl_total": 120.0,
        "lifetime_family_sample_size": 80,
    }
    assert main_module._family_is_profitable_from_context(context) is False


def test_family_profitable_blend_requires_lifetime_sample() -> None:
    context = {
        "historical_family_pnl_total": -5.0,
        "historical_family_sample_size": 25,
        "lifetime_family_pnl_total": 50.0,
        "lifetime_family_sample_size": 10,
    }
    assert main_module._family_is_profitable_from_context(context) is False


def test_family_profitable_rejects_negative_lifetime() -> None:
    context = {
        "historical_family_pnl_total": -77.0,
        "historical_family_sample_size": 80,
        "lifetime_family_pnl_total": -50.0,
        "lifetime_family_sample_size": 120,
    }
    assert main_module._family_is_profitable_from_context(context) is False


class TestLmsrSignalPosterior(unittest.TestCase):
    """LMSR mispricing signal must use the same posterior as the edge gate."""

    def test_direct_floor_lifts_posterior(self) -> None:
        self.assertAlmostEqual(
            _posterior_for_lmsr_signal(
                bayesian_posterior_applied=None,
                effective_confidence=0.70,
                execution_posterior_floor=0.88,
            ),
            0.88,
        )

    def test_without_floor_uses_bayesian_then_confidence(self) -> None:
        self.assertAlmostEqual(
            _posterior_for_lmsr_signal(
                bayesian_posterior_applied=0.63,
                effective_confidence=0.70,
                execution_posterior_floor=None,
            ),
            0.63,
        )
        self.assertAlmostEqual(
            _posterior_for_lmsr_signal(
                bayesian_posterior_applied=None,
                effective_confidence=0.70,
                execution_posterior_floor=None,
            ),
            0.70,
        )

    def test_floored_posterior_unblocks_real_mispricing(self) -> None:
        # Regression for KXWTI-26JUN0814-T89.99: model prob 0.925 / floor 0.88
        # at LMSR price 0.69 was blocked because the signal used calibrated
        # confidence (0.70), yielding |0.0099| < 0.03 min inefficiency.
        from lmsr import inefficiency_signal

        lmsr_price = 0.690003352383589
        old_signal = inefficiency_signal(0.70, lmsr_price)
        self.assertLess(abs(old_signal), 0.03)

        posterior = _posterior_for_lmsr_signal(
            bayesian_posterior_applied=None,
            effective_confidence=0.70,
            execution_posterior_floor=0.88,
        )
        new_signal = inefficiency_signal(posterior, lmsr_price)
        self.assertGreaterEqual(abs(new_signal), 0.03)


class TestDailyDrawdownBasis(unittest.TestCase):
    """Drawdown gates should measure today's entries, not legacy settlements."""

    class _StubStateManager:
        def __init__(self, value: float | None = None, error: bool = False) -> None:
            self.value = value
            self.error = error
            self.last_since: datetime | None = None

        def get_attributed_daily_realized_pnl(self, since: datetime) -> float:
            self.last_since = since
            if self.error:
                raise RuntimeError("db unavailable")
            return float(self.value or 0.0)

    def test_prefers_attributed_realized_pnl(self) -> None:
        stub = self._StubStateManager(value=-12.5)
        delta, basis = _daily_drawdown_basis_usdc(
            state_manager=stub,
            trade_day=date(2026, 6, 9),
            day_start_balance=100.0,
            current_balance=50.0,
        )
        self.assertEqual(basis, "attributed_realized")
        self.assertAlmostEqual(delta, -12.5)
        self.assertEqual(stub.last_since, datetime(2026, 6, 9, tzinfo=timezone.utc))
        self.assertTrue(
            _daily_drawdown_cap_reached(
                daily_balance_delta=delta,
                max_daily_drawdown_usdc=10.0,
            )
        )

    def test_falls_back_to_balance_delta_on_query_failure(self) -> None:
        stub = self._StubStateManager(error=True)
        delta, basis = _daily_drawdown_basis_usdc(
            state_manager=stub,
            trade_day=date(2026, 6, 9),
            day_start_balance=100.0,
            current_balance=50.0,
        )
        self.assertEqual(basis, "balance_delta")
        self.assertAlmostEqual(delta, -50.0)

    def test_legacy_settlement_losses_do_not_trip_cap(self) -> None:
        # Balance fell $24.21 from legacy April/May settlements, but today's
        # entries realized zero loss: the gate must stay open.
        stub = self._StubStateManager(value=0.0)
        delta, _basis = _daily_drawdown_basis_usdc(
            state_manager=stub,
            trade_day=date(2026, 6, 9),
            day_start_balance=80.0,
            current_balance=55.79,
        )
        self.assertFalse(
            _daily_drawdown_cap_reached(
                daily_balance_delta=delta,
                max_daily_drawdown_usdc=15.0,
            )
        )


class TestSatelliteRecapBet(unittest.TestCase):
    """Satellite cap should resize, not hard-skip, execution-eligible bets."""

    def test_clamps_bet_back_to_cap(self) -> None:
        recap = _satellite_recap_bet(
            bet_pct=0.417,
            satellite_cap_pct=0.25,
            min_bet_floor_applied=False,
            max_bet_usdc=50.0,
            min_bet_usdc=1.0,
        )
        self.assertIsNotNone(recap)
        clamped_pct, clamped_amount = recap
        self.assertAlmostEqual(clamped_pct, 0.25)
        self.assertAlmostEqual(clamped_amount, 12.5)

    def test_recap_amount_never_drops_below_min_bet(self) -> None:
        recap = _satellite_recap_bet(
            bet_pct=0.90,
            satellite_cap_pct=0.05,
            min_bet_floor_applied=False,
            max_bet_usdc=50.0,
            min_bet_usdc=5.0,
        )
        self.assertIsNotNone(recap)
        clamped_pct, clamped_amount = recap
        self.assertAlmostEqual(clamped_amount, 5.0)
        self.assertAlmostEqual(clamped_pct, 0.10)

    def test_no_recap_when_under_cap_or_uncapped(self) -> None:
        self.assertIsNone(
            _satellite_recap_bet(
                bet_pct=0.20,
                satellite_cap_pct=0.25,
                min_bet_floor_applied=False,
                max_bet_usdc=50.0,
                min_bet_usdc=1.0,
            )
        )
        self.assertIsNone(
            _satellite_recap_bet(
                bet_pct=0.90,
                satellite_cap_pct=None,
                min_bet_floor_applied=False,
                max_bet_usdc=50.0,
                min_bet_usdc=1.0,
            )
        )


class TestConvictionRepairEdgeEligibility(unittest.TestCase):
    """Repair eligibility aligned with the execution edge standard (0.12)."""

    def test_mid_band_edge_now_triggers_repair(self) -> None:
        market = Market(
            id="KXGOLDD-26JUN0917-T4252",
            question="Will the gold close price be above 4252?",
            outcomes=[
                MarketOutcome(name="YES", price=0.55),
                MarketOutcome(name="NO", price=0.45),
            ],
            liquidity_usdc=800.0,
            resolution_criteria="Official close",
        )
        decision = TradeDecision(
            should_trade=False,
            outcome="YES",
            confidence=0.68,
            bet_size_pct=0.0,
            reasoning="Edge present but conviction held back.",
            edge_external=0.14,
            edge_source="computed",
            evidence_basis="direct",
            evidence_quality=0.95,
            primary_source_url="https://www.wsj.com/market-data/quotes/futures/GC00",
            source_match_class="settlement_aligned",
        )
        diagnostics: dict[str, object] = {}

        reason = _conviction_repair_reason(
            decision=decision,
            market=market,
            settings=Settings(),
            score_result=None,
            score_threshold=None,
            diagnostics=diagnostics,
        )

        self.assertEqual(reason, "conviction_repair_no_trade_contradiction")
        self.assertTrue(diagnostics["conviction_repair_triggerable"])

    def test_edge_below_new_min_still_misses(self) -> None:
        market = Market(
            id="KXGOLDD-26JUN0917-T4252",
            question="Will the gold close price be above 4252?",
            outcomes=[
                MarketOutcome(name="YES", price=0.55),
                MarketOutcome(name="NO", price=0.45),
            ],
            liquidity_usdc=800.0,
            resolution_criteria="Official close",
        )
        decision = TradeDecision(
            should_trade=False,
            outcome="YES",
            confidence=0.60,
            bet_size_pct=0.0,
            reasoning="Edge too small for repair.",
            edge_external=0.08,
            edge_source="computed",
            evidence_basis="direct",
            evidence_quality=0.95,
            primary_source_url="https://www.wsj.com/market-data/quotes/futures/GC00",
            source_match_class="settlement_aligned",
        )
        diagnostics: dict[str, object] = {}

        self.assertIsNone(
            _conviction_repair_reason(
                decision=decision,
                market=market,
                settings=Settings(),
                score_result=None,
                score_threshold=None,
                diagnostics=diagnostics,
            )
        )
        self.assertEqual(
            diagnostics["conviction_repair_missed_reason"],
            "edge_below_repair_min",
        )


class SelfConsistencyGatingTest(unittest.TestCase):
    def _candidate(self, market_id: str, score: float) -> dict:
        return {
            "market": Market(
                id=market_id,
                question="q",
                outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
            ),
            "pre_analysis_score": score,
        }

    def test_selects_top_n_by_pre_analysis_score(self) -> None:
        candidates = [
            self._candidate("a", 0.9),
            self._candidate("b", 0.4),
            self._candidate("c", 0.7),
        ]

        class _Settings:
            GROK_SELF_CONSISTENCY_TOP_CANDIDATES = 2

        allowed = main_module._self_consistency_allowed_market_ids(candidates, _Settings())
        self.assertEqual(allowed, {"a", "c"})

    def test_zero_top_candidates_disables_gating(self) -> None:
        candidates = [self._candidate("a", 0.9), self._candidate("b", 0.4)]

        class _Settings:
            GROK_SELF_CONSISTENCY_TOP_CANDIDATES = 0

        self.assertIsNone(
            main_module._self_consistency_allowed_market_ids(candidates, _Settings())
        )


if __name__ == "__main__":
    unittest.main()
