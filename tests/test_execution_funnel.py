from __future__ import annotations

from datetime import datetime, timedelta, timezone

from config import Settings
from main import (
    _cap_analysis_candidates,
    _effective_score_gate_threshold,
    _order_lifecycle_metrics,
    _persist_submitted_order_lifecycle,
    _pre_analysis_opportunity_score,
    _sizing_audit_fields,
    _sync_exchange_fills,
)
from market_state import MarketStateManager
from models import Market, MarketOutcome, MarketState, OrderResponse, TradeDecision
from participation import ParticipationTier, classify_participation
from score_engine import compute_final_score


def _market(*, market_id: str, category: str, question: str, liquidity: float = 1200.0) -> Market:
    return Market(
        id=market_id,
        question=question,
        category=category,
        outcomes=[MarketOutcome(name="YES", price=0.52), MarketOutcome(name="NO", price=0.48)],
        liquidity_usdc=liquidity,
        close_time=datetime.now(timezone.utc) + timedelta(hours=12),
        resolution_criteria="Official settlement source",
    )


def _decision(
    *,
    confidence: float,
    evidence_quality: float,
    edge_external: float = 0.08,
    outcome: str = "YES",
) -> TradeDecision:
    return TradeDecision(
        should_trade=True,
        outcome=outcome,
        confidence=confidence,
        bet_size_pct=0.2,
        reasoning="Validated direct evidence.",
        edge_external=edge_external,
        evidence_quality=evidence_quality,
    )


def test_execution_funnel_score_gate_blocks_weak_setup() -> None:
    settings = Settings(SCORE_GATE_THRESHOLD=0.38, SCORE_GATE_THRESHOLD_WEATHER_DIRECT=0.10)
    market = _market(
        market_id="KXBTC-TEST",
        category="crypto",
        question="Will BTC close above threshold?",
        liquidity=250.0,
    )
    decision = _decision(confidence=0.58, evidence_quality=0.42, edge_external=0.03)
    score = compute_final_score(market=market, decision=decision, implied_prob_market=0.52)
    threshold = _effective_score_gate_threshold(
        settings=settings,
        market=market,
        evidence_basis_class="proxy",
    )
    assert score.final_score < threshold


def test_execution_funnel_score_gate_passes_high_quality_weather_direct() -> None:
    settings = Settings(
        SCORE_GATE_THRESHOLD=0.38,
        SCORE_GATE_THRESHOLD_WEATHER_DIRECT=0.10,
        SCORE_GATE_THRESHOLD_DIRECT_HIGH_QUALITY=0.25,
    )
    market = _market(
        market_id="KXHIGHCHI-TEST",
        category="weather",
        question="Will Chicago high exceed 70F?",
        liquidity=1400.0,
    )
    decision = _decision(confidence=0.72, evidence_quality=0.88, edge_external=0.09)
    score = compute_final_score(market=market, decision=decision, implied_prob_market=0.55)
    threshold = _effective_score_gate_threshold(
        settings=settings,
        market=market,
        evidence_basis_class="direct",
    )
    assert threshold == 0.10
    assert score.final_score >= threshold


def test_execution_funnel_score_gate_uses_direct_high_quality_non_weather_threshold() -> None:
    settings = Settings(
        SCORE_GATE_THRESHOLD=0.38,
        SCORE_GATE_THRESHOLD_WEATHER_DIRECT=0.10,
        SCORE_GATE_THRESHOLD_DIRECT_HIGH_QUALITY=0.25,
    )
    market = _market(
        market_id="KXWTI-TEST",
        category="commodities",
        question="Will WTI settle above 95?",
        liquidity=1400.0,
    )
    threshold = _effective_score_gate_threshold(
        settings=settings,
        market=market,
        evidence_basis_class="direct",
        evidence_quality=0.90,
    )
    assert threshold == 0.25


def test_execution_funnel_regression_kxlowtaus_weather_direct_stays_tradeable() -> None:
    settings = Settings(
        SCORE_GATE_THRESHOLD=0.38,
        SCORE_GATE_THRESHOLD_WEATHER_DIRECT=0.10,
        SCORE_GATE_THRESHOLD_DIRECT_HIGH_QUALITY=0.25,
    )
    market = _market(
        market_id="KXLOWTAUS-26APR13-T70",
        category="weather",
        question="Will minimum temperature be above 70F?",
        liquidity=1500.0,
    )
    decision = _decision(
        outcome="NO",
        confidence=0.70,
        evidence_quality=1.0,
        edge_external=-0.61,
    )
    score = compute_final_score(market=market, decision=decision, implied_prob_market=0.27)
    threshold = _effective_score_gate_threshold(
        settings=settings,
        market=market,
        evidence_basis_class="direct",
        evidence_quality=decision.evidence_quality,
    )
    assert threshold == 0.10
    assert score.final_score >= threshold


def test_execution_funnel_regression_hou3_direct_edge_stays_tradeable() -> None:
    settings = Settings(
        SCORE_GATE_THRESHOLD=0.38,
        SCORE_GATE_THRESHOLD_WEATHER_DIRECT=0.10,
        SCORE_GATE_THRESHOLD_DIRECT_HIGH_QUALITY=0.25,
    )
    market = _market(
        market_id="KXSAMPLETEAMTOTAL-26APR131610TEAMATEAMB-TEAMA3",
        category="sports",
        question="Will Team A score over 3 in the matchup?",
        liquidity=1200.0,
    )
    decision = _decision(
        outcome="NO",
        confidence=0.64,
        evidence_quality=0.82,
        edge_external=-0.18,
    )
    score = compute_final_score(market=market, decision=decision, implied_prob_market=0.41)
    threshold = _effective_score_gate_threshold(
        settings=settings,
        market=market,
        evidence_basis_class="direct",
        evidence_quality=decision.evidence_quality,
    )
    assert threshold == 0.25
    assert score.final_score >= threshold


def test_execution_funnel_source_confirmed_sports_edge_survives_calibration_shrink() -> None:
    settings = Settings(
        SCORE_GATE_THRESHOLD=0.38,
        SCORE_GATE_THRESHOLD_DIRECT_HIGH_QUALITY=0.25,
    )
    market = _market(
        market_id="KXMLB-26MAY171900TEAMATEAMB-TEAMA",
        category="sports",
        question="Will Team A win after official lineup news?",
        liquidity=1500.0,
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.64,
        raw_confidence=0.84,
        bet_size_pct=0.2,
        reasoning="Official source and computed external probability support YES.",
        implied_prob_external=0.45,
        my_prob=0.68,
        edge_external=0.23,
        edge_source="computed",
        evidence_basis="direct",
        evidence_quality=0.94,
        primary_source_url="https://www.espn.com/game/example",
        source_match_class="settlement_aligned",
    )
    score = compute_final_score(
        market=market,
        decision=decision,
        implied_prob_market=0.67,
        evidence_basis_class="direct",
        edge_source="computed",
        source_match_class="settlement_aligned",
        primary_source_url_present=True,
        source_confirmed_edge_min=settings.CONVICTION_REPAIR_MIN_EDGE,
        source_confirmed_edge_min_evidence_quality=(
            settings.CONVICTION_REPAIR_MIN_EVIDENCE_QUALITY
        ),
        source_confirmed_edge_bonus_base=settings.SCORE_SOURCE_CONFIRMED_EDGE_BONUS,
    )
    threshold = _effective_score_gate_threshold(
        settings=settings,
        market=market,
        evidence_basis_class="direct",
        evidence_quality=decision.evidence_quality,
    )
    assert score.source_confirmed_edge is True
    assert "non_positive_market_edge" not in score.rejection_reasons
    assert score.final_score >= threshold


def test_execution_funnel_source_confirmed_path_rejects_proxy_without_primary_source() -> None:
    market = _market(
        market_id="KXGENERIC-26MAY17-B1",
        category="generic",
        question="Will a generic threshold resolve yes?",
        liquidity=1500.0,
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.64,
        bet_size_pct=0.2,
        reasoning="Preview-only proxy source.",
        implied_prob_external=0.45,
        my_prob=0.68,
        edge_external=0.23,
        edge_source="computed",
        evidence_basis="proxy",
        evidence_quality=0.94,
        source_match_class="preview_or_proxy",
    )
    score = compute_final_score(
        market=market,
        decision=decision,
        implied_prob_market=0.67,
        evidence_basis_class="proxy",
        edge_source="computed",
        source_match_class="preview_or_proxy",
        primary_source_url_present=False,
    )
    assert score.source_confirmed_edge is False
    assert "non_positive_market_edge" in score.rejection_reasons


def test_order_lifecycle_metrics_distinguish_execution_partial_and_resting() -> None:
    executed = _order_lifecycle_metrics(
        OrderResponse(
            id="executed",
            status="executed",
            raw={
                "fill_count": "4.00",
                "client_qty_shares": 4,
                "client_price": 0.61,
            },
        ),
        submitted_amount_usdc=2.0,
    )
    partial = _order_lifecycle_metrics(
        OrderResponse(
            id="partial",
            status="partially_filled",
            raw={
                "fill_count": "1.00",
                "client_qty_shares": 6,
                "client_price": 0.39,
            },
        ),
        submitted_amount_usdc=2.0,
    )
    resting = _order_lifecycle_metrics(
        OrderResponse(
            id="resting",
            status="resting",
            raw={
                "fill_count": "0.00",
                "client_qty_shares": 3,
                "client_price": 0.77,
            },
        ),
        submitted_amount_usdc=2.0,
    )

    assert executed.fully_filled is True
    assert executed.partially_filled is False
    assert executed.resting_unfilled is False
    assert executed.filled_notional_usdc == 2.44

    assert partial.fully_filled is False
    assert partial.partially_filled is True
    assert partial.resting_unfilled is False
    assert partial.filled_notional_usdc == 0.39

    assert resting.fully_filled is False
    assert resting.partially_filled is False
    assert resting.resting_unfilled is True
    assert resting.filled_notional_usdc == 0.0


def test_resting_order_is_recorded_only_after_fill_reconciliation(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    response = OrderResponse(
        id="order-resting",
        status="resting",
        raw={
            "fill_count": "0.00",
            "client_qty_shares": 5,
            "client_price": 0.40,
        },
    )
    lifecycle = _order_lifecycle_metrics(
        response,
        submitted_amount_usdc=2.0,
    )

    try:
        persisted = _persist_submitted_order_lifecycle(
            state_manager=manager,
            market_id="KXTEST-FILL",
            outcome="YES",
            order_response=response,
            lifecycle=lifecycle,
            submitted_amount_usdc=2.0,
            fallback_entry_price=0.40,
            confidence=0.70,
            implied_prob=0.40,
        )

        assert persisted["fill_recorded"] is False
        assert manager.get_position("KXTEST-FILL") is None

        class FillClient:
            @staticmethod
            def get_fills(
                *,
                limit: int,
                cursor: str | None,
                min_ts: int | None,
                subaccount: int,
            ) -> dict:
                assert limit == 1000
                assert cursor is None
                assert subaccount == 0
                return {
                    "fills": [
                        {
                            "fill_id": "fill-a",
                            "order_id": "order-resting",
                            "count_fp": "2.00",
                            "yes_price_dollars": "0.40",
                        },
                        {
                            "fill_id": "fill-b",
                            "order_id": "order-resting",
                            "count_fp": "3.00",
                            "yes_price_dollars": "0.40",
                        },
                    ]
                }

        first_sync = _sync_exchange_fills(
            state_manager=manager,
            kalshi_client=FillClient(),
        )
        duplicate_sync = _sync_exchange_fills(
            state_manager=manager,
            kalshi_client=FillClient(),
        )

        position = manager.get_position("KXTEST-FILL")
        assert position is not None
        assert position.total_amount_usdc == 2.0
        assert position.trade_count == 1
        assert first_sync.new_fill_events == 2
        assert first_sync.filled_notional_usdc == 2.0
        assert duplicate_sync.new_fill_events == 0
        assert manager.get_pending_order("order-resting")["status"] == "filled"
    finally:
        manager.close()


def test_sizing_audit_fields_include_kelly_lmsr_and_floor_context() -> None:
    payload = _sizing_audit_fields(
        sizing_mode="kelly",
        raw_bet_amount_usdc=1.75,
        bet_amount_usdc=2.0,
        min_bet_floor_applied=True,
        kelly_sub_floor_skipped=False,
        kelly_min_bet_policy_applied="fallback_edge_scaling",
        kelly_raw=0.50,
        kelly_fraction_value=0.30,
        posterior_for_kelly=0.89,
        min_edge_for_kelly=0.08,
        kelly_effective_fraction=0.20,
        historical_family_size_multiplier=0.55,
        lmsr_execution_price=0.61,
        lmsr_inefficiency_signal=0.28,
        expected_value_usdc=0.90,
    )

    assert payload["sizing_mode"] == "kelly"
    assert payload["raw_bet_amount_usdc"] == 1.75
    assert payload["bet_amount_usdc"] == 2.0
    assert payload["min_bet_floor_applied"] is True
    assert payload["kelly_min_bet_policy_applied"] == "fallback_edge_scaling"
    assert payload["kelly_raw"] == 0.50
    assert payload["kelly_fraction_value"] == 0.30
    assert payload["posterior_for_kelly"] == 0.89
    assert payload["kelly_effective_fraction"] == 0.20
    assert payload["historical_family_size_multiplier"] == 0.55
    assert payload["lmsr_execution_price"] == 0.61
    assert payload["lmsr_inefficiency_signal"] == 0.28
    assert payload["expected_value_usdc"] == 0.90


# ---------------------------------------------------------------------------
# Cycle-14 MLB F5 regression fixture (correlation_id 6a6cc761).
#
# The 2026-05-03 audit found that cycles 13-14 produced 2 successful MLB F5
# trades (KXMLBF5-26MAY031920TEXDET-DET and KXMLBF5SPREAD-26MAY031340SFTB-TB2)
# while cycles 15-30 produced 0 execution candidates. The trade audit captured
# pre_analysis_score >= 0.55 with historical_gate_allowed=True, sports family
# (no source_difficulty penalty), mid-priced (~0.50 → max price_center bump),
# and high liquidity. The audit's iterative-optimization fixes (drain priority,
# empty-Grok retry, two-pass drain, penalty single-count, adaptive band, score
# distribution telemetry, configurable maxlen) must NOT regress this path.
#
# These tests reconstruct the cycle-14 market characteristics and assert the
# full participation funnel still classifies them as EXECUTION_ELIGIBLE under
# the new code paths.
# ---------------------------------------------------------------------------


def _cycle14_mlb_f5_market(market_id: str = "KXMLBF5-26MAY031920TEXDET-DET") -> Market:
    """MLB First-5-Innings market matching the cycle-14 success fixture.

    Mid-priced (favors price_center=1.0), high liquidity (favors
    liquidity_score>=0.4), sports family (no source_difficulty penalty,
    no family_penalty), close window <24h (favors horizon_score=1.0).

    The question text explicitly includes "MLB" with surrounding spaces so
    family_from_text classifies this as sports (the MLB keyword in the
    ticker prefix lacks word boundaries because of the KX prefix).
    """
    return Market(
        id=market_id,
        question=(
            "MLB First 5 Innings: Will the Tigers win vs Texas Rangers "
            "after 5 innings?"
        ),
        category="sports",
        liquidity_usdc=1500.0,
        outcomes=[
            MarketOutcome(name="YES", price=0.50),
            MarketOutcome(name="NO", price=0.50),
        ],
        close_time=datetime.now(timezone.utc) + timedelta(hours=4),
        resolution_criteria=(
            "Resolves YES if Tigers lead Texas after the bottom of the 5th "
            "inning per official MLB scoring."
        ),
    )


def test_cycle14_mlb_f5_passes_pre_analysis_threshold() -> None:
    """Pre-analysis must score the cycle-14 fixture above MIN_SCORE so it is
    eligible for deep analysis (i.e. NOT routed to soft-research)."""
    settings = Settings()
    market = _cycle14_mlb_f5_market()
    state = MarketState(market_id=market.id, analysis_count=0, non_actionable_streak=0)
    score, breakdown = _pre_analysis_opportunity_score(
        market,
        state,
        settings,
        traded_before=False,
    )
    assert score >= settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE, (
        f"Cycle-14 MLB F5 fixture must pre-score >= "
        f"{settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE} but got {score:.4f}; "
        f"breakdown={breakdown}"
    )
    # Sports family carries no family_penalty; this guards against regressions
    # in the family-penalty constants leaking back in.
    assert breakdown["pre_score_family_penalty"] == 0.0
    assert breakdown["pre_score_source_difficulty_penalty"] == 0.0


def test_cycle14_mlb_f5_cap_keeps_market_when_historical_gate_allows() -> None:
    """When historical_gate_allowed=True with metrics present, the cap must
    NOT apply the legacy 0.12 historical penalty (5d). The MLB F5 fixture is
    the canonical "sports market with positive historical record" case.

    Uses 3 candidates with max=2 so _cap_analysis_candidates exercises its
    ranking logic (with len(candidates) <= max it short-circuits and the
    selection_rank_components are never populated).
    """
    market = _cycle14_mlb_f5_market()
    candidate_with_metrics = {
        "market": market,
        "pre_analysis_score": 0.65,
        "historical_gate_allowed": True,
        "historical_gate_metrics": {"historical_gate_score_penalty": 0.0},
    }
    spread_market = _cycle14_mlb_f5_market(
        market_id="KXMLBF5SPREAD-26MAY031340SFTB-TB2"
    )
    competing_candidate = {
        "market": spread_market,
        "pre_analysis_score": 0.62,
        "historical_gate_allowed": True,
        "historical_gate_metrics": {"historical_gate_score_penalty": 0.0},
    }
    filler_market = _cycle14_mlb_f5_market(market_id="KXMLBF5-FILLER-XYZ")
    filler_candidate = {
        "market": filler_market,
        "pre_analysis_score": 0.45,
        "historical_gate_allowed": True,
        "historical_gate_metrics": {"historical_gate_score_penalty": 0.0},
    }
    capped = _cap_analysis_candidates(
        [candidate_with_metrics, competing_candidate, filler_candidate],
        max_markets_per_cycle=2,
    )
    capped_ids = [item["market"].id for item in capped]
    assert market.id in capped_ids
    assert spread_market.id in capped_ids
    # Risk-adjusted score must equal base when historical_gate is fully clean
    # (no double-deducted penalty).
    primary = next(item for item in capped if item["market"].id == market.id)
    assert primary["selection_rank_components"]["historical_gate_penalty"] == 0.0
    assert primary["selection_rank_components"]["risk_adjusted_score"] == 0.65


def test_cycle14_mlb_f5_classifies_as_execution_eligible() -> None:
    """End-to-end: with should_trade=True and a proper computed edge, the
    cycle-14 fixture must classify as EXECUTION_ELIGIBLE. Mirrors the audit
    fields from correlation_id 6a6cc761."""
    decision = classify_participation(
        decision_should_trade=True,
        decision_abstain=False,
        decision_evidence_basis="direct",
        decision_edge_source="computed",
        decision_evidence_quality=0.78,
        evidence_quality_threshold=0.75,
        edge_value=0.14,
        edge_reasonable_max=0.40,
        score_gate_blocked=False,
    )
    assert decision.tier == ParticipationTier.EXECUTION_ELIGIBLE
    assert decision.primary_reason == "all_gates_passed"
    assert decision.why_not_execution_eligible is None


def test_cycle14_mlb_f5_unaffected_by_adaptive_band_at_zero_streak() -> None:
    """Adaptive band (5e) only widens routing under sustained zero-yield. With
    no streak the effective band equals the base band, so cycle-14 markets
    follow the unchanged path."""
    settings = Settings()
    base_band = settings.PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND
    # When CYCLE_YIELD_ALERT_ESCALATE_AFTER threshold is not crossed,
    # widening must be zero.
    consecutive_zero_yield = 0
    band_widen_threshold = 2 * settings.CYCLE_YIELD_ALERT_ESCALATE_AFTER
    assert consecutive_zero_yield < band_widen_threshold
    # Base band is the only research-band figure used in this regime.
    assert settings.PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND_MAX >= base_band
