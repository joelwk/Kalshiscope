from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from config import Settings
from main import (
    _adjust_bet_size_for_edge,
    _apply_definitive_outcome_floors,
    _build_execution_audit,
    _direct_evidence_posterior_floor,
    _effective_score_gate_threshold,
    _extract_winning_outcome,
    _filter_markets,
    _is_confidence_override_allowed,
    _is_definitive_outcome_eligible,
    _is_definitive_validated,
    _is_high_quality_settled_evidence,
    _is_uniform_implied_probability,
    _min_evidence_quality_for_market,
    _passes_edge_threshold,
    _should_suppress_hallucinated_edge_penalty,
    _sizing_mode_label,
    _zero_bet_skip_message,
)
from models import Market, MarketOutcome, TradeDecision


def _decision(confidence: float, bet_size_pct: float = 0.5) -> TradeDecision:
    return TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=confidence,
        bet_size_pct=bet_size_pct,
        reasoning="test",
        edge_source="computed",
    )


def test_edge_gate_requires_implied_prob_when_configured() -> None:
    settings = Settings(REQUIRE_IMPLIED_PRICE=True)
    ok, edge, reason = _passes_edge_threshold(None, _decision(0.7), settings)
    assert ok is False
    assert edge is None
    assert "missing implied probability" in reason


def test_edge_gate_blocks_low_edge_for_low_price() -> None:
    settings = Settings(
        MIN_EDGE=0.05,
        LOW_PRICE_THRESHOLD=0.50,
        LOW_PRICE_MIN_EDGE=0.08,
    )
    implied_prob = 0.45
    ok, edge, reason = _passes_edge_threshold(implied_prob, _decision(0.50), settings)
    assert ok is False
    assert edge is not None
    assert "below min" in reason


def test_edge_gate_allows_when_edge_clears_threshold() -> None:
    settings = Settings(
        MIN_EDGE=0.05,
        LOW_PRICE_THRESHOLD=0.50,
        LOW_PRICE_MIN_EDGE=0.08,
    )
    implied_prob = 0.56
    ok, edge, reason = _passes_edge_threshold(implied_prob, _decision(0.62), settings)
    assert ok is True
    assert round(edge, 4) == 0.06
    assert reason == ""


def test_edge_gate_blocks_non_sports_without_direct_evidence() -> None:
    settings = Settings(
        MIN_EDGE=0.10,
        NON_SPORTS_REQUIRES_DIRECT_EVIDENCE=True,
        MAX_REASONABLE_EDGE=0.45,
    )
    market = Market(
        id="KXBTCD-TEST",
        question="Bitcoin threshold",
        category="crypto",
    )
    decision = _decision(0.70).model_copy(update={"evidence_basis": "proxy"})
    ok, edge, reason = _passes_edge_threshold(
        0.50,
        decision,
        settings,
        market=market,
    )
    assert ok is False
    assert edge == pytest.approx(0.20)
    assert reason == "non_sports_needs_direct_evidence"


def test_edge_gate_keeps_sports_path_with_direct_evidence() -> None:
    settings = Settings(
        MIN_EDGE=0.10,
        NON_SPORTS_REQUIRES_DIRECT_EVIDENCE=True,
        MAX_REASONABLE_EDGE=0.45,
    )
    market = Market(
        id="KXSAMPLEGAME-TEST",
        question="Sports: Team A vs Team B winner",
        category="sports",
    )
    decision = _decision(0.78).model_copy(update={"evidence_basis": "direct"})
    ok, edge, reason = _passes_edge_threshold(
        0.58,
        decision,
        settings,
        market=market,
    )
    assert ok is True
    assert round(edge or 0.0, 2) == 0.20
    assert reason == ""


def test_edge_gate_allows_low_price_with_sufficient_edge() -> None:
    """Verifies that underdog outcomes pass when edge exceeds LOW_PRICE_MIN_EDGE."""
    settings = Settings(
        MIN_EDGE=0.05,
        LOW_PRICE_THRESHOLD=0.50,
        LOW_PRICE_MIN_EDGE=0.08,
    )
    implied_prob = 0.45
    ok, edge, reason = _passes_edge_threshold(implied_prob, _decision(0.55), settings)
    assert ok is True
    assert round(edge, 2) == 0.10
    assert reason == ""


def test_mid_price_outcome_uses_standard_edge() -> None:
    """Outcomes above LOW_PRICE_THRESHOLD use MIN_EDGE, not the elevated bar."""
    settings = Settings(
        MIN_EDGE=0.05,
        LOW_PRICE_THRESHOLD=0.50,
        LOW_PRICE_MIN_EDGE=0.08,
    )
    implied_prob = 0.576
    ok, edge, reason = _passes_edge_threshold(implied_prob, _decision(0.67), settings)
    assert ok is True
    assert round(edge, 3) == 0.094
    assert reason == ""


def test_edge_based_sizing_scales_down_for_small_edge() -> None:
    settings = Settings(
        MIN_EDGE=0.05,
        LOW_PRICE_THRESHOLD=0.50,
        LOW_PRICE_MIN_EDGE=0.08,
        EDGE_SCALING_RANGE=0.10,
        LOW_PRICE_BET_PENALTY=0.5,
    )
    decision = _decision(0.66, bet_size_pct=0.6)
    implied_prob = 0.56
    edge = 0.06
    adjusted = _adjust_bet_size_for_edge(decision, implied_prob, edge, settings)
    assert 0 < adjusted < decision.bet_size_pct


def test_edge_based_sizing_caps_fallback_edge_to_min_bet() -> None:
    settings = Settings(
        MIN_BET_USDC=2.0,
        MAX_BET_USDC=8.0,
        MIN_EDGE=0.05,
        FALLBACK_EDGE_MIN_EDGE=0.15,
        EDGE_SCALING_RANGE=0.05,
    )
    decision = _decision(0.78, bet_size_pct=1.0).model_copy(update={"edge_source": "fallback"})
    adjusted = _adjust_bet_size_for_edge(
        decision,
        implied_prob=0.50,
        edge=0.20,
        settings=settings,
    )
    assert adjusted == 0.25


def test_extract_winning_outcome_from_index() -> None:
    market = Market(
        id="m1",
        question="Test market",
        outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
        winningOption=1,
    )
    assert _extract_winning_outcome(market) == "NO"


def test_extract_winning_outcome_ignores_unresolved_sentinel() -> None:
    market = Market(
        id="m2",
        question="Test market",
        outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
        status=0,
        winningOption="18446744073709551615",
    )
    assert _extract_winning_outcome(market) is None


def test_extract_winning_outcome_invalid_index_returns_none() -> None:
    market = Market(
        id="m3",
        question="Test market",
        outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
        winningOption="99",
    )
    assert _extract_winning_outcome(market) is None


def test_filter_markets_excludes_closed_now() -> None:
    now = datetime.now(timezone.utc)
    markets = [
        Market(id="open", question="Open", close_time=now + timedelta(minutes=15)),
        Market(id="closed", question="Closed", close_time=now - timedelta(minutes=1)),
    ]
    filtered = _filter_markets(
        markets,
        min_liquidity=0.0,
        allowlist=(),
        blocklist=(),
    )
    assert [market.id for market in filtered] == ["open"]


def test_filter_markets_excludes_resolved_without_close_time() -> None:
    markets = [
        Market(id="open", question="Open market", close_time=None, status=0),
        Market(
            id="resolved",
            question="Resolved market",
            close_time=None,
            status="resolved",
            outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
        ),
    ]
    stats: dict[str, int] = {}
    filtered = _filter_markets(
        markets,
        min_liquidity=0.0,
        allowlist=(),
        blocklist=(),
        stats=stats,
    )
    assert [market.id for market in filtered] == ["open"]
    assert stats.get("skipped_resolved") == 1


def test_uniform_distribution_guard_for_multi_outcome_market() -> None:
    outcomes = [
        MarketOutcome(name="A"),
        MarketOutcome(name="B"),
        MarketOutcome(name="C"),
        MarketOutcome(name="D"),
    ]
    assert _is_uniform_implied_probability(0.25, outcomes) is True
    assert _is_uniform_implied_probability(0.30, outcomes) is False


def test_sizing_mode_label_for_kelly_and_edge_scaling() -> None:
    assert _sizing_mode_label(True) == "kelly"
    assert _sizing_mode_label(False) == "edge_scaling"


def test_zero_bet_skip_message_is_mode_aware() -> None:
    assert "Kelly" in _zero_bet_skip_message("kelly")
    assert "edge scaling" in _zero_bet_skip_message("edge_scaling")


def test_min_evidence_quality_floor_default_allows_crypto_volume_tuning() -> None:
    settings = Settings()
    generic_market = Market(id="m-eq-floor", question="Will BTC close above threshold?", category="crypto")
    assert _min_evidence_quality_for_market(generic_market, settings) == 0.55


def test_edge_gate_blocks_below_tightened_global_min_edge() -> None:
    settings = Settings(MIN_EDGE=0.07, LOW_PRICE_THRESHOLD=0.50, LOW_PRICE_MIN_EDGE=0.10)
    implied_prob = 0.56
    ok, edge, reason = _passes_edge_threshold(implied_prob, _decision(0.625), settings)
    assert ok is False
    assert round(edge or 0.0, 3) == 0.065
    assert "below min" in reason


def test_edge_gate_blocks_fallback_edge_below_tightened_threshold() -> None:
    settings = Settings(
        MIN_EDGE=0.07,
        FALLBACK_EDGE_MIN_EDGE=0.22,
        LOW_PRICE_THRESHOLD=0.50,
        LOW_PRICE_MIN_EDGE=0.10,
    )
    implied_prob = 0.60
    decision = _decision(0.73).model_copy(update={"edge_source": "fallback"})
    ok, edge, reason = _passes_edge_threshold(implied_prob, decision, settings)
    assert ok is False
    assert round(edge or 0.0, 2) == 0.13
    assert "below min" in reason


def test_confidence_override_requires_floor_even_with_edge_and_evidence() -> None:
    settings = Settings(
        CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED=True,
        CONFIDENCE_GATE_MIN_EDGE=0.10,
        CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY=0.70,
        CONFIDENCE_GATE_OVERRIDE_MIN_CONFIDENCE=0.50,
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.35,
        bet_size_pct=0.3,
        reasoning="test",
        evidence_quality=0.9,
    )
    allowed, min_confidence, override_path = _is_confidence_override_allowed(
        settings=settings,
        decision=decision,
        override_edge=0.20,
    )
    assert min_confidence == 0.50
    assert allowed is False
    assert override_path == "none"


def test_confidence_override_allows_when_floor_and_thresholds_met() -> None:
    settings = Settings(
        CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED=True,
        CONFIDENCE_GATE_MIN_EDGE=0.10,
        CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY=0.70,
        CONFIDENCE_GATE_OVERRIDE_MIN_CONFIDENCE=0.50,
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.52,
        bet_size_pct=0.3,
        reasoning="test",
        evidence_quality=0.9,
    )
    allowed, _, override_path = _is_confidence_override_allowed(
        settings=settings,
        decision=decision,
        override_edge=0.20,
    )
    assert allowed is True
    assert override_path == "edge_default"


def test_confidence_override_uses_pre_calibration_confidence() -> None:
    settings = Settings(
        CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED=True,
        CONFIDENCE_GATE_MIN_EDGE=0.08,
        CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY=0.65,
        CONFIDENCE_GATE_OVERRIDE_MIN_CONFIDENCE=0.55,
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.48,
        bet_size_pct=0.3,
        reasoning="test",
        evidence_quality=0.70,
    )
    blocked, _, blocked_path = _is_confidence_override_allowed(
        settings=settings,
        decision=decision,
        override_edge=0.20,
    )
    assert blocked is False
    assert blocked_path == "none"

    allowed, _, override_path = _is_confidence_override_allowed(
        settings=settings,
        decision=decision,
        override_edge=0.20,
        pre_calibration_confidence=0.58,
    )
    assert allowed is True
    assert override_path == "edge_default"


def test_confidence_override_pre_calibration_still_enforces_evidence_quality() -> None:
    settings = Settings(
        CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED=True,
        CONFIDENCE_GATE_MIN_EDGE=0.08,
        CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY=0.65,
        CONFIDENCE_GATE_OVERRIDE_MIN_CONFIDENCE=0.55,
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.48,
        bet_size_pct=0.3,
        reasoning="test",
        evidence_quality=0.60,
    )
    allowed, _, override_path = _is_confidence_override_allowed(
        settings=settings,
        decision=decision,
        override_edge=0.20,
        pre_calibration_confidence=0.62,
    )
    assert allowed is False
    assert override_path == "none"


def test_confidence_override_pre_calibration_still_enforces_edge() -> None:
    settings = Settings(
        CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED=True,
        CONFIDENCE_GATE_MIN_EDGE=0.08,
        CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY=0.65,
        CONFIDENCE_GATE_OVERRIDE_MIN_CONFIDENCE=0.55,
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.48,
        bet_size_pct=0.3,
        reasoning="test",
        evidence_quality=0.90,
    )
    allowed, _, override_path = _is_confidence_override_allowed(
        settings=settings,
        decision=decision,
        override_edge=0.05,
        pre_calibration_confidence=0.62,
    )
    assert allowed is False
    assert override_path == "none"


def _direct_decision(
    *,
    confidence: float = 0.60,
    edge_external: float | None = 0.13,
    evidence_quality: float = 0.85,
    evidence_basis: str = "direct",
    edge_source: str = "computed",
) -> TradeDecision:
    return TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=confidence,
        bet_size_pct=0.5,
        reasoning="direct settlement-aligned read",
        edge_source=edge_source,
        edge_external=edge_external,
        evidence_quality=evidence_quality,
        evidence_basis=evidence_basis,
    )


def _floor_settings(**overrides) -> Settings:
    base = dict(
        DIRECT_POSTERIOR_FLOOR_ENABLED=True,
        DIRECT_POSTERIOR_FLOOR_MIN_EVIDENCE_QUALITY=0.80,
        MAX_GLOBAL_CONFIDENCE_DIRECT=0.89,
        MIN_EDGE=0.08,
        MAX_REASONABLE_EDGE=0.40,
    )
    base.update(overrides)
    return Settings(**base)


def test_direct_posterior_floor_reconstructs_model_estimate() -> None:
    settings = _floor_settings()
    floor = _direct_evidence_posterior_floor(_direct_decision(), 0.67, settings)
    assert floor == pytest.approx(0.80)


def test_direct_posterior_floor_capped_at_direct_ceiling() -> None:
    settings = _floor_settings()
    # implied 0.66 + edge 0.30 = 0.96 model estimate, capped at 0.89.
    floor = _direct_evidence_posterior_floor(
        _direct_decision(edge_external=0.30), 0.66, settings
    )
    assert floor == pytest.approx(0.89)


def test_direct_posterior_floor_none_for_proxy_or_low_eq_or_nonpositive_edge() -> None:
    settings = _floor_settings()
    assert _direct_evidence_posterior_floor(
        _direct_decision(evidence_basis="proxy"), 0.67, settings
    ) is None
    assert _direct_evidence_posterior_floor(
        _direct_decision(evidence_quality=0.70), 0.67, settings
    ) is None
    assert _direct_evidence_posterior_floor(
        _direct_decision(edge_external=0.0), 0.67, settings
    ) is None
    assert _direct_evidence_posterior_floor(
        _direct_decision(edge_source="fallback"), 0.67, settings
    ) is None
    assert _direct_evidence_posterior_floor(_direct_decision(), None, settings) is None


def test_direct_posterior_floor_disabled_returns_none() -> None:
    settings = _floor_settings(DIRECT_POSTERIOR_FLOOR_ENABLED=False)
    assert _direct_evidence_posterior_floor(_direct_decision(), 0.67, settings) is None


def test_direct_posterior_floor_unblocks_edge_gate_after_calibration_inversion() -> None:
    settings = _floor_settings()
    market = Market(
        id="KXAAA-FLOOR",
        question="Direct evidence generic market",
        outcomes=[MarketOutcome(name="YES", price=0.67), MarketOutcome(name="NO", price=0.33)],
        liquidity_usdc=600.0,
        close_time=datetime.now(timezone.utc) + timedelta(days=1),
    )
    # Calibration crushed confidence to 0.60; raw model edge is +0.13.
    decision = _direct_decision(confidence=0.60, edge_external=0.13)
    floor = _direct_evidence_posterior_floor(decision, 0.67, settings, market=market)
    assert floor == pytest.approx(0.80)

    # Without the floor the calibrated confidence yields a negative market edge.
    blocked_ok, blocked_edge, _ = _passes_edge_threshold(
        0.67, decision, settings, market=market,
        effective_confidence_override=decision.confidence,
    )
    assert blocked_ok is False
    assert blocked_edge is not None and blocked_edge < 0

    # With the floor applied the real positive edge clears the gate.
    ok, edge, reason = _passes_edge_threshold(
        0.67, decision, settings, market=market,
        effective_confidence_override=max(decision.confidence, floor),
    )
    assert ok is True
    assert edge == pytest.approx(0.13)
    assert reason == ""


def test_effective_score_gate_threshold_uses_weather_direct_threshold() -> None:
    settings = Settings(
        SCORE_GATE_THRESHOLD=0.25,
        SCORE_GATE_THRESHOLD_WEATHER_DIRECT=0.10,
    )
    weather_market = Market(
        id="KXHIGHCHI-26APR10-T50",
        question="Will Chicago high be below 50F?",
        category="weather",
    )
    threshold = _effective_score_gate_threshold(
        settings=settings,
        market=weather_market,
        evidence_basis_class="direct",
    )
    assert threshold == 0.10


def test_effective_score_gate_threshold_defaults_for_non_direct_or_non_weather() -> None:
    settings = Settings(
        SCORE_GATE_THRESHOLD=0.38,
        SCORE_GATE_THRESHOLD_WEATHER_DIRECT=0.10,
        SCORE_GATE_THRESHOLD_DIRECT_HIGH_QUALITY=0.25,
    )
    weather_market = Market(
        id="KXHIGHCHI-26APR10-T50",
        question="Will Chicago high be below 50F?",
        category="weather",
    )
    crypto_market = Market(
        id="KXBTCD-26APR1001-T71999.99",
        question="Bitcoin price on Apr 10, 2026?",
        category="crypto",
    )
    weather_proxy_threshold = _effective_score_gate_threshold(
        settings=settings,
        market=weather_market,
        evidence_basis_class="proxy",
    )
    crypto_direct_threshold = _effective_score_gate_threshold(
        settings=settings,
        market=crypto_market,
        evidence_basis_class="direct",
    )
    assert weather_proxy_threshold == 0.38
    assert crypto_direct_threshold == 0.38


def test_effective_score_gate_threshold_uses_direct_high_quality_override() -> None:
    settings = Settings(
        SCORE_GATE_THRESHOLD=0.38,
        SCORE_GATE_THRESHOLD_WEATHER_DIRECT=0.10,
        SCORE_GATE_THRESHOLD_DIRECT_HIGH_QUALITY=0.25,
    )
    crypto_market = Market(
        id="KXBTCD-26APR1001-T71999.99",
        question="Bitcoin price on Apr 10, 2026?",
        category="crypto",
    )
    threshold = _effective_score_gate_threshold(
        settings=settings,
        market=crypto_market,
        evidence_basis_class="direct",
        evidence_quality=0.85,
    )
    assert threshold == 0.25


def test_effective_score_gate_threshold_profitable_family_convergent_bypass() -> None:
    settings = Settings(
        SCORE_GATE_THRESHOLD=0.52,
        SCORE_GATE_PROFITABLE_FAMILY_CONVERGENT_ENABLED=True,
        SCORE_GATE_THRESHOLD_PROFITABLE_FAMILY_CONVERGENT=0.08,
        SCORE_GATE_PROFITABLE_FAMILY_CONVERGENT_MIN_SAMPLES=30,
    )
    sports_market = Market(
        id="KXMLB-26MAY23-TOR",
        question="Will Toronto win?",
        category="sports",
    )
    threshold = _effective_score_gate_threshold(
        settings=settings,
        market=sports_market,
        evidence_basis_class="proxy",
        evidence_quality=0.50,
        family_is_profitable=True,
        self_consistency_passed=True,
        family_sample_size=40,
    )
    assert threshold == 0.08

    thin_history = _effective_score_gate_threshold(
        settings=settings,
        market=sports_market,
        evidence_basis_class="proxy",
        evidence_quality=0.50,
        family_is_profitable=True,
        self_consistency_passed=True,
        family_sample_size=10,
    )
    assert thin_history == 0.52

    disabled = _effective_score_gate_threshold(
        settings=Settings(
            SCORE_GATE_THRESHOLD=0.52,
            SCORE_GATE_PROFITABLE_FAMILY_CONVERGENT_ENABLED=False,
        ),
        market=sports_market,
        evidence_basis_class="proxy",
        evidence_quality=0.50,
        family_is_profitable=True,
        self_consistency_passed=True,
        family_sample_size=40,
    )
    assert disabled == 0.52


def test_research_queue_receipt_emitted_on_soft_block() -> None:
    payload = _build_execution_audit(
        decision_terminal=False,
        final_action="research_queued",
        final_reason="edge_gate_blocked",
        gate_edge_required=0.14,
        gate_edge_actual=0.11,
        edge_shortfall=0.03,
    )
    assert payload["final_action"] == "research_queued"
    assert payload["final_reason"] == "edge_gate_blocked"
    assert payload["skip_reasons"] == ["edge_gate_blocked"]
    assert payload["rejection_reason"] == "edge_gate_blocked"


def test_research_queue_receipt_emitted_for_edge_above_reasonable_max() -> None:
    payload = _build_execution_audit(
        decision_terminal=False,
        final_action="research_queued",
        final_reason="edge_above_reasonable_max",
        gate_edge_required=0.14,
        gate_edge_actual=0.36,
        edge_shortfall=0.0,
        gate_edge_reason="edge_above_reasonable_max",
    )
    assert payload["final_action"] == "research_queued"
    assert payload["final_reason"] == "edge_above_reasonable_max"
    assert payload["skip_reasons"] == ["edge_above_reasonable_max"]
    assert payload["gate_edge_reason"] == "edge_above_reasonable_max"


def test_research_queue_receipt_emitted_for_hallucinated_edge() -> None:
    payload = _build_execution_audit(
        decision_terminal=False,
        final_action="research_queued",
        final_reason="hallucinated_edge",
        score_threshold=0.48,
        score_gap=0.07,
    )
    assert payload["final_action"] == "research_queued"
    assert payload["final_reason"] == "hallucinated_edge"
    assert payload["skip_reasons"] == ["hallucinated_edge"]


def test_research_queue_receipt_emitted_for_extreme_market_edge() -> None:
    payload = _build_execution_audit(
        decision_terminal=False,
        final_action="research_queued",
        final_reason="extreme_market_edge",
        score_threshold=0.48,
        score_gap=0.11,
    )
    assert payload["final_action"] == "research_queued"
    assert payload["final_reason"] == "extreme_market_edge"
    assert payload["skip_reasons"] == ["extreme_market_edge"]


def test_execution_audit_preserves_gated_should_trade_field() -> None:
    payload = _build_execution_audit(
        decision_terminal=True,
        final_action="skip",
        final_reason="score_gate_blocked",
        should_trade=False,
        gated_should_trade=True,
        ticker_prefix_short="KXETH",
        ticker_prefix_short_pnl=-9.1,
    )
    assert payload["should_trade"] is False
    assert payload["gated_should_trade"] is True
    assert payload["ticker_prefix_short"] == "KXETH"
    assert payload["ticker_prefix_short_pnl"] == -9.1


def test_execution_audit_includes_ranking_and_research_only_fields() -> None:
    payload = _build_execution_audit(
        decision_terminal=False,
        final_action="research_queued",
        final_reason="repeated_non_actionable_research_only",
        ranking_rank=2,
        ranking_total_candidates=8,
        historical_family_pnl_total=14.5,
        historical_family_samples=12,
        primary_source_url_present=False,
        fallback_high_confidence_penalty_applied=True,
        research_only=True,
    )
    assert payload["final_action"] == "research_queued"
    assert payload["ranking_rank"] == 2
    assert payload["ranking_total_candidates"] == 8
    assert payload["historical_family_pnl_total"] == 14.5
    assert payload["historical_family_samples"] == 12
    assert payload["fallback_high_confidence_penalty_applied"] is True
    assert payload["research_only"] is True


_WHITELIST_WITH_AP = (
    "weather.gov", "noaa.gov", "wsj.com", "bloomberg.com",
    "reuters.com", "coindesk.com", "kalshi.com", "apnews.com",
)


def test_definitive_outcome_bypasses_evidence_quality_gate() -> None:
    """When definitive_outcome_detected is True with a whitelisted URL and
    direct basis, _min_evidence_quality_for_market should return the lower
    definitive floor instead of the normal threshold."""
    market = Market(
        id="KXSAMPLEGAME-26APR251915TEAMATEAMB-TEAMB",
        question="Will Team B win?",
        category="sports",
        outcomes=[
            MarketOutcome(name="YES", price=0.45),
            MarketOutcome(name="NO", price=0.55),
        ],
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.95,
        bet_size_pct=0.5,
        reasoning="Game over per Reuters source",
        evidence_basis="direct",
        evidence_quality=0.60,
        primary_source_url="https://reuters.com/article/test",
        definitive_outcome_detected=True,
    )
    settings = Settings(
        DEFINITIVE_OUTCOME_EVIDENCE_QUALITY_FLOOR=0.78,
        MIN_EVIDENCE_QUALITY_FOR_TRADE=0.75,
    )
    threshold = _min_evidence_quality_for_market(market, settings, decision)
    assert threshold <= 0.78


def test_definitive_outcome_edge_cap_raised() -> None:
    """Edge cap should be raised for validated-definitive outcomes."""
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.95,
        bet_size_pct=0.5,
        reasoning="Game resolved",
        evidence_basis="direct",
        evidence_quality=0.85,
        raw_evidence_quality=0.85,
        primary_source_url="https://reuters.com/article/test",
        definitive_outcome_detected=True,
        source_match_class="settlement_aligned",
        my_prob=0.95,
    )
    market = Market(
        id="KXSAMPLEPERIOD-26APR251415TEAMATEAMB-TEAMA",
        question="Will Team A win the first period?",
        category="sports",
        outcomes=[
            MarketOutcome(name="YES", price=0.45),
            MarketOutcome(name="NO", price=0.55),
        ],
    )
    settings = Settings(
        DEFINITIVE_OUTCOME_EDGE_REASONABLE_MAX=0.65,
        MAX_REASONABLE_EDGE=0.35,
    )
    passed, edge_val, reason = _passes_edge_threshold(
        0.45, decision, settings, market=market
    )
    assert passed is True
    assert edge_val is not None
    assert abs(edge_val - 0.50) < 0.01


def test_non_definitive_edge_049_still_blocked() -> None:
    """Without definitive_outcome_detected, edge=0.49 is still blocked."""
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.94,
        bet_size_pct=0.5,
        reasoning="Test",
        evidence_basis="direct",
        evidence_quality=0.85,
    )
    market = Market(
        id="KXTEST-001",
        question="Test?",
        category="generic",
        outcomes=[
            MarketOutcome(name="YES", price=0.45),
            MarketOutcome(name="NO", price=0.55),
        ],
    )
    settings = Settings(
        DEFINITIVE_OUTCOME_EDGE_REASONABLE_MAX=0.65,
        MAX_REASONABLE_EDGE=0.35,
    )
    passed, edge_val, reason = _passes_edge_threshold(
        0.45, decision, settings, market=market
    )
    assert passed is False
    assert reason == "edge_above_reasonable_max"


# --- High-quality settlement-aligned evidence path tests ---
#
# Cycle 1 review found the bot's only high-conviction trade
# (KXPUREALBUMS-KEH26APR30-39K) blocked by edge_above_reasonable_max even
# though the analysis had evidence_basis=direct, source_match_class=
# settlement_aligned, evidence_quality=1.0, and a whitelisted primary
# source URL. ``_is_high_quality_settled_evidence`` recognizes this case
# and unlocks the higher DEFINITIVE_OUTCOME_EDGE_REASONABLE_MAX cap so the
# trade can be evaluated by the score gate instead of hard-blocked.


def _kehlani_style_decision(
    confidence: float = 0.80,
    evidence_quality: float = 1.0,
    primary_source_url: str = "https://www.hitsdailydouble.com/charts/hits-top-50",
    source_match_class: str = "settlement_aligned",
    evidence_basis: str = "direct",
    definitive_outcome_detected: bool = False,
) -> TradeDecision:
    return TradeDecision(
        should_trade=True,
        outcome="NO",
        confidence=confidence,
        bet_size_pct=0.5,
        reasoning="Hits Daily Double chart shows 23,291 sales vs 39K threshold",
        evidence_basis=evidence_basis,
        evidence_quality=evidence_quality,
        edge_source="computed",
        primary_source_url=primary_source_url,
        source_match_class=source_match_class,
        definitive_outcome_detected=definitive_outcome_detected,
    )


def _kehlani_settings(**overrides) -> Settings:
    defaults = dict(
        MAX_REASONABLE_EDGE=0.40,
        DEFINITIVE_OUTCOME_EDGE_REASONABLE_MAX=0.50,
        HIGH_QUALITY_SETTLED_EVIDENCE_MIN_EQ=0.95,
        DIRECT_SOURCE_WHITELIST=("hitsdailydouble.com", "billboard.com"),
        NON_SPORTS_REQUIRES_DIRECT_EVIDENCE=False,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def test_high_quality_settled_evidence_recognizes_kehlani_canary() -> None:
    settings = _kehlani_settings()
    decision = _kehlani_style_decision()
    assert _is_high_quality_settled_evidence(decision, settings) is True
    assert _is_definitive_validated(decision, settings) is True


def test_high_quality_settled_unlocks_higher_edge_cap() -> None:
    """Cycle 1 canary: edge=0.50 (90 conf - 40 implied) clears the new
    ``DEFINITIVE_OUTCOME_EDGE_REASONABLE_MAX`` cap of 0.50 instead of
    being hard-blocked at the legacy 0.32 cap."""
    settings = _kehlani_settings()
    decision = _kehlani_style_decision(confidence=0.90)
    market = Market(
        id="KXPUREALBUMS-KEH26APR30-39K",
        question="Will Kehlani have above 39000 Pure Album Sales?",
        category="music",
        outcomes=[
            MarketOutcome(name="YES", price=0.66),
            MarketOutcome(name="NO", price=0.34),
        ],
    )
    passed, edge_val, reason = _passes_edge_threshold(
        0.40, decision, settings, market=market
    )
    assert passed is True, f"reason={reason}"
    assert edge_val is not None
    assert abs(edge_val - 0.50) < 0.001
    assert reason == ""


def test_high_quality_settled_requires_eq_floor() -> None:
    """eq below the configured floor (default 0.95) does not qualify."""
    settings = _kehlani_settings()
    decision = _kehlani_style_decision(evidence_quality=0.94)
    assert _is_high_quality_settled_evidence(decision, settings) is False
    assert _is_definitive_validated(decision, settings) is False


def test_high_quality_settled_requires_settlement_aligned() -> None:
    """Direct + whitelisted + eq=1.0 but source_match_class=verifiable_unmatched
    is not enough — settlement_aligned signal is required."""
    settings = _kehlani_settings()
    decision = _kehlani_style_decision(source_match_class="verifiable_unmatched")
    assert _is_high_quality_settled_evidence(decision, settings) is False


def test_high_quality_settled_requires_whitelisted_source() -> None:
    """Direct + settlement_aligned + eq=1.0 but a non-whitelisted source
    does not unlock the higher cap."""
    settings = _kehlani_settings(
        DIRECT_SOURCE_WHITELIST=("billboard.com",),
    )
    decision = _kehlani_style_decision(
        primary_source_url="https://randomblog.example.com/charts"
    )
    assert _is_high_quality_settled_evidence(decision, settings) is False


def test_high_quality_settled_requires_direct_basis() -> None:
    """Proxy basis cannot trigger the high-quality settled exemption."""
    settings = _kehlani_settings()
    decision = _kehlani_style_decision(evidence_basis="proxy")
    assert _is_high_quality_settled_evidence(decision, settings) is False


def test_high_quality_settled_suppresses_hallucinated_edge_penalty() -> None:
    """The hallucinated_edge penalty should be suppressed for the new path
    so the score gate sees the unmodified score."""
    settings = _kehlani_settings()
    decision = _kehlani_style_decision()
    assert (
        _should_suppress_hallucinated_edge_penalty(
            decision=decision,
            evidence_basis="direct",
            settings=settings,
        )
        is True
    )


def test_legacy_definitive_validated_path_unchanged() -> None:
    """A definitive_outcome_detected=True trade with eq>=0.80 still
    qualifies via the legacy strict path even when eq < the new
    HIGH_QUALITY_SETTLED_EVIDENCE_MIN_EQ floor."""
    settings = _kehlani_settings(HIGH_QUALITY_SETTLED_EVIDENCE_MIN_EQ=0.99)
    decision = _kehlani_style_decision(
        evidence_quality=0.85,
        definitive_outcome_detected=True,
    ).model_copy(
        update={
            "my_prob": 0.99,
            "raw_evidence_quality": 0.85,
        }
    )
    assert _is_definitive_validated(decision, settings) is True
    assert _is_high_quality_settled_evidence(decision, settings) is False


def test_definitive_outcome_requires_structured_near_binary_my_prob() -> None:
    settings = Settings(
        DIRECT_SOURCE_WHITELIST=("espn.com",),
        DEFINITIVE_OUTCOME_EVIDENCE_QUALITY_FLOOR=0.80,
    )
    market = Market(
        id="KXMLBHRR-SAMPLE",
        question="Brandon Nimmo: 2+ hits + runs + RBIs?",
        category="sports",
        outcomes=[
            MarketOutcome(name="YES", price=0.45),
            MarketOutcome(name="NO", price=0.55),
        ],
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.80,
        bet_size_pct=0.5,
        reasoning="ESPN box score says Brandon Nimmo cleared the prop.",
        evidence_basis="direct",
        evidence_quality=0.80,
        raw_evidence_quality=0.95,
        primary_source_url="https://www.espn.com/mlb/boxscore/_/gameId/1",
        definitive_outcome_detected=True,
        source_match_class="settlement_aligned",
        edge_source="none",
    )

    updated, applied = _apply_definitive_outcome_floors(decision, market, settings)
    assert updated is decision
    assert applied is False
    assert _is_definitive_outcome_eligible(decision, settings, market=market) is False
    ok, edge, reason = _passes_edge_threshold(0.45, decision, settings, market=market)
    assert ok is False
    assert edge == pytest.approx(0.35)
    assert reason == "missing_structured_probability"


def test_definitive_outcome_rejects_mismatched_sports_entity() -> None:
    settings = Settings(
        DIRECT_SOURCE_WHITELIST=("espn.com",),
        DEFINITIVE_OUTCOME_EVIDENCE_QUALITY_FLOOR=0.80,
        DEFINITIVE_OUTCOME_EDGE_REASONABLE_MAX=0.60,
        MAX_REASONABLE_EDGE=0.35,
    )
    market = Market(
        id="KXMLBHRR-SAMPLE",
        question="Brandon Nimmo: 2+ hits + runs + RBIs?",
        category="sports",
        outcomes=[
            MarketOutcome(name="YES", price=0.45),
            MarketOutcome(name="NO", price=0.55),
        ],
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.95,
        bet_size_pct=0.5,
        reasoning="ESPN box score says Adley Rutschman cleared the prop.",
        evidence_basis="direct",
        evidence_quality=0.95,
        raw_evidence_quality=0.95,
        primary_source_url="https://www.espn.com/mlb/boxscore/_/gameId/1",
        definitive_outcome_detected=True,
        source_match_class="settlement_aligned",
        edge_source="computed",
        my_prob=0.99,
    )

    assert _is_definitive_validated(decision, settings, market=market) is False
    ok, edge, reason = _passes_edge_threshold(0.45, decision, settings, market=market)
    assert ok is False
    assert edge == pytest.approx(0.50)
    assert reason == "edge_above_reasonable_max"


def test_definitive_outcome_accepts_matching_sports_entity_with_structured_probability() -> None:
    settings = Settings(
        DIRECT_SOURCE_WHITELIST=("espn.com",),
        DEFINITIVE_OUTCOME_EVIDENCE_QUALITY_FLOOR=0.80,
        DEFINITIVE_OUTCOME_EDGE_REASONABLE_MAX=0.60,
        MAX_REASONABLE_EDGE=0.35,
    )
    market = Market(
        id="KXMLBHRR-SAMPLE",
        question="Brandon Nimmo: 2+ hits + runs + RBIs?",
        category="sports",
        outcomes=[
            MarketOutcome(name="YES", price=0.45),
            MarketOutcome(name="NO", price=0.55),
        ],
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.95,
        bet_size_pct=0.5,
        reasoning="ESPN box score confirms Brandon Nimmo cleared the prop.",
        evidence_basis="direct",
        evidence_quality=0.95,
        raw_evidence_quality=0.95,
        primary_source_url="https://www.espn.com/mlb/boxscore/_/gameId/1",
        definitive_outcome_detected=True,
        source_match_class="settlement_aligned",
        edge_source="computed",
        my_prob=0.99,
    )

    assert _is_definitive_validated(decision, settings, market=market) is True
    ok, edge, reason = _passes_edge_threshold(0.45, decision, settings, market=market)
    assert ok is True
    assert edge == pytest.approx(0.50)
    assert reason == ""


def test_high_quality_settled_caps_extreme_edge_at_095() -> None:
    """Even high-quality settled evidence cannot bypass the 0.95 hard limit."""
    settings = _kehlani_settings()
    decision = _kehlani_style_decision(confidence=0.99)
    market = Market(
        id="KXPUREALBUMS-CANARY",
        question="Test?",
        category="music",
        outcomes=[
            MarketOutcome(name="YES", price=0.97),
            MarketOutcome(name="NO", price=0.03),
        ],
    )
    passed, _edge_val, reason = _passes_edge_threshold(
        0.02, decision, settings, market=market
    )
    assert passed is False
    assert reason == "edge_above_reasonable_max"


def test_strong_direct_evidence_override_unlocks_high_eq_low_conf_trade() -> None:
    """KXHIGHMIA-style: whitelisted source, basis=direct, eq=1.0, edge=0.15,
    conf=0.58 -> override_path=strong_direct_evidence, allowed=True."""
    settings = Settings(
        CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED=True,
        CONFIDENCE_GATE_MIN_EDGE=0.10,
        CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY=0.70,
        CONFIDENCE_GATE_OVERRIDE_MIN_CONFIDENCE=0.62,
        STRONG_EVIDENCE_CONFIDENCE_FLOOR=0.55,
        STRONG_EVIDENCE_MIN_EVIDENCE_QUALITY=0.85,
        DIRECT_SOURCE_WHITELIST=("weather.gov", "espn.com"),
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="NO",
        confidence=0.58,
        bet_size_pct=0.2,
        reasoning="NWS forecast evidence",
        evidence_basis="direct",
        evidence_quality=1.0,
        edge_source="computed",
        edge_external=0.15,
        primary_source_url="https://forecast.weather.gov/product.php?site=MFL",
    )
    allowed, floor, override_path = _is_confidence_override_allowed(
        settings=settings,
        decision=decision,
        override_edge=0.15,
    )
    assert allowed is True
    assert override_path == "strong_direct_evidence"
    assert floor == 0.55


def test_strong_direct_evidence_override_rejects_low_eq() -> None:
    """Same as above but evidence_quality=0.70 < 0.85 threshold -> blocked."""
    settings = Settings(
        CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED=True,
        CONFIDENCE_GATE_MIN_EDGE=0.10,
        CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY=0.70,
        CONFIDENCE_GATE_OVERRIDE_MIN_CONFIDENCE=0.62,
        STRONG_EVIDENCE_CONFIDENCE_FLOOR=0.55,
        STRONG_EVIDENCE_MIN_EVIDENCE_QUALITY=0.85,
        DIRECT_SOURCE_WHITELIST=("weather.gov",),
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="NO",
        confidence=0.58,
        bet_size_pct=0.2,
        reasoning="Weather forecast",
        evidence_basis="direct",
        evidence_quality=0.70,
        edge_source="computed",
        edge_external=0.15,
        primary_source_url="https://forecast.weather.gov/product.php?site=MFL",
    )
    allowed, _, override_path = _is_confidence_override_allowed(
        settings=settings,
        decision=decision,
        override_edge=0.15,
    )
    assert allowed is False
    assert override_path == "none"


def test_strong_direct_evidence_override_rejects_non_whitelisted_source() -> None:
    """Non-whitelisted source URL -> strong override path doesn't trigger."""
    settings = Settings(
        CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED=True,
        CONFIDENCE_GATE_MIN_EDGE=0.10,
        CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY=0.70,
        CONFIDENCE_GATE_OVERRIDE_MIN_CONFIDENCE=0.62,
        STRONG_EVIDENCE_CONFIDENCE_FLOOR=0.55,
        STRONG_EVIDENCE_MIN_EVIDENCE_QUALITY=0.85,
        DIRECT_SOURCE_WHITELIST=("weather.gov",),
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="NO",
        confidence=0.58,
        bet_size_pct=0.2,
        reasoning="Blog source",
        evidence_basis="direct",
        evidence_quality=0.95,
        edge_source="computed",
        edge_external=0.15,
        primary_source_url="https://randomblog.example.com/forecast",
    )
    allowed, _, override_path = _is_confidence_override_allowed(
        settings=settings,
        decision=decision,
        override_edge=0.15,
    )
    assert allowed is False
    assert override_path == "none"


# --- Strong proxy evidence override tests ---


def _proxy_override_settings(**overrides) -> Settings:
    defaults = dict(
        CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED=True,
        CONFIDENCE_GATE_MIN_EDGE=0.10,
        CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY=0.70,
        CONFIDENCE_GATE_OVERRIDE_MIN_CONFIDENCE=0.62,
        STRONG_EVIDENCE_CONFIDENCE_FLOOR=0.55,
        STRONG_EVIDENCE_MIN_EVIDENCE_QUALITY=0.85,
        STRONG_EVIDENCE_PROXY_MIN_EVIDENCE_QUALITY=0.95,
        STRONG_EVIDENCE_PROXY_MIN_EDGE=0.20,
        DIRECT_SOURCE_WHITELIST=("wsj.com", "weather.gov", "espn.com"),
    )
    defaults.update(overrides)
    return Settings(**defaults)


def test_strong_proxy_evidence_override_allows_kxgoldw_canary() -> None:
    """KXGOLDW canary: whitelisted WSJ source, basis=proxy, eq=1.0,
    edge=0.24, conf=0.55 -> strong_proxy_evidence, allowed=True."""
    settings = _proxy_override_settings()
    decision = TradeDecision(
        should_trade=True,
        outcome="NO",
        confidence=0.55,
        bet_size_pct=0.38,
        reasoning="Gold futures proxy evidence from WSJ",
        evidence_basis="proxy",
        evidence_quality=1.0,
        edge_source="computed",
        edge_external=0.24,
        primary_source_url="https://www.wsj.com/market-data/quotes/futures/GCK26",
    )
    allowed, floor, override_path = _is_confidence_override_allowed(
        settings=settings,
        decision=decision,
        override_edge=0.24,
    )
    assert allowed is True
    assert override_path == "strong_proxy_evidence"
    assert floor == 0.55


def test_strong_proxy_evidence_rejects_low_eq() -> None:
    """eq=0.85 passes the direct floor (0.85) but fails the proxy floor
    (0.95) -> blocked."""
    settings = _proxy_override_settings()
    decision = TradeDecision(
        should_trade=True,
        outcome="NO",
        confidence=0.55,
        bet_size_pct=0.30,
        reasoning="Proxy evidence with moderate quality",
        evidence_basis="proxy",
        evidence_quality=0.85,
        edge_source="computed",
        edge_external=0.24,
        primary_source_url="https://www.wsj.com/market-data/quotes/futures/GCK26",
    )
    allowed, _, override_path = _is_confidence_override_allowed(
        settings=settings,
        decision=decision,
        override_edge=0.24,
    )
    assert allowed is False
    assert override_path == "none"


def test_strong_proxy_evidence_rejects_low_edge() -> None:
    """edge=0.15 passes the default threshold (0.10) but fails the proxy
    threshold (0.20) -> blocked."""
    settings = _proxy_override_settings()
    decision = TradeDecision(
        should_trade=True,
        outcome="NO",
        confidence=0.55,
        bet_size_pct=0.25,
        reasoning="Proxy evidence with small edge",
        evidence_basis="proxy",
        evidence_quality=0.95,
        edge_source="computed",
        edge_external=0.15,
        primary_source_url="https://www.wsj.com/market-data/quotes/futures/GCK26",
    )
    allowed, _, override_path = _is_confidence_override_allowed(
        settings=settings,
        decision=decision,
        override_edge=0.15,
    )
    assert allowed is False
    assert override_path == "none"


def test_strong_proxy_evidence_rejects_non_whitelisted_source() -> None:
    """Non-whitelisted source URL -> proxy override doesn't trigger even
    when eq and edge are strong."""
    settings = _proxy_override_settings()
    decision = TradeDecision(
        should_trade=True,
        outcome="NO",
        confidence=0.55,
        bet_size_pct=0.30,
        reasoning="Proxy evidence from untrusted source",
        evidence_basis="proxy",
        evidence_quality=1.0,
        edge_source="computed",
        edge_external=0.30,
        primary_source_url="https://randomblog.example.com/futures/gold",
    )
    allowed, _, override_path = _is_confidence_override_allowed(
        settings=settings,
        decision=decision,
        override_edge=0.30,
    )
    assert allowed is False
    assert override_path == "none"


# --- Cycle 4 recovery: confidence override floor below MIN_CONFIDENCE ---
#
# Cycle 4 receipts showed should_trade=True candidates blocked at
# `confidence_below_min` because runtime CONFIDENCE_GATE_OVERRIDE_MIN_CONFIDENCE
# was set equal to MIN_CONFIDENCE, defeating the override path. These cases
# pin the intended behavior: when the override floor is below the hard gate
# AND the edge/evidence-quality thresholds are met, the candidate must be
# allowed past the confidence gate.


def test_confidence_override_below_min_confidence_allows_strong_edge_candidate() -> None:
    """Override floor 0.55 < MIN_CONFIDENCE 0.62: a 0.58-confidence
    candidate with strong edge and evidence must clear via edge_default."""
    settings = Settings(
        MIN_CONFIDENCE=0.62,
        CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED=True,
        CONFIDENCE_GATE_MIN_EDGE=0.10,
        CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY=0.70,
        CONFIDENCE_GATE_OVERRIDE_MIN_CONFIDENCE=0.55,
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.58,
        bet_size_pct=0.3,
        reasoning="High-quality edge with sub-MIN_CONFIDENCE rating",
        evidence_quality=0.80,
        edge_source="computed",
        edge_external=0.15,
    )
    allowed, floor, override_path = _is_confidence_override_allowed(
        settings=settings,
        decision=decision,
        override_edge=0.15,
    )
    assert allowed is True
    assert floor == 0.55
    assert override_path == "edge_default"
    assert decision.confidence < settings.MIN_CONFIDENCE
    assert decision.confidence >= floor


def test_confidence_override_floor_equal_to_min_confidence_blocks_sub_threshold() -> None:
    """Regression for the cycle 4 misconfiguration: when the override
    floor matches MIN_CONFIDENCE the override cannot rescue any candidate
    below the hard gate, even with strong edge and evidence."""
    settings = Settings(
        MIN_CONFIDENCE=0.62,
        CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED=True,
        CONFIDENCE_GATE_MIN_EDGE=0.10,
        CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY=0.70,
        CONFIDENCE_GATE_OVERRIDE_MIN_CONFIDENCE=0.62,
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.585,
        bet_size_pct=0.3,
        reasoning="Cycle 4-style sub-floor candidate",
        evidence_quality=0.80,
        edge_source="computed",
        edge_external=0.20,
    )
    allowed, floor, override_path = _is_confidence_override_allowed(
        settings=settings,
        decision=decision,
        override_edge=0.20,
    )
    assert allowed is False
    assert floor == 0.62
    assert override_path == "none"


def test_confidence_override_below_min_confidence_still_requires_floor() -> None:
    """Even when the override floor sits below MIN_CONFIDENCE, candidates
    below the floor remain blocked: the override path must not collapse
    safety to zero."""
    settings = Settings(
        MIN_CONFIDENCE=0.62,
        CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED=True,
        CONFIDENCE_GATE_MIN_EDGE=0.10,
        CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY=0.70,
        CONFIDENCE_GATE_OVERRIDE_MIN_CONFIDENCE=0.55,
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.50,
        bet_size_pct=0.3,
        reasoning="Sub-floor confidence even with strong edge",
        evidence_quality=0.85,
        edge_source="computed",
        edge_external=0.20,
    )
    allowed, _, override_path = _is_confidence_override_allowed(
        settings=settings,
        decision=decision,
        override_edge=0.20,
    )
    assert allowed is False
    assert override_path == "none"
