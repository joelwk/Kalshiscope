from __future__ import annotations

from datetime import datetime, timedelta, timezone

from config import Settings
from main import (
    _adjust_bet_size_for_edge,
    _effective_score_gate_threshold,
    _extract_winning_outcome,
    _filter_markets,
    _is_confidence_override_allowed,
    _is_uniform_implied_probability,
    _min_evidence_quality_for_market,
    _passes_edge_threshold,
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


def test_min_evidence_quality_floor_default_is_raised() -> None:
    settings = Settings()
    generic_market = Market(id="m-eq-floor", question="Will BTC close above threshold?", category="crypto")
    assert _min_evidence_quality_for_market(generic_market, settings) == 0.75


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
        FALLBACK_EDGE_MIN_EDGE=0.15,
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
    allowed, min_confidence = _is_confidence_override_allowed(
        settings=settings,
        decision=decision,
        override_edge=0.20,
    )
    assert min_confidence == 0.50
    assert allowed is False


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
    allowed, _ = _is_confidence_override_allowed(
        settings=settings,
        decision=decision,
        override_edge=0.20,
    )
    assert allowed is True


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
        SCORE_GATE_THRESHOLD=0.25,
        SCORE_GATE_THRESHOLD_WEATHER_DIRECT=0.10,
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
    assert weather_proxy_threshold == 0.25
    assert crypto_direct_threshold == 0.25
