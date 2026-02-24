from __future__ import annotations

from datetime import datetime, timedelta, timezone

from config import Settings
from main import (
    _adjust_bet_size_for_edge,
    _extract_winning_outcome,
    _filter_markets,
    _is_uniform_implied_probability,
    _passes_edge_threshold,
)
from models import Market, MarketOutcome, TradeDecision


def _decision(confidence: float, bet_size_pct: float = 0.5) -> TradeDecision:
    return TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=confidence,
        bet_size_pct=bet_size_pct,
        reasoning="test",
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
        LOW_PRICE_THRESHOLD=0.58,
        LOW_PRICE_MIN_EDGE=0.10,
    )
    implied_prob = 0.55
    ok, edge, reason = _passes_edge_threshold(implied_prob, _decision(0.62), settings)
    assert ok is False
    assert edge is not None
    assert "below min" in reason


def test_edge_gate_allows_when_edge_clears_threshold() -> None:
    settings = Settings(
        MIN_EDGE=0.05,
        LOW_PRICE_THRESHOLD=0.58,
        LOW_PRICE_MIN_EDGE=0.10,
    )
    implied_prob = 0.60
    ok, edge, reason = _passes_edge_threshold(implied_prob, _decision(0.67), settings)
    assert ok is True
    assert round(edge, 4) == 0.07
    assert reason == ""


def test_edge_based_sizing_scales_down_for_small_edge() -> None:
    settings = Settings(
        MIN_EDGE=0.05,
        LOW_PRICE_THRESHOLD=0.58,
        LOW_PRICE_MIN_EDGE=0.10,
        EDGE_SCALING_RANGE=0.10,
        LOW_PRICE_BET_PENALTY=0.5,
    )
    decision = _decision(0.66, bet_size_pct=0.6)
    implied_prob = 0.60
    edge = 0.06
    adjusted = _adjust_bet_size_for_edge(decision, implied_prob, edge, settings)
    assert 0 < adjusted < decision.bet_size_pct


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
