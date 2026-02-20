from __future__ import annotations

from datetime import datetime, timedelta, timezone

from models import Market, MarketOutcome, TradeDecision
from score_engine import compute_final_score


def test_compute_final_score_higher_with_edge_and_evidence() -> None:
    market = Market(
        id="m1",
        question="Test",
        outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
        liquidity_usdc=1000.0,
        close_time=datetime.now(timezone.utc) + timedelta(days=1),
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.68,
        bet_size_pct=0.5,
        reasoning="test",
        edge_external=0.10,
        evidence_quality=0.8,
    )
    score = compute_final_score(market, decision, implied_prob_market=0.55)
    assert score.final_score > 0


def test_compute_final_score_penalizes_low_liquidity() -> None:
    now = datetime.now(timezone.utc)
    low_liq_market = Market(
        id="m2",
        question="Test",
        outcomes=[MarketOutcome(name="YES", price=0.60)],
        liquidity_usdc=20.0,
        close_time=now + timedelta(days=1),
    )
    high_liq_market = low_liq_market.model_copy(update={"id": "m3", "liquidity_usdc": 2000.0})
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.67,
        bet_size_pct=0.5,
        reasoning="test",
        edge_external=0.08,
        evidence_quality=0.7,
    )
    low_score = compute_final_score(low_liq_market, decision, implied_prob_market=0.60)
    high_score = compute_final_score(high_liq_market, decision, implied_prob_market=0.60)
    assert high_score.final_score > low_score.final_score

