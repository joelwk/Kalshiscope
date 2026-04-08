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


def test_compute_final_score_defaults_new_optional_fields() -> None:
    market = Market(
        id="m4",
        question="Test",
        outcomes=[MarketOutcome(name="YES", price=0.52), MarketOutcome(name="NO", price=0.48)],
        liquidity_usdc=500.0,
        close_time=datetime.now(timezone.utc) + timedelta(days=2),
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.60,
        bet_size_pct=0.3,
        reasoning="test",
        edge_external=0.04,
        evidence_quality=0.6,
    )
    score = compute_final_score(market, decision, implied_prob_market=0.52)
    assert score.bayesian_posterior is None
    assert score.lmsr_price is None
    assert score.inefficiency_signal is None
    assert score.kelly_raw is None
    assert score.weather_uncertainty_penalty == 0.0


def test_compute_final_score_penalizes_low_evidence_multiplicatively() -> None:
    market = Market(
        id="m5",
        question="Test",
        outcomes=[MarketOutcome(name="YES", price=0.40), MarketOutcome(name="NO", price=0.60)],
        liquidity_usdc=1000.0,
        close_time=datetime.now(timezone.utc) + timedelta(days=1),
    )
    high_evidence = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.70,
        bet_size_pct=0.3,
        reasoning="test",
        edge_external=0.12,
        evidence_quality=0.95,
    )
    low_evidence = high_evidence.model_copy(update={"evidence_quality": 0.35})
    high_score = compute_final_score(market, high_evidence, implied_prob_market=0.40)
    low_score = compute_final_score(market, low_evidence, implied_prob_market=0.40)
    assert high_score.final_score > low_score.final_score


def test_compute_final_score_uses_kelly_and_inefficiency_signals() -> None:
    market = Market(
        id="m6",
        question="Test",
        outcomes=[MarketOutcome(name="YES", price=0.45), MarketOutcome(name="NO", price=0.55)],
        liquidity_usdc=1200.0,
        close_time=datetime.now(timezone.utc) + timedelta(days=1),
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.66,
        bet_size_pct=0.4,
        reasoning="test",
        edge_external=0.09,
        evidence_quality=0.75,
    )
    baseline = compute_final_score(market, decision, implied_prob_market=0.45)
    boosted = compute_final_score(
        market,
        decision,
        implied_prob_market=0.45,
        bayesian_posterior=0.72,
        inefficiency_signal=0.18,
        kelly_raw=0.35,
    )
    assert boosted.final_score > baseline.final_score
    assert boosted.kelly_component > 0
    assert boosted.inefficiency_component > 0


def test_compute_final_score_applies_weather_uncertainty_penalty() -> None:
    now = datetime.now(timezone.utc)
    market = Market(
        id="m-weather",
        question="Will rainfall exceed 2 inches in Miami?",
        outcomes=[MarketOutcome(name="YES", price=0.42), MarketOutcome(name="NO", price=0.58)],
        liquidity_usdc=900.0,
        close_time=now + timedelta(days=8),
        category="weather",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.64,
        bet_size_pct=0.35,
        reasoning="test",
        edge_external=0.07,
        evidence_quality=0.7,
    )
    baseline = compute_final_score(
        market,
        decision,
        implied_prob_market=0.42,
        now=now,
    )
    weather_adjusted = compute_final_score(
        market,
        decision,
        implied_prob_market=0.42,
        is_weather_market=True,
        weather_score_penalty=0.03,
        now=now,
    )
    assert weather_adjusted.final_score < baseline.final_score
    assert weather_adjusted.weather_uncertainty_penalty == 0.06

