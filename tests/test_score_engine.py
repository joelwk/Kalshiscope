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
    assert score.weather_bin_penalty == 0.0
    assert score.low_information_penalty == 0.0
    assert score.observed_data_bonus == 0.0
    assert score.no_external_odds_penalty == 0.0
    assert score.repeated_analysis_penalty == 0.0
    assert isinstance(score.rejection_reasons, tuple)


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


def test_compute_final_score_applies_low_information_penalty_for_fallback_edge() -> None:
    market = Market(
        id="m-low-info",
        question="Will event happen?",
        outcomes=[MarketOutcome(name="YES", price=0.50), MarketOutcome(name="NO", price=0.50)],
        liquidity_usdc=1000.0,
        close_time=datetime.now(timezone.utc) + timedelta(days=1),
    )
    baseline_decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.65,
        bet_size_pct=0.3,
        reasoning="test",
        edge_external=0.08,
        edge_source="computed",
        evidence_quality=0.45,
    )
    low_info_decision = baseline_decision.model_copy(update={"edge_source": "fallback"})
    baseline_score = compute_final_score(market, baseline_decision, implied_prob_market=0.50)
    low_info_score = compute_final_score(market, low_info_decision, implied_prob_market=0.50)
    assert low_info_score.low_information_penalty > 0.0
    assert low_info_score.final_score < baseline_score.final_score
    assert "low_information_penalty" in low_info_score.rejection_reasons


def test_compute_final_score_adds_observed_data_bonus() -> None:
    market = Market(
        id="m-observed",
        question="Observed weather threshold",
        outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
        liquidity_usdc=1000.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=10),
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.72,
        bet_size_pct=0.3,
        reasoning="Observed",
        edge_external=0.10,
        evidence_quality=0.85,
    )
    boosted = compute_final_score(market, decision, implied_prob_market=0.55)
    baseline = compute_final_score(
        market,
        decision.model_copy(update={"evidence_quality": 0.79}),
        implied_prob_market=0.55,
    )
    assert boosted.observed_data_bonus == 0.05
    assert boosted.final_score > baseline.final_score


def test_compute_final_score_applies_no_external_odds_penalty() -> None:
    market = Market(
        id="m-no-ext",
        question="No external odds",
        outcomes=[MarketOutcome(name="YES", price=0.50), MarketOutcome(name="NO", price=0.50)],
        liquidity_usdc=1000.0,
        close_time=datetime.now(timezone.utc) + timedelta(days=1),
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.63,
        bet_size_pct=0.2,
        reasoning="No external odds found",
        edge_external=0.05,
        edge_source="fallback",
        evidence_quality=0.6,
    )
    score = compute_final_score(market, decision, implied_prob_market=0.50)
    assert score.no_external_odds_penalty == 0.02
    assert "no_external_odds_penalty" in score.rejection_reasons


def test_compute_final_score_applies_repeated_analysis_penalty() -> None:
    market = Market(
        id="m-repeat",
        question="Repeated market",
        outcomes=[MarketOutcome(name="YES", price=0.52), MarketOutcome(name="NO", price=0.48)],
        liquidity_usdc=1000.0,
        close_time=datetime.now(timezone.utc) + timedelta(days=1),
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.66,
        bet_size_pct=0.3,
        reasoning="test",
        edge_external=0.07,
        evidence_quality=0.7,
    )
    score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.52,
        repeated_analysis_count=5,
    )
    assert score.repeated_analysis_penalty == 0.20
    assert "repeated_analysis_penalty" in score.rejection_reasons


def test_compute_final_score_applies_strengthened_repeated_analysis_penalty() -> None:
    market = Market(
        id="m-repeat-strong",
        question="Repeated market",
        outcomes=[MarketOutcome(name="YES", price=0.52), MarketOutcome(name="NO", price=0.48)],
        liquidity_usdc=1000.0,
        close_time=datetime.now(timezone.utc) + timedelta(days=1),
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.66,
        bet_size_pct=0.3,
        reasoning="test",
        edge_external=0.07,
        evidence_quality=0.7,
    )
    score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.52,
        repeated_analysis_count=5,
        repeated_analysis_penalty_base=0.05,
        repeated_analysis_penalty_start_count=1,
    )
    assert score.repeated_analysis_penalty == 0.20
    assert "repeated_analysis_penalty" in score.rejection_reasons


def test_compute_final_score_applies_mention_market_penalty() -> None:
    market = Market(
        id="KXPERSONMENTION-26APR08-TOIL",
        question="Will person mention term?",
        outcomes=[MarketOutcome(name="YES", price=0.51), MarketOutcome(name="NO", price=0.49)],
        liquidity_usdc=800.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=6),
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.62,
        bet_size_pct=0.2,
        reasoning="test",
        edge_external=0.06,
        evidence_quality=0.68,
    )
    score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.51,
        mention_market_penalty_base=0.05,
    )
    assert score.mention_market_penalty == 0.05
    assert "mention_market_penalty" in score.rejection_reasons


def test_compute_final_score_applies_confidence_calibration_penalty() -> None:
    market = Market(
        id="m-calibration",
        question="Calibration test",
        outcomes=[MarketOutcome(name="YES", price=0.40), MarketOutcome(name="NO", price=0.60)],
        liquidity_usdc=1000.0,
        close_time=datetime.now(timezone.utc) + timedelta(days=1),
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.35,
        bet_size_pct=0.3,
        reasoning="test",
        edge_external=0.08,
        evidence_quality=0.7,
    )
    score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.40,
        confidence_calibration_floor=0.50,
        confidence_calibration_penalty_scale=0.08,
    )
    assert score.confidence_calibration_penalty > 0.0
    assert "confidence_calibration_penalty" in score.rejection_reasons


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


def test_compute_final_score_applies_weather_bin_penalty_for_narrow_bins() -> None:
    now = datetime.now(timezone.utc)
    market = Market(
        id="KXHIGHMIA-26APR08-B78.5",
        question="Will the high temp in Miami be 78-79°?",
        outcomes=[MarketOutcome(name="YES", price=0.33), MarketOutcome(name="NO", price=0.67)],
        liquidity_usdc=900.0,
        close_time=now + timedelta(hours=8),
        category="weather",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="NO",
        confidence=0.73,
        bet_size_pct=0.25,
        reasoning="test",
        edge_external=0.08,
        evidence_quality=0.75,
    )
    bin_adjusted = compute_final_score(
        market,
        decision,
        implied_prob_market=0.67,
        is_weather_market=True,
        weather_score_penalty=0.10,
        now=now,
    )
    market_no_bin = market.model_copy(update={"id": "KXHIGHMIA-26APR08-T78"})
    no_bin_adjusted = compute_final_score(
        market_no_bin,
        decision,
        implied_prob_market=0.67,
        is_weather_market=True,
        weather_score_penalty=0.10,
        now=now,
    )
    assert bin_adjusted.weather_bin_penalty == 0.03
    assert no_bin_adjusted.weather_bin_penalty == 0.0
    assert bin_adjusted.final_score < no_bin_adjusted.final_score

