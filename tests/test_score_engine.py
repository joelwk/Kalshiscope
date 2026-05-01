from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from models import Market, MarketOutcome, TradeDecision
from score_engine import calibrate_confidence, compute_final_score


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


def test_calibrate_confidence_shrinks_high_values() -> None:
    assert calibrate_confidence(1.0) == pytest.approx(0.75, rel=1e-9)
    assert calibrate_confidence(0.9) == pytest.approx(0.70, rel=1e-9)
    assert calibrate_confidence(0.8) == pytest.approx(0.65, rel=1e-9)
    assert calibrate_confidence(0.7) == pytest.approx(0.60, rel=1e-9)
    assert calibrate_confidence(0.4) == pytest.approx(0.4, rel=1e-9)


def test_calibrate_confidence_respects_tuned_floor_and_factor() -> None:
    calibrated = calibrate_confidence(
        0.70,
        shrinkage_floor=0.52,
        shrinkage_factor=0.30,
    )
    assert calibrated == pytest.approx(0.574, rel=1e-9)


def test_calibrate_confidence_relaxes_shrinkage_for_direct_evidence() -> None:
    baseline = calibrate_confidence(
        0.85,
        shrinkage_floor=0.52,
        shrinkage_factor=0.30,
    )
    direct = calibrate_confidence(
        0.85,
        shrinkage_floor=0.52,
        shrinkage_factor=0.30,
        evidence_basis_class="direct",
    )
    assert baseline == pytest.approx(0.619, rel=1e-9)
    assert direct == pytest.approx(0.685, rel=1e-9)
    assert direct > baseline


def test_calibrate_confidence_direct_with_source_preserves_more_edge() -> None:
    direct_without_source = calibrate_confidence(
        0.85,
        shrinkage_floor=0.52,
        shrinkage_factor=0.30,
        evidence_basis_class="direct",
        has_primary_source_url=False,
        direct_shrinkage_boost_factor=2.0,
    )
    direct_with_source = calibrate_confidence(
        0.85,
        shrinkage_floor=0.52,
        shrinkage_factor=0.30,
        evidence_basis_class="direct",
        has_primary_source_url=True,
        direct_shrinkage_boost_factor=2.0,
    )
    assert direct_without_source == pytest.approx(0.685, rel=1e-9)
    assert direct_with_source == pytest.approx(0.79, rel=1e-9)
    assert direct_with_source > direct_without_source


def test_calibrate_confidence_relaxes_shrinkage_for_definitive_outcome() -> None:
    baseline = calibrate_confidence(
        0.85,
        shrinkage_floor=0.52,
        shrinkage_factor=0.30,
        evidence_basis_class="direct",
    )
    definitive = calibrate_confidence(
        0.85,
        shrinkage_floor=0.52,
        shrinkage_factor=0.30,
        evidence_basis_class="direct",
        definitive_outcome=True,
    )
    assert baseline == pytest.approx(0.685, rel=1e-9)
    assert definitive == pytest.approx(0.85, rel=1e-9)
    assert definitive > baseline


def test_compute_final_score_adds_overconfidence_penalty() -> None:
    market = Market(
        id="m-overconfidence",
        question="Test",
        outcomes=[MarketOutcome(name="YES", price=0.52), MarketOutcome(name="NO", price=0.48)],
        liquidity_usdc=1000.0,
        close_time=datetime.now(timezone.utc) + timedelta(days=1),
    )
    overconfident_decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.95,
        bet_size_pct=0.5,
        reasoning="test",
        edge_external=0.10,
        evidence_quality=0.50,
    )
    evidence_supported_decision = overconfident_decision.model_copy(update={"evidence_quality": 0.90})
    overconfident_score = compute_final_score(
        market,
        overconfident_decision,
        implied_prob_market=0.52,
        overconfidence_penalty_base=0.10,
    )
    evidence_supported_score = compute_final_score(
        market,
        evidence_supported_decision,
        implied_prob_market=0.52,
        overconfidence_penalty_base=0.10,
    )
    assert overconfident_score.overconfidence_penalty > 0
    assert "overconfidence_penalty" in overconfident_score.rejection_reasons
    assert evidence_supported_score.overconfidence_penalty == pytest.approx(0.0, abs=1e-9)
    assert overconfident_score.final_score < evidence_supported_score.final_score


def test_compute_final_score_overconfidence_penalty_hits_full_scale_with_large_gap() -> None:
    market = Market(
        id="m-overconfidence-max",
        question="Test",
        outcomes=[MarketOutcome(name="YES", price=0.52), MarketOutcome(name="NO", price=0.48)],
        liquidity_usdc=1200.0,
        close_time=datetime.now(timezone.utc) + timedelta(days=1),
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.95,
        bet_size_pct=0.3,
        reasoning="test",
        edge_external=0.09,
        evidence_quality=0.50,
    )
    score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.50,
        overconfidence_penalty_base=0.10,
    )
    assert score.overconfidence_penalty == pytest.approx(0.15, rel=1e-9)


def test_compute_final_score_adds_hallucinated_edge_penalty() -> None:
    market = Market(
        id="m-hallucinated-edge",
        question="Test",
        outcomes=[MarketOutcome(name="YES", price=0.20), MarketOutcome(name="NO", price=0.80)],
        liquidity_usdc=1500.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=8),
        resolution_criteria="Official settlement source",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.92,
        bet_size_pct=0.4,
        reasoning="test",
        edge_external=0.55,
        evidence_quality=0.70,
    )
    score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.20,
        max_reasonable_edge=0.45,
    )
    assert score.hallucinated_edge_penalty == pytest.approx(0.15, rel=1e-9)
    assert "hallucinated_edge" in score.rejection_reasons


def test_compute_final_score_suppresses_hallucinated_edge_penalty_when_direct_whitelist_definitive() -> None:
    market = Market(
        id="m-hallucinated-edge-suppressed",
        question="Test",
        outcomes=[MarketOutcome(name="YES", price=0.20), MarketOutcome(name="NO", price=0.80)],
        liquidity_usdc=1500.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=8),
        resolution_criteria="Official settlement source",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.92,
        bet_size_pct=0.4,
        reasoning="test",
        edge_external=0.55,
        evidence_quality=0.70,
        definitive_outcome_detected=True,
        primary_source_url="https://apnews.com/article/example",
    )
    score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.20,
        max_reasonable_edge=0.45,
        evidence_basis_class="direct",
        suppress_hallucinated_edge_penalty=True,
    )
    assert score.hallucinated_edge_penalty == pytest.approx(0.0, abs=1e-9)
    assert score.hallucinated_edge_penalty_suppressed is True
    assert "hallucinated_edge" not in score.rejection_reasons


def test_compute_final_score_applies_hallucinated_edge_penalty_when_not_definitive() -> None:
    market = Market(
        id="m-hallucinated-edge-not-definitive",
        question="Test",
        outcomes=[MarketOutcome(name="YES", price=0.20), MarketOutcome(name="NO", price=0.80)],
        liquidity_usdc=1500.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=8),
        resolution_criteria="Official settlement source",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.92,
        bet_size_pct=0.4,
        reasoning="test",
        edge_external=0.55,
        evidence_quality=0.70,
        definitive_outcome_detected=False,
        primary_source_url="https://apnews.com/article/example",
    )
    score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.20,
        max_reasonable_edge=0.45,
        evidence_basis_class="direct",
        suppress_hallucinated_edge_penalty=False,
    )
    assert score.hallucinated_edge_penalty == pytest.approx(0.15, rel=1e-9)
    assert score.hallucinated_edge_penalty_suppressed is False
    assert "hallucinated_edge" in score.rejection_reasons


def test_compute_final_score_adds_high_edge_calibration_penalty() -> None:
    market = Market(
        id="m-high-edge-cal",
        question="High edge calibration guard",
        outcomes=[MarketOutcome(name="YES", price=0.35), MarketOutcome(name="NO", price=0.65)],
        liquidity_usdc=1500.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=8),
        resolution_criteria="Official settlement source",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.78,
        bet_size_pct=0.3,
        reasoning="Large non-definitive edge.",
        edge_external=0.40,
        edge_source="computed",
        evidence_quality=0.70,
    )
    score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.35,
        max_reasonable_edge=0.45,
        evidence_basis_class="proxy",
    )
    assert score.high_edge_calibration_penalty > 0
    assert "high_edge_calibration_penalty" in score.rejection_reasons
    assert "extreme_edge_learning_queue" not in score.rejection_reasons


def test_compute_final_score_routes_extreme_edge_to_learning_queue() -> None:
    market = Market(
        id="m-extreme-edge-learning",
        question="Extreme edge learning queue guard",
        outcomes=[MarketOutcome(name="YES", price=0.20), MarketOutcome(name="NO", price=0.80)],
        liquidity_usdc=1500.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=8),
        resolution_criteria="Official settlement source",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.82,
        bet_size_pct=0.3,
        reasoning="Huge non-definitive edge.",
        edge_external=0.52,
        edge_source="computed",
        evidence_quality=0.75,
    )
    score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.20,
        max_reasonable_edge=0.45,
        evidence_basis_class="proxy",
    )
    assert score.extreme_edge_learning_queue is True
    assert score.high_edge_calibration_penalty >= 0.12
    assert "extreme_edge_learning_queue" in score.rejection_reasons


def test_compute_final_score_adds_late_stage_overconfidence_penalty_for_non_direct() -> None:
    market = Market(
        id="m-late-overconfidence",
        question="Test",
        outcomes=[MarketOutcome(name="YES", price=0.60), MarketOutcome(name="NO", price=0.40)],
        liquidity_usdc=1200.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=8),
        resolution_criteria="Official settlement source",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.90,
        bet_size_pct=0.4,
        reasoning="test",
        edge_external=0.05,
        evidence_quality=0.80,
    )
    non_direct_score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.60,
        evidence_basis_class="proxy",
    )
    direct_score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.60,
        evidence_basis_class="direct",
    )
    assert non_direct_score.late_stage_overconfidence_penalty == pytest.approx(0.08, rel=1e-9)
    assert direct_score.late_stage_overconfidence_penalty == pytest.approx(0.0, abs=1e-9)
    assert "late_stage_overconfidence" in non_direct_score.rejection_reasons


def test_compute_final_score_penalizes_numeric_strike_bins_without_direct_evidence() -> None:
    market = Market(
        id="KXINXU-26APR20H1100-T7119.9999",
        question="Index threshold",
        outcomes=[MarketOutcome(name="YES", price=0.56), MarketOutcome(name="NO", price=0.44)],
        liquidity_usdc=700.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=8),
        resolution_criteria="Official settlement source",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.67,
        bet_size_pct=0.4,
        reasoning="test",
        edge_external=0.10,
        edge_source="computed",
        evidence_quality=0.55,
    )
    proxy_score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.56,
        evidence_basis_class="proxy",
    )
    direct_score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.56,
        evidence_basis_class="direct",
    )
    assert proxy_score.numeric_strike_bin_penalty > 0.0
    assert direct_score.numeric_strike_bin_penalty == pytest.approx(0.0, abs=1e-9)
    assert "numeric_strike_bin" in proxy_score.rejection_reasons


def test_compute_final_score_defaults_new_optional_fields() -> None:
    market = Market(
        id="m4",
        question="Test",
        outcomes=[MarketOutcome(name="YES", price=0.52), MarketOutcome(name="NO", price=0.48)],
        liquidity_usdc=500.0,
        close_time=datetime.now(timezone.utc) + timedelta(days=2),
        resolution_criteria="Official settlement source",
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
    assert score.generic_bin_penalty == 0.0
    assert score.ambiguous_resolution_penalty == 0.0
    assert score.fallback_edge_penalty == 0.0
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
    assert score.no_external_odds_penalty == 0.04
    assert "no_external_odds_penalty" in score.rejection_reasons
    assert "fallback_without_external_odds" in score.rejection_reasons


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
        repeated_analysis_count=3,
        repeated_analysis_penalty_base=0.10,
        repeated_analysis_penalty_start_count=0,
    )
    assert score.repeated_analysis_penalty == pytest.approx(0.30)
    assert "repeated_analysis_penalty" in score.rejection_reasons


def test_compute_final_score_applies_fallback_edge_penalty() -> None:
    market = Market(
        id="m-fallback-penalty",
        question="Fallback edge market",
        outcomes=[MarketOutcome(name="YES", price=0.52), MarketOutcome(name="NO", price=0.48)],
        liquidity_usdc=1000.0,
        close_time=datetime.now(timezone.utc) + timedelta(days=1),
    )
    computed_decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.66,
        bet_size_pct=0.3,
        reasoning="test",
        edge_external=0.07,
        edge_source="computed",
        evidence_quality=0.7,
    )
    fallback_decision = computed_decision.model_copy(update={"edge_source": "fallback"})
    computed_score = compute_final_score(market, computed_decision, implied_prob_market=0.52)
    fallback_score = compute_final_score(market, fallback_decision, implied_prob_market=0.52)
    assert fallback_score.fallback_edge_penalty == 0.04
    assert "fallback_edge_penalty" in fallback_score.rejection_reasons
    assert fallback_score.final_score < computed_score.final_score


def test_compute_final_score_applies_proxy_evidence_penalty_for_fallback_confidence() -> None:
    market = Market(
        id="m-fallback-proxy-penalty",
        question="Fallback edge market",
        outcomes=[MarketOutcome(name="YES", price=0.52), MarketOutcome(name="NO", price=0.48)],
        liquidity_usdc=1000.0,
        close_time=datetime.now(timezone.utc) + timedelta(days=1),
    )
    fallback_decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.88,
        bet_size_pct=0.3,
        reasoning="No transcript found and no external odds available.",
        edge_external=0.06,
        edge_source="fallback",
        evidence_quality=0.40,
    )
    score = compute_final_score(
        market,
        fallback_decision,
        implied_prob_market=0.52,
        proxy_evidence_penalty_base=0.07,
    )
    assert score.proxy_evidence_penalty > 0.0
    assert "proxy_evidence_penalty" in score.rejection_reasons


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


def test_compute_final_score_applies_generic_bin_penalty_for_non_weather_bins() -> None:
    market = Market(
        id="KXNASDAQ100-26APR10H1600-B25250",
        question="Will Nasdaq-100 close in this bin?",
        outcomes=[MarketOutcome(name="YES", price=0.33), MarketOutcome(name="NO", price=0.67)],
        liquidity_usdc=900.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=8),
        category="finance",
        resolution_criteria="Exchange close print",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="NO",
        confidence=0.73,
        bet_size_pct=0.25,
        reasoning="test",
        edge_external=0.08,
        evidence_quality=0.50,
    )
    penalized = compute_final_score(
        market,
        decision,
        implied_prob_market=0.67,
        generic_bin_penalty_base=0.04,
    )
    unpenalized = compute_final_score(
        market.model_copy(update={"id": "KXNASDAQ100-26APR10H1600-ALT"}),
        decision,
        implied_prob_market=0.67,
        generic_bin_penalty_base=0.04,
    )
    assert penalized.generic_bin_penalty > 0.0
    assert "generic_bin_penalty" in penalized.rejection_reasons
    assert penalized.final_score < unpenalized.final_score


def test_compute_final_score_applies_ambiguous_resolution_penalty() -> None:
    market = Market(
        id="m-ambiguous",
        question="Will event happen?",
        outcomes=[MarketOutcome(name="YES", price=0.52), MarketOutcome(name="NO", price=0.48)],
        liquidity_usdc=900.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=8),
        resolution_criteria="",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="NO",
        confidence=0.73,
        bet_size_pct=0.25,
        reasoning="test",
        edge_external=0.08,
        evidence_quality=0.70,
    )
    score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.48,
        ambiguous_resolution_penalty_base=0.08,
    )
    assert score.ambiguous_resolution_penalty == pytest.approx(0.08)
    assert "ambiguous_resolution_penalty" in score.rejection_reasons


def test_compute_final_score_weak_setup_falls_below_score_gate_threshold() -> None:
    market = Market(
        id="m-weak-setup",
        question="Will event happen?",
        outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
        liquidity_usdc=80.0,
        close_time=datetime.now(timezone.utc) + timedelta(days=10),
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.60,
        bet_size_pct=0.2,
        reasoning="Fallback estimate with limited supporting data.",
        edge_external=0.0,
        edge_source="fallback",
        implied_prob_external=None,
        evidence_quality=0.55,
    )
    score = compute_final_score(market, decision, implied_prob_market=0.55)
    assert score.final_score < 0.22


def test_compute_final_score_adds_evidence_basis_bonus_for_direct_evidence() -> None:
    market = Market(
        id="m-direct-bonus",
        question="Direct evidence market",
        outcomes=[MarketOutcome(name="YES", price=0.50), MarketOutcome(name="NO", price=0.50)],
        liquidity_usdc=1200.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=6),
        resolution_criteria="Official settlement source",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.68,
        bet_size_pct=0.3,
        reasoning="direct evidence",
        edge_external=0.09,
        evidence_quality=0.85,
    )
    baseline = compute_final_score(
        market,
        decision,
        implied_prob_market=0.50,
        evidence_basis_class="proxy",
    )
    boosted = compute_final_score(
        market,
        decision,
        implied_prob_market=0.50,
        evidence_basis_class="direct",
    )
    assert boosted.evidence_basis_bonus == pytest.approx(0.08)
    assert boosted.final_score > baseline.final_score


def test_compute_final_score_scales_weather_penalty_for_direct_evidence() -> None:
    now = datetime.now(timezone.utc)
    market = Market(
        id="m-weather-direct",
        question="Will rainfall exceed 2 inches in Miami?",
        outcomes=[MarketOutcome(name="YES", price=0.42), MarketOutcome(name="NO", price=0.58)],
        liquidity_usdc=900.0,
        close_time=now + timedelta(days=8),
        category="weather",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.70,
        bet_size_pct=0.35,
        reasoning="observed station reports",
        edge_external=0.10,
        evidence_quality=0.9,
    )
    proxy_score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.42,
        is_weather_market=True,
        weather_score_penalty=0.15,
        evidence_basis_class="proxy",
        now=now,
    )
    direct_score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.42,
        is_weather_market=True,
        weather_score_penalty=0.15,
        evidence_basis_class="direct",
        now=now,
    )
    assert proxy_score.weather_uncertainty_penalty == pytest.approx(0.30)
    assert direct_score.weather_uncertainty_penalty == pytest.approx(0.075)
    assert direct_score.final_score > proxy_score.final_score


def test_compute_final_score_penalizes_proxy_high_confidence_low_evidence() -> None:
    market = Market(
        id="m-proxy-hc",
        question="Proxy high confidence test",
        outcomes=[MarketOutcome(name="YES", price=0.50), MarketOutcome(name="NO", price=0.50)],
        liquidity_usdc=1000.0,
        close_time=datetime.now(timezone.utc) + timedelta(days=1),
    )
    high_conf_low_eq = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.78,
        bet_size_pct=0.3,
        reasoning="No odds found, proxy reasoning only",
        edge_external=0.06,
        edge_source="fallback",
        evidence_quality=0.42,
    )
    moderate_conf = high_conf_low_eq.model_copy(update={"confidence": 0.60})
    high_score = compute_final_score(market, high_conf_low_eq, implied_prob_market=0.50)
    mod_score = compute_final_score(market, moderate_conf, implied_prob_market=0.50)
    assert high_score.proxy_evidence_penalty > mod_score.proxy_evidence_penalty
    assert high_score.final_score < mod_score.final_score


def test_compute_final_score_no_external_odds_penalty_increased() -> None:
    market = Market(
        id="m-no-ext-v2",
        question="Increased no external odds penalty",
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
    assert score.no_external_odds_penalty == 0.04


def test_compute_final_score_reduces_fallback_penalties_for_direct_evidence() -> None:
    market = Market(
        id="m-direct-fallback",
        question="Fallback edge with direct evidence",
        outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
        liquidity_usdc=1200.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=6),
        resolution_criteria="Official settlement source",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.88,
        bet_size_pct=0.3,
        reasoning="Observed direct station reports and official source.",
        edge_external=0.05,
        edge_source="fallback",
        implied_prob_external=None,
        evidence_quality=0.85,
    )
    proxy_score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.55,
        evidence_basis_class="proxy",
    )
    direct_score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.55,
        evidence_basis_class="direct",
    )
    assert direct_score.fallback_edge_penalty == pytest.approx(
        proxy_score.fallback_edge_penalty * 0.20
    )
    assert direct_score.proxy_evidence_penalty == pytest.approx(
        proxy_score.proxy_evidence_penalty * 0.20
    )
    assert direct_score.final_score > proxy_score.final_score


def test_compute_final_score_uses_moderate_direct_multiplier_for_mid_quality_fallback() -> None:
    market = Market(
        id="m-direct-fallback-mid",
        question="Fallback edge with direct evidence",
        outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
        liquidity_usdc=1200.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=6),
        resolution_criteria="Official settlement source",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.72,
        bet_size_pct=0.3,
        reasoning="Official recap with direct source details.",
        edge_external=0.05,
        edge_source="fallback",
        implied_prob_external=None,
        evidence_quality=0.60,
    )
    proxy_score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.55,
        evidence_basis_class="proxy",
    )
    direct_score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.55,
        evidence_basis_class="direct",
    )
    assert direct_score.proxy_evidence_penalty == pytest.approx(
        proxy_score.proxy_evidence_penalty * 0.25
    )
    assert direct_score.final_score > proxy_score.final_score


def test_compute_final_score_adds_definitive_outcome_bonus_for_direct_fallback() -> None:
    market = Market(
        id="m-definitive-outcome",
        question="Post-game player prop",
        outcomes=[MarketOutcome(name="YES", price=0.56), MarketOutcome(name="NO", price=0.44)],
        liquidity_usdc=900.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=4),
        resolution_criteria="Official box score",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.85,
        bet_size_pct=0.4,
        reasoning="Final score confirmed in official recap.",
        edge_external=0.29,
        edge_source="fallback",
        evidence_quality=0.60,
        likelihood_ratio=50.0,
    )
    with_bonus = compute_final_score(
        market,
        decision,
        implied_prob_market=0.56,
        evidence_basis_class="direct",
    )
    without_bonus = compute_final_score(
        market,
        decision.model_copy(update={"likelihood_ratio": 5.0}),
        implied_prob_market=0.56,
        evidence_basis_class="direct",
    )
    assert with_bonus.definitive_outcome_bonus == pytest.approx(0.06, rel=1e-9)
    assert without_bonus.definitive_outcome_bonus == pytest.approx(0.0, rel=1e-9)
    assert with_bonus.final_score > without_bonus.final_score


def test_compute_final_score_adds_computed_edge_bonus() -> None:
    market = Market(
        id="m-computed-bonus",
        question="Computed edge quality bonus",
        outcomes=[MarketOutcome(name="YES", price=0.45), MarketOutcome(name="NO", price=0.55)],
        liquidity_usdc=800.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=12),
        resolution_criteria="Settlement docs",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.74,
        bet_size_pct=0.3,
        reasoning="External odds imply 0.58; model estimate is 0.74.",
        implied_prob_external=0.58,
        my_prob=0.74,
        edge_external=0.16,
        edge_source="computed",
        evidence_quality=0.8,
    )
    score = compute_final_score(market, decision, implied_prob_market=0.55)
    assert score.computed_edge_bonus > 0.0


def test_compute_final_score_increases_repeat_penalty_for_non_actionable_streak() -> None:
    market = Market(
        id="m-repeat-streak",
        question="Repeat penalty scaling",
        outcomes=[MarketOutcome(name="YES", price=0.48), MarketOutcome(name="NO", price=0.52)],
        liquidity_usdc=900.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=24),
        resolution_criteria="Settlement docs",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.67,
        bet_size_pct=0.2,
        reasoning="Moderate edge setup.",
        edge_external=0.07,
        edge_source="fallback",
        evidence_quality=0.7,
    )
    baseline = compute_final_score(
        market,
        decision,
        implied_prob_market=0.50,
        repeated_analysis_count=4,
        repeated_analysis_penalty_base=0.10,
        repeated_analysis_penalty_start_count=1,
        non_actionable_streak=0,
    )
    streaked = compute_final_score(
        market,
        decision,
        implied_prob_market=0.50,
        repeated_analysis_count=4,
        repeated_analysis_penalty_base=0.10,
        repeated_analysis_penalty_start_count=1,
        non_actionable_streak=6,
    )
    assert streaked.repeated_analysis_penalty > baseline.repeated_analysis_penalty


def test_compute_final_score_applies_extreme_confidence_penalty() -> None:
    market = Market(
        id="m-extreme-confidence",
        question="Extreme confidence check",
        outcomes=[MarketOutcome(name="YES", price=0.51), MarketOutcome(name="NO", price=0.49)],
        liquidity_usdc=1500.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=6),
        resolution_criteria="Official settlement source",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.93,
        bet_size_pct=0.4,
        reasoning="Strong conviction.",
        edge_external=0.10,
        edge_source="computed",
        evidence_quality=0.9,
    )
    direct_score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.70,
        evidence_basis_class="direct",
        extreme_confidence_threshold=0.90,
        extreme_confidence_penalty_base=0.08,
    )
    proxy_score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.70,
        evidence_basis_class="proxy",
        extreme_confidence_threshold=0.90,
        extreme_confidence_penalty_base=0.08,
    )
    assert direct_score.extreme_confidence_penalty == pytest.approx(0.08)
    assert proxy_score.extreme_confidence_penalty == pytest.approx(0.12)
    assert "extreme_confidence_penalty" in proxy_score.rejection_reasons


def test_compute_final_score_applies_short_prefix_penalty() -> None:
    market = Market(
        id="m-short-prefix",
        question="Short prefix penalty",
        outcomes=[MarketOutcome(name="YES", price=0.53), MarketOutcome(name="NO", price=0.47)],
        liquidity_usdc=1200.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=12),
        resolution_criteria="Official settlement source",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.72,
        bet_size_pct=0.3,
        reasoning="Moderate edge.",
        edge_external=0.09,
        edge_source="computed",
        evidence_quality=0.8,
    )
    baseline = compute_final_score(
        market,
        decision,
        implied_prob_market=0.60,
        evidence_basis_class="direct",
    )
    penalized = compute_final_score(
        market,
        decision,
        implied_prob_market=0.60,
        evidence_basis_class="direct",
        short_prefix_penalty=0.10,
    )
    assert penalized.short_prefix_penalty == pytest.approx(0.10)
    assert penalized.final_score == pytest.approx(baseline.final_score - 0.10)
    assert "historical_short_prefix_pnl" in penalized.rejection_reasons


def test_compute_final_score_applies_fallback_high_confidence_penalty() -> None:
    market = Market(
        id="m-fallback-high-confidence",
        question="Fallback high-confidence check",
        outcomes=[MarketOutcome(name="YES", price=0.49), MarketOutcome(name="NO", price=0.51)],
        liquidity_usdc=900.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=8),
        resolution_criteria="Official settlement source",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.90,
        bet_size_pct=0.3,
        reasoning="Proxy confidence only.",
        edge_external=0.08,
        edge_source="fallback",
        evidence_quality=0.55,
    )
    score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.60,
        evidence_basis_class="proxy",
    )
    assert score.fallback_high_confidence_penalty == pytest.approx(0.30)
    assert "fallback_high_confidence_trade" in score.rejection_reasons


def test_compute_final_score_fallback_high_confidence_penalty_scales_with_evidence_gap() -> None:
    market = Market(
        id="m-fallback-high-confidence-gap",
        question="Fallback gap scaling check",
        outcomes=[MarketOutcome(name="YES", price=0.44), MarketOutcome(name="NO", price=0.56)],
        liquidity_usdc=800.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=6),
        resolution_criteria="Official settlement source",
    )
    lower_gap = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.90,
        bet_size_pct=0.3,
        reasoning="Fallback confidence with stronger evidence.",
        edge_external=0.08,
        edge_source="fallback",
        evidence_quality=0.75,
    )
    higher_gap = lower_gap.model_copy(update={"evidence_quality": 0.45})
    lower_gap_score = compute_final_score(
        market,
        lower_gap,
        implied_prob_market=0.60,
        evidence_basis_class="proxy",
    )
    higher_gap_score = compute_final_score(
        market,
        higher_gap,
        implied_prob_market=0.60,
        evidence_basis_class="proxy",
    )
    assert higher_gap_score.fallback_high_confidence_penalty > lower_gap_score.fallback_high_confidence_penalty
    assert higher_gap_score.final_score < lower_gap_score.final_score


def test_compute_final_score_extreme_market_edge_penalty_triggers_without_direct_evidence() -> None:
    market = Market(
        id="m-extreme-edge-penalty",
        question="Extreme market edge guard",
        outcomes=[MarketOutcome(name="YES", price=0.20), MarketOutcome(name="NO", price=0.80)],
        liquidity_usdc=900.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=5),
        resolution_criteria="Official settlement source",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.82,
        bet_size_pct=0.2,
        reasoning="Large edge from fallback evidence.",
        edge_external=0.40,
        edge_source="fallback",
        evidence_quality=0.55,
    )
    score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.45,
        evidence_basis_class="proxy",
        max_reasonable_edge=0.35,
        extreme_market_edge_penalty_base=0.08,
    )
    assert score.extreme_market_edge_penalty == pytest.approx(0.08)
    assert "extreme_market_edge_penalty" in score.rejection_reasons


def test_compute_final_score_extreme_market_edge_penalty_suppressed_for_direct_computed() -> None:
    market = Market(
        id="m-extreme-edge-direct-computed",
        question="Extreme market edge direct computed",
        outcomes=[MarketOutcome(name="YES", price=0.20), MarketOutcome(name="NO", price=0.80)],
        liquidity_usdc=900.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=5),
        resolution_criteria="Official settlement source",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.82,
        bet_size_pct=0.2,
        reasoning="Large edge from direct computed evidence.",
        edge_external=0.40,
        edge_source="computed",
        evidence_quality=0.88,
    )
    score = compute_final_score(
        market,
        decision,
        implied_prob_market=0.45,
        evidence_basis_class="direct",
        max_reasonable_edge=0.35,
        extreme_market_edge_penalty_base=0.08,
    )
    assert score.extreme_market_edge_penalty == pytest.approx(0.0)
    assert "extreme_market_edge_penalty" not in score.rejection_reasons


def test_compute_final_score_applies_historical_family_bonus_for_profitable_family() -> None:
    market = Market(
        id="m-family-bonus",
        question="Family bonus check",
        outcomes=[MarketOutcome(name="YES", price=0.48), MarketOutcome(name="NO", price=0.52)],
        liquidity_usdc=700.0,
        close_time=datetime.now(timezone.utc) + timedelta(hours=6),
        resolution_criteria="Official settlement source",
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.72,
        bet_size_pct=0.25,
        reasoning="Direct evidence for a profitable family.",
        edge_external=0.11,
        edge_source="computed",
        evidence_quality=0.82,
    )
    with_bonus = compute_final_score(
        market,
        decision,
        implied_prob_market=0.58,
        evidence_basis_class="direct",
        historical_family_pnl_total=18.0,
        historical_family_sample_size=12,
    )
    without_bonus = compute_final_score(
        market,
        decision,
        implied_prob_market=0.58,
        evidence_basis_class="direct",
        historical_family_pnl_total=18.0,
        historical_family_sample_size=0,
    )
    assert with_bonus.historical_family_bonus == pytest.approx(0.04)
    assert without_bonus.historical_family_bonus == pytest.approx(0.0, abs=1e-9)
    assert with_bonus.final_score == pytest.approx(without_bonus.final_score + 0.04)


def _make_market(market_id: str = "KXMLBGAME-26APR191420NYMCHC-CHC", **kw):
    defaults = dict(
        id=market_id,
        question="Test",
        outcomes=[MarketOutcome(name="YES", price=0.55), MarketOutcome(name="NO", price=0.45)],
        liquidity_usdc=1000.0,
        close_time=datetime.now(timezone.utc) + timedelta(days=1),
    )
    defaults.update(kw)
    return Market(**defaults)


def _make_decision(**kw):
    defaults = dict(
        should_trade=True,
        outcome="YES",
        confidence=0.75,
        bet_size_pct=0.5,
        reasoning="test",
        edge_external=0.10,
        evidence_quality=0.80,
    )
    defaults.update(kw)
    return TradeDecision(**defaults)


def test_historical_prefix_bonus_credits_proven_winners() -> None:
    market = _make_market("KXMLBGAME-26APR191420NYMCHC-CHC")
    decision = _make_decision(confidence=0.75)
    result = compute_final_score(
        market, decision, implied_prob_market=0.55,
        historical_prefix_pnl_per_trade=2.36,
        historical_prefix_sample_size=22,
    )
    assert result.historical_prefix_bonus > 0
    assert result.historical_prefix_bonus == pytest.approx(
        0.08 * min(1.0, 2.36 / 3.0), abs=1e-4
    )
    assert "historical_prefix_unprofitable" not in result.rejection_reasons


def test_historical_prefix_penalty_marks_proven_losers() -> None:
    market = _make_market("KXINXU-26APR25-T5200")
    decision = _make_decision(confidence=0.75)
    result = compute_final_score(
        market, decision, implied_prob_market=0.55,
        historical_prefix_pnl_per_trade=-1.69,
        historical_prefix_sample_size=11,
    )
    assert result.historical_prefix_penalty > 0
    assert result.historical_prefix_bonus == pytest.approx(0.0, abs=1e-9)
    assert "historical_prefix_unprofitable" in result.rejection_reasons


def test_historical_prefix_no_effect_when_small_sample() -> None:
    market = _make_market("KXNEWPREFIX-123")
    decision = _make_decision()
    result = compute_final_score(
        market, decision, implied_prob_market=0.55,
        historical_prefix_pnl_per_trade=-5.0,
        historical_prefix_sample_size=3,
    )
    assert result.historical_prefix_penalty == pytest.approx(0.0, abs=1e-9)
    assert result.historical_prefix_bonus == pytest.approx(0.0, abs=1e-9)


def test_extreme_confidence_band_penalty_skips_definitive_direct() -> None:
    market = _make_market()
    decision = _make_decision(confidence=0.97, evidence_quality=0.90)
    result = compute_final_score(
        market, decision, implied_prob_market=0.55,
        evidence_basis_class="direct",
        definitive_outcome_eligible=True,
    )
    assert result.extreme_confidence_band_penalty == pytest.approx(0.0, abs=1e-9)
    assert "extreme_confidence_band" not in result.rejection_reasons


def test_extreme_confidence_band_penalty_fires_on_proxy_high_confidence() -> None:
    market = _make_market()
    decision = _make_decision(confidence=0.97, evidence_quality=0.70)
    result = compute_final_score(
        market, decision, implied_prob_market=0.55,
        evidence_basis_class="proxy",
    )
    assert result.extreme_confidence_band_penalty > 0
    expected = 0.10 * (0.97 - 0.94) / 0.05
    assert result.extreme_confidence_band_penalty == pytest.approx(expected, abs=1e-4)
    assert "extreme_confidence_band" in result.rejection_reasons


def test_extreme_confidence_band_no_penalty_below_095() -> None:
    market = _make_market()
    decision = _make_decision(confidence=0.94, evidence_quality=0.70)
    result = compute_final_score(
        market, decision, implied_prob_market=0.55,
        evidence_basis_class="proxy",
    )
    assert result.extreme_confidence_band_penalty == pytest.approx(0.0, abs=1e-9)


def test_numeric_strike_computed_overconfidence_penalty_targets_KXBTCD() -> None:
    market = _make_market("KXBTCD-26APR0717-T68499.99")
    decision = _make_decision(confidence=0.97, evidence_quality=0.85)
    result = compute_final_score(
        market, decision, implied_prob_market=0.55,
        edge_source="computed",
    )
    assert result.numeric_strike_computed_overconfidence_penalty > 0
    expected = min(0.06, 0.05 * (0.97 - 0.85) * 4)
    assert result.numeric_strike_computed_overconfidence_penalty == pytest.approx(
        expected, abs=1e-4
    )
    assert "numeric_strike_computed_overconfidence" in result.rejection_reasons


def test_numeric_strike_penalty_not_applied_with_fallback_edge() -> None:
    market = _make_market("KXBTCD-26APR0717-T68499.99")
    decision = _make_decision(confidence=0.90, evidence_quality=0.80)
    result = compute_final_score(
        market, decision, implied_prob_market=0.55,
        edge_source="fallback",
    )
    assert result.numeric_strike_computed_overconfidence_penalty == pytest.approx(0.0, abs=1e-9)


def test_default_zero_when_no_historical_kwargs() -> None:
    market = _make_market()
    decision = _make_decision()
    result = compute_final_score(market, decision, implied_prob_market=0.55)
    assert result.historical_prefix_bonus == pytest.approx(0.0, abs=1e-9)
    assert result.historical_prefix_penalty == pytest.approx(0.0, abs=1e-9)
    assert result.extreme_confidence_band_penalty == pytest.approx(0.0, abs=1e-9)
    assert result.numeric_strike_computed_overconfidence_penalty == pytest.approx(0.0, abs=1e-9)


def test_confidence_calibration_penalty_scales_below_floor() -> None:
    market = _make_market()
    low_conf = _make_decision(confidence=0.65, evidence_quality=0.70)
    high_conf = _make_decision(confidence=0.82, evidence_quality=0.70)
    low_result = compute_final_score(
        market, low_conf, implied_prob_market=0.55,
        confidence_calibration_floor=0.78,
        confidence_calibration_penalty_scale=0.50,
    )
    high_result = compute_final_score(
        market, high_conf, implied_prob_market=0.55,
        confidence_calibration_floor=0.78,
        confidence_calibration_penalty_scale=0.50,
    )
    assert low_result.confidence_calibration_penalty > 0
    assert low_result.confidence_calibration_penalty == pytest.approx(
        (0.78 - 0.65) * 0.50, abs=1e-4
    )
    assert high_result.confidence_calibration_penalty == pytest.approx(0.0, abs=1e-9)


def test_edge_source_none_extra_penalty() -> None:
    market = _make_market()
    decision = _make_decision(confidence=0.75, evidence_quality=0.70)
    none_result = compute_final_score(
        market, decision, implied_prob_market=0.55,
        edge_source="none",
    )
    fallback_result = compute_final_score(
        market, decision, implied_prob_market=0.55,
        edge_source="fallback",
    )
    assert none_result.no_external_odds_penalty > fallback_result.no_external_odds_penalty
    penalty_diff = none_result.no_external_odds_penalty - fallback_result.no_external_odds_penalty
    assert penalty_diff == pytest.approx(0.06, abs=1e-4)


def test_coinflip_sports_penalty_triggers() -> None:
    market = _make_market()
    decision = _make_decision(confidence=0.82, evidence_quality=0.70)
    sports_coinflip = compute_final_score(
        market, decision, implied_prob_market=0.50,
        market_family="sports",
        evidence_basis_class="proxy",
    )
    assert sports_coinflip.coinflip_sports_penalty == pytest.approx(0.10, abs=1e-9)

    sports_away_from_coinflip = compute_final_score(
        market, decision, implied_prob_market=0.70,
        market_family="sports",
        evidence_basis_class="proxy",
    )
    assert sports_away_from_coinflip.coinflip_sports_penalty == pytest.approx(0.0, abs=1e-9)

    sports_direct = compute_final_score(
        market, decision, implied_prob_market=0.50,
        market_family="sports",
        evidence_basis_class="direct",
    )
    assert sports_direct.coinflip_sports_penalty == pytest.approx(0.0, abs=1e-9)

    non_sports = compute_final_score(
        market, decision, implied_prob_market=0.50,
        market_family="weather",
        evidence_basis_class="proxy",
    )
    assert non_sports.coinflip_sports_penalty == pytest.approx(0.0, abs=1e-9)


def test_kxmlbgame_positive_prefix_lifts_rank() -> None:
    game_market = _make_market("KXMLBGAME-26APR191420NYMCHC-CHC")
    hit_market = _make_market("KXMLBHIT-26APR271940SEAMIN-SEARAROZARENA56-1")
    decision = _make_decision(confidence=0.80, evidence_quality=0.80)
    game_result = compute_final_score(
        game_market, decision, implied_prob_market=0.55,
        historical_prefix_pnl_per_trade=1.88,
        historical_prefix_sample_size=26,
    )
    hit_result = compute_final_score(
        hit_market, decision, implied_prob_market=0.55,
        historical_prefix_pnl_per_trade=-3.61,
        historical_prefix_sample_size=4,
    )
    assert game_result.historical_prefix_bonus > 0
    assert game_result.final_score > hit_result.final_score


def test_historical_prefix_amplified_penalty() -> None:
    market = _make_market("KXMLBHIT-26APR271940SEAMIN-SEARAROZARENA56-1")
    decision = _make_decision(confidence=0.75, evidence_quality=0.70)
    result = compute_final_score(
        market, decision, implied_prob_market=0.55,
        historical_prefix_pnl_per_trade=-2.0,
        historical_prefix_sample_size=10,
    )
    assert result.historical_prefix_penalty > 0
    assert result.historical_prefix_penalty >= 0.04


def test_non_definitive_confidence_ceiling_caps_proxy_at_089() -> None:
    """Non-definitive proxy-evidence trades should be capped at MAX_GLOBAL_CONFIDENCE_DIRECT."""
    from config import Settings
    from main import _non_definitive_confidence_ceiling

    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.95,
        bet_size_pct=0.5,
        reasoning="test",
        evidence_quality=0.80,
        evidence_basis="proxy",
    )
    settings = Settings(MAX_GLOBAL_CONFIDENCE_DIRECT=0.89)
    ceiling = _non_definitive_confidence_ceiling(decision, settings)
    assert ceiling == pytest.approx(0.89, rel=1e-9)


def test_non_definitive_confidence_ceiling_allows_definitive() -> None:
    """Definitive-outcome-eligible decisions bypass the 0.89 ceiling."""
    from config import Settings
    from main import _non_definitive_confidence_ceiling

    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.95,
        bet_size_pct=0.5,
        reasoning="test",
        evidence_quality=0.90,
        evidence_basis="direct",
        definitive_outcome_detected=True,
        primary_source_url="https://apnews.com/article/example",
    )
    settings = Settings(MAX_GLOBAL_CONFIDENCE_DIRECT=0.89)
    ceiling = _non_definitive_confidence_ceiling(decision, settings)
    assert ceiling > 0.89

