from __future__ import annotations

from datetime import datetime, timedelta, timezone

from config import Settings
from main import _effective_score_gate_threshold
from models import Market, MarketOutcome, TradeDecision
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
        edge_external=0.61,
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
        market_id="KXMLBTEAMTOTAL-26APR131610HOUSEA-HOU3",
        category="sports",
        question="Will Houston score over 3 runs?",
        liquidity=1200.0,
    )
    decision = _decision(
        outcome="NO",
        confidence=0.64,
        evidence_quality=0.82,
        edge_external=0.18,
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
