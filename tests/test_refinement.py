from __future__ import annotations

from datetime import datetime, timedelta, timezone

from models import Market, MarketOutcome, MarketState, TradeDecision
from refinement import RefinementStrategy


class DummyGrok:
    def __init__(self, decisions: list[TradeDecision]) -> None:
        self.decisions = decisions
        self.calls = 0

    def analyze_market_deep(
        self,
        market: Market,
        previous_analysis: TradeDecision | None = None,
        search_config=None,
    ) -> TradeDecision:
        decision = self.decisions[self.calls]
        self.calls += 1
        return decision


def _market(market_id: str, close_time: datetime | None) -> Market:
    return Market(
        id=market_id,
        question="Test market?",
        outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
        close_time=close_time,
    )


def _decision(confidence: float) -> TradeDecision:
    return TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=confidence,
        bet_size_pct=0.5,
        reasoning="test",
    )


def test_should_refine_borderline_confidence() -> None:
    market = _market("m1", None)
    refinement = RefinementStrategy(market=market)
    decision = _decision(0.6)
    assert refinement.should_refine(decision, None) is True
    reasons = refinement.get_refinement_reasons(decision, None)
    assert "borderline_trade_confidence" in reasons


def test_should_refine_previous_high_confidence() -> None:
    market = _market("m2", None)
    refinement = RefinementStrategy(market=market, high_confidence_threshold=0.75)
    decision = _decision(0.6)
    state = MarketState(
        market_id="m2",
        last_analysis=datetime.now(timezone.utc),
        analysis_count=1,
        last_confidence=0.82,
        confidence_trend=[0.82],
    )
    assert refinement.should_refine(decision, state) is True


def test_should_refine_urgent_close() -> None:
    close_time = datetime.now(timezone.utc) + timedelta(days=1)
    market = _market("m3", close_time)
    refinement = RefinementStrategy(
        market=market,
        urgent_days_before_close=2,
    )
    decision = _decision(0.6)
    assert refinement.should_refine(decision, None) is True


def test_should_skip_refinement_for_high_confidence() -> None:
    close_time = datetime.now(timezone.utc) + timedelta(days=1)
    market = _market("m6", close_time)
    refinement = RefinementStrategy(
        market=market,
        urgent_days_before_close=2,
        high_confidence_threshold=0.70,
    )
    decision = _decision(0.75)
    assert refinement.should_refine(decision, None) is True


def test_perform_refinement_stops_when_confidence_leaves_borderline() -> None:
    market = _market("m4", None)
    decisions = [_decision(0.6), _decision(0.9)]
    grok = DummyGrok(decisions)
    refinement = RefinementStrategy(market=market)

    result = refinement.perform_refinement(grok, market, decisions[0])
    assert result.confidence == 0.9
    assert grok.calls == 2


def test_perform_refinement_max_passes() -> None:
    market = _market("m5", None)
    decisions = [_decision(0.6), _decision(0.7)]
    grok = DummyGrok(decisions)
    refinement = RefinementStrategy(market=market)

    result = refinement.perform_refinement(grok, market, decisions[0])
    assert result.confidence == 0.7
    assert grok.calls == 2


def test_refinement_reasons_include_low_evidence() -> None:
    market = _market("m7", None)
    refinement = RefinementStrategy(market=market)
    decision = TradeDecision(
        should_trade=False,
        outcome="YES",
        confidence=0.5,
        bet_size_pct=0.0,
        reasoning="test",
        evidence_quality=0.2,
    )
    reasons = refinement.get_refinement_reasons(decision, None, implied_prob=None, evidence_quality=0.2)
    assert "missing_implied_probability" in reasons
    assert "low_evidence_quality" in reasons
