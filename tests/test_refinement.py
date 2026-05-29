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
        **kwargs,
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
    decisions = [_decision(0.64), _decision(0.9)]
    grok = DummyGrok(decisions)
    refinement = RefinementStrategy(market=market)

    result = refinement.perform_refinement(grok, market, _decision(0.6))
    assert result.confidence == 0.9
    assert grok.calls == 2


def test_perform_refinement_max_passes() -> None:
    market = _market("m5", None)
    decisions = [_decision(0.64), _decision(0.7)]
    grok = DummyGrok(decisions)
    refinement = RefinementStrategy(market=market)

    result = refinement.perform_refinement(grok, market, _decision(0.6))
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


def test_perform_refinement_early_stop_when_no_material_change() -> None:
    market = _market("m8", None)
    initial = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.66,
        bet_size_pct=0.3,
        implied_prob_external=0.56,
        my_prob=0.66,
        edge_external=0.10,
        evidence_quality=0.7,
        reasoning="initial",
    )
    pass_one = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.669,
        bet_size_pct=0.3,
        implied_prob_external=0.56,
        my_prob=0.66,
        edge_external=0.101,
        evidence_quality=0.72,
        reasoning="pass one",
    )
    # Provide a second decision to prove second pass is skipped.
    pass_two = _decision(0.75)
    grok = DummyGrok([pass_one, pass_two])
    refinement = RefinementStrategy(market=market)

    result = refinement.perform_refinement(grok, market, initial)
    assert result.confidence == pass_one.confidence
    assert grok.calls == 1


def test_perform_refinement_skips_second_pass_on_negative_edge() -> None:
    market = _market("m9", None)
    initial = TradeDecision(
        should_trade=False,
        outcome="YES",
        confidence=0.62,
        bet_size_pct=0.0,
        implied_prob_external=0.66,
        my_prob=0.62,
        edge_external=-0.04,
        evidence_quality=0.7,
        reasoning="initial",
    )
    pass_one = TradeDecision(
        should_trade=False,
        outcome="YES",
        confidence=0.61,
        bet_size_pct=0.0,
        implied_prob_external=0.66,
        my_prob=0.61,
        edge_external=-0.05,
        evidence_quality=0.75,
        reasoning="pass one",
    )
    pass_two = _decision(0.7)
    grok = DummyGrok([pass_one, pass_two])
    refinement = RefinementStrategy(market=market)

    result = refinement.perform_refinement(grok, market, initial)
    assert result.edge_external == -0.05
    assert grok.calls == 1


def test_flip_rejected_when_confidence_below_078_non_direct() -> None:
    """A non-direct flip at confidence < 0.78 should be rejected."""
    from refinement import FLIP_CONFIDENCE_FLOOR_NON_DIRECT

    market = _market("m-flip-reject", datetime.now(timezone.utc) + timedelta(hours=12))
    initial = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.72,
        bet_size_pct=0.4,
        reasoning="initial",
        edge_external=0.08,
        evidence_quality=0.60,
        evidence_basis="proxy",
    )
    flipped = TradeDecision(
        should_trade=True,
        outcome="NO",
        confidence=0.75,
        bet_size_pct=0.5,
        reasoning="flip to NO with proxy evidence",
        edge_external=0.15,
        evidence_quality=0.60,
        evidence_basis="proxy",
    )
    grok = DummyGrok([flipped])
    refinement = RefinementStrategy(market=market)
    result = refinement.perform_refinement(grok, market, initial)
    assert result.outcome == "YES", "Flip should be rejected for non-direct below 0.78"
    assert result.confidence < initial.confidence, "Confidence should be reduced on rejected flip"
    assert FLIP_CONFIDENCE_FLOOR_NON_DIRECT == 0.78


def test_flip_accepted_when_evidence_basis_direct_and_primary_source_url() -> None:
    """A flip with direct evidence + primary_source_url at conf >= 0.70 should be accepted."""
    market = _market("m-flip-accept", datetime.now(timezone.utc) + timedelta(hours=12))
    initial = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.65,
        bet_size_pct=0.4,
        reasoning="initial",
        edge_external=0.05,
        evidence_quality=0.70,
        evidence_basis="proxy",
    )
    flipped = TradeDecision(
        should_trade=True,
        outcome="NO",
        confidence=0.75,
        bet_size_pct=0.5,
        reasoning="flip to NO with direct AP source",
        edge_external=0.15,
        evidence_quality=0.85,
        evidence_basis="direct",
        primary_source_url="https://apnews.com/article/example",
    )
    grok = DummyGrok([flipped])
    refinement = RefinementStrategy(market=market)
    result = refinement.perform_refinement(grok, market, initial)
    assert result.outcome == "NO", "Flip should be accepted with direct evidence + URL"
    assert result.confidence >= 0.70


def _sports_market(market_id: str = "KXMLBF5-26MAY041840TORTB-TB") -> Market:
    """Sports family fixture (MLB ticker prefix triggers _SPORTS_TICKER_PATTERN)."""
    return Market(
        id=market_id,
        question="Toronto vs Tampa Bay first 5 innings winner?",
        category="sports",
        outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
        close_time=datetime.now(timezone.utc) + timedelta(hours=4),
    )


def _weather_market(market_id: str = "KXLOWTLV-26MAY03-B61.5") -> Market:
    """Weather family fixture (weather keyword triggers family detection)."""
    return Market(
        id=market_id,
        question="Will the minimum temperature be 61-62 on May 3?",
        category="weather",
        outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
        close_time=datetime.now(timezone.utc) + timedelta(hours=4),
    )


def test_skip_borderline_families_suppresses_borderline_trade_trigger() -> None:
    """When the market's family is in skip_borderline_families, the
    borderline_trade_confidence trigger must NOT fire even though confidence
    is in the [0.60, 0.78] window. This protects fast-moving sports markets
    from the deep-refinement edge-erosion failure mode (TORTB F5 case)."""
    market = _sports_market()
    refinement = RefinementStrategy(
        market=market,
        skip_borderline_families=("sports",),
    )
    decision = _decision(0.62)
    reasons = refinement.get_refinement_reasons(decision, None)
    assert "borderline_trade_confidence" not in reasons


def test_skip_borderline_families_does_not_affect_non_listed_families() -> None:
    """Weather markets must still trigger borderline_trade_confidence
    refinement when sports is the only entry in skip_borderline_families."""
    market = _weather_market()
    refinement = RefinementStrategy(
        market=market,
        skip_borderline_families=("sports",),
    )
    decision = _decision(0.62)
    reasons = refinement.get_refinement_reasons(decision, None)
    assert "borderline_trade_confidence" in reasons


def test_skip_borderline_families_default_empty_preserves_existing_behavior() -> None:
    """Backward-compat: when skip_borderline_families is empty (default),
    sports markets still get the borderline_trade_confidence trigger.
    Operators must opt in via the env setting."""
    market = _sports_market()
    refinement = RefinementStrategy(market=market)
    decision = _decision(0.62)
    reasons = refinement.get_refinement_reasons(decision, None)
    assert "borderline_trade_confidence" in reasons


def test_skip_borderline_families_case_insensitive() -> None:
    """The skip list normalizes input so 'Sports', 'SPORTS', and 'sports'
    all suppress the trigger \u2014 operators may write any case in .env."""
    market = _sports_market()
    for variant in ("Sports", "SPORTS", "sports", "  sports  "):
        refinement = RefinementStrategy(
            market=market,
            skip_borderline_families=(variant,),
        )
        decision = _decision(0.62)
        reasons = refinement.get_refinement_reasons(decision, None)
        assert "borderline_trade_confidence" not in reasons, (
            f"Variant {variant!r} should suppress trigger"
        )


def test_skip_borderline_families_does_not_block_other_triggers() -> None:
    """The skip only affects borderline_trade_confidence. Other triggers
    (low_evidence_quality, high_conf_small_edge) must still fire on sports
    markets so genuine evidence/edge problems are not masked."""
    market = _sports_market()
    refinement = RefinementStrategy(
        market=market,
        skip_borderline_families=("sports",),
    )
    low_evidence_decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.62,
        bet_size_pct=0.5,
        reasoning="low evidence",
        evidence_quality=0.30,
    )
    reasons = refinement.get_refinement_reasons(low_evidence_decision, None)
    assert "low_evidence_quality" in reasons
    assert "borderline_trade_confidence" not in reasons


def test_skip_borderline_families_handles_missing_market() -> None:
    """When self.market is None the helper must not crash; treat as
    'do not skip' so refinement runs normally."""
    refinement = RefinementStrategy(
        market=None,
        skip_borderline_families=("sports",),
    )
    decision = _decision(0.62)
    reasons = refinement.get_refinement_reasons(decision, None)
    assert "borderline_trade_confidence" in reasons


def test_settings_default_skip_borderline_families_is_empty() -> None:
    """Sports borderline critique is enabled so profitable-family proxy paths can refine."""
    from config import Settings
    s = Settings()
    assert s.REFINEMENT_SKIP_BORDERLINE_FAMILIES == ()


def test_borderline_pre_execution_score_trigger() -> None:
    from config import Settings

    market = _market("m-borderline", None)
    refinement = RefinementStrategy(market=market)
    settings = Settings(
        BORDERLINE_CRITIQUE_REFINEMENT_ENABLED=True,
        BORDERLINE_CRITIQUE_REFINEMENT_SCORE_BAND=0.10,
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.70,
        bet_size_pct=0.3,
        reasoning="test",
        primary_source_url="https://example.com/source",
    )
    reasons = refinement.get_refinement_reasons(
        decision,
        None,
        settings=settings,
        pre_execution_score=0.48,
        score_threshold=0.52,
    )
    assert "borderline_pre_execution_score" in reasons
