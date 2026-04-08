from __future__ import annotations

from datetime import datetime, timedelta, timezone

from market_scheduler import MarketScheduler, remaining_reanalysis_cooldown_seconds
from models import Market, MarketOutcome, MarketState


class DummyStateManager:
    def __init__(self, mapping: dict[str, MarketState | None]) -> None:
        self.mapping = mapping

    def get_market_state(self, market_id: str) -> MarketState | None:
        return self.mapping.get(market_id)


def _market(market_id: str, close_time: datetime | None) -> Market:
    return Market(
        id=market_id,
        question="Test market?",
        outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
        close_time=close_time,
    )


def test_prioritize_markets_urgent_then_stale() -> None:
    now = datetime.now(timezone.utc)
    m1 = _market("m1", now + timedelta(days=1))
    m2 = _market("m2", now + timedelta(days=10))
    m3 = _market("m3", now + timedelta(days=10))

    state_m2 = MarketState(
        market_id="m2",
        last_analysis=now - timedelta(hours=2),
        analysis_count=1,
        last_confidence=0.6,
        confidence_trend=[0.6],
    )

    state_manager = DummyStateManager({"m1": None, "m2": state_m2, "m3": None})
    scheduler = MarketScheduler(urgent_days_before_close=2)

    ordered = scheduler.prioritize_markets([m2, m3, m1], state_manager)
    assert [m.id for m in ordered] == ["m1", "m3", "m2"]


def test_should_skip_closed_market() -> None:
    scheduler = MarketScheduler()
    market = _market("m4", datetime.now(timezone.utc) - timedelta(days=1))
    should_skip, reason = scheduler.should_skip(market, None)
    assert should_skip is True
    assert reason == "market closed"


def test_should_skip_recently_analyzed_non_urgent() -> None:
    now = datetime.now(timezone.utc)
    scheduler = MarketScheduler(reanalysis_cooldown_hours=6, urgent_days_before_close=2)
    market = _market("m5", now + timedelta(days=10))
    state = MarketState(
        market_id="m5",
        last_analysis=now - timedelta(hours=1),
        analysis_count=1,
        last_confidence=0.7,
        confidence_trend=[0.7],
    )

    should_skip, reason = scheduler.should_skip(market, state)
    assert should_skip is True
    assert reason == "recently analyzed"


def test_should_skip_if_urgent_within_urgent_cooldown() -> None:
    now = datetime.now(timezone.utc)
    scheduler = MarketScheduler(
        reanalysis_cooldown_hours=6,
        urgent_days_before_close=2,
        urgent_reanalysis_cooldown_hours=2,
    )
    market = _market("m6", now + timedelta(days=1))
    state = MarketState(
        market_id="m6",
        last_analysis=now - timedelta(hours=1),
        analysis_count=1,
        last_confidence=0.7,
        confidence_trend=[0.7],
    )

    should_skip, reason = scheduler.should_skip(market, state)
    assert should_skip is True
    assert reason == "recently analyzed"


def test_should_not_skip_if_urgent_outside_urgent_cooldown() -> None:
    now = datetime.now(timezone.utc)
    scheduler = MarketScheduler(
        reanalysis_cooldown_hours=6,
        urgent_days_before_close=2,
        urgent_reanalysis_cooldown_hours=1,
    )
    market = _market("m7", now + timedelta(days=1))
    state = MarketState(
        market_id="m7",
        last_analysis=now - timedelta(hours=2),
        analysis_count=1,
        last_confidence=0.7,
        confidence_trend=[0.7],
    )

    should_skip, reason = scheduler.should_skip(market, state)
    assert should_skip is False
    assert reason == ""


def test_should_not_skip_non_actionable_terminal_outcome_after_short_cooldown() -> None:
    now = datetime.now(timezone.utc)
    scheduler = MarketScheduler(reanalysis_cooldown_hours=6, urgent_days_before_close=2)
    market = _market("m8", now + timedelta(days=4))
    state = MarketState(
        market_id="m8",
        last_analysis=now - timedelta(minutes=20),
        analysis_count=1,
        last_confidence=0.58,
        confidence_trend=[0.58],
        last_terminal_outcome="no_trade_recommended",
    )

    should_skip, reason = scheduler.should_skip(market, state)
    assert should_skip is False
    assert reason == ""


def test_should_skip_non_actionable_terminal_outcome_on_second_streak() -> None:
    now = datetime.now(timezone.utc)
    scheduler = MarketScheduler(reanalysis_cooldown_hours=6, urgent_days_before_close=2)
    market = _market("m8b", now + timedelta(days=4))
    state = MarketState(
        market_id="m8b",
        last_analysis=now - timedelta(minutes=20),
        analysis_count=2,
        last_confidence=0.58,
        confidence_trend=[0.58, 0.57],
        last_terminal_outcome="no_trade_recommended",
        non_actionable_streak=2,
    )

    should_skip, reason = scheduler.should_skip(market, state)
    assert should_skip is True
    assert reason == "recently analyzed"


def test_should_skip_non_actionable_terminal_outcome_on_third_streak() -> None:
    now = datetime.now(timezone.utc)
    scheduler = MarketScheduler(reanalysis_cooldown_hours=6, urgent_days_before_close=2)
    market = _market("m8c-third", now + timedelta(days=4))
    state = MarketState(
        market_id="m8c-third",
        last_analysis=now - timedelta(hours=3),
        analysis_count=3,
        last_confidence=0.58,
        confidence_trend=[0.58, 0.57, 0.56],
        last_terminal_outcome="no_trade_recommended",
        non_actionable_streak=3,
    )

    should_skip, reason = scheduler.should_skip(market, state)
    assert should_skip is True
    assert reason == "recently analyzed"


def test_should_not_skip_orderbook_spread_terminal_outcome_after_short_cooldown() -> None:
    now = datetime.now(timezone.utc)
    scheduler = MarketScheduler(reanalysis_cooldown_hours=6, urgent_days_before_close=2)
    market = _market("m8c", now + timedelta(days=4))
    state = MarketState(
        market_id="m8c",
        last_analysis=now - timedelta(minutes=20),
        analysis_count=1,
        last_confidence=0.58,
        confidence_trend=[0.58],
        last_terminal_outcome="orderbook_spread_too_wide",
    )

    should_skip, reason = scheduler.should_skip(market, state)
    assert should_skip is False
    assert reason == ""


def test_hard_backoff_doubles_cooldown_for_abstain_low_evidence() -> None:
    now = datetime.now(timezone.utc)
    scheduler = MarketScheduler(reanalysis_cooldown_hours=6, urgent_days_before_close=2)
    market = _market("m-hard-backoff", now + timedelta(days=4))
    state = MarketState(
        market_id="m-hard-backoff",
        last_analysis=now - timedelta(minutes=20),
        analysis_count=1,
        last_confidence=0.55,
        confidence_trend=[0.55],
        last_terminal_outcome="abstain_low_evidence",
    )

    should_skip, reason = scheduler.should_skip(market, state)
    assert should_skip is True
    assert reason == "recently analyzed"


def test_should_skip_actionable_terminal_outcome_under_full_cooldown() -> None:
    now = datetime.now(timezone.utc)
    scheduler = MarketScheduler(reanalysis_cooldown_hours=6, urgent_days_before_close=2)
    market = _market("m9", now + timedelta(days=4))
    state = MarketState(
        market_id="m9",
        last_analysis=now - timedelta(minutes=20),
        analysis_count=1,
        last_confidence=0.72,
        confidence_trend=[0.72],
        last_terminal_outcome="order_submitted",
    )

    should_skip, reason = scheduler.should_skip(market, state)
    assert should_skip is True
    assert reason == "recently analyzed"


def test_remaining_cooldown_helper_matches_scheduler_skip() -> None:
    now = datetime.now(timezone.utc)
    scheduler = MarketScheduler(reanalysis_cooldown_hours=2, urgent_days_before_close=2)
    market = _market("m10", now + timedelta(days=5))
    state = MarketState(
        market_id="m10",
        last_analysis=now - timedelta(minutes=30),
        analysis_count=1,
        last_confidence=0.65,
        confidence_trend=[0.65],
        last_terminal_outcome="order_submitted",
    )

    remaining = remaining_reanalysis_cooldown_seconds(
        market,
        state,
        reanalysis_cooldown_hours=scheduler.reanalysis_cooldown_hours,
        urgent_days_before_close=scheduler.urgent_days_before_close,
        urgent_reanalysis_cooldown_hours=scheduler.urgent_reanalysis_cooldown_hours,
        now=now,
    )
    should_skip, _ = scheduler.should_skip(market, state)
    assert remaining is not None
    assert remaining > 0
    assert should_skip is True


def test_fill_failure_count_applies_cooldown_multiplier() -> None:
    now = datetime.now(timezone.utc)
    scheduler = MarketScheduler(reanalysis_cooldown_hours=2, urgent_days_before_close=2)
    market = _market("m-fill-fail", now + timedelta(days=3))
    state = MarketState(
        market_id="m-fill-fail",
        last_analysis=now - timedelta(hours=3),
        analysis_count=4,
        last_confidence=0.61,
        confidence_trend=[0.6, 0.61],
        fill_failure_count=3,
    )
    should_skip, reason = scheduler.should_skip(market, state)
    assert should_skip is True
    assert reason == "recently analyzed"
