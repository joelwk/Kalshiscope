from __future__ import annotations

from datetime import datetime, timedelta, timezone

from market_scheduler import MarketScheduler
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


def test_should_not_skip_if_urgent() -> None:
    now = datetime.now(timezone.utc)
    scheduler = MarketScheduler(reanalysis_cooldown_hours=6, urgent_days_before_close=2)
    market = _market("m6", now + timedelta(days=1))
    state = MarketState(
        market_id="m6",
        last_analysis=now - timedelta(hours=1),
        analysis_count=1,
        last_confidence=0.7,
        confidence_trend=[0.7],
    )

    should_skip, reason = scheduler.should_skip(market, state)
    assert should_skip is False
    assert reason == ""
