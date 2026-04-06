from __future__ import annotations

from datetime import datetime, timezone

from config import Settings
from main import _should_adjust_position
from models import Market, MarketState, Position, TradeDecision


def _decision(confidence: float, bet_size_pct: float = 0.5) -> TradeDecision:
    return TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=confidence,
        bet_size_pct=bet_size_pct,
        reasoning="test",
    )


def test_should_adjust_new_position() -> None:
    settings = Settings(
        MAX_BET_USDC=50.0,
        MAX_POSITION_PER_MARKET_USDC=200.0,
        MIN_CONFIDENCE_INCREASE_FOR_ADD=0.1,
    )
    decision = _decision(0.8, 0.4)
    should_add, bet_pct, reason = _should_adjust_position(
        decision, None, None, None, settings
    )
    assert should_add is True
    assert bet_pct == 0.4
    assert reason == "new_position"


def test_should_adjust_blocked_at_max_position() -> None:
    settings = Settings(
        MAX_BET_USDC=50.0,
        MAX_POSITION_PER_MARKET_USDC=200.0,
        MIN_CONFIDENCE_INCREASE_FOR_ADD=0.1,
    )
    position = Position(
        market_id="m1",
        outcome="YES",
        total_amount_usdc=200.0,
        avg_confidence=0.7,
        trade_count=3,
        first_trade=datetime.now(timezone.utc),
        last_trade=datetime.now(timezone.utc),
    )
    should_add, bet_pct, reason = _should_adjust_position(
        _decision(0.9), None, position, None, settings
    )
    assert should_add is False
    assert bet_pct == 0.0
    assert reason == "max_position_reached"


def test_should_adjust_blocked_on_confidence_increase() -> None:
    settings = Settings(
        MAX_BET_USDC=50.0,
        MAX_POSITION_PER_MARKET_USDC=200.0,
        MIN_CONFIDENCE_INCREASE_FOR_ADD=0.1,
    )
    position = Position(
        market_id="m1",
        outcome="YES",
        total_amount_usdc=150.0,
        avg_confidence=0.75,
        trade_count=2,
        first_trade=datetime.now(timezone.utc),
        last_trade=datetime.now(timezone.utc),
    )
    should_add, bet_pct, reason = _should_adjust_position(
        _decision(0.8), None, position, None, settings
    )
    assert should_add is False
    assert bet_pct == 0.0
    assert reason == "insufficient_confidence_increase"


def test_should_adjust_allows_small_position_with_scaled_threshold() -> None:
    settings = Settings(
        MAX_BET_USDC=50.0,
        MAX_POSITION_PER_MARKET_USDC=200.0,
        MIN_CONFIDENCE_INCREASE_FOR_ADD=0.1,
    )
    position = Position(
        market_id="m1",
        outcome="YES",
        total_amount_usdc=1.0,
        avg_confidence=0.65,
        trade_count=1,
        first_trade=datetime.now(timezone.utc),
        last_trade=datetime.now(timezone.utc),
    )
    should_add, bet_pct, reason = _should_adjust_position(
        _decision(0.70, 0.3), None, position, None, settings
    )
    assert should_add is True
    assert bet_pct == 0.3
    assert reason == "confidence_increase_threshold_met"


def test_should_adjust_caps_to_remaining_room() -> None:
    settings = Settings(
        MAX_BET_USDC=50.0,
        MAX_POSITION_PER_MARKET_USDC=200.0,
        MIN_CONFIDENCE_INCREASE_FOR_ADD=0.1,
    )
    position = Position(
        market_id="m1",
        outcome="YES",
        total_amount_usdc=190.0,
        avg_confidence=0.7,
        trade_count=2,
        first_trade=datetime.now(timezone.utc),
        last_trade=datetime.now(timezone.utc),
    )
    decision = _decision(0.85, 0.5)
    should_add, bet_pct, reason = _should_adjust_position(
        decision, None, position, MarketState(market_id="m1"), settings
    )
    assert should_add is True
    assert round(bet_pct, 4) == 0.2
    assert reason == "high_confidence_override"


def test_position_override_respects_sports_cap() -> None:
    settings = Settings(
        MAX_BET_USDC=50.0,
        MAX_POSITION_PER_MARKET_USDC=200.0,
        MIN_CONFIDENCE_INCREASE_FOR_ADD=0.1,
        MAX_SPORTS_CONFIDENCE=0.80,
        HIGH_CONFIDENCE_POSITION_OVERRIDE=0.85,
    )
    position = Position(
        market_id="m1",
        outcome="YES",
        total_amount_usdc=50.0,
        avg_confidence=0.79,
        trade_count=2,
        first_trade=datetime.now(timezone.utc),
        last_trade=datetime.now(timezone.utc),
    )
    market = MarketState(market_id="m1")
    decision = _decision(0.80, 0.4)
    should_add, bet_pct, reason = _should_adjust_position(
        decision,
        Market(id="m1", question="NBA: Test", outcomes=[]),
        position,
        market,
        settings,
    )
    assert should_add is False
    assert bet_pct == 0.0
    assert reason == "insufficient_confidence_increase"
