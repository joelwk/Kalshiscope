import pytest

from config import Settings
from models import Market, MarketOutcome, TradeDecision


@pytest.fixture()
def sample_market() -> Market:
    return Market(
        id="m1",
        question="Will it rain?",
        outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
        liquidity_usdc=200.0,
        category="weather",
    )


@pytest.fixture()
def sample_decision() -> TradeDecision:
    return TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.9,
        bet_size_pct=0.5,
        reasoning="test",
    )


@pytest.fixture()
def dummy_settings() -> Settings:
    return Settings(
        XAI_API_KEY="xai-key",
        KALSHI_API_BASE_URL="https://api.example/trade-api/v2",
        KALSHI_API_KEY_ID="kalshi-key-id",
        KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        DRY_RUN=True,
        MIN_CONFIDENCE=0.7,
        MIN_LIQUIDITY_USDC=100.0,
        MAX_BET_USDC=50.0,
    )
