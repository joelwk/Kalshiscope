import json
from pathlib import Path

import pytest

from config import Settings
from models import Market, MarketOutcome, TradeDecision


@pytest.fixture()
def predictbase_markets_snapshot() -> dict:
    path = Path(__file__).parent / "fixtures" / "predictbase_markets.json"
    return json.loads(path.read_text(encoding="utf-8"))


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
        ALCHEMY_RPC_URL="https://rpc.example",
        WALLET_PRIVATE_KEY="0xabc",
        PREDICTBASE_API_BASE_URL="https://api.example",
        DRY_RUN=True,
        EXECUTE_ONCHAIN=False,
        AUTO_APPROVE_USDC=False,
        MIN_CONFIDENCE=0.7,
        MIN_LIQUIDITY_USDC=100.0,
        MAX_BET_USDC=50.0,
        PREDICTBASE_API_KEY=None,
    )
