from __future__ import annotations

from datetime import datetime, timedelta, timezone

from config import Settings
from models import Market
from research_profiles import build_market_search_config, market_family


def test_market_family_sports() -> None:
    market = Market(id="1", question="NBA: Lakers vs Celtics", category="sports")
    assert market_family(market) == "sports"


def test_market_family_crypto() -> None:
    market = Market(id="2", question="Will $BTC close above 120k?", category="crypto")
    assert market_family(market) == "crypto"


def test_market_family_politics() -> None:
    market = Market(id="3", question="Portugal Presidential Election Winner", category="politics")
    assert market_family(market) == "politics"


def test_dynamic_lookback_short_horizon() -> None:
    now = datetime.now(timezone.utc)
    settings = Settings(
        SEARCH_LOOKBACK_SHORT_HOURS=24,
        SEARCH_LOOKBACK_MEDIUM_HOURS=72,
        SEARCH_LOOKBACK_LONG_HOURS=168,
    )
    market = Market(
        id="4",
        question="NFL: Team A vs Team B",
        close_time=now + timedelta(hours=8),
        category="sports",
    )
    config = build_market_search_config(settings, market, now=now)
    assert config.lookback_hours == 24
    assert config.profile_name == "sports"


def test_dynamic_lookback_long_horizon() -> None:
    now = datetime.now(timezone.utc)
    settings = Settings(
        SEARCH_LOOKBACK_SHORT_HOURS=24,
        SEARCH_LOOKBACK_MEDIUM_HOURS=72,
        SEARCH_LOOKBACK_LONG_HOURS=168,
    )
    market = Market(
        id="5",
        question="Presidential Election Winner",
        close_time=now + timedelta(days=30),
        category="politics",
    )
    config = build_market_search_config(settings, market, now=now)
    assert config.lookback_hours == 168
    assert config.profile_name == "politics"

