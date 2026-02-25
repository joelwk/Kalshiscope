from __future__ import annotations

from datetime import datetime, timedelta, timezone

from config import Settings
from models import Market
from research_profiles import build_market_search_config, market_category_flags, market_family


def test_market_family_sports() -> None:
    market = Market(id="1", question="NBA: Lakers vs Celtics", category="sports")
    assert market_family(market) == "sports"


def test_market_family_olympics_hockey_question() -> None:
    market = Market(
        id="1b",
        question="Olympics Ice Hockey FINAL: Canada vs USA",
        category=None,
    )
    assert market_family(market) == "sports"


def test_market_family_sports_from_category_keyword() -> None:
    market = Market(
        id="1c",
        question="Who wins this matchup?",
        category="ncaa tournament",
    )
    assert market_family(market) == "sports"


def test_market_family_champions_league() -> None:
    market = Market(
        id="1d",
        question="UEFA Champions League: Atletico Madrid vs Club Brugge",
        category="soccer",
    )
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


def test_dynamic_lookback_medium_fallback_without_close_time() -> None:
    now = datetime.now(timezone.utc)
    settings = Settings(
        SEARCH_LOOKBACK_SHORT_HOURS=24,
        SEARCH_LOOKBACK_MEDIUM_HOURS=72,
        SEARCH_LOOKBACK_LONG_HOURS=168,
    )
    market = Market(
        id="6",
        question="Will this product launch this quarter?",
        close_time=None,
        category="business",
    )
    config = build_market_search_config(settings, market, now=now)
    assert config.lookback_hours == 72


def test_market_category_flags_esports() -> None:
    market = Market(
        id="7",
        question="Valorant: Team A vs Team B",
        category="esports",
    )
    is_sports, is_esports = market_category_flags(market)
    assert is_sports is False
    assert is_esports is True
