from __future__ import annotations

from datetime import datetime, timedelta, timezone

from config import Settings
from models import Market
from research_profiles import (
    build_market_search_config,
    is_commodity_market,
    market_category_flags,
    market_family,
    profile_for_market,
)


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


def test_market_family_weather() -> None:
    market = Market(
        id="w1",
        question="Will the minimum temperature be below 40F tomorrow?",
        category="weather",
    )
    assert market_family(market) == "weather"


def test_market_family_weather_precipitation_keyword() -> None:
    market = Market(
        id="w2",
        question="Will rainfall exceed 2 inches in Miami?",
        category=None,
    )
    assert market_family(market) == "weather"


def test_market_family_weather_severe_keyword() -> None:
    market = Market(
        id="w3",
        question="Will a hurricane make landfall in Florida this week?",
        category=None,
    )
    assert market_family(market) == "weather"


def test_market_family_speech_detected_from_ticker() -> None:
    market = Market(
        id="KXCARNEYMENTION-26APR08-ROCK",
        question="Will Carney say 'rocket' during remarks?",
        category="politics",
    )
    assert market_family(market) == "speech"


def test_market_family_music_detected_from_streaming_keywords() -> None:
    market = Market(
        id="KXARTISTSTREAMS-YEEZY26APR09-479.0M",
        question="Will Kanye West have above 479000000 Streams on Luminate from Apr 1 to Apr 7?",
        category="entertainment",
    )
    assert market_family(market) == "music"


def test_profile_for_market_returns_music_profile() -> None:
    settings = Settings()
    market = Market(
        id="KXALBUMSALES-THU-ACT-5000",
        question="Will Distracted have at least 5,000 Activity sales this week?",
        category="music",
    )
    profile = profile_for_market(settings, market)
    assert profile.name == "music"
    assert "billboard.com" in profile.domains
    assert "SpotifyCharts" in profile.x_handles


def test_profile_for_market_returns_speech_profile() -> None:
    settings = Settings()
    market = Market(
        id="KXPOLITICSMENTION-26APR08-MAGA",
        question="Will the speaker mention MAGA at the press conference?",
        category=None,
    )
    profile = profile_for_market(settings, market)
    assert profile.name == "speech"
    assert "c-span.org" in profile.domains
    assert "CSPAN" in profile.x_handles


def test_is_commodity_market_detects_gold() -> None:
    market = Market(
        id="c1",
        question="Will the gold close price be above 4677 on Apr 7?",
        category="commodities",
    )
    assert is_commodity_market(market) is True


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


def test_dynamic_lookback_weather_short_horizon() -> None:
    now = datetime.now(timezone.utc)
    settings = Settings(
        SEARCH_LOOKBACK_SHORT_HOURS=24,
        SEARCH_LOOKBACK_MEDIUM_HOURS=72,
        SEARCH_LOOKBACK_LONG_HOURS=168,
    )
    market = Market(
        id="w4",
        question="Will it rain in Boston tomorrow?",
        close_time=now + timedelta(hours=18),
        category="weather",
    )
    config = build_market_search_config(settings, market, now=now)
    assert config.lookback_hours == 24


def test_dynamic_lookback_weather_long_horizon() -> None:
    now = datetime.now(timezone.utc)
    settings = Settings(
        SEARCH_LOOKBACK_SHORT_HOURS=24,
        SEARCH_LOOKBACK_MEDIUM_HOURS=72,
        SEARCH_LOOKBACK_LONG_HOURS=168,
    )
    market = Market(
        id="w5",
        question="Will snowfall exceed 6 inches in Chicago in 10 days?",
        close_time=now + timedelta(days=10),
        category="weather",
    )
    config = build_market_search_config(settings, market, now=now)
    assert config.lookback_hours == 168


def test_market_category_flags_esports() -> None:
    market = Market(
        id="7",
        question="Valorant: Team A vs Team B",
        category="esports",
    )
    is_sports, is_esports = market_category_flags(market)
    assert is_sports is False
    assert is_esports is True
