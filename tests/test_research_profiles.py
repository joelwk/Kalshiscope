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


def test_market_family_mlb_player_prop_via_ticker_prefix() -> None:
    """Regression: KXMLBTB-* (player-prop questions) must classify as sports
    even when the natural-language question never names "MLB". The 7-cycle
    follow-up audit found a settled-game trade (KXMLBTB-26MAY031605CLEATH-
    CLEJRAMREZ11-2) was misrouted to "generic" and skipped because the
    keyword regex `\\bmlb\\b` could not match the leading "KXMLB" without a
    word boundary. Ticker-prefix detection in family_from_text now anchors
    on `\\bKXMLB`."""
    market = Market(
        id="KXMLBTB-26MAY031605CLEATH-CLEJRAMREZ11-2",
        question="Jose Ramirez: 2+ total bases?",
        category=None,
    )
    assert market_family(market) == "sports"


def test_market_family_mlb_first_5_innings_via_ticker_prefix() -> None:
    """KXMLBF5-* (cycle-14 success fixture) classifies as sports."""
    market = Market(
        id="KXMLBF5-26MAY031920TEXDET-DET",
        question="Will Detroit win first 5 innings vs Texas?",
        category="sports",
    )
    assert market_family(market) == "sports"


def test_market_family_nba_player_prop_via_ticker_prefix() -> None:
    market = Market(
        id="KXNBA-PLAYER-PROP-XYZ",
        question="Will player score 20+ points?",
        category=None,
    )
    assert market_family(market) == "sports"


def test_market_family_nfl_via_ticker_prefix() -> None:
    market = Market(
        id="KXNFL-26WK1-PHI-DAL",
        question="Will the home team win?",
        category=None,
    )
    assert market_family(market) == "sports"


def test_market_family_nhl_via_ticker_prefix() -> None:
    market = Market(
        id="KXNHL-26FEB14-RANGERS-BRUINS",
        question="Will the road team win?",
        category=None,
    )
    assert market_family(market) == "sports"


def test_market_family_ncaa_via_ticker_prefix() -> None:
    market = Market(
        id="KXNCAAB-MARMAD2026-WIN",
        question="Will the favored team advance?",
        category=None,
    )
    assert market_family(market) == "sports"


def test_market_family_ticker_prefix_takes_precedence_over_keywords() -> None:
    """Even if the question text contains a non-sports keyword, the sports
    ticker prefix wins. Guards against accidental misclassification when
    a sports market question references e.g. weather conditions."""
    market = Market(
        id="KXMLB-26AUG-RAINOUT",
        question="Will the game be a rain-out (weather affecting play)?",
        category=None,
    )
    assert market_family(market) == "sports"


def test_market_family_no_false_positive_on_non_sports_kx_ticker() -> None:
    """Tickers like KXBTC, KXHIGHCHI, KXLOWTLAX must not be miscategorized
    as sports just because they share the KX prefix. This guards against
    the sports-ticker pattern accidentally matching everything starting
    with KX. Family-specific classification (crypto/weather/etc) is
    asserted in the dedicated tests above; here we only verify that the
    sports-ticker pattern doesn't fire on non-sports prefixes."""
    crypto = Market(
        id="KXBTCD-26MAY0317-T78649.99",
        question="Bitcoin range bin?",
        category="crypto",
    )
    weather_temp = Market(
        id="KXHIGHCHI-26MAY03-T70",
        question="Will the high temperature in Chicago be above 70F?",
        category=None,
    )
    music = Market(
        id="KXBBCHARTPOSITIONSONG-26MAY09SOE-3",
        question="Will Ordinary be on the Billboard Hot 100?",
        category=None,
    )
    assert market_family(crypto) != "sports"
    assert market_family(weather_temp) != "sports"
    assert market_family(music) != "sports"
    # And the keyword-based detection still works for these too.
    assert market_family(crypto) == "crypto"
    assert market_family(weather_temp) == "weather"
    assert market_family(music) == "music"


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


def test_market_family_cricket_kbo_and_isl_tickers_from_logs() -> None:
    """Recent participation logs showed these sports tickers using generic
    search profiles and generic-family penalties."""
    for ticker in (
        "KXIPLSIX-26MAY11DCPBKS-20",
        "KXT20MATCH-26MAY10COOPNG-COO",
        "KXKBOGAME-26MAY100100KIALOT-KIA",
        "KXISLGAME-26APR291230HAHHGE-HGE",
        "KXNPBGAME-26AUG050500HOKFUK-FUK",
        "KXARGNACBTOTAL-26AUG03GUETRI-5",
        "KXARGNACBSPREAD-26AUG03GUETRI-TRI2",
    ):
        market = Market(id=ticker, question="Will the home team win?", category=None)
        assert market_family(market) == "sports", ticker


def test_market_family_crypto() -> None:
    market = Market(id="2", question="Will $BTC close above 120k?", category="crypto")
    assert market_family(market) == "crypto"


def test_market_family_crypto_15m_ticker_prefixes() -> None:
    """15-minute crypto tickers must not fall into the generic family.

    The cycle-7 logs showed KXSOL15M/KXXRP15M/KXBNB15M decisions using the
    generic search profile, which gives them generic-family PnL penalties and
    non-crypto source domains.
    """
    for ticker in (
        "KXSOL15M-26MAY051345-45",
        "KXXRP15M-26MAY051345-45",
        "KXBNB15M-26MAY051345-45",
        "KXDOGE15M-26MAY051345-45",
    ):
        market = Market(id=ticker, question="Price up in next 15 mins?", category=None)
        assert market_family(market) == "crypto"


def test_profile_for_crypto_15m_ticker_uses_crypto_profile() -> None:
    settings = Settings()
    market = Market(
        id="KXSOL15M-26MAY051345-45",
        question="SOL price up in next 15 mins?",
        category=None,
    )
    profile = profile_for_market(settings, market)
    assert profile.name == "crypto"
    assert "coinbase.com" in profile.domains


def test_market_family_crypto_daily_alt_tickers_from_logs() -> None:
    """KXSOLE/KXSHIBA were showing up as generic in recent participation logs."""
    for ticker in (
        "KXSOLE-26MAY0817-B88",
        "KXSHIBA-26MAY0717-B0.0000062",
    ):
        market = Market(id=ticker, question="Crypto spot price range?", category=None)
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


def test_market_family_entertainment_ticker_precedes_streaming_keyword() -> None:
    market = Market(
        id="KXNETFLIXTOPVIEWSMOVIE-26MAY11-21",
        question="Will this streaming movie be #1 on Netflix?",
        category="entertainment",
    )
    assert market_family(market) == "entertainment"


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


def test_profile_for_market_returns_entertainment_profile() -> None:
    settings = Settings()
    market = Market(
        id="KXNETFLIXTOPVIEWSMOVIE-26MAY11-21",
        question="Will the #1 Movie on Netflix have at least 21 million views?",
        category=None,
    )
    profile = profile_for_market(settings, market)
    assert profile.name == "entertainment"
    assert "netflix.com" in profile.domains
    assert "flixpatrol.com" in profile.domains


def test_build_search_config_keeps_full_source_pool_for_extended_research() -> None:
    now = datetime.now(timezone.utc)
    settings = Settings(
        ENTERTAINMENT_ALLOWED_DOMAINS=(
            "netflix.com",
            "top10.netflix.com",
            "flixpatrol.com",
            "boxofficemojo.com",
            "the-numbers.com",
            "variety.com",
        ),
        SEARCH_PROFILE_MAX_DOMAINS=3,
    )
    market = Market(
        id="KXNETFLIXRANKMOVIE-26MAY11-REM",
        question="Will this movie rank #1 on Netflix?",
        close_time=now + timedelta(hours=12),
    )
    search_config = build_market_search_config(settings, market, now=now)
    assert search_config.allowed_domains == [
        "netflix.com",
        "top10.netflix.com",
        "flixpatrol.com",
    ]
    assert search_config.source_domains_pool == [
        "netflix.com",
        "top10.netflix.com",
        "flixpatrol.com",
        "boxofficemojo.com",
        "the-numbers.com",
        "variety.com",
    ]


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


def test_is_commodity_market_detects_wti_ticker() -> None:
    market = Market(
        id="KXWTI-26JUL12-T70.00",
        question="Will the close price be above 70?",
        category="finance",
    )
    assert is_commodity_market(market) is True


def test_is_commodity_market_detects_natgas_keyword() -> None:
    market = Market(
        id="KXNG-26JUL12-T3.5",
        question="Will natural gas settle above 3.5?",
        category="finance",
    )
    assert is_commodity_market(market) is True


def test_profile_for_market_commodity_routes_to_commodity_profile() -> None:
    """Commodities classify under the generic taxonomy family but must route to
    the dedicated commodity search profile so the model can reach (and cite) the
    CME/ICE/EIA settlement pages required for direct evidence. Without this they
    inherited GENERIC news-wire domains and were blocked as
    missing_primary_source_url (weather-only sourcing regression)."""
    settings = Settings()
    market = Market(
        id="KXSILVERD-26JUN2217-T66.25",
        question="Will the silver close price be above 66.25 on Jun 26?",
        category="commodities",
    )
    # Taxonomy family is unchanged (no scoring/penalty impact)...
    assert market_family(market) == "generic"
    # ...but the search profile is the commodity one.
    profile = profile_for_market(settings, market)
    assert profile.name == "commodity"
    assert "cmegroup.com" in profile.domains
    assert "theice.com" in profile.domains


def test_commodity_search_config_reaches_exchange_settlement_domains() -> None:
    """The trimmed (top SEARCH_PROFILE_MAX_DOMAINS) searchable set for a commodity
    market includes the exchange settlement page, so a cited URL can satisfy the
    settlement-grade primary_source_url requirement."""
    settings = Settings()
    market = Market(
        id="KXCOPPERW-26JUN2617-T6.35",
        question="Will the copper close price be above 6.35 USD/Lbs on June 26?",
        category="commodities",
    )
    search_config = build_market_search_config(settings, market)
    assert search_config.profile_name == "commodity"
    assert "cmegroup.com" in search_config.allowed_domains


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


def test_market_family_valorant_map_ticker_is_sports() -> None:
    market = Market(
        id="KXVALORANTMAP-26AUG121100GMSGE-1-SGE",
        question="Map 1 winner?",
        category=None,
    )
    assert market_family(market) == "sports"
    _, is_esports = market_category_flags(market)
    assert is_esports is True


def test_market_family_cs2_ticker_is_sports() -> None:
    market = Market(
        id="KXCS2-26AUG12-TEAM",
        question="Match winner?",
        category=None,
    )
    assert market_family(market) == "sports"
