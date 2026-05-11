from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from config import SearchConfig, Settings
from models import Market

_SPORTS_KEYWORDS = (
    "nba",
    "nhl",
    "nfl",
    "mlb",
    "soccer",
    "football",
    "tennis",
    "atp",
    "wta",
    "premier league",
    "la liga",
    "serie a",
    "bundesliga",
    "hockey",
    "ice hockey",
    "olympics",
    "olympic",
    "mma",
    "ufc",
    "boxing",
    "ncaa",
    "college basketball",
    "college football",
    "champions league",
    "ucl",
    "europa league",
    "uefa",
    "ligue 1",
    "eredivisie",
    "copa",
    "cricket",
    "ipl",
    "t20",
    "kbo",
    "rugby",
    "f1",
    "formula 1",
    "grand prix",
    "mls",
    "wnba",
    "afl",
)
_ESPORTS_KEYWORDS = ("cs2", "csgo", "dota", "league of legends", "valorant", "esports")
_CRYPTO_KEYWORDS = (
    "crypto",
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "dogecoin",
    "solana",
    "defi",
    "fdv",
    "token",
    "listing",
)
_POLITICS_KEYWORDS = (
    "election",
    "president",
    "presidential",
    "senate",
    "house",
    "prime minister",
    "poll",
    "referendum",
)
_SPEECH_KEYWORDS = (
    "mention",
    "will say",
    "say ",
    "speak",
    "speech",
    "press conference",
    "briefing",
    "transcript",
)
_WEATHER_KEYWORDS = (
    "temperature",
    "temp",
    "weather",
    "high temp",
    "low temp",
    "minimum temperature",
    "maximum temperature",
    "rain",
    "rainfall",
    "precipitation",
    "snow",
    "snowfall",
    "inches of snow",
    "hurricane",
    "tropical storm",
    "cyclone",
    "tornado",
    "severe weather",
    "wind",
    "wind speed",
    "windchill",
    "heat index",
    "humidity",
    "flood",
    "drought",
    "wildfire",
    "air quality",
    "aqi",
    "forecast",
    "nws",
    "noaa",
)
_COMMODITY_KEYWORDS = (
    "copper",
    "gold",
    "silver",
    "brent",
    "crude",
    "oil",
    "gas prices",
)
_MUSIC_KEYWORDS = (
    "streams",
    "streaming",
    "spotify",
    "luminate",
    "album sales",
    "pure sales",
    "activity sales",
    "billboard",
    "hits daily double",
    "hot 100",
)
_ENTERTAINMENT_KEYWORDS = (
    "netflix",
    "top 10",
    "movie",
    "film",
    "box office",
    "views",
    "tv show",
    "television",
)
_LONG_HORIZON_HINTS = ("election", "presidential", "winner", "nominee")
_SPEECH_TICKER_PATTERN = re.compile(r"MENTION", re.IGNORECASE)
_ENTERTAINMENT_TICKER_PATTERN = re.compile(
    r"\bKX(?:NETFLIX|BOXOFFICE|MOVIE|TVSHOW|STREAMING|APPSTORE|YOUTUBE|TWITCH)",
    re.IGNORECASE,
)

# Sports-league ticker prefixes used by Kalshi (case-insensitive). Required
# because keyword-based detection on natural-language questions misses
# markets like KXMLBTB-26MAY031605CLEATH-CLEJRAMREZ11-2 ("Will Jose Ramirez
# get 2+ total bases?") whose question text never names "MLB". The
# alphanumeric ticker also defeats `\bmlb\b` regex matching because the
# leading "KX" prefix prevents a word boundary before "MLB". Without this
# pattern, settled MLB outcomes were classified as "generic" family,
# inheriting the negative generic-family historical penalty and mis-
# triggering the confidence calibration shrink against the wrong bucket.
# Word-boundary anchored: `\bKX...` matches when the ticker is preceded by
# whitespace or beginning-of-string. `_market_text` concatenates
# `f"{category} {question} {market_id}"`, so the ticker is at end-of-text
# preceded by a space \u2014 the boundary fires correctly.
_SPORTS_TICKER_PATTERN = re.compile(
    r"\bKX(?:MLB|NBA|NFL|NHL|NCAA|EPL|UCL|UEFA|MLS|WNBA|AFL|ATP|WTA|UFC|MMA|BOX|F1|IPL|T20|KBO|ISL)",
    re.IGNORECASE,
)
_CRYPTO_TICKER_PATTERN = re.compile(
    r"\bKX(?:BTC|ETH|DOGE|SOL|SOLE|BNB|XRP|HYPE|SHIB|SHIBA)(?:D|15M|E)?(?:-|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ResearchProfile:
    name: str
    domains: tuple[str, ...]
    x_handles: tuple[str, ...]


def build_market_search_config(
    settings: Settings,
    market: Market,
    now: datetime | None = None,
) -> SearchConfig:
    now = now or datetime.now(timezone.utc)
    profile = profile_for_market(settings, market)
    lookback_hours = _lookback_hours(settings, market, now)
    from_date = now - timedelta(hours=lookback_hours)
    return SearchConfig(
        from_date=from_date,
        to_date=now,
        allowed_domains=_prioritized_trim(
            profile.domains,
            settings.SEARCH_PROFILE_MAX_DOMAINS,
        ),
        allowed_x_handles=_prioritized_trim(
            profile.x_handles,
            settings.SEARCH_PROFILE_MAX_X_HANDLES,
        ),
        source_domains_pool=list(
            _prioritized_trim(profile.domains, len(profile.domains))
        ),
        source_x_handles_pool=list(
            _prioritized_trim(profile.x_handles, len(profile.x_handles))
        ),
        max_allowed_domains=settings.SEARCH_PROFILE_MAX_DOMAINS,
        max_allowed_x_handles=settings.SEARCH_PROFILE_MAX_X_HANDLES,
        multimedia_confidence_range=settings.MULTIMEDIA_CONFIDENCE_THRESHOLD,
        profile_name=profile.name,
        lookback_hours=lookback_hours,
    )


def profile_for_market(settings: Settings, market: Market) -> ResearchProfile:
    family = market_family(market)
    if family == "sports":
        return ResearchProfile(
            name=family,
            domains=settings.SPORTS_ALLOWED_DOMAINS,
            x_handles=settings.SPORTS_ALLOWED_X_HANDLES,
        )
    if family == "crypto":
        return ResearchProfile(
            name=family,
            domains=settings.CRYPTO_ALLOWED_DOMAINS,
            x_handles=settings.CRYPTO_ALLOWED_X_HANDLES,
        )
    if family == "politics":
        return ResearchProfile(
            name=family,
            domains=settings.POLITICS_ALLOWED_DOMAINS,
            x_handles=settings.POLITICS_ALLOWED_X_HANDLES,
        )
    if family == "speech":
        return ResearchProfile(
            name=family,
            domains=settings.SPEECH_ALLOWED_DOMAINS,
            x_handles=settings.SPEECH_ALLOWED_X_HANDLES,
        )
    if family == "music":
        return ResearchProfile(
            name=family,
            domains=settings.MUSIC_ALLOWED_DOMAINS,
            x_handles=settings.MUSIC_ALLOWED_X_HANDLES,
        )
    if family == "weather":
        return ResearchProfile(
            name=family,
            domains=settings.WEATHER_ALLOWED_DOMAINS,
            x_handles=settings.WEATHER_ALLOWED_X_HANDLES,
        )
    if family == "entertainment":
        return ResearchProfile(
            name=family,
            domains=settings.ENTERTAINMENT_ALLOWED_DOMAINS,
            x_handles=settings.ENTERTAINMENT_ALLOWED_X_HANDLES,
        )
    return ResearchProfile(
        name="generic",
        domains=settings.GENERIC_ALLOWED_DOMAINS,
        x_handles=settings.GENERIC_ALLOWED_X_HANDLES,
    )


def market_family(market: Market) -> str:
    text = _market_text(market)
    return family_from_text(text)


def family_from_text(text: str) -> str:
    """Classify a market family from any text blob (category, question, ticker).

    Used by live trading via `market_family(market)` and by historical analytics
    in `analytics.py` and `market_state.py` so every consumer shares the same
    keyword-driven taxonomy without per-product ticker hardcoding.

    Detection precedence:
    1. Sports-league ticker prefixes (KXMLB, KXNBA, KXNFL, ...) and crypto
       ticker prefixes (KXBTC, KXETH, KXSOL15M, ...) take priority. These
       markets often have short questions with no family keyword, and the
       leading "KX" prefix defeats word-boundary keyword checks.
    2. Sports still takes precedence over crypto because sports markets often
       mention weather or finance context incidentally, while crypto prefixes
       are product-specific.
    3. Otherwise fall back to keyword matching across the concatenated
       category + question + ticker text.

    Historical note: the sports-prefix bug originally misrouted settled MLB
    markets to "generic"; the same pattern later appeared for 15-minute crypto
    tickers such as KXSOL15M/KXXRP15M/KXBNB15M, which then inherited generic
    search domains and generic-family PnL penalties.
    """
    if _SPORTS_TICKER_PATTERN.search(text or ""):
        return "sports"
    if _CRYPTO_TICKER_PATTERN.search(text or ""):
        return "crypto"
    normalized = (text or "").lower()
    if _has_keyword_match(normalized, _SPORTS_KEYWORDS) or _has_keyword_match(
        normalized, _ESPORTS_KEYWORDS
    ):
        return "sports"
    if _has_keyword_match(normalized, _CRYPTO_KEYWORDS):
        return "crypto"
    if _has_keyword_match(normalized, _POLITICS_KEYWORDS):
        return "politics"
    if _SPEECH_TICKER_PATTERN.search(text or "") or _has_keyword_match(
        normalized, _SPEECH_KEYWORDS
    ):
        return "speech"
    if _ENTERTAINMENT_TICKER_PATTERN.search(text or "") or _has_keyword_match(
        normalized,
        _ENTERTAINMENT_KEYWORDS,
    ):
        return "entertainment"
    if _has_keyword_match(normalized, _MUSIC_KEYWORDS):
        return "music"
    if _has_keyword_match(normalized, _WEATHER_KEYWORDS):
        return "weather"
    return "generic"


def is_commodity_market(market: Market) -> bool:
    return _has_keyword_match(_market_text(market), _COMMODITY_KEYWORDS)


def _has_keyword_match(text: str, keywords: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(kw)}\b", text) for kw in keywords)


def _market_text(market: Market) -> str:
    category = (market.category or "").lower()
    question = (market.question or "").lower()
    market_id = (market.id or "")
    return f"{category} {question} {market_id}"


def market_category_flags(market: Market) -> tuple[bool, bool]:
    text = _market_text(market)
    is_esports = _has_keyword_match(text, _ESPORTS_KEYWORDS)
    is_sports = _has_keyword_match(text, _SPORTS_KEYWORDS)
    return is_sports, is_esports


def _lookback_hours(settings: Settings, market: Market, now: datetime) -> int:
    family = market_family(market)
    if family == "weather":
        return _weather_lookback_hours(settings, market, now)
    if family == "speech":
        if market.close_time:
            close_time = market.close_time
            if close_time.tzinfo is None:
                close_time = close_time.replace(tzinfo=timezone.utc)
            if close_time - now <= timedelta(days=2):
                return settings.SEARCH_LOOKBACK_SHORT_HOURS
        return min(settings.SEARCH_LOOKBACK_MEDIUM_HOURS, 36)
    if family == "music":
        if market.close_time:
            close_time = market.close_time
            if close_time.tzinfo is None:
                close_time = close_time.replace(tzinfo=timezone.utc)
            if close_time - now <= timedelta(days=2):
                return settings.SEARCH_LOOKBACK_SHORT_HOURS
        return min(settings.SEARCH_LOOKBACK_LONG_HOURS, 168)

    if market.close_time:
        close_time = market.close_time
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=timezone.utc)
        delta = close_time - now
        if delta <= timedelta(hours=48):
            return settings.SEARCH_LOOKBACK_SHORT_HOURS
        if delta <= timedelta(days=7):
            return settings.SEARCH_LOOKBACK_MEDIUM_HOURS
    question = (market.question or "").lower()
    if any(token in question for token in _LONG_HORIZON_HINTS):
        return settings.SEARCH_LOOKBACK_LONG_HOURS
    return settings.SEARCH_LOOKBACK_MEDIUM_HOURS


def _weather_lookback_hours(settings: Settings, market: Market, now: datetime) -> int:
    if market.close_time is None:
        return settings.SEARCH_LOOKBACK_MEDIUM_HOURS

    close_time = market.close_time
    if close_time.tzinfo is None:
        close_time = close_time.replace(tzinfo=timezone.utc)
    delta = close_time - now

    if delta <= timedelta(hours=24):
        return settings.SEARCH_LOOKBACK_SHORT_HOURS
    if delta <= timedelta(days=3):
        return settings.SEARCH_LOOKBACK_MEDIUM_HOURS
    if delta <= timedelta(days=7):
        return settings.SEARCH_LOOKBACK_MEDIUM_HOURS
    return settings.SEARCH_LOOKBACK_LONG_HOURS


def _prioritized_trim(items: tuple[str, ...], limit: int) -> list[str]:
    if limit <= 0:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
        if len(ordered) >= limit:
            break
    return ordered
