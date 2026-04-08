from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from config import SearchConfig, Settings
from models import Market

_MAX_SEARCH_DOMAINS = 5
_MAX_SEARCH_HANDLES = 10

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
_LONG_HORIZON_HINTS = ("election", "presidential", "winner", "nominee")
_SPEECH_TICKER_PATTERN = re.compile(r"MENTION", re.IGNORECASE)


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
        allowed_domains=_prioritized_trim(profile.domains, _MAX_SEARCH_DOMAINS),
        allowed_x_handles=_prioritized_trim(profile.x_handles, _MAX_SEARCH_HANDLES),
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
    if family == "weather":
        return ResearchProfile(
            name=family,
            domains=settings.WEATHER_ALLOWED_DOMAINS,
            x_handles=settings.WEATHER_ALLOWED_X_HANDLES,
        )
    return ResearchProfile(
        name="generic",
        domains=settings.GENERIC_ALLOWED_DOMAINS,
        x_handles=settings.GENERIC_ALLOWED_X_HANDLES,
    )


def market_family(market: Market) -> str:
    is_sports, is_esports = market_category_flags(market)
    if is_sports or is_esports:
        return "sports"
    category = (market.category or "").lower()
    question = market.question.lower()
    text = f"{category} {question}"
    if _has_keyword_match(text, _CRYPTO_KEYWORDS):
        return "crypto"
    if _has_keyword_match(text, _POLITICS_KEYWORDS):
        return "politics"
    if _SPEECH_TICKER_PATTERN.search(market.id or "") or _has_keyword_match(text, _SPEECH_KEYWORDS):
        return "speech"
    if _has_keyword_match(text, _WEATHER_KEYWORDS):
        return "weather"
    return "generic"


def is_commodity_market(market: Market) -> bool:
    category = (market.category or "").lower()
    question = market.question.lower()
    text = f"{category} {question}"
    return _has_keyword_match(text, _COMMODITY_KEYWORDS)


def _has_keyword_match(text: str, keywords: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(kw)}\b", text) for kw in keywords)


def market_category_flags(market: Market) -> tuple[bool, bool]:
    category = (market.category or "").lower()
    question = market.question.lower()
    text = f"{category} {question}"
    is_esports = _has_keyword_match(text, _ESPORTS_KEYWORDS)
    is_sports = _has_keyword_match(text, _SPORTS_KEYWORDS)
    return is_sports, is_esports


def _lookback_hours(settings: Settings, market: Market, now: datetime) -> int:
    if market_family(market) == "weather":
        return _weather_lookback_hours(settings, market, now)

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
