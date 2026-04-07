from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Risk controls - Conservative defaults for value betting
    MIN_BET_USDC: float = 1.0
    MAX_BET_USDC: float = 50.0
    MIN_CONFIDENCE: float = 0.50  # Allows mid-probability bets; edge + evidence gates handle quality
    CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED: bool = True
    CONFIDENCE_GATE_MIN_EDGE: float = 0.10
    CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY: float = 0.70
    SLIPPAGE_CONFIDENCE_THRESHOLD: float = 0.70
    SLIPPAGE_PCT: float = 0.02
    MIN_LIQUIDITY_USDC: float = 100.0
    POLL_INTERVAL_SEC: int = 300

    # Edge gating / sizing
    MIN_EDGE: float = 0.05
    LOW_PRICE_THRESHOLD: float = 0.50
    HIGH_PRICE_THRESHOLD: float = 0.65
    LOW_PRICE_MIN_EDGE: float = 0.08
    COINFLIP_PRICE_LOWER: float = 0.45
    COINFLIP_PRICE_UPPER: float = 0.55
    EDGE_SCALING_RANGE: float = 0.15
    LOW_PRICE_BET_PENALTY: float = 0.50
    FALLBACK_EDGE_MIN_EDGE: float = 0.08
    REQUIRE_IMPLIED_PRICE: bool = True
    
    # Confidence caps to prevent overconfidence on high-variance events
    MAX_SPORTS_CONFIDENCE: float = 0.80  # Cap sports bets at 80% confidence
    MAX_ESPORTS_CONFIDENCE: float = 0.75  # Cap esports at 75%

    # Filtering
    MARKET_CATEGORIES_ALLOWLIST: tuple[str, ...] = ()
    MARKET_CATEGORIES_BLOCKLIST: tuple[str, ...] = ()
    MARKET_TICKER_BLOCKLIST_PREFIXES: tuple[str, ...] = (
        "KXBTC15M-",
        "KXETH15M-",
        "KXSOL15M-",
        "KXDOGE15M-",
        "KXBNB15M-",
        "KXXRP15M-",
        "KXHYPE15M-",
        "KXNETFLIX",
        "KXSPOTIFY",
        "KXMADDOW",
    )
    SKIP_WEATHER_BIN_MARKETS: bool = True
    MIN_VOLUME_24H: float = 0.0
    EXTREME_YES_PRICE_LOWER: float = 0.05
    EXTREME_YES_PRICE_UPPER: float = 0.95
    LADDER_COLLAPSE_THRESHOLD: int = 5
    MAX_BRACKETS_PER_EVENT: int = 3
    # Date range filtering: only consider markets closing within this window (days from now)
    # Set to 0 or None to disable the filter
    MARKET_MIN_CLOSE_DAYS: int | None = None  # Minimum days until close (skip markets closing too soon)
    MARKET_MAX_CLOSE_DAYS: int | None = None  # Maximum days until close (skip markets closing too far out)

    # xAI Grok
    XAI_API_KEY: str = ""
    GROK_MODEL: str = "grok-4-1-fast-reasoning"
    GROK_MODEL_DEEP: str = "grok-4.20-beta-0309-reasoning"
    SEARCH_LOOKBACK_HOURS: int = 24
    SEARCH_ALLOWED_DOMAINS: tuple[str, ...] = (
        "espn.com",
        "cbssports.com",
        "nba.com",
        "covers.com",
        "sportsbookreview.com",
        "theathletic.com",
        "rotowire.com",
        "actionnetwork.com",
        "atptour.com",
        "wtatennis.com",
        "tennisexplorer.com",
        "flashscore.com",
    )
    SEARCH_ALLOWED_X_HANDLES: tuple[str, ...] = (
        "ESPN",
        "CBSSports",
        "NBA",
        "SportsCenter",
        "ShamsCharania",
        "wojespn",
        "FDSportsbook",
        "DKSportsbook",
        "BetMGM",
        "coinbase",
        "krakenfx",
        "business",
        "Reuters",
        "ReutersBiz",
        "WSJ",
        "FT",
        "CNBC",
        "MarketWatch",
        "TheEconomist",
        "YahooFinance",
        "GoUncensored",
        "ZssBecker",
        "WallStreetMav",
        "CryptoHayes",
        "elonmusk",
        "TrustlessState",
        "WhaleInsider",
        "WallStreetApes",
        "WatcherGuru",
        "intocryptoverse",
    )
    MULTIMEDIA_CONFIDENCE_THRESHOLD: tuple[float, float] = (0.55, 0.75)
    # Dynamic search windows by market horizon
    SEARCH_LOOKBACK_SHORT_HOURS: int = 24
    SEARCH_LOOKBACK_MEDIUM_HOURS: int = 72
    SEARCH_LOOKBACK_LONG_HOURS: int = 168
    # Category-specific source profiles
    SPORTS_ALLOWED_DOMAINS: tuple[str, ...] = (
        "espn.com",
        "cbssports.com",
        "nba.com",
        "covers.com",
        "sportsbookreview.com",
        "theathletic.com",
        "rotowire.com",
        "actionnetwork.com",
        "atptour.com",
        "wtatennis.com",
        "tennisexplorer.com",
        "flashscore.com",
    )
    SPORTS_ALLOWED_X_HANDLES: tuple[str, ...] = (
        "ESPN",
        "CBSSports",
        "NBA",
        "SportsCenter",
        "ShamsCharania",
        "wojespn",
        "FDSportsbook",
        "DKSportsbook",
        "BetMGM",
        "ataborasso",
        "TennisChannel",
        "WTA",
        "atptour",
    )
    CRYPTO_ALLOWED_DOMAINS: tuple[str, ...] = (
        "coindesk.com",
        "cointelegraph.com",
        "theblock.co",
        "decrypt.co",
        "messari.io",
        "coinbase.com",
        "kraken.com",
    )
    CRYPTO_ALLOWED_X_HANDLES: tuple[str, ...] = (
        "coinbase",
        "krakenfx",
        "CoinDesk",
        "TheBlock__",
        "WatcherGuru",
        "intocryptoverse",
        "WhaleInsider",
    )
    POLITICS_ALLOWED_DOMAINS: tuple[str, ...] = (
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "politico.com",
        "economist.com",
        "ft.com",
    )
    POLITICS_ALLOWED_X_HANDLES: tuple[str, ...] = (
        "Reuters",
        "ReutersBiz",
        "AP",
        "BBCWorld",
        "politico",
        "WSJ",
        "FT",
    )
    WEATHER_ALLOWED_DOMAINS: tuple[str, ...] = (
        "weather.gov",
        "weather.com",
        "wunderground.com",
        "accuweather.com",
        "metoffice.gov.uk",
    )
    WEATHER_ALLOWED_X_HANDLES: tuple[str, ...] = (
        "NWS",
        "weatherchannel",
        "NWSSPC",
        "NHC_Atlantic",
        "metoffice",
    )
    GENERIC_ALLOWED_DOMAINS: tuple[str, ...] = (
        "reuters.com",
        "apnews.com",
        "wsj.com",
        "ft.com",
        "economist.com",
    )
    GENERIC_ALLOWED_X_HANDLES: tuple[str, ...] = (
        "Reuters",
        "ReutersBiz",
        "WSJ",
        "FT",
        "CNBC",
        "MarketWatch",
        "YahooFinance",
    )

    # Kalshi
    KALSHI_API_BASE_URL: str = "https://api.elections.kalshi.com/trade-api/v2"
    KALSHI_API_KEY_ID: str = ""
    KALSHI_PRIVATE_KEY_PATH: str = "kalshi-scope.txt"
    KALSHI_SERVER_SIDE_FILTERS_ENABLED: bool = True

    # Execution
    DRY_RUN: bool = True
    PRE_ORDER_MARKET_REFRESH: bool = False
    ORDERBOOK_PRECHECK_ENABLED: bool = True
    ORDERBOOK_PRECHECK_MIN_CONFIDENCE: float = 0.75
    ORDER_PRICE_IMPROVEMENT_CENTS: int = 2
    CALIBRATION_MODE_ENABLED: bool = False
    CALIBRATION_MIN_SAMPLES: int = 20

    # State management
    STATE_DB_PATH: str = "data/market_state.db"
    STATE_JSON_EXPORT_PATH: str = "data/market_state.json"
    EXPORT_STATE_JSON: bool = True

    # Re-analysis controls
    REANALYSIS_COOLDOWN_HOURS: int = 6
    URGENT_REANALYSIS_DAYS_BEFORE_CLOSE: int = 1
    URGENT_REANALYSIS_COOLDOWN_HOURS: int = 1
    PARALLEL_ANALYSIS_ENABLED: bool = True
    ANALYSIS_MAX_WORKERS: int = 3
    MAX_MARKETS_PER_CYCLE: int = 50

    # Resolution tracking
    RESOLUTION_SYNC_INTERVAL_CYCLES: int = 3

    # Position limits
    MAX_POSITION_PER_MARKET_USDC: float = 200.0
    MAX_POSITION_PCT_OF_BANKROLL: float = 0.15
    MIN_CONFIDENCE_INCREASE_FOR_ADD: float = 0.10
    HIGH_CONFIDENCE_POSITION_OVERRIDE: float = 0.85  # Allow adding to position if conf >= this
    OPPOSITE_OUTCOME_STRATEGY: str = "block"  # block|hedge

    # Score gate (phase A/B can run in shadow mode)
    SCORE_GATE_MODE: str = "shadow"  # off|shadow|active
    SCORE_GATE_THRESHOLD: float = 0.08

    # Bayesian + LMSR + Kelly experimental layers
    BAYESIAN_ENABLED: bool = False
    BAYESIAN_SKIP_STALE_UPDATES: bool = True
    BAYESIAN_PRIOR_DEFAULT: float = 0.50
    BAYESIAN_MIN_UPDATES_FOR_TRADE: int = 1
    LMSR_ENABLED: bool = False
    LMSR_LIQUIDITY_PARAM_B: float = 100000.0
    LMSR_MIN_INEFFICIENCY: float = 0.05
    KELLY_SIZING_ENABLED: bool = False
    KELLY_FRACTION_DEFAULT: float = 0.25
    KELLY_FRACTION_SHORT_HORIZON_HOURS: int = 1
    KELLY_FRACTION_SHORT_HORIZON: float = 0.10
    KELLY_MIN_BET_POLICY: str = "fallback_edge_scaling"  # skip|floor|fallback_edge_scaling
    KELLY_MIN_BANKROLL_USDC: float = 50.0

    # Side-flip guardrails
    FLIP_GUARD_ENABLED: bool = True
    FLIP_GUARD_MIN_ABS_CONFIDENCE: float = 0.65
    FLIP_GUARD_MIN_CONF_GAIN: float = 0.08
    FLIP_GUARD_MIN_EDGE_GAIN: float = 0.03
    FLIP_GUARD_MIN_EVIDENCE_QUALITY: float = 0.60
    FLIP_CIRCUIT_BREAKER_ENABLED: bool = True
    FLIP_CIRCUIT_BREAKER_MAX_FLIPS: int = 3

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE_LEVEL: str = "DEBUG"
    LOG_DIR: str = "logs"
    ENABLE_FILE_LOGGING: bool = True
    ENABLE_JSON_LOGGING: bool = True
    ENABLE_COLORED_LOGGING: bool = True


BASE_REQUIRED_ENV_VARS = (
    "XAI_API_KEY",
    "KALSHI_API_KEY_ID",
    "KALSHI_PRIVATE_KEY_PATH",
)


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    items = [item.strip() for item in value.split(",")]
    return tuple(item for item in items if item)


def _read_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _read_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _read_env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw


def _read_env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    return _split_csv(raw)


def _read_env_float_pair(
    name: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        left, right = [part.strip() for part in raw.split(",", maxsplit=1)]
        return (float(left), float(right))
    except (ValueError, TypeError):
        return default


def _read_env_int_optional(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if not raw or raw.strip().lower() in {"", "none", "null"}:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_settings() -> Settings:
    legacy_model_aliases: dict[str, str] = {}

    requested_model_initial = _read_env_str("GROK_MODEL", Settings.GROK_MODEL).strip()
    normalized_model_initial = legacy_model_aliases.get(
        requested_model_initial,
        requested_model_initial,
    )
    requested_model_deep = _read_env_str(
        "GROK_MODEL_DEEP",
        Settings.GROK_MODEL_DEEP,
    ).strip()
    normalized_model_deep = legacy_model_aliases.get(
        requested_model_deep,
        requested_model_deep,
    )

    settings = Settings(
        MIN_BET_USDC=_read_env_float("MIN_BET_USDC", Settings.MIN_BET_USDC),
        MAX_BET_USDC=_read_env_float("MAX_BET_USDC", Settings.MAX_BET_USDC),
        MIN_CONFIDENCE=_read_env_float("MIN_CONFIDENCE", Settings.MIN_CONFIDENCE),
        CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED=_read_env_bool(
            "CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED",
            Settings.CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED,
        ),
        CONFIDENCE_GATE_MIN_EDGE=_read_env_float(
            "CONFIDENCE_GATE_MIN_EDGE", Settings.CONFIDENCE_GATE_MIN_EDGE
        ),
        CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY=_read_env_float(
            "CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY",
            Settings.CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY,
        ),
        MIN_EDGE=_read_env_float("MIN_EDGE", Settings.MIN_EDGE),
        LOW_PRICE_THRESHOLD=_read_env_float(
            "LOW_PRICE_THRESHOLD", Settings.LOW_PRICE_THRESHOLD
        ),
        HIGH_PRICE_THRESHOLD=_read_env_float(
            "HIGH_PRICE_THRESHOLD", Settings.HIGH_PRICE_THRESHOLD
        ),
        LOW_PRICE_MIN_EDGE=_read_env_float(
            "LOW_PRICE_MIN_EDGE", Settings.LOW_PRICE_MIN_EDGE
        ),
        COINFLIP_PRICE_LOWER=_read_env_float(
            "COINFLIP_PRICE_LOWER", Settings.COINFLIP_PRICE_LOWER
        ),
        COINFLIP_PRICE_UPPER=_read_env_float(
            "COINFLIP_PRICE_UPPER", Settings.COINFLIP_PRICE_UPPER
        ),
        EDGE_SCALING_RANGE=_read_env_float(
            "EDGE_SCALING_RANGE", Settings.EDGE_SCALING_RANGE
        ),
        LOW_PRICE_BET_PENALTY=_read_env_float(
            "LOW_PRICE_BET_PENALTY", Settings.LOW_PRICE_BET_PENALTY
        ),
        FALLBACK_EDGE_MIN_EDGE=_read_env_float(
            "FALLBACK_EDGE_MIN_EDGE", Settings.FALLBACK_EDGE_MIN_EDGE
        ),
        REQUIRE_IMPLIED_PRICE=_read_env_bool(
            "REQUIRE_IMPLIED_PRICE", Settings.REQUIRE_IMPLIED_PRICE
        ),
        MAX_SPORTS_CONFIDENCE=_read_env_float(
            "MAX_SPORTS_CONFIDENCE", Settings.MAX_SPORTS_CONFIDENCE
        ),
        MAX_ESPORTS_CONFIDENCE=_read_env_float(
            "MAX_ESPORTS_CONFIDENCE", Settings.MAX_ESPORTS_CONFIDENCE
        ),
        SLIPPAGE_CONFIDENCE_THRESHOLD=_read_env_float(
            "SLIPPAGE_CONFIDENCE_THRESHOLD",
            Settings.SLIPPAGE_CONFIDENCE_THRESHOLD,
        ),
        SLIPPAGE_PCT=_read_env_float("SLIPPAGE_PCT", Settings.SLIPPAGE_PCT),
        MIN_LIQUIDITY_USDC=_read_env_float(
            "MIN_LIQUIDITY_USDC", Settings.MIN_LIQUIDITY_USDC
        ),
        POLL_INTERVAL_SEC=_read_env_int(
            "POLL_INTERVAL_SEC", Settings.POLL_INTERVAL_SEC
        ),
        MARKET_CATEGORIES_ALLOWLIST=_split_csv(
            os.getenv("MARKET_CATEGORIES_ALLOWLIST")
        ),
        MARKET_CATEGORIES_BLOCKLIST=_split_csv(
            os.getenv("MARKET_CATEGORIES_BLOCKLIST")
        ),
        MARKET_TICKER_BLOCKLIST_PREFIXES=_read_env_csv(
            "MARKET_TICKER_BLOCKLIST_PREFIXES",
            Settings.MARKET_TICKER_BLOCKLIST_PREFIXES,
        ),
        SKIP_WEATHER_BIN_MARKETS=_read_env_bool(
            "SKIP_WEATHER_BIN_MARKETS", Settings.SKIP_WEATHER_BIN_MARKETS
        ),
        MIN_VOLUME_24H=_read_env_float("MIN_VOLUME_24H", Settings.MIN_VOLUME_24H),
        EXTREME_YES_PRICE_LOWER=_read_env_float(
            "EXTREME_YES_PRICE_LOWER",
            Settings.EXTREME_YES_PRICE_LOWER,
        ),
        EXTREME_YES_PRICE_UPPER=_read_env_float(
            "EXTREME_YES_PRICE_UPPER",
            Settings.EXTREME_YES_PRICE_UPPER,
        ),
        LADDER_COLLAPSE_THRESHOLD=_read_env_int(
            "LADDER_COLLAPSE_THRESHOLD",
            Settings.LADDER_COLLAPSE_THRESHOLD,
        ),
        MAX_BRACKETS_PER_EVENT=_read_env_int(
            "MAX_BRACKETS_PER_EVENT",
            Settings.MAX_BRACKETS_PER_EVENT,
        ),
        MARKET_MIN_CLOSE_DAYS=_read_env_int_optional(
            "MARKET_MIN_CLOSE_DAYS", Settings.MARKET_MIN_CLOSE_DAYS
        ),
        MARKET_MAX_CLOSE_DAYS=_read_env_int_optional(
            "MARKET_MAX_CLOSE_DAYS", Settings.MARKET_MAX_CLOSE_DAYS
        ),
        XAI_API_KEY=_read_env_str("XAI_API_KEY", Settings.XAI_API_KEY),
        GROK_MODEL=normalized_model_initial,
        GROK_MODEL_DEEP=normalized_model_deep,
        SEARCH_LOOKBACK_HOURS=_read_env_int(
            "SEARCH_LOOKBACK_HOURS", Settings.SEARCH_LOOKBACK_HOURS
        ),
        SEARCH_ALLOWED_DOMAINS=_read_env_csv(
            "SEARCH_ALLOWED_DOMAINS", Settings.SEARCH_ALLOWED_DOMAINS
        ),
        SEARCH_ALLOWED_X_HANDLES=_read_env_csv(
            "SEARCH_ALLOWED_X_HANDLES", Settings.SEARCH_ALLOWED_X_HANDLES
        ),
        MULTIMEDIA_CONFIDENCE_THRESHOLD=_read_env_float_pair(
            "MULTIMEDIA_CONFIDENCE_THRESHOLD",
            Settings.MULTIMEDIA_CONFIDENCE_THRESHOLD,
        ),
        SEARCH_LOOKBACK_SHORT_HOURS=_read_env_int(
            "SEARCH_LOOKBACK_SHORT_HOURS",
            Settings.SEARCH_LOOKBACK_SHORT_HOURS,
        ),
        SEARCH_LOOKBACK_MEDIUM_HOURS=_read_env_int(
            "SEARCH_LOOKBACK_MEDIUM_HOURS",
            Settings.SEARCH_LOOKBACK_MEDIUM_HOURS,
        ),
        SEARCH_LOOKBACK_LONG_HOURS=_read_env_int(
            "SEARCH_LOOKBACK_LONG_HOURS",
            Settings.SEARCH_LOOKBACK_LONG_HOURS,
        ),
        SPORTS_ALLOWED_DOMAINS=_read_env_csv(
            "SPORTS_ALLOWED_DOMAINS", Settings.SPORTS_ALLOWED_DOMAINS
        ),
        SPORTS_ALLOWED_X_HANDLES=_read_env_csv(
            "SPORTS_ALLOWED_X_HANDLES", Settings.SPORTS_ALLOWED_X_HANDLES
        ),
        CRYPTO_ALLOWED_DOMAINS=_read_env_csv(
            "CRYPTO_ALLOWED_DOMAINS", Settings.CRYPTO_ALLOWED_DOMAINS
        ),
        CRYPTO_ALLOWED_X_HANDLES=_read_env_csv(
            "CRYPTO_ALLOWED_X_HANDLES", Settings.CRYPTO_ALLOWED_X_HANDLES
        ),
        POLITICS_ALLOWED_DOMAINS=_read_env_csv(
            "POLITICS_ALLOWED_DOMAINS", Settings.POLITICS_ALLOWED_DOMAINS
        ),
        POLITICS_ALLOWED_X_HANDLES=_read_env_csv(
            "POLITICS_ALLOWED_X_HANDLES", Settings.POLITICS_ALLOWED_X_HANDLES
        ),
        WEATHER_ALLOWED_DOMAINS=_read_env_csv(
            "WEATHER_ALLOWED_DOMAINS", Settings.WEATHER_ALLOWED_DOMAINS
        ),
        WEATHER_ALLOWED_X_HANDLES=_read_env_csv(
            "WEATHER_ALLOWED_X_HANDLES", Settings.WEATHER_ALLOWED_X_HANDLES
        ),
        GENERIC_ALLOWED_DOMAINS=_read_env_csv(
            "GENERIC_ALLOWED_DOMAINS", Settings.GENERIC_ALLOWED_DOMAINS
        ),
        GENERIC_ALLOWED_X_HANDLES=_read_env_csv(
            "GENERIC_ALLOWED_X_HANDLES", Settings.GENERIC_ALLOWED_X_HANDLES
        ),
        KALSHI_API_BASE_URL=_read_env_str(
            "KALSHI_API_BASE_URL", Settings.KALSHI_API_BASE_URL
        ),
        KALSHI_API_KEY_ID=_read_env_str(
            "KALSHI_API_KEY_ID", Settings.KALSHI_API_KEY_ID
        ),
        KALSHI_PRIVATE_KEY_PATH=_read_env_str(
            "KALSHI_PRIVATE_KEY_PATH", Settings.KALSHI_PRIVATE_KEY_PATH
        ),
        KALSHI_SERVER_SIDE_FILTERS_ENABLED=_read_env_bool(
            "KALSHI_SERVER_SIDE_FILTERS_ENABLED",
            Settings.KALSHI_SERVER_SIDE_FILTERS_ENABLED,
        ),
        DRY_RUN=_read_env_bool("DRY_RUN", Settings.DRY_RUN),
        PRE_ORDER_MARKET_REFRESH=_read_env_bool(
            "PRE_ORDER_MARKET_REFRESH", Settings.PRE_ORDER_MARKET_REFRESH
        ),
        ORDERBOOK_PRECHECK_ENABLED=_read_env_bool(
            "ORDERBOOK_PRECHECK_ENABLED", Settings.ORDERBOOK_PRECHECK_ENABLED
        ),
        ORDERBOOK_PRECHECK_MIN_CONFIDENCE=_read_env_float(
            "ORDERBOOK_PRECHECK_MIN_CONFIDENCE",
            Settings.ORDERBOOK_PRECHECK_MIN_CONFIDENCE,
        ),
        ORDER_PRICE_IMPROVEMENT_CENTS=_read_env_int(
            "ORDER_PRICE_IMPROVEMENT_CENTS",
            Settings.ORDER_PRICE_IMPROVEMENT_CENTS,
        ),
        CALIBRATION_MODE_ENABLED=_read_env_bool(
            "CALIBRATION_MODE_ENABLED", Settings.CALIBRATION_MODE_ENABLED
        ),
        CALIBRATION_MIN_SAMPLES=_read_env_int(
            "CALIBRATION_MIN_SAMPLES", Settings.CALIBRATION_MIN_SAMPLES
        ),
        STATE_DB_PATH=_read_env_str(
            "STATE_DB_PATH", Settings.STATE_DB_PATH
        ),
        STATE_JSON_EXPORT_PATH=_read_env_str(
            "STATE_JSON_EXPORT_PATH", Settings.STATE_JSON_EXPORT_PATH
        ),
        EXPORT_STATE_JSON=_read_env_bool(
            "EXPORT_STATE_JSON", Settings.EXPORT_STATE_JSON
        ),
        REANALYSIS_COOLDOWN_HOURS=_read_env_int(
            "REANALYSIS_COOLDOWN_HOURS",
            Settings.REANALYSIS_COOLDOWN_HOURS,
        ),
        URGENT_REANALYSIS_DAYS_BEFORE_CLOSE=_read_env_int(
            "URGENT_REANALYSIS_DAYS_BEFORE_CLOSE",
            Settings.URGENT_REANALYSIS_DAYS_BEFORE_CLOSE,
        ),
        URGENT_REANALYSIS_COOLDOWN_HOURS=_read_env_int(
            "URGENT_REANALYSIS_COOLDOWN_HOURS",
            Settings.URGENT_REANALYSIS_COOLDOWN_HOURS,
        ),
        PARALLEL_ANALYSIS_ENABLED=_read_env_bool(
            "PARALLEL_ANALYSIS_ENABLED", Settings.PARALLEL_ANALYSIS_ENABLED
        ),
        ANALYSIS_MAX_WORKERS=_read_env_int(
            "ANALYSIS_MAX_WORKERS", Settings.ANALYSIS_MAX_WORKERS
        ),
        MAX_MARKETS_PER_CYCLE=_read_env_int(
            "MAX_MARKETS_PER_CYCLE", Settings.MAX_MARKETS_PER_CYCLE
        ),
        RESOLUTION_SYNC_INTERVAL_CYCLES=_read_env_int(
            "RESOLUTION_SYNC_INTERVAL_CYCLES",
            Settings.RESOLUTION_SYNC_INTERVAL_CYCLES,
        ),
        MAX_POSITION_PER_MARKET_USDC=_read_env_float(
            "MAX_POSITION_PER_MARKET_USDC",
            Settings.MAX_POSITION_PER_MARKET_USDC,
        ),
        MAX_POSITION_PCT_OF_BANKROLL=_read_env_float(
            "MAX_POSITION_PCT_OF_BANKROLL",
            Settings.MAX_POSITION_PCT_OF_BANKROLL,
        ),
        MIN_CONFIDENCE_INCREASE_FOR_ADD=_read_env_float(
            "MIN_CONFIDENCE_INCREASE_FOR_ADD",
            Settings.MIN_CONFIDENCE_INCREASE_FOR_ADD,
        ),
        HIGH_CONFIDENCE_POSITION_OVERRIDE=_read_env_float(
            "HIGH_CONFIDENCE_POSITION_OVERRIDE",
            Settings.HIGH_CONFIDENCE_POSITION_OVERRIDE,
        ),
        OPPOSITE_OUTCOME_STRATEGY=_read_env_str(
            "OPPOSITE_OUTCOME_STRATEGY",
            Settings.OPPOSITE_OUTCOME_STRATEGY,
        ),
        SCORE_GATE_MODE=_read_env_str(
            "SCORE_GATE_MODE",
            Settings.SCORE_GATE_MODE,
        ),
        SCORE_GATE_THRESHOLD=_read_env_float(
            "SCORE_GATE_THRESHOLD",
            Settings.SCORE_GATE_THRESHOLD,
        ),
        BAYESIAN_ENABLED=_read_env_bool(
            "BAYESIAN_ENABLED",
            Settings.BAYESIAN_ENABLED,
        ),
        BAYESIAN_SKIP_STALE_UPDATES=_read_env_bool(
            "BAYESIAN_SKIP_STALE_UPDATES",
            Settings.BAYESIAN_SKIP_STALE_UPDATES,
        ),
        BAYESIAN_PRIOR_DEFAULT=_read_env_float(
            "BAYESIAN_PRIOR_DEFAULT",
            Settings.BAYESIAN_PRIOR_DEFAULT,
        ),
        BAYESIAN_MIN_UPDATES_FOR_TRADE=_read_env_int(
            "BAYESIAN_MIN_UPDATES",
            _read_env_int(
                "BAYESIAN_MIN_UPDATES_FOR_TRADE",
                Settings.BAYESIAN_MIN_UPDATES_FOR_TRADE,
            ),
        ),
        LMSR_ENABLED=_read_env_bool(
            "LMSR_ENABLED",
            Settings.LMSR_ENABLED,
        ),
        LMSR_LIQUIDITY_PARAM_B=_read_env_float(
            "LMSR_LIQUIDITY_PARAM_B",
            Settings.LMSR_LIQUIDITY_PARAM_B,
        ),
        LMSR_MIN_INEFFICIENCY=_read_env_float(
            "LMSR_MIN_INEFFICIENCY",
            Settings.LMSR_MIN_INEFFICIENCY,
        ),
        KELLY_SIZING_ENABLED=_read_env_bool(
            "KELLY_SIZING_ENABLED",
            Settings.KELLY_SIZING_ENABLED,
        ),
        KELLY_FRACTION_DEFAULT=_read_env_float(
            "KELLY_FRACTION_DEFAULT",
            Settings.KELLY_FRACTION_DEFAULT,
        ),
        KELLY_FRACTION_SHORT_HORIZON_HOURS=_read_env_int(
            "KELLY_FRACTION_SHORT_HORIZON_HOURS",
            Settings.KELLY_FRACTION_SHORT_HORIZON_HOURS,
        ),
        KELLY_FRACTION_SHORT_HORIZON=_read_env_float(
            "KELLY_FRACTION_SHORT_HORIZON",
            Settings.KELLY_FRACTION_SHORT_HORIZON,
        ),
        KELLY_MIN_BET_POLICY=_read_env_str(
            "KELLY_MIN_BET_POLICY",
            Settings.KELLY_MIN_BET_POLICY,
        ),
        KELLY_MIN_BANKROLL_USDC=_read_env_float(
            "KELLY_MIN_BANKROLL_USDC",
            Settings.KELLY_MIN_BANKROLL_USDC,
        ),
        FLIP_GUARD_ENABLED=_read_env_bool(
            "FLIP_GUARD_ENABLED",
            Settings.FLIP_GUARD_ENABLED,
        ),
        FLIP_GUARD_MIN_ABS_CONFIDENCE=_read_env_float(
            "FLIP_GUARD_MIN_ABS_CONFIDENCE",
            Settings.FLIP_GUARD_MIN_ABS_CONFIDENCE,
        ),
        FLIP_GUARD_MIN_CONF_GAIN=_read_env_float(
            "FLIP_GUARD_MIN_CONF_GAIN",
            Settings.FLIP_GUARD_MIN_CONF_GAIN,
        ),
        FLIP_GUARD_MIN_EDGE_GAIN=_read_env_float(
            "FLIP_GUARD_MIN_EDGE_GAIN",
            Settings.FLIP_GUARD_MIN_EDGE_GAIN,
        ),
        FLIP_GUARD_MIN_EVIDENCE_QUALITY=_read_env_float(
            "FLIP_GUARD_MIN_EVIDENCE_QUALITY",
            Settings.FLIP_GUARD_MIN_EVIDENCE_QUALITY,
        ),
        FLIP_CIRCUIT_BREAKER_ENABLED=_read_env_bool(
            "FLIP_CIRCUIT_BREAKER_ENABLED",
            Settings.FLIP_CIRCUIT_BREAKER_ENABLED,
        ),
        FLIP_CIRCUIT_BREAKER_MAX_FLIPS=_read_env_int(
            "FLIP_CIRCUIT_BREAKER_MAX_FLIPS",
            Settings.FLIP_CIRCUIT_BREAKER_MAX_FLIPS,
        ),
        LOG_LEVEL=_read_env_str("LOG_LEVEL", Settings.LOG_LEVEL),
        LOG_FILE_LEVEL=_read_env_str("LOG_FILE_LEVEL", Settings.LOG_FILE_LEVEL),
        LOG_DIR=_read_env_str("LOG_DIR", Settings.LOG_DIR),
        ENABLE_FILE_LOGGING=_read_env_bool(
            "ENABLE_FILE_LOGGING", Settings.ENABLE_FILE_LOGGING
        ),
        ENABLE_JSON_LOGGING=_read_env_bool(
            "ENABLE_JSON_LOGGING", Settings.ENABLE_JSON_LOGGING
        ),
        ENABLE_COLORED_LOGGING=_read_env_bool(
            "ENABLE_COLORED_LOGGING", Settings.ENABLE_COLORED_LOGGING
        ),
    )
    strategy = settings.OPPOSITE_OUTCOME_STRATEGY.strip().lower()
    if strategy not in {"block", "hedge"}:
        strategy = Settings.OPPOSITE_OUTCOME_STRATEGY
    score_mode = settings.SCORE_GATE_MODE.strip().lower()
    if score_mode not in {"off", "shadow", "active"}:
        score_mode = Settings.SCORE_GATE_MODE
    kelly_min_bet_policy = settings.KELLY_MIN_BET_POLICY.strip().lower()
    if kelly_min_bet_policy not in {"skip", "floor", "fallback_edge_scaling"}:
        kelly_min_bet_policy = Settings.KELLY_MIN_BET_POLICY

    settings = Settings(
        **{
            **settings.__dict__,
            "OPPOSITE_OUTCOME_STRATEGY": strategy,
            "SCORE_GATE_MODE": score_mode,
            "KELLY_MIN_BET_POLICY": kelly_min_bet_policy,
        }
    )

    _validate_required(settings)
    return settings


def _required_env_vars(settings: Settings) -> tuple[str, ...]:
    return tuple(BASE_REQUIRED_ENV_VARS)


def _validate_required(
    settings: Settings, required: Iterable[str] | None = None
) -> None:
    required_vars = tuple(required) if required is not None else _required_env_vars(settings)
    missing = [name for name in required_vars if not getattr(settings, name)]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Missing required environment variables: {names}")


def build_search_config(settings: Settings) -> SearchConfig:
    """Build SearchConfig from settings to keep wiring centralized."""
    from datetime import datetime, timedelta, timezone

    search_now = datetime.now(timezone.utc)
    return SearchConfig(
        from_date=search_now - timedelta(hours=settings.SEARCH_LOOKBACK_HOURS),
        to_date=search_now,
        allowed_domains=list(settings.SEARCH_ALLOWED_DOMAINS),
        allowed_x_handles=list(settings.SEARCH_ALLOWED_X_HANDLES),
        multimedia_confidence_range=settings.MULTIMEDIA_CONFIDENCE_THRESHOLD,
    )


@dataclass
class SearchConfig:
    from_date: "datetime | None" = None
    to_date: "datetime | None" = None
    allowed_domains: list[str] = field(default_factory=list)
    allowed_x_handles: list[str] = field(default_factory=list)
    enable_multimedia: bool = False
    multimedia_confidence_range: tuple[float, float] = (0.55, 0.75)
    profile_name: str = "generic"
    lookback_hours: int | None = None
