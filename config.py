from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Risk controls - Conservative defaults for value betting
    MIN_BET_USDC: float = 5.0
    MAX_BET_USDC: float = 50.0
    MIN_CONFIDENCE: float = 0.65  # Raised from 0.60 - require meaningful edge
    SLIPPAGE_CONFIDENCE_THRESHOLD: float = 0.70
    SLIPPAGE_PCT: float = 0.02
    MIN_LIQUIDITY_USDC: float = 100.0
    POLL_INTERVAL_SEC: int = 300

    # Edge gating / sizing
    MIN_EDGE: float = 0.05
    LOW_PRICE_THRESHOLD: float = 0.58
    HIGH_PRICE_THRESHOLD: float = 0.65
    LOW_PRICE_MIN_EDGE: float = 0.10
    EDGE_SCALING_RANGE: float = 0.15
    LOW_PRICE_BET_PENALTY: float = 0.50
    REQUIRE_IMPLIED_PRICE: bool = True
    
    # Confidence caps to prevent overconfidence on high-variance events
    MAX_SPORTS_CONFIDENCE: float = 0.80  # Cap sports bets at 80% confidence
    MAX_ESPORTS_CONFIDENCE: float = 0.75  # Cap esports at 75%

    # Filtering
    MARKET_CATEGORIES_ALLOWLIST: tuple[str, ...] = ()
    MARKET_CATEGORIES_BLOCKLIST: tuple[str, ...] = ()
    # Date range filtering: only consider markets closing within this window (days from now)
    # Set to 0 or None to disable the filter
    MARKET_MIN_CLOSE_DAYS: int | None = None  # Minimum days until close (skip markets closing too soon)
    MARKET_MAX_CLOSE_DAYS: int | None = None  # Maximum days until close (skip markets closing too far out)

    # xAI Grok
    XAI_API_KEY: str = ""
    GROK_MODEL: str = "grok-3"
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

    # PredictBase
    PREDICTBASE_API_BASE_URL: str = "https://api.predictbase.app"
    PREDICTBASE_API_KEY: str | None = None
    PREDICTBASE_API_KEY_HEADER: str = "x-api-key"
    PREDICTBASE_API_KEY_PREFIX: str = ""

    # Web3
    ALCHEMY_RPC_URL: str = ""
    WALLET_PRIVATE_KEY: str = ""
    USDC_TOKEN_ADDRESS: str = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
    PREDICTBASE_CONTRACT_ADDRESS: str = ""
    USDC_DECIMALS: int = 6
    CHAIN_ID: int | None = 8453

    # Execution
    DRY_RUN: bool = True
    AUTO_APPROVE_USDC: bool = False
    EXECUTE_ONCHAIN: bool = False

    # State management
    STATE_DB_PATH: str = "data/market_state.db"
    STATE_JSON_EXPORT_PATH: str = "data/market_state.json"
    EXPORT_STATE_JSON: bool = True

    # Re-analysis controls
    REANALYSIS_COOLDOWN_HOURS: int = 6
    URGENT_REANALYSIS_DAYS_BEFORE_CLOSE: int = 1

    # Resolution tracking
    RESOLUTION_SYNC_INTERVAL_CYCLES: int = 3

    # Position limits
    MAX_POSITION_PER_MARKET_USDC: float = 200.0
    MIN_CONFIDENCE_INCREASE_FOR_ADD: float = 0.10
    HIGH_CONFIDENCE_POSITION_OVERRIDE: float = 0.85  # Allow adding to position if conf >= this
    OPPOSITE_OUTCOME_STRATEGY: str = "block"  # block|hedge

    # Score gate (phase A/B can run in shadow mode)
    SCORE_GATE_MODE: str = "shadow"  # off|shadow|active
    SCORE_GATE_THRESHOLD: float = 0.08

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE_LEVEL: str = "DEBUG"
    LOG_DIR: str = "logs"
    ENABLE_FILE_LOGGING: bool = True
    ENABLE_JSON_LOGGING: bool = True
    ENABLE_COLORED_LOGGING: bool = True


BASE_REQUIRED_ENV_VARS = (
    "XAI_API_KEY",
    "WALLET_PRIVATE_KEY",
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
    settings = Settings(
        MIN_BET_USDC=_read_env_float("MIN_BET_USDC", Settings.MIN_BET_USDC),
        MAX_BET_USDC=_read_env_float("MAX_BET_USDC", Settings.MAX_BET_USDC),
        MIN_CONFIDENCE=_read_env_float("MIN_CONFIDENCE", Settings.MIN_CONFIDENCE),
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
        EDGE_SCALING_RANGE=_read_env_float(
            "EDGE_SCALING_RANGE", Settings.EDGE_SCALING_RANGE
        ),
        LOW_PRICE_BET_PENALTY=_read_env_float(
            "LOW_PRICE_BET_PENALTY", Settings.LOW_PRICE_BET_PENALTY
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
        MARKET_MIN_CLOSE_DAYS=_read_env_int_optional(
            "MARKET_MIN_CLOSE_DAYS", Settings.MARKET_MIN_CLOSE_DAYS
        ),
        MARKET_MAX_CLOSE_DAYS=_read_env_int_optional(
            "MARKET_MAX_CLOSE_DAYS", Settings.MARKET_MAX_CLOSE_DAYS
        ),
        XAI_API_KEY=_read_env_str("XAI_API_KEY", Settings.XAI_API_KEY),
        GROK_MODEL=_read_env_str("GROK_MODEL", Settings.GROK_MODEL),
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
        GENERIC_ALLOWED_DOMAINS=_read_env_csv(
            "GENERIC_ALLOWED_DOMAINS", Settings.GENERIC_ALLOWED_DOMAINS
        ),
        GENERIC_ALLOWED_X_HANDLES=_read_env_csv(
            "GENERIC_ALLOWED_X_HANDLES", Settings.GENERIC_ALLOWED_X_HANDLES
        ),
        PREDICTBASE_API_BASE_URL=_read_env_str(
            "PREDICTBASE_API_BASE_URL", Settings.PREDICTBASE_API_BASE_URL
        ),
        PREDICTBASE_API_KEY=os.getenv("PREDICTBASE_API_KEY"),
        PREDICTBASE_API_KEY_HEADER=_read_env_str(
            "PREDICTBASE_API_KEY_HEADER", Settings.PREDICTBASE_API_KEY_HEADER
        ),
        PREDICTBASE_API_KEY_PREFIX=_read_env_str(
            "PREDICTBASE_API_KEY_PREFIX", Settings.PREDICTBASE_API_KEY_PREFIX
        ),
        ALCHEMY_RPC_URL=_read_env_str(
            "ALCHEMY_RPC_URL", Settings.ALCHEMY_RPC_URL
        ),
        WALLET_PRIVATE_KEY=_read_env_str(
            "WALLET_PRIVATE_KEY", Settings.WALLET_PRIVATE_KEY
        ),
        USDC_TOKEN_ADDRESS=_read_env_str(
            "USDC_TOKEN_ADDRESS", Settings.USDC_TOKEN_ADDRESS
        ),
        PREDICTBASE_CONTRACT_ADDRESS=_read_env_str(
            "PREDICTBASE_CONTRACT_ADDRESS", Settings.PREDICTBASE_CONTRACT_ADDRESS
        ),
        USDC_DECIMALS=_read_env_int("USDC_DECIMALS", Settings.USDC_DECIMALS),
        CHAIN_ID=(
            int(os.getenv("CHAIN_ID"))
            if os.getenv("CHAIN_ID")
            else Settings.CHAIN_ID
        ),
        DRY_RUN=_read_env_bool("DRY_RUN", Settings.DRY_RUN),
        AUTO_APPROVE_USDC=_read_env_bool(
            "AUTO_APPROVE_USDC", Settings.AUTO_APPROVE_USDC
        ),
        EXECUTE_ONCHAIN=_read_env_bool(
            "EXECUTE_ONCHAIN", Settings.EXECUTE_ONCHAIN
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
        RESOLUTION_SYNC_INTERVAL_CYCLES=_read_env_int(
            "RESOLUTION_SYNC_INTERVAL_CYCLES",
            Settings.RESOLUTION_SYNC_INTERVAL_CYCLES,
        ),
        MAX_POSITION_PER_MARKET_USDC=_read_env_float(
            "MAX_POSITION_PER_MARKET_USDC",
            Settings.MAX_POSITION_PER_MARKET_USDC,
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

    settings = Settings(
        **{
            **settings.__dict__,
            "OPPOSITE_OUTCOME_STRATEGY": strategy,
            "SCORE_GATE_MODE": score_mode,
        }
    )

    _validate_required(settings)
    return settings


def _required_env_vars(settings: Settings) -> tuple[str, ...]:
    required = list(BASE_REQUIRED_ENV_VARS)
    if settings.EXECUTE_ONCHAIN or settings.AUTO_APPROVE_USDC:
        required.append("ALCHEMY_RPC_URL")
    if settings.AUTO_APPROVE_USDC:
        required.append("PREDICTBASE_CONTRACT_ADDRESS")
    return tuple(required)


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
