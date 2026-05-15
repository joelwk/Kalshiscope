from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import re
from typing import Any
from urllib.parse import urlparse

from bayesian_engine import (
    BayesianState,
    initial_state,
    log_likelihood_from_ratio,
    posterior_from_state,
)
from calibration_gates import (
    GateTier,
    evaluate_market,
    evaluate_market_tiered,
    evaluate_short_prefix_penalty,
    load_family_stats,
    load_short_prefix_stats,
    load_ticker_prefix_stats,
)
from participation import (
    HistoricalGateResult,
    ParticipationDecision,
    ParticipationTier,
    TimeoutState,
    bayesian_shrunk_pnl,
    classify_participation,
)
from calibration import (
    build_counterfactual_flags,
    compute_adaptive_thresholds,
    historical_confidence_shrink,
)
from config import SearchConfig, Settings, load_settings
from grok_client import GrokClient
from kelly import kelly_bet_pct, kelly_fraction
from lmsr import (
    infer_quantities_from_prices,
    inefficiency_signal as lmsr_inefficiency_signal,
    lmsr_prices,
    trade_cost as lmsr_trade_cost,
)
from logging_config import (
    get_logger,
    log_trade_decision as _base_log_trade_decision,
    set_correlation_id,
    setup_logging,
)
from market_scheduler import MarketScheduler, remaining_reanalysis_cooldown_seconds
from market_state import MarketStateManager
from models import (
    InsufficientBalanceError,
    Market,
    MarketClosedError,
    MarketOutcome,
    MarketState,
    OrderRequest,
    Position,
    TradeDecision,
)
from kalshi_client import KalshiClient
from refinement import RefinementStrategy
from research_profiles import build_market_search_config, market_category_flags, market_family
from bootstrap_checks import BootstrapError, run_bootstrap_checks
from score_engine import calibrate_confidence, compute_final_score, score_breakdown_explanation
from xai_provider import XAIProvider

try:
    import certifi
except Exception:  # pragma: no cover - optional runtime diagnostic
    certifi = None

logger = get_logger("predictbot")

_MATCHUP_SEPARATOR = re.compile(r"\s+(?:vs\.?|v\.?|at)\s+|\s*@\s*", re.IGNORECASE)
_OPEN_MARKET_STATUS = {"", "0", "open", "active", "trading"}
_RESOLVED_MARKET_STATUS = {
    "1",
    "2",
    "3",
    "closed",
    "resolved",
    "settled",
    "finalized",
    "ended",
    "cancelled",
    "canceled",
    "inactive",
}
_ADAPTIVE_SLEEP_CAP_SECONDS = 1800
_ORDERBOOK_SPREAD_CUTOFF_DEFAULT = 0.08
_STALE_REFRESH_RETRY_DELAY_SECONDS = 1.0
_STALE_REFRESH_LENIENT_AGE_MULTIPLIER = 2.5
_MAX_CONFIDENCE = 1.0
_AGGRESSIVE_CONFIDENCE_SHRINKAGE_FACTOR = 0.30
_INDEX_MARKET_PREFIXES = ("KXNASDAQ100U-", "KXINXU-")
_COMMODITY_MARKET_TOKENS = (
    "GOLD",
    "SILVER",
    "WTI",
    "NATGAS",
    "COPPER",
    "CORN",
    "SOY",
    "WHEAT",
    "AAA",
)
_SPORTS_ENTITY_STOPWORDS = frozenset(
    {
        "above",
        "after",
        "against",
        "base",
        "bases",
        "below",
        "first",
        "game",
        "have",
        "hits",
        "home",
        "inning",
        "innings",
        "matchup",
        "over",
        "period",
        "player",
        "points",
        "prop",
        "runs",
        "score",
        "sports",
        "team",
        "than",
        "total",
        "under",
        "will",
        "with",
    }
)
_HISTORICAL_WIN_RATE_BY_BUCKET = {
    0.7: 0.43,
    0.8: 0.50,
    0.9: 0.52,
    1.0: 0.47,
}
_TICKER_DATE_PATTERN = re.compile(
    r"-(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})(?:-|$)",
    re.IGNORECASE,
)
_MONTH_ABBREVIATIONS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}
_KELLY_MIN_BET_POLICY_SKIP = "skip"
_KELLY_MIN_BET_POLICY_FLOOR = "floor"
_KELLY_MIN_BET_POLICY_FALLBACK_EDGE = "fallback_edge_scaling"
_RE_VALIDATED_PREFIX = re.compile(r"^\[Validated\b[^\]]*\]\s*")
_XAI_RETRIABLE_ERROR_MARKERS = (
    "statuscode.internal",
    "internal server error",
    "service temporarily unavailable",
    "temporarily unavailable",
)
_XAI_QUOTA_EXHAUSTED_MARKERS = (
    "resource_exhausted",
    "available credits",
    "monthly spending limit",
    "reached its monthly spending limit",
)
_WEATHER_BIN_TICKER_PATTERN = re.compile(r"-B\d", re.IGNORECASE)
_PRE_ANALYSIS_HARD_REJECTION_TERMINAL_OUTCOMES = {
    "no_trade_recommended",
    "evidence_quality_below_min",
    "confidence_below_min",
    "abstain_low_evidence",
}
# In-memory cap on the per-cycle research-queue capture log is governed by
# settings.RESEARCH_QUEUE_CYCLE_LOG_MAXLEN (default 200). The DB table
# research_queue_entries persists EVERY entry regardless of that cap; the
# in-memory deque only bounds the per-cycle "Research queue captured N blocked
# opportunities" log line and the per-cycle receipt summary.
_RESEARCH_QUEUE_EVIDENCE_GAP_MAX = 0.08
_RESEARCH_QUEUE_EDGE_GAP_MAX = 0.08
_SCORE_GATE_ALWAYS_BLOCK_REASONS = frozenset(
    {
        "non_positive_market_edge",
        "low_evidence_quality",
        "low_information_penalty",
        "hallucinated_edge",
        "extreme_edge_learning_queue",
        "extreme_market_edge_penalty",
        "ambiguous_resolution_penalty",
    }
)
_SCORE_GATE_SOURCE_BLOCK_REASONS = frozenset(
    {
        "fallback_edge_penalty",
        "fallback_without_external_odds",
        "proxy_evidence_penalty",
        "no_external_odds_penalty",
    }
)
_PRE_ANALYSIS_RESEARCH_ONLY_SCORE = 0.0
_PRE_ANALYSIS_SOURCE_DIFFICULTY_PENALTIES = {
    "crypto": 0.08,
    "generic": 0.06,
    "politics": 0.08,
    "speech": 0.12,
    "weather": 0.10,
}
_PRE_ANALYSIS_AMBIGUOUS_MARKET_PENALTY = 0.08
_PRE_ANALYSIS_PROFITABLE_HISTORY_BONUS = 0.06
_PRE_ANALYSIS_POSITIVE_FAMILY_VOLUME_BONUS = 0.03
_PRE_ANALYSIS_POSITIVE_FAMILY_PNL_BONUS = 0.02
_PRE_ANALYSIS_NEGATIVE_PREFIX_PENALTY = 0.08
_AMBIGUOUS_MARKET_TOKENS = (
    "attend",
    "mention",
    "say ",
    "speech",
    "stream",
    "album",
    "views",
)
_GENERIC_SUBFAMILY_KEYWORDS = (
    ("commodity", ("oil", "wti", "brent", "gas", "fuel", "jet fuel", "gold", "silver")),
    ("macro_release", ("jobless", "claims", "adp", "payroll", "cpi", "inflation", "gdp")),
    ("rates", ("fed", "rate", "yield", "treasury", "basis point")),
    ("index", ("nasdaq", "s&p", "sp500", "dow", "russell")),
    ("transport", ("tsa", "airport", "flight", "airline")),
)


def _normalize_outcome_key(outcome: str | None) -> str:
    return re.sub(r"\s+", " ", (outcome or "").strip()).lower()


def _is_retriable_xai_error(error_text: str | None) -> bool:
    normalized = (error_text or "").strip().lower()
    if not normalized:
        return False
    return any(marker in normalized for marker in _XAI_RETRIABLE_ERROR_MARKERS)


def _is_quota_exhausted_xai_error(error_text: str | None) -> bool:
    normalized = (error_text or "").strip().lower()
    if not normalized:
        return False
    return any(marker in normalized for marker in _XAI_QUOTA_EXHAUSTED_MARKERS)


@dataclass(frozen=True)
class _CryptoPreflightResult:
    should_skip: bool = False
    reason: str = ""
    spot_price: float | None = None
    threshold: float | None = None
    buffer_pct: float | None = None


def _crypto_threshold_preflight(
    market: Market,
    settings: Settings,
) -> _CryptoPreflightResult:
    """Lightweight preflight for crypto threshold markets (default OFF).

    When enabled, parses the threshold from the ticker and compares to
    a cached spot price to skip Grok when the buffer is too wide.
    Currently a stub — full implementation requires an HTTP call to
    CoinDesk/Coinbase for spot prices.
    """
    if not settings.CRYPTO_PREFLIGHT_ENABLED:
        return _CryptoPreflightResult()
    market_id = (market.id or "").upper()
    threshold_match = re.search(r"-T([\d.]+)$", market_id)
    if not threshold_match:
        return _CryptoPreflightResult()
    return _CryptoPreflightResult()


def _build_reasoning_hash(decision: TradeDecision) -> str:
    reasoning_text = _RE_VALIDATED_PREFIX.sub("", (decision.reasoning or "").strip())[:200]
    outcome_text = (decision.outcome or "").strip().lower()
    rounded_confidence = round(float(decision.confidence), 2)
    payload = f"{outcome_text}|{rounded_confidence:.2f}|{reasoning_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _outcomes_match(left: str | None, right: str | None) -> bool:
    left_key = _normalize_outcome_key(left)
    right_key = _normalize_outcome_key(right)
    if not left_key or not right_key:
        return False
    return left_key == right_key


def _status_indicates_closed(status: object) -> bool:
    if status is None:
        return False
    status_text = str(status).strip().lower()
    if status_text in _OPEN_MARKET_STATUS:
        return False
    if status_text in _RESOLVED_MARKET_STATUS:
        return True
    if status_text.lstrip("-").isdigit():
        try:
            return int(status_text) > 0
        except ValueError:
            return False
    return False


def _filter_markets(
    markets,
    min_liquidity,
    allowlist,
    blocklist,
    ticker_prefix_blocklist=(),
    min_close_days=None,
    max_close_days=None,
    stats: dict[str, int] | None = None,
    min_volume_24h: float = 0.0,
    min_open_interest: float = 0.0,
    extreme_yes_price_lower: float | None = None,
    extreme_yes_price_upper: float | None = None,
    min_tradeable_yes_price: float | None = None,
    max_tradeable_yes_price: float | None = None,
    skip_weather_bin_markets: bool = False,
    skip_crypto_bin_markets: bool = False,
    family_blocklist=(),
):
    """Filter markets based on liquidity, category, and close date constraints."""
    filtered = []
    skipped_liquidity = 0
    skipped_volume_24h = 0
    skipped_open_interest = 0
    activity_passed_by_open_interest = 0
    skipped_extreme_price = 0
    skipped_untradeable_price = 0
    skipped_allowlist = 0
    skipped_blocklist = 0
    skipped_family_blocklist = 0
    skipped_close_too_soon = 0
    skipped_close_too_far = 0
    skipped_closed_now = 0
    skipped_resolved = 0
    skipped_ticker_prefix_blocklist = 0
    skipped_weather_bin_markets = 0
    skipped_crypto_bin_markets = 0
    skipped_likely_resolved_by_ticker = 0

    now = datetime.now(timezone.utc)
    min_close_date = (
        now + timedelta(days=min_close_days)
        if min_close_days is not None
        else None
    )
    max_close_date = (
        now + timedelta(days=max_close_days)
        if max_close_days is not None
        else None
    )

    for market in markets:
        close_time = market.close_time
        if close_time and close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=timezone.utc)
        # Liquidity can be sparse/noisy in some market payloads; use a conservative
        # proxy that falls back to open interest and recent volume signals.
        effective_liquidity = max(
            0.0,
            float(market.liquidity_usdc or 0.0),
            float(market.open_interest or 0.0),
            float(market.volume_24h or 0.0),
        )
        if effective_liquidity < min_liquidity:
            skipped_liquidity += 1
            continue
        effective_volume_24h = market.volume_24h if market.volume_24h is not None else 0.0
        effective_open_interest = (
            market.open_interest if market.open_interest is not None else 0.0
        )
        volume_threshold_enabled = min_volume_24h > 0.0
        open_interest_threshold_enabled = min_open_interest > 0.0
        meets_volume_threshold = (
            (not volume_threshold_enabled) or effective_volume_24h >= min_volume_24h
        )
        meets_open_interest_threshold = (
            open_interest_threshold_enabled and effective_open_interest >= min_open_interest
        )
        meets_activity_threshold = True
        if volume_threshold_enabled and open_interest_threshold_enabled:
            meets_activity_threshold = meets_volume_threshold or meets_open_interest_threshold
        elif volume_threshold_enabled:
            meets_activity_threshold = meets_volume_threshold
        elif open_interest_threshold_enabled:
            meets_activity_threshold = meets_open_interest_threshold
        if not meets_activity_threshold:
            skipped_volume_24h += 1
            skipped_open_interest += 1
            continue
        if not meets_volume_threshold and meets_open_interest_threshold:
            activity_passed_by_open_interest += 1
        yes_price = _get_outcome_entry_price(market, "YES")
        if yes_price is not None:
            if (
                min_tradeable_yes_price is not None
                and yes_price <= min_tradeable_yes_price
            ) or (
                max_tradeable_yes_price is not None
                and yes_price >= max_tradeable_yes_price
            ):
                skipped_untradeable_price += 1
                continue
            if (
                extreme_yes_price_lower is not None
                and yes_price <= extreme_yes_price_lower
            ) or (
                extreme_yes_price_upper is not None
                and yes_price >= extreme_yes_price_upper
            ):
                skipped_extreme_price += 1
                continue
        if allowlist and (market.category not in allowlist):
            skipped_allowlist += 1
            continue
        if blocklist:
            market_category = (market.category or "").strip()
            family = market_family(market)
            if market_category in blocklist or family in blocklist:
                if not market_category:
                    logger.warning(
                        "Market blocked via inferred family because category is missing: market=%s family=%s",
                        market.id,
                        family,
                        data={
                            "market_id": market.id,
                            "family": family,
                        },
                    )
                skipped_blocklist += 1
                continue
        if family_blocklist:
            family = market_family(market)
            if family in family_blocklist:
                skipped_family_blocklist += 1
                continue
        if ticker_prefix_blocklist:
            market_id = (market.id or "").upper()
            if any(market_id.startswith(prefix.upper()) for prefix in ticker_prefix_blocklist):
                skipped_ticker_prefix_blocklist += 1
                continue
        market_id_upper = (market.id or "").upper()
        if skip_weather_bin_markets and _is_weather_bin_market(market_id_upper):
            skipped_weather_bin_markets += 1
            continue
        if skip_crypto_bin_markets and _is_crypto_bin_market(market_id_upper):
            skipped_crypto_bin_markets += 1
            continue
        if _is_likely_resolved_by_ticker_date(market, now):
            skipped_likely_resolved_by_ticker += 1
            continue
        if _is_market_resolved_or_closed(market):
            skipped_resolved += 1
            continue
        if min_close_date and close_time:
            if close_time < min_close_date:
                skipped_close_too_soon += 1
                continue
        if max_close_date and close_time:
            if close_time > max_close_date:
                skipped_close_too_far += 1
                continue
        if close_time and close_time <= now:
            skipped_closed_now += 1
            continue
        filtered.append(market)

    logger.debug(
        "Market filtering complete: kept=%d, skipped_liquidity=%d, skipped_volume_24h=%d, "
        "skipped_untradeable_price=%d, skipped_extreme_price=%d, skipped_allowlist=%d, "
        "skipped_blocklist=%d, skipped_family_blocklist=%d, skipped_ticker_prefix_blocklist=%d, skipped_resolved=%d, skipped_close_too_soon=%d, "
        "skipped_close_too_far=%d, skipped_closed_now=%d, skipped_open_interest=%d, activity_passed_by_open_interest=%d, skipped_weather_bin_markets=%d, "
        "skipped_crypto_bin_markets=%d, skipped_likely_resolved_by_ticker=%d",
        len(filtered),
        skipped_liquidity,
        skipped_volume_24h,
        skipped_untradeable_price,
        skipped_extreme_price,
        skipped_allowlist,
        skipped_blocklist,
        skipped_family_blocklist,
        skipped_ticker_prefix_blocklist,
        skipped_resolved,
        skipped_close_too_soon,
        skipped_close_too_far,
        skipped_closed_now,
        skipped_open_interest,
        activity_passed_by_open_interest,
        skipped_weather_bin_markets,
        skipped_crypto_bin_markets,
        skipped_likely_resolved_by_ticker,
        data={
            "kept": len(filtered),
            "skipped_liquidity": skipped_liquidity,
            "skipped_volume_24h": skipped_volume_24h,
            "skipped_untradeable_price": skipped_untradeable_price,
            "skipped_extreme_price": skipped_extreme_price,
            "skipped_allowlist": skipped_allowlist,
            "skipped_blocklist": skipped_blocklist,
            "skipped_family_blocklist": skipped_family_blocklist,
            "skipped_ticker_prefix_blocklist": skipped_ticker_prefix_blocklist,
            "skipped_resolved": skipped_resolved,
            "skipped_close_too_soon": skipped_close_too_soon,
            "skipped_close_too_far": skipped_close_too_far,
            "skipped_closed_now": skipped_closed_now,
            "skipped_open_interest": skipped_open_interest,
            "activity_passed_by_open_interest": activity_passed_by_open_interest,
            "skipped_weather_bin_markets": skipped_weather_bin_markets,
            "skipped_crypto_bin_markets": skipped_crypto_bin_markets,
            "skipped_likely_resolved_by_ticker": skipped_likely_resolved_by_ticker,
        },
    )
    if stats is not None:
        stats.update(
            {
                "kept": len(filtered),
                "skipped_liquidity": skipped_liquidity,
                "skipped_volume_24h": skipped_volume_24h,
                "skipped_untradeable_price": skipped_untradeable_price,
                "skipped_extreme_price": skipped_extreme_price,
                "skipped_allowlist": skipped_allowlist,
                "skipped_blocklist": skipped_blocklist,
                "skipped_family_blocklist": skipped_family_blocklist,
                "skipped_ticker_prefix_blocklist": skipped_ticker_prefix_blocklist,
                "skipped_resolved": skipped_resolved,
                "skipped_close_too_soon": skipped_close_too_soon,
                "skipped_close_too_far": skipped_close_too_far,
                "skipped_closed_now": skipped_closed_now,
                "skipped_open_interest": skipped_open_interest,
                "activity_passed_by_open_interest": activity_passed_by_open_interest,
                "skipped_weather_bin_markets": skipped_weather_bin_markets,
                "skipped_crypto_bin_markets": skipped_crypto_bin_markets,
                "skipped_likely_resolved_by_ticker": skipped_likely_resolved_by_ticker,
            }
        )
    return filtered


def _log_filter_diagnostics(
    markets: list[Market],
    *,
    min_liquidity: float,
    min_volume_24h: float,
    min_open_interest: float,
    sample_size: int = 8,
) -> None:
    if not markets:
        return
    liquidity_pass = 0
    volume_pass = 0
    open_interest_pass = 0
    for market in markets:
        liquidity = market.liquidity_usdc if market.liquidity_usdc is not None else 0.0
        volume_24h = market.volume_24h if market.volume_24h is not None else 0.0
        open_interest = market.open_interest if market.open_interest is not None else 0.0
        if liquidity >= min_liquidity:
            liquidity_pass += 1
        if min_volume_24h <= 0.0 or volume_24h >= min_volume_24h:
            volume_pass += 1
        if min_open_interest <= 0.0 or open_interest >= min_open_interest:
            open_interest_pass += 1

    sample_payload: list[dict[str, Any]] = []
    for market in markets[: max(1, sample_size)]:
        sample_payload.append(
            {
                "market_id": market.id,
                "liquidity_usdc": market.liquidity_usdc,
                "volume_24h": market.volume_24h,
                "open_interest": market.open_interest,
                "yes_price": _get_outcome_entry_price(market, "YES"),
            }
        )

    logger.info(
        "Filter diagnostics: liquidity_pass=%d/%d volume_pass=%d/%d open_interest_pass=%d/%d thresholds(liquidity=%.2f volume_24h=%.2f open_interest=%.2f)",
        liquidity_pass,
        len(markets),
        volume_pass,
        len(markets),
        open_interest_pass,
        len(markets),
        min_liquidity,
        min_volume_24h,
        min_open_interest,
        data={
            "filter_diagnostics": {
                "total_markets": len(markets),
                "liquidity_pass": liquidity_pass,
                "volume_24h_pass": volume_pass,
                "open_interest_pass": open_interest_pass,
                "min_liquidity_usdc": min_liquidity,
                "min_volume_24h": min_volume_24h,
                "min_open_interest": min_open_interest,
                "sample_markets": sample_payload,
            }
        },
    )


def _is_weather_bin_market(market_id: str) -> bool:
    if not market_id:
        return False
    return bool(
        re.match(
            r"^KX(?:HIGH|LOW|LOWT|HIGHT|TEMP|PRECIP|SNOW|WIND)[A-Z0-9-]*-.*-B\d",
            market_id.upper(),
        )
    )


def _is_weather_market_by_ticker(market_id: str) -> bool:
    if not market_id:
        return False
    return bool(
        re.match(
            r"^KX(?:HIGH|LOW|LOWT|HIGHT|TEMP|PRECIP|SNOW|WIND)[A-Z0-9-]*-",
            market_id.upper(),
        )
    )


def _is_crypto_bin_market(market_id: str) -> bool:
    if not market_id:
        return False
    return bool(
        re.match(
            r"^KX(?:BTC|ETH|DOGE|SOL|BNB|XRP|HYPE)[A-Z0-9-]*-.*-B\d",
            market_id.upper(),
        )
    )


def _ticker_resolution_date(market_id: str) -> datetime | None:
    match = _TICKER_DATE_PATTERN.search((market_id or "").upper())
    if not match:
        return None
    year_token, month_token, day_token = match.groups()
    month = _MONTH_ABBREVIATIONS.get(month_token.upper())
    if month is None:
        return None
    try:
        return datetime(
            year=2000 + int(year_token),
            month=month,
            day=int(day_token),
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None


def _is_likely_resolved_by_ticker_date(market: Market, now: datetime) -> bool:
    resolution_date = _ticker_resolution_date(market.id or "")
    if resolution_date is None:
        return False
    return resolution_date.date() < now.date()


def _extract_order_cancel_reason(order_response: Any) -> str | None:
    if order_response is None or not isinstance(order_response, dict):
        return None
    reason_keys = (
        "cancel_reason",
        "cancellation_reason",
        "status_reason",
        "reject_reason",
        "reason",
        "error",
    )
    for key in reason_keys:
        value = order_response.get(key)
        if value:
            return str(value)
    nested_order = order_response.get("order")
    if isinstance(nested_order, dict):
        for key in reason_keys:
            value = nested_order.get(key)
            if value:
                return str(value)
    return None


def _extract_order_fill_count(order_response: Any) -> float | None:
    if order_response is None or not isinstance(order_response, dict):
        return None
    candidate_keys = ("fill_count_fp", "fill_count", "filled_count")
    for key in candidate_keys:
        value = order_response.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    nested_order = order_response.get("order")
    if isinstance(nested_order, dict):
        for key in candidate_keys:
            value = nested_order.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _collapse_event_ladders(
    markets: list[Market],
    *,
    ladder_collapse_threshold: int,
    max_brackets_per_event: int,
) -> list[Market]:
    """Collapse large event ladders to the most price-informative brackets."""
    if not markets:
        return markets
    if ladder_collapse_threshold <= 0 or max_brackets_per_event <= 0:
        return markets

    event_groups: dict[str, list[Market]] = {}
    for market in markets:
        event_ticker = (market.event_ticker or "").strip()
        if not event_ticker:
            continue
        event_groups.setdefault(event_ticker, []).append(market)

    collapsed_events = 0
    removed_markets = 0
    keep_ids: set[str] = set()
    for event_ticker, event_markets in event_groups.items():
        if len(event_markets) <= ladder_collapse_threshold:
            for market in event_markets:
                keep_ids.add(market.id)
            continue

        collapsed_events += 1
        ranked = sorted(
            event_markets,
            key=lambda market: (
                abs((_get_outcome_entry_price(market, "YES") or -1.0) - 0.5),
                -(market.liquidity_usdc or 0.0),
                market.id,
            ),
        )
        selected = ranked[:max_brackets_per_event]
        for market in selected:
            keep_ids.add(market.id)
        removed_markets += max(0, len(event_markets) - len(selected))
        logger.debug(
            "Collapsed ladder event=%s total=%d kept=%d",
            event_ticker,
            len(event_markets),
            len(selected),
            data={
                "event_ticker": event_ticker,
                "total_markets": len(event_markets),
                "kept_markets": [market.id for market in selected],
                "removed_count": max(0, len(event_markets) - len(selected)),
            },
        )

    if collapsed_events == 0:
        return markets

    collapsed: list[Market] = []
    for market in markets:
        if market.id in keep_ids or not (market.event_ticker or "").strip():
            collapsed.append(market)

    logger.info(
        "Collapsed event ladders: events=%d removed=%d kept=%d",
        collapsed_events,
        removed_markets,
        len(collapsed),
        data={
            "collapsed_events": collapsed_events,
            "removed_markets": removed_markets,
            "kept_markets": len(collapsed),
        },
    )
    return collapsed


def _dedupe_markets_by_matchup(markets: list[Market]) -> list[Market]:
    """Remove duplicate matchup markets with flipped team order."""
    if not markets:
        return markets

    kept: list[Market] = []
    seen: dict[str, Market] = {}
    seen_index: dict[str, int] = {}
    duplicates = 0

    for market in markets:
        key = _normalize_matchup_key(market.question)
        if not key:
            kept.append(market)
            continue

        if key not in seen:
            seen[key] = market
            seen_index[key] = len(kept)
            kept.append(market)
            continue

        duplicates += 1
        existing = seen[key]
        preferred = _select_preferred_market(existing, market)
        if preferred is existing:
            logger.debug(
                "Skipping duplicate market: id=%s matchup=%s",
                market.id,
                key,
                data={"market_id": market.id, "matchup_key": key},
            )
            continue

        logger.debug(
            "Replacing duplicate market: old_id=%s new_id=%s matchup=%s",
            existing.id,
            market.id,
            key,
            data={
                "matchup_key": key,
                "replaced_market_id": existing.id,
                "kept_market_id": market.id,
            },
        )
        kept[seen_index[key]] = market
        seen[key] = market

    if duplicates:
        logger.info(
            "Duplicate matchups removed: duplicates=%d kept=%d",
            duplicates,
            len(kept),
            data={"duplicates": duplicates, "kept": len(kept)},
        )
    return kept


def _normalize_matchup_key(question: str) -> str | None:
    """Normalize Team A vs Team B questions to a stable key."""
    if not question:
        return None

    text = question.strip()
    league = ""
    if ":" in text:
        prefix, rest = text.split(":", 1)
        league = prefix.strip().lower()
        text = rest.strip()

    text = text.rstrip("?")
    parts = _MATCHUP_SEPARATOR.split(text, maxsplit=1)
    if len(parts) != 2:
        return None

    left = _clean_matchup_team(parts[0])
    right = _clean_matchup_team(parts[1])
    if not left or not right:
        return None

    teams = sorted([left.lower(), right.lower()])
    key = f"{teams[0]} vs {teams[1]}"
    if league:
        key = f"{league}|{key}"
    return key


def _clean_matchup_team(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name.strip())
    cleaned = re.sub(r"\s*\(.*\)$", "", cleaned).strip()
    cleaned = cleaned.strip(" -")
    cleaned = cleaned.rstrip("?")
    return cleaned


def _select_preferred_market(existing: Market, candidate: Market) -> Market:
    """Choose a stable market when duplicates exist."""
    existing_key = _market_id_sort_key(existing)
    candidate_key = _market_id_sort_key(candidate)
    return existing if existing_key <= candidate_key else candidate


def _market_id_sort_key(market: Market) -> tuple[int, int | str]:
    try:
        return (0, int(market.id))
    except (TypeError, ValueError):
        return (1, str(market.id))


_MIN_VALID_PRICE = 0.01
_MAX_VALID_PRICE = 1.0
_PRICE_BUCKET_LOW = "lt_low_threshold"
_PRICE_BUCKET_MID = "mid_range"
_PRICE_BUCKET_HIGH = "gt_high_threshold"
_UNRESOLVED_WINNING_TOKENS = {"", "-1", "18446744073709551615"}
_UNIFORM_IMPLIED_EPSILON = 0.02


def _find_market_outcome(market: Market, outcome: str) -> MarketOutcome | None:
    if not market.outcomes:
        return None
    outcome_upper = outcome.upper()
    for market_outcome in market.outcomes:
        if market_outcome.name.upper() == outcome_upper:
            return market_outcome
    return None


def _get_outcome_entry_price(market: Market, outcome: str) -> float | None:
    market_outcome = _find_market_outcome(market, outcome)
    if not market_outcome:
        return None
    price = market_outcome.price
    if price is None:
        return None
    if _MIN_VALID_PRICE <= price <= _MAX_VALID_PRICE:
        return price
    return None


def _set_outcome_entry_price(market: Market, outcome: str, price: float) -> bool:
    market_outcome = _find_market_outcome(market, outcome)
    if not market_outcome:
        return False
    if not (_MIN_VALID_PRICE <= price <= _MAX_VALID_PRICE):
        return False
    market_outcome.price = price
    return True


def _build_order_request_from_market(
    market: Market,
    decision: TradeDecision,
    amount_usdc: float,
) -> OrderRequest:
    order_data: dict[str, Any] = {
        "market_id": market.id,
        "outcome": decision.outcome,
        "amount_usdc": amount_usdc,
        "confidence": decision.confidence,
    }
    outcome_price = _get_outcome_entry_price(market, decision.outcome)
    if outcome_price is not None:
        order_data["yes_price"] = int(round(outcome_price * 100))
    return OrderRequest(**order_data)


def _get_implied_probability(market: Market, outcome: str) -> float | None:
    market_outcome = _find_market_outcome(market, outcome)
    if not market_outcome:
        return None
    price = market_outcome.price
    if price is not None and _MIN_VALID_PRICE <= price <= _MAX_VALID_PRICE:
        return price
    odds = market_outcome.odds
    if odds is None or odds <= 0:
        return None
    implied = 1.0 / odds
    if _MIN_VALID_PRICE <= implied <= _MAX_VALID_PRICE:
        return implied
    return None


_SELF_CONSISTENCY_REPAIR_MARKER = "self_consistency_disagreement"


def _decision_has_near_binary_structured_probability(decision: TradeDecision) -> bool:
    for attr_name in ("my_prob", "probability_yes"):
        value = getattr(decision, attr_name, None)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric >= 0.95 or numeric <= 0.05:
            return True
    return False


def _edge_repair_reason(
    *,
    decision: TradeDecision,
    market: Market,
    settings: Settings,
    implied_prob: float | None,
) -> str | None:
    if not getattr(settings, "EDGE_REPAIR_ENABLED", True):
        return None
    repair_text = (
        f"{getattr(decision, 'self_critique', '') or ''} "
        f"{getattr(decision, 'reasoning', '') or ''}"
    ).lower()
    if _SELF_CONSISTENCY_REPAIR_MARKER in repair_text:
        return _SELF_CONSISTENCY_REPAIR_MARKER
    if not decision.should_trade or decision.abstain:
        return None
    edge_source = str(decision.edge_source or "").strip().lower()
    missing_structured_probability = (
        decision.my_prob is None and decision.probability_yes is None
    )
    if edge_source == "none":
        return "edge_source_none"
    if missing_structured_probability and edge_source in {"none", "fallback"}:
        return "missing_structured_probability"
    definitive = _is_definitive_outcome_eligible(decision, settings, market=market)
    if _decision_has_near_binary_structured_probability(decision) and not definitive:
        return "near_binary_without_definitive_evidence"
    confidence_for_edge = (
        decision.raw_confidence
        if decision.raw_confidence is not None
        else decision.confidence
    )
    edge_value: float | None = None
    if implied_prob is not None:
        edge_value = confidence_for_edge - implied_prob
    elif decision.edge_external is not None:
        edge_value = decision.edge_external
    if (
        getattr(settings, "EDGE_BAND_CALIBRATION_ENABLED", True)
        and edge_value is not None
        and abs(float(edge_value)) > 0.35
        and not definitive
    ):
        return "high_edge_without_definitive_evidence"
    return None


def _research_queue_context_text(context: dict[str, Any] | None) -> str | None:
    if not isinstance(context, dict):
        return None
    parts: list[str] = []
    reason = str(context.get("reason") or "").strip()
    if reason:
        parts.append(f"queue_reason={reason}")
    what_to_learn = str(context.get("what_to_learn_next") or "").strip()
    if what_to_learn:
        parts.append(f"what_to_learn_next={what_to_learn}")
    gate_name = str(context.get("gate_name") or "").strip()
    if gate_name:
        parts.append(f"gate_name={gate_name}")
    if not parts:
        return None
    parts.append(
        "repair_goal=compute probability_yes, market-implied probability, "
        "edge_market, base rate, counter-evidence, and explain what is already priced in"
    )
    return "Research queue repair target: " + "; ".join(parts)


def _market_with_research_queue_context(
    market: Market,
    research_queue_context: dict[str, Any] | None,
) -> Market:
    context_text = _research_queue_context_text(research_queue_context)
    if not context_text:
        return market
    existing_resolution = str(market.resolution_criteria or "").strip()
    updated_resolution = (
        f"{existing_resolution}\n\n{context_text}"
        if existing_resolution
        else context_text
    )
    return market.model_copy(update={"resolution_criteria": updated_resolution})


def _edge_band_label(edge_value: float | None) -> str | None:
    if edge_value is None:
        return None
    abs_edge = abs(float(edge_value))
    if abs_edge < 0.15:
        return "<15pp"
    if abs_edge < 0.25:
        return "15-25pp"
    if abs_edge < 0.35:
        return "25-35pp"
    if abs_edge < 0.45:
        return "35-45pp"
    return "45pp+"


def _expected_value_usdc(
    *,
    probability: float | None,
    entry_price: float | None,
    amount_usdc: float | None,
) -> float | None:
    if probability is None or entry_price is None or amount_usdc is None:
        return None
    try:
        p = max(0.0, min(1.0, float(probability)))
        price = float(entry_price)
        amount = float(amount_usdc)
    except (TypeError, ValueError):
        return None
    if price <= 0.0 or price >= 1.0 or amount <= 0.0:
        return None
    contracts = amount / price
    return (p * contracts * (1.0 - price)) - ((1.0 - p) * amount)


def _decision_outcome_probability(
    market: Market,
    decision: TradeDecision,
) -> float | None:
    probability_yes = getattr(decision, "probability_yes", None)
    if probability_yes is not None:
        try:
            yes_prob = max(0.0, min(1.0, float(probability_yes)))
        except (TypeError, ValueError):
            yes_prob = None
        if yes_prob is not None:
            normalized_outcome = _normalize_outcome_key(decision.outcome)
            if normalized_outcome == "yes":
                return yes_prob
            if normalized_outcome == "no":
                return 1.0 - yes_prob
    confidence = (
        decision.raw_confidence
        if decision.raw_confidence is not None
        else decision.confidence
    )
    try:
        return max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        return None


def _daily_expectancy_role(
    *,
    settings: Settings,
    opportunity_rank: int,
) -> tuple[str, float | None]:
    if not getattr(settings, "DAILY_EXPECTANCY_ENABLED", True):
        return "standard", None
    primary_target_limit = max(
        0,
        int(getattr(settings, "DAILY_EXPECTANCY_PRIMARY_TARGETS", 2) or 0),
    )
    if max(1, int(opportunity_rank)) <= primary_target_limit:
        return "primary_target", None
    satellite_cap_pct = max(
        0.0,
        min(
            1.0,
            float(getattr(settings, "DAILY_EXPECTANCY_SATELLITE_MAX_BET_PCT", 0.25)),
        ),
    )
    return "satellite", satellite_cap_pct


def _edge_threshold_for_market(
    implied_prob: float,
    settings: Settings,
    edge_source: str | None = None,
    market: Market | None = None,
    definitive_outcome_eligible: bool = False,
) -> float:
    """Effective minimum edge threshold for the edge gate.

    For definitive-outcome cases (game settled per whitelisted primary
    source) the FALLBACK_EDGE_MIN_EDGE / VERY_LOW_PRICE_MIN_EDGE /
    LOW_PRICE_MIN_EDGE bumps are bypassed — those bumps exist because of
    uncertain pricing, but a settled-game read against a whitelisted
    source has ground-truth pricing semantics.
    """
    min_edge = settings.MIN_EDGE
    if market is not None and not definitive_outcome_eligible:
        liquidity = float(market.liquidity_usdc or 0.0)
        high_liquidity_threshold = max(0.0, float(settings.MIN_EDGE_HIGH_LIQUIDITY_THRESHOLD))
        medium_liquidity_threshold = max(0.0, float(settings.MIN_EDGE_MEDIUM_LIQUIDITY_THRESHOLD))
        if high_liquidity_threshold > 0 and liquidity > high_liquidity_threshold:
            min_edge *= max(0.0, float(settings.MIN_EDGE_HIGH_LIQUIDITY_MULTIPLIER))
        elif medium_liquidity_threshold > 0 and liquidity > medium_liquidity_threshold:
            min_edge *= max(0.0, float(settings.MIN_EDGE_MEDIUM_LIQUIDITY_MULTIPLIER))
    is_weather_market = market is not None and market_family(market) == "weather"
    if is_weather_market and not definitive_outcome_eligible:
        min_edge = max(min_edge, settings.WEATHER_MIN_EDGE)
    if not definitive_outcome_eligible:
        low_price_multiplier = max(0.0, float(settings.LOW_PRICE_MIN_EDGE_MULTIPLIER))
        if implied_prob < settings.VERY_LOW_PRICE_THRESHOLD:
            min_edge = max(min_edge, settings.VERY_LOW_PRICE_MIN_EDGE * low_price_multiplier)
        if implied_prob < settings.LOW_PRICE_THRESHOLD:
            min_edge = max(min_edge, settings.LOW_PRICE_MIN_EDGE * low_price_multiplier)
        if settings.COINFLIP_PRICE_LOWER <= implied_prob <= settings.COINFLIP_PRICE_UPPER:
            min_edge = max(min_edge, settings.LOW_PRICE_MIN_EDGE * low_price_multiplier)
    if (edge_source or "").lower() == "fallback" and not definitive_outcome_eligible:
        fallback_multiplier = max(0.0, float(settings.FALLBACK_EDGE_MIN_EDGE_MULTIPLIER))
        min_edge = max(min_edge, settings.FALLBACK_EDGE_MIN_EDGE * fallback_multiplier)
        if is_weather_market:
            min_edge = max(
                min_edge,
                settings.WEATHER_FALLBACK_EDGE_MIN_EDGE * fallback_multiplier,
            )
    return min_edge


def _passes_edge_threshold(
    implied_prob: float | None,
    decision: TradeDecision,
    settings: Settings,
    market: Market | None = None,
) -> tuple[bool, float | None, str]:
    if implied_prob is None:
        if settings.REQUIRE_IMPLIED_PRICE:
            return False, None, "missing implied probability"
        return True, None, ""
    effective_confidence = (
        decision.raw_confidence
        if decision.raw_confidence is not None
        else decision.confidence
    )
    edge = effective_confidence - implied_prob
    if (
        str(getattr(decision, "edge_source", "") or "").strip().lower() == "none"
        and getattr(decision, "my_prob", None) is None
    ):
        return False, edge, "missing_structured_probability"
    is_definitive = _is_definitive_outcome_eligible(
        decision,
        settings,
        market=market,
    )
    is_definitive_validated = _is_definitive_validated(
        decision,
        settings,
        market=market,
    )
    max_reasonable_edge = (
        max(0.0, min(1.0, float(settings.DEFINITIVE_OUTCOME_EDGE_REASONABLE_MAX)))
        if is_definitive_validated
        else max(0.0, min(1.0, float(settings.MAX_REASONABLE_EDGE)))
    )
    if abs(edge) > max_reasonable_edge + 1e-9 and not is_definitive_validated:
        return False, edge, "edge_above_reasonable_max"
    if abs(edge) > max_reasonable_edge + 1e-9 and is_definitive_validated:
        if abs(edge) > 0.95:
            return False, edge, "edge_above_reasonable_max"
    if (
        market is not None
        and settings.NON_SPORTS_REQUIRES_DIRECT_EVIDENCE
        and market_family(market) != "sports"
    ):
        evidence_basis = _decision_evidence_basis(decision)
        if evidence_basis != "direct":
            return False, edge, "non_sports_needs_direct_evidence"
    min_edge = _edge_threshold_for_market(
        implied_prob,
        settings,
        decision.edge_source,
        market=market,
        definitive_outcome_eligible=is_definitive,
    )
    if edge < min_edge - 1e-9:
        return False, edge, f"edge {edge:.4f} below min {min_edge:.4f}"
    return True, edge, ""


def _adjust_bet_size_for_edge(
    decision: TradeDecision,
    implied_prob: float | None,
    edge: float | None,
    settings: Settings,
    market: Market | None = None,
) -> float:
    if edge is None or implied_prob is None:
        return decision.bet_size_pct
    min_edge = _edge_threshold_for_market(
        implied_prob,
        settings,
        decision.edge_source,
        market=market,
    )
    edge_over = edge - min_edge
    if edge_over <= 0:
        return 0.0
    scaling_range = max(settings.EDGE_SCALING_RANGE, 0.01)
    scale = min(1.0, edge_over / scaling_range)
    bet_pct = decision.bet_size_pct * scale
    if implied_prob < settings.LOW_PRICE_THRESHOLD:
        bet_pct *= settings.LOW_PRICE_BET_PENALTY
    normalized_edge_source = str(decision.edge_source or "").strip().lower()
    if normalized_edge_source in {"fallback", "none"}:
        max_bet_safe = max(settings.MAX_BET_USDC, 1e-9)
        fallback_max_pct = max(0.0, min(1.0, settings.MIN_BET_USDC / max_bet_safe))
        bet_pct = min(bet_pct, fallback_max_pct)
    return max(0.0, min(1.0, bet_pct))


def _market_confidence_family(market: Market) -> str:
    family = market_family(market)
    if family in {"weather", "crypto", "speech"}:
        return family
    market_id = (market.id or "").upper()
    if "LCATTLE" in market_id or "LIVECATTLE" in market_id:
        return "livestock"
    if "HOIL" in market_id:
        return "heating_oil"
    if "CORN" in market_id:
        return "corn"
    if any(market_id.startswith(prefix) for prefix in _INDEX_MARKET_PREFIXES):
        return "index"
    if any(token in market_id for token in _COMMODITY_MARKET_TOKENS):
        return "commodity"
    return "generic"


def _confidence_shrinkage_override_for_market(market: Market) -> float | None:
    confidence_family = _market_confidence_family(market)
    if confidence_family in {"weather", "crypto", "index"}:
        return _AGGRESSIVE_CONFIDENCE_SHRINKAGE_FACTOR
    return None


def _is_within_order_submission_band(
    price: float | None,
    settings: Settings,
) -> bool:
    if price is None:
        return False
    return settings.ORDER_SUBMISSION_MIN_PRICE <= price <= settings.ORDER_SUBMISSION_MAX_PRICE


def _max_confidence_for_market(market: Market | None, settings: Settings) -> float:
    if not market:
        return settings.MAX_GLOBAL_CONFIDENCE
    is_sports, is_esports = market_category_flags(market)
    if is_sports:
        return min(settings.MAX_SPORTS_CONFIDENCE, settings.MAX_GLOBAL_CONFIDENCE)
    if is_esports:
        return min(settings.MAX_ESPORTS_CONFIDENCE, settings.MAX_GLOBAL_CONFIDENCE)
    confidence_family = _market_confidence_family(market)
    if confidence_family == "weather":
        return min(settings.MAX_WEATHER_CONFIDENCE, settings.MAX_GLOBAL_CONFIDENCE)
    if confidence_family == "index":
        return min(settings.MAX_INDEX_CONFIDENCE, settings.MAX_GLOBAL_CONFIDENCE)
    if confidence_family == "commodity":
        return min(settings.MAX_COMMODITY_CONFIDENCE, settings.MAX_GLOBAL_CONFIDENCE)
    if confidence_family == "livestock":
        return min(settings.MAX_LIVESTOCK_CONFIDENCE, settings.MAX_GLOBAL_CONFIDENCE)
    if confidence_family == "heating_oil":
        return min(
            settings.MAX_HEATING_OIL_CONFIDENCE,
            settings.MAX_GLOBAL_CONFIDENCE,
        )
    if confidence_family == "corn":
        return min(settings.MAX_CORN_CONFIDENCE, settings.MAX_GLOBAL_CONFIDENCE)
    if confidence_family == "crypto":
        return min(settings.MAX_CRYPTO_CONFIDENCE, settings.MAX_GLOBAL_CONFIDENCE)
    if confidence_family == "speech":
        return min(settings.MAX_SPEECH_CONFIDENCE, settings.MAX_GLOBAL_CONFIDENCE)
    return settings.MAX_GLOBAL_CONFIDENCE


def _non_definitive_confidence_ceiling(
    decision: TradeDecision,
    settings: Settings,
    market: Market | None = None,
) -> float:
    """Hard ceiling for non-definitive trades per historical calibration data.

    The 0.90+ confidence tier historically wins only ~43% vs 60% at 0.85-0.89.
    Allow definitive-outcome-eligible decisions to bypass this cap since they
    are backed by settlement-aligned primary sources.
    """
    if _is_definitive_outcome_eligible(decision, settings, market=market):
        return 1.0
    evidence_basis = str(getattr(decision, "evidence_basis", "") or "").strip().lower()
    if evidence_basis == "direct":
        return min(1.0, max(0.0, settings.MAX_GLOBAL_CONFIDENCE_DIRECT))
    return min(1.0, max(0.0, settings.MAX_GLOBAL_CONFIDENCE_DIRECT))


def _historical_win_rate_at_bucket(confidence: float) -> float | None:
    rounded = round(max(0.0, min(1.0, confidence)) * 10.0) / 10.0
    return _HISTORICAL_WIN_RATE_BY_BUCKET.get(rounded)


def _is_whitelisted_primary_source_url(url: str, settings: Settings) -> bool:
    normalized_url = str(url or "").strip().lower()
    if not normalized_url:
        return False
    parsed = urlparse(normalized_url)
    host = (parsed.netloc or "").split("@")[-1].split(":")[0].lower()
    for whitelist_entry in settings.DIRECT_SOURCE_WHITELIST:
        normalized_entry = str(whitelist_entry or "").strip().lower()
        if not normalized_entry:
            continue
        if "/" in normalized_entry and normalized_entry in normalized_url:
            return True
        if host and (host == normalized_entry or host.endswith(f".{normalized_entry}")):
            return True
    return False


def _decision_raw_evidence_quality(decision: TradeDecision) -> float:
    raw_value = getattr(decision, "raw_evidence_quality", None)
    if raw_value is None:
        raw_value = getattr(decision, "evidence_quality", 0.0)
    try:
        return max(0.0, min(1.0, float(raw_value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _decision_has_near_binary_my_prob(decision: TradeDecision) -> bool:
    my_prob = getattr(decision, "my_prob", None)
    if my_prob is None:
        return False
    try:
        my_prob_value = float(my_prob)
    except (TypeError, ValueError):
        return False
    return my_prob_value >= 0.95 or my_prob_value <= 0.05


def _significant_market_tokens(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[a-zA-Z][a-zA-Z']{3,}", text or ""):
        normalized = token.strip("'").lower()
        if normalized and normalized not in _SPORTS_ENTITY_STOPWORDS:
            tokens.add(normalized)
    return tokens


def _decision_source_text(decision: TradeDecision) -> str:
    parts = [
        str(getattr(decision, "reasoning", "") or ""),
        str(getattr(decision, "primary_source_url", "") or ""),
    ]
    key_sources = getattr(decision, "key_sources", None)
    if isinstance(key_sources, (list, tuple)):
        parts.extend(str(item) for item in key_sources)
    elif key_sources:
        parts.append(str(key_sources))
    return " ".join(parts).lower()


def _sports_settlement_source_matches_market(
    decision: TradeDecision,
    market: Market | None,
) -> bool:
    if market is None or market_family(market) != "sports":
        return True
    source_match = str(getattr(decision, "source_match_class", "") or "").strip().lower()
    if source_match != "settlement_aligned":
        return True

    tokens = _significant_market_tokens(
        " ".join(
            (
                str(getattr(market, "question", "") or ""),
                str(getattr(market, "resolution_criteria", "") or ""),
            )
        )
    )
    if not tokens:
        return True

    decision_text = _decision_source_text(decision)
    matched_tokens = {
        token
        for token in tokens
        if re.search(rf"\b{re.escape(token)}\b", decision_text)
    }
    required_matches = 2 if len(tokens) >= 2 else 1
    return len(matched_tokens) >= required_matches


def _is_high_quality_settled_evidence(
    decision: TradeDecision,
    settings: Settings,
    market: Market | None = None,
) -> bool:
    """Recognize concrete settlement-aligned chart/observation evidence.

    Returns True when the decision has ALL of:
    - ``evidence_basis == "direct"``
    - ``source_match_class == "settlement_aligned"`` (Grok detected concrete
      chart or observation data tied to the market's settlement criterion)
    - validated ``evidence_quality`` at or above
      ``settings.HIGH_QUALITY_SETTLED_EVIDENCE_MIN_EQ`` (default 0.95)
    - whitelisted ``primary_source_url``

    This handles cases where the underlying numeric reading isn't strictly
    binary (e.g. 23,291 vs a 39,000 threshold yielding ``my_prob`` ~0.90)
    so ``definitive_outcome_detected`` was not set, but the evidence is
    concrete, well-sourced, and aligned with the settlement criterion. The
    exemption lifts the ``MAX_REASONABLE_EDGE`` cap to
    ``DEFINITIVE_OUTCOME_EDGE_REASONABLE_MAX`` and suppresses the matching
    hallucinated-edge / high-edge calibration penalties so the trade can
    reach the score gate, where calibration shrinkage and other penalties
    continue to scale.
    """
    if _decision_evidence_basis(decision) != "direct":
        return False
    source_match = (
        str(getattr(decision, "source_match_class", "") or "").strip().lower()
    )
    if source_match != "settlement_aligned":
        return False
    eq = float(decision.evidence_quality or 0.0)
    eq_floor = float(
        getattr(
            settings,
            "HIGH_QUALITY_SETTLED_EVIDENCE_MIN_EQ",
            0.95,
        )
    )
    if eq < eq_floor:
        return False
    primary_source_url = str(getattr(decision, "primary_source_url", "") or "").strip()
    if not primary_source_url:
        return False
    if not _is_whitelisted_primary_source_url(primary_source_url, settings):
        return False
    return _sports_settlement_source_matches_market(decision, market)


def _should_suppress_hallucinated_edge_penalty(
    *,
    decision: TradeDecision,
    evidence_basis: str,
    settings: Settings,
    market: Market | None = None,
) -> bool:
    if str(evidence_basis or "").strip().lower() != "direct":
        return False
    primary_source_url = str(getattr(decision, "primary_source_url", "") or "").strip()
    if not primary_source_url:
        return False
    if not _is_whitelisted_primary_source_url(primary_source_url, settings):
        return False
    if bool(getattr(decision, "definitive_outcome_detected", False)):
        return _is_definitive_outcome_eligible(decision, settings, market=market)
    return _is_high_quality_settled_evidence(decision, settings, market=market)


def _is_definitive_outcome_eligible(
    decision: TradeDecision,
    settings: Settings,
    market: Market | None = None,
) -> bool:
    """Check whether a decision qualifies for definitive-outcome floor overrides.

    Definitive overrides are intentionally strict because they bypass several
    normal uncertainty penalties. A candidate needs direct, whitelisted,
    settlement-aligned evidence, high raw evidence quality, and a structured
    near-binary ``my_prob``. Sports settlement sources also need to mention
    the market's own entity tokens so an unrelated boxscore cannot unlock
    definitive handling.
    """
    evidence_basis = _decision_evidence_basis(decision)
    if evidence_basis != "direct":
        return False
    primary_source_url = str(getattr(decision, "primary_source_url", "") or "").strip()
    if not primary_source_url:
        return False
    if not _is_whitelisted_primary_source_url(primary_source_url, settings):
        return False
    source_match = str(getattr(decision, "source_match_class", "") or "").strip().lower()
    if source_match != "settlement_aligned":
        return False
    if not _sports_settlement_source_matches_market(decision, market):
        return False
    if not _decision_has_near_binary_my_prob(decision):
        return False
    raw_eq_floor = max(
        0.0,
        min(1.0, float(settings.DEFINITIVE_OUTCOME_EVIDENCE_QUALITY_FLOOR)),
    )
    if _decision_raw_evidence_quality(decision) < raw_eq_floor:
        return False
    return True


def _is_definitive_validated(
    decision: TradeDecision,
    settings: Settings,
    market: Market | None = None,
) -> bool:
    """Strict validation that gates the higher ``DEFINITIVE_OUTCOME_EDGE_REASONABLE_MAX`` cap.

    A decision is validated when EITHER:
    - It is ``_is_definitive_outcome_eligible`` AND has the model-flagged
      ``definitive_outcome_detected=True`` AND eq>=0.80 AND direct +
      settlement_aligned source_match_class (the legacy strict path), OR
    - It satisfies ``_is_high_quality_settled_evidence`` (direct +
      settlement_aligned + eq>=HIGH_QUALITY_SETTLED_EVIDENCE_MIN_EQ +
      whitelisted source). The strict eq floor of 0.95 in the new path
      compensates for the relaxed ``definitive_outcome_detected`` requirement.
    """
    legacy_validated = (
        _is_definitive_outcome_eligible(decision, settings, market=market)
        and bool(getattr(decision, "definitive_outcome_detected", False))
        and float(getattr(decision, "evidence_quality", 0.0) or 0.0) >= 0.80
        and _decision_evidence_basis(decision) == "direct"
        and (getattr(decision, "source_match_class", "") or "").strip().lower()
        == "settlement_aligned"
    )
    if legacy_validated:
        return True
    return _is_high_quality_settled_evidence(decision, settings, market=market)


def _apply_definitive_outcome_floors(
    decision: TradeDecision,
    market: Market,
    settings: Settings,
) -> tuple[TradeDecision, bool]:
    """Floor evidence_quality when the decision is definitive-eligible.

    Returns ``(possibly_modified_decision, floor_was_applied)``. When the
    decision qualifies, the returned copy has ``evidence_quality`` raised
    to at least ``settings.DEFINITIVE_OUTCOME_EVIDENCE_QUALITY_FLOOR`` and
    ``definitive_outcome_detected`` set to True so downstream consumers
    pick up the auto-detection.
    """
    if not _is_definitive_outcome_eligible(decision, settings, market=market):
        return decision, False
    floor = max(0.0, min(1.0, float(settings.DEFINITIVE_OUTCOME_EVIDENCE_QUALITY_FLOOR)))
    current_eq = float(decision.evidence_quality or 0.0)
    if current_eq >= floor and bool(getattr(decision, "definitive_outcome_detected", False)):
        return decision, False
    updated = decision.model_copy(
        update={
            "evidence_quality": max(current_eq, floor),
            "definitive_outcome_detected": True,
            "evidence_quality_floor_applied": "definitive_outcome",
        }
    )
    return updated, True


def _min_evidence_quality_for_market(
    market: Market,
    settings: Settings,
    decision: TradeDecision | None = None,
) -> float:
    family = market_family(market)
    if family == "weather":
        base_minimum = settings.WEATHER_MIN_EVIDENCE_QUALITY
    elif family == "sports":
        base_minimum = settings.SPORTS_MIN_EVIDENCE_QUALITY
    else:
        base_minimum = settings.MIN_EVIDENCE_QUALITY_FOR_TRADE
    if decision is None:
        return base_minimum
    evidence_basis = _decision_evidence_basis(decision)
    primary_source_url = str(getattr(decision, "primary_source_url", "") or "").strip()
    if evidence_basis != "direct" or not primary_source_url:
        return base_minimum
    if not _is_whitelisted_primary_source_url(primary_source_url, settings):
        return base_minimum
    if family == "weather":
        direct_source_minimum = settings.DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_WEATHER
    elif family == "sports":
        direct_source_minimum = settings.DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_SPORTS
    else:
        direct_source_minimum = settings.DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_DEFAULT
    return min(base_minimum, max(0.0, min(1.0, float(direct_source_minimum))))


def _should_queue_research_for_blocked_trade(
    *,
    settings: Settings,
    decision: TradeDecision,
    evidence_basis: str,
    gate_name: str,
    threshold_gap: float,
    edge_reason: str | None = None,
) -> bool:
    if not settings.RESEARCH_QUEUE_ENABLED:
        return False
    if not decision.should_trade or decision.abstain:
        return False
    normalized_evidence_basis = str(evidence_basis or "").strip().lower()
    normalized_edge_source = str(getattr(decision, "edge_source", "") or "").strip().lower()
    if normalized_evidence_basis != "direct":
        if gate_name in {"evidence", "source"} and (
            normalized_evidence_basis in {"absence_only", "proxy"}
            or normalized_edge_source in {"", "none"}
            or float(getattr(decision, "evidence_quality", 0.0) or 0.0) <= 0.0
        ):
            return True
        return False
    normalized_gap = max(0.0, float(threshold_gap))
    if gate_name == "evidence":
        return normalized_gap <= _RESEARCH_QUEUE_EVIDENCE_GAP_MAX
    if gate_name == "edge":
        if str(edge_reason or "") in {
            "edge_above_reasonable_max",
            "missing_structured_probability",
        }:
            return True
        return normalized_gap <= _RESEARCH_QUEUE_EDGE_GAP_MAX
    if gate_name in {"hallucinated_edge", "extreme_market_edge"}:
        return True
    return False


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    normalized_quantile = max(0.0, min(1.0, float(quantile)))
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = normalized_quantile * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    if lower_index == upper_index:
        return ordered[lower_index]
    interpolation = position - lower_index
    return ordered[lower_index] + ((ordered[upper_index] - ordered[lower_index]) * interpolation)


def _cap_effective_confidence_for_market(
    confidence: float,
    market: Market | None,
    settings: Settings,
) -> float:
    return min(confidence, _max_confidence_for_market(market, settings))


def _effective_position_override_threshold(
    market: Market | None,
    settings: Settings,
) -> float:
    return settings.HIGH_CONFIDENCE_POSITION_OVERRIDE


def _price_bucket(
    implied_prob: float | None,
    settings: Settings,
) -> str:
    if implied_prob is None:
        return _PRICE_BUCKET_LOW
    if implied_prob < settings.LOW_PRICE_THRESHOLD:
        return _PRICE_BUCKET_LOW
    if implied_prob <= settings.HIGH_PRICE_THRESHOLD:
        return _PRICE_BUCKET_MID
    return _PRICE_BUCKET_HIGH


def _canonical_outcome_name(market: Market, outcome: str) -> str:
    market_outcome = _find_market_outcome(market, outcome)
    if market_outcome:
        return market_outcome.name
    return outcome


def _load_or_initialize_bayesian_states(
    market: Market,
    state_manager: MarketStateManager,
    settings: Settings,
) -> dict[str, BayesianState]:
    states = state_manager.get_bayesian_state(market.id)
    outcome_names = [outcome.name for outcome in market.outcomes]
    if states:
        return states

    if len(outcome_names) == 2:
        seeded_states = initial_state(len(outcome_names), prior=settings.BAYESIAN_PRIOR_DEFAULT)
    else:
        seeded_states = initial_state(len(outcome_names), prior=None)

    initialized: dict[str, BayesianState] = {}
    for outcome_name, state in zip(outcome_names, seeded_states):
        initialized[outcome_name] = state
        state_manager.update_bayesian_state(
            market_id=market.id,
            outcome=outcome_name,
            log_prior=state.log_prior,
            log_likelihood=0.0,
            count_as_update=False,
        )
    return initialized


def _applied_bayesian_posterior(
    bayesian_posterior_raw: float | None,
    bayesian_update_count: int,
    min_updates_for_trade: int,
) -> float | None:
    if bayesian_posterior_raw is None:
        return None
    if bayesian_update_count < max(0, int(min_updates_for_trade)):
        return None
    return bayesian_posterior_raw


def _cap_bayesian_confidence_boost(
    *,
    base_confidence: float,
    candidate_confidence: float,
    max_boost: float,
) -> float:
    boost_ceiling = min(_MAX_CONFIDENCE, base_confidence + max(0.0, max_boost))
    return min(candidate_confidence, boost_ceiling)


def _kelly_fraction_for_market_horizon(market: Market, settings: Settings) -> float:
    weather_multiplier = max(0.0, settings.KELLY_FRACTION_WEATHER)
    is_weather_market = market_family(market) == "weather"

    if market.close_time is None:
        base_fraction = settings.KELLY_FRACTION_DEFAULT
        if is_weather_market:
            return max(0.0, min(1.0, base_fraction * weather_multiplier))
        return base_fraction
    close_time = market.close_time
    if close_time.tzinfo is None:
        close_time = close_time.replace(tzinfo=timezone.utc)
    horizon_seconds = (close_time - datetime.now(timezone.utc)).total_seconds()
    short_horizon_seconds = max(0, settings.KELLY_FRACTION_SHORT_HORIZON_HOURS) * 3600
    base_fraction = settings.KELLY_FRACTION_DEFAULT
    if short_horizon_seconds > 0 and horizon_seconds <= short_horizon_seconds:
        base_fraction = settings.KELLY_FRACTION_SHORT_HORIZON
    if is_weather_market:
        return max(0.0, min(1.0, base_fraction * weather_multiplier))
    return base_fraction


def _sizing_mode_label(kelly_enabled: bool) -> str:
    return "kelly" if kelly_enabled else "edge_scaling"


def _zero_bet_skip_message(sizing_mode: str) -> str:
    if sizing_mode == "kelly":
        return "bet size reduced to zero by Kelly sizing"
    return "bet size reduced to zero by edge scaling"


def _resolve_min_bet_floor(
    bet_amount: float,
    *,
    min_bet_usdc: float,
    max_bet_usdc: float,
    kelly_path_active: bool,
    min_bet_policy: str,
    edge_scaling_bet_pct: float | None = None,
) -> tuple[float, float, bool, bool, str]:
    """Resolve minimum bet handling and return amount, pct, flags, and policy."""
    max_bet_safe = max(0.0, max_bet_usdc)
    if max_bet_safe <= 0:
        return 0.0, 0.0, False, False, _KELLY_MIN_BET_POLICY_SKIP
    original_pct = max(0.0, min(1.0, bet_amount / max_bet_safe))
    if bet_amount >= min_bet_usdc:
        return bet_amount, original_pct, False, False, _KELLY_MIN_BET_POLICY_FLOOR
    if not kelly_path_active:
        floored_amount = min_bet_usdc
        floored_pct = max(0.0, min(1.0, floored_amount / max_bet_safe))
        return floored_amount, floored_pct, True, False, _KELLY_MIN_BET_POLICY_FLOOR

    normalized_policy = (min_bet_policy or "").strip().lower()
    if normalized_policy not in {
        _KELLY_MIN_BET_POLICY_SKIP,
        _KELLY_MIN_BET_POLICY_FLOOR,
        _KELLY_MIN_BET_POLICY_FALLBACK_EDGE,
    }:
        normalized_policy = _KELLY_MIN_BET_POLICY_SKIP

    if normalized_policy == _KELLY_MIN_BET_POLICY_SKIP:
        return bet_amount, original_pct, False, True, normalized_policy
    if normalized_policy == _KELLY_MIN_BET_POLICY_FLOOR:
        floored_amount = min_bet_usdc
        floored_pct = max(0.0, min(1.0, floored_amount / max_bet_safe))
        return floored_amount, floored_pct, True, False, normalized_policy

    fallback_pct = max(0.0, min(1.0, edge_scaling_bet_pct or 0.0))
    fallback_amount = _calculate_bet(max_bet_safe, fallback_pct)
    if fallback_amount < min_bet_usdc:
        fallback_amount = min_bet_usdc
    fallback_pct = max(0.0, min(1.0, fallback_amount / max_bet_safe))
    min_floor_applied = fallback_amount == min_bet_usdc
    return fallback_amount, fallback_pct, min_floor_applied, False, normalized_policy


def _compute_lmsr_execution_price_for_outcome(
    market: Market,
    decision_outcome: str,
    amount_usdc: float,
    settings: Settings,
) -> float | None:
    if not market.outcomes:
        return None
    prices: list[float] = []
    outcome_names: list[str] = []
    for market_outcome in market.outcomes:
        implied = _get_implied_probability(market, market_outcome.name)
        if implied is None:
            continue
        prices.append(implied)
        outcome_names.append(market_outcome.name)

    if len(prices) < 2:
        return None
    selected_idx = next(
        (idx for idx, name in enumerate(outcome_names) if _outcomes_match(name, decision_outcome)),
        None,
    )
    if selected_idx is None:
        return None
    if amount_usdc <= 0:
        return None
    try:
        quantities = infer_quantities_from_prices(prices, settings.LMSR_LIQUIDITY_PARAM_B)
        current_prices = lmsr_prices(quantities, settings.LMSR_LIQUIDITY_PARAM_B)
        current_price = current_prices[selected_idx]
        if current_price <= 0:
            return None
        trade_delta_shares = amount_usdc / current_price
        if trade_delta_shares <= 0:
            return None
        estimated_cost = lmsr_trade_cost(
            quantities=quantities,
            outcome_idx=selected_idx,
            delta=trade_delta_shares,
            b=settings.LMSR_LIQUIDITY_PARAM_B,
        )
    except (ValueError, OverflowError):
        return None
    if estimated_cost <= 0:
        return None
    average_execution_price = estimated_cost / trade_delta_shares
    return average_execution_price


def _best_orderbook_sell_price(
    orderbook: dict[str, Any],
    option_index: int,
) -> float | None:
    sells = orderbook.get("sells")
    if not isinstance(sells, list):
        return None
    best_price: float | None = None
    for entry in sells:
        if not isinstance(entry, dict):
            continue
        if entry.get("optionIndex") != option_index:
            continue
        candidate = _coerce_float(entry.get("price"))
        if candidate is None:
            continue
        if best_price is None or candidate < best_price:
            best_price = candidate
    return best_price


def _orderbook_entry_quantity(entry: dict[str, Any]) -> float | None:
    quantity_keys = (
        "quantity",
        "quantity_shares",
        "quantityShares",
        "size",
        "count",
        "remaining_count",
        "remainingCount",
        "resting_count",
    )
    for key in quantity_keys:
        quantity = _coerce_float(entry.get(key))
        if quantity is not None and quantity > 0:
            return quantity
    return None


def _available_orderbook_sell_quantity(
    orderbook: dict[str, Any],
    option_index: int,
    max_price: float | None,
) -> float | None:
    sells = orderbook.get("sells")
    if not isinstance(sells, list):
        return None
    available_quantity = 0.0
    quantity_seen = False
    for entry in sells:
        if not isinstance(entry, dict):
            continue
        if entry.get("optionIndex") != option_index:
            continue
        entry_price = _coerce_float(entry.get("price"))
        if max_price is not None and entry_price is not None and entry_price > max_price:
            continue
        entry_quantity = _orderbook_entry_quantity(entry)
        if entry_quantity is None:
            continue
        quantity_seen = True
        available_quantity += entry_quantity
    if not quantity_seen:
        return None
    return available_quantity


def _is_uniform_implied_probability(
    implied_prob: float | None,
    outcomes: list[MarketOutcome],
) -> bool:
    if implied_prob is None or len(outcomes) <= 2:
        return False
    uniform_implied = 1.0 / len(outcomes)
    return abs(implied_prob - uniform_implied) < _UNIFORM_IMPLIED_EPSILON


def _extract_winning_outcome(market: Market) -> str | None:
    status_text = str(market.status).strip().lower() if market.status is not None else ""
    if status_text in {"0", "open", "active"}:
        return None
    candidates = (
        "winning_option_raw",
        "winningOption",
        "winning_option",
        "winningOptionIndex",
        "winning_option_index",
        "winningOutcome",
        "winning_outcome",
        "market_result",
        "result",
        "settlement_result",
        "settled_value",
    )
    for key in candidates:
        value = getattr(market, key, None)
        if value is None:
            continue
        if _is_unresolved_winning_value(value):
            return None
        if isinstance(value, (int, float)) or (
            isinstance(value, str)
            and value.strip().lstrip("-").isdigit()
        ):
            index = int(value)
            if 0 <= index < len(market.outcomes):
                return market.outcomes[index].name
            return None
        if isinstance(value, str):
            normalized_value = value.strip()
            if normalized_value.lower() in {"yes", "no"}:
                return normalized_value.upper()
            return normalized_value
    return None


def _is_unresolved_winning_value(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if text in _UNRESOLVED_WINNING_TOKENS:
        return True
    if text.startswith("-") and text[1:].isdigit():
        return True
    if text.isdigit():
        index = int(text)
        if index < 0:
            return True
    return False


def _is_market_resolved_or_closed(market: Market) -> bool:
    """Return True when market appears settled/closed based on status/winner signals."""
    winning_outcome = _extract_winning_outcome(market)
    if winning_outcome:
        return True
    return _status_indicates_closed(market.status)


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _decision_edge_for_outcome(
    market: Market,
    outcome: str,
    confidence: float,
) -> float | None:
    implied = _get_implied_probability(market, outcome)
    if implied is None:
        return None
    return confidence - implied


def _apply_flip_guard(
    market: Market,
    decision: TradeDecision,
    anchor_analysis: dict[str, Any] | None,
    settings: Settings,
) -> tuple[TradeDecision, bool, bool]:
    """Apply strict flip guardrails against switching sides across cycles."""
    if not settings.FLIP_GUARD_ENABLED or anchor_analysis is None:
        return decision, False, False

    anchor_outcome_raw = anchor_analysis.get("outcome")
    anchor_outcome = str(anchor_outcome_raw).strip() if anchor_outcome_raw is not None else ""
    if not anchor_outcome:
        return decision, False, False
    if _outcomes_match(decision.outcome, anchor_outcome):
        return decision, False, False

    anchor_confidence = _coerce_float(anchor_analysis.get("confidence")) or 0.0
    evidence_basis_class = str(decision.evidence_basis or "").strip().lower()
    likelihood_ratio = _coerce_float(decision.likelihood_ratio) or 0.0
    use_raw_confidence_for_flip_guard = (
        evidence_basis_class == "direct"
        and decision.raw_confidence is not None
        and likelihood_ratio >= 5.0
    )
    evaluated_confidence = (
        max(0.0, min(1.0, float(decision.raw_confidence)))
        if use_raw_confidence_for_flip_guard
        else decision.confidence
    )
    if anchor_confidence < settings.MIN_CONFIDENCE:
        logger.debug(
            "FlipGuard bypassed due to low-confidence anchor: market=%s anchor_conf=%.3f threshold=%.3f",
            market.id,
            anchor_confidence,
            settings.MIN_CONFIDENCE,
            data={
                "market_id": market.id,
                "anchor_outcome": anchor_outcome,
                "proposed_outcome": decision.outcome,
                "anchor_confidence": anchor_confidence,
                "min_confidence_threshold": settings.MIN_CONFIDENCE,
                "use_raw_confidence_for_flip_guard": use_raw_confidence_for_flip_guard,
            },
        )
        return decision, False, False

    confidence_delta = evaluated_confidence - anchor_confidence
    new_edge = _decision_edge_for_outcome(market, decision.outcome, evaluated_confidence)
    anchor_edge = _decision_edge_for_outcome(market, anchor_outcome, anchor_confidence)
    edge_delta = None
    edge_gain_ok = True
    if new_edge is not None and anchor_edge is not None:
        edge_delta = abs(new_edge) - abs(anchor_edge)
        edge_gain_ok = edge_delta >= settings.FLIP_GUARD_MIN_EDGE_GAIN

    abs_conf_ok = evaluated_confidence >= settings.FLIP_GUARD_MIN_ABS_CONFIDENCE
    conf_gain_ok = confidence_delta >= settings.FLIP_GUARD_MIN_CONF_GAIN
    evidence_quality = decision.evidence_quality or 0.0
    evidence_ok = evidence_quality >= settings.FLIP_GUARD_MIN_EVIDENCE_QUALITY
    high_evidence_flip_override = evidence_quality >= 0.90 and decision.confidence >= 0.90

    payload = {
        "market_id": market.id,
        "anchor_outcome": anchor_outcome,
        "proposed_outcome": decision.outcome,
        "anchor_confidence": anchor_confidence,
        "proposed_confidence": decision.confidence,
        "flip_guard_evaluated_confidence": evaluated_confidence,
        "confidence_delta": confidence_delta,
        "anchor_edge": anchor_edge,
        "proposed_edge": new_edge,
        "edge_delta": edge_delta,
        "evidence_quality": evidence_quality,
        "abs_conf_ok": abs_conf_ok,
        "conf_gain_ok": conf_gain_ok,
        "edge_gain_ok": edge_gain_ok,
        "evidence_ok": evidence_ok,
        "high_evidence_flip_override": high_evidence_flip_override,
        "use_raw_confidence_for_flip_guard": use_raw_confidence_for_flip_guard,
    }

    if high_evidence_flip_override or (abs_conf_ok and conf_gain_ok and edge_gain_ok and evidence_ok):
        logger.info(
            "FlipGuard passed: market=%s anchor=%s proposed=%s conf_delta=%.3f edge_delta=%s",
            market.id,
            anchor_outcome,
            decision.outcome,
            confidence_delta,
            f"{edge_delta:.3f}" if edge_delta is not None else "n/a",
            data=payload,
        )
        return decision, True, False

    reasons: list[str] = []
    if not abs_conf_ok:
        reasons.append(
            f"abs_conf {evaluated_confidence:.2f} < {settings.FLIP_GUARD_MIN_ABS_CONFIDENCE:.2f}"
        )
    if not conf_gain_ok:
        reasons.append(
            f"conf_gain {confidence_delta:.2f} < {settings.FLIP_GUARD_MIN_CONF_GAIN:.2f}"
        )
    if edge_delta is not None and not edge_gain_ok:
        reasons.append(
            f"edge_gain {edge_delta:.3f} < {settings.FLIP_GUARD_MIN_EDGE_GAIN:.3f}"
        )
    if not evidence_ok:
        reasons.append(
            "evidence_quality "
            f"{evidence_quality:.2f} < {settings.FLIP_GUARD_MIN_EVIDENCE_QUALITY:.2f}"
        )
    block_reason = "; ".join(reasons) if reasons else "criteria not met"
    payload["block_reason"] = block_reason

    blocked_decision = decision.model_copy(
        update={
            "should_trade": False,
            "bet_size_pct": 0.0,
            "outcome": anchor_outcome,
            "reasoning": (
                f"[FlipGuard blocked: {block_reason}; anchor={anchor_outcome}; "
                f"proposed={decision.outcome}] {decision.reasoning}"
            ),
        }
    )
    logger.warning(
        "FlipGuard blocked: market=%s anchor=%s proposed=%s conf_delta=%.3f edge_delta=%s reason=%s",
        market.id,
        anchor_outcome,
        decision.outcome,
        confidence_delta,
        f"{edge_delta:.3f}" if edge_delta is not None else "n/a",
        block_reason,
        data=payload,
    )
    return blocked_decision, True, True


def _update_resolved_markets(
    markets: list[Market],
    state_manager: MarketStateManager,
    kalshi_client: KalshiClient,
    settings: Settings | None = None,
) -> None:
    traded_ids = state_manager.get_unresolved_traded_market_ids()
    if not traded_ids:
        return
    market_map = {market.id: market for market in markets}
    resolved_count = 0
    fetched_market_count = 0
    for market_id in traded_ids:
        market = market_map.get(market_id)
        if not market:
            try:
                market = kalshi_client.get_market(market_id)
                fetched_market_count += 1
            except Exception as exc:
                logger.debug(
                    "Resolution sync lookup failed for traded market %s: %s",
                    market_id,
                    exc,
                    data={"market_id": market_id, "error": str(exc)},
                )
                continue
        winning_outcome = _extract_winning_outcome(market)
        if not winning_outcome:
            continue
        updated = state_manager.record_resolution(
            market_id=market_id,
            winning_outcome=winning_outcome,
            resolved_at=market.close_time,
            online_calibration_enabled=(
                bool(getattr(settings, "CALIBRATION_ONLINE_UPDATE_ENABLED", False))
                if settings is not None
                else False
            ),
            online_calibration_alpha=(
                float(getattr(settings, "CALIBRATION_ONLINE_ALPHA", 0.15))
                if settings is not None
                else 0.15
            ),
            online_calibration_max_samples_per_bucket=(
                int(getattr(settings, "CALIBRATION_ONLINE_MAX_SAMPLES_PER_BUCKET", 500))
                if settings is not None
                else 500
            ),
        )
        if updated:
            state_manager.reset_bayesian_state(market_id)
            resolved_count += 1
    if resolved_count:
        logger.info(
            "Resolved markets updated: count=%d fetched_missing=%d",
            resolved_count,
            fetched_market_count,
            data={
                "resolved_count": resolved_count,
                "fetched_missing_markets": fetched_market_count,
            },
        )


def _should_skip_flip_refinement(
    market: Market,
    decision: TradeDecision,
    anchor_analysis: dict[str, Any] | None,
    settings: Settings,
) -> tuple[bool, str | None, dict[str, Any] | None]:
    """Detect side-flip candidates that cannot pass flip-guard thresholds."""
    if not settings.FLIP_GUARD_ENABLED or anchor_analysis is None:
        return False, None, None
    anchor_outcome_raw = anchor_analysis.get("outcome")
    anchor_outcome = str(anchor_outcome_raw).strip() if anchor_outcome_raw is not None else ""
    if not anchor_outcome or _outcomes_match(decision.outcome, anchor_outcome):
        return False, None, None

    anchor_confidence = _coerce_float(anchor_analysis.get("confidence")) or 0.0
    confidence_delta = decision.confidence - anchor_confidence
    max_confidence_delta = _MAX_CONFIDENCE - anchor_confidence
    implied_new = _get_implied_probability(market, decision.outcome)
    anchor_edge = _decision_edge_for_outcome(market, anchor_outcome, anchor_confidence)
    edge_delta_ceiling: float | None = None
    if implied_new is not None and anchor_edge is not None:
        max_new_edge = _MAX_CONFIDENCE - implied_new
        edge_delta_ceiling = max_new_edge - anchor_edge

    blocked_reasons: list[str] = []
    if settings.FLIP_GUARD_MIN_ABS_CONFIDENCE > _MAX_CONFIDENCE:
        blocked_reasons.append("abs_confidence_unreachable")
    if max_confidence_delta < settings.FLIP_GUARD_MIN_CONF_GAIN:
        blocked_reasons.append("conf_gain_unreachable")
    if (
        edge_delta_ceiling is not None
        and edge_delta_ceiling < settings.FLIP_GUARD_MIN_EDGE_GAIN
    ):
        blocked_reasons.append("edge_gain_unreachable")

    if not blocked_reasons:
        return False, None, None

    payload = {
        "market_id": market.id,
        "anchor_outcome": anchor_outcome,
        "proposed_outcome": decision.outcome,
        "anchor_confidence": anchor_confidence,
        "proposed_confidence": decision.confidence,
        "confidence_delta": confidence_delta,
        "max_confidence_delta": max_confidence_delta,
        "edge_delta_ceiling": edge_delta_ceiling,
        "flip_guard_min_conf_gain": settings.FLIP_GUARD_MIN_CONF_GAIN,
        "flip_guard_min_edge_gain": settings.FLIP_GUARD_MIN_EDGE_GAIN,
        "flip_guard_min_abs_confidence": settings.FLIP_GUARD_MIN_ABS_CONFIDENCE,
        "precheck_block_reasons": blocked_reasons,
    }
    return True, ",".join(blocked_reasons), payload


def _calculate_bet(max_bet, bet_pct):
    """Calculate bet amount based on confidence-adjusted percentage."""
    bet_pct = max(0.0, min(1.0, bet_pct))
    return max_bet * bet_pct


def _cap_confidence_for_category(
    decision: TradeDecision,
    market: Market,
    settings: Settings,
) -> TradeDecision:
    """Apply confidence caps based on market category to prevent overconfidence."""
    max_conf = _max_confidence_for_market(market, settings)
    is_sports, is_esports = market_category_flags(market)
    family = market_family(market)
    if is_sports:
        cap_reason = "sports"
    elif is_esports:
        cap_reason = "esports"
    elif family == "weather":
        cap_reason = "weather"
    else:
        cap_reason = _market_confidence_family(market)

    if decision.confidence > max_conf:
        logger.info(
            "Capping confidence: market=%s original=%.2f capped=%.2f reason=%s",
            market.id,
            decision.confidence,
            max_conf,
            cap_reason,
            data={
                "market_id": market.id,
                "original_confidence": decision.confidence,
                "capped_confidence": max_conf,
                "cap_reason": cap_reason,
            },
        )
        return decision.model_copy(
            update={
                "confidence": max_conf,
                "bet_size_pct": decision.bet_size_pct * (max_conf / decision.confidence),
                "reasoning": (
                    f"[Confidence capped from {decision.confidence:.2f} to {max_conf:.2f} "
                    f"for {cap_reason}] {decision.reasoning}"
                ),
            }
        )
    
    return decision


def _build_previous_analysis(anchor: dict[str, Any] | None) -> TradeDecision | None:
    if not anchor:
        return None
    outcome = str(anchor.get("outcome") or "").strip()
    confidence = _coerce_float(anchor.get("confidence"))
    reasoning = str(anchor.get("reasoning") or "").strip()
    if not outcome or confidence is None:
        return None
    # Preserve anchor evidence fields so the next Grok turn sees the real
    # quality/source/basis instead of always-zero defaults. Without this,
    # previous_analysis biases the model toward "research gap" framing on
    # every retry, which collapses repeat-analysis quality.
    evidence_quality = _coerce_float(anchor.get("evidence_quality"))
    if evidence_quality is None:
        evidence_quality = 0.0
    edge_source_raw = anchor.get("edge_source")
    edge_source = (
        str(edge_source_raw).strip()
        if isinstance(edge_source_raw, str) and edge_source_raw.strip()
        else None
    )
    evidence_basis_raw = anchor.get("evidence_basis")
    evidence_basis = (
        str(evidence_basis_raw).strip()
        if isinstance(evidence_basis_raw, str) and evidence_basis_raw.strip()
        else None
    )
    implied_prob_external = _coerce_float(anchor.get("implied_prob_external"))
    edge_external = _coerce_float(anchor.get("edge_external"))
    my_prob = _coerce_float(anchor.get("my_prob"))
    if not reasoning:
        reasoning = "Previous cycle analysis."
    # Soft hint: when prior cycle found direct evidence, the next pass should
    # check for material change before defaulting to abstain. Stays advisory.
    if evidence_basis == "direct" and "material change" not in reasoning.lower():
        reasoning = (
            f"{reasoning}\n\n"
            "Note: prior cycle reached direct evidence; check for material "
            "change before defaulting to abstain."
        )
    return TradeDecision(
        should_trade=False,
        outcome=outcome,
        confidence=max(0.0, min(1.0, confidence)),
        bet_size_pct=0.0,
        reasoning=reasoning,
        edge_source=edge_source,
        evidence_basis=evidence_basis,
        evidence_quality=max(0.0, min(1.0, evidence_quality)),
        implied_prob_external=implied_prob_external,
        edge_external=edge_external,
        my_prob=my_prob,
    )


def _should_adjust_position(
    decision: TradeDecision,
    market: Market | None,
    existing_position: Position | None,
    state: MarketState | None,
    settings: Settings,
    cycle_bankroll: float | None = None,
    current_entry_price: float | None = None,
    last_entry_price: float | None = None,
) -> tuple[bool, float, str]:
    """Determine if position should be added to and calculate amount."""
    if not existing_position:
        return True, decision.bet_size_pct, "new_position"

    if (
        settings.OPPOSITE_OUTCOME_STRATEGY == "block"
        and existing_position.outcome
        and existing_position.outcome.upper() != decision.outcome.upper()
    ):
        return False, 0.0, "opposite_outcome_blocked"

    effective_max_position = _effective_max_position_limit_usdc(settings, cycle_bankroll)

    if existing_position.total_amount_usdc >= effective_max_position:
        return False, 0.0, "max_position_reached"

    remaining = effective_max_position - existing_position.total_amount_usdc
    if remaining <= 0:
        return False, 0.0, "no_remaining_capacity"

    override_threshold = _effective_position_override_threshold(market, settings)
    is_high_confidence = decision.confidence >= override_threshold

    # Otherwise, require minimum confidence increase over existing position
    confidence_increase = decision.confidence - existing_position.avg_confidence
    position_fill_ratio = (
        existing_position.total_amount_usdc / max(effective_max_position, 0.01)
    )
    scaled_increase_threshold = settings.MIN_CONFIDENCE_INCREASE_FOR_ADD * max(
        0.25,
        min(1.0, position_fill_ratio),
    )
    meets_increase_threshold = confidence_increase >= scaled_increase_threshold

    if not is_high_confidence and not meets_increase_threshold:
        return False, 0.0, "insufficient_confidence_increase"

    if (
        current_entry_price is not None
        and last_entry_price is not None
        and current_entry_price > 0.0
        and last_entry_price > 0.0
    ):
        relative_move = abs(current_entry_price - last_entry_price) / last_entry_price
        if relative_move < settings.MIN_PRICE_MOVE_FOR_READD:
            return False, 0.0, "insufficient_price_move_for_readd"

    reason = (
        "high_confidence_override"
        if is_high_confidence
        else "confidence_increase_threshold_met"
    )
    return True, min(decision.bet_size_pct, remaining / settings.MAX_BET_USDC), reason


def _effective_max_position_limit_usdc(
    settings: Settings,
    cycle_bankroll: float | None = None,
) -> float:
    """Compute effective per-market position cap for this cycle."""
    effective_max_position = settings.MAX_POSITION_PER_MARKET_USDC
    if cycle_bankroll is not None and cycle_bankroll > 0:
        bankroll_position_cap = cycle_bankroll * settings.MAX_POSITION_PCT_OF_BANKROLL
        effective_max_position = min(effective_max_position, bankroll_position_cap)
    return effective_max_position


def _log_settings_summary(settings) -> None:
    """Log a sanitized summary of current settings."""
    close_days_info = _format_close_days_info(
        settings.MARKET_MIN_CLOSE_DAYS, settings.MARKET_MAX_CLOSE_DAYS
    )
    logger.info(
        "Configuration loaded: dry_run=%s, bet_range=$%.2f-$%.2f, min_confidence=%.2f, "
        "poll_interval=%ds%s",
        settings.DRY_RUN,
        settings.MIN_BET_USDC,
        settings.MAX_BET_USDC,
        settings.MIN_CONFIDENCE,
        settings.POLL_INTERVAL_SEC,
        close_days_info,
        data={
            "dry_run": settings.DRY_RUN,
            "min_bet_usdc": settings.MIN_BET_USDC,
            "max_bet_usdc": settings.MAX_BET_USDC,
            "min_confidence": settings.MIN_CONFIDENCE,
            "confidence_gate_edge_override_enabled": settings.CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED,
            "confidence_gate_min_edge": settings.CONFIDENCE_GATE_MIN_EDGE,
            "confidence_gate_min_evidence_quality": settings.CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY,
            "confidence_gate_override_min_confidence": settings.CONFIDENCE_GATE_OVERRIDE_MIN_CONFIDENCE,
            "min_evidence_quality_for_trade": settings.MIN_EVIDENCE_QUALITY_FOR_TRADE,
            "sports_min_evidence_quality": settings.SPORTS_MIN_EVIDENCE_QUALITY,
            "min_liquidity_usdc": settings.MIN_LIQUIDITY_USDC,
            "min_volume_24h": settings.MIN_VOLUME_24H,
            "min_open_interest": settings.MIN_OPEN_INTEREST,
            "min_tradeable_implied_price": settings.MIN_TRADEABLE_IMPLIED_PRICE,
            "max_tradeable_implied_price": settings.MAX_TRADEABLE_IMPLIED_PRICE,
            "poll_interval_sec": settings.POLL_INTERVAL_SEC,
            "market_min_close_days": settings.MARKET_MIN_CLOSE_DAYS,
            "market_max_close_days": settings.MARKET_MAX_CLOSE_DAYS,
            "grok_model": settings.GROK_MODEL,
            "grok_self_consistency_enabled": settings.GROK_SELF_CONSISTENCY_ENABLED,
            "grok_self_consistency_liquidity_threshold": settings.GROK_SELF_CONSISTENCY_LIQUIDITY_THRESHOLD,
            "grok_self_consistency_edge_threshold": settings.GROK_SELF_CONSISTENCY_EDGE_THRESHOLD,
            "categories_allowlist": settings.MARKET_CATEGORIES_ALLOWLIST,
            "categories_blocklist": settings.MARKET_CATEGORIES_BLOCKLIST,
            "ticker_prefix_blocklist": settings.MARKET_TICKER_BLOCKLIST_PREFIXES,
            "skip_weather_bin_markets": settings.SKIP_WEATHER_BIN_MARKETS,
            "crypto_bin_market_blocklist_enabled": settings.CRYPTO_BIN_MARKET_BLOCKLIST_ENABLED,
            "max_weather_candidates_per_cycle": settings.MAX_WEATHER_CANDIDATES_PER_CYCLE,
            "max_crypto_candidates_per_cycle": settings.MAX_CRYPTO_CANDIDATES_PER_CYCLE,
            "max_speech_candidates_per_cycle": settings.MAX_SPEECH_CANDIDATES_PER_CYCLE,
            "max_music_candidates_per_cycle": settings.MAX_MUSIC_CANDIDATES_PER_CYCLE,
            "max_sports_candidates_per_cycle": settings.MAX_SPORTS_CANDIDATES_PER_CYCLE,
            "weather_min_evidence_quality": settings.WEATHER_MIN_EVIDENCE_QUALITY,
            "direct_source_min_evidence_quality_sports": settings.DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_SPORTS,
            "weather_fallback_edge_min_edge": settings.WEATHER_FALLBACK_EDGE_MIN_EDGE,
            "kalshi_server_side_filters_enabled": settings.KALSHI_SERVER_SIDE_FILTERS_ENABLED,
            "kalshi_max_fetch_pages": settings.KALSHI_MAX_FETCH_PAGES,
            "kalshi_mve_filter": settings.KALSHI_MVE_FILTER,
            "kalshi_eligible_floor": settings.KALSHI_ELIGIBLE_FLOOR,
            "kalshi_fetch_topup_enabled": settings.KALSHI_FETCH_TOPUP_ENABLED,
            "score_gate_mode": settings.SCORE_GATE_MODE,
            "score_gate_threshold": settings.SCORE_GATE_THRESHOLD,
            "score_computed_edge_bonus": settings.SCORE_COMPUTED_EDGE_BONUS,
            "score_repeated_analysis_penalty_base": settings.SCORE_REPEATED_ANALYSIS_PENALTY_BASE,
            "score_repeated_analysis_penalty_start_count": settings.SCORE_REPEATED_ANALYSIS_PENALTY_START_COUNT,
            "score_volume_amplifier_enabled": settings.SCORE_VOLUME_AMPLIFIER_ENABLED,
            "score_confidence_calibration_floor": settings.SCORE_CONFIDENCE_CALIBRATION_FLOOR,
            "score_confidence_calibration_penalty_scale": settings.SCORE_CONFIDENCE_CALIBRATION_PENALTY_SCALE,
            "calibration_online_update_enabled": settings.CALIBRATION_ONLINE_UPDATE_ENABLED,
            "calibration_online_alpha": settings.CALIBRATION_ONLINE_ALPHA,
            "mention_market_score_penalty": settings.MENTION_MARKET_SCORE_PENALTY,
            "pre_analysis_opportunity_enabled": settings.PRE_ANALYSIS_OPPORTUNITY_ENABLED,
            "pre_analysis_opportunity_min_score": settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE,
            "max_markets_per_cycle": settings.MAX_MARKETS_PER_CYCLE,
            "max_trades_per_cycle": settings.MAX_TRADES_PER_CYCLE,
            "bayesian_enabled": settings.BAYESIAN_ENABLED,
            "bayesian_skip_stale_updates": settings.BAYESIAN_SKIP_STALE_UPDATES,
            "bayesian_max_posterior": settings.BAYESIAN_MAX_POSTERIOR,
            "bayesian_max_confidence_boost": settings.BAYESIAN_MAX_CONFIDENCE_BOOST,
            "lmsr_enabled": settings.LMSR_ENABLED,
            "kelly_sizing_enabled": settings.KELLY_SIZING_ENABLED,
            "kelly_dynamic_enabled": settings.KELLY_DYNAMIC_ENABLED,
            "kelly_fraction_default": settings.KELLY_FRACTION_DEFAULT,
            "kelly_fraction_short_horizon_hours": settings.KELLY_FRACTION_SHORT_HORIZON_HOURS,
            "kelly_fraction_short_horizon": settings.KELLY_FRACTION_SHORT_HORIZON,
            "kelly_min_bet_policy": settings.KELLY_MIN_BET_POLICY,
            "fallback_edge_min_edge": settings.FALLBACK_EDGE_MIN_EDGE,
            "coinflip_price_lower": settings.COINFLIP_PRICE_LOWER,
            "coinflip_price_upper": settings.COINFLIP_PRICE_UPPER,
            "max_position_pct_of_bankroll": settings.MAX_POSITION_PCT_OF_BANKROLL,
            "parallel_analysis_enabled": settings.PARALLEL_ANALYSIS_ENABLED,
            "analysis_max_workers": settings.ANALYSIS_MAX_WORKERS,
            "xai_circuit_breaker_max_failures": settings.XAI_CIRCUIT_BREAKER_MAX_FAILURES,
            "xai_client_timeout_seconds": settings.XAI_CLIENT_TIMEOUT_SECONDS,
            "grok_stream_timeout_seconds": settings.GROK_STREAM_TIMEOUT_SECONDS,
            "grok_analysis_max_budget_seconds": settings.GROK_ANALYSIS_MAX_BUDGET_SECONDS,
            "pre_order_market_refresh": settings.PRE_ORDER_MARKET_REFRESH,
            "max_market_data_age_seconds": settings.MAX_MARKET_DATA_AGE_SECONDS,
            "orderbook_precheck_enabled": settings.ORDERBOOK_PRECHECK_ENABLED,
            "orderbook_precheck_min_confidence": settings.ORDERBOOK_PRECHECK_MIN_CONFIDENCE,
            "orderbook_min_resting_volume": settings.ORDERBOOK_MIN_RESTING_VOLUME,
            "order_default_tif": settings.ORDER_DEFAULT_TIF,
            "order_submission_min_price": settings.ORDER_SUBMISSION_MIN_PRICE,
            "order_submission_max_price": settings.ORDER_SUBMISSION_MAX_PRICE,
            "order_fallback_to_market": settings.ORDER_FALLBACK_TO_MARKET,
            "order_fallback_min_confidence": settings.ORDER_FALLBACK_MIN_CONFIDENCE,
            "order_fallback_min_liquidity_usdc": settings.ORDER_FALLBACK_MIN_LIQUIDITY_USDC,
            "calibration_mode_enabled": settings.CALIBRATION_MODE_ENABLED,
            "calibration_min_samples": settings.CALIBRATION_MIN_SAMPLES,
            "position_sync_enabled": settings.POSITION_SYNC_ENABLED,
            "position_sync_interval_cycles": settings.POSITION_SYNC_INTERVAL_CYCLES,
            "opposite_outcome_strategy": settings.OPPOSITE_OUTCOME_STRATEGY,
            "flip_guard_enabled": settings.FLIP_GUARD_ENABLED,
            "flip_guard_min_abs_confidence": settings.FLIP_GUARD_MIN_ABS_CONFIDENCE,
            "flip_guard_min_conf_gain": settings.FLIP_GUARD_MIN_CONF_GAIN,
            "flip_guard_min_edge_gain": settings.FLIP_GUARD_MIN_EDGE_GAIN,
            "flip_guard_min_evidence_quality": settings.FLIP_GUARD_MIN_EVIDENCE_QUALITY,
            "flip_circuit_breaker_enabled": settings.FLIP_CIRCUIT_BREAKER_ENABLED,
            "flip_circuit_breaker_max_flips": settings.FLIP_CIRCUIT_BREAKER_MAX_FLIPS,
            "evidence_quality_high_confidence_override": settings.EVIDENCE_QUALITY_HIGH_CONFIDENCE_OVERRIDE,
        },
    )
    if settings.DRY_RUN:
        logger.warning(
            "DRY_RUN is enabled. No live Kalshi orders will be submitted until DRY_RUN=false.",
            data={"dry_run": True},
        )
    if settings.KELLY_SIZING_ENABLED and settings.MAX_BET_USDC > 0:
        effective_min_bet_pct = settings.MIN_BET_USDC / settings.MAX_BET_USDC
        logger.info(
            "Kelly min-bet policy active: policy=%s min_bet_pct=%.3f",
            settings.KELLY_MIN_BET_POLICY,
            effective_min_bet_pct,
            data={
                "kelly_sizing_enabled": settings.KELLY_SIZING_ENABLED,
                "kelly_min_bet_policy": settings.KELLY_MIN_BET_POLICY,
                "min_bet_usdc": settings.MIN_BET_USDC,
                "max_bet_usdc": settings.MAX_BET_USDC,
                "effective_min_bet_pct": round(effective_min_bet_pct, 6),
            },
        )


def _format_close_days_info(min_days, max_days) -> str:
    """Format close days filter info for logging."""
    if min_days is None and max_days is None:
        return ""
    parts = []
    if min_days is not None:
        parts.append(f"min={min_days}d")
    if max_days is not None:
        parts.append(f"max={max_days}d")
    return f", close_window=[{', '.join(parts)}]"


def _build_kalshi_market_fetch_window(
    min_close_days: int | None,
    max_close_days: int | None,
) -> tuple[datetime | None, datetime | None]:
    now = datetime.now(timezone.utc)
    start = now + timedelta(days=min_close_days) if min_close_days is not None else None
    end = now + timedelta(days=max_close_days) if max_close_days is not None else None
    return start, end


def _fetch_markets_with_optional_server_filters(
    kalshi_client: KalshiClient,
    *,
    use_server_side_filters: bool,
    fetch_window_start: datetime | None,
    fetch_window_end: datetime | None,
    mve_filter: str | None = None,
) -> list[Market]:
    if not use_server_side_filters:
        return kalshi_client.get_markets(mve_filter=mve_filter)
    try:
        return kalshi_client.get_markets(
            close_time_start=fetch_window_start,
            close_time_end=fetch_window_end,
            mve_filter=mve_filter,
        )
    except Exception as exc:
        logger.warning(
            "Kalshi server-side filters failed; attempting filtered retry before unfiltered fallback: %s",
            exc,
            data={
                "error": str(exc),
                "close_time_start": fetch_window_start.isoformat()
                if fetch_window_start
                else None,
                "close_time_end": fetch_window_end.isoformat()
                if fetch_window_end
                else None,
                "mve_filter": mve_filter,
            },
        )
        kalshi_client.reset_session()
        try:
            return kalshi_client.get_markets(
                close_time_start=fetch_window_start,
                close_time_end=fetch_window_end,
                mve_filter=mve_filter,
            )
        except Exception as retry_exc:
            logger.warning(
                "Kalshi filtered retry failed; falling back to unfiltered fetch: %s",
                retry_exc,
                data={
                    "error": str(retry_exc),
                    "close_time_start": fetch_window_start.isoformat()
                    if fetch_window_start
                    else None,
                    "close_time_end": fetch_window_end.isoformat()
                    if fetch_window_end
                    else None,
                    "mve_filter": mve_filter,
                },
            )
            return kalshi_client.get_markets(mve_filter=mve_filter)


def _requires_market_refresh(
    *,
    pre_order_market_refresh: bool,
    market_data_age_seconds: float | None,
    max_market_data_age_seconds: int,
) -> bool:
    if pre_order_market_refresh:
        return True
    if market_data_age_seconds is None:
        return False
    return market_data_age_seconds > max_market_data_age_seconds


def _can_use_lenient_stale_refresh_fallback(
    *,
    evidence_basis_class: str,
    pre_execution_final_score: float,
    market_data_age_seconds: float | None,
    settings: Settings,
) -> bool:
    if market_data_age_seconds is None:
        return False
    if market_data_age_seconds <= float(settings.MAX_MARKET_DATA_AGE_SECONDS):
        return False
    if str(evidence_basis_class or "").strip().lower() != "direct":
        return False
    if float(pre_execution_final_score) < float(settings.SCORE_GATE_THRESHOLD):
        return False
    lenient_max_age_seconds = (
        float(settings.MAX_MARKET_DATA_AGE_SECONDS)
        * _STALE_REFRESH_LENIENT_AGE_MULTIPLIER
    )
    return market_data_age_seconds <= lenient_max_age_seconds


def _confidence_gate_override_metrics(
    market: Market,
    decision: TradeDecision,
) -> tuple[float | None, float | None]:
    implied_prob = _get_implied_probability(market, decision.outcome)
    market_edge = (decision.confidence - implied_prob) if implied_prob is not None else None
    model_edge = decision.edge_external
    if model_edge is not None and market_edge is not None:
        return (max(model_edge, market_edge), market_edge)
    if model_edge is not None:
        return (model_edge, market_edge)
    return (market_edge, market_edge)


def _is_confidence_override_allowed(
    *,
    settings: Settings,
    decision: TradeDecision,
    override_edge: float | None,
) -> tuple[bool, float, str]:
    override_min_confidence = max(
        0.0,
        min(1.0, settings.CONFIDENCE_GATE_OVERRIDE_MIN_CONFIDENCE),
    )
    edge_default_allowed = (
        settings.CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED
        and override_edge is not None
        and override_edge >= settings.CONFIDENCE_GATE_MIN_EDGE
        and decision.evidence_quality >= settings.CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY
        and decision.confidence >= override_min_confidence
    )
    if edge_default_allowed:
        return True, override_min_confidence, "edge_default"

    strong_floor = max(0.0, min(1.0, settings.STRONG_EVIDENCE_CONFIDENCE_FLOOR))
    strong_eq_min = max(0.0, settings.STRONG_EVIDENCE_MIN_EVIDENCE_QUALITY)
    source_url = str(getattr(decision, "primary_source_url", "") or "").strip().lower()
    source_whitelisted = any(
        domain in source_url
        for domain in settings.DIRECT_SOURCE_WHITELIST
    ) if source_url else False
    evidence_basis = str(getattr(decision, "evidence_basis", "") or "").strip().lower()
    edge_source = str(getattr(decision, "edge_source", "") or "").strip().lower()
    strong_evidence_allowed = (
        evidence_basis == "direct"
        and decision.evidence_quality >= strong_eq_min
        and source_whitelisted
        and edge_source == "computed"
        and override_edge is not None
        and abs(override_edge) >= settings.CONFIDENCE_GATE_MIN_EDGE
        and decision.confidence >= strong_floor
    )
    if strong_evidence_allowed:
        return True, strong_floor, "strong_direct_evidence"

    proxy_eq_min = max(0.0, settings.STRONG_EVIDENCE_PROXY_MIN_EVIDENCE_QUALITY)
    proxy_edge_min = max(0.0, settings.STRONG_EVIDENCE_PROXY_MIN_EDGE)
    strong_proxy_allowed = (
        evidence_basis == "proxy"
        and decision.evidence_quality >= proxy_eq_min
        and source_whitelisted
        and edge_source == "computed"
        and override_edge is not None
        and abs(override_edge) >= proxy_edge_min
        and decision.confidence >= strong_floor
    )
    if strong_proxy_allowed:
        return True, strong_floor, "strong_proxy_evidence"

    return False, override_min_confidence, "none"


def _record_terminal_outcome(
    state_manager: MarketStateManager,
    market_id: str,
    terminal_outcome: str,
) -> None:
    try:
        state_manager.record_terminal_outcome(market_id, terminal_outcome)
    except Exception as exc:
        logger.debug(
            "Failed to persist terminal outcome: market=%s outcome=%s error=%s",
            market_id,
            terminal_outcome,
            exc,
            data={
                "market_id": market_id,
                "terminal_outcome": terminal_outcome,
                "error": str(exc),
            },
        )


def _record_rejection_reason(
    rejection_breakdown: dict[str, int],
    reason: str,
) -> None:
    rejection_breakdown[reason] = rejection_breakdown.get(reason, 0) + 1


def _summarize_distribution(samples: list[float]) -> dict[str, float | int]:
    """Return a compact distribution summary for cycle-receipt telemetry.

    Empty samples produce ``{"count": 0}`` so consumers can disambiguate
    "no markets scored" from "all markets scored zero". Percentiles use a
    simple linear interpolation that matches numpy's default ``linear``
    method without pulling numpy into this module.
    """
    if not samples:
        return {"count": 0}
    ordered = sorted(samples)
    count = len(ordered)

    def _percentile(p: float) -> float:
        if count == 1:
            return float(ordered[0])
        rank = (count - 1) * p
        low = int(rank)
        high = min(count - 1, low + 1)
        weight = rank - low
        return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)

    return {
        "count": count,
        "min": round(float(ordered[0]), 4),
        "p25": round(_percentile(0.25), 4),
        "p50": round(_percentile(0.50), 4),
        "p75": round(_percentile(0.75), 4),
        "max": round(float(ordered[-1]), 4),
    }


def _iter_exchange_position_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in ("market_positions", "positions", "portfolio_positions", "data"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _parse_exchange_position_row(row: dict[str, Any]) -> tuple[str, str, float, int] | None:
    market_id = str(
        row.get("ticker")
        or row.get("market_ticker")
        or row.get("market_id")
        or ""
    ).strip()
    if not market_id:
        return None
    contracts_raw = row.get("position") or row.get("position_fp")
    if contracts_raw is None:
        yes_count = float(row.get("yes_count") or row.get("yes_count_fp") or 0.0)
        no_count = float(row.get("no_count") or row.get("no_count_fp") or 0.0)
        contracts_raw = yes_count - no_count
    try:
        contracts = int(float(contracts_raw or 0.0))
    except (TypeError, ValueError):
        return None
    if contracts == 0:
        return None
    outcome = "YES" if contracts > 0 else "NO"
    exposure_raw = row.get("market_exposure_dollars")
    amount_usdc = _coerce_float(exposure_raw) or 0.0
    if amount_usdc <= 0:
        amount_usdc = float(abs(contracts))
    return market_id, outcome, abs(amount_usdc), abs(contracts)


def _sync_positions_from_exchange(
    *,
    state_manager: MarketStateManager,
    kalshi_client: KalshiClient,
) -> tuple[int, int]:
    payload = kalshi_client.get_positions()
    rows = _iter_exchange_position_rows(payload)
    synced = 0
    local_updates = 0
    for row in rows:
        parsed = _parse_exchange_position_row(row)
        if parsed is None:
            continue
        market_id, outcome, amount_usdc, contracts = parsed
        existing = state_manager.get_position(market_id)
        if existing is not None and (
            existing.outcome != outcome
            or abs(existing.total_amount_usdc - amount_usdc) > 0.01
        ):
            local_updates += 1
        state_manager.upsert_position_snapshot(
            market_id=market_id,
            outcome=outcome,
            total_amount_usdc=amount_usdc,
        )
        synced += 1
        logger.debug(
            "Position sync row: market=%s outcome=%s contracts=%d amount_usdc=%.4f",
            market_id,
            outcome,
            contracts,
            amount_usdc,
            data={
                "market_id": market_id,
                "outcome": outcome,
                "contracts": contracts,
                "amount_usdc": amount_usdc,
            },
        )
    return synced, local_updates


def _iter_exchange_settlement_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in ("settlements", "market_settlements", "data"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _parse_exchange_settlement_row(row: dict[str, Any]) -> dict[str, Any] | None:
    settlement_id = str(
        row.get("settlement_id")
        or row.get("id")
        or row.get("trade_id")
        or row.get("market_ticker")
        or row.get("ticker")
        or ""
    ).strip()
    market_id = str(
        row.get("market_ticker")
        or row.get("ticker")
        or row.get("market_id")
        or ""
    ).strip()
    if not settlement_id or not market_id:
        return None
    winning_outcome_raw = str(
        row.get("market_result")
        or row.get("result")
        or row.get("winning_outcome")
        or ""
    ).strip().upper()
    winning_outcome = winning_outcome_raw if winning_outcome_raw in {"YES", "NO"} else None
    yes_contracts = int(
        _coerce_float(
            row.get("yes_count")
            or row.get("yes_count_fp")
            or row.get("yes_contracts_owned")
            or 0
        ) or 0.0
    )
    no_contracts = int(
        _coerce_float(
            row.get("no_count")
            or row.get("no_count_fp")
            or row.get("no_contracts_owned")
            or 0
        ) or 0.0
    )
    predicted_outcome: str | None = None
    contracts = 0
    avg_price: float | None = None
    cost_dollars: float = 0.0
    if yes_contracts > 0:
        predicted_outcome = "YES"
        contracts = yes_contracts
        cost_dollars_raw = _coerce_float(row.get("yes_total_cost_dollars"))
        if cost_dollars_raw is not None:
            cost_dollars = cost_dollars_raw
            avg_price = cost_dollars / yes_contracts if yes_contracts > 0 else None
        else:
            avg_price = _coerce_float(
                row.get("yes_total_cost") or row.get("yes_contracts_average_price")
            )
            if avg_price is None:
                avg_price = _coerce_float(row.get("yes_contracts_average_price_in_cents"))
                if avg_price is not None and avg_price > 1.0:
                    avg_price /= 100.0
            if avg_price is not None:
                cost_dollars = avg_price * yes_contracts
    elif no_contracts > 0:
        predicted_outcome = "NO"
        contracts = no_contracts
        cost_dollars_raw = _coerce_float(row.get("no_total_cost_dollars"))
        if cost_dollars_raw is not None:
            cost_dollars = cost_dollars_raw
            avg_price = cost_dollars / no_contracts if no_contracts > 0 else None
        else:
            avg_price = _coerce_float(
                row.get("no_total_cost") or row.get("no_contracts_average_price")
            )
            if avg_price is None:
                avg_price = _coerce_float(row.get("no_contracts_average_price_in_cents"))
                if avg_price is not None and avg_price > 1.0:
                    avg_price /= 100.0
            if avg_price is not None:
                cost_dollars = avg_price * no_contracts

    profit = _coerce_float(
        row.get("profit")
        or row.get("profit_in_dollars")
        or row.get("pnl")
        or row.get("realized_pnl")
    )
    if profit is None:
        revenue_raw = _coerce_float(row.get("revenue"))
        fee_raw = _coerce_float(row.get("fee_cost"))
        revenue_dollars = (revenue_raw / 100.0) if revenue_raw is not None else 0.0
        fee_dollars = fee_raw if fee_raw is not None else 0.0
        profit = revenue_dollars - cost_dollars - fee_dollars

    settled_at = (
        _coerce_datetime(row.get("settled_time"))
        or _coerce_datetime(row.get("created_time"))
        or _coerce_datetime(row.get("created_at"))
    )
    return {
        "settlement_id": settlement_id,
        "market_id": market_id,
        "winning_outcome": winning_outcome,
        "predicted_outcome": predicted_outcome,
        "pnl_realized": float(profit or 0.0),
        "contracts": contracts if contracts > 0 else None,
        "avg_price": avg_price,
        "settled_at": settled_at,
        "raw": row,
    }


def _sync_settlements_from_exchange(
    *,
    state_manager: MarketStateManager,
    kalshi_client: KalshiClient,
    settings: Settings | None = None,
    limit: int = 200,
) -> int:
    payload = kalshi_client.get_settlements(limit=limit)
    rows = _iter_exchange_settlement_rows(payload)
    imported = 0
    for row in rows:
        parsed = _parse_exchange_settlement_row(row)
        if parsed is None:
            continue
        state_manager.record_exchange_settlement(
            **parsed,
            online_calibration_enabled=(
                bool(getattr(settings, "CALIBRATION_ONLINE_UPDATE_ENABLED", False))
                if settings is not None
                else False
            ),
            online_calibration_alpha=(
                float(getattr(settings, "CALIBRATION_ONLINE_ALPHA", 0.15))
                if settings is not None
                else 0.15
            ),
            online_calibration_max_samples_per_bucket=(
                int(getattr(settings, "CALIBRATION_ONLINE_MAX_SAMPLES_PER_BUCKET", 500))
                if settings is not None
                else 500
            ),
        )
        imported += 1
    return imported


def _detect_external_fills(
    *,
    state_manager: MarketStateManager,
    kalshi_client: KalshiClient,
    limit: int = 200,
) -> int:
    payload = kalshi_client.get_fills(limit=limit)
    rows = payload.get("fills")
    if not isinstance(rows, list):
        rows = payload.get("data")
    if not isinstance(rows, list):
        return 0
    known_order_ids = state_manager.get_known_order_ids()
    external_count = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        order_id = str(
            row.get("order_id")
            or row.get("orderId")
            or row.get("id")
            or row.get("trade_id")
            or ""
        ).strip()
        if not order_id:
            continue
        if order_id not in known_order_ids:
            external_count += 1
    return external_count


def _is_coinflip_signal(decision: TradeDecision) -> bool:
    return decision.confidence <= 0.55 and decision.evidence_quality < 0.60


def _analysis_result_rank(
    result: dict[str, Any] | None,
    *,
    historical_family_pnl_total: float | None = None,
    historical_family_sample_size: int = 0,
    historical_family_win_rate: float | None = None,
) -> tuple[float, float, float, float, float, float, float, float, float, float, float]:
    if not result:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    decision = result.get("decision")
    if not isinstance(decision, TradeDecision):
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    should_trade_rank = 1.0 if decision.should_trade and not decision.abstain else 0.0
    final_score_rank = float(result.get("pre_execution_final_score", 0.0) or 0.0)
    score_rank = final_score_rank + (0.02 * should_trade_rank)
    critical_rejection_rank = 0.0
    critical_rejection_reasons = _score_gate_critical_rejection_reasons(
        rejection_reasons=result.get("pre_execution_rejection_reasons", ()),
        evidence_basis_class=_decision_evidence_basis(decision),
        edge_source=decision.edge_source,
        definitive_outcome_eligible=bool(
            getattr(decision, "definitive_outcome_detected", False)
        ),
    )
    if critical_rejection_reasons:
        critical_rejection_rank = -1.0
    profitable_family_rank = 1.0 if (
        historical_family_pnl_total is not None
        and int(historical_family_sample_size) >= 8
        and float(historical_family_pnl_total) > 10.0
    ) else 0.0
    historical_family_win_rate_rank = 0.0
    if int(historical_family_sample_size) >= 8:
        historical_family_win_rate_rank = max(
            0.0,
            min(1.0, float(historical_family_win_rate or 0.0)),
        )
    overconfidence_gap_penalty_rank = -max(0.0, decision.confidence - decision.evidence_quality)
    evidence_basis_rank = 1.0 if _decision_evidence_basis(decision) == "direct" else 0.0
    primary_source_rank = (
        1.0 if str(getattr(decision, "primary_source_url", "") or "").strip() else 0.0
    )
    edge_external_rank = float(decision.edge_external or 0.0)
    evidence_rank = max(0.0, min(1.0, decision.evidence_quality))
    confidence_rank = max(0.0, min(1.0, decision.confidence))
    return (
        critical_rejection_rank,
        score_rank,
        profitable_family_rank,
        historical_family_win_rate_rank,
        overconfidence_gap_penalty_rank,
        evidence_basis_rank,
        primary_source_rank,
        edge_external_rank,
        evidence_rank,
        confidence_rank,
        should_trade_rank,
    )


def _event_ticker_prefix(market: Market) -> str:
    event_ticker = str(market.event_ticker or "").strip().upper()
    if event_ticker:
        return event_ticker
    market_id = str(market.id or "").strip().upper()
    if "-" in market_id:
        return market_id.rsplit("-", maxsplit=1)[0]
    return market_id


def _daily_balance_delta_usdc(
    *,
    day_start_balance: float | None,
    current_balance: float | None,
) -> float | None:
    if day_start_balance is None or current_balance is None:
        return None
    return float(current_balance) - float(day_start_balance)


def _event_concentration_blocked(
    *,
    max_bets_per_event: int,
    open_other_positions_count: int,
    cycle_other_attempts_count: int,
) -> bool:
    if max_bets_per_event <= 0:
        return False
    return (open_other_positions_count + cycle_other_attempts_count) >= max_bets_per_event


def _event_side_conflict_blocked(
    *,
    proposed_outcome: str,
    open_event_outcomes: set[str],
    cycle_event_outcomes: set[str],
) -> tuple[bool, list[str]]:
    normalized_proposed = _normalize_outcome_key(proposed_outcome)
    if not normalized_proposed:
        return False, []
    existing_outcomes = sorted(
        {
            _normalize_outcome_key(outcome)
            for outcome in (open_event_outcomes | cycle_event_outcomes)
            if _normalize_outcome_key(outcome)
        }
    )
    if not existing_outcomes:
        return False, []
    has_conflict = any(outcome != normalized_proposed for outcome in existing_outcomes)
    return has_conflict, existing_outcomes


def _should_apply_definitive_side_override(
    *,
    decision: TradeDecision,
    evidence_basis: str,
    primary_source_whitelisted: bool = False,
    cycle_overrides_applied: int,
    max_overrides_per_cycle: int,
) -> bool:
    """Allow an event-side-conflict override only under strict definitive conditions."""
    if cycle_overrides_applied >= max_overrides_per_cycle:
        return False
    if not getattr(decision, "definitive_outcome_detected", False):
        return False
    if str(evidence_basis or "").strip().lower() != "direct":
        return False
    is_whitelisted = primary_source_whitelisted or getattr(
        decision, "primary_source_whitelisted", False
    )
    if not is_whitelisted:
        return False
    lr = getattr(decision, "likelihood_ratio", None)
    if lr is None or float(lr) < 10.0:
        return False
    raw_conf = float(getattr(decision, "raw_confidence", None) or decision.confidence)
    if raw_conf < 0.80:
        return False
    return True


def _daily_trade_cap_reached(*, daily_trade_count: int, max_trades_per_day: int) -> bool:
    if max_trades_per_day <= 0:
        return False
    return daily_trade_count >= max_trades_per_day


def _daily_drawdown_cap_reached(
    *,
    daily_balance_delta: float | None,
    max_daily_drawdown_usdc: float,
) -> bool:
    if max_daily_drawdown_usdc <= 0:
        return False
    if daily_balance_delta is None:
        return False
    return max(0.0, -float(daily_balance_delta)) >= max_daily_drawdown_usdc


def _estimate_api_cost_usd(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    settings: Settings,
) -> float:
    input_rate = max(0.0, float(settings.API_COST_INPUT_PER_1K_TOKENS_USD))
    output_rate = max(0.0, float(settings.API_COST_OUTPUT_PER_1K_TOKENS_USD))
    billable_prompt_tokens = max(0, int(prompt_tokens) - max(0, int(cached_tokens)))
    return ((billable_prompt_tokens / 1000.0) * input_rate) + (
        (max(0, int(completion_tokens)) / 1000.0) * output_rate
    )


def _build_execution_audit(
    *,
    decision_phase: str | None = None,
    decision_terminal: bool | None = None,
    final_action: str | None = None,
    final_reason: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a compact execution audit payload with canonical keys."""
    alias_to_canonical = {
        "amount_usdc": "bet_amount_usdc",
        "score_value": "score_final",
        "confidence_gate_override_edge": "override_edge",
        "confidence_gate_override_market_edge": "market_edge",
        "implied_prob": "implied_prob_market",
        "edge": "edge_market",
        "audit_entry_price": "entry_price",
        "audit_implied_prob_market": "implied_prob_market",
        "audit_edge_source": "edge_source",
    }
    payload: dict[str, Any] = {}
    normalized_final_action = str(final_action or "").strip().lower()
    if decision_phase is not None:
        payload["decision_phase"] = decision_phase
    if decision_terminal is not None:
        payload["decision_terminal"] = decision_terminal
    if final_action is not None:
        payload["final_action"] = final_action
    if final_reason is not None:
        payload["final_reason"] = final_reason
        if normalized_final_action in {"skip", "research_queued"}:
            payload["rejection_reason"] = final_reason
        if normalized_final_action in {"skip", "research_queued"}:
            payload["skip_reasons"] = [str(final_reason)]
    for key, value in extra.items():
        if value is not None:
            if key in payload and payload[key] != value:
                logger.debug(
                    "Execution audit key override: key=%s old=%r new=%r",
                    key,
                    payload[key],
                    value,
                    data={"key": key, "old_value": payload[key], "new_value": value},
                )
            payload[key] = value
    # Mark synthetic placeholder decisions so analytics scripts can partition
    # "real Grok output" from queue/timeout/cap routing without string-matching.
    decision_origin_value = payload.get("decision_origin")
    if isinstance(decision_origin_value, str) and decision_origin_value.startswith("synthetic_"):
        payload.setdefault("synthetic_decision", True)
    elif "synthetic_decision" not in payload:
        payload["synthetic_decision"] = False
    # Flatten historical family fields from the pre_analysis_breakdown blob to
    # top-level audit keys so SQL analytics can read them without JSON-path
    # navigation. Top-level fields win if both are set.
    breakdown = payload.get("pre_analysis_breakdown")
    if isinstance(breakdown, dict):
        for breakdown_key, audit_key in (
            ("pre_score_historical_family_samples", "historical_family_samples"),
            ("pre_score_historical_family_pnl_total", "historical_family_pnl_total"),
            ("pre_score_historical_family_win_rate", "historical_family_win_rate"),
            ("pre_score_historical_family_pnl_ratio", "historical_family_pnl_ratio"),
        ):
            value = breakdown.get(breakdown_key)
            if value is not None:
                payload.setdefault(audit_key, value)
    if final_reason and "rejection_stage" not in payload:
        if str(final_reason).startswith("pre_analysis_"):
            payload["rejection_stage"] = "pre_analysis"
        elif str(final_reason) in {"no_trade_recommended", "abstain_low_evidence"}:
            payload["rejection_stage"] = "validation"
        elif str(final_reason) in {"score_gate_blocked", "score_gate_critical_rejection"}:
            payload["rejection_stage"] = "score_gate"
        elif str(final_reason).endswith("_blocked") or str(final_reason).endswith("_below_min"):
            payload["rejection_stage"] = "execution_gate"
    for alias_key, canonical_key in alias_to_canonical.items():
        if (
            alias_key in payload
            and canonical_key in payload
            and payload[alias_key] != payload[canonical_key]
        ):
            logger.debug(
                "Execution audit canonical key preferred over alias: alias=%s canonical=%s",
                alias_key,
                canonical_key,
                data={
                    "alias_key": alias_key,
                    "canonical_key": canonical_key,
                    "alias_value": payload[alias_key],
                    "canonical_value": payload[canonical_key],
                },
            )
        if alias_key in payload and canonical_key not in payload:
            payload[canonical_key] = payload[alias_key]
        payload.pop(alias_key, None)
    return payload


def _participation_decision_for_audit(
    *,
    audit: dict[str, Any],
    decision: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> ParticipationDecision | None:
    """Infer canonical participation fields for receipts that did not set them.

    This is audit-only. It does not change routing, order submission, sizing, or
    any risk gate; it only normalizes downstream skip/research/order receipts so
    performance review can compare gates with the same participation vocabulary.
    """
    audit_payload = audit or {}
    decision_payload = decision or {}
    final_action = str(audit_payload.get("final_action") or "").strip().lower()
    final_reason = str(audit_payload.get("final_reason") or "").strip()
    normalized_reason = final_reason.lower()
    if not final_action and not normalized_reason:
        return None

    if final_action in {"order_attempt", "order_submitted", "dry_run"}:
        return ParticipationDecision(
            tier=ParticipationTier.EXECUTION_ELIGIBLE,
            primary_reason=final_reason or "all_gates_passed",
            tier_metadata={"skip_due_to": None},
        )

    if "timeout" in normalized_reason:
        timeout_streak = int(audit_payload.get("timeout_streak") or 1)
        if final_action == "monitor_only":
            timeout_streak = max(timeout_streak, 2)
        return classify_participation(
            timeout_state=TimeoutState(
                timed_out=True,
                retriable=True,
                timeout_streak=timeout_streak,
                search_profile=str(audit_payload.get("search_profile") or "generic"),
            )
        )

    if "analysis_failure" in normalized_reason:
        return classify_participation(
            analysis_failed=True,
            analysis_error_retriable="retriable" in normalized_reason,
        )

    if normalized_reason.startswith("pre_analysis_"):
        historical_gate = None
        if (
            audit_payload.get("historical_gate_prefix_sample_size") is not None
            or audit_payload.get("historical_gate_sample_size") is not None
        ):
            sample_size_raw = (
                audit_payload.get("historical_gate_prefix_sample_size")
                if audit_payload.get("historical_gate_prefix_sample_size") is not None
                else audit_payload.get("historical_gate_sample_size")
            )
            sample_size = int(_coerce_float(sample_size_raw) or 0)
            historical_gate = HistoricalGateResult(
                allowed=bool(audit_payload.get("historical_gate_allowed", True)),
                reason=audit_payload.get("historical_gate_reason"),
                metrics=audit_payload,
                sample_size=sample_size,
                wilson_win_rate_lower_bound=_coerce_float(
                    audit_payload.get("historical_gate_wilson_lb")
                ),
                shrunk_pnl_per_trade=_coerce_float(
                    audit_payload.get("historical_gate_shrunk_pnl_per_trade")
                ),
            )
        return classify_participation(
            historical_gate=historical_gate,
            pre_analysis_rejection_reason=final_reason,
            pre_analysis_metadata=audit_payload,
        )

    evidence_quality_for_audit = _coerce_float(
        decision_payload.get("evidence_quality", audit_payload.get("evidence_quality"))
    )
    evidence_basis_for_audit = str(
        decision_payload.get("evidence_basis")
        or audit_payload.get("evidence_basis")
        or audit_payload.get("evidence_basis_class")
        or ""
    )
    edge_source_for_audit = str(
        decision_payload.get("edge_source")
        or audit_payload.get("edge_source")
        or ""
    )
    primary_source_url_for_audit = str(
        decision_payload.get("primary_source_url")
        or audit_payload.get("primary_source_url")
        or ""
    ).strip()
    primary_source_whitelisted_for_audit = (
        _is_whitelisted_primary_source_url(primary_source_url_for_audit, settings)
        if settings is not None and primary_source_url_for_audit
        else False
    )
    min_evidence_quality_for_audit = _coerce_float(
        audit_payload.get("min_evidence_quality")
    )
    if min_evidence_quality_for_audit is None and settings is not None:
        min_evidence_quality_for_audit = float(settings.MIN_EVIDENCE_QUALITY_FOR_TRADE)

    if normalized_reason in {"abstain_low_evidence"}:
        return classify_participation(
            decision_abstain=True,
            decision_evidence_basis=evidence_basis_for_audit,
            decision_edge_source=edge_source_for_audit,
            decision_primary_source_whitelisted=primary_source_whitelisted_for_audit,
            decision_evidence_quality=evidence_quality_for_audit,
            evidence_quality_threshold=min_evidence_quality_for_audit,
            pre_analysis_metadata=audit_payload,
        )
    if normalized_reason in {"no_trade_recommended"}:
        return classify_participation(
            decision_should_trade=False,
            decision_evidence_basis=evidence_basis_for_audit,
            decision_edge_source=edge_source_for_audit,
            decision_primary_source_whitelisted=primary_source_whitelisted_for_audit,
            decision_evidence_quality=evidence_quality_for_audit,
            evidence_quality_threshold=min_evidence_quality_for_audit,
            pre_analysis_metadata=audit_payload,
        )

    should_trade_raw = decision_payload.get("should_trade")
    should_trade = (
        bool(should_trade_raw)
        if isinstance(should_trade_raw, bool)
        else str(should_trade_raw).strip().lower() in {"1", "true", "yes"}
    )
    if should_trade:
        evidence_quality = evidence_quality_for_audit
        confidence = _coerce_float(decision_payload.get("confidence"))
        min_evidence_quality = min_evidence_quality_for_audit
        min_confidence = _coerce_float(
            audit_payload.get("counterfactual_required_confidence")
        )
        if min_confidence is None and settings is not None:
            min_confidence = float(settings.MIN_CONFIDENCE)
        edge_value = _coerce_float(
            audit_payload.get(
                "gate_edge_actual",
                audit_payload.get("edge_market", audit_payload.get("score_edge_market")),
            )
        )
        primary_source_whitelisted = primary_source_whitelisted_for_audit
        score_gate_reason = (
            normalized_reason
            if "score_gate" in normalized_reason
            else None
        )
        downstream_gate_reason = None
        if (
            normalized_reason
            and normalized_reason
            not in {
                "confidence_below_min",
                "evidence_quality_below_min",
                "weather_evidence_quality_below_min",
            }
            and score_gate_reason is None
        ):
            downstream_gate_reason = normalized_reason
        return classify_participation(
            pre_analysis_metadata=audit_payload,
            decision_should_trade=True,
            decision_abstain=bool(decision_payload.get("abstain")),
            decision_definitive_outcome=bool(
                decision_payload.get("definitive_outcome_detected")
            ),
            decision_evidence_basis=str(evidence_basis_for_audit),
            decision_edge_source=str(edge_source_for_audit),
            decision_primary_source_whitelisted=primary_source_whitelisted,
            decision_evidence_quality=evidence_quality,
            evidence_quality_threshold=min_evidence_quality,
            confidence_value=confidence,
            confidence_threshold=(
                min_confidence
                if normalized_reason == "confidence_below_min"
                else None
            ),
            edge_value=(
                edge_value
                if normalized_reason in {"edge_gate_blocked", "edge_above_reasonable_max"}
                else None
            ),
            edge_reasonable_max=(
                float(settings.MAX_REASONABLE_EDGE)
                if settings is not None
                else 0.35
            ),
            definitive_edge_reasonable_max=(
                float(settings.DEFINITIVE_OUTCOME_EDGE_REASONABLE_MAX)
                if settings is not None
                else 0.65
            ),
            score_gate_blocked=score_gate_reason is not None,
            score_gate_reason=score_gate_reason,
            downstream_gate_reason=downstream_gate_reason,
        )

    if final_action in {"skip", "research_queued", "monitor_only"}:
        tier = (
            ParticipationTier.MONITOR_ONLY
            if final_action == "monitor_only"
            else ParticipationTier.SKIP_FOR_NOW_WITH_REASON
        )
        return ParticipationDecision(
            tier=tier,
            primary_reason=final_reason or final_action,
            why_not_execution_eligible=f"Final action {final_action}: {final_reason}",
            what_to_learn_next="Review final gate reason and market outcome before promoting.",
            tier_metadata={
                **audit_payload,
                "skip_due_to": _skip_due_to_for_reason(final_reason, audit_payload),
            },
        )

    return None


def _apply_participation_audit_fields(
    audit: dict[str, Any],
    *,
    decision: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> bool:
    """Stamp missing canonical participation fields onto an audit payload."""
    if not isinstance(audit, dict):
        return False
    had_tier = bool(audit.get("participation_tier"))
    participation_decision = _participation_decision_for_audit(
        audit=audit,
        decision=decision,
        settings=settings,
    )
    if participation_decision is None:
        return False
    _, _, metadata = participation_decision.to_metadata_tuple()
    for key, value in metadata.items():
        if value is not None:
            audit.setdefault(key, value)
    audit.setdefault("participation_tier", str(participation_decision.tier))
    audit.setdefault("participation_decision", participation_decision.primary_reason)
    audit.setdefault(
        "participation_terminal_reject",
        participation_decision.tier == ParticipationTier.TERMINAL_REJECT,
    )
    if (
        audit.get("final_action") in {"skip", "research_queued", "monitor_only"}
        and "skip_due_to" not in audit
    ):
        audit["skip_due_to"] = _skip_due_to_for_reason(
            audit.get("final_reason"),
            audit,
        )
    return not had_tier and bool(audit.get("participation_tier"))


# Audit/decision fields stamped on synthetic research-queue / cap / non-actionable
# decisions where Grok was never invoked. They make it explicit downstream that
# the placeholder (eq=0.0/edge_source=none/basis=absence_only) is a research gap,
# not a Grok finding, and that final_action="research_queued" is NOT a hard reject.
_SYNTHETIC_DECISION_AUDIT_FIELDS: dict[str, Any] = {
    "analysis_skipped": True,
    "evidence_quality_unevaluated": True,
    "edge_source_unevaluated": True,
    "pre_analysis_hard_reject": False,
}


def _skip_due_to_for_reason(reason: str | None, audit: dict[str, Any] | None = None) -> str:
    normalized = str(reason or "").strip().lower()
    audit_payload = audit or {}
    if "timeout" in normalized:
        return "timeout"
    if "analysis_failure" in normalized or "api" in normalized or "quota" in normalized:
        return "operational_failure"
    if "source" in normalized or "evidence" in normalized or "hallucinated" in normalized:
        return "lack_of_evidence"
    if "stale" in normalized:
        return "stale_evidence"
    if "daily_drawdown" in normalized:
        return "risk_cap"
    if "historical" in normalized or "prefix" in normalized or "family_pnl" in normalized:
        return "poor_historical_prefix"
    if "resolution" in normalized or "ambiguous" in normalized or "outcome_mismatch" in normalized:
        return "ambiguous_resolution"
    if "pre_analysis_score" in normalized:
        return "weak_pre_analysis_score"
    if "fallback_edge_high_churn" in normalized or "churn" in normalized:
        return "repeated_churn"
    if "edge" in normalized or "score" in normalized or "confidence" in normalized:
        return "weak_edge"
    if "daily" in normalized or "position" in normalized or "balance" in normalized or "risk" in normalized:
        return "risk_cap"
    source_status = audit_payload.get("source_requirement_status")
    if isinstance(source_status, dict):
        basis = str(source_status.get("evidence_basis") or "").strip().lower()
        edge_source = str(source_status.get("edge_source") or "").strip().lower()
        if basis == "absence_only" or edge_source == "none":
            return "lack_of_evidence"
    return "not_execution_quality_now"


_TIER_LABEL_FOR_LOG: dict[str, str] = {
    "execution_eligible": "exec",
    "deep_research_required": "deep",
    "research_only_learning_queue": "research",
    "monitor_only": "monitor",
    "skip_for_now_with_reason": "skip",
    "operational_error_retry": "retry",
    "terminal_reject": "terminal",
}


def _format_tier_breakdown_for_log(breakdown: dict[str, int] | None) -> str:
    """Render participation tier counts for the Cycle funnel: log line."""
    if not breakdown:
        return "{}"
    parts: list[str] = []
    for tier_key in sorted(breakdown.keys()):
        count = int(breakdown.get(tier_key) or 0)
        if count <= 0:
            continue
        label = _TIER_LABEL_FOR_LOG.get(str(tier_key), str(tier_key))
        parts.append(f"{label}:{count}")
    if not parts:
        return "{}"
    return "{" + ",".join(parts) + "}"


def _build_counterfactual_audit_fields(
    *,
    reason: str | None,
    settings: Settings,
    pre_analysis_score: float | None = None,
    historical_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build counterfactual audit fields explaining what would have unblocked a market.

    Returns a dict of "what would the system need to see to participate?" answers
    so analytics can quantify the gap between the current state and execution
    eligibility. Always-applicable trade thresholds (confidence, evidence quality,
    edge minimum) are emitted everywhere; gate-specific fields (sample size for
    historical gates, threshold gap for score gates) are only added when relevant.
    """
    fields: dict[str, Any] = {
        "counterfactual_required_confidence": settings.MIN_CONFIDENCE,
        "counterfactual_required_evidence_quality": settings.MIN_EVIDENCE_QUALITY_FOR_TRADE,
        "counterfactual_required_edge_min": settings.MIN_EDGE,
        "counterfactual_required_pre_analysis_score": settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE,
    }
    normalized_reason = str(reason or "").strip().lower()
    if pre_analysis_score is not None:
        fields["pre_analysis_threshold_gap"] = float(
            settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE - pre_analysis_score
        )
    if "prefix" in normalized_reason or "historical_prefix" in normalized_reason:
        fields["counterfactual_required_prefix_sample_size"] = (
            settings.HISTORICAL_TICKER_PREFIX_HARD_BLOCK_MIN_SAMPLES
        )
    if "family" in normalized_reason and "historical" in normalized_reason:
        fields["counterfactual_required_family_sample_size"] = (
            settings.HISTORICAL_FAMILY_MIN_SAMPLES
        )
    if "daily_drawdown" in normalized_reason:
        fields["counterfactual_required_for_drawdown_block"] = (
            "drawdown_reset_or_position_close"
        )
        fields["counterfactual_max_daily_drawdown_usdc"] = (
            settings.MAX_DAILY_DRAWDOWN_USDC
        )
    if historical_metrics:
        prefix_n = historical_metrics.get("historical_gate_prefix_sample_size")
        if isinstance(prefix_n, (int, float)):
            shortfall = max(
                0,
                int(settings.HISTORICAL_TICKER_PREFIX_HARD_BLOCK_MIN_SAMPLES) - int(prefix_n),
            )
            fields["counterfactual_prefix_samples_short_by"] = shortfall
        family_n = historical_metrics.get("historical_gate_family_sample_size")
        if isinstance(family_n, (int, float)):
            shortfall = max(
                0,
                int(settings.HISTORICAL_FAMILY_MIN_SAMPLES) - int(family_n),
            )
            fields["counterfactual_family_samples_short_by"] = shortfall
    return fields


def _research_priority_for_reason(
    *,
    gate_name: str,
    reason: str,
    threshold_gap: float = 0.0,
    participation_tier: str | None = None,
) -> float:
    normalized_gate = str(gate_name or "").strip().lower()
    normalized_reason = str(reason or "").strip().lower()
    normalized_tier = str(participation_tier or "").strip().lower()
    priority = 0.35
    if "edge" in normalized_gate or "edge" in normalized_reason:
        priority = 0.90
    elif "evidence" in normalized_gate or "source" in normalized_reason:
        priority = 0.85
    elif "timeout" in normalized_gate or "timeout" in normalized_reason:
        priority = 0.75
    elif "historical" in normalized_gate or "historical" in normalized_reason:
        priority = 0.65
    elif "score_soft_research" in normalized_reason:
        priority = 0.58
    elif "analysis_cap" in normalized_gate or "lifetime" in normalized_reason:
        priority = 0.30
    if normalized_tier == str(ParticipationTier.OPERATIONAL_ERROR_RETRY):
        priority = max(priority, 0.78)
    elif normalized_tier == str(ParticipationTier.DEEP_RESEARCH_REQUIRED):
        priority = max(priority, 0.72)
    elif normalized_tier == str(ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE):
        priority = max(priority, 0.55)
    gap_penalty = min(0.20, max(0.0, float(threshold_gap)) * 0.50)
    return round(max(0.0, min(1.0, priority - gap_penalty)), 4)


def _safe_json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


_RESEARCH_QUEUE_AUDIT_MIRROR_KEYS = frozenset(
    {
        "decision_origin",
        "market_judgment_available",
        "final_action",
        "final_reason",
        "participation_tier",
        "participation_decision",
        "skip_due_to",
        "why_not_execution_eligible",
        "what_to_learn_next",
        "pre_analysis_score",
        "pre_analysis_breakdown",
        "pre_analysis_hard_reject",
        "research_priority",
        "research_queue_position",
        "synthetic_decision",
    }
)


def _research_queue_last_decision_json(
    decision: TradeDecision,
    audit_fields: dict[str, Any],
) -> str:
    """Persist a decision payload with a nested audit object for queue replay.

    Older queue rows stored only top-level decision/audit fragments. New rows
    keep those compatibility fields while adding ``audit`` so drain priority and
    performance review can read the exact participation context.
    """
    audit = {
        str(key): value
        for key, value in (audit_fields or {}).items()
        if value is not None
    }
    payload = dict(decision.model_dump())
    for key, value in audit.items():
        if key in _RESEARCH_QUEUE_AUDIT_MIRROR_KEYS or key.startswith("counterfactual_"):
            payload[key] = value
    payload["audit"] = audit
    return _safe_json_dumps(payload)


def _score_receipt_fields(score_result: Any) -> dict[str, Any]:
    if score_result is None:
        return {}
    weather_penalty = float(getattr(score_result, "weather_uncertainty_penalty", 0.0) or 0.0)
    weather_penalty += float(getattr(score_result, "weather_bin_penalty", 0.0) or 0.0)
    return {
        "score_final": float(getattr(score_result, "final_score", 0.0) or 0.0),
        "score_edge_market": float(getattr(score_result, "edge_market", 0.0) or 0.0),
        "score_edge_external": float(getattr(score_result, "edge_external", 0.0) or 0.0),
        "score_evidence_quality": float(
            getattr(score_result, "evidence_quality", 0.0) or 0.0
        ),
        "score_evidence_component": float(
            getattr(score_result, "evidence_component", 0.0) or 0.0
        ),
        "score_observed_data_bonus": float(
            getattr(score_result, "observed_data_bonus", 0.0) or 0.0
        ),
        "score_evidence_basis_bonus": float(
            getattr(score_result, "evidence_basis_bonus", 0.0) or 0.0
        ),
        "score_computed_edge_bonus": float(
            getattr(score_result, "computed_edge_bonus", 0.0) or 0.0
        ),
        "score_bayesian_component": float(
            getattr(score_result, "bayesian_component", 0.0) or 0.0
        ),
        "score_inefficiency_component": float(
            getattr(score_result, "inefficiency_component", 0.0) or 0.0
        ),
        "score_kelly_component": float(
            getattr(score_result, "kelly_component", 0.0) or 0.0
        ),
        "score_confidence_alignment_bonus": float(
            getattr(score_result, "confidence_alignment_bonus", 0.0) or 0.0
        ),
        "score_definitive_outcome_bonus": float(
            getattr(score_result, "definitive_outcome_bonus", 0.0) or 0.0
        ),
        "score_liquidity_penalty": float(
            getattr(score_result, "liquidity_penalty", 0.0) or 0.0
        ),
        "score_staleness_penalty": float(
            getattr(score_result, "staleness_penalty", 0.0) or 0.0
        ),
        "score_low_information_penalty": float(
            getattr(score_result, "low_information_penalty", 0.0) or 0.0
        ),
        "score_no_external_odds_penalty": float(
            getattr(score_result, "no_external_odds_penalty", 0.0) or 0.0
        ),
        "score_repeated_penalty": float(
            getattr(score_result, "repeated_analysis_penalty", 0.0) or 0.0
        ),
        "score_mention_market_penalty": float(
            getattr(score_result, "mention_market_penalty", 0.0) or 0.0
        ),
        "score_confidence_calibration_penalty": float(
            getattr(score_result, "confidence_calibration_penalty", 0.0) or 0.0
        ),
        "score_fallback_edge_penalty": float(
            getattr(score_result, "fallback_edge_penalty", 0.0) or 0.0
        ),
        "score_overconfidence_penalty": float(
            getattr(score_result, "overconfidence_penalty", 0.0) or 0.0
        ),
        "score_extreme_confidence_penalty": float(
            getattr(score_result, "extreme_confidence_penalty", 0.0) or 0.0
        ),
        "score_late_stage_overconfidence_penalty": float(
            getattr(score_result, "late_stage_overconfidence_penalty", 0.0) or 0.0
        ),
        "score_fallback_high_confidence_penalty": float(
            getattr(score_result, "fallback_high_confidence_penalty", 0.0) or 0.0
        ),
        "score_extreme_market_edge_penalty": float(
            getattr(score_result, "extreme_market_edge_penalty", 0.0) or 0.0
        ),
        "score_hallucinated_edge_penalty": float(
            getattr(score_result, "hallucinated_edge_penalty", 0.0) or 0.0
        ),
        "score_hallucinated_edge_penalty_suppressed": bool(
            getattr(score_result, "hallucinated_edge_penalty_suppressed", False)
        ),
        "score_high_edge_calibration_penalty": float(
            getattr(score_result, "high_edge_calibration_penalty", 0.0) or 0.0
        ),
        "score_extreme_edge_learning_queue": bool(
            getattr(score_result, "extreme_edge_learning_queue", False)
        ),
        "score_proxy_evidence_penalty": float(
            getattr(score_result, "proxy_evidence_penalty", 0.0) or 0.0
        ),
        "score_generic_bin_penalty": float(
            getattr(score_result, "generic_bin_penalty", 0.0) or 0.0
        ),
        "score_numeric_strike_bin_penalty": float(
            getattr(score_result, "numeric_strike_bin_penalty", 0.0) or 0.0
        ),
        "score_short_prefix_penalty": float(
            getattr(score_result, "short_prefix_penalty", 0.0) or 0.0
        ),
        "score_historical_family_bonus": float(
            getattr(score_result, "historical_family_bonus", 0.0) or 0.0
        ),
        "score_ambiguous_resolution_penalty": float(
            getattr(score_result, "ambiguous_resolution_penalty", 0.0) or 0.0
        ),
        "score_historical_prefix_bonus": float(
            getattr(score_result, "historical_prefix_bonus", 0.0) or 0.0
        ),
        "score_historical_prefix_penalty": float(
            getattr(score_result, "historical_prefix_penalty", 0.0) or 0.0
        ),
        "score_extreme_confidence_band_penalty": float(
            getattr(score_result, "extreme_confidence_band_penalty", 0.0) or 0.0
        ),
        "score_numeric_strike_computed_overconfidence_penalty": float(
            getattr(score_result, "numeric_strike_computed_overconfidence_penalty", 0.0) or 0.0
        ),
        "score_volume_amplifier_discount": float(
            getattr(score_result, "volume_amplifier_discount", 0.0) or 0.0
        ),
        "score_coinflip_sports_penalty": float(
            getattr(score_result, "coinflip_sports_penalty", 0.0) or 0.0
        ),
        "score_weather_penalty": weather_penalty,
        "score_bayesian_posterior": getattr(score_result, "bayesian_posterior", None),
        "score_lmsr_price": getattr(score_result, "lmsr_price", None),
        "score_inefficiency_signal": getattr(score_result, "inefficiency_signal", None),
        "score_kelly_raw": getattr(score_result, "kelly_raw", None),
        "score_rejection_reasons": list(getattr(score_result, "rejection_reasons", ()) or ()),
        "score_breakdown_explanation": (
            score_breakdown_explanation(score_result)
            if hasattr(score_result, "final_score")
            else None
        ),
    }


def _score_breakdown_from_execution_audit(
    *,
    execution_audit: dict[str, Any] | None,
    explicit_score_breakdown: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if isinstance(explicit_score_breakdown, dict):
        return explicit_score_breakdown
    if not isinstance(execution_audit, dict):
        return None
    candidate_score_breakdown = execution_audit.get("score_breakdown")
    if isinstance(candidate_score_breakdown, dict):
        return candidate_score_breakdown
    inferred_score_breakdown = {
        key: value
        for key, value in execution_audit.items()
        if str(key).startswith("score_")
    }
    return inferred_score_breakdown or None


def _compact_score_breakdown(score_fields: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(score_fields, dict):
        return {}
    compact: dict[str, Any] = {}
    for key, value in score_fields.items():
        if key == "score_rejection_reasons":
            reasons = [str(reason) for reason in (value or []) if str(reason).strip()]
            if reasons:
                compact[key] = reasons
            continue
        if isinstance(value, bool):
            if value:
                compact[key] = value
            continue
        if isinstance(value, (int, float)):
            if key in {
                "score_final",
                "score_edge_market",
                "score_edge_external",
                "score_evidence_quality",
            } or abs(float(value)) > 1e-9:
                compact[key] = float(value)
            continue
        if value is not None:
            compact[key] = value
    return compact


def _apply_runtime_score_receipt(
    audit_context: dict[str, Any],
    *,
    score_result: Any,
    score_threshold_effective: float,
    pre_execution_final_score: float | None,
    score_gate_score_source: str,
    score_gate_critical_reasons: tuple[str, ...] | list[str] | set[str] = (),
) -> dict[str, Any]:
    """Persist the score actually used by the execution gate into audit fields."""
    score_fields = _score_receipt_fields(score_result)
    if score_fields:
        audit_context.update(score_fields)
        compact_score = _compact_score_breakdown(score_fields)
        if compact_score:
            audit_context["score_breakdown"] = compact_score

    runtime_final_score = score_fields.get("score_final")
    audit_context["execution_score_final"] = runtime_final_score
    audit_context["execution_score_threshold"] = float(score_threshold_effective)
    audit_context["execution_score_rejection_reasons"] = list(
        score_fields.get("score_rejection_reasons", []) or []
    )
    audit_context["score_threshold_effective"] = float(score_threshold_effective)
    audit_context["score_gate_score_source"] = str(score_gate_score_source or "")
    audit_context["score_gate_critical_reasons"] = [
        str(reason) for reason in score_gate_critical_reasons if str(reason).strip()
    ]
    if runtime_final_score is not None and pre_execution_final_score is not None:
        audit_context["pre_vs_runtime_score_delta"] = (
            float(runtime_final_score) - float(pre_execution_final_score)
        )
    return score_fields


def _resolved_pnl_estimate_total(state_manager: MarketStateManager) -> float:
    """Estimate cumulative resolved PnL from family outcome snapshots."""
    try:
        snapshot = state_manager.get_family_outcome_snapshot(lookback=2000)
    except Exception:
        return 0.0
    return float(
        sum(float((stats or {}).get("pnl_total", 0.0) or 0.0) for stats in snapshot.values())
    )


def _score_kwargs(
    *,
    settings: Settings,
    repeated_analysis_count: int,
    non_actionable_streak: int,
    is_weather_market: bool,
    evidence_basis_class: str,
    edge_source: str,
    market_family: str = "",
    short_prefix_penalty: float = 0.0,
    suppress_hallucinated_edge_penalty: bool = False,
    definitive_outcome_eligible: bool = False,
    historical_family_pnl_total: float | None = None,
    historical_family_sample_size: int = 0,
    historical_prefix_pnl_per_trade: float | None = None,
    historical_prefix_sample_size: int = 0,
) -> dict[str, Any]:
    return {
        "is_weather_market": is_weather_market,
        "weather_score_penalty": settings.WEATHER_SCORE_PENALTY,
        "low_info_penalty_threshold": settings.SCORE_LOW_INFO_PENALTY_THRESHOLD,
        "low_info_penalty_base": settings.SCORE_LOW_INFO_PENALTY_BASE,
        "repeated_analysis_count": repeated_analysis_count,
        "non_actionable_streak": non_actionable_streak,
        "repeated_analysis_penalty_base": settings.SCORE_REPEATED_ANALYSIS_PENALTY_BASE,
        "repeated_analysis_penalty_start_count": settings.SCORE_REPEATED_ANALYSIS_PENALTY_START_COUNT,
        "mention_market_penalty_base": settings.MENTION_MARKET_SCORE_PENALTY,
        "confidence_calibration_floor": settings.SCORE_CONFIDENCE_CALIBRATION_FLOOR,
        "confidence_calibration_penalty_scale": settings.SCORE_CONFIDENCE_CALIBRATION_PENALTY_SCALE,
        "fallback_edge_penalty_base": settings.SCORE_FALLBACK_EDGE_PENALTY_BASE,
        "overconfidence_penalty_base": settings.SCORE_OVERCONFIDENCE_PENALTY_BASE,
        "computed_edge_bonus_base": settings.SCORE_COMPUTED_EDGE_BONUS,
        "proxy_evidence_penalty_base": settings.SCORE_PROXY_EVIDENCE_PENALTY_BASE,
        "generic_bin_penalty_base": settings.SCORE_GENERIC_BIN_PENALTY_BASE,
        "ambiguous_resolution_penalty_base": settings.SCORE_AMBIGUOUS_RESOLUTION_PENALTY_BASE,
        "max_reasonable_edge": settings.MAX_REASONABLE_EDGE,
        "hallucinated_edge_penalty_base": settings.SCORE_HALLUCINATED_EDGE_PENALTY_BASE,
        "volume_amplifier_enabled": settings.SCORE_VOLUME_AMPLIFIER_ENABLED,
        "extreme_market_edge_penalty_base": settings.SCORE_EXTREME_MARKET_EDGE_PENALTY_BASE,
        "late_stage_overconfidence_penalty_base": settings.SCORE_LATE_STAGE_OVERCONFIDENCE_PENALTY_BASE,
        "extreme_confidence_threshold": settings.SCORE_EXTREME_CONFIDENCE_THRESHOLD,
        "extreme_confidence_penalty_base": settings.SCORE_EXTREME_CONFIDENCE_PENALTY_BASE,
        "short_prefix_penalty": short_prefix_penalty,
        "evidence_basis_class": evidence_basis_class,
        "edge_source": edge_source,
        "market_family": market_family,
        "coinflip_price_lower": settings.COINFLIP_PRICE_LOWER,
        "coinflip_price_upper": settings.COINFLIP_PRICE_UPPER,
        "suppress_hallucinated_edge_penalty": suppress_hallucinated_edge_penalty,
        "definitive_outcome_eligible": definitive_outcome_eligible,
        "historical_family_pnl_total": historical_family_pnl_total,
        "historical_family_sample_size": historical_family_sample_size,
        "historical_prefix_pnl_per_trade": historical_prefix_pnl_per_trade,
        "historical_prefix_sample_size": historical_prefix_sample_size,
    }


def _effective_score_gate_threshold(
    *,
    settings: Settings,
    market: Market,
    evidence_basis_class: str,
    evidence_quality: float = 0.0,
) -> float:
    if market_family(market) == "weather" and evidence_basis_class == "direct":
        return settings.SCORE_GATE_THRESHOLD_WEATHER_DIRECT
    if evidence_basis_class == "direct" and evidence_quality >= 0.80:
        return settings.SCORE_GATE_THRESHOLD_DIRECT_HIGH_QUALITY
    return settings.SCORE_GATE_THRESHOLD


def _score_gate_critical_rejection_reasons(
    *,
    rejection_reasons: tuple[str, ...] | list[str] | set[str],
    evidence_basis_class: str,
    edge_source: str | None,
    definitive_outcome_eligible: bool = False,
) -> tuple[str, ...]:
    """Identify score failures that should block execution even if bonuses lift the score.

    When ``definitive_outcome_eligible`` is True (game already settled per
    a whitelisted primary source), source-block penalties are suppressed —
    a model citing AP/Reuters with my_prob~1.0 isn't relying on fallback
    edges or proxy evidence; the outcome is observable.
    """
    normalized_reasons = {
        str(reason or "").strip()
        for reason in rejection_reasons
        if str(reason or "").strip()
    }
    critical_reasons = set(normalized_reasons & _SCORE_GATE_ALWAYS_BLOCK_REASONS)
    if definitive_outcome_eligible:
        return tuple(sorted(critical_reasons))
    normalized_evidence_basis = str(evidence_basis_class or "").strip().lower()
    normalized_edge_source = str(edge_source or "").strip().lower()
    if normalized_edge_source in {"fallback", "none"} or normalized_evidence_basis != "direct":
        critical_reasons.update(normalized_reasons & _SCORE_GATE_SOURCE_BLOCK_REASONS)
    return tuple(sorted(critical_reasons))


def _order_response_receipt(order_response: Any | None) -> dict[str, Any] | None:
    if order_response is None:
        return None
    if hasattr(order_response, "model_dump"):
        payload = order_response.model_dump()
    elif isinstance(order_response, dict):
        payload = dict(order_response)
    else:
        return {"raw": str(order_response)}
    raw_payload = payload.get("raw")
    if isinstance(raw_payload, dict):
        payload["raw_summary"] = {
            key: raw_payload.get(key)
            for key in (
                "client_order_id",
                "client_price",
                "client_qty_shares",
                "fill_count",
                "status",
            )
            if key in raw_payload
        }
    return payload


def _should_skip_for_balance(
    *,
    available_balance: float | None,
    min_bet_usdc: float,
) -> bool:
    if available_balance is None:
        return False
    return float(available_balance) < float(min_bet_usdc)


def _decision_evidence_basis(decision: TradeDecision) -> str:
    explicit_basis = str(getattr(decision, "evidence_basis", "") or "").strip().lower()
    if explicit_basis in {"direct", "proxy", "absence_only"}:
        if explicit_basis == "direct":
            reasoning = str(decision.reasoning or "").lower()
            has_absence_marker = any(
                token in reasoning
                for token in (
                    "absence-only",
                    "absence_only",
                    "absence of",
                    "no direct",
                    "no settlement-aligned",
                    "no external odds",
                    "no relevant",
                )
            )
            has_proxy_marker = any(
                token in reasoning
                for token in (
                    "proxy evidence",
                    "proxy-only",
                    "sparse evidence",
                    "evidence gap",
                )
            )
            if has_absence_marker and str(decision.edge_source or "").strip().lower() in {"fallback", "none"}:
                return "absence_only"
            if has_proxy_marker and not bool(getattr(decision, "definitive_outcome_detected", False)):
                return "proxy"
        return explicit_basis
    reasoning = str(decision.reasoning or "").lower()
    for marker in ("basis=direct", "basis=proxy", "basis=absence_only"):
        if marker in reasoning:
            return marker.split("=", 1)[1]
    has_direct_source_signal = any(
        token in reasoning
        for token in (
            "official",
            "transcript",
            "resolution source",
            "settlement",
            "weather.gov",
            "nws",
            "metar",
            "exchange",
            "as of",
        )
    )
    has_absence_signal = any(
        token in reasoning
        for token in (
            "no transcript",
            "no evidence",
            "no mentions",
            "no data",
            "no chart",
            "no external odds",
        )
    )
    edge_source = str(decision.edge_source or "").strip().lower()
    if has_absence_signal and edge_source in {"fallback", "none"}:
        return "absence_only"
    if has_direct_source_signal:
        return "direct"
    return "proxy"


def _generic_market_subfamily(market: Market) -> str:
    """Coarse scoring-only split for broad generic markets."""
    text = " ".join(
        (
            str(getattr(market, "id", "") or ""),
            str(getattr(market, "question", "") or ""),
            str(getattr(market, "category", "") or ""),
            str(getattr(market, "resolution_criteria", "") or ""),
        )
    ).lower()
    for subfamily, keywords in _GENERIC_SUBFAMILY_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return f"generic_{subfamily}"
    return "generic_other"


def _passes_refreshed_edge_guard(
    market: Market,
    decision: TradeDecision,
    settings: Settings,
) -> tuple[bool, float | None, float | None, str]:
    implied_prob = _get_implied_probability(market, decision.outcome)
    edge_ok, edge_value, edge_reason = _passes_edge_threshold(
        implied_prob,
        decision,
        settings,
        market=market,
    )
    return edge_ok, implied_prob, edge_value, edge_reason


def _compute_next_wakeup_seconds(
    markets: list[Market],
    state_manager: MarketStateManager,
    settings: Settings,
    now: datetime | None = None,
) -> int | None:
    """Compute next useful wake-up based on per-market cooldown expiry."""
    if not markets:
        return None

    now_utc = now or datetime.now(timezone.utc)
    earliest_remaining: float | None = None

    for market in markets:
        try:
            state = state_manager.get_market_state(market.id)
        except Exception as exc:
            logger.debug(
                "Adaptive wake-up skipped state lookup for market=%s: %s",
                market.id,
                exc,
                data={"market_id": market.id, "error": str(exc)},
            )
            continue

        if not state or not state.last_analysis:
            continue

        close_time = market.close_time
        if close_time and close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=timezone.utc)
        if close_time and close_time <= now_utc:
            continue

        remaining_seconds = remaining_reanalysis_cooldown_seconds(
            market,
            state,
            reanalysis_cooldown_hours=settings.REANALYSIS_COOLDOWN_HOURS,
            urgent_days_before_close=settings.URGENT_REANALYSIS_DAYS_BEFORE_CLOSE,
            urgent_reanalysis_cooldown_hours=settings.URGENT_REANALYSIS_COOLDOWN_HOURS,
            now=now_utc,
        )
        if remaining_seconds is None:
            continue
        if remaining_seconds <= 0:
            return 1
        if earliest_remaining is None or remaining_seconds < earliest_remaining:
            earliest_remaining = remaining_seconds

    if earliest_remaining is None:
        return None
    capped = min(earliest_remaining, float(_ADAPTIVE_SLEEP_CAP_SECONDS))
    return max(1, int(capped))


def _dry_streak_sleep_seconds(
    *,
    base_poll_interval_sec: int,
    consecutive_zero_order_cycles: int,
    enabled: bool = True,
) -> int | None:
    if not enabled:
        return None
    if consecutive_zero_order_cycles < 3:
        return None
    return min(
        int(_ADAPTIVE_SLEEP_CAP_SECONDS),
        max(1, int(base_poll_interval_sec) * 2),
    )


def _build_grok_client_for_worker(
    settings: Settings,
    provider: XAIProvider | None = None,
) -> GrokClient:
    """Create a Grok client for threaded analysis workers."""
    return GrokClient(
        api_key=settings.XAI_API_KEY,
        model=settings.GROK_MODEL,
        model_deep=settings.GROK_MODEL_DEEP,
        min_bet_usdc=settings.MIN_BET_USDC,
        max_bet_usdc=settings.MAX_BET_USDC,
        settings=settings,
        provider=provider,
    )


# Per-thread storage for analysis worker GrokClient instances. The
# ``ThreadPoolExecutor`` reuses worker threads across submitted tasks, so
# stashing the client in ``threading.local`` lets each thread pay the
# initialization cost once instead of every candidate. Cycle 1 review
# observed 8+ "GrokClient initialized" debug messages per cycle (one per
# candidate); with this cache it drops to one per worker thread.
_worker_grok_client_storage = threading.local()


def reset_worker_grok_client_cache() -> None:
    """Reset the thread-local Grok client cache.

    Primarily exposed for tests that need to verify per-thread reuse without
    contamination between cycles. Each worker thread that calls
    :func:`_get_or_create_worker_grok_client` after this reset will rebuild
    its client on first use.
    """
    storage = _worker_grok_client_storage
    if hasattr(storage, "client"):
        delattr(storage, "client")


def _get_or_create_worker_grok_client(
    settings: Settings,
    provider: XAIProvider | None = None,
) -> GrokClient:
    """Return the calling thread's GrokClient, building it lazily on first use."""
    storage = _worker_grok_client_storage
    client = getattr(storage, "client", None)
    if client is None:
        client = _build_grok_client_for_worker(settings, provider=provider)
        storage.client = client
    return client


def _analyze_market_candidate_via_thread_local_client(
    market: Market,
    state: MarketState | None,
    anchor_analysis: dict[str, Any] | None,
    settings: Settings,
    provider: XAIProvider | None,
    historical_confidence_buckets: dict[str, dict[float, dict[str, float | int]]] | None = None,
    correlation_id: str | None = None,
    force_extended_research: bool = False,
    research_queue_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Worker entry point that reuses one GrokClient per worker thread."""
    grok_client = _get_or_create_worker_grok_client(settings, provider)
    return _analyze_market_candidate_for_worker(
        market=market,
        state=state,
        anchor_analysis=anchor_analysis,
        settings=settings,
        grok_client=grok_client,
        historical_confidence_buckets=historical_confidence_buckets,
        correlation_id=correlation_id,
        force_extended_research=force_extended_research,
        research_queue_context=research_queue_context,
    )


def _analyze_market_candidate_for_worker(
    market: Market,
    state: MarketState | None,
    anchor_analysis: dict[str, Any] | None,
    settings: Settings,
    grok_client: GrokClient,
    historical_confidence_buckets: dict[str, dict[float, dict[str, float | int]]] | None = None,
    correlation_id: str | None = None,
    force_extended_research: bool = False,
    research_queue_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Worker-safe wrapper that restores cycle correlation context."""
    if correlation_id:
        set_correlation_id(correlation_id)
    return _analyze_market_candidate(
        market=market,
        state=state,
        anchor_analysis=anchor_analysis,
        settings=settings,
        grok_client=grok_client,
        historical_confidence_buckets=historical_confidence_buckets,
        force_extended_research=force_extended_research,
        research_queue_context=research_queue_context,
    )


def _analysis_candidate_family_counts(
    analysis_candidates: list[dict[str, Any]],
) -> dict[str, int]:
    """Build per-family counts for analysis candidate observability."""
    counts: dict[str, int] = {}
    for candidate in analysis_candidates:
        market = candidate.get("market")
        if not isinstance(market, Market):
            continue
        family = market_family(market)
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def _analysis_candidate_attempt_limit(
    settings: Settings,
    dynamic_max_markets_per_cycle: int,
    *,
    parallel_analysis_enabled: bool,
) -> int:
    """Return how many candidates may be sent to Grok this cycle.

    Sequential mode keeps a small failure buffer because later candidates can
    replace failed calls without increasing concurrent work. Parallel mode
    should not add that buffer: all selected candidates are submitted at once,
    so the old max+failure-buffer path paid for extra Grok calls and then the
    execution loop processed only MAX_MARKETS_PER_CYCLE decisions.
    """
    base_limit = max(0, int(dynamic_max_markets_per_cycle))
    if parallel_analysis_enabled:
        return base_limit
    return base_limit + max(0, int(settings.XAI_CIRCUIT_BREAKER_MAX_FAILURES))


def _pre_analysis_opportunity_score(
    market: Market,
    state: MarketState | None,
    settings: Settings,
    traded_before: bool,
    fallback_family_edge_rate: float | None = None,
    fallback_family_sample_size: int = 0,
    historical_family_stats: dict[str, float | int] | None = None,
    historical_prefix_stats: dict[str, Any] | None = None,
    historical_gate_metrics: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Estimate opportunity quality before expensive enrichment/analysis."""
    now_utc = datetime.now(timezone.utc)
    implied_prob_yes = _get_implied_probability(market, "YES")
    implied_prob_no = _get_implied_probability(market, "NO")
    implied_prob = implied_prob_yes if implied_prob_yes is not None else implied_prob_no
    liquidity_usdc = max(0.0, float(market.liquidity_usdc or 0.0))
    liquidity_score = min(1.0, liquidity_usdc / 500.0)
    coinflip_penalty = 0.0
    if implied_prob is not None and settings.COINFLIP_PRICE_LOWER <= implied_prob <= settings.COINFLIP_PRICE_UPPER:
        coinflip_penalty = 0.15
    price_center_score = 0.0
    if implied_prob is not None:
        price_center_score = max(0.0, 1.0 - (abs(implied_prob - 0.5) / 0.5))
    horizon_score = 0.5
    raw_hours_to_close: float | None = None
    if market.close_time is not None:
        close_time = market.close_time
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=timezone.utc)
        raw_hours_to_close = (close_time - now_utc).total_seconds() / 3600.0
        hours_to_close = max(0.0, raw_hours_to_close)
        if hours_to_close <= 24:
            horizon_score = 1.0
        elif hours_to_close <= 48:
            horizon_score = 0.8
        elif hours_to_close <= 96:
            horizon_score = 0.6
        else:
            horizon_score = 0.35
    post_event_bonus = 0.0
    if raw_hours_to_close is not None and -6.0 <= raw_hours_to_close <= 0.0:
        post_event_bonus = 0.10
    family = market_family(market)
    market_subfamily = _generic_market_subfamily(market) if family == "generic" else family
    market_id_upper = (market.id or "").upper()
    analysis_count = int(state.analysis_count) if state is not None and state.analysis_count is not None else 0
    non_actionable_streak = int(state.non_actionable_streak) if state is not None else 0
    analysis_penalty_start = max(0, int(settings.PRE_ANALYSIS_ANALYSIS_COUNT_START))
    repeated_analysis_penalty = 0.0
    if analysis_count > analysis_penalty_start:
        repeated_analysis_penalty = (
            float(max(0.0, settings.PRE_ANALYSIS_ANALYSIS_COUNT_PENALTY))
            * float(analysis_count - analysis_penalty_start)
        )
    if not traded_before:
        repeated_analysis_penalty *= 1.15
    non_actionable_penalty = (
        float(max(0.0, settings.PRE_ANALYSIS_NON_ACTIONABLE_STREAK_PENALTY))
        * float(
            min(
                max(0, non_actionable_streak),
                max(0, settings.PRE_ANALYSIS_NON_ACTIONABLE_STREAK_CAP),
            )
        )
    )
    family_penalty = 0.0
    generic_bin_penalty = 0.0
    crypto_bin_penalty = 0.0
    fallback_family_penalty = 0.0
    if family == "speech":
        family_penalty += max(0.0, settings.PRE_ANALYSIS_FAMILY_PENALTY_SPEECH)
    elif family == "music":
        family_penalty += max(0.0, settings.PRE_ANALYSIS_FAMILY_PENALTY_MUSIC)
    elif family == "sports":
        family_penalty += max(0.0, settings.PRE_ANALYSIS_FAMILY_PENALTY_SPORTS)
    if family == "weather" and _WEATHER_BIN_TICKER_PATTERN.search(market.id or ""):
        family_penalty += max(0.0, settings.PRE_ANALYSIS_FAMILY_PENALTY_WEATHER_BIN)
    if family == "generic" and _WEATHER_BIN_TICKER_PATTERN.search(market.id or ""):
        generic_bin_penalty = max(0.0, settings.PRE_ANALYSIS_FAMILY_PENALTY_GENERIC_BIN)
    if family == "crypto" and _WEATHER_BIN_TICKER_PATTERN.search(market.id or ""):
        crypto_bin_penalty = max(0.0, settings.PRE_ANALYSIS_FAMILY_PENALTY_CRYPTO_BIN)
    fallback_rate = max(0.0, min(1.0, float(fallback_family_edge_rate or 0.0)))
    fallback_samples = max(0, int(fallback_family_sample_size))
    fallback_rate_threshold = max(0.0, min(1.0, settings.PRE_ANALYSIS_FALLBACK_FAMILY_RATE_THRESHOLD))
    if (
        family in {"generic", "crypto"}
        and fallback_samples >= max(1, settings.PRE_ANALYSIS_FALLBACK_FAMILY_MIN_SAMPLES)
        and fallback_rate >= fallback_rate_threshold
    ):
        fallback_family_penalty = max(0.0, settings.PRE_ANALYSIS_FALLBACK_FAMILY_PENALTY) * (
            1.0 + (fallback_rate - fallback_rate_threshold)
        )
    historical_family_penalty = 0.0
    historical_family_pnl_penalty = 0.0
    historical_family_win_rate = 0.0
    historical_family_sample_size = 0
    historical_family_pnl = 0.0
    historical_family_pnl_ratio = 0.0
    fallback_family_penalty_scale = 1.0
    if historical_family_stats:
        historical_family_win_rate = max(
            0.0, min(1.0, float(historical_family_stats.get("win_rate", 0.0) or 0.0))
        )
        historical_family_sample_size = max(
            0, int(historical_family_stats.get("sample_size", 0) or 0)
        )
        historical_family_pnl = float(historical_family_stats.get("pnl_total", 0.0) or 0.0)
        historical_min_samples = max(1, int(settings.PRE_ANALYSIS_HISTORICAL_FAMILY_MIN_SAMPLES))
        historical_win_rate_threshold = max(
            0.0,
            min(1.0, float(settings.PRE_ANALYSIS_HISTORICAL_FAMILY_WIN_RATE_THRESHOLD)),
        )
        if (
            historical_family_sample_size >= historical_min_samples
            and historical_family_pnl < 0.0
            and historical_family_win_rate < historical_win_rate_threshold
        ):
            win_rate_shortfall = historical_win_rate_threshold - historical_family_win_rate
            historical_family_penalty = max(
                0.0, settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PENALTY
            ) * (1.0 + win_rate_shortfall)
        historical_pnl_min_samples = max(
            1, int(settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_MIN_SAMPLES)
        )
        historical_pnl_threshold = float(settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_THRESHOLD)
        if (
            historical_family_sample_size >= historical_pnl_min_samples
            and historical_family_pnl <= historical_pnl_threshold
        ):
            pnl_per_trade = historical_family_pnl / max(
                historical_family_sample_size,
                historical_pnl_min_samples,
            )
            historical_pnl_penalty_base = max(
                0.0,
                settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_PENALTY,
            )
            if pnl_per_trade <= -1.0:
                historical_family_pnl_penalty = min(
                    0.25,
                    max(0.0, settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_SEVERE_PENALTY),
                )
            elif pnl_per_trade <= -0.30:
                historical_family_pnl_penalty = min(0.25, historical_pnl_penalty_base)
            elif pnl_per_trade <= -0.05:
                historical_family_pnl_penalty = min(0.25, historical_pnl_penalty_base * 0.25)
            else:
                historical_family_pnl_penalty = 0.0
            historical_family_pnl_ratio = abs(pnl_per_trade)
            severe_pnl_threshold = float(
                settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_SEVERE_THRESHOLD
            )
            if historical_family_pnl <= severe_pnl_threshold:
                historical_family_pnl_penalty = max(
                    historical_family_pnl_penalty,
                    max(0.0, settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_SEVERE_PENALTY),
                )
                historical_family_pnl_penalty = min(0.25, historical_family_pnl_penalty)
            if family == "generic" and historical_family_win_rate >= 0.50:
                # "generic" blends unrelated market types; broad negative PnL
                # with a solid hit rate points more to sizing/entry than to
                # market-selection failure.
                generic_penalty_cap = 0.04
                historical_family_pnl_penalty = min(
                    historical_family_pnl_penalty,
                    generic_penalty_cap,
                )
    if (
        fallback_family_penalty > 0.0
        and historical_family_pnl > 0.0
        and historical_family_win_rate > 0.55
    ):
        fallback_family_penalty *= 0.5
        fallback_family_penalty_scale = 0.5
    historical_profit_bonus = 0.0
    historical_family_volume_bonus = 0.0
    positive_family_pnl_bonus = 0.0
    if historical_family_sample_size >= 8 and (
        historical_family_win_rate > 0.55 or historical_family_pnl > 0.0
    ):
        historical_family_volume_bonus = max(
            0.0,
            float(
                getattr(
                    settings,
                    "PRE_ANALYSIS_ADAPTIVE_BOOST",
                    _PRE_ANALYSIS_POSITIVE_FAMILY_VOLUME_BONUS,
                )
            ),
        )
    if (
        historical_family_sample_size >= max(1, settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_MIN_SAMPLES)
        and historical_family_pnl > abs(float(settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_THRESHOLD))
        and historical_family_win_rate >= 0.55
    ):
        historical_profit_bonus = _PRE_ANALYSIS_PROFITABLE_HISTORY_BONUS
    source_difficulty_penalty = _PRE_ANALYSIS_SOURCE_DIFFICULTY_PENALTIES.get(family, 0.0)
    ambiguous_resolution_penalty = 0.0
    if not (market.resolution_criteria or "").strip():
        ambiguous_resolution_penalty = 0.08
    ambiguous_market_penalty = 0.0
    market_text = f"{market.id or ''} {market.question or ''} {market.resolution_criteria or ''}".lower()
    if family not in {"sports", "weather"} and any(
        token in market_text for token in _AMBIGUOUS_MARKET_TOKENS
    ):
        ambiguous_market_penalty = _PRE_ANALYSIS_AMBIGUOUS_MARKET_PENALTY
    churn_penalty = 0.0
    if analysis_count >= max(6, settings.PRE_ANALYSIS_HARD_REJECTION_MIN_ANALYSES // 2):
        churn_penalty = 0.05
        if not traded_before:
            churn_penalty += 0.03
    zero_trade_rate_penalty = 0.0
    negative_prefix_penalty = 0.0
    historical_gate_score_penalty = 0.0
    historical_gate_sample_weight = 0.0
    if historical_prefix_stats and not traded_before:
        prefix_len = max(1, int(settings.HISTORICAL_TICKER_PREFIX_LEN))
        market_prefix = market_id_upper[:prefix_len]
        prefix_snapshot = historical_prefix_stats.get(market_prefix)
        if prefix_snapshot is not None:
            prefix_sample_size = max(
                0,
                int(getattr(prefix_snapshot, "sample_size", 0) or 0),
            )
            prefix_wins = max(
                0,
                int(getattr(prefix_snapshot, "wins", 0) or 0),
            )
            prefix_pnl_total = float(getattr(prefix_snapshot, "pnl_total", 0.0) or 0.0)
            if (
                prefix_sample_size >= max(1, int(settings.HISTORICAL_TICKER_PREFIX_MIN_SAMPLES))
                and prefix_wins <= 0
                and prefix_pnl_total <= 0.0
            ):
                zero_trade_rate_penalty = max(
                    0.0,
                    settings.PRE_ANALYSIS_ZERO_TRADE_RATE_PENALTY,
                )
            elif (
                prefix_sample_size >= max(1, int(settings.HISTORICAL_TICKER_PREFIX_MIN_SAMPLES))
                and prefix_pnl_total <= float(settings.HISTORICAL_TICKER_PREFIX_PNL_CUTOFF)
            ):
                negative_prefix_penalty = _PRE_ANALYSIS_NEGATIVE_PREFIX_PENALTY
    if historical_gate_metrics:
        historical_gate_tier = str(
            historical_gate_metrics.get("historical_gate_tier") or ""
        ).strip().lower()
        try:
            historical_gate_sample_weight = float(
                historical_gate_metrics.get("historical_gate_sample_weight", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            historical_gate_sample_weight = 0.0
        if historical_gate_tier in {GateTier.SOFT_DEMOTE, GateTier.HARD_DENY}:
            try:
                historical_gate_score_penalty = float(
                    historical_gate_metrics.get(
                        "historical_gate_score_penalty",
                        settings.HISTORICAL_TICKER_PREFIX_SOFT_DEMOTE_SCORE_PENALTY,
                    )
                    or 0.0
                )
            except (TypeError, ValueError):
                historical_gate_score_penalty = 0.0
            historical_gate_score_penalty = max(0.0, historical_gate_score_penalty)
    # Cap stacked historical-family penalties so overlapping signals from the
    # same poor-history data source (family PnL, prefix PnL, fallback rate,
    # zero-trade-rate, gate soft-demote) cannot collapse a market's score by
    # 0.6pp+ and force soft-research routing on its own. Each individual
    # penalty is preserved at full strength when it is the only one firing.
    stacked_historical_penalty = (
        fallback_family_penalty
        + historical_family_penalty
        + historical_family_pnl_penalty
        + zero_trade_rate_penalty
        + negative_prefix_penalty
        + historical_gate_score_penalty
    )
    stacked_cap = max(
        0.0,
        float(settings.PRE_ANALYSIS_STACKED_HISTORICAL_PENALTY_CAP),
    )
    stacked_historical_excess_credited = 0.0
    if stacked_cap > 0.0 and stacked_historical_penalty > stacked_cap:
        stacked_historical_excess_credited = stacked_historical_penalty - stacked_cap
    score = (
        (0.40 * price_center_score)
        + (0.35 * liquidity_score)
        + (0.25 * horizon_score)
        + post_event_bonus
        + historical_profit_bonus
        + historical_family_volume_bonus
        - repeated_analysis_penalty
        - non_actionable_penalty
        - family_penalty
        - generic_bin_penalty
        - crypto_bin_penalty
        - fallback_family_penalty
        - historical_family_penalty
        - historical_family_pnl_penalty
        - source_difficulty_penalty
        - ambiguous_resolution_penalty
        - ambiguous_market_penalty
        - churn_penalty
        - zero_trade_rate_penalty
        - negative_prefix_penalty
        - historical_gate_score_penalty
        - coinflip_penalty
        + stacked_historical_excess_credited
    )
    if historical_family_pnl > 0.0:
        positive_family_pnl_bonus = _PRE_ANALYSIS_POSITIVE_FAMILY_PNL_BONUS
        score += positive_family_pnl_bonus
    return score, {
        "pre_score_price_center": price_center_score,
        "pre_score_liquidity": liquidity_score,
        "pre_score_horizon": horizon_score,
        "pre_score_post_event_bonus": post_event_bonus,
        "pre_score_repeated_analysis_penalty": repeated_analysis_penalty,
        "pre_score_non_actionable_penalty": non_actionable_penalty,
        "pre_score_family_penalty": family_penalty,
        "pre_score_generic_bin_penalty": generic_bin_penalty,
        "pre_score_crypto_bin_penalty": crypto_bin_penalty,
        "pre_score_fallback_family_penalty": fallback_family_penalty,
        "pre_score_fallback_family_penalty_scale": fallback_family_penalty_scale,
        "pre_score_fallback_family_rate": fallback_rate,
        "pre_score_fallback_family_samples": float(fallback_samples),
        "pre_score_historical_family_penalty": historical_family_penalty,
        "pre_score_historical_family_pnl_penalty": historical_family_pnl_penalty,
        "pre_score_historical_family_pnl_ratio": historical_family_pnl_ratio,
        "pre_score_historical_family_win_rate": historical_family_win_rate,
        "pre_score_historical_family_samples": float(historical_family_sample_size),
        "pre_score_historical_family_pnl_total": historical_family_pnl,
        "pre_score_historical_profit_bonus": historical_profit_bonus,
        "pre_score_historical_family_volume_bonus": historical_family_volume_bonus,
        "pre_score_positive_family_pnl_bonus": positive_family_pnl_bonus,
        "pre_score_source_difficulty_penalty": source_difficulty_penalty,
        "pre_score_ambiguous_resolution_penalty": ambiguous_resolution_penalty,
        "pre_score_ambiguous_market_penalty": ambiguous_market_penalty,
        "pre_score_churn_penalty": churn_penalty,
        "pre_score_zero_trade_rate_penalty": zero_trade_rate_penalty,
        "pre_score_negative_prefix_penalty": negative_prefix_penalty,
        "pre_score_historical_gate_score_penalty": historical_gate_score_penalty,
        "pre_score_historical_gate_sample_weight": historical_gate_sample_weight,
        "pre_score_market_subfamily": market_subfamily,
        "pre_score_stacked_historical_penalty_raw": stacked_historical_penalty,
        "pre_score_stacked_historical_penalty_cap": stacked_cap,
        "pre_score_stacked_historical_excess_credited": stacked_historical_excess_credited,
        "pre_score_coinflip_penalty": coinflip_penalty,
        "pre_score_analysis_count": float(analysis_count),
        "pre_score_non_actionable_streak": float(non_actionable_streak),
        "pre_score_traded_before": 1.0 if traded_before else 0.0,
        "pre_score_hours_to_close": (
            float(raw_hours_to_close) if raw_hours_to_close is not None else 0.0
        ),
    }


def _pre_analysis_participation_hold(
    *,
    market: Market,
    state: MarketState | None,
    settings: Settings,
    traded_before: bool,
    had_recent_fallback_edge: bool = False,
    historical_family_stats: dict[str, float | int] | None = None,
    fallback_family_edge_rate: float | None = None,
    fallback_family_sample_size: int = 0,
    historical_gate_allowed: bool | None = None,
    historical_gate_reason: str | None = None,
    historical_gate_metrics: dict[str, Any] | None = None,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Evaluate pre-analysis participation gates.

    Returns ``(demoted, reason, metadata)`` where ``demoted`` indicates the
    market should be routed away from deep analysis (to research queue or
    skip), and ``metadata`` describes which gate fired with structured fields.
    Note: most "demoted" outcomes route to research/monitor tiers, not true
    terminal rejection.
    """
    if not settings.PRE_ANALYSIS_HARD_REJECTION_ENABLED:
        return False, None, {}
    analysis_count = int(state.analysis_count or 0) if state is not None else 0
    non_actionable_streak = int(state.non_actionable_streak or 0) if state is not None else 0
    if (
        historical_gate_allowed is False
        and historical_gate_reason
        and not traded_before
    ):
        metadata = {
            "participation_demotion_reason": historical_gate_reason,
            "participation_demotion_analysis_count": analysis_count,
            "participation_demotion_traded_before": traded_before,
            **(historical_gate_metrics or {}),
        }
        return False, None, metadata
    if state is None:
        return False, None, {}
    if (
        non_actionable_streak >= 3
        and had_recent_fallback_edge
        and not traded_before
    ):
        metadata = {
            "participation_demotion_reason": "fallback_edge_high_churn",
            "participation_demotion_non_actionable_streak": non_actionable_streak,
            "participation_demotion_analysis_count": analysis_count,
            "participation_demotion_traded_before": traded_before,
            "participation_demotion_had_recent_fallback_edge": had_recent_fallback_edge,
        }
        return True, "pre_analysis_fallback_edge_high_churn", metadata
    family = market_family(market)
    if (
        family == "crypto"
        and settings.PRE_ANALYSIS_CRYPTO_NEGATIVE_PNL_BLOCK_ENABLED
        and not traded_before
    ):
        historical_sample_size = int(
            (historical_family_stats or {}).get("sample_size", 0) or 0
        )
        historical_pnl_total = float(
            (historical_family_stats or {}).get("pnl_total", 0.0) or 0.0
        )
        fallback_rate = max(0.0, min(1.0, float(fallback_family_edge_rate or 0.0)))
        fallback_samples = max(0, int(fallback_family_sample_size))
        min_samples = max(1, int(settings.PRE_ANALYSIS_CRYPTO_MIN_SAMPLES))
        if (
            historical_sample_size >= min_samples
            and fallback_samples >= min_samples
            and historical_pnl_total <= settings.PRE_ANALYSIS_CRYPTO_NEGATIVE_PNL_THRESHOLD
            and fallback_rate >= settings.PRE_ANALYSIS_CRYPTO_FALLBACK_RATE_BLOCK_THRESHOLD
        ):
            metadata = {
                "participation_demotion_reason": "crypto_historically_unprofitable",
                "participation_demotion_family": family,
                "participation_demotion_historical_pnl": historical_pnl_total,
                "participation_demotion_historical_samples": historical_sample_size,
                "participation_demotion_fallback_rate": fallback_rate,
                "participation_demotion_fallback_samples": fallback_samples,
            }
            return False, None, metadata
    terminal_outcome = str(state.last_terminal_outcome or "").strip().lower()
    has_high_churn = (
        non_actionable_streak >= max(1, settings.PRE_ANALYSIS_HARD_REJECTION_MIN_STREAK)
        and analysis_count >= max(1, settings.PRE_ANALYSIS_HARD_REJECTION_MIN_ANALYSES)
    )
    if (
        has_high_churn
        and terminal_outcome in _PRE_ANALYSIS_HARD_REJECTION_TERMINAL_OUTCOMES
        and not traded_before
    ):
        demotion_reason = "repeated_non_actionable_market"
        if family in {"generic", "crypto"} and _WEATHER_BIN_TICKER_PATTERN.search(market.id or ""):
            demotion_reason = "repeated_non_actionable_bin_market"
        metadata = {
            "participation_demotion_reason": demotion_reason,
            "participation_demotion_family": family,
            "participation_demotion_terminal_outcome": terminal_outcome,
            "participation_demotion_non_actionable_streak": non_actionable_streak,
            "participation_demotion_analysis_count": analysis_count,
            "participation_demotion_traded_before": traded_before,
        }
        return True, f"pre_analysis_{demotion_reason}", metadata
    if analysis_count >= 4 and non_actionable_streak >= 3 and not traded_before:
        metadata = {
            "participation_demotion_reason": "repeated_churn_market",
            "participation_demotion_non_actionable_streak": non_actionable_streak,
            "participation_demotion_analysis_count": analysis_count,
            "participation_demotion_traded_before": traded_before,
        }
        return True, "pre_analysis_repeated_churn_market", metadata
    return False, None, {}


def _cap_analysis_candidates(
    analysis_candidates: list[dict[str, Any]],
    max_markets_per_cycle: int,
    max_weather_candidates_per_cycle: int | None = None,
    max_crypto_candidates_per_cycle: int | None = None,
    max_speech_candidates_per_cycle: int | None = None,
    max_music_candidates_per_cycle: int | None = None,
    max_sports_candidates_per_cycle: int | None = None,
    pre_scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Apply a hard cap using global risk-adjusted rank, then family caps."""
    if max_markets_per_cycle <= 0:
        return []
    if len(analysis_candidates) <= max_markets_per_cycle:
        return analysis_candidates

    ranked_candidates: list[tuple[tuple[float, int, int, str], dict[str, Any]]] = []
    invalid_candidates: list[dict[str, Any]] = []
    for input_index, candidate in enumerate(analysis_candidates):
        market = candidate.get("market")
        if not isinstance(market, Market):
            invalid_candidates.append(candidate)
            continue
        family = market_family(market)
        market_id = str(getattr(market, "id", ""))
        base_score = float(
            (pre_scores or {}).get(
                market_id,
                candidate.get("pre_analysis_score") or 0.0,
            )
        )
        non_actionable_streak = max(0, int(candidate.get("non_actionable_streak", 0) or 0))
        historical_pnl = float(candidate.get("historical_family_pnl_total", 0.0) or 0.0)
        historical_samples = max(0, int(candidate.get("historical_family_sample_size", 0) or 0))
        historical_win_rate = float(candidate.get("historical_family_win_rate", 0.0) or 0.0)
        historical_gate_allowed = candidate.get("historical_gate_allowed")
        historical_gate_metrics = candidate.get("historical_gate_metrics")
        historical_gate_metric_penalty = 0.0
        historical_gate_metrics_present = isinstance(historical_gate_metrics, dict)
        if historical_gate_metrics_present:
            try:
                historical_gate_metric_penalty = float(
                    historical_gate_metrics.get("historical_gate_score_penalty", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                historical_gate_metric_penalty = 0.0
        short_prefix_penalty = float(candidate.get("short_prefix_score_penalty", 0.0) or 0.0)
        historical_loss_penalty = 0.0
        if historical_samples >= 8 and historical_pnl < 0.0:
            historical_loss_penalty = min(0.20, abs(historical_pnl) / 250.0)
            if historical_win_rate and historical_win_rate < 0.50:
                historical_loss_penalty += min(0.06, (0.50 - historical_win_rate) * 0.20)
        # Avoid double-counting the historical-gate penalty: when the gate
        # surfaced metrics, _pre_analysis_opportunity_score has already absorbed
        # historical_gate_score_penalty into the base score. Re-deducting the
        # 0.12 flat here would punish the same signal twice and over-demote
        # markets that the gate already softened. Only fall back to the flat
        # 0.12 when the gate ran but metrics never reached this candidate
        # (legacy/backward-compat path).
        if historical_gate_metrics_present:
            historical_gate_penalty = max(0.0, historical_gate_metric_penalty)
        elif historical_gate_allowed is False:
            historical_gate_penalty = 0.12
        else:
            historical_gate_penalty = 0.0
        repeated_penalty = min(0.12, non_actionable_streak * 0.02)
        source_difficulty_penalty = _PRE_ANALYSIS_SOURCE_DIFFICULTY_PENALTIES.get(
            family,
            0.0,
        ) * 0.5
        risk_adjusted_score = (
            base_score
            - repeated_penalty
            - historical_loss_penalty
            - historical_gate_penalty
            - short_prefix_penalty
            - source_difficulty_penalty
        )
        selection_components = {
            "base_pre_analysis_score": round(base_score, 4),
            "risk_adjusted_score": round(risk_adjusted_score, 4),
            "repeated_penalty": round(repeated_penalty, 4),
            "historical_loss_penalty": round(historical_loss_penalty, 4),
            "historical_gate_penalty": round(historical_gate_penalty, 4),
            "short_prefix_penalty": round(short_prefix_penalty, 4),
            "source_difficulty_penalty": round(source_difficulty_penalty, 4),
        }
        candidate["selection_rank_score"] = round(risk_adjusted_score, 4)
        candidate["selection_rank_components"] = selection_components
        # Research-queue drain probes always rank ahead of normal candidates so
        # the cap doesn't silently drop the longest-waiting research entries.
        drain_probe_priority = 0 if candidate.get("is_research_queue_drain_probe") else 1
        ranked_candidates.append(
            (
                (
                    drain_probe_priority,
                    -risk_adjusted_score,
                    non_actionable_streak,
                    input_index,
                    market_id,
                ),
                candidate,
            )
        )

    if not ranked_candidates:
        return analysis_candidates[:max_markets_per_cycle]

    selected: list[dict[str, Any]] = []
    selected_weather_count = 0
    selected_crypto_count = 0
    selected_speech_count = 0
    selected_music_count = 0
    selected_sports_count = 0
    for _, candidate in sorted(ranked_candidates, key=lambda item: item[0]):
        if len(selected) >= max_markets_per_cycle:
            break
        market = candidate.get("market")
        if not isinstance(market, Market):
            continue
        family = market_family(market)
        if (
            family == "weather"
            and max_weather_candidates_per_cycle is not None
            and selected_weather_count >= max_weather_candidates_per_cycle
        ):
            continue
        if (
            family == "crypto"
            and max_crypto_candidates_per_cycle is not None
            and selected_crypto_count >= max_crypto_candidates_per_cycle
        ):
            continue
        if (
            family == "speech"
            and max_speech_candidates_per_cycle is not None
            and selected_speech_count >= max_speech_candidates_per_cycle
        ):
            continue
        if (
            family == "music"
            and max_music_candidates_per_cycle is not None
            and selected_music_count >= max_music_candidates_per_cycle
        ):
            continue
        if (
            family == "sports"
            and max_sports_candidates_per_cycle is not None
            and selected_sports_count >= max_sports_candidates_per_cycle
        ):
            continue
        selected.append(candidate)
        if family == "weather":
            selected_weather_count += 1
        elif family == "crypto":
            selected_crypto_count += 1
        elif family == "speech":
            selected_speech_count += 1
        elif family == "music":
            selected_music_count += 1
        elif family == "sports":
            selected_sports_count += 1
    if len(selected) < max_markets_per_cycle and invalid_candidates:
        selected.extend(invalid_candidates[: max_markets_per_cycle - len(selected)])
    return selected


def _resolve_dynamic_analysis_candidate_cap(
    *,
    settings: Settings,
    best_pre_analysis_score: float,
) -> tuple[int, bool, bool]:
    """Return (cap, reduced_applied, negative_score_floor_applied)."""
    dynamic_max_markets_per_cycle = settings.MAX_MARKETS_PER_CYCLE
    reduced_candidate_cap_applied = False
    negative_score_floor_applied = False
    if best_pre_analysis_score < settings.NEGATIVE_BEST_SCORE_DEEP_ANALYSIS_FLOOR:
        dynamic_max_markets_per_cycle = 1
        reduced_candidate_cap_applied = True
        negative_score_floor_applied = True
    elif best_pre_analysis_score < settings.PRE_ANALYSIS_MUST_ANALYZE_THRESHOLD:
        dynamic_max_markets_per_cycle = min(
            dynamic_max_markets_per_cycle,
            max(1, settings.PRE_ANALYSIS_REDUCED_MAX_CANDIDATES),
        )
        reduced_candidate_cap_applied = True
    return dynamic_max_markets_per_cycle, reduced_candidate_cap_applied, negative_score_floor_applied


def _research_queue_drain_sort_key(entry: dict[str, Any]) -> tuple[float, str, str]:
    """Prioritize near-threshold research entries before older weak entries."""
    raw_gap = entry.get("threshold_gap")
    try:
        threshold_gap = max(0.0, float(raw_gap))
    except (TypeError, ValueError):
        threshold_gap = float("inf")
    return (
        threshold_gap,
        str(entry.get("queued_at") or ""),
        str(entry.get("market_id") or ""),
    )


def _research_queue_zero_yield_sort_key(entry: dict[str, Any]) -> tuple[float, int, str, str]:
    """Promotion ranking when zero-yield cycles show the queue needs active repair."""
    raw_gap = entry.get("threshold_gap")
    try:
        threshold_gap = max(0.0, float(raw_gap))
    except (TypeError, ValueError):
        threshold_gap = float("inf")
    try:
        times_seen = max(0, int(entry.get("times_seen") or 0))
    except (TypeError, ValueError):
        times_seen = 0
    return (
        threshold_gap,
        -times_seen,
        str(entry.get("queued_at") or ""),
        str(entry.get("market_id") or ""),
    )


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = str(item or "").strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


def _research_source_window(
    *,
    current_items: list[str],
    pool_items: list[str],
    offset: int,
    limit: int,
) -> list[str]:
    """Pick a bounded source window, preferring fallback sources when present."""
    max_items = max(1, int(limit or len(current_items) or 1))
    pool = _dedupe_preserve_order(pool_items or current_items)
    if not pool:
        return []
    resolved_offset = max(0, int(offset))
    if resolved_offset >= len(pool):
        resolved_offset = 0
    rotated = [*pool[resolved_offset:], *pool[:resolved_offset]]
    return rotated[:max_items]


def _build_speech_reanalysis_search_config(
    base_config: SearchConfig,
    settings: Settings,
) -> SearchConfig:
    """Expand lookback and rotate sources for low-evidence speech reanalysis."""
    now = datetime.now(timezone.utc)
    base_lookback_hours = base_config.lookback_hours or 24
    expanded_lookback_hours = max(base_lookback_hours, base_lookback_hours * 2)
    rotated_domains = _research_source_window(
        current_items=base_config.allowed_domains,
        pool_items=base_config.source_domains_pool,
        offset=settings.EXTENDED_RESEARCH_SOURCE_OFFSET,
        limit=base_config.max_allowed_domains,
    )
    rotated_handles = _research_source_window(
        current_items=base_config.allowed_x_handles,
        pool_items=base_config.source_x_handles_pool,
        offset=settings.EXTENDED_RESEARCH_X_HANDLE_OFFSET,
        limit=base_config.max_allowed_x_handles,
    )
    return SearchConfig(
        from_date=now - timedelta(hours=expanded_lookback_hours),
        to_date=now,
        allowed_domains=rotated_domains,
        allowed_x_handles=rotated_handles,
        source_domains_pool=list(base_config.source_domains_pool),
        source_x_handles_pool=list(base_config.source_x_handles_pool),
        max_allowed_domains=base_config.max_allowed_domains,
        max_allowed_x_handles=base_config.max_allowed_x_handles,
        enable_multimedia=True,
        multimedia_confidence_range=base_config.multimedia_confidence_range,
        profile_name=base_config.profile_name,
        lookback_hours=expanded_lookback_hours,
    )


def _build_extended_reanalysis_search_config(
    base_config: SearchConfig,
    settings: Settings,
) -> SearchConfig:
    """Increase lookback and rotate sources for stale non-actionable markets."""
    now = datetime.now(timezone.utc)
    base_lookback_hours = base_config.lookback_hours or 24
    expanded_lookback_hours = max(base_lookback_hours + 24, base_lookback_hours * 2)
    rotated_domains = _research_source_window(
        current_items=base_config.allowed_domains,
        pool_items=base_config.source_domains_pool,
        offset=settings.EXTENDED_RESEARCH_SOURCE_OFFSET,
        limit=base_config.max_allowed_domains,
    )
    rotated_handles = _research_source_window(
        current_items=base_config.allowed_x_handles,
        pool_items=base_config.source_x_handles_pool,
        offset=settings.EXTENDED_RESEARCH_X_HANDLE_OFFSET,
        limit=base_config.max_allowed_x_handles,
    )
    return SearchConfig(
        from_date=now - timedelta(hours=expanded_lookback_hours),
        to_date=now,
        allowed_domains=rotated_domains,
        allowed_x_handles=rotated_handles,
        source_domains_pool=list(base_config.source_domains_pool),
        source_x_handles_pool=list(base_config.source_x_handles_pool),
        max_allowed_domains=base_config.max_allowed_domains,
        max_allowed_x_handles=base_config.max_allowed_x_handles,
        enable_multimedia=True,
        multimedia_confidence_range=base_config.multimedia_confidence_range,
        profile_name=base_config.profile_name,
        lookback_hours=expanded_lookback_hours,
    )


def _analyze_market_candidate(
    market: Market,
    state: MarketState | None,
    anchor_analysis: dict[str, Any] | None,
    settings: Settings,
    grok_client: GrokClient,
    historical_confidence_buckets: dict[str, dict[float, dict[str, float | int]]] | None = None,
    force_extended_research: bool = False,
    research_queue_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run analysis/refinement/guardrails for a market candidate."""
    previous_analysis = _build_previous_analysis(anchor_analysis)
    analysis_market = _market_with_research_queue_context(market, research_queue_context)
    search_config = build_market_search_config(settings, analysis_market)
    used_extended_research = bool(force_extended_research)
    if used_extended_research:
        search_config = _build_extended_reanalysis_search_config(
            search_config,
            settings,
        )
    try:
        decision = grok_client.analyze_market(
            analysis_market,
            search_config=search_config,
            previous_analysis=previous_analysis,
        )
    except Exception as exc:
        error_text = str(exc)
        is_timeout = (
            isinstance(exc, TimeoutError)
            or "grok stream exceeded" in error_text.lower()
        )
        logger.error(
            "Initial market analysis failed for %s: %s",
            market.id,
            exc,
            data={
                "market_id": market.id,
                "error": error_text,
                "error_type": type(exc).__name__,
                "analysis_phase": "initial",
                "is_timeout": is_timeout,
            },
        )
        return {
            "analysis_failed": True,
            "analysis_error": error_text,
            "analysis_error_type": type(exc).__name__,
            "analysis_error_retriable_xai": _is_retriable_xai_error(error_text),
            "analysis_error_quota_exhausted": _is_quota_exhausted_xai_error(error_text),
            "analysis_is_timeout": is_timeout,
            "analysis_search_profile": getattr(search_config, "profile_name", None),
            "was_refined": False,
            "refinement_reason_text": None,
            "used_extended_research": used_extended_research,
            "flip_triggered": False,
            "flip_blocked": False,
            "refinement_skipped_by_flip_precheck": False,
            "flip_precheck_reason": None,
            "market_outcome_mismatch_counted": False,
        }

    anchor_outcome: str | None = None
    if anchor_analysis and anchor_analysis.get("outcome") is not None:
        anchor_outcome = str(anchor_analysis["outcome"]).strip() or None

    edge_repair_attempted = False
    edge_repair_reason_text: str | None = None
    edge_repair_unresolved_reason: str | None = None
    repair_reason = _edge_repair_reason(
        decision=decision,
        market=market,
        settings=settings,
        implied_prob=_get_implied_probability(market, decision.outcome),
    )
    if repair_reason is not None:
        edge_repair_attempted = True
        edge_repair_reason_text = repair_reason
        repair_search_config = _build_extended_reanalysis_search_config(
            search_config,
            settings,
        )
        repair_previous = decision.model_copy(
            update={
                "reasoning": (
                    f"[EdgeRepairRequired reason={repair_reason}] "
                    "Compute probability_yes, market-implied probability, edge_market, "
                    "base rate, counter-evidence, and pricing-in explanation. "
                    f"{decision.reasoning}"
                )
            }
        )
        try:
            repaired_decision = grok_client.analyze_market_deep(
                analysis_market,
                previous_analysis=repair_previous,
                search_config=repair_search_config,
            )
            decision = repaired_decision
            was_refined = True
            used_extended_research = True
        except Exception as exc:
            edge_repair_unresolved_reason = f"repair_exception:{type(exc).__name__}"
            logger.warning(
                "Edge repair failed: market=%s reason=%s error=%s",
                market.id,
                repair_reason,
                exc,
                data={
                    "market_id": market.id,
                    "edge_repair_reason": repair_reason,
                    "error": str(exc),
                },
            )
            decision = decision.model_copy(
                update={
                    "should_trade": False,
                    "abstain": True,
                    "bet_size_pct": 0.0,
                    "reasoning": (
                        f"[EdgeRepair unresolved reason={edge_repair_unresolved_reason}] "
                        f"{decision.reasoning}"
                    ),
                }
            )
        if edge_repair_unresolved_reason is None:
            edge_repair_unresolved_reason = _edge_repair_reason(
                decision=decision,
                market=market,
                settings=settings,
                implied_prob=_get_implied_probability(market, decision.outcome),
            )
            if edge_repair_unresolved_reason is not None:
                decision = decision.model_copy(
                    update={
                        "should_trade": False,
                        "abstain": True,
                        "bet_size_pct": 0.0,
                        "reasoning": (
                            f"[EdgeRepair unresolved reason={edge_repair_unresolved_reason}] "
                            f"{decision.reasoning}"
                        ),
                    }
                )

    was_refined = edge_repair_attempted
    refinement = RefinementStrategy(
        market=analysis_market,
        urgent_days_before_close=settings.URGENT_REANALYSIS_DAYS_BEFORE_CLOSE,
        skip_borderline_families=settings.REFINEMENT_SKIP_BORDERLINE_FAMILIES,
    )
    refinement_skipped_by_flip_precheck = False
    flip_precheck_reason: str | None = None
    implied_prob_for_refine = _get_implied_probability(market, decision.outcome)
    edge_for_refine = (
        decision.confidence - implied_prob_for_refine
        if implied_prob_for_refine is not None
        else decision.edge_external
    )
    refinement_reasons = refinement.get_refinement_reasons(
        decision,
        state,
        implied_prob=implied_prob_for_refine,
        evidence_quality=decision.evidence_quality,
        edge_value=edge_for_refine,
    )
    if anchor_outcome and not _outcomes_match(decision.outcome, anchor_outcome):
        if "side_flip_vs_anchor" not in refinement_reasons:
            refinement_reasons.append("side_flip_vs_anchor")
    if edge_repair_attempted:
        refinement_reasons = []
    refinement_reason_text = ",".join(refinement_reasons) if refinement_reasons else None
    if refinement_reasons:
        refinement_search_config = search_config
        if (
            search_config.profile_name == "speech"
            and decision.evidence_quality < 0.5
        ):
            refinement_search_config = _build_speech_reanalysis_search_config(
                search_config,
                settings,
            )
            logger.debug(
                "Expanded speech reanalysis search config: market=%s initial_lookback=%s expanded_lookback=%s",
                market.id,
                search_config.lookback_hours,
                refinement_search_config.lookback_hours,
                data={
                    "market_id": market.id,
                    "profile_name": search_config.profile_name,
                    "initial_lookback_hours": search_config.lookback_hours,
                    "expanded_lookback_hours": refinement_search_config.lookback_hours,
                    "expanded_domains": refinement_search_config.allowed_domains,
                    "expanded_x_handles": refinement_search_config.allowed_x_handles,
                },
            )
        (
            should_skip_refinement,
            flip_precheck_reason,
            flip_precheck_payload,
        ) = _should_skip_flip_refinement(
            market=market,
            decision=decision,
            anchor_analysis=anchor_analysis,
            settings=settings,
        )
        if should_skip_refinement:
            refinement_skipped_by_flip_precheck = True
            logger.info(
                "Skipped refinement by flip pre-check: market=%s reason=%s",
                market.id,
                flip_precheck_reason,
                data=flip_precheck_payload,
            )
        else:
            decision = refinement.perform_refinement(
                grok_client,
                analysis_market,
                decision,
                search_config=refinement_search_config,
            )
            was_refined = True

    decision = _cap_confidence_for_category(decision, market, settings)
    confidence_before_calibration = decision.confidence
    evidence_basis_for_calibration = _decision_evidence_basis(decision)
    definitive_outcome_for_calibration = _is_definitive_outcome_eligible(
        decision,
        settings,
        market=market,
    )
    stage_one_confidence = calibrate_confidence(
        decision.confidence,
        shrinkage_floor=settings.CONFIDENCE_SHRINKAGE_FLOOR,
        shrinkage_factor=(
            settings.CONFIDENCE_SHRINKAGE_FACTOR_HIGH
            if confidence_before_calibration >= 0.88
            else settings.CONFIDENCE_SHRINKAGE_FACTOR
        ),
        family_shrinkage_override=_confidence_shrinkage_override_for_market(market),
        evidence_basis_class=evidence_basis_for_calibration,
        definitive_outcome=definitive_outcome_for_calibration,
        has_primary_source_url=bool(
            str(getattr(decision, "primary_source_url", "") or "").strip()
        ),
        direct_shrinkage_boost_factor=settings.CALIBRATION_DIRECT_SHRINKAGE_FACTOR_BOOST,
    )
    confidence_family = market_family(market)
    historical_win_rate_at_bucket = _historical_win_rate_at_bucket(confidence_before_calibration)
    historical_bucket_sample_size = 0
    historical_bucket_family = "none"
    confidence_history_gap_applied = 0.0
    if settings.HISTORICAL_CONFIDENCE_SHRINK_ENABLED:
        historical_shrink = historical_confidence_shrink(
            stage_one_confidence,
            family=confidence_family,
            calibration_buckets=historical_confidence_buckets,
            min_samples=settings.HISTORICAL_CONFIDENCE_SHRINK_MIN_SAMPLES,
        )
        calibrated_confidence = historical_shrink.calibrated_confidence
        confidence_history_gap_applied = max(0.0, stage_one_confidence - calibrated_confidence)
        historical_bucket_sample_size = historical_shrink.sample_size
        historical_bucket_family = historical_shrink.family_used
        if historical_shrink.observed_win_rate is not None:
            historical_win_rate_at_bucket = historical_shrink.observed_win_rate
    else:
        calibrated_confidence = stage_one_confidence
    calibration_delta = confidence_before_calibration - calibrated_confidence
    confidence_calibration_applied = calibration_delta > 0
    if confidence_calibration_applied:
        scaled_bet_size_pct = decision.bet_size_pct
        if confidence_before_calibration > 0:
            scaled_bet_size_pct = decision.bet_size_pct * (
                calibrated_confidence / confidence_before_calibration
            )
        decision = decision.model_copy(
            update={
                "confidence": calibrated_confidence,
                "bet_size_pct": max(0.0, min(1.0, scaled_bet_size_pct)),
                "reasoning": (
                    f"[Confidence calibrated from {confidence_before_calibration:.2f} "
                    f"to {calibrated_confidence:.2f}] {decision.reasoning}"
                ),
            }
        )
    _nd_ceiling = _non_definitive_confidence_ceiling(
        decision,
        settings,
        market=market,
    )
    if decision.confidence > _nd_ceiling:
        _nd_original = decision.confidence
        _nd_scaled_bet = decision.bet_size_pct
        if _nd_original > 0:
            _nd_scaled_bet = decision.bet_size_pct * (_nd_ceiling / _nd_original)
        decision = decision.model_copy(
            update={
                "confidence": _nd_ceiling,
                "bet_size_pct": max(0.0, min(1.0, _nd_scaled_bet)),
                "reasoning": (
                    f"[Confidence capped {_nd_original:.2f} -> {_nd_ceiling:.2f} "
                    f"non-definitive ceiling] {decision.reasoning}"
                ),
            }
        )
        logger.info(
            "Non-definitive confidence ceiling applied: market=%s original=%.2f capped=%.2f",
            market.id,
            _nd_original,
            _nd_ceiling,
            data={
                "market_id": market.id,
                "original_confidence": _nd_original,
                "capped_confidence": _nd_ceiling,
                "ceiling_source": "non_definitive_confidence_ceiling",
            },
        )
    decision, flip_triggered, flip_blocked = _apply_flip_guard(
        market,
        decision,
        anchor_analysis,
        settings,
    )
    market_outcome_mismatch_counted = "[Outcome mismatch]" in (decision.reasoning or "")
    return {
        "decision": decision,
        "analysis_search_profile": getattr(search_config, "profile_name", None),
        "was_refined": was_refined,
        "refinement_reason_text": refinement_reason_text,
        "used_extended_research": used_extended_research,
        "flip_triggered": flip_triggered,
        "flip_blocked": flip_blocked,
        "refinement_skipped_by_flip_precheck": refinement_skipped_by_flip_precheck,
        "flip_precheck_reason": flip_precheck_reason,
        "market_outcome_mismatch_counted": market_outcome_mismatch_counted,
        "edge_repair_attempted": edge_repair_attempted,
        "edge_repair_reason": edge_repair_reason_text,
        "edge_repair_unresolved_reason": edge_repair_unresolved_reason,
        "confidence_before_calibration": confidence_before_calibration,
        "confidence_after_calibration": decision.confidence,
        "confidence_calibration_applied": confidence_calibration_applied,
        "raw_vs_calibrated_delta": calibration_delta,
        "historical_win_rate_at_bucket": historical_win_rate_at_bucket,
        "historical_bucket_sample_size": historical_bucket_sample_size,
        "historical_bucket_family": historical_bucket_family,
        "confidence_history_gap_applied": confidence_history_gap_applied,
        "historical_confidence_shrink_applied": confidence_history_gap_applied > 0.0,
        "definitive_outcome_for_calibration": definitive_outcome_for_calibration,
    }


def main(max_cycles: int | None = None) -> None:
    if max_cycles is not None and max_cycles <= 0:
        raise ValueError("max_cycles must be greater than zero when provided")

    settings = load_settings()

    setup_logging(
        level=settings.LOG_LEVEL,
        file_level=settings.LOG_FILE_LEVEL,
        log_dir=settings.LOG_DIR,
        enable_file_logging=settings.ENABLE_FILE_LOGGING,
        enable_json_logging=settings.ENABLE_JSON_LOGGING,
        enable_colors=settings.ENABLE_COLORED_LOGGING,
    )

    _log_settings_summary(settings)
    logger.info("PredictBot initializing...")

    state_manager = MarketStateManager(settings.STATE_DB_PATH)
    backfilled = state_manager.backfill_outcomes_from_settlements()
    if backfilled:
        logger.info(
            "Backfilled %d trade_outcomes from exchange_settlements",
            backfilled,
            data={"backfilled_outcomes": backfilled},
        )
    scheduler = MarketScheduler(
        reanalysis_cooldown_hours=settings.REANALYSIS_COOLDOWN_HOURS,
        urgent_days_before_close=settings.URGENT_REANALYSIS_DAYS_BEFORE_CLOSE,
        urgent_reanalysis_cooldown_hours=settings.URGENT_REANALYSIS_COOLDOWN_HOURS,
        max_reanalyses_per_market_per_day=settings.MAX_REANALYSES_PER_MARKET_PER_DAY,
    )
    shared_xai_provider = XAIProvider(
        api_key=settings.XAI_API_KEY,
        timeout_seconds=settings.XAI_CLIENT_TIMEOUT_SECONDS,
    )
    grok_client = GrokClient(
        api_key=settings.XAI_API_KEY,
        model=settings.GROK_MODEL,
        model_deep=settings.GROK_MODEL_DEEP,
        min_bet_usdc=settings.MIN_BET_USDC,
        max_bet_usdc=settings.MAX_BET_USDC,
        settings=settings,
        provider=shared_xai_provider,
    )
    logger.debug(
        "Grok client initialized with model=%s model_deep=%s",
        settings.GROK_MODEL,
        settings.GROK_MODEL_DEEP,
    )

    kalshi_client = KalshiClient(
        base_url=settings.KALSHI_API_BASE_URL,
        api_key_id=settings.KALSHI_API_KEY_ID,
        private_key_path=settings.KALSHI_PRIVATE_KEY_PATH,
        order_price_improvement_cents=settings.ORDER_PRICE_IMPROVEMENT_CENTS,
        default_time_in_force=settings.ORDER_DEFAULT_TIF,
        max_fetch_pages=settings.KALSHI_MAX_FETCH_PAGES,
    )
    logger.debug("Kalshi client initialized with base_url=%s", settings.KALSHI_API_BASE_URL)

    try:
        run_bootstrap_checks(
            kalshi_client=kalshi_client,
            skip_api_checks=settings.DRY_RUN,
        )
    except BootstrapError as exc:
        logger.critical(
            "Bootstrap check failed — aborting to avoid wasting API tokens: %s",
            exc,
            data={"error": str(exc)},
        )
        raise

    logger.info(
        "PredictBot started (dry_run=%s, max_cycles=%s)",
        settings.DRY_RUN,
        max_cycles if max_cycles is not None else "unlimited",
    )
    cycle_count = 0
    current_trade_day = datetime.now(timezone.utc).date()
    daily_trade_count = 0
    daily_start_balance: float | None = None
    cumulative_api_cost_estimate_usd = 0.0
    consecutive_zero_order_cycles = 0
    consecutive_zero_execution_yield_cycles = 0
    xai_quota_paused_until: datetime | None = None

    while True:
        cycle_count += 1
        cycle_id = set_correlation_id()
        cycle_start = time.monotonic()
        sleep_seconds = settings.POLL_INTERVAL_SEC

        logger.info("Starting bot cycle #%d", cycle_count)

        try:
            fetch_window_start, fetch_window_end = _build_kalshi_market_fetch_window(
                settings.MARKET_MIN_CLOSE_DAYS,
                settings.MARKET_MAX_CLOSE_DAYS,
            )
            markets = _fetch_markets_with_optional_server_filters(
                kalshi_client,
                use_server_side_filters=settings.KALSHI_SERVER_SIDE_FILTERS_ENABLED,
                fetch_window_start=fetch_window_start,
                fetch_window_end=fetch_window_end,
                mve_filter=settings.KALSHI_MVE_FILTER,
            )
            # Cycle 2 review: snapshot per-fetch pagination metadata so the
            # cycle receipt can record catalog topology (pages_fetched,
            # page_cap_hit, mve_filter active) without scanning DEBUG logs.
            fetch_pages_fetched = int(getattr(kalshi_client, "last_fetch_pages", 0))
            fetch_page_cap_hit = bool(getattr(kalshi_client, "last_fetch_cap_hit", False))
            mve_filter_setting = (settings.KALSHI_MVE_FILTER or "").strip().lower()
            mve_filter_active = mve_filter_setting in {"exclude", "only"}
            fetched_count = len(markets)
            logger.info("Fetched %d raw markets", fetched_count)
            _log_filter_diagnostics(
                markets,
                min_liquidity=settings.MIN_LIQUIDITY_USDC,
                min_volume_24h=settings.MIN_VOLUME_24H,
                min_open_interest=settings.MIN_OPEN_INTEREST,
            )

            filter_stats: dict[str, int] = {}
            markets = _filter_markets(
                markets,
                settings.MIN_LIQUIDITY_USDC,
                settings.MARKET_CATEGORIES_ALLOWLIST,
                settings.MARKET_CATEGORIES_BLOCKLIST,
                family_blocklist=settings.MARKET_FAMILY_BLOCKLIST,
                ticker_prefix_blocklist=settings.MARKET_TICKER_BLOCKLIST_PREFIXES,
                skip_weather_bin_markets=settings.SKIP_WEATHER_BIN_MARKETS,
                skip_crypto_bin_markets=settings.CRYPTO_BIN_MARKET_BLOCKLIST_ENABLED,
                min_close_days=settings.MARKET_MIN_CLOSE_DAYS,
                max_close_days=settings.MARKET_MAX_CLOSE_DAYS,
                stats=filter_stats,
                min_volume_24h=settings.MIN_VOLUME_24H,
                min_open_interest=settings.MIN_OPEN_INTEREST,
                extreme_yes_price_lower=settings.EXTREME_YES_PRICE_LOWER,
                extreme_yes_price_upper=settings.EXTREME_YES_PRICE_UPPER,
                min_tradeable_yes_price=settings.MIN_TRADEABLE_IMPLIED_PRICE,
                max_tradeable_yes_price=settings.MAX_TRADEABLE_IMPLIED_PRICE,
            )
            logger.info("Filtered to %d eligible markets", len(markets))

            # Cycle 2 review: catalog-coverage early-warning. When the page
            # cap was hit AND the post-filter eligible count is below the
            # operator-set floor, log a structured WARNING so we can detect
            # "running out of catalog before running out of cap" before the
            # symptom becomes a sustained cycle_yield_alert ERROR. The auto
            # top-up branch is gated by KALSHI_FETCH_TOPUP_ENABLED (default
            # off) — we want the warning telemetry first, automation later.
            eligible_floor = max(0, int(settings.KALSHI_ELIGIBLE_FLOOR))
            eligible_floor_warning_triggered = False
            if (
                fetch_page_cap_hit
                and eligible_floor > 0
                and len(markets) < eligible_floor
            ):
                eligible_floor_warning_triggered = True
                logger.warning(
                    "Catalog-coverage gap: eligible_markets=%d below floor=%d "
                    "with page cap hit (pages=%d, max=%d, mve_filter=%s); "
                    "consider raising KALSHI_MAX_FETCH_PAGES or enabling "
                    "KALSHI_FETCH_TOPUP_ENABLED",
                    len(markets),
                    eligible_floor,
                    fetch_pages_fetched,
                    settings.KALSHI_MAX_FETCH_PAGES,
                    mve_filter_setting or "unset",
                    data={
                        "eligible_markets": len(markets),
                        "kalshi_eligible_floor": eligible_floor,
                        "pages_fetched": fetch_pages_fetched,
                        "kalshi_max_fetch_pages": settings.KALSHI_MAX_FETCH_PAGES,
                        "page_cap_hit": fetch_page_cap_hit,
                        "mve_filter": mve_filter_setting or None,
                        "kalshi_fetch_topup_enabled": settings.KALSHI_FETCH_TOPUP_ENABLED,
                    },
                )

            markets = _collapse_event_ladders(
                markets,
                ladder_collapse_threshold=settings.LADDER_COLLAPSE_THRESHOLD,
                max_brackets_per_event=settings.MAX_BRACKETS_PER_EVENT,
            )
            markets = _dedupe_markets_by_matchup(markets)

            markets = scheduler.prioritize_markets(markets, state_manager)
            # Counter of market_family for the post-prioritization eligible
            # list. This shows operators *which* families survived the page
            # cap (the exact question raised by the cycle 2 review).
            eligible_market_families = dict(
                Counter(market_family(m) for m in markets)
            )

            cycle_bankroll: float | None = None
            cycle_cash_balance: float | None = None
            cycle_portfolio_value: float | None = None
            try:
                portfolio_balance = kalshi_client.get_portfolio_balance()
                cycle_cash_balance = portfolio_balance.available_balance
                cycle_portfolio_value = portfolio_balance.total_portfolio_value
                cycle_bankroll = cycle_portfolio_value
            except Exception as exc:
                logger.debug(
                    "Kalshi balance lookup failed for position cap: %s",
                    exc,
                    data={"error": str(exc)},
                )
            cycle_trade_day = datetime.now(timezone.utc).date()
            if cycle_trade_day != current_trade_day:
                current_trade_day = cycle_trade_day
                daily_trade_count = 0
                daily_start_balance = cycle_bankroll
            elif daily_start_balance is None and cycle_bankroll is not None:
                daily_start_balance = cycle_bankroll
            if (
                settings.POSITION_SYNC_ENABLED
                and settings.POSITION_SYNC_INTERVAL_CYCLES > 0
                and cycle_count % settings.POSITION_SYNC_INTERVAL_CYCLES == 0
            ):
                try:
                    synced_positions, reconciled_positions = _sync_positions_from_exchange(
                        state_manager=state_manager,
                        kalshi_client=kalshi_client,
                    )
                    logger.info(
                        "Kalshi position sync complete: synced=%d reconciled=%d",
                        synced_positions,
                        reconciled_positions,
                        data={
                            "synced_positions": synced_positions,
                            "reconciled_positions": reconciled_positions,
                        },
                    )
                except Exception as exc:
                    logger.warning(
                        "Kalshi position sync failed: %s",
                        exc,
                        data={"error": str(exc)},
                    )

            if settings.RESOLUTION_SYNC_INTERVAL_CYCLES > 0:
                if cycle_count % settings.RESOLUTION_SYNC_INTERVAL_CYCLES == 0:
                    try:
                        _update_resolved_markets(
                            markets,
                            state_manager,
                            kalshi_client,
                            settings=settings,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Resolution sync failed: %s",
                            exc,
                            data={"error": str(exc)},
                        )
                    try:
                        synced_settlements = _sync_settlements_from_exchange(
                            state_manager=state_manager,
                            kalshi_client=kalshi_client,
                            settings=settings,
                        )
                        if synced_settlements > 0:
                            logger.info(
                                "Kalshi settlement sync complete: imported=%d",
                                synced_settlements,
                                data={"synced_settlements": synced_settlements},
                            )
                        confidence_tier_snapshot = state_manager.get_confidence_tier_outcomes()
                        if confidence_tier_snapshot:
                            logger.info(
                                "Confidence-tier outcome snapshot: %s",
                                ", ".join(
                                    (
                                        f"{row['tier']}: "
                                        f"n={row['sample_size']} "
                                        f"wr={float(row['win_rate']) * 100:.1f}% "
                                        f"pnl={float(row['pnl_total']):.2f}"
                                    )
                                    for row in confidence_tier_snapshot
                                ),
                                data={
                                    "confidence_tier_outcomes": confidence_tier_snapshot,
                                },
                            )
                    except Exception as exc:
                        logger.warning(
                            "Kalshi settlement sync failed: %s",
                            exc,
                            data={"error": str(exc)},
                        )
                    try:
                        external_fill_count = _detect_external_fills(
                            state_manager=state_manager,
                            kalshi_client=kalshi_client,
                        )
                        if external_fill_count > 0:
                            logger.info(
                                "Detected external fills not present in local trade log: count=%d",
                                external_fill_count,
                                data={"external_fill_count": external_fill_count},
                            )
                    except Exception as exc:
                        logger.debug(
                            "External fill detection failed: %s",
                            exc,
                            data={"error": str(exc)},
                        )

            trades_attempted = 0
            trades_filled = 0
            trades_canceled_unfilled = 0
            total_usd_deployed = 0.0
            trades_skipped_confidence = 0
            trades_skipped_balance = 0
            trades_skipped_no_trade = 0
            trades_skipped_edge = 0
            trades_skipped_position = 0
            cycle_definitive_overrides_applied = 0
            trades_skipped_kelly_sub_floor = 0
            cycle_projected_daily_ev_usdc = 0.0
            cycle_primary_targets_selected = 0
            cycle_satellites_selected = 0
            scheduler_skipped_closed = 0
            scheduler_skipped_recently = 0
            scheduler_skipped_other = 0
            position_skipped_saturated = 0
            position_skipped_anchor_opposite = 0
            markets_analyzed = 0
            markets_refined = 0
            execution_candidates = 0
            pre_analysis_passed = 0
            validation_passed = 0
            edge_gate_passed = 0
            score_gate_passed = 0
            decisions_made = 0
            score_gate_blocked = 0
            flip_guard_triggered = 0
            flip_guard_blocked = 0
            flip_precheck_skipped_refinement = 0
            outcome_mismatch_blocked = 0
            analysis_only_mode = False  # Set True when balance is insufficient
            price_bucket_stats = {
                _PRICE_BUCKET_LOW: 0,
                _PRICE_BUCKET_MID: 0,
                _PRICE_BUCKET_HIGH: 0,
            }
            calibration_samples: list[dict[str, Any]] = []
            rejection_breakdown: dict[str, int] = {}
            score_rejection_reason_breakdown: dict[str, int] = {}
            score_near_misses: list[dict[str, Any]] = []
            pre_vs_runtime_score_deltas: list[float] = []
            runtime_score_below_threshold_order_count = 0
            runtime_score_evaluation_count = 0
            score_gate_score_source_counts: dict[str, int] = {}
            rejection_funnel_summary: list[dict[str, Any]] = []
            pre_analysis_rejection_breakdown: dict[str, int] = {}
            execution_family_stats: dict[str, dict[str, float]] = {}
            evidence_basis_breakdown: dict[str, int] = {}
            family_edge_samples: dict[str, list[float]] = {}
            research_queue_cycle_log_maxlen = max(
                1, int(settings.RESEARCH_QUEUE_CYCLE_LOG_MAXLEN)
            )
            research_queue: deque[dict[str, Any]] = deque(
                maxlen=research_queue_cycle_log_maxlen
            )
            pre_analysis_blocked = 0
            pre_analysis_research_routed_count = 0
            # Per-cycle samples for score-distribution telemetry (5f). Surfaces
            # calibration drift in the cycle receipt without forcing operators
            # to grep individual trade audits.
            cycle_pre_score_samples: list[float] = []
            cycle_soft_research_threshold_gap_samples: list[float] = []
            deprioritized_market_samples: list[dict[str, Any]] = []
            # Effective research band (5e): widens linearly under sustained
            # zero-execution drought to capture more markets for learning. Only
            # affects soft-research routing; the deep-analysis MIN_SCORE itself
            # is never moved by this knob, so execution gating is unchanged.
            _research_band_base = max(
                0.0, float(settings.PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND)
            )
            effective_research_band = _research_band_base
            research_band_widened_by = 0.0
            if (
                settings.PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND_ADAPTIVE_ENABLED
                and settings.CYCLE_YIELD_ALERT_ESCALATE_AFTER > 0
            ):
                _band_widen_threshold = (
                    2 * settings.CYCLE_YIELD_ALERT_ESCALATE_AFTER
                )
                if (
                    consecutive_zero_execution_yield_cycles
                    >= _band_widen_threshold
                ):
                    _band_max = max(
                        _research_band_base,
                        float(settings.PRE_ANALYSIS_OPPORTUNITY_RESEARCH_BAND_MAX),
                    )
                    research_band_widened_by = min(
                        _band_max - _research_band_base,
                        0.02
                        * (
                            consecutive_zero_execution_yield_cycles
                            - _band_widen_threshold
                            + 1
                        ),
                    )
                    effective_research_band = (
                        _research_band_base + research_band_widened_by
                    )
            should_trade_but_blocked = 0
            blocked_direct_evidence_count = 0
            participation_tier_breakdown: dict[str, int] = {}
            definitive_outcome_floor_applied_count = 0
            evidence_floor_suppressed_count = 0
            timeout_routed_to_monitor_only_count = 0
            negative_best_score_skipped_count = 0
            should_trade_blocked_breakdown: dict[str, int] = {}
            cycle_prompt_tokens = 0
            cycle_completion_tokens = 0
            cycle_reasoning_tokens = 0
            cycle_cached_tokens = 0
            event_cycle_traded_market_ids: dict[str, set[str]] = {}
            event_cycle_traded_outcomes: dict[str, set[str]] = {}
            confidence_calibration_applied_count = 0
            confidence_calibration_delta_sum = 0.0
            confidence_delta_samples: list[float] = []
            confidence_calibration_historical_win_rates: list[float] = []
            extended_research_market_ids: set[str] = set()
            cycle_balance_start = cycle_bankroll
            last_known_balance = cycle_cash_balance
            last_known_portfolio_value = cycle_bankroll

            def _refresh_last_known_balance() -> None:
                nonlocal last_known_balance, last_known_portfolio_value
                try:
                    refreshed_portfolio = kalshi_client.get_portfolio_balance()
                except Exception:
                    return
                last_known_balance = refreshed_portfolio.available_balance
                last_known_portfolio_value = refreshed_portfolio.total_portfolio_value

            def log_trade_decision(
                *,
                market_id: str,
                question: str,
                decision: dict[str, Any],
                order: dict[str, Any] | None = None,
                execution_audit: dict[str, Any] | None = None,
                score_breakdown: dict[str, Any] | None = None,
            ) -> None:
                normalized_decision = dict(decision or {})
                audit_payload = dict(execution_audit or {}) if isinstance(execution_audit, dict) else {}
                normalized_score_breakdown = _score_breakdown_from_execution_audit(
                    execution_audit=audit_payload,
                    explicit_score_breakdown=score_breakdown,
                )
                if not audit_payload:
                    inferred_action = "order_attempt" if order is not None else "decision_recorded"
                    inferred_reason = "order_attempt" if order is not None else "missing_execution_audit"
                    audit_payload = _build_execution_audit(
                        decision_terminal=(order is None),
                        final_action=inferred_action,
                        final_reason=inferred_reason,
                    )
                if (
                    isinstance(normalized_score_breakdown, dict)
                    and "score_breakdown" not in audit_payload
                ):
                    audit_payload["score_breakdown"] = normalized_score_breakdown
                if "analysis_prompt_tokens" not in audit_payload:
                    audit_payload["analysis_prompt_tokens"] = normalized_decision.get("prompt_tokens")
                if "analysis_completion_tokens" not in audit_payload:
                    audit_payload["analysis_completion_tokens"] = normalized_decision.get("completion_tokens")
                if "analysis_cached_tokens" not in audit_payload:
                    audit_payload["analysis_cached_tokens"] = normalized_decision.get("cached_tokens")
                if "source_requirement_status" not in audit_payload:
                    normalized_family = str(audit_payload.get("market_family") or "").strip().lower()
                    normalized_evidence_basis = str(
                        normalized_decision.get("evidence_basis") or ""
                    ).strip().lower()
                    normalized_edge_source = str(
                        normalized_decision.get("edge_source") or ""
                    ).strip().lower()
                    has_primary_source = bool(
                        str(normalized_decision.get("primary_source_url") or "").strip()
                    )
                    audit_payload["source_requirement_status"] = {
                        "evidence_basis": normalized_evidence_basis,
                        "edge_source": normalized_edge_source,
                        "primary_source_url_present": has_primary_source,
                        "requires_primary_source": normalized_family not in {"sports", "generic"},
                    }
                for _source_audit_key in (
                    "source_match_class",
                    "evidence_floor_suppressed_reason",
                    "evidence_quality_floor_applied",
                ):
                    if (
                        _source_audit_key not in audit_payload
                        and normalized_decision.get(_source_audit_key) is not None
                    ):
                        audit_payload[_source_audit_key] = normalized_decision.get(
                            _source_audit_key
                        )
                if settings.PARTICIPATION_TIER_AUDIT_ENABLED:
                    audit_payload["participation_tier_audit_enabled"] = True
                    audit_payload["participation_tier_gating_enabled"] = (
                        settings.PARTICIPATION_TIER_GATING_ENABLED
                    )
                    participation_tier_inferred = _apply_participation_audit_fields(
                        audit_payload,
                        decision=normalized_decision,
                        settings=settings,
                    )
                    inferred_tier = str(
                        audit_payload.get("participation_tier") or ""
                    ).strip()
                    if participation_tier_inferred and inferred_tier:
                        _record_rejection_reason(
                            participation_tier_breakdown,
                            inferred_tier,
                        )
                if market_id in extended_research_market_ids:
                    audit_payload["extended_research_used"] = True
                normalized_final_action = str(audit_payload.get("final_action") or "").strip().lower()
                for _audit_to_decision_key in (
                    "decision_origin",
                    "market_judgment_available",
                    "participation_tier",
                    "participation_decision",
                    "blocked_conviction",
                    "skip_due_to",
                ):
                    if (
                        _audit_to_decision_key in audit_payload
                        and _audit_to_decision_key not in normalized_decision
                    ):
                        normalized_decision[_audit_to_decision_key] = audit_payload[
                            _audit_to_decision_key
                        ]
                if "calibration_outcome_key" not in normalized_decision:
                    normalized_decision["calibration_outcome_key"] = (
                        f"{market_id}:{str(normalized_decision.get('outcome') or '').strip()}"
                    )
                if normalized_final_action in {"skip", "research_queued"}:
                    final_reason_text = str(
                        audit_payload.get("final_reason") or normalized_final_action
                    )
                    audit_payload.setdefault(
                        "skip_due_to",
                        _skip_due_to_for_reason(final_reason_text, audit_payload),
                    )
                    normalized_decision.setdefault(
                        "no_action_reason",
                        final_reason_text,
                    )
                    if normalized_decision.get("should_trade") is True:
                        audit_payload.setdefault("blocked_conviction", True)
                        audit_payload.setdefault(
                            "blocked_conviction_reason",
                            final_reason_text,
                        )
                        audit_payload.setdefault(
                            "why_not_execution_eligible",
                            f"Downstream gate blocked should_trade=True: {final_reason_text}",
                        )
                        audit_payload.setdefault(
                            "what_to_learn_next",
                            "Compare this blocked conviction against settlement and source quality.",
                        )
                        normalized_decision.setdefault("blocked_conviction", True)
                        normalized_decision.setdefault(
                            "blocked_conviction_reason",
                            final_reason_text,
                        )
                        normalized_decision.setdefault(
                            "skip_due_to",
                            audit_payload.get("skip_due_to"),
                        )
                inferred_edge_market = audit_payload.get("edge_market")
                if inferred_edge_market is None:
                    inferred_edge_market = audit_payload.get("score_edge_market")
                inferred_edge_external = audit_payload.get("edge_external")
                if inferred_edge_external is None:
                    inferred_edge_external = audit_payload.get("score_edge_external")
                if inferred_edge_market is not None and "edge" not in normalized_decision:
                    normalized_decision["edge"] = inferred_edge_market
                if inferred_edge_market is not None and "edge_market" not in normalized_decision:
                    normalized_decision["edge_market"] = inferred_edge_market
                if (
                    inferred_edge_external is not None
                    and "edge_external" not in normalized_decision
                ):
                    normalized_decision["edge_external"] = inferred_edge_external
                _base_log_trade_decision(
                    market_id=market_id,
                    question=question,
                    decision=normalized_decision,
                    order=order,
                    execution_audit=audit_payload,
                )
                try:
                    state_manager.record_decision_receipt(
                        cycle_id=cycle_id,
                        market_id=market_id,
                        decision=normalized_decision,
                        order=order,
                        execution_audit=audit_payload,
                        score_breakdown=normalized_score_breakdown,
                    )
                except Exception as receipt_exc:
                    logger.debug(
                        "Decision receipt persistence failed: market=%s error=%s",
                        market_id,
                        receipt_exc,
                        data={"market_id": market_id, "error": str(receipt_exc)},
                    )
                if market_id in extended_research_market_ids:
                    next_eligible_cycle: int | None = None
                    cooldown_cycles = max(
                        0,
                        int(settings.EXTENDED_RESEARCH_COOLDOWN_CYCLES),
                    )
                    if normalized_final_action in {"skip", "research_queued"}:
                        if cooldown_cycles > 0:
                            next_eligible_cycle = cycle_count + cooldown_cycles
                            audit_payload["extended_research_next_eligible_cycle"] = (
                                next_eligible_cycle
                            )
                    elif normalized_final_action in {"order_attempt", "order_submitted", "dry_run"}:
                        next_eligible_cycle = 0
                    if next_eligible_cycle is not None:
                        try:
                            state_manager.set_market_cooldown_cycle(
                                market_id,
                                next_eligible_cycle,
                            )
                        except Exception as cooldown_exc:
                            logger.debug(
                                "Extended research cooldown persistence failed: market=%s error=%s",
                                market_id,
                                cooldown_exc,
                                data={"market_id": market_id, "error": str(cooldown_exc)},
                            )
                audit = audit_payload
                if normalized_final_action in {"skip", "research_queued"}:
                    rejection_funnel_summary.append(
                        {
                            "market_id": market_id,
                            "market_family": audit.get("market_family"),
                            "evidence_basis": audit.get("evidence_basis_class"),
                            "score": audit.get(
                                "execution_score_final",
                                audit.get("pre_execution_final_score"),
                            ),
                            "final_action": normalized_final_action,
                            "rejection_stage": audit.get("rejection_stage"),
                            "rejection_reason": audit.get("final_reason"),
                        }
                    )

            def _record_should_trade_blocked(reason: str) -> None:
                nonlocal should_trade_but_blocked
                should_trade_but_blocked += 1
                should_trade_blocked_breakdown[reason] = (
                    should_trade_blocked_breakdown.get(reason, 0) + 1
                )

            def _research_learning_target(
                *,
                gate_name: str,
                reason: str,
                market: Market,
                decision: TradeDecision,
            ) -> str:
                normalized_gate = str(gate_name or "").strip().lower()
                normalized_reason = str(reason or "").strip().lower()
                if "historical" in normalized_gate or "historical" in normalized_reason:
                    return (
                        "Review settled prefix outcomes, compare direct evidence quality, "
                        "and require a current primary source plus refreshed market edge before execution."
                    )
                if (
                    "extreme_market_edge" in normalized_gate
                    or "extreme_edge_learning_queue" in normalized_gate
                    or "high_edge" in normalized_gate
                    or "edge_above_reasonable_max" in normalized_reason
                ):
                    return (
                        "Verify the edge against a current orderbook and an independent primary source; "
                        "learn whether the apparent edge was stale, ambiguous, or already priced in."
                    )
                if "hallucinated" in normalized_gate or "hallucinated" in normalized_reason:
                    return (
                        "Find direct primary-source evidence for the exact resolution criteria; "
                        "do not reuse proxy or inferred evidence without a market-specific source."
                    )
                if "research_only" in normalized_gate or "analysis_cap" in normalized_gate:
                    return (
                        "Wait for outcome/settlement data and compare it with the repeated analyses "
                        "before spending more model tokens."
                    )
                source_hint = str(getattr(decision, "primary_source_url", "") or "").strip()
                if source_hint:
                    return (
                        "Recheck the primary source, current orderbook, and exact resolution text "
                        "to decide whether the held edge is now execution-quality."
                    )
                return (
                    "Find a direct primary source, current market price, and explicit pricing-in explanation "
                    "before reconsidering execution."
                )

            def _enqueue_research_candidate(
                *,
                market: Market,
                decision: TradeDecision,
                reason: str,
                gate_name: str,
                threshold_gap: float,
                edge_market: float | None = None,
                edge_required: float | None = None,
                participation_tier: str | None = None,
                why_not_execution_eligible: str | None = None,
                what_to_learn_next: str | None = None,
                decision_origin: str = "grok_analysis",
            ) -> int:
                learning_target = what_to_learn_next or _research_learning_target(
                    gate_name=gate_name,
                    reason=reason,
                    market=market,
                    decision=decision,
                )
                priority = _research_priority_for_reason(
                    gate_name=gate_name,
                    reason=reason,
                    threshold_gap=threshold_gap,
                    participation_tier=participation_tier,
                )
                entry = {
                    "market_id": market.id,
                    "market_family": market_family(market),
                    "gate_name": gate_name,
                    "reason": reason,
                    "learning_hold_reason": reason,
                    "what_to_learn_next": learning_target,
                    "threshold_gap": round(max(0.0, threshold_gap), 4),
                    "edge_market": edge_market,
                    "edge_required": edge_required,
                    "confidence": decision.confidence,
                    "evidence_quality": decision.evidence_quality,
                    "edge_external": decision.edge_external,
                    "primary_source_url": getattr(decision, "primary_source_url", None),
                    "participation_tier": participation_tier,
                    "why_not_execution_eligible": why_not_execution_eligible,
                    "decision_origin": decision_origin,
                    "research_priority": priority,
                }
                if settings.RESEARCH_QUEUE_PRIORITY_ENABLED:
                    queued = list(research_queue)
                    queued.append(entry)
                    queued.sort(
                        key=lambda item: (
                            -float(item.get("research_priority", 0.0) or 0.0),
                            str(item.get("market_id") or ""),
                        )
                    )
                    research_queue.clear()
                    research_queue.extend(queued[:research_queue_cycle_log_maxlen])
                    for index, item in enumerate(research_queue, start=1):
                        if item.get("market_id") == market.id and item.get("reason") == reason:
                            return index
                    return len(research_queue)
                research_queue.append(entry)
                return len(research_queue)

            def _record_runtime_score_order_attempt_if_below(
                audit_context: dict[str, Any],
            ) -> None:
                nonlocal runtime_score_below_threshold_order_count
                try:
                    runtime_score = float(audit_context.get("execution_score_final"))
                    runtime_threshold = float(
                        audit_context.get("execution_score_threshold")
                    )
                except (TypeError, ValueError):
                    return
                if runtime_score < runtime_threshold:
                    runtime_score_below_threshold_order_count += 1

            def _register_order_attempt(
                event_key: str,
                market_id: str,
                outcome: str,
            ) -> None:
                nonlocal daily_trade_count
                daily_trade_count += 1
                if event_key:
                    event_cycle_traded_market_ids.setdefault(event_key, set()).add(market_id)
                    normalized_outcome = _normalize_outcome_key(outcome)
                    if normalized_outcome:
                        event_cycle_traded_outcomes.setdefault(event_key, set()).add(
                            normalized_outcome
                        )

            traded_market_ids: set[str] = set()
            try:
                traded_market_ids = set(state_manager.get_traded_market_ids())
            except Exception as exc:
                logger.debug(
                    "Failed to load traded market ids for pre-analysis funnel: %s",
                    exc,
                    data={"error": str(exc)},
                )

            analysis_candidates: list[dict[str, Any]] = []
            fallback_family_rate_cache: dict[str, tuple[float, int]] = {}
            historical_family_outcome_snapshot: dict[str, dict[str, float | int]] = {}
            historical_prefix_stats: dict[str, Any] = {}
            historical_short_prefix_stats: dict[str, Any] = {}
            historical_family_stats_recent: dict[str, Any] = {}
            historical_confidence_buckets: dict[str, dict[float, dict[str, float | int]]] = {}

            try:
                historical_family_outcome_snapshot = state_manager.get_family_outcome_snapshot(
                    lookback=max(100, settings.PRE_ANALYSIS_HISTORICAL_FAMILY_MIN_SAMPLES * 20),
                )
            except Exception as exc:
                logger.debug(
                    "Historical family outcome snapshot lookup failed: %s",
                    exc,
                    data={"error": str(exc)},
                )
                historical_family_outcome_snapshot = {}
            try:
                historical_prefix_stats = load_ticker_prefix_stats(
                    state_manager,
                    prefix_len=settings.HISTORICAL_TICKER_PREFIX_LEN,
                    lookback_days=settings.HISTORICAL_TICKER_PREFIX_LOOKBACK_DAYS,
                )
            except Exception as exc:
                logger.debug(
                    "Historical ticker-prefix snapshot lookup failed: %s",
                    exc,
                    data={"error": str(exc)},
                )
                historical_prefix_stats = {}
            try:
                historical_short_prefix_stats = load_short_prefix_stats(
                    state_manager,
                    prefix_len=settings.HISTORICAL_SHORT_PREFIX_LEN,
                    lookback_days=settings.HISTORICAL_TICKER_PREFIX_LOOKBACK_DAYS,
                )
            except Exception as exc:
                logger.debug(
                    "Historical short-prefix snapshot lookup failed: %s",
                    exc,
                    data={"error": str(exc)},
                )
                historical_short_prefix_stats = {}
            try:
                historical_family_stats_recent = load_family_stats(
                    state_manager,
                    lookback_days=settings.HISTORICAL_FAMILY_LOOKBACK_DAYS,
                )
            except Exception as exc:
                logger.debug(
                    "Historical family snapshot lookup failed: %s",
                    exc,
                    data={"error": str(exc)},
                )
            if settings.HISTORICAL_CONFIDENCE_SHRINK_ENABLED:
                try:
                    historical_confidence_buckets = (
                        state_manager.load_confidence_calibration_buckets(
                            days=settings.HISTORICAL_CONFIDENCE_SHRINK_LOOKBACK_DAYS,
                        )
                    )
                except Exception as exc:
                    logger.debug(
                        "Historical confidence bucket calibration lookup failed: %s",
                        exc,
                        data={"error": str(exc)},
                    )

            recent_research_entries: dict[str, dict[str, Any]] = {}
            if settings.RESEARCH_QUEUE_ENABLED and settings.RESEARCH_QUEUE_PERSIST_TO_DB:
                try:
                    raw_entries = state_manager.get_active_research_entries(
                        lookback_hours=max(1, settings.RESEARCH_QUEUE_REUSE_LOOKBACK_HOURS),
                        limit=200,
                    )
                    for entry in raw_entries:
                        mid = str(entry.get("market_id") or "").strip()
                        if mid and mid not in recent_research_entries:
                            recent_research_entries[mid] = entry
                except Exception as exc:
                    logger.debug(
                        "Research queue consumer lookup failed: %s",
                        exc,
                        data={"error": str(exc)},
                    )
                if recent_research_entries:
                    logger.debug(
                        "Research queue consumer loaded %d recent entries",
                        len(recent_research_entries),
                        data={"research_queue_consumer_loaded": len(recent_research_entries)},
                    )

            drainable_research_entries: dict[str, dict[str, Any]] = {}
            research_queue_drained_count = 0
            research_queue_drain_skipped_stale_count = 0
            research_queue_drain_skipped_low_priority_count = 0
            research_queue_emergency_probes_count = 0
            research_queue_zero_yield_promotions_count = 0
            if (
                settings.RESEARCH_QUEUE_ENABLED
                and settings.RESEARCH_QUEUE_PERSIST_TO_DB
                and settings.RESEARCH_QUEUE_DRAIN_ENABLED
                and settings.RESEARCH_QUEUE_DRAIN_PER_CYCLE > 0
            ):
                try:
                    excluded_ids = tuple(traded_market_ids)
                except Exception:
                    excluded_ids = ()
                # Snapshot the current cycle's tradeable market IDs so we don't
                # "select" drain candidates that fell out of `_filter_markets`
                # (closed, low-liquidity, expired, etc.) — without this guard
                # the drain log claims selection but the per-market loop never
                # marks the candidate as a probe and research_queue_drained_count
                # stays at 0, producing misleading telemetry.
                current_market_ids: set[str] = {
                    str(getattr(m, "id", "") or "")
                    for m in markets
                    if isinstance(m, Market)
                }
                # Over-fetch beyond the per-cycle drain limit so the priority
                # filter and stale-market filter don't starve the per-cycle quota.
                drain_pool_limit = max(
                    1, settings.RESEARCH_QUEUE_DRAIN_PER_CYCLE * 5
                )
                _drain_min_priority = max(
                    0.0, float(settings.RESEARCH_QUEUE_DRAIN_MIN_PRIORITY)
                )
                drain_per_cycle_quota = max(
                    1, settings.RESEARCH_QUEUE_DRAIN_PER_CYCLE
                )
                try:
                    drain_rows = state_manager.get_drainable_research_entries(
                        min_age_hours=settings.RESEARCH_QUEUE_DRAIN_MIN_AGE_HOURS,
                        max_age_hours=settings.RESEARCH_QUEUE_DRAIN_MAX_AGE_HOURS,
                        limit=drain_pool_limit,
                        excluded_market_ids=excluded_ids,
                        included_market_ids=tuple(current_market_ids),
                    )
                    drain_rows = sorted(
                        drain_rows,
                        key=_research_queue_drain_sort_key,
                    )
                    selected = 0
                    for entry in drain_rows:
                        mid = str(entry.get("market_id") or "").strip()
                        if not mid:
                            continue
                        if mid not in current_market_ids:
                            research_queue_drain_skipped_stale_count += 1
                            continue
                        # Apply the operator-tuned priority floor here (not in
                        # the SQL helper) so we can count low-priority skips
                        # for cycle-receipt observability. Entries with no
                        # priority signal (None) are admitted under the same
                        # "unknown is admissible" rule the helper uses.
                        if _drain_min_priority > 0.0:
                            entry_priority = (
                                state_manager.estimate_research_entry_priority(
                                    entry
                                )
                            )
                            if (
                                entry_priority is not None
                                and entry_priority < _drain_min_priority
                            ):
                                research_queue_drain_skipped_low_priority_count += 1
                                continue
                        drainable_research_entries[mid] = entry
                        selected += 1
                        if selected >= drain_per_cycle_quota:
                            break
                except Exception as exc:
                    logger.debug(
                        "Research queue drain lookup failed: %s",
                        exc,
                        data={"error": str(exc)},
                    )

                # Emergency second-pass drain: when the normal drain selected
                # nothing AND we're in sustained zero-execution-yield mode, the
                # research queue has effectively become a write-only sink. Try
                # again with a halved minimum age so younger current-cycle
                # entries qualify. Keep the current_market_ids guard: entries
                # outside this cycle's filtered market list cannot be marked as
                # probes by the per-market loop and would produce misleading
                # "selected" telemetry without any actual analysis.
                zero_yield_eligible = (
                    settings.CYCLE_YIELD_ALERT_ESCALATE_AFTER > 0
                    and consecutive_zero_execution_yield_cycles
                    >= settings.CYCLE_YIELD_ALERT_ESCALATE_AFTER
                )
                emergency_eligible = (
                    not drainable_research_entries
                    and zero_yield_eligible
                )
                if emergency_eligible:
                    try:
                        emergency_min_age = max(
                            0.0,
                            float(settings.RESEARCH_QUEUE_DRAIN_MIN_AGE_HOURS) / 2.0,
                        )
                        emergency_rows = state_manager.get_drainable_research_entries(
                            min_age_hours=emergency_min_age,
                            max_age_hours=settings.RESEARCH_QUEUE_DRAIN_MAX_AGE_HOURS,
                            limit=drain_pool_limit,
                            excluded_market_ids=excluded_ids,
                            included_market_ids=tuple(current_market_ids),
                        )
                        emergency_rows = sorted(
                            emergency_rows,
                            key=_research_queue_drain_sort_key,
                        )
                        emergency_selected = 0
                        for entry in emergency_rows:
                            mid = str(entry.get("market_id") or "").strip()
                            if not mid:
                                continue
                            if mid in drainable_research_entries:
                                continue
                            if mid not in current_market_ids:
                                research_queue_drain_skipped_stale_count += 1
                                continue
                            if _drain_min_priority > 0.0:
                                entry_priority = (
                                    state_manager.estimate_research_entry_priority(
                                        entry
                                    )
                                )
                                if (
                                    entry_priority is not None
                                    and entry_priority < _drain_min_priority
                                ):
                                    research_queue_drain_skipped_low_priority_count += 1
                                    continue
                            entry["is_drain_emergency_probe"] = True
                            drainable_research_entries[mid] = entry
                            emergency_selected += 1
                            if emergency_selected >= drain_per_cycle_quota:
                                break
                        research_queue_emergency_probes_count = emergency_selected
                    except Exception as exc:
                        logger.debug(
                            "Emergency research-queue drain lookup failed: %s",
                            exc,
                            data={"error": str(exc)},
                        )
                zero_yield_target = max(
                    0,
                    int(getattr(settings, "RESEARCH_QUEUE_ZERO_YIELD_PROMOTIONS", 0) or 0),
                )
                if zero_yield_eligible and zero_yield_target > len(drainable_research_entries):
                    try:
                        promotion_min_age = max(
                            0.0,
                            float(settings.RESEARCH_QUEUE_DRAIN_MIN_AGE_HOURS) / 2.0,
                        )
                        promotion_rows = state_manager.get_drainable_research_entries(
                            min_age_hours=promotion_min_age,
                            max_age_hours=settings.RESEARCH_QUEUE_DRAIN_MAX_AGE_HOURS,
                            limit=max(drain_pool_limit, zero_yield_target * 8),
                            excluded_market_ids=excluded_ids,
                            included_market_ids=tuple(current_market_ids),
                        )
                        promotion_rows = sorted(
                            promotion_rows,
                            key=_research_queue_zero_yield_sort_key,
                        )
                        for entry in promotion_rows:
                            if len(drainable_research_entries) >= zero_yield_target:
                                break
                            mid = str(entry.get("market_id") or "").strip()
                            if not mid:
                                continue
                            if mid in drainable_research_entries:
                                continue
                            if mid not in current_market_ids:
                                research_queue_drain_skipped_stale_count += 1
                                continue
                            entry["is_zero_yield_promotion"] = True
                            entry["zero_yield_promotion_bypassed_priority_floor"] = True
                            drainable_research_entries[mid] = entry
                            research_queue_zero_yield_promotions_count += 1
                    except Exception as exc:
                        logger.debug(
                            "Zero-yield research-queue promotion lookup failed: %s",
                            exc,
                            data={"error": str(exc)},
                        )
                if drainable_research_entries:
                    logger.info(
                        "Research queue drain selected %d candidate(s) for forced re-analysis",
                        len(drainable_research_entries),
                        data={
                            "research_queue_drain_candidates": [
                                {
                                    "market_id": mid,
                                    "queued_at": entry.get("queued_at"),
                                    "reason": entry.get("reason"),
                                    "what_to_learn_next": entry.get(
                                        "what_to_learn_next"
                                    ),
                                    "is_drain_emergency_probe": bool(
                                        entry.get("is_drain_emergency_probe")
                                    ),
                                    "is_zero_yield_promotion": bool(
                                        entry.get("is_zero_yield_promotion")
                                    ),
                                }
                                for mid, entry in drainable_research_entries.items()
                            ],
                            "research_queue_drain_skipped_stale_count": (
                                research_queue_drain_skipped_stale_count
                            ),
                            "research_queue_drain_skipped_low_priority_count": (
                                research_queue_drain_skipped_low_priority_count
                            ),
                            "research_queue_drain_min_priority": _drain_min_priority,
                            "research_queue_emergency_probes_count": (
                                research_queue_emergency_probes_count
                            ),
                            "research_queue_zero_yield_promotions_count": (
                                research_queue_zero_yield_promotions_count
                            ),
                            "consecutive_zero_execution_yield_cycles": (
                                consecutive_zero_execution_yield_cycles
                            ),
                        },
                    )
                elif (
                    research_queue_drain_skipped_stale_count > 0
                    or research_queue_drain_skipped_low_priority_count > 0
                ):
                    logger.debug(
                        "Research queue drain skipped %d stale + %d low-priority "
                        "entries; no candidates promoted",
                        research_queue_drain_skipped_stale_count,
                        research_queue_drain_skipped_low_priority_count,
                        data={
                            "research_queue_drain_skipped_stale_count": (
                                research_queue_drain_skipped_stale_count
                            ),
                            "research_queue_drain_skipped_low_priority_count": (
                                research_queue_drain_skipped_low_priority_count
                            ),
                            "research_queue_drain_min_priority": _drain_min_priority,
                            "consecutive_zero_execution_yield_cycles": (
                                consecutive_zero_execution_yield_cycles
                            ),
                        },
                    )

            _RESEARCH_QUEUE_SCORE_BUMP = 0.05

            def _get_fallback_family_stats(family_name: str) -> tuple[float, int]:
                normalized_family = str(family_name or "").strip().lower()
                if not normalized_family:
                    return 0.0, 0
                cached = fallback_family_rate_cache.get(normalized_family)
                if cached is not None:
                    return cached
                try:
                    computed = state_manager.get_family_fallback_edge_rate(
                        normalized_family,
                        lookback=max(50, settings.PRE_ANALYSIS_FALLBACK_FAMILY_MIN_SAMPLES * 5),
                    )
                except Exception as exc:
                    logger.debug(
                        "Fallback edge family rate lookup failed for %s: %s",
                        normalized_family,
                        exc,
                        data={"family": normalized_family, "error": str(exc)},
                    )
                    computed = (0.0, 0)
                fallback_family_rate_cache[normalized_family] = computed
                return computed

            def _get_historical_family_stats(family_name: str) -> dict[str, float | int]:
                normalized_family = str(family_name or "").strip().lower()
                if not normalized_family:
                    return {}
                return dict(historical_family_outcome_snapshot.get(normalized_family, {}))

            def _evaluate_historical_gate(
                *,
                market_id: str,
                family_name: str,
            ) -> tuple[bool, str | None, dict[str, Any]]:
                return evaluate_market(
                    market_id=market_id,
                    family=family_name,
                    prefix_stats=historical_prefix_stats,
                    family_stats=historical_family_stats_recent,
                    prefix_len=settings.HISTORICAL_TICKER_PREFIX_LEN,
                    prefix_gate_enabled=settings.HISTORICAL_TICKER_PREFIX_GATE_ENABLED,
                    prefix_min_samples=settings.HISTORICAL_TICKER_PREFIX_MIN_SAMPLES,
                    prefix_hard_block_min_samples=settings.HISTORICAL_TICKER_PREFIX_HARD_BLOCK_MIN_SAMPLES,
                    prefix_pnl_cutoff=settings.HISTORICAL_TICKER_PREFIX_PNL_CUTOFF,
                    prefix_win_rate_cutoff=settings.HISTORICAL_TICKER_PREFIX_WIN_RATE_CUTOFF,
                    prefix_shrinkage_enabled=settings.HISTORICAL_TICKER_PREFIX_SHRINKAGE_ENABLED,
                    prefix_prior_win_rate=settings.HISTORICAL_TICKER_PREFIX_PRIOR_WIN_RATE,
                    prefix_prior_strength=settings.HISTORICAL_TICKER_PREFIX_PRIOR_STRENGTH,
                    prefix_shrunk_pnl_cutoff=settings.HISTORICAL_TICKER_PREFIX_SHRUNK_PNL_CUTOFF,
                    prefix_soft_demote_score_penalty=(
                        settings.HISTORICAL_TICKER_PREFIX_SOFT_DEMOTE_SCORE_PENALTY
                    ),
                    family_gate_enabled=settings.HISTORICAL_FAMILY_GATE_ENABLED,
                    family_min_samples=settings.HISTORICAL_FAMILY_MIN_SAMPLES,
                    family_pnl_cutoff=settings.HISTORICAL_FAMILY_PNL_CUTOFF,
                    family_win_rate_cutoff=settings.HISTORICAL_FAMILY_WIN_RATE_CUTOFF,
                )

            def _evaluate_short_prefix_score_penalty(
                *,
                market_id: str,
            ) -> tuple[float, dict[str, Any]]:
                return evaluate_short_prefix_penalty(
                    market_id=market_id,
                    short_prefix_stats=historical_short_prefix_stats,
                    prefix_len=settings.HISTORICAL_SHORT_PREFIX_LEN,
                    min_samples=settings.HISTORICAL_SHORT_PREFIX_MIN_SAMPLES,
                    pnl_cutoff=settings.HISTORICAL_SHORT_PREFIX_PNL_CUTOFF,
                    score_penalty=settings.HISTORICAL_SHORT_PREFIX_SCORE_PENALTY,
                )

            for market in markets:
                logger.debug(
                    "Analyzing market: id=%s, question='%s'",
                    market.id,
                    market.question[:80],
                )
                try:
                    state = state_manager.get_market_state(market.id)
                except Exception as exc:
                    logger.warning(
                        "State lookup failed for market %s: %s",
                        market.id,
                        exc,
                        data={"market_id": market.id, "error": str(exc)},
                    )
                    state = None
                next_eligible_cycle = int(
                    state.next_eligible_cycle if isinstance(state, MarketState) else 0
                )
                if next_eligible_cycle > cycle_count:
                    pre_analysis_blocked += 1
                    _record_rejection_reason(
                        pre_analysis_rejection_breakdown,
                        "extended_research_cooldown",
                    )
                    _record_rejection_reason(
                        rejection_breakdown,
                        "extended_research_cooldown",
                    )
                    logger.debug(
                        "Skipping %s: extended research cooldown until cycle %d",
                        market.id,
                        next_eligible_cycle,
                        data={
                            "market_id": market.id,
                            "next_eligible_cycle": next_eligible_cycle,
                            "current_cycle": cycle_count,
                        },
                    )
                    continue

                should_skip, skip_reason = scheduler.should_skip(market, state, state_manager=state_manager)
                if should_skip:
                    if skip_reason == "market closed":
                        scheduler_skipped_closed += 1
                    elif skip_reason == "recently analyzed":
                        scheduler_skipped_recently += 1
                    elif skip_reason == "daily_reanalysis_cap_reached":
                        scheduler_skipped_other += 1
                    else:
                        scheduler_skipped_other += 1
                    logger.debug(
                        "Skipping %s: %s",
                        market.id,
                        skip_reason,
                        data={"market_id": market.id, "reason": skip_reason},
                    )
                    continue

                existing_position: Position | None = None
                try:
                    existing_position = state_manager.get_position(market.id)
                except Exception as exc:
                    logger.warning(
                        "Position lookup failed for market %s: %s",
                        market.id,
                        exc,
                        data={"market_id": market.id, "error": str(exc)},
                    )

                effective_max_position = _effective_max_position_limit_usdc(
                    settings,
                    cycle_bankroll,
                )
                if (
                    existing_position
                    and existing_position.total_amount_usdc >= effective_max_position
                ):
                    position_skipped_saturated += 1
                    logger.debug(
                        "Skipping %s: position_saturated",
                        market.id,
                        data={
                            "market_id": market.id,
                            "reason": "position_saturated",
                            "existing_position_usdc": existing_position.total_amount_usdc,
                            "effective_max_position_usdc": effective_max_position,
                        },
                    )
                    continue

                anchor_analysis: dict[str, Any] | None = None
                try:
                    anchor_analysis = state_manager.get_anchor_analysis(
                        market.id,
                        settings.MIN_CONFIDENCE,
                    )
                except Exception as exc:
                    logger.warning(
                        "Anchor analysis lookup failed for market %s: %s",
                        market.id,
                        exc,
                        data={"market_id": market.id, "error": str(exc)},
                    )
                if (
                    existing_position
                    and settings.OPPOSITE_OUTCOME_STRATEGY == "block"
                    and anchor_analysis
                ):
                    anchor_outcome = str(anchor_analysis.get("outcome") or "").strip()
                    if anchor_outcome and not _outcomes_match(
                        existing_position.outcome,
                        anchor_outcome,
                    ):
                        position_skipped_anchor_opposite += 1
                        logger.debug(
                            "Skipping %s: position_anchor_outcome_conflict",
                            market.id,
                            data={
                                "market_id": market.id,
                                "reason": "position_anchor_outcome_conflict",
                                "position_outcome": existing_position.outcome,
                                "anchor_outcome": anchor_outcome,
                            },
                        )
                        continue
                traded_before = market.id in traded_market_ids
                had_recent_fallback_edge = False
                try:
                    had_recent_fallback_edge = state_manager.market_has_recent_fallback_edge(
                        market.id,
                        lookback=3,
                    )
                except Exception as exc:
                    logger.debug(
                        "Recent fallback edge lookup failed for market %s: %s",
                        market.id,
                        exc,
                        data={"market_id": market.id, "error": str(exc)},
                    )
                family_name = market_family(market)
                family_fallback_rate, family_fallback_samples = _get_fallback_family_stats(
                    family_name
                )
                historical_family_stats = _get_historical_family_stats(family_name)
                recent_family_stats = historical_family_stats_recent.get(family_name)
                historical_family_pnl_total = float(
                    getattr(recent_family_stats, "pnl_total", 0.0) or 0.0
                )
                historical_family_sample_size = int(
                    getattr(recent_family_stats, "sample_size", 0) or 0
                )
                historical_family_win_rate = float(
                    getattr(recent_family_stats, "win_rate", 0.0) or 0.0
                )
                historical_gate_allowed, historical_gate_reason, historical_gate_metrics = (
                    _evaluate_historical_gate(
                        market_id=market.id,
                        family_name=family_name,
                    )
                )
                short_prefix_score_penalty, short_prefix_metrics = (
                    _evaluate_short_prefix_score_penalty(market_id=market.id)
                )
                pre_analysis_demoted, pre_analysis_demotion_reason, pre_analysis_demotion_data = (
                    _pre_analysis_participation_hold(
                        market=market,
                        state=state if isinstance(state, MarketState) else None,
                        settings=settings,
                        traded_before=traded_before,
                        had_recent_fallback_edge=had_recent_fallback_edge,
                        historical_family_stats=historical_family_stats,
                        fallback_family_edge_rate=family_fallback_rate,
                        fallback_family_sample_size=family_fallback_samples,
                        historical_gate_allowed=historical_gate_allowed,
                        historical_gate_reason=historical_gate_reason,
                        historical_gate_metrics=historical_gate_metrics,
                    )
                )
                if pre_analysis_demoted:
                    pre_analysis_blocked += 1
                    pre_analysis_research_routed_count += 1
                    if pre_analysis_demotion_reason:
                        _record_rejection_reason(
                            pre_analysis_rejection_breakdown,
                            pre_analysis_demotion_reason,
                        )
                        _record_rejection_reason(
                            rejection_breakdown,
                            pre_analysis_demotion_reason,
                        )
                    participation_result = classify_participation(
                        historical_gate=HistoricalGateResult(
                            allowed=historical_gate_allowed or False,
                            reason=historical_gate_reason,
                            metrics=historical_gate_metrics or {},
                            sample_size=int(
                                (historical_gate_metrics or {}).get(
                                    "historical_gate_prefix_sample_size", 0
                                )
                                or 0
                            ),
                            wilson_win_rate_lower_bound=(
                                (historical_gate_metrics or {}).get(
                                    "historical_gate_wilson_lb"
                                )
                            ),
                        ) if historical_gate_metrics else None,
                        pre_analysis_rejection_reason=pre_analysis_demotion_reason,
                        pre_analysis_metadata=pre_analysis_demotion_data,
                    )
                    participation_tier_str = str(participation_result.tier)
                    _record_rejection_reason(
                        participation_tier_breakdown,
                        participation_tier_str,
                    )
                    default_outcome = market.outcomes[0].name if market.outcomes else "YES"
                    research_only_decision = TradeDecision(
                        should_trade=False,
                        outcome=default_outcome,
                        confidence=0.50,
                        bet_size_pct=0.0,
                        reasoning=(
                            f"[ResearchOnly reason={pre_analysis_demotion_reason or 'pre_analysis_soft_research'}] "
                            "Soft-demoted by historical or repeated no-action performance; queued for learning, not execution."
                        ),
                        edge_source="none",
                        evidence_basis="absence_only",
                        evidence_quality=0.0,
                        abstain=True,
                    )
                    research_queue_position: int | None = None
                    queue_for_pre_analysis_research = bool(settings.RESEARCH_QUEUE_ENABLED)
                    if queue_for_pre_analysis_research:
                        research_queue_position = _enqueue_research_candidate(
                            market=market,
                            decision=research_only_decision,
                            reason=pre_analysis_demotion_reason
                            or "pre_analysis_soft_research",
                            gate_name="pre_analysis_performance",
                            threshold_gap=0.0,
                            participation_tier=participation_tier_str,
                            why_not_execution_eligible=(
                                participation_result.why_not_execution_eligible
                            ),
                            what_to_learn_next=participation_result.what_to_learn_next,
                            decision_origin="synthetic_research_queue",
                        )
                    # Strip canonical keys emitted explicitly by _build_execution_audit
                    # below so pre_analysis metadata cannot double-write them.
                    _demotion_audit_extra = {
                        k: v
                        for k, v in pre_analysis_demotion_data.items()
                        if k not in {
                            "participation_tier",
                            "participation_decision",
                            "participation_demotion_reason",
                            "why_not_execution_eligible",
                            "what_to_learn_next",
                            "sample_size_signal",
                        }
                    }
                    _demotion_counterfactuals = _build_counterfactual_audit_fields(
                        reason=pre_analysis_demotion_reason,
                        settings=settings,
                        pre_analysis_score=None,
                        historical_metrics=historical_gate_metrics,
                    )
                    _research_audit = _build_execution_audit(
                        decision_terminal=not queue_for_pre_analysis_research,
                        final_action=(
                            "research_queued"
                            if queue_for_pre_analysis_research
                            else "skip"
                        ),
                        final_reason=pre_analysis_demotion_reason
                        or "pre_analysis_soft_research",
                        market_family=family_name,
                        pre_analysis_score=_PRE_ANALYSIS_RESEARCH_ONLY_SCORE,
                        pre_analysis_soft_research_only=True,
                        pre_analysis_soft_research_reason=pre_analysis_demotion_reason,
                        research_queue_position=research_queue_position,
                        historical_gate_allowed=historical_gate_allowed,
                        historical_gate_reason=historical_gate_reason,
                        historical_gate_metrics=historical_gate_metrics,
                        historical_family_pnl_total=historical_family_pnl_total,
                        historical_family_sample_size=historical_family_sample_size,
                        historical_family_win_rate=historical_family_win_rate,
                        short_prefix_score_penalty=short_prefix_score_penalty,
                        participation_tier=participation_tier_str,
                        participation_decision=str(participation_result.primary_reason),
                        why_not_execution_eligible=participation_result.why_not_execution_eligible,
                        what_to_learn_next=participation_result.what_to_learn_next,
                        pre_analysis_demotion_reason=pre_analysis_demotion_reason,
                        decision_origin="synthetic_research_queue",
                        market_judgment_available=False,
                        skip_due_to=_skip_due_to_for_reason(
                            pre_analysis_demotion_reason,
                            historical_gate_metrics,
                        ),
                        **_SYNTHETIC_DECISION_AUDIT_FIELDS,
                        **_demotion_counterfactuals,
                        **_demotion_audit_extra,
                    )
                    if queue_for_pre_analysis_research and settings.RESEARCH_QUEUE_PERSIST_TO_DB:
                        try:
                            state_manager.record_research_queue_entry(
                                market_id=market.id,
                                cycle_id=cycle_id,
                                gate_name="pre_analysis_performance",
                                reason=pre_analysis_demotion_reason or "pre_analysis_soft_research",
                                threshold_gap=0.0,
                                what_to_learn_next=participation_result.what_to_learn_next,
                                last_decision_json=_research_queue_last_decision_json(
                                    research_only_decision,
                                    _research_audit,
                                ),
                            )
                        except Exception:
                            pass
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=research_only_decision.model_dump(),
                        execution_audit=_research_audit,
                    )
                    logger.debug(
                        "Soft-demoted %s to research-only pre-analysis path (%s) tier=%s",
                        market.id,
                        pre_analysis_demotion_reason or "pre_analysis_soft_research",
                        participation_tier_str,
                        data={
                            "market_id": market.id,
                            "final_action": (
                                "research_queued"
                                if queue_for_pre_analysis_research
                                else "skip"
                            ),
                            "research_queue_position": research_queue_position,
                            "participation_tier": participation_tier_str,
                            "what_to_learn_next": participation_result.what_to_learn_next,
                            **_demotion_audit_extra,
                        },
                    )
                    continue
                _analysis_count_for_cap = (
                    int(state.analysis_count)
                    if isinstance(state, MarketState)
                    and state.analysis_count is not None
                    else 0
                )
                if (
                    settings.MAX_LIFETIME_ANALYSES_PER_MARKET > 0
                    and _analysis_count_for_cap >= settings.MAX_LIFETIME_ANALYSES_PER_MARKET
                    and not traded_before
                ):
                    _cap_reason = "pre_analysis_lifetime_analysis_cap"
                    pre_analysis_research_routed_count += 1
                    _record_rejection_reason(rejection_breakdown, _cap_reason)
                    _record_rejection_reason(pre_analysis_rejection_breakdown, _cap_reason)
                    default_outcome = market.outcomes[0].name if market.outcomes else "YES"
                    _cap_decision = TradeDecision(
                        should_trade=False,
                        outcome=default_outcome,
                        confidence=0.50,
                        bet_size_pct=0.0,
                        reasoning=(
                            f"[ResearchOnly reason={_cap_reason}] "
                            f"Market has {_analysis_count_for_cap} lifetime analyses >= cap "
                            f"{settings.MAX_LIFETIME_ANALYSES_PER_MARKET}; "
                            "queued for outcome learning, not re-analysis."
                        ),
                        edge_source="none",
                        evidence_basis="absence_only",
                        evidence_quality=0.0,
                        abstain=True,
                    )
                    _cap_rq_pos: int | None = None
                    if settings.RESEARCH_QUEUE_ENABLED:
                        _cap_rq_pos = _enqueue_research_candidate(
                            market=market,
                            decision=_cap_decision,
                            reason=_cap_reason,
                            gate_name="lifetime_analysis_cap",
                            threshold_gap=0.0,
                            participation_tier=str(
                                ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE
                            ),
                            why_not_execution_eligible=(
                                "Lifetime analysis cap reached without execution-quality signal"
                            ),
                            what_to_learn_next=(
                                f"Reached {_analysis_count_for_cap} analyses (cap "
                                f"{settings.MAX_LIFETIME_ANALYSES_PER_MARKET}); "
                                "waiting for settlement outcome."
                            ),
                            decision_origin="synthetic_research_queue",
                        )
                    _cap_counterfactuals = _build_counterfactual_audit_fields(
                        reason=_cap_reason,
                        settings=settings,
                        pre_analysis_score=None,
                        historical_metrics=None,
                    )
                    _cap_audit = _build_execution_audit(
                        decision_terminal=False,
                        final_action="research_queued",
                        final_reason=_cap_reason,
                        market_family=family_name,
                        analysis_count=_analysis_count_for_cap,
                        research_queue_position=_cap_rq_pos,
                        participation_tier=str(
                            ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE
                        ),
                        participation_decision=_cap_reason,
                        decision_origin="synthetic_research_queue",
                        market_judgment_available=False,
                        why_not_execution_eligible=(
                            "Lifetime analysis cap reached without execution-quality signal"
                        ),
                        what_to_learn_next=(
                            f"Reached {_analysis_count_for_cap} analyses; wait for outcome learning."
                        ),
                        skip_due_to="not_execution_quality_now",
                        **_SYNTHETIC_DECISION_AUDIT_FIELDS,
                        **_cap_counterfactuals,
                    )
                    if settings.RESEARCH_QUEUE_PERSIST_TO_DB:
                        try:
                            state_manager.record_research_queue_entry(
                                market_id=market.id,
                                cycle_id=cycle_id,
                                gate_name="lifetime_analysis_cap",
                                reason=_cap_reason,
                                threshold_gap=0.0,
                                what_to_learn_next=(
                                    f"Reached {_analysis_count_for_cap} analyses (cap "
                                    f"{settings.MAX_LIFETIME_ANALYSES_PER_MARKET}); "
                                    "waiting for settlement outcome."
                                ),
                                last_decision_json=_research_queue_last_decision_json(
                                    _cap_decision,
                                    _cap_audit,
                                ),
                            )
                        except Exception:
                            pass
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=_cap_decision.model_dump(),
                        execution_audit=_cap_audit,
                    )
                    logger.debug(
                        "Lifetime analysis cap reached for %s: %d >= %d",
                        market.id,
                        _analysis_count_for_cap,
                        settings.MAX_LIFETIME_ANALYSES_PER_MARKET,
                        data={
                            "market_id": market.id,
                            "analysis_count": _analysis_count_for_cap,
                            "cap": settings.MAX_LIFETIME_ANALYSES_PER_MARKET,
                            "research_queue_position": _cap_rq_pos,
                        },
                    )
                    continue
                pre_analysis_score = None
                pre_analysis_breakdown: dict[str, Any] | None = None
                if settings.PRE_ANALYSIS_OPPORTUNITY_ENABLED:
                    pre_analysis_score, pre_analysis_breakdown = _pre_analysis_opportunity_score(
                        market,
                        state if isinstance(state, MarketState) else None,
                        settings,
                        traded_before=traded_before,
                        fallback_family_edge_rate=family_fallback_rate,
                        fallback_family_sample_size=family_fallback_samples,
                        historical_family_stats=historical_family_stats,
                        historical_prefix_stats=historical_prefix_stats,
                        historical_gate_metrics=historical_gate_metrics,
                    )
                    research_entry = recent_research_entries.get(market.id)
                    if research_entry is not None:
                        pre_analysis_score += _RESEARCH_QUEUE_SCORE_BUMP
                        if pre_analysis_breakdown is None:
                            pre_analysis_breakdown = {}
                        pre_analysis_breakdown["research_queue_bump"] = _RESEARCH_QUEUE_SCORE_BUMP
                        pre_analysis_breakdown["previous_research_reason"] = str(
                            research_entry.get("reason") or ""
                        )
                    # Record the post-bump score for cycle-level distribution
                    # telemetry. Includes drain probes so the receipt reflects
                    # the actual score landscape across all scored markets.
                    cycle_pre_score_samples.append(float(pre_analysis_score))
                    drain_entry = drainable_research_entries.get(market.id)
                    is_drain_probe = drain_entry is not None
                    if is_drain_probe:
                        # Forced probe from research-queue drain: bypass the
                        # pre-analysis score gate entirely so the longest-waiting
                        # research-queued markets get a fresh look at deep analysis.
                        # Do NOT mutate pre_analysis_score itself; record the bypass
                        # in the breakdown so receipts show why this market was
                        # admitted despite a low score.
                        if pre_analysis_breakdown is None:
                            pre_analysis_breakdown = {}
                        pre_analysis_breakdown["research_queue_drain_probe"] = True
                        pre_analysis_breakdown["research_queue_drain_queued_at"] = str(
                            drain_entry.get("queued_at") or ""
                        )
                        pre_analysis_breakdown["research_queue_drain_reason"] = str(
                            drain_entry.get("reason") or ""
                        )
                        pre_analysis_breakdown["research_queue_zero_yield_promotion"] = bool(
                            drain_entry.get("is_zero_yield_promotion")
                        )
                        pre_analysis_breakdown[
                            "research_queue_priority_floor_bypassed"
                        ] = bool(
                            drain_entry.get(
                                "zero_yield_promotion_bypassed_priority_floor"
                            )
                        )
                        research_queue_drained_count += 1
                    if not is_drain_probe and pre_analysis_score < settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE:
                        _research_floor = (
                            settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE
                            - effective_research_band
                        )
                        _route_to_soft_research = (
                            settings.PRE_ANALYSIS_OPPORTUNITY_RESEARCH_ENABLED
                            and settings.RESEARCH_QUEUE_ENABLED
                            and pre_analysis_score >= _research_floor
                        )
                        if _route_to_soft_research:
                            pre_analysis_research_routed_count += 1
                            cycle_soft_research_threshold_gap_samples.append(
                                float(
                                    settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE
                                    - pre_analysis_score
                                )
                            )
                            _rejection_tag = "pre_analysis_score_soft_research"
                            _record_rejection_reason(
                                pre_analysis_rejection_breakdown,
                                _rejection_tag,
                            )
                            _record_rejection_reason(
                                rejection_breakdown,
                                _rejection_tag,
                            )
                            _soft_participation_result = classify_participation(
                                pre_analysis_rejection_reason=_rejection_tag,
                                pre_analysis_metadata={
                                    "pre_analysis_score": pre_analysis_score,
                                    "pre_analysis_threshold": (
                                        settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE
                                    ),
                                    "pre_analysis_threshold_gap": float(
                                        settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE
                                        - pre_analysis_score
                                    ),
                                    **(pre_analysis_breakdown or {}),
                                },
                            )
                            _soft_participation_tier = str(
                                _soft_participation_result.tier
                            )
                            _record_rejection_reason(
                                participation_tier_breakdown,
                                _soft_participation_tier,
                            )
                            default_outcome = (
                                market.outcomes[0].name if market.outcomes else "YES"
                            )
                            _soft_research_decision = TradeDecision(
                                should_trade=False,
                                outcome=default_outcome,
                                confidence=0.50,
                                bet_size_pct=0.0,
                                reasoning=(
                                    f"[ResearchOnly reason={_rejection_tag}] "
                                    f"Pre-analysis score {pre_analysis_score:.4f} below "
                                    f"min {settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE:.4f} "
                                    "but within research band; queued for learning."
                                ),
                                edge_source="none",
                                evidence_basis="absence_only",
                                evidence_quality=0.0,
                                abstain=True,
                            )
                            _soft_rq_pos = _enqueue_research_candidate(
                                market=market,
                                decision=_soft_research_decision,
                                reason=_rejection_tag,
                                gate_name="pre_analysis_opportunity_score",
                                threshold_gap=float(
                                    settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE
                                    - pre_analysis_score
                                ),
                                participation_tier=_soft_participation_tier,
                                why_not_execution_eligible=(
                                    _soft_participation_result.why_not_execution_eligible
                                ),
                                what_to_learn_next=(
                                    _soft_participation_result.what_to_learn_next
                                ),
                                decision_origin="synthetic_research_queue",
                            )
                            _soft_counterfactuals = _build_counterfactual_audit_fields(
                                reason=_rejection_tag,
                                settings=settings,
                                pre_analysis_score=pre_analysis_score,
                                historical_metrics=None,
                            )
                            _soft_audit = _build_execution_audit(
                                decision_terminal=False,
                                final_action="research_queued",
                                final_reason=_rejection_tag,
                                market_family=family_name,
                                pre_analysis_score=pre_analysis_score,
                                pre_analysis_breakdown=pre_analysis_breakdown,
                                research_queue_position=_soft_rq_pos,
                                participation_tier=_soft_participation_tier,
                                participation_decision=(
                                    _soft_participation_result.primary_reason
                                ),
                                why_not_execution_eligible=(
                                    _soft_participation_result.why_not_execution_eligible
                                ),
                                what_to_learn_next=(
                                    _soft_participation_result.what_to_learn_next
                                ),
                                decision_origin="synthetic_research_queue",
                                market_judgment_available=False,
                                skip_due_to="weak_pre_analysis_score",
                                **_SYNTHETIC_DECISION_AUDIT_FIELDS,
                                **_soft_counterfactuals,
                            )
                            if settings.RESEARCH_QUEUE_PERSIST_TO_DB:
                                try:
                                    state_manager.record_research_queue_entry(
                                        market_id=market.id,
                                        cycle_id=cycle_id,
                                        gate_name="pre_analysis_opportunity_score",
                                        reason=_rejection_tag,
                                        threshold_gap=float(
                                            settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE
                                            - pre_analysis_score
                                        ),
                                        what_to_learn_next=(
                                            f"Score {pre_analysis_score:.4f} near threshold "
                                            f"{settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE:.4f}; "
                                            "monitor for improved conditions."
                                        ),
                                        last_decision_json=_research_queue_last_decision_json(
                                            _soft_research_decision,
                                            _soft_audit,
                                        ),
                                    )
                                except Exception:
                                    pass
                            log_trade_decision(
                                market_id=market.id,
                                question=market.question,
                                decision=_soft_research_decision.model_dump(),
                                execution_audit=_soft_audit,
                            )
                            logger.debug(
                                "Soft-research routed %s: pre-analysis score %.4f in research band [%.4f, %.4f)",
                                market.id,
                                pre_analysis_score,
                                _research_floor,
                                settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE,
                                data={
                                    "market_id": market.id,
                                    "pre_analysis_score": pre_analysis_score,
                                    "pre_analysis_threshold": settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE,
                                    "research_band_floor": _research_floor,
                                    "research_queue_position": _soft_rq_pos,
                                    "participation_tier": _soft_participation_tier,
                                    "what_to_learn_next": (
                                        _soft_participation_result.what_to_learn_next
                                    ),
                                    **(pre_analysis_breakdown or {}),
                                },
                            )
                        else:
                            pre_analysis_blocked += 1
                            _far_tag = (
                                "pre_analysis_score_far_below_min"
                                if pre_analysis_score < _research_floor
                                else "pre_analysis_score_below_min"
                            )
                            _record_rejection_reason(
                                pre_analysis_rejection_breakdown,
                                _far_tag,
                            )
                            _record_rejection_reason(
                                rejection_breakdown,
                                _far_tag,
                            )
                            if len(deprioritized_market_samples) < 25:
                                _far_participation = classify_participation(
                                    pre_analysis_rejection_reason=_far_tag,
                                    pre_analysis_metadata={
                                        "pre_analysis_score": pre_analysis_score,
                                        "pre_analysis_threshold": (
                                            settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE
                                        ),
                                        "pre_analysis_threshold_gap": float(
                                            settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE
                                            - pre_analysis_score
                                        ),
                                        **(pre_analysis_breakdown or {}),
                                    },
                                )
                                deprioritized_market_samples.append(
                                    {
                                        "market_id": market.id,
                                        "market_family": family_name,
                                        "reason": _far_tag,
                                        "participation_tier": str(
                                            _far_participation.tier
                                        ),
                                        "skip_due_to": _skip_due_to_for_reason(
                                            _far_tag
                                        ),
                                        "pre_analysis_score": round(
                                            float(pre_analysis_score),
                                            4,
                                        ),
                                        "pre_analysis_threshold_gap": round(
                                            float(
                                                settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE
                                                - pre_analysis_score
                                            ),
                                            4,
                                        ),
                                        "what_to_learn_next": (
                                            _far_participation.what_to_learn_next
                                        ),
                                        "breakdown": {
                                            k: v
                                            for k, v in (pre_analysis_breakdown or {}).items()
                                            if k
                                            in {
                                                "pre_score_price_center",
                                                "pre_score_liquidity",
                                                "pre_score_horizon",
                                                "pre_score_family_penalty",
                                                "pre_score_historical_gate_score_penalty",
                                                "pre_score_source_difficulty_penalty",
                                                "pre_score_coinflip_penalty",
                                                "research_queue_bump",
                                            }
                                        },
                                    }
                                )
                            logger.debug(
                                "Skipping %s: pre-analysis opportunity score %.4f < %.4f",
                                market.id,
                                pre_analysis_score,
                                settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE,
                                data={
                                    "market_id": market.id,
                                    "pre_analysis_score": pre_analysis_score,
                                    "pre_analysis_threshold": settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE,
                                    **(pre_analysis_breakdown or {}),
                                },
                            )
                        continue
                _research_context = recent_research_entries.get(market.id)
                analysis_candidates.append(
                    {
                        "market": market,
                        "state": state,
                        "anchor_analysis": anchor_analysis,
                        "market_family": market_family(market),
                        "traded_before": traded_before,
                        "non_actionable_streak": int(
                            state.non_actionable_streak if state else 0
                        ),
                        "pre_analysis_score": pre_analysis_score,
                        "pre_analysis_breakdown": pre_analysis_breakdown,
                        "historical_gate_allowed": historical_gate_allowed,
                        "historical_gate_reason": historical_gate_reason,
                        "historical_gate_metrics": historical_gate_metrics,
                        "historical_family_pnl_total": historical_family_pnl_total,
                        "historical_family_sample_size": historical_family_sample_size,
                        "historical_family_win_rate": historical_family_win_rate,
                        "short_prefix_score_penalty": short_prefix_score_penalty,
                        "short_prefix_metrics": short_prefix_metrics,
                        "force_extended_research": (
                            (
                                is_drain_probe
                                and settings.RESEARCH_QUEUE_DRAIN_FORCE_EXTENDED_RESEARCH
                            )
                            or int(state.non_actionable_streak if state else 0)
                            >= settings.EXTENDED_RESEARCH_AFTER_STREAK
                        ),
                        "market_snapshot_monotonic": time.monotonic(),
                        "previous_research_reason": (
                            str(_research_context.get("reason") or "")
                            if _research_context else None
                        ),
                        "previous_research_what_to_learn": (
                            str(_research_context.get("what_to_learn_next") or "")
                            if _research_context else None
                        ),
                        "is_research_queue_drain_probe": is_drain_probe,
                        "research_queue_drain_entry": (
                            drain_entry if is_drain_probe else None
                        ),
                    }
                )
                pre_analysis_passed += 1

            original_analysis_candidates_count = len(analysis_candidates)
            available_family_distribution = _analysis_candidate_family_counts(
                analysis_candidates
            )
            pre_analysis_scores = {
                str(getattr(candidate.get("market"), "id", "")): float(
                    candidate.get("pre_analysis_score") or 0.0
                )
                for candidate in analysis_candidates
                if isinstance(candidate.get("market"), Market)
            }
            best_pre_analysis_score = max(pre_analysis_scores.values(), default=0.0)
            (
                dynamic_max_markets_per_cycle,
                reduced_candidate_cap_applied,
                negative_score_floor_applied,
            ) = _resolve_dynamic_analysis_candidate_cap(
                settings=settings,
                best_pre_analysis_score=best_pre_analysis_score,
            )
            if negative_score_floor_applied:
                negative_best_score_skipped_count += 1
            analysis_candidate_attempt_limit = _analysis_candidate_attempt_limit(
                settings,
                dynamic_max_markets_per_cycle,
                parallel_analysis_enabled=bool(settings.PARALLEL_ANALYSIS_ENABLED),
            )
            sports_candidate_cap = (
                settings.MAX_SPORTS_CANDIDATES_PER_CYCLE
                if settings.MAX_SPORTS_CANDIDATES_PER_CYCLE > 0
                else None
            )
            analysis_candidates = _cap_analysis_candidates(
                analysis_candidates,
                analysis_candidate_attempt_limit,
                max_weather_candidates_per_cycle=settings.MAX_WEATHER_CANDIDATES_PER_CYCLE,
                max_crypto_candidates_per_cycle=settings.MAX_CRYPTO_CANDIDATES_PER_CYCLE,
                max_speech_candidates_per_cycle=settings.MAX_SPEECH_CANDIDATES_PER_CYCLE,
                max_music_candidates_per_cycle=settings.MAX_MUSIC_CANDIDATES_PER_CYCLE,
                max_sports_candidates_per_cycle=sports_candidate_cap,
                pre_scores=pre_analysis_scores,
            )
            selected_family_distribution = _analysis_candidate_family_counts(
                analysis_candidates
            )
            if len(analysis_candidates) < original_analysis_candidates_count:
                logger.info(
                    "Capped analysis candidates from %d to %d",
                    original_analysis_candidates_count,
                    len(analysis_candidates),
                    data={
                        "analysis_candidates_original": original_analysis_candidates_count,
                        "analysis_candidates_capped": len(analysis_candidates),
                        "max_markets_per_cycle": dynamic_max_markets_per_cycle,
                        "analysis_candidate_attempt_limit": analysis_candidate_attempt_limit,
                        "best_pre_analysis_score": best_pre_analysis_score,
                        "reduced_candidate_cap_applied": reduced_candidate_cap_applied,
                    },
                )
            logger.info(
                "Analysis candidate funnel: available=%d selected=%d",
                original_analysis_candidates_count,
                len(analysis_candidates),
                data={
                    "analysis_candidates_available": original_analysis_candidates_count,
                    "analysis_candidates_selected": len(analysis_candidates),
                    "analysis_candidate_family_distribution_available": available_family_distribution,
                    "analysis_candidate_family_distribution_selected": selected_family_distribution,
                    "scheduler_skipped_closed": scheduler_skipped_closed,
                    "scheduler_skipped_recently": scheduler_skipped_recently,
                    "scheduler_skipped_other": scheduler_skipped_other,
                    "position_skipped_saturated": position_skipped_saturated,
                    "position_skipped_anchor_opposite": position_skipped_anchor_opposite,
                    "pre_analysis_blocked": pre_analysis_blocked,
                    "pre_analysis_rejection_breakdown": pre_analysis_rejection_breakdown,
                    "max_markets_per_cycle": dynamic_max_markets_per_cycle,
                    "analysis_candidate_attempt_limit": analysis_candidate_attempt_limit,
                    "best_pre_analysis_score": best_pre_analysis_score,
                    "reduced_candidate_cap_applied": reduced_candidate_cap_applied,
                },
            )

            daily_drawdown_preflight_blocked_count = 0
            if (
                settings.DAILY_DRAWDOWN_PREFLIGHT_ENABLED
                and analysis_candidates
                and settings.MAX_DAILY_DRAWDOWN_USDC > 0
            ):
                preflight_balance_delta = _daily_balance_delta_usdc(
                    day_start_balance=daily_start_balance,
                    current_balance=last_known_portfolio_value,
                )
                preflight_drawdown = max(
                    0.0,
                    -(preflight_balance_delta if preflight_balance_delta is not None else 0.0),
                )
                if _daily_drawdown_cap_reached(
                    daily_balance_delta=preflight_balance_delta,
                    max_daily_drawdown_usdc=settings.MAX_DAILY_DRAWDOWN_USDC,
                ):
                    _drawdown_reason = "pre_analysis_daily_drawdown_blocked"
                    monitor_tier_str = str(ParticipationTier.MONITOR_ONLY)
                    drawdown_what_to_learn = (
                        "Re-evaluate after drawdown reset (new trading day or "
                        "position close); would-be conviction held for next session."
                    )
                    drawdown_why_not = (
                        f"Daily drawdown cap reached (drawdown=${preflight_drawdown:.2f}, "
                        f"cap=${settings.MAX_DAILY_DRAWDOWN_USDC:.2f}); analysis skipped"
                        " to avoid wasted Grok cost on trades that would be blocked."
                    )
                    for candidate in analysis_candidates:
                        market_for_drawdown = candidate.get("market")
                        if not isinstance(market_for_drawdown, Market):
                            continue
                        family_name_drawdown = str(
                            candidate.get("market_family")
                            or market_family(market_for_drawdown)
                        )
                        default_outcome_drawdown = (
                            market_for_drawdown.outcomes[0].name
                            if market_for_drawdown.outcomes
                            else "YES"
                        )
                        drawdown_decision = TradeDecision(
                            should_trade=False,
                            outcome=default_outcome_drawdown,
                            confidence=0.50,
                            bet_size_pct=0.0,
                            reasoning=(
                                f"[MonitorOnly reason={_drawdown_reason}] "
                                f"{drawdown_why_not}"
                            ),
                            edge_source="none",
                            evidence_basis="absence_only",
                            evidence_quality=0.0,
                            abstain=True,
                        )
                        drawdown_rq_pos: int | None = None
                        if settings.RESEARCH_QUEUE_ENABLED:
                            drawdown_rq_pos = _enqueue_research_candidate(
                                market=market_for_drawdown,
                                decision=drawdown_decision,
                                reason=_drawdown_reason,
                                gate_name="daily_drawdown_preflight",
                                threshold_gap=0.0,
                                participation_tier=monitor_tier_str,
                                why_not_execution_eligible=drawdown_why_not,
                                what_to_learn_next=drawdown_what_to_learn,
                                decision_origin="synthetic_research_queue",
                            )
                        _record_rejection_reason(
                            rejection_breakdown,
                            _drawdown_reason,
                        )
                        _record_rejection_reason(
                            participation_tier_breakdown,
                            monitor_tier_str,
                        )
                        drawdown_counterfactuals = _build_counterfactual_audit_fields(
                            reason=_drawdown_reason,
                            settings=settings,
                            pre_analysis_score=candidate.get("pre_analysis_score"),
                            historical_metrics=candidate.get("historical_gate_metrics"),
                        )
                        drawdown_audit = _build_execution_audit(
                            decision_terminal=False,
                            final_action="research_queued",
                            final_reason=_drawdown_reason,
                            market_family=family_name_drawdown,
                            pre_analysis_score=candidate.get("pre_analysis_score"),
                            pre_analysis_breakdown=candidate.get("pre_analysis_breakdown"),
                            research_queue_position=drawdown_rq_pos,
                            participation_tier=monitor_tier_str,
                            participation_decision=_drawdown_reason,
                            why_not_execution_eligible=drawdown_why_not,
                            what_to_learn_next=drawdown_what_to_learn,
                            decision_origin="synthetic_research_queue",
                            market_judgment_available=False,
                            skip_due_to=_skip_due_to_for_reason(_drawdown_reason),
                            daily_drawdown_usdc=round(preflight_drawdown, 2),
                            max_daily_drawdown_usdc=settings.MAX_DAILY_DRAWDOWN_USDC,
                            **_SYNTHETIC_DECISION_AUDIT_FIELDS,
                            **drawdown_counterfactuals,
                        )
                        if settings.RESEARCH_QUEUE_PERSIST_TO_DB:
                            try:
                                state_manager.record_research_queue_entry(
                                    market_id=market_for_drawdown.id,
                                    cycle_id=cycle_id,
                                    gate_name="daily_drawdown_preflight",
                                    reason=_drawdown_reason,
                                    threshold_gap=0.0,
                                    what_to_learn_next=drawdown_what_to_learn,
                                    last_decision_json=_research_queue_last_decision_json(
                                        drawdown_decision,
                                        drawdown_audit,
                                    ),
                                )
                            except Exception:
                                pass
                        log_trade_decision(
                            market_id=market_for_drawdown.id,
                            question=market_for_drawdown.question,
                            decision=drawdown_decision.model_dump(),
                            execution_audit=drawdown_audit,
                        )
                        daily_drawdown_preflight_blocked_count += 1
                    logger.warning(
                        "Daily drawdown preflight engaged: drawdown=$%.2f cap=$%.2f; "
                        "routed %d candidate(s) to research_queue (MONITOR_ONLY) and skipped Grok",
                        preflight_drawdown,
                        settings.MAX_DAILY_DRAWDOWN_USDC,
                        daily_drawdown_preflight_blocked_count,
                        data={
                            "daily_drawdown_preflight_engaged": True,
                            "daily_drawdown_usdc": round(preflight_drawdown, 2),
                            "max_daily_drawdown_usdc": settings.MAX_DAILY_DRAWDOWN_USDC,
                            "daily_drawdown_preflight_blocked_count": (
                                daily_drawdown_preflight_blocked_count
                            ),
                        },
                    )
                    analysis_candidates = []

            research_only_emissions = 0
            if settings.RESEARCH_QUEUE_ENABLED and analysis_candidates:
                analyzable_candidates: list[dict[str, Any]] = []
                for candidate in analysis_candidates:
                    market = candidate.get("market")
                    if not isinstance(market, Market):
                        continue
                    candidate_state = candidate.get("state")
                    analysis_count_for_research = int(
                        candidate_state.analysis_count
                        if isinstance(candidate_state, MarketState)
                        and candidate_state.analysis_count is not None
                        else 0
                    )
                    non_actionable_streak_for_research = int(
                        candidate_state.non_actionable_streak
                        if isinstance(candidate_state, MarketState)
                        and candidate_state.non_actionable_streak is not None
                        else 0
                    )
                    should_route_research_only = (
                        analysis_count_for_research >= 5
                        and non_actionable_streak_for_research >= 3
                    )
                    if not should_route_research_only:
                        analyzable_candidates.append(candidate)
                        continue
                    candidate["research_only"] = True
                    default_outcome = (
                        market.outcomes[0].name
                        if market.outcomes
                        else "YES"
                    )
                    research_only_decision = TradeDecision(
                        should_trade=False,
                        outcome=default_outcome,
                        confidence=0.50,
                        bet_size_pct=0.0,
                        reasoning=(
                            "[ResearchOnly reason=repeated_non_actionable_research_only] "
                            "Skipping repeated analysis for this cycle and queueing for future calibration learning."
                        ),
                        edge_source="none",
                        evidence_basis="absence_only",
                        evidence_quality=0.0,
                        abstain=True,
                    )
                    research_queue_position = _enqueue_research_candidate(
                        market=market,
                        decision=research_only_decision,
                        reason="repeated_non_actionable_research_only",
                        gate_name="research_only",
                        threshold_gap=0.0,
                    )
                    _record_rejection_reason(
                        rejection_breakdown,
                        "research_only_repeated_non_actionable",
                    )
                    _research_only_counterfactuals = _build_counterfactual_audit_fields(
                        reason="repeated_non_actionable_research_only",
                        settings=settings,
                        pre_analysis_score=candidate.get("pre_analysis_score"),
                        historical_metrics=candidate.get("historical_gate_metrics"),
                    )
                    _research_only_audit = _build_execution_audit(
                        decision_terminal=False,
                        final_action="research_queued",
                        final_reason="repeated_non_actionable_research_only",
                        market_family=str(candidate.get("market_family") or market_family(market)),
                        analysis_count=analysis_count_for_research,
                        non_actionable_streak=non_actionable_streak_for_research,
                        pre_analysis_score=candidate.get("pre_analysis_score"),
                        pre_analysis_breakdown=candidate.get("pre_analysis_breakdown"),
                        historical_gate_allowed=candidate.get("historical_gate_allowed"),
                        historical_gate_reason=candidate.get("historical_gate_reason"),
                        historical_gate_metrics=candidate.get("historical_gate_metrics"),
                        historical_family_pnl_total=candidate.get("historical_family_pnl_total"),
                        historical_family_sample_size=candidate.get("historical_family_sample_size"),
                        short_prefix_score_penalty=candidate.get("short_prefix_score_penalty"),
                        research_only=True,
                        research_queue_position=research_queue_position,
                        participation_tier=str(
                            ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE
                        ),
                        participation_decision="repeated_non_actionable_research_only",
                        decision_origin="synthetic_research_queue",
                        market_judgment_available=False,
                        skip_due_to="not_execution_quality_now",
                        why_not_execution_eligible=(
                            "Repeated non-actionable analyses without execution-quality signal"
                        ),
                        what_to_learn_next=(
                            "Wait for materially new evidence, price movement, or settlement outcome "
                            "before spending another full analysis call."
                        ),
                        **_SYNTHETIC_DECISION_AUDIT_FIELDS,
                        **_research_only_counterfactuals,
                    )
                    if settings.RESEARCH_QUEUE_PERSIST_TO_DB:
                        try:
                            state_manager.record_research_queue_entry(
                                market_id=market.id,
                                cycle_id=cycle_id,
                                gate_name="research_only",
                                reason="repeated_non_actionable_research_only",
                                threshold_gap=0.0,
                                what_to_learn_next=(
                                    "Wait for materially new evidence, price movement, "
                                    "or settlement outcome before spending another full analysis call."
                                ),
                                last_decision_json=_research_queue_last_decision_json(
                                    research_only_decision,
                                    _research_only_audit,
                                ),
                            )
                        except Exception:
                            pass
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=research_only_decision.model_dump(),
                        execution_audit=_research_only_audit,
                    )
                    research_only_emissions += 1
                if research_only_emissions > 0:
                    logger.info(
                        "Rerouted %d repetitive non-actionable markets to research-only queue",
                        research_only_emissions,
                        data={"research_only_emissions": research_only_emissions},
                    )
                analysis_candidates = analyzable_candidates

            # -- xAI quota-exhaustion cross-cycle breaker --
            xai_quota_paused_this_cycle = False
            if (
                settings.XAI_QUOTA_BREAKER_ENABLED
                and xai_quota_paused_until is not None
            ):
                if datetime.now(timezone.utc) < xai_quota_paused_until:
                    logger.info(
                        "xAI quota pause active; skipping analysis phase (resumes %s)",
                        xai_quota_paused_until.isoformat(),
                        data={
                            "xai_quota_paused": True,
                            "paused_until_utc": xai_quota_paused_until.isoformat(),
                        },
                    )
                    xai_quota_paused_this_cycle = True
                else:
                    logger.info(
                        "xAI quota pause expired; resuming analysis",
                        data={"xai_quota_paused": False},
                    )
                    xai_quota_paused_until = None

            analysis_results: dict[str, dict[str, Any]] = {}
            analysis_phase_start = time.monotonic()
            analysis_candidates_count = len(analysis_candidates)
            parallel_analysis_requested = (
                settings.PARALLEL_ANALYSIS_ENABLED
                and analysis_candidates_count > 1
            )
            parallel_analysis_used = False
            analysis_worker_count = 1
            xai_circuit_breaker_triggered = False

            if xai_quota_paused_this_cycle:
                analysis_candidates = []
                analysis_candidates_count = 0
                parallel_analysis_requested = False
                analysis_phase_duration_ms = 0.0

            if parallel_analysis_requested:
                configured_workers = max(1, settings.ANALYSIS_MAX_WORKERS)
                analysis_worker_count = min(configured_workers, analysis_candidates_count)
                logger.info(
                    "Parallel analysis requested: enabled=%s candidates=%d workers=%d",
                    settings.PARALLEL_ANALYSIS_ENABLED,
                    analysis_candidates_count,
                    analysis_worker_count,
                    data={
                        "parallel_analysis_enabled": settings.PARALLEL_ANALYSIS_ENABLED,
                        "analysis_candidates": analysis_candidates_count,
                        "analysis_workers": analysis_worker_count,
                    },
                )
                try:
                    # Drop any cached worker GrokClient from a prior cycle so
                    # the thread-local cache is rebuilt with the current
                    # settings/provider on first use of each worker thread.
                    reset_worker_grok_client_cache()
                    with ThreadPoolExecutor(max_workers=analysis_worker_count) as executor:
                        parallel_analysis_used = True
                        future_to_market = {}
                        for candidate in analysis_candidates:
                            future = executor.submit(
                                _analyze_market_candidate_via_thread_local_client,
                                candidate["market"],
                                candidate["state"],
                                candidate["anchor_analysis"],
                                settings,
                                shared_xai_provider,
                                historical_confidence_buckets,
                                cycle_id,
                                bool(candidate.get("force_extended_research")),
                                candidate.get("research_queue_drain_entry"),
                            )
                            future_to_market[future] = candidate["market"]

                        for future in as_completed(future_to_market):
                            market = future_to_market[future]
                            try:
                                analysis_results[market.id] = future.result()
                            except Exception as exc:
                                error_text = str(exc)
                                is_timeout = (
                                    isinstance(exc, TimeoutError)
                                    or "grok stream exceeded" in error_text.lower()
                                )
                                is_retriable = is_timeout or _is_retriable_xai_error(error_text)
                                try:
                                    failure_search_profile = build_market_search_config(
                                        settings,
                                        market,
                                    ).profile_name
                                except Exception:
                                    failure_search_profile = None
                                analysis_results[market.id] = {
                                    "analysis_failed": True,
                                    "analysis_error": error_text,
                                    "analysis_error_type": type(exc).__name__,
                                    "analysis_error_retriable_xai": is_retriable,
                                    "analysis_error_quota_exhausted": _is_quota_exhausted_xai_error(
                                        error_text
                                    ),
                                    "analysis_is_timeout": is_timeout,
                                    "analysis_search_profile": failure_search_profile,
                                    "was_refined": False,
                                    "refinement_reason_text": None,
                                    "used_extended_research": False,
                                    "flip_triggered": False,
                                    "flip_blocked": False,
                                    "refinement_skipped_by_flip_precheck": False,
                                    "flip_precheck_reason": None,
                                    "market_outcome_mismatch_counted": False,
                                }
                                logger.error(
                                    "Failed to analyze market %s: %s",
                                    market.id,
                                    exc,
                                    data={
                                        "market_id": market.id,
                                        "error": error_text,
                                        "is_timeout": is_timeout,
                                        "is_retriable": is_retriable,
                                    },
                                )
                                if _is_quota_exhausted_xai_error(error_text):
                                    if settings.XAI_QUOTA_BREAKER_ENABLED:
                                        pause_until = datetime.now(timezone.utc) + timedelta(
                                            minutes=max(1, settings.XAI_QUOTA_PAUSE_MINUTES)
                                        )
                                        xai_quota_paused_until = pause_until
                                        logger.error(
                                            "xAI quota exhausted; pausing analysis for %d minutes until %s",
                                            settings.XAI_QUOTA_PAUSE_MINUTES,
                                            pause_until.isoformat(),
                                            data={
                                                "xai_quota_exhausted": True,
                                                "pause_minutes": settings.XAI_QUOTA_PAUSE_MINUTES,
                                                "paused_until_utc": pause_until.isoformat(),
                                            },
                                        )
                                logger.info(
                                    "Market %s captured for participation-tier failure routing",
                                    market.id,
                                    data={
                                        "market_id": market.id,
                                        "final_action": "pending_failure_routing",
                                        "analysis_is_timeout": is_timeout,
                                        "analysis_error_retriable_xai": is_retriable,
                                    },
                                )
                    if parallel_analysis_used and settings.XAI_CIRCUIT_BREAKER_MAX_FAILURES > 0:
                        xai_failure_count = sum(
                            1 for r in analysis_results.values()
                            if r.get("analysis_failed") and r.get("analysis_error_retriable_xai")
                        )
                        if xai_failure_count >= settings.XAI_CIRCUIT_BREAKER_MAX_FAILURES:
                            xai_circuit_breaker_triggered = True
                            logger.warning(
                                "xAI circuit breaker would have triggered: %d retriable failures detected in parallel batch",
                                xai_failure_count,
                                data={
                                    "xai_circuit_breaker_max_failures": settings.XAI_CIRCUIT_BREAKER_MAX_FAILURES,
                                    "xai_retriable_failures": xai_failure_count,
                                    "total_results": len(analysis_results),
                                },
                            )
                    if parallel_analysis_used and settings.XAI_QUOTA_BREAKER_ENABLED:
                        quota_hit = any(
                            r.get("analysis_failed") and r.get("analysis_error_quota_exhausted")
                            for r in analysis_results.values()
                        )
                        if quota_hit:
                            pause_until = datetime.now(timezone.utc) + timedelta(
                                minutes=max(1, settings.XAI_QUOTA_PAUSE_MINUTES)
                            )
                            xai_quota_paused_until = pause_until
                            logger.error(
                                "xAI quota exhausted (parallel batch); pausing analysis for %d minutes until %s",
                                settings.XAI_QUOTA_PAUSE_MINUTES,
                                pause_until.isoformat(),
                                data={
                                    "xai_quota_exhausted": True,
                                    "pause_minutes": settings.XAI_QUOTA_PAUSE_MINUTES,
                                    "paused_until_utc": pause_until.isoformat(),
                                },
                            )
                except Exception as exc:
                    parallel_analysis_used = False
                    analysis_results.clear()
                    logger.exception(
                        "Parallel analysis failed; falling back to serial path: %s",
                        exc,
                        data={
                            "error": str(exc),
                            "analysis_candidates": analysis_candidates_count,
                            "analysis_workers": analysis_worker_count,
                        },
                    )

            if not parallel_analysis_used:
                successful_analysis_count = 0
                consecutive_xai_failures = 0
                for candidate_index, candidate in enumerate(analysis_candidates):
                    if successful_analysis_count >= settings.MAX_MARKETS_PER_CYCLE:
                        break
                    market = candidate["market"]
                    try:
                        result = _analyze_market_candidate(
                            market=market,
                            state=candidate["state"],
                            anchor_analysis=candidate["anchor_analysis"],
                            settings=settings,
                            grok_client=grok_client,
                            historical_confidence_buckets=historical_confidence_buckets,
                            force_extended_research=bool(
                                candidate.get("force_extended_research")
                            ),
                            research_queue_context=candidate.get(
                                "research_queue_drain_entry"
                            ),
                        )
                        analysis_results[market.id] = result
                        if result.get("analysis_failed"):
                            if result.get("analysis_error_retriable_xai"):
                                consecutive_xai_failures += 1
                            else:
                                consecutive_xai_failures = 0
                            if result.get("analysis_error_quota_exhausted"):
                                if settings.XAI_QUOTA_BREAKER_ENABLED:
                                    pause_until = datetime.now(timezone.utc) + timedelta(
                                        minutes=max(1, settings.XAI_QUOTA_PAUSE_MINUTES)
                                    )
                                    xai_quota_paused_until = pause_until
                                    logger.error(
                                        "xAI quota exhausted (from result); pausing analysis for %d minutes until %s",
                                        settings.XAI_QUOTA_PAUSE_MINUTES,
                                        pause_until.isoformat(),
                                        data={
                                            "xai_quota_exhausted": True,
                                            "pause_minutes": settings.XAI_QUOTA_PAUSE_MINUTES,
                                            "paused_until_utc": pause_until.isoformat(),
                                        },
                                    )
                                    break
                        else:
                            consecutive_xai_failures = 0
                            successful_analysis_count += 1

                        if (
                            not xai_circuit_breaker_triggered
                            and settings.XAI_CIRCUIT_BREAKER_MAX_FAILURES > 0
                            and consecutive_xai_failures >= settings.XAI_CIRCUIT_BREAKER_MAX_FAILURES
                        ):
                            xai_circuit_breaker_triggered = True
                            remaining_candidates = analysis_candidates[candidate_index + 1 :]
                            for skipped_candidate in remaining_candidates:
                                skipped_market = skipped_candidate["market"]
                                _record_terminal_outcome(
                                    state_manager,
                                    skipped_market.id,
                                    "analysis_skipped_xai_circuit_breaker",
                                )
                            logger.warning(
                                "xAI analysis circuit breaker triggered after %d consecutive failures; skipped %d remaining markets",
                                consecutive_xai_failures,
                                len(remaining_candidates),
                                data={
                                    "xai_circuit_breaker_max_failures": settings.XAI_CIRCUIT_BREAKER_MAX_FAILURES,
                                    "consecutive_xai_failures": consecutive_xai_failures,
                                    "skipped_remaining_markets": len(remaining_candidates),
                                },
                            )
                            break
                    except Exception as exc:
                        error_text = str(exc)
                        logger.error(
                            "Failed to analyze market %s: %s",
                            market.id,
                            exc,
                            data={"market_id": market.id, "error": error_text},
                        )
                        logger.info(
                            "Market %s skipped for this cycle due to analysis failure after retries",
                            market.id,
                            data={
                                "market_id": market.id,
                                "final_action": "skip",
                                "final_reason": "analysis_failure_after_retries",
                            },
                        )
                        _record_terminal_outcome(
                            state_manager,
                            market.id,
                            "analysis_failure",
                        )
                        if _is_retriable_xai_error(error_text):
                            consecutive_xai_failures += 1
                        else:
                            consecutive_xai_failures = 0
                        if _is_quota_exhausted_xai_error(error_text):
                            if settings.XAI_QUOTA_BREAKER_ENABLED:
                                pause_until = datetime.now(timezone.utc) + timedelta(
                                    minutes=max(1, settings.XAI_QUOTA_PAUSE_MINUTES)
                                )
                                xai_quota_paused_until = pause_until
                                logger.error(
                                    "xAI quota exhausted; pausing analysis for %d minutes until %s",
                                    settings.XAI_QUOTA_PAUSE_MINUTES,
                                    pause_until.isoformat(),
                                    data={
                                        "xai_quota_exhausted": True,
                                        "pause_minutes": settings.XAI_QUOTA_PAUSE_MINUTES,
                                        "paused_until_utc": pause_until.isoformat(),
                                    },
                                )
                                break
                        if (
                            not xai_circuit_breaker_triggered
                            and settings.XAI_CIRCUIT_BREAKER_MAX_FAILURES > 0
                            and consecutive_xai_failures >= settings.XAI_CIRCUIT_BREAKER_MAX_FAILURES
                        ):
                            xai_circuit_breaker_triggered = True
                            remaining_candidates = analysis_candidates[candidate_index + 1 :]
                            for skipped_candidate in remaining_candidates:
                                skipped_market = skipped_candidate["market"]
                                _record_terminal_outcome(
                                    state_manager,
                                    skipped_market.id,
                                    "analysis_skipped_xai_circuit_breaker",
                                )
                            logger.warning(
                                "xAI analysis circuit breaker triggered after %d consecutive failures; skipped %d remaining markets",
                                consecutive_xai_failures,
                                len(remaining_candidates),
                                data={
                                    "xai_circuit_breaker_max_failures": settings.XAI_CIRCUIT_BREAKER_MAX_FAILURES,
                                    "consecutive_xai_failures": consecutive_xai_failures,
                                    "skipped_remaining_markets": len(remaining_candidates),
                                },
                            )
                            break
            analysis_phase_duration_ms = round(
                (time.monotonic() - analysis_phase_start) * 1000,
                2,
            )
            logger.info(
                "Analysis phase complete: requested_parallel=%s used_parallel=%s candidates=%d workers=%d duration=%.2fms completed=%d circuit_breaker=%s",
                parallel_analysis_requested,
                parallel_analysis_used,
                analysis_candidates_count,
                analysis_worker_count,
                analysis_phase_duration_ms,
                len(analysis_results),
                xai_circuit_breaker_triggered,
                data={
                    "parallel_analysis_requested": parallel_analysis_requested,
                    "parallel_analysis_used": parallel_analysis_used,
                    "analysis_candidates": analysis_candidates_count,
                    "analysis_workers": analysis_worker_count,
                    "analysis_phase_duration_ms": analysis_phase_duration_ms,
                    "analysis_completed": len(analysis_results),
                    "xai_circuit_breaker_triggered": xai_circuit_breaker_triggered,
                    "xai_circuit_breaker_max_failures": settings.XAI_CIRCUIT_BREAKER_MAX_FAILURES,
                    "analysis_candidate_attempt_limit": analysis_candidate_attempt_limit,
                },
            )
            for analysis_result in analysis_results.values():
                if not isinstance(analysis_result, dict):
                    continue
                decision_for_usage = analysis_result.get("decision")
                if not isinstance(decision_for_usage, TradeDecision):
                    continue
                cycle_prompt_tokens += int(decision_for_usage.prompt_tokens or 0)
                cycle_completion_tokens += int(decision_for_usage.completion_tokens or 0)
                cycle_reasoning_tokens += int(decision_for_usage.reasoning_tokens or 0)
                cycle_cached_tokens += int(decision_for_usage.cached_tokens or 0)

            _prefix_pnl_cache: dict[str, dict[str, float | int]] = {}

            def _get_prefix_pnl(market_id: str) -> dict[str, float | int]:
                prefix = market_id[:settings.HISTORICAL_TICKER_PREFIX_LEN].upper()
                if prefix not in _prefix_pnl_cache:
                    _prefix_pnl_cache[prefix] = state_manager.get_prefix_pnl_stats(prefix)
                return _prefix_pnl_cache[prefix]

            for candidate in analysis_candidates:
                market = candidate.get("market")
                if not isinstance(market, Market):
                    continue
                analysis_result = analysis_results.get(market.id)
                if not isinstance(analysis_result, dict):
                    continue
                decision = analysis_result.get("decision")
                if not isinstance(decision, TradeDecision):
                    continue
                state_for_rank = candidate.get("state")
                repeated_analysis_count = (
                    int(state_for_rank.analysis_count)
                    if isinstance(state_for_rank, MarketState)
                    and state_for_rank.analysis_count is not None
                    else 0
                )
                evidence_basis_for_rank = _decision_evidence_basis(decision)
                suppress_hallucinated_edge_penalty = _should_suppress_hallucinated_edge_penalty(
                    decision=decision,
                    evidence_basis=evidence_basis_for_rank,
                    settings=settings,
                    market=market,
                )
                definitive_eligible_for_rank = _is_definitive_outcome_eligible(
                    decision,
                    settings,
                    market=market,
                )
                implied_prob_for_rank = _get_implied_probability(market, decision.outcome)
                short_prefix_penalty_for_rank = float(
                    candidate.get("short_prefix_score_penalty", 0.0) or 0.0
                )
                pfx_stats = _get_prefix_pnl(market.id or "")
                pfx_n = int(pfx_stats.get("n", 0))
                pfx_pnl = float(pfx_stats.get("total_pnl", 0.0))
                pfx_shrunk = bayesian_shrunk_pnl(pfx_pnl, pfx_n) if pfx_n > 0 else None
                rank_score = compute_final_score(
                    market=market,
                    decision=decision,
                    implied_prob_market=implied_prob_for_rank,
                    **_score_kwargs(
                        settings=settings,
                        repeated_analysis_count=repeated_analysis_count,
                        non_actionable_streak=(
                            int(state_for_rank.non_actionable_streak)
                            if isinstance(state_for_rank, MarketState)
                            and state_for_rank.non_actionable_streak is not None
                            else 0
                        ),
                        is_weather_market=(market_family(market) == "weather"),
                        evidence_basis_class=evidence_basis_for_rank,
                        edge_source=decision.edge_source or "",
                        market_family=market_family(market),
                        short_prefix_penalty=short_prefix_penalty_for_rank,
                        suppress_hallucinated_edge_penalty=suppress_hallucinated_edge_penalty,
                        definitive_outcome_eligible=definitive_eligible_for_rank,
                        historical_family_pnl_total=float(
                            candidate.get("historical_family_pnl_total", 0.0) or 0.0
                        ),
                        historical_family_sample_size=int(
                            candidate.get("historical_family_sample_size", 0) or 0
                        ),
                        historical_prefix_pnl_per_trade=pfx_shrunk,
                        historical_prefix_sample_size=pfx_n,
                    ),
                )
                analysis_result["historical_family_win_rate"] = float(
                    candidate.get("historical_family_win_rate", 0.0) or 0.0
                )
                analysis_result["pre_execution_final_score"] = rank_score.final_score
                analysis_result["pre_execution_rejection_reasons"] = list(
                    rank_score.rejection_reasons
                )
                analysis_result["pre_execution_score_result"] = rank_score
                analysis_result["pre_execution_score_breakdown"] = {
                    "edge_market": rank_score.edge_market,
                    "edge_external": rank_score.edge_external,
                    "evidence_quality": rank_score.evidence_quality,
                    "repeated_analysis_penalty": rank_score.repeated_analysis_penalty,
                    "fallback_edge_penalty": rank_score.fallback_edge_penalty,
                    "proxy_evidence_penalty": rank_score.proxy_evidence_penalty,
                    "overconfidence_penalty": rank_score.overconfidence_penalty,
                    "late_stage_overconfidence_penalty": rank_score.late_stage_overconfidence_penalty,
                    "fallback_high_confidence_penalty": rank_score.fallback_high_confidence_penalty,
                    "extreme_market_edge_penalty": rank_score.extreme_market_edge_penalty,
                    "hallucinated_edge_penalty": rank_score.hallucinated_edge_penalty,
                    "hallucinated_edge_penalty_suppressed": (
                        rank_score.hallucinated_edge_penalty_suppressed
                    ),
                    "liquidity_penalty": rank_score.liquidity_penalty,
                    "staleness_penalty": rank_score.staleness_penalty,
                    "evidence_basis_bonus": rank_score.evidence_basis_bonus,
                    "generic_bin_penalty": rank_score.generic_bin_penalty,
                    "numeric_strike_bin_penalty": rank_score.numeric_strike_bin_penalty,
                    "extreme_confidence_penalty": rank_score.extreme_confidence_penalty,
                    "short_prefix_penalty": rank_score.short_prefix_penalty,
                    "historical_family_bonus": rank_score.historical_family_bonus,
                    "ambiguous_resolution_penalty": rank_score.ambiguous_resolution_penalty,
                }

            analysis_candidates = sorted(
                analysis_candidates,
                key=lambda candidate: _analysis_result_rank(
                    analysis_results.get(candidate["market"].id),
                    historical_family_pnl_total=float(
                        candidate.get("historical_family_pnl_total", 0.0) or 0.0
                    ),
                    historical_family_sample_size=int(
                        candidate.get("historical_family_sample_size", 0) or 0
                    ),
                    historical_family_win_rate=float(
                        candidate.get("historical_family_win_rate", 0.0) or 0.0
                    ),
                ),
                reverse=True,
            )
            logger.debug(
                "Ranked execution queue prepared by pre-execution score",
                data={
                    "top_ranked_markets": [
                        {
                            "market_id": candidate["market"].id,
                            "pre_execution_final_score": (
                                analysis_results.get(candidate["market"].id, {}).get(
                                    "pre_execution_final_score"
                                )
                            ),
                            "pre_execution_rejection_reasons": (
                                analysis_results.get(candidate["market"].id, {}).get(
                                    "pre_execution_rejection_reasons"
                                )
                            ),
                            "pre_execution_should_trade": bool(
                                getattr(
                                    analysis_results.get(candidate["market"].id, {}).get("decision"),
                                    "should_trade",
                                    False,
                                )
                            ),
                        }
                        for candidate in analysis_candidates[:5]
                    ],
                },
            )

            ranking_total_candidates = len(analysis_candidates)
            for ranking_rank, candidate in enumerate(analysis_candidates, start=1):
                if markets_analyzed >= settings.MAX_MARKETS_PER_CYCLE:
                    break
                market = candidate["market"]
                market_family_name = market_family(market)
                state = candidate["state"]
                market_start = time.monotonic()
                market_snapshot_monotonic = candidate.get("market_snapshot_monotonic")
                analysis_result = analysis_results.get(market.id)
                if analysis_result is None:
                    continue
                if analysis_result.get("analysis_failed"):
                    is_timeout_failure = bool(analysis_result.get("analysis_is_timeout"))
                    is_retriable = bool(
                        analysis_result.get("analysis_error_retriable_xai")
                        or analysis_result.get("analysis_error_retriable")
                        or is_timeout_failure
                    )
                    timeout_streak = 1
                    if is_timeout_failure and settings.TIMEOUT_RETRY_AS_MONITOR_ONLY_ENABLED:
                        previous_timeout_signal = False
                        if isinstance(state, MarketState):
                            previous_terminal = str(
                                state.last_terminal_outcome or ""
                            ).strip().lower()
                            previous_timeout_signal = previous_terminal in {
                                "grok_stream_timeout",
                                "monitor_only_timeout",
                            }
                        previous_research = recent_research_entries.get(market.id)
                        if isinstance(previous_research, dict):
                            previous_reason = str(
                                previous_research.get("reason") or ""
                            ).strip().lower()
                            previous_timeout_signal = (
                                previous_timeout_signal
                                or previous_reason == "grok_stream_timeout"
                            )
                        if previous_timeout_signal:
                            timeout_streak = 2
                    if is_timeout_failure:
                        participation_result = classify_participation(
                            timeout_state=TimeoutState(
                                timed_out=True,
                                retriable=is_retriable,
                                timeout_streak=timeout_streak,
                                search_profile=str(
                                    analysis_result.get("analysis_search_profile")
                                    or "generic"
                                ),
                            )
                        )
                    else:
                        participation_result = classify_participation(
                            analysis_failed=True,
                            analysis_error_retriable=is_retriable,
                        )
                    _failure_tier = str(participation_result.tier)
                    _failure_reason = (
                        "grok_stream_timeout"
                        if is_timeout_failure
                        else (
                            "analysis_failure_retriable"
                            if is_retriable
                            else "analysis_failure_after_retries"
                        )
                    )
                    _record_rejection_reason(
                        participation_tier_breakdown,
                        _failure_tier,
                    )
                    if participation_result.tier == ParticipationTier.MONITOR_ONLY:
                        timeout_routed_to_monitor_only_count += 1
                        _record_terminal_outcome(
                            state_manager,
                            market.id,
                            "monitor_only_timeout",
                        )
                    else:
                        if participation_result.tier != ParticipationTier.OPERATIONAL_ERROR_RETRY:
                            _record_terminal_outcome(
                                state_manager,
                                market.id,
                                _failure_reason,
                            )
                    default_outcome = market.outcomes[0].name if market.outcomes else "YES"
                    _failure_decision = TradeDecision(
                        should_trade=False,
                        outcome=default_outcome,
                        confidence=0.50,
                        bet_size_pct=0.0,
                        reasoning=(
                            f"[OperationalHold reason={_failure_reason}] "
                            "Analysis failed before a market judgment could be made."
                        ),
                        edge_source="none",
                        evidence_basis="absence_only",
                        evidence_quality=0.0,
                        abstain=True,
                    )
                    _failure_queue_position: int | None = None
                    queue_failure_for_research = (
                        settings.RESEARCH_QUEUE_ENABLED
                        and participation_result.tier
                        == ParticipationTier.OPERATIONAL_ERROR_RETRY
                    )
                    if queue_failure_for_research:
                        _failure_queue_position = _enqueue_research_candidate(
                            market=market,
                            decision=_failure_decision,
                            reason=_failure_reason,
                            gate_name="analysis_failure",
                            threshold_gap=0.0,
                            participation_tier=_failure_tier,
                            why_not_execution_eligible=(
                                participation_result.why_not_execution_eligible
                            ),
                            what_to_learn_next=participation_result.what_to_learn_next,
                            decision_origin="synthetic_operational_hold",
                        )
                    _failure_action = (
                        "research_queued"
                        if queue_failure_for_research
                        else (
                            "monitor_only"
                            if participation_result.tier == ParticipationTier.MONITOR_ONLY
                            else "skip"
                        )
                    )
                    _failure_counterfactuals = _build_counterfactual_audit_fields(
                        reason=_failure_reason,
                        settings=settings,
                        pre_analysis_score=None,
                        historical_metrics=None,
                    )
                    _failure_audit = _build_execution_audit(
                        decision_terminal=not queue_failure_for_research,
                        final_action=_failure_action,
                        final_reason=_failure_reason,
                        market_family=market_family_name,
                        timeout_streak=timeout_streak if is_timeout_failure else None,
                        participation_tier=_failure_tier,
                        participation_decision=participation_result.primary_reason,
                        why_not_execution_eligible=(
                            participation_result.why_not_execution_eligible
                        ),
                        what_to_learn_next=participation_result.what_to_learn_next,
                        research_queue_position=_failure_queue_position,
                        decision_origin="synthetic_operational_hold",
                        market_judgment_available=False,
                        skip_due_to=_skip_due_to_for_reason(_failure_reason),
                        search_profile=analysis_result.get("analysis_search_profile"),
                        error_type=analysis_result.get("analysis_error_type"),
                        error_message=analysis_result.get("analysis_error"),
                        **_SYNTHETIC_DECISION_AUDIT_FIELDS,
                        **_failure_counterfactuals,
                    )
                    if settings.RESEARCH_QUEUE_PERSIST_TO_DB:
                        try:
                            state_manager.record_research_queue_entry(
                                market_id=market.id,
                                cycle_id=cycle_id,
                                gate_name=(
                                    "grok_timeout"
                                    if is_timeout_failure
                                    else "analysis_failure"
                                ),
                                reason=_failure_reason,
                                what_to_learn_next=participation_result.what_to_learn_next,
                                last_decision_json=_research_queue_last_decision_json(
                                    _failure_decision,
                                    _failure_audit,
                                ),
                            )
                        except Exception:
                            pass
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=_failure_decision.model_dump(),
                        execution_audit=_failure_audit,
                    )
                    logger.info(
                        "Market %s routed after analysis failure: %s",
                        market.id,
                        analysis_result.get("analysis_error"),
                        data={
                            "market_id": market.id,
                            "final_action": _failure_action,
                            "final_reason": _failure_reason,
                            "participation_tier": _failure_tier,
                            "error_type": analysis_result.get("analysis_error_type"),
                            "search_profile": analysis_result.get("analysis_search_profile"),
                        },
                    )
                    continue
                markets_analyzed += 1
                decision = analysis_result["decision"]
                decisions_made += 1
                evidence_basis = _decision_evidence_basis(decision)
                used_extended_research = bool(
                    analysis_result.get("used_extended_research")
                )
                if used_extended_research:
                    extended_research_market_ids.add(market.id)
                evidence_basis_breakdown[evidence_basis] = (
                    evidence_basis_breakdown.get(evidence_basis, 0) + 1
                )
                if getattr(decision, "evidence_floor_suppressed_reason", None):
                    evidence_floor_suppressed_count += 1
                candidate_edge_value = decision.edge_external
                if candidate_edge_value is None:
                    my_prob_value = getattr(decision, "my_prob", None)
                    implied_external_value = getattr(decision, "implied_prob_external", None)
                    if my_prob_value is not None and implied_external_value is not None:
                        candidate_edge_value = float(my_prob_value) - float(implied_external_value)
                if candidate_edge_value is not None:
                    family_edge_samples.setdefault(market_family_name, []).append(
                        float(candidate_edge_value)
                    )
                pre_execution_final_score = float(
                    analysis_result.get("pre_execution_final_score", 0.0) or 0.0
                )
                score_receipt_fields: dict[str, Any] = {}
                pre_execution_score_result = analysis_result.get("pre_execution_score_result")
                if pre_execution_score_result is not None:
                    score_receipt_fields = _score_receipt_fields(pre_execution_score_result)
                analysis_count_for_market = int(
                    state.analysis_count if state is not None and state.analysis_count is not None else 0
                )
                non_actionable_streak_for_market = int(
                    state.non_actionable_streak if state is not None else 0
                )
                event_ticker_prefix = _event_ticker_prefix(market)
                correlated_position_market_ids: list[str] = []
                try:
                    correlated_position_market_ids = (
                        state_manager.get_open_position_market_ids_for_event(event_ticker_prefix)
                    )
                except Exception as exc:
                    logger.debug(
                        "Event position concentration lookup failed for market %s: %s",
                        market.id,
                        exc,
                        data={"market_id": market.id, "error": str(exc)},
                    )
                correlated_positions_count = len(correlated_position_market_ids)
                correlated_position_outcomes: set[str] = set()
                for correlated_market_id in correlated_position_market_ids:
                    if not correlated_market_id or correlated_market_id == market.id:
                        continue
                    try:
                        correlated_position = state_manager.get_position(correlated_market_id)
                    except Exception:
                        continue
                    if correlated_position is None:
                        continue
                    normalized_outcome = _normalize_outcome_key(correlated_position.outcome)
                    if normalized_outcome:
                        correlated_position_outcomes.add(normalized_outcome)
                daily_pnl_estimate = _daily_balance_delta_usdc(
                    day_start_balance=daily_start_balance,
                    current_balance=last_known_portfolio_value,
                )
                short_prefix_metrics = (
                    candidate.get("short_prefix_metrics")
                    if isinstance(candidate.get("short_prefix_metrics"), dict)
                    else {}
                )
                short_prefix_score_penalty = float(
                    candidate.get("short_prefix_score_penalty", 0.0) or 0.0
                )
                audit_context: dict[str, Any] = {
                    "market_family": market_family_name,
                    "research_profile": analysis_result.get("analysis_search_profile"),
                    "ranking_rank": ranking_rank,
                    "ranking_total_candidates": ranking_total_candidates,
                    "pre_analysis_score": candidate.get("pre_analysis_score"),
                    "pre_analysis_breakdown": candidate.get("pre_analysis_breakdown"),
                    "analysis_count": analysis_count_for_market,
                    "non_actionable_streak": non_actionable_streak_for_market,
                    "traded_before": bool(candidate.get("traded_before", False)),
                    "historical_gate_allowed": candidate.get("historical_gate_allowed"),
                    "historical_gate_reason": candidate.get("historical_gate_reason"),
                    "historical_gate_metrics": candidate.get("historical_gate_metrics"),
                    "historical_family_pnl_total": candidate.get(
                        "historical_family_pnl_total"
                    ),
                    "historical_family_samples": candidate.get(
                        "historical_family_sample_size"
                    ),
                    "historical_family_win_rate": candidate.get(
                        "historical_family_win_rate"
                    ),
                    "pre_execution_final_score": pre_execution_final_score,
                    "score_breakdown": analysis_result.get("pre_execution_score_breakdown"),
                    "evidence_basis_class": evidence_basis,
                    "confidence_before_calibration": analysis_result.get(
                        "confidence_before_calibration"
                    ),
                    "confidence_after_calibration": analysis_result.get(
                        "confidence_after_calibration"
                    ),
                    "confidence_calibration_applied": analysis_result.get(
                        "confidence_calibration_applied"
                    ),
                    "raw_vs_calibrated_delta": analysis_result.get(
                        "raw_vs_calibrated_delta"
                    ),
                    "historical_win_rate_at_bucket": analysis_result.get(
                        "historical_win_rate_at_bucket"
                    ),
                    "historical_bucket_sample_size": analysis_result.get(
                        "historical_bucket_sample_size"
                    ),
                    "historical_bucket_family": analysis_result.get(
                        "historical_bucket_family"
                    ),
                    "confidence_history_gap_applied": analysis_result.get(
                        "confidence_history_gap_applied"
                    ),
                    "historical_confidence_shrink_applied": analysis_result.get(
                        "historical_confidence_shrink_applied"
                    ),
                    "definitive_outcome_for_calibration": analysis_result.get(
                        "definitive_outcome_for_calibration"
                    ),
                    "evidence_quality_raw": getattr(
                        decision,
                        "raw_evidence_quality",
                        None,
                    ),
                    "evidence_quality_validated": decision.evidence_quality,
                    "definitive_outcome_detected": bool(
                        getattr(decision, "definitive_outcome_detected", False)
                    ),
                    "evidence_quality_floor_applied": getattr(
                        decision,
                        "evidence_quality_floor_applied",
                        None,
                    ),
                    "source_match_class": getattr(decision, "source_match_class", None),
                    "evidence_floor_suppressed_reason": getattr(
                        decision,
                        "evidence_floor_suppressed_reason",
                        None,
                    ),
                    "event_ticker_prefix": event_ticker_prefix,
                    "correlated_positions_count": correlated_positions_count,
                    "correlated_position_outcomes": sorted(correlated_position_outcomes),
                    "daily_trade_count": daily_trade_count,
                    "daily_pnl_estimate": daily_pnl_estimate,
                    "gated_should_trade": bool(decision.should_trade),
                    "ticker_prefix_short": short_prefix_metrics.get("historical_short_prefix"),
                    "ticker_prefix_short_pnl": short_prefix_metrics.get(
                        "historical_short_prefix_pnl_total"
                    ),
                    "ticker_prefix_short_sample_size": short_prefix_metrics.get(
                        "historical_short_prefix_sample_size"
                    ),
                    "short_prefix_score_penalty": short_prefix_score_penalty,
                    "extended_research_used": used_extended_research,
                    "edge_repair_attempted": bool(
                        analysis_result.get("edge_repair_attempted", False)
                    ),
                    "edge_repair_reason": analysis_result.get("edge_repair_reason"),
                    "edge_repair_unresolved_reason": analysis_result.get(
                        "edge_repair_unresolved_reason"
                    ),
                    "research_only": bool(candidate.get("research_only", False)),
                    "research_queue_drain_probe": bool(
                        candidate.get("is_research_queue_drain_probe", False)
                    ),
                    "primary_source_url_present": bool(
                        str(getattr(decision, "primary_source_url", "") or "").strip()
                    ),
                }
                if candidate.get("is_research_queue_drain_probe"):
                    drain_meta = candidate.get("research_queue_drain_entry") or {}
                    audit_context["research_queue_drain_queued_at"] = (
                        drain_meta.get("queued_at")
                    )
                    audit_context["research_queue_drain_reason"] = (
                        drain_meta.get("reason")
                    )
                audit_context.update(score_receipt_fields)
                compact_pre_execution_score = _compact_score_breakdown(score_receipt_fields)
                if compact_pre_execution_score:
                    audit_context["score_breakdown"] = compact_pre_execution_score
                if pre_execution_score_result is not None:
                    audit_context["fallback_high_confidence_penalty_applied"] = bool(
                        float(
                            getattr(
                                pre_execution_score_result,
                                "fallback_high_confidence_penalty",
                                0.0,
                            )
                            or 0.0
                        )
                        > 0.0
                    )
                if analysis_result.get("confidence_calibration_applied"):
                    confidence_calibration_applied_count += 1
                raw_vs_calibrated_delta = float(
                    analysis_result.get("raw_vs_calibrated_delta", 0.0) or 0.0
                )
                confidence_calibration_delta_sum += raw_vs_calibrated_delta
                confidence_delta_samples.append(raw_vs_calibrated_delta)
                historical_bucket_rate = analysis_result.get("historical_win_rate_at_bucket")
                if isinstance(historical_bucket_rate, (float, int)):
                    confidence_calibration_historical_win_rates.append(
                        float(historical_bucket_rate)
                    )
                was_refined = analysis_result["was_refined"]
                if was_refined:
                    markets_refined += 1
                refinement_reason_text = analysis_result["refinement_reason_text"]
                if analysis_result["refinement_skipped_by_flip_precheck"]:
                    flip_precheck_skipped_refinement += 1
                if analysis_result["flip_triggered"]:
                    flip_guard_triggered += 1
                if analysis_result["flip_blocked"]:
                    flip_guard_blocked += 1
                market_outcome_mismatch_counted = bool(
                    analysis_result["market_outcome_mismatch_counted"]
                )
                if market_outcome_mismatch_counted:
                    outcome_mismatch_blocked += 1
                    logger.warning(
                        "Outcome mismatch blocked trade path: market=%s outcome=%s",
                        market.id,
                        decision.outcome,
                        data={
                            "market_id": market.id,
                            "outcome": decision.outcome,
                        },
                    )

                previous_reasoning_hash: str | None = None
                current_reasoning_hash = _build_reasoning_hash(decision)
                if settings.BAYESIAN_SKIP_STALE_UPDATES:
                    try:
                        previous_reasoning_hash = state_manager.get_last_reasoning_hash(market.id)
                    except Exception as exc:
                        logger.debug(
                            "Reasoning hash lookup failed for market %s: %s",
                            market.id,
                            exc,
                            data={"market_id": market.id, "error": str(exc)},
                        )

                try:
                    state_manager.record_analysis(
                        market.id,
                        decision,
                        is_refined=was_refined,
                        refinement_reason=refinement_reason_text,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to record analysis for market %s: %s",
                        market.id,
                        exc,
                        data={"market_id": market.id, "error": str(exc)},
                    )

                if settings.FLIP_CIRCUIT_BREAKER_ENABLED and decision.should_trade:
                    try:
                        flip_count = state_manager.get_outcome_flip_count(market.id)
                    except Exception as exc:
                        logger.debug(
                            "Flip count lookup failed for market %s: %s",
                            market.id,
                            exc,
                            data={"market_id": market.id, "error": str(exc)},
                        )
                    else:
                        if flip_count >= settings.FLIP_CIRCUIT_BREAKER_MAX_FLIPS:
                            decision = decision.model_copy(
                                update={"should_trade": False, "bet_size_pct": 0.0}
                            )
                            logger.info(
                                "SKIP [%s] '%s' -> flip circuit breaker (flips=%d, max=%d)",
                                market.id,
                                market.question[:40] + "..."
                                if len(market.question) > 40
                                else market.question,
                                flip_count,
                                settings.FLIP_CIRCUIT_BREAKER_MAX_FLIPS,
                                data={
                                    "market_id": market.id,
                                    "flip_count": flip_count,
                                    "flip_circuit_breaker_max": settings.FLIP_CIRCUIT_BREAKER_MAX_FLIPS,
                                },
                            )

                if _is_coinflip_signal(decision):
                    logger.debug(
                        "Coinflip-quality signal noted (penalty-only): market=%s conf=%.2f evidence=%.2f",
                        market.id,
                        decision.confidence,
                        decision.evidence_quality,
                        data={
                            "market_id": market.id,
                            "confidence": decision.confidence,
                            "evidence_quality": decision.evidence_quality,
                        },
                    )

                if decision.abstain:
                    trades_skipped_no_trade += 1
                    _record_rejection_reason(rejection_breakdown, "abstain_low_evidence")
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision.model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="abstain_low_evidence",
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(state_manager, market.id, "abstain_low_evidence")
                    question_short = market.question[:40] + "..." if len(market.question) > 40 else market.question
                    logger.info(
                        "SKIP [%s] '%s' -> abstain (low evidence quality %.2f)",
                        market.id,
                        question_short,
                        decision.evidence_quality,
                    )
                    continue

                if not decision.should_trade:
                    trades_skipped_no_trade += 1
                    _record_rejection_reason(rejection_breakdown, "no_trade_recommended")
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision.model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="no_trade_recommended",
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(state_manager, market.id, "no_trade_recommended")
                    question_short = market.question[:40] + "..." if len(market.question) > 40 else market.question
                    logger.info(
                        "SKIP [%s] '%s' -> no trade recommended",
                        market.id,
                        question_short,
                    )
                    continue
                if (
                    decision.should_trade
                    and settings.NON_SPORTS_REQUIRES_PRIMARY_SOURCE_URL
                    and market_family_name != "sports"
                    and not str(getattr(decision, "primary_source_url", "") or "").strip()
                ):
                    trades_skipped_no_trade += 1
                    _record_should_trade_blocked("non_sports_missing_primary_source")
                    _record_rejection_reason(
                        rejection_breakdown,
                        "non_sports_missing_primary_source",
                    )
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision.model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="non_sports_missing_primary_source",
                            primary_source_url=getattr(decision, "primary_source_url", None),
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(
                        state_manager,
                        market.id,
                        "non_sports_missing_primary_source",
                    )
                    logger.info(
                        "SKIP [%s] -> missing primary_source_url for non-sports should_trade decision",
                        market.id,
                        data={
                            "market_id": market.id,
                            "final_reason": "non_sports_missing_primary_source",
                            "market_family": market_family_name,
                        },
                    )
                    continue
                validation_passed += 1

                if _should_skip_for_balance(
                    available_balance=last_known_balance,
                    min_bet_usdc=settings.MIN_BET_USDC,
                ):
                    analysis_only_mode = True
                    trades_skipped_balance += 1
                    _record_should_trade_blocked("balance_exhausted_skip")
                    _record_rejection_reason(
                        rejection_breakdown,
                        "balance_exhausted_skip",
                    )
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision.model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="balance_exhausted_skip",
                            available_balance=last_known_balance,
                            min_bet_usdc=settings.MIN_BET_USDC,
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(state_manager, market.id, "balance_exhausted_skip")
                    logger.info(
                        "SKIP [%s] -> balance exhausted (available=$%.2f < min_bet=$%.2f)",
                        market.id,
                        last_known_balance,
                        settings.MIN_BET_USDC,
                        data={
                            "market_id": market.id,
                            "final_reason": "balance_exhausted_skip",
                            "available_balance": last_known_balance,
                            "min_bet_usdc": settings.MIN_BET_USDC,
                        },
                    )
                    continue

                if decision.confidence < settings.MIN_CONFIDENCE:
                    override_edge, market_edge = _confidence_gate_override_metrics(market, decision)
                    (
                        confidence_override_allowed,
                        override_min_confidence,
                        override_path,
                    ) = _is_confidence_override_allowed(
                        settings=settings,
                        decision=decision,
                        override_edge=override_edge,
                    )
                    if not confidence_override_allowed and _is_definitive_outcome_eligible(
                        decision,
                        settings,
                        market=market,
                    ):
                        confidence_override_allowed = True
                        override_min_confidence = settings.MIN_CONFIDENCE
                        override_path = "definitive_outcome"
                        logger.info(
                            "Confidence gate override [%s]: definitive_outcome_detected with whitelisted source bypasses MIN_CONFIDENCE (conf=%.2f)",
                            market.id,
                            decision.confidence,
                            data={
                                "market_id": market.id,
                                "confidence": decision.confidence,
                                "min_confidence": settings.MIN_CONFIDENCE,
                                "definitive_outcome_detected": True,
                                "primary_source_url": getattr(
                                    decision, "primary_source_url", None
                                ),
                            },
                        )
                    if confidence_override_allowed:
                        logger.info(
                            "Confidence gate override [%s]: conf %.2f < min %.2f but edge %.3f and evidence %.2f meet override thresholds (path=%s)",
                            market.id,
                            decision.confidence,
                            settings.MIN_CONFIDENCE,
                            override_edge,
                            decision.evidence_quality,
                            override_path,
                            data={
                                "market_id": market.id,
                                "confidence": decision.confidence,
                                "min_confidence": settings.MIN_CONFIDENCE,
                                "override_edge": override_edge,
                                "market_edge": market_edge,
                                "model_edge": decision.edge_external,
                                "evidence_quality": decision.evidence_quality,
                                "override_min_confidence": override_min_confidence,
                                "confidence_override_path": override_path,
                            },
                        )
                    else:
                        trades_skipped_confidence += 1
                        _record_should_trade_blocked("confidence_below_min")
                        _record_rejection_reason(rejection_breakdown, "confidence_below_min")
                        log_trade_decision(
                            market_id=market.id,
                            question=market.question,
                            decision=decision.model_dump(),
                            execution_audit=_build_execution_audit(
                                decision_terminal=True,
                                final_action="skip",
                                final_reason="confidence_below_min",
                                confidence_gate_override_allowed=False,
                                confidence_override_path=override_path,
                                override_edge=override_edge,
                                market_edge=market_edge,
                                override_min_confidence=override_min_confidence,
                                counterfactual_required_confidence=settings.MIN_CONFIDENCE,
                                **audit_context,
                            ),
                        )
                        _record_terminal_outcome(state_manager, market.id, "confidence_below_min")
                        question_short = market.question[:40] + "..." if len(market.question) > 40 else market.question
                        logger.warning(
                            "SKIP [%s] '%s' -> conf %.2f < min %.2f",
                            market.id,
                            question_short,
                            decision.confidence,
                            settings.MIN_CONFIDENCE,
                            data={
                                "market_id": market.id,
                                "final_reason": "confidence_below_min",
                                "pre_execution_final_score": pre_execution_final_score,
                            },
                        )
                        continue

                entry_price = _get_outcome_entry_price(market, decision.outcome)
                implied_prob = _get_implied_probability(market, decision.outcome)
                audit_context["audit_entry_price"] = entry_price
                audit_context["audit_implied_prob_market"] = implied_prob
                audit_context["audit_edge_source"] = decision.edge_source
                if (
                    entry_price is not None
                    and entry_price < settings.VERY_LOW_PRICE_THRESHOLD
                ):
                    trades_skipped_edge += 1
                    _record_should_trade_blocked("entry_price_too_low")
                    _record_rejection_reason(rejection_breakdown, "entry_price_too_low")
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision.model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="entry_price_too_low",
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(state_manager, market.id, "entry_price_too_low")
                    logger.warning(
                        "SKIP [%s] -> entry price %.3f below floor %.3f",
                        market.id,
                        entry_price,
                        settings.VERY_LOW_PRICE_THRESHOLD,
                        data={
                            "market_id": market.id,
                            "final_reason": "entry_price_too_low",
                            "entry_price": entry_price,
                            "entry_price_floor": settings.VERY_LOW_PRICE_THRESHOLD,
                            "score_breakdown": score_receipt_fields,
                        },
                    )
                    continue
                bayesian_posterior_raw: float | None = None
                bayesian_posterior_applied: float | None = None
                bayesian_update_count: int = 0
                lmsr_execution_price: float | None = None
                ineff_signal: float | None = None
                effective_confidence = decision.confidence
                likelihood_ratio = decision.likelihood_ratio

                if settings.BAYESIAN_ENABLED and market.outcomes:
                    try:
                        canonical_outcome = _canonical_outcome_name(market, decision.outcome)
                        skip_stale_update = (
                            settings.BAYESIAN_SKIP_STALE_UPDATES
                            and previous_reasoning_hash is not None
                            and previous_reasoning_hash == current_reasoning_hash
                        )
                        if skip_stale_update:
                            logger.debug(
                                "Bayesian update skipped for stale reasoning: market=%s",
                                market.id,
                                data={
                                    "market_id": market.id,
                                    "reasoning_hash": current_reasoning_hash,
                                },
                            )
                        bayesian_states = _load_or_initialize_bayesian_states(
                            market=market,
                            state_manager=state_manager,
                            settings=settings,
                        )

                        if likelihood_ratio is not None and likelihood_ratio > 0:
                            log_likelihood = log_likelihood_from_ratio(likelihood_ratio)
                            is_binary_market = len(market.outcomes) == 2
                            for market_outcome in market.outcomes:
                                outcome_name = market_outcome.name
                                state_for_outcome = bayesian_states.get(outcome_name)
                                if state_for_outcome is None:
                                    seeded_state = initial_state(1, prior=None)[0]
                                    state_for_outcome = seeded_state
                                    bayesian_states[outcome_name] = seeded_state
                                if _outcomes_match(outcome_name, canonical_outcome):
                                    outcome_log_likelihood = log_likelihood
                                elif is_binary_market:
                                    outcome_log_likelihood = -log_likelihood
                                else:
                                    outcome_log_likelihood = 0.0
                                state_manager.update_bayesian_state(
                                    market_id=market.id,
                                    outcome=outcome_name,
                                    log_prior=state_for_outcome.log_prior,
                                    log_likelihood=outcome_log_likelihood,
                                    count_as_update=not skip_stale_update,
                                )
                            bayesian_states = state_manager.get_bayesian_state(market.id)
                        elif likelihood_ratio is None:
                            logger.debug(
                                "Bayesian update skipped: missing likelihood ratio for market %s",
                                market.id,
                                data={
                                    "market_id": market.id,
                                    "bayesian_enabled": settings.BAYESIAN_ENABLED,
                                },
                            )
                        else:
                            logger.warning(
                                "Bayesian update skipped: invalid likelihood ratio for market %s",
                                market.id,
                                data={
                                    "market_id": market.id,
                                    "likelihood_ratio": likelihood_ratio,
                                },
                            )

                        ordered_states = [
                            bayesian_states[outcome.name]
                            for outcome in market.outcomes
                            if outcome.name in bayesian_states
                        ]
                        if len(ordered_states) == len(market.outcomes):
                            posterior_values = posterior_from_state(ordered_states)
                            for idx, market_outcome in enumerate(market.outcomes):
                                if _outcomes_match(market_outcome.name, canonical_outcome):
                                    bayesian_posterior_raw = posterior_values[idx]
                                    selected_state = bayesian_states.get(market_outcome.name)
                                    bayesian_update_count = (
                                        selected_state.update_count if selected_state else 0
                                    )
                                    break
                            bayesian_posterior_applied = _applied_bayesian_posterior(
                                bayesian_posterior_raw=bayesian_posterior_raw,
                                bayesian_update_count=bayesian_update_count,
                                min_updates_for_trade=settings.BAYESIAN_MIN_UPDATES_FOR_TRADE,
                            )
                            if bayesian_posterior_applied is not None:
                                bayesian_posterior_applied = min(
                                    bayesian_posterior_applied,
                                    settings.BAYESIAN_MAX_POSTERIOR,
                                )
                                capped_confidence = _cap_effective_confidence_for_market(
                                    bayesian_posterior_applied,
                                    market,
                                    settings,
                                )
                                base_confidence = decision.raw_confidence or decision.confidence
                                boost_capped_confidence = _cap_bayesian_confidence_boost(
                                    base_confidence=base_confidence,
                                    candidate_confidence=capped_confidence,
                                    max_boost=settings.BAYESIAN_MAX_CONFIDENCE_BOOST,
                                )
                                if boost_capped_confidence < capped_confidence:
                                    logger.debug(
                                        "Clamped Bayesian confidence boost: market=%s base=%.4f capped=%.4f boost_ceiling=%.4f",
                                        market.id,
                                        base_confidence,
                                        capped_confidence,
                                        boost_capped_confidence,
                                        data={
                                            "market_id": market.id,
                                            "base_confidence": base_confidence,
                                            "bayesian_posterior_applied": bayesian_posterior_applied,
                                            "capped_confidence_before_boost_cap": capped_confidence,
                                            "bayesian_max_confidence_boost": settings.BAYESIAN_MAX_CONFIDENCE_BOOST,
                                            "boost_ceiling": boost_capped_confidence,
                                        },
                                    )
                                    capped_confidence = boost_capped_confidence
                                if capped_confidence < bayesian_posterior_applied:
                                    logger.debug(
                                        "Capped Bayesian posterior: market=%s posterior=%.4f capped=%.4f",
                                        market.id,
                                        bayesian_posterior_applied,
                                        capped_confidence,
                                        data={
                                            "market_id": market.id,
                                            "bayesian_posterior_raw": bayesian_posterior_applied,
                                            "bayesian_posterior_capped": capped_confidence,
                                        },
                                    )
                                effective_confidence = capped_confidence
                            elif bayesian_posterior_raw is not None:
                                logger.debug(
                                    "Bayesian posterior not applied yet: market=%s updates=%d min_updates=%d",
                                    market.id,
                                    bayesian_update_count,
                                    settings.BAYESIAN_MIN_UPDATES_FOR_TRADE,
                                    data={
                                        "market_id": market.id,
                                        "bayesian_update_count": bayesian_update_count,
                                        "bayesian_min_updates": settings.BAYESIAN_MIN_UPDATES_FOR_TRADE,
                                        "bayesian_posterior_raw": bayesian_posterior_raw,
                                    },
                                )
                    except Exception as exc:
                        logger.warning(
                            "Bayesian update failed for market %s: %s",
                            market.id,
                            exc,
                            data={"market_id": market.id, "error": str(exc)},
                        )

                decision_for_edge = (
                    decision.model_copy(update={"confidence": effective_confidence})
                    if effective_confidence != decision.confidence
                    else decision
                )
                decision_for_edge, definitive_floor_was_applied = (
                    _apply_definitive_outcome_floors(
                        decision_for_edge, market, settings
                    )
                )
                if definitive_floor_was_applied:
                    definitive_outcome_floor_applied_count += 1
                    auto_detected = not bool(
                        getattr(decision, "definitive_outcome_detected", False)
                    )
                    decision = decision_for_edge
                    analysis_result["pre_execution_score_result"] = None
                    analysis_result["pre_execution_rejection_reasons"] = []
                    analysis_result["pre_execution_score_breakdown"] = None
                    logger.info(
                        "Definitive outcome floor applied: market=%s eq_after=%.2f conf=%.2f source=%s",
                        market.id,
                        decision_for_edge.evidence_quality,
                        decision_for_edge.confidence,
                        getattr(decision_for_edge, "primary_source_url", None),
                        data={
                            "market_id": market.id,
                            "evidence_quality_after": decision_for_edge.evidence_quality,
                            "confidence": decision_for_edge.confidence,
                            "primary_source_url": getattr(
                                decision_for_edge, "primary_source_url", None
                            ),
                            "definitive_outcome_auto_detected": auto_detected,
                            "score_cache_invalidated": True,
                        },
                    )
                if (
                    decision_for_edge.evidence_quality
                    < _min_evidence_quality_for_market(market, settings, decision_for_edge)
                ):
                    min_evidence_quality = _min_evidence_quality_for_market(
                        market,
                        settings,
                        decision_for_edge,
                    )
                    evidence_rejection_reason = (
                        "weather_evidence_quality_below_min"
                        if market_family(market) == "weather"
                        else "evidence_quality_below_min"
                    )
                    evidence_gap = max(
                        0.0,
                        float(min_evidence_quality - float(decision_for_edge.evidence_quality)),
                    )
                    queue_for_research = _should_queue_research_for_blocked_trade(
                        settings=settings,
                        decision=decision_for_edge,
                        evidence_basis=evidence_basis,
                        gate_name="evidence",
                        threshold_gap=evidence_gap,
                    )
                    final_action = "research_queued" if queue_for_research else "skip"
                    final_outcome_reason = (
                        "research_queued" if queue_for_research else evidence_rejection_reason
                    )
                    research_queue_position: int | None = None
                    if queue_for_research:
                        research_queue_position = _enqueue_research_candidate(
                            market=market,
                            decision=decision_for_edge,
                            reason=evidence_rejection_reason,
                            gate_name="evidence",
                            threshold_gap=evidence_gap,
                        )
                    if evidence_basis == "direct":
                        blocked_direct_evidence_count += 1
                    trades_skipped_no_trade += 1
                    _record_should_trade_blocked(evidence_rejection_reason)
                    _record_rejection_reason(
                        rejection_breakdown,
                        evidence_rejection_reason,
                    )
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_terminal=not queue_for_research,
                            final_action=final_action,
                            final_reason=evidence_rejection_reason,
                            evidence_quality=decision_for_edge.evidence_quality,
                            min_evidence_quality=min_evidence_quality,
                            evidence_quality_gap=evidence_gap,
                            research_queue_position=research_queue_position,
                            **audit_context,
                        ),
                    )
                    if not queue_for_research:
                        _record_terminal_outcome(
                            state_manager,
                            market.id,
                            final_outcome_reason,
                        )
                    logger.warning(
                        "%s [%s] -> %s after should_trade=True",
                        "RESEARCH" if queue_for_research else "SKIP",
                        market.id,
                        evidence_rejection_reason,
                        data={
                            "market_id": market.id,
                            "final_action": final_action,
                            "final_reason": evidence_rejection_reason,
                            "evidence_quality_gap": evidence_gap,
                            "research_queue_position": research_queue_position,
                            "score_breakdown": score_receipt_fields,
                        },
                    )
                    continue
                bucket = _price_bucket(implied_prob, settings)
                price_bucket_stats[bucket] += 1
                required_edge_threshold = _edge_threshold_for_market(
                    implied_prob,
                    settings,
                    market=market,
                )
                baseline_edge_threshold = _edge_threshold_for_market(
                    implied_prob,
                    settings,
                    decision_for_edge.edge_source,
                    market=None,
                    definitive_outcome_eligible=_is_definitive_outcome_eligible(
                        decision_for_edge,
                        settings,
                        market=market,
                    ),
                )
                edge_threshold_reduction = max(
                    0.0,
                    float(baseline_edge_threshold - required_edge_threshold),
                )
                edge_ok, edge_value, edge_reason = _passes_edge_threshold(
                    implied_prob,
                    decision_for_edge,
                    settings,
                    market=market,
                )
                audit_context["edge_market"] = edge_value
                audit_context["edge_external"] = decision_for_edge.edge_external
                audit_context["gate_edge_required_baseline"] = baseline_edge_threshold
                audit_context["gate_edge_dynamic_reduction"] = edge_threshold_reduction
                audit_context["gate_edge_dynamic_reduction_applied"] = (
                    edge_threshold_reduction > 1e-9
                )
                _def_validated = _is_definitive_validated(
                    decision_for_edge,
                    settings,
                    market=market,
                )
                if _def_validated:
                    audit_context["definitive_edge_bypass_validated"] = True
                calibration_payload = {
                    "market_id": market.id,
                    "cycle": cycle_count,
                    "edge_market": edge_value,
                    "implied_prob_market": implied_prob,
                    "confidence": decision_for_edge.confidence,
                    "evidence_quality": decision_for_edge.evidence_quality,
                    "liquidity_usdc": market.liquidity_usdc,
                    "analysis_duration_ms": round((time.monotonic() - market_start) * 1000, 2),
                    "edge_gate_pass": edge_ok,
                    "gate_edge_required": required_edge_threshold,
                    "gate_edge_actual": edge_value,
                    "gate_edge_required_baseline": baseline_edge_threshold,
                    "gate_edge_dynamic_reduction": edge_threshold_reduction,
                }
                calibration_payload.update(build_counterfactual_flags(edge_value))
                calibration_samples.append(calibration_payload)
                logger.info(
                    "Calibration sample: market=%s edge=%s edge_gate_pass=%s",
                    market.id,
                    f"{edge_value:.4f}" if edge_value is not None else "n/a",
                    edge_ok,
                    data=calibration_payload,
                )
                if not edge_ok:
                    trades_skipped_edge += 1
                    _record_should_trade_blocked("edge_gate_blocked")
                    _record_rejection_reason(rejection_breakdown, "edge_gate_blocked")
                    if edge_reason == "non_sports_needs_direct_evidence":
                        _record_rejection_reason(
                            rejection_breakdown,
                            "non_sports_needs_direct",
                        )
                    elif edge_reason == "edge_above_reasonable_max":
                        _record_rejection_reason(
                            rejection_breakdown,
                            "edge_above_reasonable_max",
                        )
                    elif "below min" in edge_reason:
                        _record_rejection_reason(
                            rejection_breakdown,
                            "edge_below_min",
                        )
                    elif edge_reason == "missing_structured_probability":
                        _record_rejection_reason(
                            rejection_breakdown,
                            "missing_structured_probability",
                        )
                    edge_shortfall = max(
                        0.0,
                        float(required_edge_threshold - float(edge_value or 0.0)),
                    )
                    queue_for_research = _should_queue_research_for_blocked_trade(
                        settings=settings,
                        decision=decision_for_edge,
                        evidence_basis=evidence_basis,
                        gate_name="edge",
                        threshold_gap=edge_shortfall,
                        edge_reason=edge_reason,
                    )
                    research_reason = (
                        edge_reason
                        if queue_for_research
                        and str(edge_reason or "")
                        in {
                            "edge_above_reasonable_max",
                            "missing_structured_probability",
                        }
                        else "edge_gate_blocked"
                    )
                    final_action = "research_queued" if queue_for_research else "skip"
                    final_outcome_reason = (
                        "research_queued" if queue_for_research else "edge_gate_blocked"
                    )
                    research_queue_position: int | None = None
                    if queue_for_research:
                        research_queue_position = _enqueue_research_candidate(
                            market=market,
                            decision=decision_for_edge,
                            reason=research_reason,
                            gate_name="edge",
                            threshold_gap=edge_shortfall,
                            edge_market=edge_value,
                            edge_required=required_edge_threshold,
                        )
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_terminal=not queue_for_research,
                            final_action=final_action,
                            final_reason=research_reason if queue_for_research else "edge_gate_blocked",
                            gate_edge_required=required_edge_threshold,
                            gate_edge_actual=edge_value,
                            gate_edge_reason=edge_reason,
                            edge_shortfall=edge_shortfall,
                            research_queue_position=research_queue_position,
                            **audit_context,
                        ),
                    )
                    if not queue_for_research:
                        _record_terminal_outcome(state_manager, market.id, final_outcome_reason)
                    question_short = market.question[:40] + "..." if len(market.question) > 40 else market.question
                    logger.warning(
                        "%s [%s] '%s' -> edge gate (%s)",
                        "RESEARCH" if queue_for_research else "SKIP",
                        market.id,
                        question_short,
                        edge_reason,
                        data={
                            "market_id": market.id,
                            "final_action": final_action,
                            "final_reason": "edge_gate_blocked",
                            "implied_prob": implied_prob,
                            "entry_price": entry_price,
                            "confidence": decision_for_edge.confidence,
                            "edge": edge_value,
                            "gate_edge_required": required_edge_threshold,
                            "gate_edge_actual": edge_value,
                            "gate_edge_reason": edge_reason,
                            "edge_shortfall": edge_shortfall,
                            "research_queue_position": research_queue_position,
                            "score_breakdown": score_receipt_fields,
                        },
                    )
                    continue
                edge_gate_passed += 1
                historical_gate_allowed_exec, historical_gate_reason_exec, historical_gate_metrics_exec = (
                    _evaluate_historical_gate(
                        market_id=market.id,
                        family_name=market_family_name,
                    )
                )
                if not historical_gate_allowed_exec and historical_gate_reason_exec:
                    trades_skipped_edge += 1
                    _record_should_trade_blocked(historical_gate_reason_exec)
                    _record_rejection_reason(
                        rejection_breakdown,
                        historical_gate_reason_exec,
                    )
                    research_queue_position: int | None = None
                    queue_for_research = bool(settings.RESEARCH_QUEUE_ENABLED)
                    if queue_for_research:
                        research_queue_position = _enqueue_research_candidate(
                            market=market,
                            decision=decision_for_edge,
                            reason=historical_gate_reason_exec,
                            gate_name="historical_performance",
                            threshold_gap=0.0,
                            edge_market=edge_value,
                        )
                    historical_learning_target = (
                        historical_gate_metrics_exec.get("what_to_learn_next")
                        if isinstance(historical_gate_metrics_exec, dict)
                        else None
                    ) or _research_learning_target(
                        gate_name="historical_performance",
                        reason=historical_gate_reason_exec,
                        market=market,
                        decision=decision_for_edge,
                    )
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_terminal=not queue_for_research,
                            final_action="research_queued" if queue_for_research else "skip",
                            final_reason=historical_gate_reason_exec,
                            research_queue_position=research_queue_position,
                            research_gate_name="historical_performance",
                            historical_performance_research_only=queue_for_research,
                            historical_prefix_action=(
                                "research_queued" if queue_for_research else "skip"
                            ),
                            learning_hold_reason=historical_gate_reason_exec,
                            what_to_learn_next=historical_learning_target,
                            **historical_gate_metrics_exec,
                            **audit_context,
                        ),
                    )
                    if not queue_for_research:
                        _record_terminal_outcome(
                            state_manager,
                            market.id,
                            historical_gate_reason_exec,
                        )
                    question_short = (
                        market.question[:40] + "..."
                        if len(market.question) > 40
                        else market.question
                    )
                    logger.warning(
                        "%s [%s] '%s' -> historical calibration gate (%s)",
                        "RESEARCH" if queue_for_research else "SKIP",
                        market.id,
                        question_short,
                        historical_gate_reason_exec,
                        data={
                            "market_id": market.id,
                            "final_action": "research_queued" if queue_for_research else "skip",
                            "final_reason": historical_gate_reason_exec,
                            "research_queue_position": research_queue_position,
                            **historical_gate_metrics_exec,
                            "score_breakdown": score_receipt_fields,
                        },
                    )
                    continue

                if _is_uniform_implied_probability(implied_prob, market.outcomes):
                    uniform_implied = 1.0 / len(market.outcomes)
                    trades_skipped_edge += 1
                    _record_should_trade_blocked("uniform_implied_probability")
                    _record_rejection_reason(
                        rejection_breakdown,
                        "uniform_implied_probability",
                    )
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="uniform_implied_probability",
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(
                        state_manager,
                        market.id,
                        "uniform_implied_probability",
                    )
                    question_short = market.question[:40] + "..." if len(market.question) > 40 else market.question
                    logger.warning(
                        "SKIP [%s] '%s' -> uniform implied probability detected (%d outcomes, implied=%.3f)",
                        market.id,
                        question_short,
                        len(market.outcomes),
                        implied_prob,
                        data={
                            "market_id": market.id,
                            "final_reason": "uniform_implied_probability",
                            "implied_prob": implied_prob,
                            "uniform_implied": uniform_implied,
                            "outcomes": [outcome.name for outcome in market.outcomes],
                            "score_breakdown": score_receipt_fields,
                        },
                    )
                    continue

                kelly_raw_value: float | None = None
                kelly_fraction_value: float | None = None
                posterior_for_kelly: float | None = None
                min_edge_for_kelly: float | None = None
                kelly_bankroll_eligible = (
                    cycle_bankroll is None or cycle_bankroll >= settings.KELLY_MIN_BANKROLL_USDC
                )
                kelly_path_active = (
                    settings.KELLY_SIZING_ENABLED
                    and implied_prob is not None
                    and kelly_bankroll_eligible
                )
                sizing_mode = _sizing_mode_label(kelly_path_active)
                if (
                    settings.KELLY_SIZING_ENABLED
                    and implied_prob is not None
                    and not kelly_bankroll_eligible
                ):
                    logger.debug(
                        "Kelly sizing disabled for cycle due to bankroll guard: market=%s bankroll=%.2f min=%.2f",
                        market.id,
                        cycle_bankroll,
                        settings.KELLY_MIN_BANKROLL_USDC,
                        data={
                            "market_id": market.id,
                            "cycle_bankroll": cycle_bankroll,
                            "kelly_min_bankroll_usdc": settings.KELLY_MIN_BANKROLL_USDC,
                        },
                    )
                if kelly_path_active:
                    posterior_for_kelly = (
                        bayesian_posterior_applied
                        if bayesian_posterior_applied is not None
                        else effective_confidence
                    )
                    kelly_raw_value = kelly_fraction(
                        posterior=posterior_for_kelly,
                        market_price=implied_prob,
                    )
                    kelly_fraction_value = _kelly_fraction_for_market_horizon(market, settings)

                score_gate_score_source = "runtime_recomputed"
                short_prefix_score_penalty = float(
                    candidate.get("short_prefix_score_penalty", 0.0) or 0.0
                )
                suppress_hallucinated_edge_penalty = (
                    _should_suppress_hallucinated_edge_penalty(
                        decision=decision_for_edge,
                        evidence_basis=evidence_basis,
                        settings=settings,
                        market=market,
                    )
                )
                exec_pfx_stats = _get_prefix_pnl(market.id or "")
                exec_pfx_n = int(exec_pfx_stats.get("n", 0))
                exec_pfx_pnl = float(exec_pfx_stats.get("total_pnl", 0.0))
                exec_pfx_shrunk = (
                    bayesian_shrunk_pnl(exec_pfx_pnl, exec_pfx_n)
                    if exec_pfx_n > 0
                    else None
                )
                score_result = compute_final_score(
                    market=market,
                    decision=decision_for_edge,
                    implied_prob_market=implied_prob,
                    bayesian_posterior=bayesian_posterior_applied,
                    lmsr_price=lmsr_execution_price,
                    inefficiency_signal=ineff_signal,
                    kelly_raw=kelly_raw_value,
                    **_score_kwargs(
                        settings=settings,
                        repeated_analysis_count=(state.analysis_count if state is not None else 0),
                        non_actionable_streak=(
                            state.non_actionable_streak if state is not None else 0
                        ),
                        is_weather_market=(market_family(market) == "weather"),
                        evidence_basis_class=evidence_basis,
                        edge_source=decision_for_edge.edge_source or "",
                        market_family=market_family_name,
                        short_prefix_penalty=short_prefix_score_penalty,
                        suppress_hallucinated_edge_penalty=suppress_hallucinated_edge_penalty,
                        definitive_outcome_eligible=_is_definitive_outcome_eligible(
                            decision_for_edge,
                            settings,
                            market=market,
                        ),
                        historical_family_pnl_total=float(
                            candidate.get("historical_family_pnl_total", 0.0) or 0.0
                        ),
                        historical_family_sample_size=int(
                            candidate.get("historical_family_sample_size", 0) or 0
                        ),
                        historical_prefix_pnl_per_trade=exec_pfx_shrunk,
                        historical_prefix_sample_size=exec_pfx_n,
                    ),
                )
                runtime_score_evaluation_count += 1
                score_gate_score_source_counts[score_gate_score_source] = (
                    score_gate_score_source_counts.get(score_gate_score_source, 0) + 1
                )
                audit_context["fallback_high_confidence_penalty_applied"] = bool(
                    float(getattr(score_result, "fallback_high_confidence_penalty", 0.0) or 0.0)
                    > 0.0
                )
                audit_context["historical_family_bonus_applied"] = bool(
                    float(getattr(score_result, "historical_family_bonus", 0.0) or 0.0) > 0.0
                )
                score_mode = settings.SCORE_GATE_MODE
                score_threshold_effective = _effective_score_gate_threshold(
                    settings=settings,
                    market=market,
                    evidence_basis_class=evidence_basis,
                    evidence_quality=decision_for_edge.evidence_quality,
                )
                score_gate_critical_reasons = _score_gate_critical_rejection_reasons(
                    rejection_reasons=score_result.rejection_reasons,
                    evidence_basis_class=evidence_basis,
                    edge_source=decision_for_edge.edge_source,
                    definitive_outcome_eligible=_is_definitive_outcome_eligible(
                        decision_for_edge,
                        settings,
                        market=market,
                    ),
                )
                score_receipt_fields = _apply_runtime_score_receipt(
                    audit_context,
                    score_result=score_result,
                    score_threshold_effective=score_threshold_effective,
                    pre_execution_final_score=pre_execution_final_score,
                    score_gate_score_source=score_gate_score_source,
                    score_gate_critical_reasons=score_gate_critical_reasons,
                )
                analysis_result["execution_final_score"] = score_result.final_score
                analysis_result["execution_rejection_reasons"] = list(
                    score_result.rejection_reasons
                )
                analysis_result["execution_score_result"] = score_result
                analysis_result["score_gate_score_source"] = score_gate_score_source
                runtime_delta = audit_context.get("pre_vs_runtime_score_delta")
                if runtime_delta is not None:
                    pre_vs_runtime_score_deltas.append(float(runtime_delta))
                score_payload: dict[str, Any] | None = None
                if score_mode != "off":
                    score_payload = {
                        "market_id": market.id,
                        "score_mode": score_mode,
                        "score_threshold": score_threshold_effective,
                        "score_threshold_default": settings.SCORE_GATE_THRESHOLD,
                        "score_threshold_weather_direct": settings.SCORE_GATE_THRESHOLD_WEATHER_DIRECT,
                        "score_threshold_direct_high_quality": settings.SCORE_GATE_THRESHOLD_DIRECT_HIGH_QUALITY,
                        "final_score": score_result.final_score,
                        "edge_market": score_result.edge_market,
                        "edge_external": score_result.edge_external,
                        "evidence_quality": score_result.evidence_quality,
                        "evidence_component": score_result.evidence_component,
                        "bayesian_component": score_result.bayesian_component,
                        "inefficiency_component": score_result.inefficiency_component,
                        "kelly_component": score_result.kelly_component,
                        "score_volume_amplifier_discount": (
                            score_result.volume_amplifier_discount
                        ),
                        "confidence_alignment_bonus": score_result.confidence_alignment_bonus,
                        "evidence_basis_bonus": score_result.evidence_basis_bonus,
                        "observed_data_bonus": score_result.observed_data_bonus,
                        "low_information_penalty": score_result.low_information_penalty,
                        "no_external_odds_penalty": score_result.no_external_odds_penalty,
                        "repeated_analysis_penalty": score_result.repeated_analysis_penalty,
                        "mention_market_penalty": score_result.mention_market_penalty,
                        "confidence_calibration_penalty": score_result.confidence_calibration_penalty,
                        "fallback_edge_penalty": score_result.fallback_edge_penalty,
                        "fallback_high_confidence_penalty": (
                            score_result.fallback_high_confidence_penalty
                        ),
                        "extreme_market_edge_penalty": (
                            score_result.extreme_market_edge_penalty
                        ),
                        "proxy_evidence_penalty": score_result.proxy_evidence_penalty,
                        "liquidity_penalty": score_result.liquidity_penalty,
                        "staleness_penalty": score_result.staleness_penalty,
                        "extreme_confidence_penalty": score_result.extreme_confidence_penalty,
                        "hallucinated_edge_penalty": score_result.hallucinated_edge_penalty,
                        "hallucinated_edge_penalty_suppressed": (
                            score_result.hallucinated_edge_penalty_suppressed
                        ),
                        "high_edge_calibration_penalty": (
                            score_result.high_edge_calibration_penalty
                        ),
                        "extreme_edge_learning_queue": score_result.extreme_edge_learning_queue,
                        "late_stage_overconfidence_penalty": score_result.late_stage_overconfidence_penalty,
                        "rejection_reasons": list(score_result.rejection_reasons),
                        "score_gate_critical_reasons": list(score_gate_critical_reasons),
                        "generic_bin_penalty": score_result.generic_bin_penalty,
                        "numeric_strike_bin_penalty": score_result.numeric_strike_bin_penalty,
                        "short_prefix_penalty": score_result.short_prefix_penalty,
                        "historical_family_bonus": score_result.historical_family_bonus,
                        "ambiguous_resolution_penalty": score_result.ambiguous_resolution_penalty,
                        "bayesian_posterior": bayesian_posterior_applied,
                        "lmsr_price": lmsr_execution_price,
                        "inefficiency_signal": ineff_signal,
                        "kelly_raw": kelly_raw_value,
                        "bayesian_posterior_raw": bayesian_posterior_raw,
                        "bayesian_posterior_applied": bayesian_posterior_applied,
                        "bayesian_applied": bayesian_posterior_applied is not None,
                        "bayesian_update_count": bayesian_update_count,
                        "bayesian_min_updates": settings.BAYESIAN_MIN_UPDATES_FOR_TRADE,
                        "likelihood_ratio": likelihood_ratio,
                        "market_family": market_family_name,
                        "evidence_basis_class": evidence_basis,
                        "pre_execution_final_score": pre_execution_final_score,
                        "execution_score_final": score_result.final_score,
                        "pre_vs_runtime_score_delta": audit_context.get(
                            "pre_vs_runtime_score_delta"
                        ),
                        "score_gate_score_source": score_gate_score_source,
                    }
                    pre_analysis_score_value = candidate.get("pre_analysis_score")
                    if pre_analysis_score_value is not None:
                        score_payload["pre_analysis_score"] = pre_analysis_score_value
                    pre_analysis_breakdown = candidate.get("pre_analysis_breakdown")
                    if isinstance(pre_analysis_breakdown, dict):
                        score_payload["pre_analysis_breakdown"] = pre_analysis_breakdown
                    logger.info(
                        "Score gate evaluation: market=%s final_score=%.4f threshold=%.4f mode=%s",
                        market.id,
                        score_result.final_score,
                        score_threshold_effective,
                        score_mode,
                        data=score_payload,
                    )
                    for rejection_reason in score_result.rejection_reasons:
                        score_rejection_reason_breakdown[rejection_reason] = (
                            score_rejection_reason_breakdown.get(rejection_reason, 0) + 1
                        )
                    if score_mode == "shadow":
                        logger.debug(
                            "Score gate shadow: market=%s final_score=%.4f threshold=%.4f",
                            market.id,
                            score_result.final_score,
                            score_threshold_effective,
                            data=score_payload,
                        )
                    elif (
                        score_result.final_score < score_threshold_effective
                        or score_gate_critical_reasons
                    ):
                        audit_context["score_gate_passed"] = False
                        score_gate_blocked += 1
                        trades_skipped_edge += 1
                        score_gap = max(
                            0.0,
                            float(score_threshold_effective - score_result.final_score),
                        )
                        score_gate_block_reason = (
                            "score_gate_critical_rejection"
                            if score_gate_critical_reasons
                            else "score_gate_blocked"
                        )
                        score_near_misses.append(
                            {
                                "market_id": market.id,
                                "final_score": float(score_result.final_score),
                                "score_threshold": float(score_threshold_effective),
                                "score_gap": score_gap,
                                "rejection_reasons": list(score_result.rejection_reasons),
                                "critical_rejection_reasons": list(score_gate_critical_reasons),
                                "score_gate_score_source": score_gate_score_source,
                            }
                        )
                        _record_should_trade_blocked(score_gate_block_reason)
                        _record_rejection_reason(rejection_breakdown, score_gate_block_reason)
                        for rejection_reason in score_result.rejection_reasons:
                            _record_rejection_reason(
                                rejection_breakdown,
                                str(rejection_reason),
                            )
                        score_rejection_reasons = {
                            str(reason) for reason in score_result.rejection_reasons
                        }
                        research_gate_name: str | None = None
                        if "extreme_edge_learning_queue" in score_rejection_reasons:
                            research_gate_name = "extreme_edge_learning_queue"
                        elif "extreme_market_edge_penalty" in score_rejection_reasons:
                            research_gate_name = "extreme_market_edge"
                        elif "hallucinated_edge" in score_rejection_reasons:
                            research_gate_name = "hallucinated_edge"
                        queue_for_research = False
                        if research_gate_name is not None:
                            queue_for_research = _should_queue_research_for_blocked_trade(
                                settings=settings,
                                decision=decision_for_edge,
                                evidence_basis=evidence_basis,
                                gate_name=research_gate_name,
                                threshold_gap=score_gap,
                            )
                        research_queue_position: int | None = None
                        if queue_for_research:
                            score_gate_learning_target = _research_learning_target(
                                gate_name=research_gate_name or "score_gate",
                                reason=research_gate_name or score_gate_block_reason,
                                market=market,
                                decision=decision_for_edge,
                            )
                            research_queue_position = _enqueue_research_candidate(
                                market=market,
                                decision=decision_for_edge,
                                reason=research_gate_name or "score_gate_blocked",
                                gate_name=research_gate_name or "score_gate",
                                threshold_gap=score_gap,
                                edge_market=score_result.edge_market,
                            )
                        else:
                            score_gate_learning_target = None
                        final_action = "research_queued" if queue_for_research else "skip"
                        final_reason = (
                            research_gate_name
                            if queue_for_research and research_gate_name
                            else score_gate_block_reason
                        )
                        log_trade_decision(
                            market_id=market.id,
                            question=market.question,
                            decision=decision_for_edge.model_dump(),
                            execution_audit=_build_execution_audit(
                                decision_terminal=not queue_for_research,
                                final_action=final_action,
                                final_reason=final_reason,
                                score_threshold=score_threshold_effective,
                                score_gap=score_gap,
                                research_queue_position=research_queue_position,
                                score_gate_research_reason=research_gate_name,
                                score_gate_block_reason=score_gate_block_reason,
                                historical_prefix_action=(
                                    "research_queued" if queue_for_research else "not_applicable"
                                ),
                                learning_hold_reason=(
                                    final_reason if queue_for_research else None
                                ),
                                what_to_learn_next=score_gate_learning_target,
                                **audit_context,
                            ),
                        )
                        if not queue_for_research:
                            _record_terminal_outcome(state_manager, market.id, "score_gate_blocked")
                        question_short = market.question[:40] + "..." if len(market.question) > 40 else market.question
                        logger.warning(
                            "%s [%s] '%s' -> score gate (%.4f < %.4f)",
                            "RESEARCH" if queue_for_research else "SKIP",
                            market.id,
                            question_short,
                            score_result.final_score,
                            score_threshold_effective,
                            data=score_payload,
                        )
                        continue
                    audit_context["score_gate_passed"] = True
                    score_gate_passed += 1
                else:
                    audit_context["score_gate_passed"] = True
                    score_gate_passed += 1

                edge_scaling_bet_pct = _adjust_bet_size_for_edge(
                    decision_for_edge,
                    implied_prob,
                    edge_value,
                    settings,
                    market=market,
                )
                if kelly_path_active:
                    if posterior_for_kelly is None:
                        posterior_for_kelly = (
                            bayesian_posterior_applied
                            if bayesian_posterior_applied is not None
                            else effective_confidence
                        )
                    if kelly_fraction_value is None:
                        kelly_fraction_value = _kelly_fraction_for_market_horizon(market, settings)
                    min_edge_for_kelly = _edge_threshold_for_market(
                        implied_prob,
                        settings,
                        market=market,
                    )
                    adjusted_bet_pct = kelly_bet_pct(
                        posterior=posterior_for_kelly,
                        market_price=implied_prob,
                        fraction=kelly_fraction_value,
                        min_edge=min_edge_for_kelly,
                        edge=edge_value,
                        dynamic_enabled=settings.KELLY_DYNAMIC_ENABLED,
                    )
                else:
                    adjusted_bet_pct = edge_scaling_bet_pct
                if (
                    kelly_path_active
                    and kelly_raw_value is not None
                    and float(kelly_raw_value) > 0.0
                ):
                    effective_kelly_fraction = max(
                        0.0,
                        float(adjusted_bet_pct) / float(kelly_raw_value),
                    )
                    audit_context["kelly_effective_fraction"] = effective_kelly_fraction
                    audit_context["kelly_dynamic_fraction_gt_0_50"] = (
                        effective_kelly_fraction > 0.50
                    )
                kelly_posterior_edge_below_min = False
                if kelly_path_active and adjusted_bet_pct <= 0:
                    kelly_posterior_edge_below_min = True
                    (
                        recovered_bet_amount,
                        recovered_bet_pct,
                        _recovered_min_bet_floor_applied,
                        _recovered_kelly_sub_floor_skipped,
                        recovered_policy,
                    ) = _resolve_min_bet_floor(
                        bet_amount=0.0,
                        min_bet_usdc=settings.MIN_BET_USDC,
                        max_bet_usdc=settings.MAX_BET_USDC,
                        kelly_path_active=True,
                        min_bet_policy=settings.KELLY_MIN_BET_POLICY,
                        edge_scaling_bet_pct=edge_scaling_bet_pct,
                    )
                    recovered_via_fallback_edge = (
                        recovered_policy == _KELLY_MIN_BET_POLICY_FALLBACK_EDGE
                        and recovered_bet_amount >= settings.MIN_BET_USDC
                        and recovered_bet_pct > 0.0
                    )
                    if recovered_via_fallback_edge:
                        adjusted_bet_pct = recovered_bet_pct
                        sizing_mode = "kelly_fallback_edge"
                        audit_context["kelly_posterior_edge_below_min"] = True
                if adjusted_bet_pct <= 0:
                    trades_skipped_edge += 1
                    _record_should_trade_blocked("zero_bet_after_sizing")
                    _record_rejection_reason(rejection_breakdown, "zero_bet_after_sizing")
                    if decision_for_edge.bet_size_pct <= 0:
                        sizing_zero_reason = "model_bet_size_zero"
                    elif kelly_path_active or kelly_posterior_edge_below_min:
                        sizing_zero_reason = "kelly_posterior_edge_below_min"
                    else:
                        sizing_zero_reason = "edge_scaling_zero"
                    question_short = market.question[:40] + "..." if len(market.question) > 40 else market.question
                    skip_reason = _zero_bet_skip_message(sizing_mode)
                    logger.warning(
                        "SKIP [%s] '%s' -> %s",
                        market.id,
                        question_short,
                        skip_reason,
                        data={
                            "market_id": market.id,
                            "final_reason": "zero_bet_after_sizing",
                            "sizing_mode": sizing_mode,
                            "implied_prob": implied_prob,
                            "entry_price": entry_price,
                            "confidence": decision_for_edge.confidence,
                            "edge": edge_value,
                            "kelly_raw": kelly_raw_value,
                            "kelly_fraction_value": kelly_fraction_value,
                            "posterior_for_kelly": posterior_for_kelly,
                            "min_edge_for_kelly": min_edge_for_kelly,
                            "sizing_zero_reason": sizing_zero_reason,
                            "score_breakdown": score_receipt_fields,
                        },
                    )
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_copy(
                            update={"bet_size_pct": adjusted_bet_pct}
                        ).model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_phase="post_sizing",
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="zero_bet_after_sizing",
                            sizing_mode=sizing_mode,
                            sizing_zero_reason=sizing_zero_reason,
                            kelly_posterior_edge_below_min=kelly_posterior_edge_below_min,
                            adjusted_bet_pct=adjusted_bet_pct,
                            bet_amount_usdc=0.0,
                            kelly_raw=kelly_raw_value,
                            kelly_fraction_value=kelly_fraction_value,
                            posterior_for_kelly=posterior_for_kelly,
                            bayesian_posterior_raw=bayesian_posterior_raw,
                            bayesian_posterior_applied=bayesian_posterior_applied,
                            bayesian_applied=bayesian_posterior_applied is not None,
                            bayesian_update_count=bayesian_update_count,
                            bayesian_min_updates=settings.BAYESIAN_MIN_UPDATES_FOR_TRADE,
                            likelihood_ratio=likelihood_ratio,
                            implied_prob_market=implied_prob,
                            min_edge_for_kelly=min_edge_for_kelly,
                            lmsr_execution_price=lmsr_execution_price,
                            lmsr_inefficiency_signal=ineff_signal,
                            lmsr_liquidity_param_b=settings.LMSR_LIQUIDITY_PARAM_B,
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(state_manager, market.id, "zero_bet_after_sizing")
                    continue

                proposed_bet_amount = _calculate_bet(settings.MAX_BET_USDC, adjusted_bet_pct)
                if settings.LMSR_ENABLED:
                    lmsr_execution_price = _compute_lmsr_execution_price_for_outcome(
                        market=market,
                        decision_outcome=decision.outcome,
                        amount_usdc=proposed_bet_amount,
                        settings=settings,
                    )
                    if lmsr_execution_price is not None:
                        posterior_for_signal = (
                            bayesian_posterior_applied
                            if bayesian_posterior_applied is not None
                            else effective_confidence
                        )
                        try:
                            ineff_signal = lmsr_inefficiency_signal(
                                posterior_for_signal,
                                lmsr_execution_price,
                            )
                        except ValueError:
                            ineff_signal = None
                    if (
                        ineff_signal is not None
                        and abs(ineff_signal) < settings.LMSR_MIN_INEFFICIENCY
                    ):
                        trades_skipped_edge += 1
                        _record_should_trade_blocked("lmsr_gate_blocked")
                        _record_rejection_reason(rejection_breakdown, "lmsr_gate_blocked")
                        question_short = market.question[:40] + "..." if len(market.question) > 40 else market.question
                        logger.warning(
                            "SKIP [%s] '%s' -> LMSR inefficiency too small (|%.4f| < %.4f)",
                            market.id,
                            question_short,
                            ineff_signal,
                            settings.LMSR_MIN_INEFFICIENCY,
                            data={
                                "market_id": market.id,
                                "final_reason": "lmsr_gate_blocked",
                                "inefficiency_signal": ineff_signal,
                                "lmsr_execution_price": lmsr_execution_price,
                                "proposed_bet_amount_usdc": proposed_bet_amount,
                                "lmsr_liquidity_param_b": settings.LMSR_LIQUIDITY_PARAM_B,
                                "bayesian_posterior_raw": bayesian_posterior_raw,
                                "bayesian_posterior_applied": bayesian_posterior_applied,
                                "bayesian_update_count": bayesian_update_count,
                                "bayesian_min_updates": settings.BAYESIAN_MIN_UPDATES_FOR_TRADE,
                                "likelihood_ratio": likelihood_ratio,
                                "score_breakdown": score_receipt_fields,
                            },
                        )
                        log_trade_decision(
                            market_id=market.id,
                            question=market.question,
                            decision=decision_for_edge.model_copy(
                                update={"bet_size_pct": adjusted_bet_pct}
                            ).model_dump(),
                            execution_audit=_build_execution_audit(
                                decision_phase="post_lmsr_gate",
                                decision_terminal=True,
                                final_action="skip",
                                final_reason="lmsr_gate_blocked",
                                lmsr_gate_decision="blocked",
                                lmsr_execution_price=lmsr_execution_price,
                                lmsr_inefficiency_signal=ineff_signal,
                                lmsr_min_inefficiency=settings.LMSR_MIN_INEFFICIENCY,
                                proposed_bet_amount_usdc=proposed_bet_amount,
                                lmsr_liquidity_param_b=settings.LMSR_LIQUIDITY_PARAM_B,
                                bayesian_posterior_raw=bayesian_posterior_raw,
                                bayesian_posterior_applied=bayesian_posterior_applied,
                                bayesian_applied=bayesian_posterior_applied is not None,
                                bayesian_update_count=bayesian_update_count,
                                bayesian_min_updates=settings.BAYESIAN_MIN_UPDATES_FOR_TRADE,
                                likelihood_ratio=likelihood_ratio,
                                **audit_context,
                            ),
                        )
                        _record_terminal_outcome(state_manager, market.id, "lmsr_gate_blocked")
                        continue

                opportunity_role = "standard"
                opportunity_rank = execution_candidates + 1
                satellite_cap_pct: float | None = None
                if getattr(settings, "DAILY_EXPECTANCY_ENABLED", True):
                    opportunity_role, satellite_cap_pct = _daily_expectancy_role(
                        settings=settings,
                        opportunity_rank=opportunity_rank,
                    )
                    if (
                        opportunity_role == "satellite"
                        and satellite_cap_pct is not None
                        and adjusted_bet_pct > satellite_cap_pct
                    ):
                        audit_context["satellite_original_bet_pct"] = adjusted_bet_pct
                        adjusted_bet_pct = satellite_cap_pct
                        audit_context["satellite_size_cap_applied"] = True
                    audit_context["daily_expectancy_enabled"] = True
                    audit_context["daily_expectancy_rank"] = opportunity_rank
                else:
                    audit_context["daily_expectancy_enabled"] = False
                audit_context["opportunity_role"] = opportunity_role
                audit_context["edge_band"] = _edge_band_label(edge_value)
                if satellite_cap_pct is not None:
                    audit_context["satellite_max_bet_pct"] = satellite_cap_pct

                execution_candidates += 1

                logger.debug(
                    "Edge passed: market=%s implied=%.3f edge=%.3f entry=%.3f bet_pct=%.3f",
                    market.id,
                    implied_prob if implied_prob is not None else 0.0,
                    edge_value if edge_value is not None else 0.0,
                    entry_price if entry_price is not None else 0.0,
                    adjusted_bet_pct,
                    data={
                        "market_id": market.id,
                        "implied_prob": implied_prob,
                        "edge": edge_value,
                        "entry_price": entry_price,
                        "adjusted_bet_size_pct": adjusted_bet_pct,
                    },
                )

                try:
                    existing_position = state_manager.get_position(market.id)
                except Exception as exc:
                    logger.warning(
                        "Position lookup failed for market %s: %s",
                        market.id,
                        exc,
                        data={"market_id": market.id, "error": str(exc)},
                    )
                    existing_position = None
                try:
                    last_entry_price = state_manager.get_last_trade_entry_price(market.id)
                except Exception as exc:
                    logger.debug(
                        "Last entry lookup failed for market %s: %s",
                        market.id,
                        exc,
                        data={"market_id": market.id, "error": str(exc)},
                    )
                    last_entry_price = None

                should_add, bet_pct, position_reason = _should_adjust_position(
                    decision_for_edge.model_copy(update={"bet_size_pct": adjusted_bet_pct}),
                    market,
                    existing_position,
                    state,
                    settings,
                    cycle_bankroll=cycle_bankroll,
                    current_entry_price=entry_price,
                    last_entry_price=last_entry_price,
                )
                if not should_add:
                    _record_should_trade_blocked("position_adjustment_blocked")
                    _record_rejection_reason(rejection_breakdown, "position_adjustment_blocked")
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_copy(
                            update={"bet_size_pct": adjusted_bet_pct}
                        ).model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_phase="post_position_gate",
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="position_adjustment_blocked",
                            sizing_mode=sizing_mode,
                            position_decision="blocked",
                            position_decision_reason=position_reason,
                            adjusted_bet_pct=adjusted_bet_pct,
                            post_position_bet_pct=bet_pct,
                            proposed_bet_amount_usdc=proposed_bet_amount,
                            kelly_raw_bet_amount_usdc=(
                                proposed_bet_amount if sizing_mode == "kelly" else None
                            ),
                            min_bet_floor_applied=False,
                            kelly_sub_floor_skipped=False,
                            kelly_raw=kelly_raw_value,
                            kelly_fraction_value=kelly_fraction_value,
                            posterior_for_kelly=posterior_for_kelly,
                            bayesian_posterior_raw=bayesian_posterior_raw,
                            bayesian_posterior_applied=bayesian_posterior_applied,
                            bayesian_applied=bayesian_posterior_applied is not None,
                            bayesian_update_count=bayesian_update_count,
                            bayesian_min_updates=settings.BAYESIAN_MIN_UPDATES_FOR_TRADE,
                            likelihood_ratio=likelihood_ratio,
                            implied_prob_market=implied_prob,
                            min_edge_for_kelly=min_edge_for_kelly,
                            lmsr_execution_price=lmsr_execution_price,
                            lmsr_inefficiency_signal=ineff_signal,
                            lmsr_liquidity_param_b=settings.LMSR_LIQUIDITY_PARAM_B,
                            **audit_context,
                        ),
                    )
                    trades_skipped_position += 1
                    _record_terminal_outcome(
                        state_manager,
                        market.id,
                        "position_adjustment_blocked",
                    )
                    question_short = market.question[:40] + "..." if len(market.question) > 40 else market.question
                    logger.warning(
                        "SKIP [%s] '%s' -> position adjustment blocked",
                        market.id,
                        question_short,
                        data={
                            "market_id": market.id,
                            "final_reason": "position_adjustment_blocked",
                            "position_decision_reason": position_reason,
                            "confidence": decision_for_edge.confidence,
                            "avg_confidence": (
                                existing_position.avg_confidence
                                if existing_position
                                else None
                            ),
                            "position_total_usdc": (
                                existing_position.total_amount_usdc
                                if existing_position
                                else None
                            ),
                            "score_breakdown": score_receipt_fields,
                        },
                    )
                    continue

                bet_amount = _calculate_bet(settings.MAX_BET_USDC, bet_pct)
                if bet_amount <= 0:
                    _record_should_trade_blocked("bet_amount_zero")
                    _record_rejection_reason(rejection_breakdown, "bet_amount_zero")
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_copy(
                            update={"bet_size_pct": bet_pct}
                        ).model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="bet_amount_zero",
                            post_position_bet_pct=bet_pct,
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(state_manager, market.id, "bet_amount_zero")
                    logger.warning(
                        "SKIP [%s] -> bet_amount_zero after should_trade=True",
                        market.id,
                        data={
                            "market_id": market.id,
                            "final_reason": "bet_amount_zero",
                            "score_breakdown": score_receipt_fields,
                        },
                    )
                    continue

                raw_bet_amount = bet_amount
                (
                    bet_amount,
                    bet_pct,
                    min_bet_floor_applied,
                    kelly_sub_floor_skipped,
                    min_bet_policy_applied,
                ) = _resolve_min_bet_floor(
                    bet_amount=bet_amount,
                    min_bet_usdc=settings.MIN_BET_USDC,
                    max_bet_usdc=settings.MAX_BET_USDC,
                    kelly_path_active=kelly_path_active,
                    min_bet_policy=settings.KELLY_MIN_BET_POLICY,
                    edge_scaling_bet_pct=edge_scaling_bet_pct,
                )
                if kelly_sub_floor_skipped:
                    trades_skipped_edge += 1
                    trades_skipped_kelly_sub_floor += 1
                    _record_should_trade_blocked("kelly_sub_floor_skip")
                    _record_rejection_reason(rejection_breakdown, "kelly_sub_floor_skip")
                    question_short = market.question[:40] + "..." if len(market.question) > 40 else market.question
                    logger.warning(
                        "SKIP [%s] '%s' -> Kelly bet below min bet floor (raw=$%.2f < min=$%.2f)",
                        market.id,
                        question_short,
                        raw_bet_amount,
                        settings.MIN_BET_USDC,
                        data={
                            "market_id": market.id,
                            "final_reason": "kelly_sub_floor_skip",
                            "sizing_mode": sizing_mode,
                            "raw_bet_amount_usdc": raw_bet_amount,
                            "min_bet_usdc": settings.MIN_BET_USDC,
                            "kelly_sub_floor_skipped": True,
                            "min_bet_floor_applied": False,
                            "kelly_min_bet_policy": settings.KELLY_MIN_BET_POLICY,
                            "score_breakdown": score_receipt_fields,
                        },
                    )
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_copy(
                            update={"bet_size_pct": bet_pct}
                        ).model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_phase="post_min_bet_floor",
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="kelly_sub_floor_skip",
                            sizing_mode=sizing_mode,
                            position_decision="blocked",
                            position_decision_reason="kelly_sub_floor_skip",
                            post_position_bet_pct=bet_pct,
                            raw_bet_amount_usdc=raw_bet_amount,
                            bet_amount_usdc=0.0,
                            min_bet_usdc=settings.MIN_BET_USDC,
                            min_bet_floor_applied=False,
                            kelly_sub_floor_skipped=True,
                            kelly_min_bet_policy=settings.KELLY_MIN_BET_POLICY,
                            kelly_min_bet_policy_applied=min_bet_policy_applied,
                            kelly_raw=kelly_raw_value,
                            kelly_fraction_value=kelly_fraction_value,
                            posterior_for_kelly=posterior_for_kelly,
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(state_manager, market.id, "kelly_sub_floor_skip")
                    continue
                if min_bet_floor_applied:
                    logger.debug(
                        "Applied bet floor: market=%s, original=$%.2f, adjusted=$%.2f, sizing_mode=%s",
                        market.id,
                        raw_bet_amount,
                        bet_amount,
                        sizing_mode,
                        data={
                            "market_id": market.id,
                            "raw_bet_amount_usdc": raw_bet_amount,
                            "bet_amount_usdc": bet_amount,
                            "min_bet_floor_applied": True,
                            "sizing_mode": sizing_mode,
                            "kelly_min_bet_policy": settings.KELLY_MIN_BET_POLICY,
                            "kelly_min_bet_policy_applied": min_bet_policy_applied,
                        },
                    )

                ev_probability = _decision_outcome_probability(market, decision_for_edge)
                expected_value_usdc = _expected_value_usdc(
                    probability=ev_probability,
                    entry_price=entry_price,
                    amount_usdc=bet_amount,
                )
                daily_ev_before = cycle_projected_daily_ev_usdc
                daily_ev_after = (
                    daily_ev_before + float(expected_value_usdc)
                    if expected_value_usdc is not None
                    else daily_ev_before
                )
                audit_context["expected_value_probability"] = ev_probability
                audit_context["expected_value_usdc"] = expected_value_usdc
                audit_context["daily_expected_value_before_usdc"] = daily_ev_before
                audit_context["daily_expected_value_after_usdc"] = daily_ev_after
                if (
                    getattr(settings, "DAILY_EXPECTANCY_ENABLED", True)
                    and opportunity_role == "satellite"
                    and (expected_value_usdc is None or daily_ev_after <= 0.0)
                ):
                    trades_skipped_edge += 1
                    _record_should_trade_blocked("daily_expectancy_satellite_ev_blocked")
                    _record_rejection_reason(
                        rejection_breakdown,
                        "daily_expectancy_satellite_ev_blocked",
                    )
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_copy(
                            update={"bet_size_pct": bet_pct}
                        ).model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_phase="post_daily_expectancy",
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="daily_expectancy_satellite_ev_blocked",
                            bet_amount_usdc=bet_amount,
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(
                        state_manager,
                        market.id,
                        "daily_expectancy_satellite_ev_blocked",
                    )
                    continue
                if (
                    getattr(settings, "DAILY_EXPECTANCY_ENABLED", True)
                    and opportunity_role == "satellite"
                    and satellite_cap_pct is not None
                    and bet_pct > satellite_cap_pct + 1e-9
                ):
                    trades_skipped_edge += 1
                    _record_should_trade_blocked("daily_expectancy_satellite_cap_blocked")
                    _record_rejection_reason(
                        rejection_breakdown,
                        "daily_expectancy_satellite_cap_blocked",
                    )
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_copy(
                            update={"bet_size_pct": bet_pct}
                        ).model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_phase="post_daily_expectancy",
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="daily_expectancy_satellite_cap_blocked",
                            bet_amount_usdc=bet_amount,
                            min_bet_floor_applied=min_bet_floor_applied,
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(
                        state_manager,
                        market.id,
                        "daily_expectancy_satellite_cap_blocked",
                    )
                    continue
                cycle_projected_daily_ev_usdc = daily_ev_after
                if opportunity_role == "primary_target":
                    cycle_primary_targets_selected += 1
                elif opportunity_role == "satellite":
                    cycle_satellites_selected += 1

                # Skip order placement if in analysis-only mode (insufficient balance)
                if analysis_only_mode:
                    question_short = market.question[:50] + "..." if len(market.question) > 50 else market.question
                    logger.info(
                        "ANALYSIS_ONLY: [%s] '%s' -> %s @ $%.2f (conf=%.2f) - skipping order, balance insufficient",
                        market.id,
                        question_short,
                        decision.outcome,
                        bet_amount,
                        decision_for_edge.confidence,
                        data={
                            "market_id": market.id,
                            "raw_bet_amount_usdc": raw_bet_amount,
                            "bet_amount_usdc": bet_amount,
                            "min_bet_floor_applied": min_bet_floor_applied,
                            "kelly_sub_floor_skipped": kelly_sub_floor_skipped,
                            "kelly_effective_fraction": audit_context.get(
                                "kelly_effective_fraction"
                            ),
                            "kelly_dynamic_fraction_gt_0_50": audit_context.get(
                                "kelly_dynamic_fraction_gt_0_50"
                            ),
                        },
                    )
                    trades_skipped_balance += 1
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_copy(
                            update={"bet_size_pct": bet_pct}
                        ).model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_phase="analysis_only_balance_skip",
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="analysis_only_insufficient_balance",
                            bet_amount_usdc=bet_amount,
                            raw_bet_amount_usdc=raw_bet_amount,
                            min_bet_floor_applied=min_bet_floor_applied,
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(
                        state_manager,
                        market.id,
                        "analysis_only_insufficient_balance",
                    )
                    continue

                if settings.DRY_RUN:
                    question_short = market.question[:50] + "..." if len(market.question) > 50 else market.question
                    logger.info(
                        "DRY_RUN: [%s] '%s' -> %s @ $%.2f (conf=%.2f)",
                        market.id,
                        question_short,
                        decision.outcome,
                        bet_amount,
                        decision_for_edge.confidence,
                        data={
                            "market_id": market.id,
                            "question": market.question,
                            "outcome": decision.outcome,
                            "raw_bet_amount_usdc": raw_bet_amount,
                            "amount_usdc": bet_amount,
                            "confidence": decision_for_edge.confidence,
                            "reasoning": decision.reasoning,
                            "min_bet_floor_applied": min_bet_floor_applied,
                            "kelly_sub_floor_skipped": kelly_sub_floor_skipped,
                        },
                    )
                    _record_runtime_score_order_attempt_if_below(audit_context)
                    trades_attempted += 1
                    _register_order_attempt(
                        event_ticker_prefix,
                        market.id,
                        decision_for_edge.outcome,
                    )
                    family_stats = execution_family_stats.setdefault(
                        market_family_name,
                        {"order_attempts": 0.0, "orders_filled": 0.0, "orders_canceled_unfilled": 0.0, "usd_deployed": 0.0},
                    )
                    family_stats["order_attempts"] += 1
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_copy(
                            update={"bet_size_pct": bet_pct}
                        ).model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_phase="dry_run_order",
                            decision_terminal=True,
                            final_action="order_attempt",
                            final_reason="dry_run",
                            bet_amount_usdc=bet_amount,
                            raw_bet_amount_usdc=raw_bet_amount,
                            min_bet_floor_applied=min_bet_floor_applied,
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(state_manager, market.id, "dry_run")
                    continue

                question_short = market.question[:50] + "..." if len(market.question) > 50 else market.question
                active_market = market
                audit_context["stale_refresh_lenient_fallback_used"] = False
                market_data_age_seconds: float | None = None
                if isinstance(market_snapshot_monotonic, (int, float)):
                    market_data_age_seconds = max(
                        0.0,
                        time.monotonic() - float(market_snapshot_monotonic),
                    )
                force_refresh_for_staleness = _requires_market_refresh(
                    pre_order_market_refresh=False,
                    market_data_age_seconds=market_data_age_seconds,
                    max_market_data_age_seconds=settings.MAX_MARKET_DATA_AGE_SECONDS,
                )
                if _requires_market_refresh(
                    pre_order_market_refresh=settings.PRE_ORDER_MARKET_REFRESH,
                    market_data_age_seconds=market_data_age_seconds,
                    max_market_data_age_seconds=settings.MAX_MARKET_DATA_AGE_SECONDS,
                ):
                    refresh_exception: Exception | None = None
                    refresh_attempts = 0
                    for refresh_attempt in range(2):
                        refresh_attempts = refresh_attempt + 1
                        try:
                            refreshed = kalshi_client.get_market(market.id)
                            if refreshed.outcomes:
                                active_market = refreshed
                            refresh_exception = None
                            break
                        except Exception as exc:
                            refresh_exception = exc
                            if refresh_attempt == 0:
                                time.sleep(_STALE_REFRESH_RETRY_DELAY_SECONDS)
                    if refresh_exception is None:
                        logger.debug(
                            "Using refreshed market snapshot for execution: market=%s",
                            market.id,
                            data={
                                "market_id": market.id,
                                "market_data_age_seconds": market_data_age_seconds,
                                "force_refresh_for_staleness": force_refresh_for_staleness,
                                "refresh_attempts": refresh_attempts,
                            },
                        )
                    elif force_refresh_for_staleness:
                        lenient_stale_refresh_allowed = _can_use_lenient_stale_refresh_fallback(
                            evidence_basis_class=evidence_basis,
                            pre_execution_final_score=pre_execution_final_score,
                            market_data_age_seconds=market_data_age_seconds,
                            settings=settings,
                        )
                        if lenient_stale_refresh_allowed:
                            audit_context["stale_refresh_lenient_fallback_used"] = True
                            logger.warning(
                                "Proceeding with stale market snapshot after refresh failures: market=%s",
                                market.id,
                                data={
                                    "market_id": market.id,
                                    "error": str(refresh_exception),
                                    "market_data_age_seconds": market_data_age_seconds,
                                    "max_market_data_age_seconds": settings.MAX_MARKET_DATA_AGE_SECONDS,
                                    "lenient_max_age_seconds": (
                                        settings.MAX_MARKET_DATA_AGE_SECONDS
                                        * _STALE_REFRESH_LENIENT_AGE_MULTIPLIER
                                    ),
                                    "refresh_attempts": refresh_attempts,
                                    "evidence_basis_class": evidence_basis,
                                    "pre_execution_final_score": pre_execution_final_score,
                                },
                            )
                        else:
                            logger.warning(
                                "SKIP [%s] '%s' -> stale market data and refresh failed",
                                market.id,
                                question_short,
                                data={
                                    "market_id": market.id,
                                    "error": str(refresh_exception),
                                    "market_data_age_seconds": market_data_age_seconds,
                                    "max_market_data_age_seconds": settings.MAX_MARKET_DATA_AGE_SECONDS,
                                    "refresh_attempts": refresh_attempts,
                                },
                            )
                            trades_skipped_edge += 1
                            _record_should_trade_blocked("stale_market_data_refresh_failed")
                            log_trade_decision(
                                market_id=market.id,
                                question=market.question,
                                decision=decision_for_edge.model_copy(
                                    update={"bet_size_pct": bet_pct}
                                ).model_dump(),
                                execution_audit=_build_execution_audit(
                                    decision_phase="post_market_refresh",
                                    decision_terminal=True,
                                    final_action="skip",
                                    final_reason="stale_market_data_refresh_failed",
                                    market_data_age_seconds=market_data_age_seconds,
                                    max_market_data_age_seconds=settings.MAX_MARKET_DATA_AGE_SECONDS,
                                    refresh_attempts=refresh_attempts,
                                    **audit_context,
                                ),
                            )
                            _record_terminal_outcome(
                                state_manager,
                                market.id,
                                "stale_market_data_refresh_failed",
                            )
                            continue
                    else:
                        logger.warning(
                            "Pre-order market refresh failed; using scheduled snapshot: market=%s error=%s",
                            market.id,
                            refresh_exception,
                            data={
                                "market_id": market.id,
                                "error": str(refresh_exception),
                                "market_data_age_seconds": market_data_age_seconds,
                                "refresh_attempts": refresh_attempts,
                            },
                        )
                if active_market is not market:
                    entry_price = _get_outcome_entry_price(active_market, decision.outcome)
                    refreshed_edge_ok, implied_prob, refreshed_edge_value, refreshed_edge_reason = _passes_refreshed_edge_guard(
                        active_market,
                        decision_for_edge,
                        settings,
                    )
                    audit_context["edge_market"] = refreshed_edge_value
                    if not refreshed_edge_ok:
                        trades_skipped_edge += 1
                        _record_should_trade_blocked("refreshed_edge_gate_blocked")
                        _record_rejection_reason(
                            rejection_breakdown,
                            "refreshed_edge_gate_blocked",
                        )
                        if refreshed_edge_reason == "non_sports_needs_direct_evidence":
                            _record_rejection_reason(
                                rejection_breakdown,
                                "non_sports_needs_direct",
                            )
                        elif refreshed_edge_reason == "edge_above_reasonable_max":
                            _record_rejection_reason(
                                rejection_breakdown,
                                "edge_above_reasonable_max",
                            )
                        logger.warning(
                            "SKIP [%s] '%s' -> refreshed edge gate (%s)",
                            market.id,
                            question_short,
                            refreshed_edge_reason,
                            data={
                                "market_id": market.id,
                                "final_reason": "refreshed_edge_gate_blocked",
                                "implied_prob": implied_prob,
                                "confidence": decision_for_edge.confidence,
                                "edge": refreshed_edge_value,
                                "score_breakdown": score_receipt_fields,
                            },
                        )
                        log_trade_decision(
                            market_id=market.id,
                            question=market.question,
                            decision=decision_for_edge.model_copy(
                                update={"bet_size_pct": bet_pct}
                            ).model_dump(),
                            execution_audit=_build_execution_audit(
                                decision_phase="post_market_refresh",
                                decision_terminal=True,
                                final_action="skip",
                                final_reason="refreshed_edge_gate_blocked",
                                implied_prob_market=implied_prob,
                                **audit_context,
                            ),
                        )
                        _record_terminal_outcome(state_manager, market.id, "refreshed_edge_gate_blocked")
                        continue

                if (
                    settings.ORDERBOOK_PRECHECK_ENABLED
                    and decision_for_edge.confidence >= settings.ORDERBOOK_PRECHECK_MIN_CONFIDENCE
                ):
                    option_index = None
                    for idx, market_outcome in enumerate(active_market.outcomes):
                        if market_outcome.name.upper() == decision.outcome.upper():
                            option_index = idx
                            break
                    if option_index is not None:
                        try:
                            orderbook = kalshi_client.get_market_orderbook(active_market.id)
                            best_sell = _best_orderbook_sell_price(orderbook, option_index)
                            if best_sell is not None:
                                _set_outcome_entry_price(
                                    active_market,
                                    decision.outcome,
                                    best_sell,
                                )
                            entry_price_for_check = _get_outcome_entry_price(active_market, decision.outcome)
                            required_order_count: int | None = None
                            if (
                                entry_price_for_check is not None
                                and entry_price_for_check > 0
                                and bet_amount > 0
                            ):
                                required_order_count = max(
                                    1,
                                    int(bet_amount / entry_price_for_check),
                                )
                            available_sell_quantity = _available_orderbook_sell_quantity(
                                orderbook=orderbook,
                                option_index=option_index,
                                max_price=entry_price_for_check,
                            )
                            if (
                                required_order_count is not None
                                and available_sell_quantity is not None
                                and available_sell_quantity
                                < max(
                                    required_order_count,
                                    settings.ORDERBOOK_MIN_RESTING_VOLUME,
                                )
                            ):
                                trades_skipped_edge += 1
                                _record_should_trade_blocked(
                                    "orderbook_insufficient_resting_volume"
                                )
                                _record_rejection_reason(
                                    rejection_breakdown,
                                    "orderbook_insufficient_resting_volume",
                                )
                                log_trade_decision(
                                    market_id=market.id,
                                    question=market.question,
                                    decision=decision_for_edge.model_copy(
                                        update={"bet_size_pct": bet_pct}
                                    ).model_dump(),
                                    execution_audit=_build_execution_audit(
                                        decision_phase="pre_orderbook_precheck",
                                        decision_terminal=True,
                                        final_action="skip",
                                        final_reason="orderbook_insufficient_resting_volume",
                                        option_index=option_index,
                                        required_order_count=required_order_count,
                                        min_resting_volume=settings.ORDERBOOK_MIN_RESTING_VOLUME,
                                        available_sell_quantity=available_sell_quantity,
                                        entry_price=entry_price_for_check,
                                        **audit_context,
                                    ),
                                )
                                _record_terminal_outcome(
                                    state_manager,
                                    market.id,
                                    "orderbook_insufficient_resting_volume",
                                )
                                logger.warning(
                                    "SKIP [%s] '%s' -> insufficient resting volume (required=%d available=%.2f)",
                                    market.id,
                                    question_short,
                                    required_order_count,
                                    available_sell_quantity,
                                    data={
                                        "market_id": market.id,
                                        "final_reason": "orderbook_insufficient_resting_volume",
                                        "option_index": option_index,
                                        "required_order_count": required_order_count,
                                        "min_resting_volume": settings.ORDERBOOK_MIN_RESTING_VOLUME,
                                        "available_sell_quantity": available_sell_quantity,
                                        "entry_price": entry_price_for_check,
                                        "score_breakdown": score_receipt_fields,
                                    },
                                )
                                continue
                            if (
                                best_sell is not None
                                and entry_price_for_check is not None
                                and best_sell > (
                                    entry_price_for_check + _ORDERBOOK_SPREAD_CUTOFF_DEFAULT
                                )
                            ):
                                if settings.CALIBRATION_MODE_ENABLED:
                                    spread_abs = best_sell - entry_price_for_check
                                    spread_payload = {
                                        "market_id": market.id,
                                        "best_sell_price": best_sell,
                                        "entry_price": entry_price_for_check,
                                        "orderbook_spread_abs": spread_abs,
                                        "analysis_duration_ms": round((time.monotonic() - market_start) * 1000, 2),
                                    }
                                    calibration_samples.append(spread_payload)
                                    logger.info(
                                        "Calibration orderbook sample: market=%s spread_abs=%.4f",
                                        market.id,
                                        spread_abs,
                                        data=spread_payload,
                                    )
                                trades_skipped_edge += 1
                                _record_should_trade_blocked("orderbook_spread_too_wide")
                                _record_rejection_reason(
                                    rejection_breakdown,
                                    "orderbook_spread_too_wide",
                                )
                                log_trade_decision(
                                    market_id=market.id,
                                    question=market.question,
                                    decision=decision_for_edge.model_copy(
                                        update={"bet_size_pct": bet_pct}
                                    ).model_dump(),
                                    execution_audit=_build_execution_audit(
                                        decision_phase="pre_orderbook_precheck",
                                        decision_terminal=True,
                                        final_action="skip",
                                        final_reason="orderbook_spread_too_wide",
                                        best_sell_price=best_sell,
                                        entry_price=entry_price_for_check,
                                        option_index=option_index,
                                        **audit_context,
                                    ),
                                )
                                _record_terminal_outcome(
                                    state_manager,
                                    market.id,
                                    "orderbook_spread_too_wide",
                                )
                                logger.warning(
                                    "SKIP [%s] '%s' -> orderbook precheck failed (best_sell=%.3f > entry=%.3f)",
                                    market.id,
                                    question_short,
                                    best_sell,
                                    entry_price_for_check,
                                    data={
                                        "market_id": market.id,
                                        "final_reason": "orderbook_spread_too_wide",
                                        "best_sell_price": best_sell,
                                        "entry_price": entry_price_for_check,
                                        "option_index": option_index,
                                        "score_breakdown": score_receipt_fields,
                                    },
                                )
                                continue
                        except Exception as exc:
                            logger.warning(
                                "Orderbook precheck failed open: market=%s error=%s",
                                market.id,
                                exc,
                                data={"market_id": market.id, "error": str(exc)},
                            )
                execution_entry_price = _get_outcome_entry_price(
                    active_market,
                    decision_for_edge.outcome,
                )
                if not _is_within_order_submission_band(
                    execution_entry_price,
                    settings,
                ):
                    trades_skipped_edge += 1
                    _record_should_trade_blocked("order_price_outside_submission_band")
                    _record_rejection_reason(
                        rejection_breakdown,
                        "order_price_outside_submission_band",
                    )
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_copy(
                            update={"bet_size_pct": bet_pct}
                        ).model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_phase="pre_order_submission_price_band",
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="order_price_outside_submission_band",
                            entry_price=execution_entry_price,
                            min_submission_price=settings.ORDER_SUBMISSION_MIN_PRICE,
                            max_submission_price=settings.ORDER_SUBMISSION_MAX_PRICE,
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(
                        state_manager,
                        market.id,
                        "order_price_outside_submission_band",
                    )
                    logger.warning(
                        "SKIP [%s] '%s' -> entry price outside submission band (price=%s, min=%.2f, max=%.2f)",
                        market.id,
                        question_short,
                        f"{execution_entry_price:.3f}" if execution_entry_price is not None else "n/a",
                        settings.ORDER_SUBMISSION_MIN_PRICE,
                        settings.ORDER_SUBMISSION_MAX_PRICE,
                        data={
                            "market_id": market.id,
                            "final_reason": "order_price_outside_submission_band",
                            "entry_price": execution_entry_price,
                            "min_submission_price": settings.ORDER_SUBMISSION_MIN_PRICE,
                            "max_submission_price": settings.ORDER_SUBMISSION_MAX_PRICE,
                            "score_breakdown": score_receipt_fields,
                        },
                    )
                    continue
                cycle_event_market_ids = event_cycle_traded_market_ids.get(
                    event_ticker_prefix,
                    set(),
                )
                cycle_event_outcomes = event_cycle_traded_outcomes.get(
                    event_ticker_prefix,
                    set(),
                )
                (
                    side_conflict_blocked,
                    existing_event_outcomes,
                ) = _event_side_conflict_blocked(
                    proposed_outcome=decision_for_edge.outcome,
                    open_event_outcomes=correlated_position_outcomes,
                    cycle_event_outcomes=cycle_event_outcomes,
                )
                if side_conflict_blocked:
                    definitive_override = _should_apply_definitive_side_override(
                        decision=decision_for_edge,
                        evidence_basis=evidence_basis,
                        primary_source_whitelisted=_is_definitive_outcome_eligible(
                            decision_for_edge,
                            settings,
                            market=market,
                        ),
                        cycle_overrides_applied=cycle_definitive_overrides_applied,
                        max_overrides_per_cycle=settings.MAX_DEFINITIVE_OVERRIDES_PER_CYCLE,
                    )
                    if definitive_override:
                        cycle_definitive_overrides_applied += 1
                        logger.info(
                            "Definitive side-conflict override applied: market=%s proposed=%s existing=%s",
                            market.id,
                            decision_for_edge.outcome,
                            existing_event_outcomes,
                            data={
                                "market_id": market.id,
                                "override_reason": "definitive_outcome_side_override",
                                "cycle_overrides_applied": cycle_definitive_overrides_applied,
                            },
                        )
                    else:
                        trades_skipped_position += 1
                        _record_should_trade_blocked("event_side_conflict_blocked")
                        _record_rejection_reason(
                            rejection_breakdown,
                            "event_side_conflict_blocked",
                        )
                        log_trade_decision(
                            market_id=market.id,
                            question=market.question,
                            decision=decision_for_edge.model_copy(
                                update={"bet_size_pct": bet_pct}
                            ).model_dump(),
                            execution_audit=_build_execution_audit(
                                decision_phase="pre_order_submission_event_side_conflict",
                                decision_terminal=True,
                                final_action="skip",
                                final_reason="event_side_conflict_blocked",
                                proposed_outcome=_normalize_outcome_key(
                                    decision_for_edge.outcome
                                ),
                                existing_event_outcomes=existing_event_outcomes,
                                **audit_context,
                            ),
                        )
                        _record_terminal_outcome(
                            state_manager,
                            market.id,
                            "event_side_conflict_blocked",
                        )
                        continue
                correlated_other_positions_count = len(
                    {
                        market_id
                        for market_id in correlated_position_market_ids
                        if market_id and market_id != market.id
                    }
                )
                correlated_cycle_other_count = len(
                    {
                        market_id
                        for market_id in cycle_event_market_ids
                        if market_id and market_id != market.id
                    }
                )
                total_other_event_exposures = (
                    correlated_other_positions_count + correlated_cycle_other_count
                )
                if _event_concentration_blocked(
                    max_bets_per_event=settings.MAX_BETS_PER_EVENT,
                    open_other_positions_count=correlated_other_positions_count,
                    cycle_other_attempts_count=correlated_cycle_other_count,
                ):
                    trades_skipped_position += 1
                    _record_should_trade_blocked("event_concentration_blocked")
                    _record_rejection_reason(
                        rejection_breakdown,
                        "event_concentration_blocked",
                    )
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_copy(
                            update={"bet_size_pct": bet_pct}
                        ).model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_phase="pre_order_submission_event_cap",
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="event_concentration_blocked",
                            max_bets_per_event=settings.MAX_BETS_PER_EVENT,
                            total_other_event_exposures=total_other_event_exposures,
                            correlated_cycle_positions_count=correlated_cycle_other_count,
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(
                        state_manager,
                        market.id,
                        "event_concentration_blocked",
                    )
                    continue
                if _daily_trade_cap_reached(
                    daily_trade_count=daily_trade_count,
                    max_trades_per_day=settings.MAX_TRADES_PER_DAY,
                ):
                    trades_skipped_position += 1
                    _record_should_trade_blocked("daily_limit_reached")
                    _record_rejection_reason(rejection_breakdown, "daily_limit_reached")
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_copy(
                            update={"bet_size_pct": bet_pct}
                        ).model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_phase="pre_order_submission_daily_cap",
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="daily_limit_reached",
                            max_trades_per_day=settings.MAX_TRADES_PER_DAY,
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(state_manager, market.id, "daily_limit_reached")
                    continue
                daily_balance_delta = _daily_balance_delta_usdc(
                    day_start_balance=daily_start_balance,
                    current_balance=last_known_portfolio_value,
                )
                daily_drawdown = max(
                    0.0,
                    -(daily_balance_delta if daily_balance_delta is not None else 0.0),
                )
                if _daily_drawdown_cap_reached(
                    daily_balance_delta=daily_balance_delta,
                    max_daily_drawdown_usdc=settings.MAX_DAILY_DRAWDOWN_USDC,
                ):
                    trades_skipped_position += 1
                    _record_should_trade_blocked("daily_drawdown_limit")
                    _record_rejection_reason(
                        rejection_breakdown,
                        "daily_drawdown_limit",
                    )
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_copy(
                            update={"bet_size_pct": bet_pct}
                        ).model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_phase="pre_order_submission_daily_cap",
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="daily_drawdown_limit",
                            daily_drawdown_usdc=daily_drawdown,
                            max_daily_drawdown_usdc=settings.MAX_DAILY_DRAWDOWN_USDC,
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(state_manager, market.id, "daily_drawdown_limit")
                    continue
                if trades_attempted >= settings.MAX_TRADES_PER_CYCLE:
                    trades_skipped_edge += 1
                    _record_should_trade_blocked("max_trades_per_cycle_reached")
                    _record_rejection_reason(
                        rejection_breakdown,
                        "max_trades_per_cycle_reached",
                    )
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_copy(
                            update={"bet_size_pct": bet_pct}
                        ).model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_phase="pre_order_submission_cap",
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="max_trades_per_cycle_reached",
                            max_trades_per_cycle=settings.MAX_TRADES_PER_CYCLE,
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(
                        state_manager,
                        market.id,
                        "max_trades_per_cycle_reached",
                    )
                    continue
                logger.info(
                    "TRADE: [%s] '%s' -> %s @ $%.2f (conf=%.2f)",
                    market.id,
                    question_short,
                    decision.outcome,
                    bet_amount,
                    decision_for_edge.confidence,
                )

                close_time_for_submission = active_market.close_time
                if close_time_for_submission and close_time_for_submission.tzinfo is None:
                    close_time_for_submission = close_time_for_submission.replace(tzinfo=timezone.utc)
                if close_time_for_submission and close_time_for_submission <= datetime.now(timezone.utc):
                    _record_should_trade_blocked("market_closed_during_cycle")
                    logger.warning(
                        "SKIP [%s] '%s' -> market closed before submission (close_time=%s)",
                        market.id,
                        question_short,
                        close_time_for_submission.isoformat(),
                        data={
                            "market_id": market.id,
                            "final_reason": "market_closed_during_cycle",
                            "close_time": close_time_for_submission.isoformat(),
                            "score_breakdown": score_receipt_fields,
                        },
                    )
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_copy(
                            update={"bet_size_pct": bet_pct}
                        ).model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_phase="order_submission",
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="market_closed_during_cycle",
                            bet_amount_usdc=bet_amount,
                            market_data_age_seconds=market_data_age_seconds,
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(state_manager, market.id, "market_closed_during_cycle")
                    continue

                order = _build_order_request_from_market(
                    active_market,
                    decision_for_edge,
                    bet_amount,
                )
                try:
                    order_response = kalshi_client.submit_order(
                        order,
                        market=active_market,
                    )
                except InsufficientBalanceError as balance_exc:
                    available = balance_exc.available
                    logger.warning(
                        "INSUFFICIENT BALANCE: available=$%.2f, needed=$%.2f - "
                        "Switching to analysis-only mode for rest of cycle",
                        available if available is not None else 0,
                        bet_amount,
                        data={
                            "market_id": market.id,
                            "amount_usdc": bet_amount,
                            "available_balance": available,
                        },
                    )
                    analysis_only_mode = True
                    if available is not None:
                        last_known_balance = available
                    else:
                        _refresh_last_known_balance()
                    trades_skipped_balance += 1
                    _record_should_trade_blocked("insufficient_balance")
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_copy(
                            update={"bet_size_pct": bet_pct}
                        ).model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_phase="order_submission",
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="insufficient_balance",
                            bet_amount_usdc=bet_amount,
                            market_data_age_seconds=market_data_age_seconds,
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(state_manager, market.id, "insufficient_balance")
                    continue  # Continue analyzing remaining markets
                except MarketClosedError as closed_exc:
                    logger.info(
                        "Order skipped because market is closed: market=%s error=%s",
                        market.id,
                        closed_exc,
                        data={"market_id": market.id, "error": str(closed_exc)},
                    )
                    _record_runtime_score_order_attempt_if_below(audit_context)
                    trades_attempted += 1
                    _register_order_attempt(
                        event_ticker_prefix,
                        market.id,
                        decision_for_edge.outcome,
                    )
                    family_stats = execution_family_stats.setdefault(
                        market_family_name,
                        {"order_attempts": 0.0, "orders_filled": 0.0, "orders_canceled_unfilled": 0.0, "usd_deployed": 0.0},
                    )
                    family_stats["order_attempts"] += 1
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_copy(
                            update={"bet_size_pct": bet_pct}
                        ).model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_phase="order_submission",
                            decision_terminal=True,
                            final_action="order_attempt",
                            final_reason="market_closed",
                            bet_amount_usdc=bet_amount,
                            order_error=str(closed_exc),
                            market_data_age_seconds=market_data_age_seconds,
                            **audit_context,
                        ),
                    )
                    _refresh_last_known_balance()
                    _record_terminal_outcome(state_manager, market.id, "market_closed")
                    continue
                except Exception as order_exc:
                    error_msg = str(order_exc)
                    normalized_order_error = error_msg.lower()
                    order_failure_reason = "order_submission_failed"
                    if "invalid parameters" in normalized_order_error:
                        order_failure_reason = "order_submission_invalid_parameters"
                    elif "timeinforce" in normalized_order_error or "time_in_force" in normalized_order_error:
                        order_failure_reason = "order_submission_invalid_time_in_force"
                    if (
                        "Could not map outcome" in error_msg
                        and not market_outcome_mismatch_counted
                    ):
                        outcome_mismatch_blocked += 1
                        market_outcome_mismatch_counted = True
                    logger.error(
                        "Order submission failed: market=%s, error=%s",
                        market.id,
                        order_exc,
                        data={"market_id": market.id, "error": error_msg},
                    )
                    _record_runtime_score_order_attempt_if_below(audit_context)
                    trades_attempted += 1
                    _register_order_attempt(
                        event_ticker_prefix,
                        market.id,
                        decision_for_edge.outcome,
                    )
                    family_stats = execution_family_stats.setdefault(
                        market_family_name,
                        {"order_attempts": 0.0, "orders_filled": 0.0, "orders_canceled_unfilled": 0.0, "usd_deployed": 0.0},
                    )
                    family_stats["order_attempts"] += 1
                    _record_rejection_reason(rejection_breakdown, order_failure_reason)
                    try:
                        state_manager.increment_fill_failure_count(market.id)
                    except Exception as fill_failure_exc:
                        logger.warning(
                            "Failed to increment fill failure count: market=%s error=%s",
                            market.id,
                            fill_failure_exc,
                            data={"market_id": market.id, "error": str(fill_failure_exc)},
                        )
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_copy(
                            update={"bet_size_pct": bet_pct}
                        ).model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_phase="order_submission",
                            decision_terminal=True,
                            final_action="order_attempt",
                            final_reason=order_failure_reason,
                            bet_amount_usdc=bet_amount,
                            order_error=error_msg,
                            market_data_age_seconds=market_data_age_seconds,
                            **audit_context,
                        ),
                    )
                    _refresh_last_known_balance()
                    _record_terminal_outcome(state_manager, market.id, order_failure_reason)
                    continue  # Continue to next market for other errors

                _record_runtime_score_order_attempt_if_below(audit_context)
                trades_attempted += 1
                _register_order_attempt(
                    event_ticker_prefix,
                    market.id,
                    decision_for_edge.outcome,
                )
                family_stats = execution_family_stats.setdefault(
                    market_family_name,
                    {"order_attempts": 0.0, "orders_filled": 0.0, "orders_canceled_unfilled": 0.0, "usd_deployed": 0.0},
                )
                family_stats["order_attempts"] += 1

                logger.info(
                    "Order submitted: id=%s, status=%s",
                    order_response.id,
                    order_response.status,
                    data={
                        "order_id": order_response.id,
                        "status": order_response.status,
                        "market_id": market.id,
                    },
                )
                normalized_order_status = (order_response.status or "").strip().lower()
                order_cancel_reason = None
                order_fill_count = None
                if normalized_order_status in {"cancelled", "canceled"}:
                    if isinstance(order_response.raw, dict):
                        order_cancel_reason = _extract_order_cancel_reason(order_response.raw)
                        order_fill_count = _extract_order_fill_count(order_response.raw)
                    logger.warning(
                        "Order was canceled by exchange: market=%s order_id=%s status=%s reason=%s fill_count=%s raw=%s",
                        market.id,
                        order_response.id,
                        order_response.status,
                        order_cancel_reason,
                        order_fill_count,
                        order_response.raw,
                        data={
                            "market_id": market.id,
                            "order_id": order_response.id,
                            "order_status": order_response.status,
                            "order_cancel_reason": order_cancel_reason,
                            "order_fill_count": order_fill_count,
                            "order_raw": order_response.raw,
                        },
                    )
                unfilled_canceled_order = (
                    normalized_order_status in {"cancelled", "canceled"}
                    and (order_fill_count is None or order_fill_count <= 0.0)
                )
                fallback_attempted = False
                fallback_order_response = None
                if (
                    unfilled_canceled_order
                    and settings.ORDER_FALLBACK_TO_MARKET
                    and decision_for_edge.confidence >= settings.ORDER_FALLBACK_MIN_CONFIDENCE
                    and (active_market.liquidity_usdc or 0.0) >= settings.ORDER_FALLBACK_MIN_LIQUIDITY_USDC
                ):
                    fallback_attempted = True
                    fallback_order = _build_order_request_from_market(
                        active_market,
                        decision_for_edge,
                        bet_amount,
                    ).model_copy(update={"order_type": "market"})
                    try:
                        fallback_order_response = kalshi_client.submit_order(
                            fallback_order,
                            market=active_market,
                            retry_suffix="fb",
                        )
                        order_response = fallback_order_response
                        normalized_order_status = (order_response.status or "").strip().lower()
                        order_cancel_reason = None
                        order_fill_count = None
                        if normalized_order_status in {"cancelled", "canceled"}:
                            if isinstance(order_response.raw, dict):
                                order_cancel_reason = _extract_order_cancel_reason(
                                    order_response.raw
                                )
                                order_fill_count = _extract_order_fill_count(
                                    order_response.raw
                                )
                        unfilled_canceled_order = (
                            normalized_order_status in {"cancelled", "canceled"}
                            and (order_fill_count is None or order_fill_count <= 0.0)
                        )
                    except Exception as fallback_exc:
                        logger.warning(
                            "Order fallback attempt failed: market=%s error=%s",
                            market.id,
                            fallback_exc,
                            data={"market_id": market.id, "error": str(fallback_exc)},
                        )
                final_reason = "order_submitted"
                terminal_outcome = "order_submitted"
                if unfilled_canceled_order:
                    trades_canceled_unfilled += 1
                    family_stats["orders_canceled_unfilled"] += 1
                    _record_rejection_reason(
                        rejection_breakdown,
                        "order_canceled_unfilled",
                    )
                    state_manager.increment_fill_failure_count(market.id)
                    final_reason = "order_canceled_unfilled"
                    terminal_outcome = "order_canceled_unfilled"
                else:
                    trades_filled += 1
                    total_usd_deployed += bet_amount
                    family_stats["orders_filled"] += 1
                    family_stats["usd_deployed"] += bet_amount
                    state_manager.reset_fill_failure_count(market.id)
                    if last_known_balance is not None:
                        last_known_balance = max(0.0, float(last_known_balance) - float(bet_amount))
                _refresh_last_known_balance()
                log_trade_decision(
                    market_id=market.id,
                    question=market.question,
                    decision=decision_for_edge.model_copy(
                        update={"bet_size_pct": bet_pct}
                    ).model_dump(),
                    order=_order_response_receipt(order_response),
                    execution_audit=_build_execution_audit(
                        decision_phase="order_submission",
                        decision_terminal=True,
                        final_action="order_attempt",
                        final_reason=final_reason,
                        bet_amount_usdc=bet_amount,
                        order_id=order_response.id,
                        order_status=order_response.status,
                        order_cancel_reason=order_cancel_reason,
                        order_fill_count=order_fill_count,
                        fallback_attempted=fallback_attempted,
                        fallback_order_id=(
                            fallback_order_response.id
                            if fallback_order_response is not None
                            else None
                        ),
                        fallback_order_status=(
                            fallback_order_response.status
                            if fallback_order_response is not None
                            else None
                        ),
                        balance_after_trade=last_known_balance if not unfilled_canceled_order else None,
                        market_data_age_seconds=market_data_age_seconds,
                        **audit_context,
                    ),
                )
                _record_terminal_outcome(state_manager, market.id, terminal_outcome)
                if unfilled_canceled_order:
                    logger.info(
                        "Skip trade recording for unfilled canceled order: market=%s order_id=%s",
                        market.id,
                        order_response.id,
                        data={
                            "market_id": market.id,
                            "order_id": order_response.id,
                            "order_status": order_response.status,
                            "order_fill_count": order_fill_count,
                        },
                    )
                    continue

                try:
                    client_price = None
                    client_shares = None
                    if isinstance(order_response.raw, dict):
                        client_price = order_response.raw.get("client_price")
                        client_shares = order_response.raw.get("client_qty_shares")
                    state_manager.record_trade(
                        market.id,
                        order_response,
                        bet_amount,
                        outcome=decision.outcome,
                        entry_price=client_price or entry_price,
                        implied_prob=implied_prob,
                        confidence=decision_for_edge.confidence,
                        shares=client_shares,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to record trade for market %s: %s",
                        market.id,
                        exc,
                        data={"market_id": market.id, "error": str(exc)},
                    )

                market_duration = (time.monotonic() - market_start) * 1000
                logger.debug(
                    "Market processing complete: id=%s, duration=%.2fms",
                    market.id,
                    market_duration,
                )

            if settings.EXPORT_STATE_JSON:
                try:
                    state_manager.export_to_json(settings.STATE_JSON_EXPORT_PATH)
                except Exception as exc:
                    logger.warning(
                        "State export failed: %s",
                        exc,
                        data={"path": settings.STATE_JSON_EXPORT_PATH, "error": str(exc)},
                    )

            cycle_duration = (time.monotonic() - cycle_start) * 1000
            mode_suffix = " [ANALYSIS_ONLY]" if analysis_only_mode else ""
            best_candidate_market_id: str | None = None
            best_candidate_score: float | None = None
            if analysis_candidates:
                top_candidate = analysis_candidates[0]
                top_market = top_candidate.get("market")
                if isinstance(top_market, Market):
                    best_candidate_market_id = top_market.id
                    top_result = analysis_results.get(top_market.id, {})
                    if isinstance(top_result, dict):
                        top_score = top_result.get("pre_execution_final_score")
                        if isinstance(top_score, (int, float)):
                            best_candidate_score = float(top_score)
            api_tokens_consumed = cycle_prompt_tokens + cycle_completion_tokens
            api_cost_estimate_usd = _estimate_api_cost_usd(
                prompt_tokens=cycle_prompt_tokens,
                completion_tokens=cycle_completion_tokens,
                cached_tokens=cycle_cached_tokens,
                settings=settings,
            )
            cumulative_api_cost_estimate_usd += api_cost_estimate_usd
            execution_family_breakdown = {
                family_name: {
                    "order_attempts": int(stats.get("order_attempts", 0)),
                    "orders_filled": int(stats.get("orders_filled", 0)),
                    "orders_canceled_unfilled": int(
                        stats.get("orders_canceled_unfilled", 0)
                    ),
                    "usd_deployed": round(float(stats.get("usd_deployed", 0.0)), 2),
                }
                for family_name, stats in sorted(execution_family_stats.items())
            }
            cumulative_cycle_pnl_estimate = _resolved_pnl_estimate_total(state_manager)
            exchange_realized_pnl_total = state_manager.get_exchange_realized_pnl_total()
            api_cost_per_fill = (
                round(api_cost_estimate_usd / trades_filled, 6)
                if trades_filled > 0
                else None
            )
            api_cost_per_usd_deployed = (
                round(api_cost_estimate_usd / total_usd_deployed, 6)
                if total_usd_deployed > 0
                else None
            )
            required_rejection_bucket_keys = (
                "ticker_prefix_pnl_block",
                "historical_prefix_pnl_block",
                "historical_family_pnl_block",
                "hallucinated_edge",
                "non_sports_needs_direct",
                "non_sports_missing_primary_source",
                "late_stage_overconfidence",
                "numeric_strike_bin",
            )
            normalized_rejection_breakdown: dict[str, int] = {
                key: int(rejection_breakdown.get(key, 0))
                for key in required_rejection_bucket_keys
            }
            for reason, count in rejection_breakdown.items():
                normalized_rejection_breakdown[str(reason)] = int(count)
            per_family_edge_p50: dict[str, float] = {}
            per_family_edge_p90: dict[str, float] = {}
            for family_name, edges in sorted(family_edge_samples.items()):
                edge_p50 = _percentile(edges, 0.50)
                edge_p90 = _percentile(edges, 0.90)
                if edge_p50 is not None:
                    per_family_edge_p50[family_name] = round(edge_p50, 4)
                if edge_p90 is not None:
                    per_family_edge_p90[family_name] = round(edge_p90, 4)
            mean_confidence_delta = (
                sum(confidence_delta_samples) / len(confidence_delta_samples)
                if confidence_delta_samples
                else 0.0
            )
            calibration_samples: list[dict[str, Any]] = []
            try:
                calibration_samples = state_manager.get_confidence_bucket_calibration(
                    lookback_days=14
                )
            except Exception as exc:
                logger.debug(
                    "Calibration sample snapshot lookup failed: %s",
                    exc,
                    data={"error": str(exc)},
                )
            research_queue_size = len(research_queue)

            _top_candidates_summary: list[dict[str, Any]] = []
            for _tc_candidate in analysis_candidates[:5]:
                _tc_market = _tc_candidate.get("market")
                if not isinstance(_tc_market, Market):
                    continue
                _tc_result = analysis_results.get(_tc_market.id, {})
                _tc_decision = _tc_result.get("decision")
                _tc_score = _tc_result.get(
                    "execution_final_score",
                    _tc_result.get("pre_execution_final_score"),
                )
                _tc_explanation = _tc_result.get("score_breakdown_explanation", "")
                _tc_final_reason = ""
                if isinstance(_tc_decision, TradeDecision):
                    if _tc_decision.should_trade:
                        _tc_final_reason = "should_trade"
                    elif _tc_decision.abstain:
                        _tc_final_reason = "abstain"
                    else:
                        _tc_final_reason = "no_trade_recommended"
                _top_candidates_summary.append({
                    "market_id": _tc_market.id,
                    "family": market_family(_tc_market),
                    "final_score": round(float(_tc_score or 0.0), 4),
                    "score_breakdown": str(_tc_explanation or "")[:200],
                    "final_reason": _tc_final_reason,
                    "selection_rank_score": _tc_candidate.get("selection_rank_score"),
                    "selection_rank_components": _tc_candidate.get(
                        "selection_rank_components"
                    ),
                    "source_match_class": getattr(
                        _tc_decision,
                        "source_match_class",
                        None,
                    )
                    if isinstance(_tc_decision, TradeDecision)
                    else None,
                })

            _confidence_bucket_decision_counts: dict[str, int] = {}
            for _cbd_mid, _cbd_result in analysis_results.items():
                if not isinstance(_cbd_result, dict):
                    continue
                _cbd_decision = _cbd_result.get("decision")
                if not isinstance(_cbd_decision, TradeDecision):
                    continue
                _cbd_conf = _cbd_decision.confidence
                _cbd_left = int(_cbd_conf * 10) / 10
                _cbd_bucket = f"{_cbd_left:.1f}-{_cbd_left + 0.1:.1f}"
                _confidence_bucket_decision_counts[_cbd_bucket] = (
                    _confidence_bucket_decision_counts.get(_cbd_bucket, 0) + 1
                )

            _near_miss_reason_counts: dict[str, int] = {}
            for _near_miss in score_near_misses:
                _near_miss_reasons = (
                    _near_miss.get("critical_rejection_reasons")
                    or _near_miss.get("rejection_reasons")
                    or ["score_below_threshold"]
                )
                for _near_miss_reason in _near_miss_reasons:
                    _reason_key = str(_near_miss_reason or "").strip()
                    if not _reason_key:
                        continue
                    _near_miss_reason_counts[_reason_key] = (
                        _near_miss_reason_counts.get(_reason_key, 0) + 1
                    )
            top_near_miss_research_reasons = [
                {"reason": reason, "count": count}
                for reason, count in sorted(
                    _near_miss_reason_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:8]
            ]
            pre_vs_runtime_score_delta = (
                round(
                    sum(pre_vs_runtime_score_deltas)
                    / len(pre_vs_runtime_score_deltas),
                    4,
                )
                if pre_vs_runtime_score_deltas
                else None
            )
            pre_vs_runtime_score_delta_abs_max = (
                round(max(abs(delta) for delta in pre_vs_runtime_score_deltas), 4)
                if pre_vs_runtime_score_deltas
                else None
            )
            cost_per_analyzed_market = (
                round(api_cost_estimate_usd / markets_analyzed, 6)
                if markets_analyzed > 0
                else None
            )
            cost_per_order_attempt = (
                round(api_cost_estimate_usd / trades_attempted, 6)
                if trades_attempted > 0
                else None
            )

            cycle_receipt = {
                "cycle": cycle_count,
                "cycle_id": cycle_id,
                "duration_ms": round(cycle_duration, 2),
                "fetched_markets": fetched_count,
                # Cycle 2 review: catalog-topology fields so future log audits
                # can answer "what survived the page cap?" without scanning
                # DEBUG-level pre-analysis rejections.
                "pages_fetched": fetch_pages_fetched,
                "page_cap_hit": fetch_page_cap_hit,
                "kalshi_max_fetch_pages": int(settings.KALSHI_MAX_FETCH_PAGES),
                "mve_filter_active": mve_filter_active,
                "mve_filter_setting": mve_filter_setting or None,
                "kalshi_eligible_floor": eligible_floor,
                "eligible_floor_warning_triggered": eligible_floor_warning_triggered,
                "eligible_market_families": eligible_market_families,
                "eligible_markets": len(markets),
                "analysis_candidates": analysis_candidates_count,
                "pre_analysis_passed": pre_analysis_passed,
                "analyzed_markets": markets_analyzed,
                "decisions_made": decisions_made,
                "validation_passed": validation_passed,
                "refined_markets": markets_refined,
                "execution_candidates": execution_candidates,
                "research_queue_size": research_queue_size,
                "research_queue_emissions": research_queue_size,
                "research_only_emissions": research_only_emissions,
                "research_queue_drained_count": research_queue_drained_count,
                "research_queue_drain_skipped_stale_count": (
                    research_queue_drain_skipped_stale_count
                ),
                "research_queue_drain_skipped_low_priority_count": (
                    research_queue_drain_skipped_low_priority_count
                ),
                "research_queue_emergency_probes_count": (
                    research_queue_emergency_probes_count
                ),
                "research_queue_zero_yield_promotions_count": (
                    research_queue_zero_yield_promotions_count
                ),
                "no_grok_research_routed": (
                    pre_analysis_research_routed_count + research_only_emissions
                ),
                "research_queue_deduped": 0,
                "edge_gate_passed": edge_gate_passed,
                "score_gate_passed": score_gate_passed,
                "order_attempted": trades_attempted,
                "order_attempts": trades_attempted,
                "orders_filled": trades_filled,
                "orders_canceled_unfilled": trades_canceled_unfilled,
                "total_usd_deployed": round(total_usd_deployed, 2),
                "projected_daily_expected_value_usdc": round(
                    cycle_projected_daily_ev_usdc,
                    4,
                ),
                "daily_expectancy_primary_targets": cycle_primary_targets_selected,
                "daily_expectancy_satellites": cycle_satellites_selected,
                "execution_family_breakdown": execution_family_breakdown,
                "should_trade_but_blocked": should_trade_but_blocked,
                "should_trade_blocked_breakdown": dict(
                    sorted(should_trade_blocked_breakdown.items())
                ),
                "participation_tier_breakdown": dict(
                    sorted(participation_tier_breakdown.items())
                ),
                "timeout_routed_to_monitor_only_count": timeout_routed_to_monitor_only_count,
                "negative_best_score_skipped_count": negative_best_score_skipped_count,
                "effective_research_band": round(effective_research_band, 4),
                "research_band_widened_by": round(research_band_widened_by, 4),
                "pre_score_distribution": _summarize_distribution(
                    cycle_pre_score_samples
                ),
                "soft_research_threshold_gap_distribution": (
                    _summarize_distribution(cycle_soft_research_threshold_gap_samples)
                ),
                "deprioritized_market_samples": deprioritized_market_samples,
                "skip_counts": {
                    "no_trade": trades_skipped_no_trade,
                    "confidence": trades_skipped_confidence,
                    "edge": trades_skipped_edge,
                    "position": trades_skipped_position,
                    "balance": trades_skipped_balance,
                    "kelly_sub_floor": trades_skipped_kelly_sub_floor,
                    "pre_analysis": pre_analysis_blocked,
                },
                "rejection_breakdown": dict(
                    sorted(normalized_rejection_breakdown.items())
                ),
                "pre_analysis_rejection_breakdown": pre_analysis_rejection_breakdown,
                "score_rejection_reason_breakdown": score_rejection_reason_breakdown,
                "blocked_direct_evidence_count": blocked_direct_evidence_count,
                "hallucinated_edge_blocks": int(
                    normalized_rejection_breakdown.get("hallucinated_edge", 0)
                ),
                "high_edge_hallucination_blocks": int(
                    normalized_rejection_breakdown.get("hallucinated_edge", 0)
                    + normalized_rejection_breakdown.get("extreme_edge_learning_queue", 0)
                    + normalized_rejection_breakdown.get("edge_above_reasonable_max", 0)
                ),
                "source_floor_suppressed": evidence_floor_suppressed_count,
                "estimated_cost_saved_usd": round(
                    (
                        pre_analysis_research_routed_count
                        + research_only_emissions
                    )
                    * cost_per_analyzed_market,
                    6,
                )
                if cost_per_analyzed_market is not None
                else None,
                "fallback_high_conf_blocks": int(
                    normalized_rejection_breakdown.get("fallback_high_confidence_trade", 0)
                ),
                "per_family_edge_p50": per_family_edge_p50,
                "per_family_edge_p90": per_family_edge_p90,
                "api_tokens_consumed": api_tokens_consumed,
                "api_prompt_tokens": cycle_prompt_tokens,
                "api_completion_tokens": cycle_completion_tokens,
                "api_reasoning_tokens": cycle_reasoning_tokens,
                "api_cached_tokens": cycle_cached_tokens,
                "api_cost_estimate_usd": round(api_cost_estimate_usd, 6),
                "cost_per_analyzed_market": cost_per_analyzed_market,
                "cost_per_order_attempt": cost_per_order_attempt,
                "api_cost_per_fill": api_cost_per_fill,
                "api_cost_per_usd_deployed": api_cost_per_usd_deployed,
                "cumulative_api_cost_estimate_usd": round(cumulative_api_cost_estimate_usd, 6),
                "grok_tokens_per_trade": round(
                    (api_tokens_consumed / trades_filled),
                    2,
                )
                if trades_filled > 0
                else None,
                "best_candidate_market_id": best_candidate_market_id,
                "best_candidate_score": best_candidate_score,
                "evidence_basis_breakdown": dict(sorted(evidence_basis_breakdown.items())),
                "confidence_calibration_applied": confidence_calibration_applied_count,
                "calibration_samples": calibration_samples,
                "runtime_score_evaluation_count": runtime_score_evaluation_count,
                "score_gate_score_source_counts": dict(
                    sorted(score_gate_score_source_counts.items())
                ),
                "pre_vs_runtime_score_delta": pre_vs_runtime_score_delta,
                "pre_vs_runtime_score_delta_abs_max": pre_vs_runtime_score_delta_abs_max,
                "runtime_score_below_threshold_order_count": (
                    runtime_score_below_threshold_order_count
                ),
                "top_near_miss_research_reasons": top_near_miss_research_reasons,
                "raw_vs_calibrated_delta": round(
                    confidence_calibration_delta_sum / max(1, markets_analyzed),
                    4,
                ),
                "mean_raw_vs_calibrated_confidence_delta": round(mean_confidence_delta, 4),
                "historical_win_rate_at_bucket": round(
                    (
                        sum(confidence_calibration_historical_win_rates)
                        / len(confidence_calibration_historical_win_rates)
                    ),
                    4,
                )
                if confidence_calibration_historical_win_rates
                else None,
                "analysis_only_mode": analysis_only_mode,
                "balance_at_cycle_start": cycle_balance_start,
                "cash_balance_at_cycle_start": cycle_cash_balance,
                "total_portfolio_value_at_cycle_start": cycle_balance_start,
                "cumulative_cycle_pnl_estimate": round(cumulative_cycle_pnl_estimate, 2),
                "exchange_realized_pnl_total": round(exchange_realized_pnl_total, 2),
                "rejection_funnel_summary": rejection_funnel_summary[:50],
                "funnel_stage_counts": {
                    "fetched": fetched_count,
                    "filtered": len(markets),
                    "deduped": len(markets),
                    "pre_scored": original_analysis_candidates_count,
                    "pre_analysis_demoted": pre_analysis_blocked,
                    "analyzed": markets_analyzed,
                    "decided": decisions_made,
                    "research_queued": research_queue_size,
                    "order_submitted": trades_attempted,
                    "orders_filled": trades_filled,
                },
                "family_trade_counts": {
                    family_name: int(stats.get("order_attempts", 0))
                    for family_name, stats in sorted(execution_family_stats.items())
                },
                "top_candidates_summary": _top_candidates_summary,
                "confidence_bucket_decision_counts": dict(
                    sorted(_confidence_bucket_decision_counts.items())
                ),
                "pre_analysis_research_routed_count": pre_analysis_research_routed_count,
                "markets_considered": len(markets),
                "markets_filtered": len(markets),
            }
            logger.info(
                "Cycle receipt",
                data={"cycle_receipt": cycle_receipt},
            )
            try:
                state_manager.record_cycle_receipt(
                    cycle_id=cycle_id,
                    cycle_number=cycle_count,
                    payload=cycle_receipt,
                )
            except Exception as receipt_exc:
                logger.warning(
                    "Cycle receipt persistence failed: cycle=%s error=%s",
                    cycle_id,
                    receipt_exc,
                    data={"cycle_id": cycle_id, "error": str(receipt_exc)},
                )
            logger.info(
                "Price bucket summary: low=%d mid=%d high=%d",
                price_bucket_stats[_PRICE_BUCKET_LOW],
                price_bucket_stats[_PRICE_BUCKET_MID],
                price_bucket_stats[_PRICE_BUCKET_HIGH],
                data={
                    "bucket_low": price_bucket_stats[_PRICE_BUCKET_LOW],
                    "bucket_mid": price_bucket_stats[_PRICE_BUCKET_MID],
                    "bucket_high": price_bucket_stats[_PRICE_BUCKET_HIGH],
                },
            )
            logger.info(
                "Pre-analysis rejections: %s",
                ", ".join(
                    f"{reason}={count}"
                    for reason, count in sorted(pre_analysis_rejection_breakdown.items())
                )
                if pre_analysis_rejection_breakdown
                else "none",
                data={"pre_analysis_rejection_breakdown": pre_analysis_rejection_breakdown},
            )
            logger.info(
                "Rejections summary: %s",
                ", ".join(
                    f"{reason}={count}"
                    for reason, count in sorted(rejection_breakdown.items())
                )
                if rejection_breakdown
                else "none",
                data={"rejection_breakdown": rejection_breakdown},
            )
            logger.info(
                "Should-trade blocked summary: %s",
                ", ".join(
                    f"{reason}={count}"
                    for reason, count in sorted(should_trade_blocked_breakdown.items())
                )
                if should_trade_blocked_breakdown
                else "none",
                data={
                    "should_trade_but_blocked": should_trade_but_blocked,
                    "should_trade_blocked_breakdown": should_trade_blocked_breakdown,
                },
            )
            logger.info(
                "Score rejection reasons: %s",
                ", ".join(
                    f"{reason}={count}"
                    for reason, count in sorted(score_rejection_reason_breakdown.items())
                )
                if score_rejection_reason_breakdown
                else "none",
                data={"score_rejection_reason_breakdown": score_rejection_reason_breakdown},
            )
            if score_near_misses:
                ranked_near_misses = sorted(
                    score_near_misses,
                    key=lambda item: float(item.get("score_gap", 0.0)),
                )[:5]
                logger.info(
                    "Score near misses (top %d): %s",
                    len(ranked_near_misses),
                    ", ".join(
                        f"{item['market_id']} gap={item['score_gap']:.4f}"
                        for item in ranked_near_misses
                    ),
                    data={"score_near_misses": ranked_near_misses},
                )
            if research_queue:
                logger.info(
                    "Research queue captured %d blocked opportunities",
                    len(research_queue),
                    data={"research_queue": list(research_queue)},
                )
            if execution_candidates == 0:
                consecutive_zero_execution_yield_cycles += 1
            else:
                consecutive_zero_execution_yield_cycles = 0
            if execution_candidates == 0 and len(research_queue) > 50:
                top_tiers = sorted(
                    participation_tier_breakdown.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:3]
                yield_alert_payload = {
                    "cycle_yield_alert": True,
                    "research_queue_size": len(research_queue),
                    "consecutive_zero_execution_yield_cycles": (
                        consecutive_zero_execution_yield_cycles
                    ),
                    "consecutive_zero_execution_yield_threshold": (
                        settings.CYCLE_YIELD_ALERT_ESCALATE_AFTER
                    ),
                    "participation_tier_breakdown": participation_tier_breakdown,
                }
                # Sustained zero-execution yield is a calibration signal, not a
                # thrown runtime error. Keep it out of predictbot_errors.log
                # while still tagging sustained alerts for dashboards.
                escalate_to_error = (
                    settings.CYCLE_YIELD_ALERT_ESCALATE_AFTER > 0
                    and consecutive_zero_execution_yield_cycles
                    >= settings.CYCLE_YIELD_ALERT_ESCALATE_AFTER
                )
                if escalate_to_error:
                    logger.warning(
                        "Cycle yield alert (sustained, %d cycles): 0 execution candidates "
                        "with %d research-queued; top tiers: %s — investigate gate calibration",
                        consecutive_zero_execution_yield_cycles,
                        len(research_queue),
                        ", ".join(f"{t}={c}" for t, c in top_tiers),
                        data=yield_alert_payload,
                    )
                else:
                    logger.warning(
                        "Cycle yield alert: 0 execution candidates with %d research-queued; "
                        "top tiers: %s — investigate gate calibration",
                        len(research_queue),
                        ", ".join(f"{t}={c}" for t, c in top_tiers),
                        data=yield_alert_payload,
                    )
            if settings.CALIBRATION_MODE_ENABLED and calibration_samples:
                recommendation = compute_adaptive_thresholds(
                    samples=calibration_samples,
                    current_edge_threshold=settings.MIN_EDGE,
                    current_spread_cutoff=_ORDERBOOK_SPREAD_CUTOFF_DEFAULT,
                    current_workers=settings.ANALYSIS_MAX_WORKERS,
                    min_samples=settings.CALIBRATION_MIN_SAMPLES,
                )
                logger.info(
                    "Calibration recommendation snapshot: edge=%.4f spread_cutoff=%.4f workers=%d samples=%d",
                    recommendation["recommended_min_market_edge_for_trade"],
                    recommendation["recommended_orderbook_spread_cutoff"],
                    recommendation["recommended_analysis_max_workers"],
                    recommendation["sample_count"],
                    data=recommendation,
                )
            logger.info(
                "Cycle funnel: fetched=%d filtered=%d skipped_resolved=%d skipped_likely_resolved_by_ticker=%d scheduler_skips=%d "
                "(closed=%d recently=%d other=%d) "
                "analyzed=%d refined=%d flip_precheck_skipped=%d flip_guard_triggered=%d "
                "flip_guard_blocked=%d execution_candidates=%d research_queue_size=%d should_trade_blocked=%d order_attempts=%d "
                "skipped_kelly_sub_floor=%d tiers=%s",
                fetched_count,
                len(markets),
                filter_stats.get("skipped_resolved", 0),
                filter_stats.get("skipped_likely_resolved_by_ticker", 0),
                scheduler_skipped_closed + scheduler_skipped_recently + scheduler_skipped_other,
                scheduler_skipped_closed,
                scheduler_skipped_recently,
                scheduler_skipped_other,
                markets_analyzed,
                markets_refined,
                flip_precheck_skipped_refinement,
                flip_guard_triggered,
                flip_guard_blocked,
                execution_candidates,
                research_queue_size,
                should_trade_but_blocked,
                trades_attempted,
                trades_skipped_kelly_sub_floor,
                _format_tier_breakdown_for_log(participation_tier_breakdown),
                data={
                    "fetched": fetched_count,
                    "filtered": len(markets),
                    "skipped_resolved": filter_stats.get("skipped_resolved", 0),
                    "skipped_likely_resolved_by_ticker": filter_stats.get(
                        "skipped_likely_resolved_by_ticker",
                        0,
                    ),
                    "scheduler_skipped_closed": scheduler_skipped_closed,
                    "scheduler_skipped_recently_analyzed": scheduler_skipped_recently,
                    "scheduler_skipped_other": scheduler_skipped_other,
                    "analyzed": markets_analyzed,
                    "refined": markets_refined,
                    "parallel_analysis_requested": parallel_analysis_requested,
                    "parallel_analysis_used": parallel_analysis_used,
                    "analysis_candidates": analysis_candidates_count,
                    "analysis_workers": analysis_worker_count,
                    "analysis_phase_duration_ms": analysis_phase_duration_ms,
                    "flip_precheck_skipped_refinement": flip_precheck_skipped_refinement,
                    "flip_guard_triggered": flip_guard_triggered,
                    "flip_guard_blocked": flip_guard_blocked,
                    "outcome_mismatch_blocked": outcome_mismatch_blocked,
                    "execution_candidates": execution_candidates,
                    "projected_daily_expected_value_usdc": round(
                        cycle_projected_daily_ev_usdc,
                        4,
                    ),
                    "daily_expectancy_primary_targets": cycle_primary_targets_selected,
                    "daily_expectancy_satellites": cycle_satellites_selected,
                    "research_queue_size": research_queue_size,
                    "research_queue_drained_count": research_queue_drained_count,
                    "research_queue_drain_skipped_stale_count": research_queue_drain_skipped_stale_count,
                    "research_queue_drain_skipped_low_priority_count": (
                        research_queue_drain_skipped_low_priority_count
                    ),
                    "research_queue_emergency_probes_count": (
                        research_queue_emergency_probes_count
                    ),
                    "research_queue_zero_yield_promotions_count": (
                        research_queue_zero_yield_promotions_count
                    ),
                    "effective_research_band": round(effective_research_band, 4),
                    "research_band_widened_by": round(research_band_widened_by, 4),
                    "pre_score_distribution": _summarize_distribution(
                        cycle_pre_score_samples
                    ),
                    "soft_research_threshold_gap_distribution": (
                        _summarize_distribution(
                            cycle_soft_research_threshold_gap_samples
                        )
                    ),
                    "daily_drawdown_preflight_blocked_count": daily_drawdown_preflight_blocked_count,
                    "should_trade_but_blocked": should_trade_but_blocked,
                    "should_trade_blocked_breakdown": should_trade_blocked_breakdown,
                    "blocked_direct_evidence_count": blocked_direct_evidence_count,
                    "order_attempts": trades_attempted,
                    "orders_filled": trades_filled,
                    "orders_canceled_unfilled": trades_canceled_unfilled,
                    "total_usd_deployed": round(total_usd_deployed, 2),
                    "execution_family_breakdown": execution_family_breakdown,
                    "skipped_kelly_sub_floor": trades_skipped_kelly_sub_floor,
                    "pre_analysis_blocked": pre_analysis_blocked,
                    "pre_analysis_rejection_breakdown": pre_analysis_rejection_breakdown,
                    "participation_tier_breakdown": dict(sorted(participation_tier_breakdown.items())),
                    "definitive_outcome_floor_applied_count": definitive_outcome_floor_applied_count,
                    "timeout_routed_to_monitor_only_count": timeout_routed_to_monitor_only_count,
                    "negative_best_score_skipped_count": negative_best_score_skipped_count,
                    "api_tokens_consumed": api_tokens_consumed,
                    "api_cost_estimate_usd": round(api_cost_estimate_usd, 6),
                    "best_candidate_market_id": best_candidate_market_id,
                    "best_candidate_score": best_candidate_score,
                    "evidence_basis_breakdown": dict(sorted(evidence_basis_breakdown.items())),
                    "per_family_edge_p50": per_family_edge_p50,
                    "per_family_edge_p90": per_family_edge_p90,
                    "mean_raw_vs_calibrated_confidence_delta": round(
                        mean_confidence_delta,
                        4,
                    ),
                    "markets_considered": len(markets),
                    "markets_filtered": len(markets),
                },
            )
            logger.info(
                "Bot cycle #%d complete: duration=%.2fms, markets=%d, trades_attempted=%d, skipped_balance=%d%s",
                cycle_count,
                cycle_duration,
                len(markets),
                trades_attempted,
                trades_skipped_balance,
                mode_suffix,
                data={
                    "cycle": cycle_count,
                    "duration_ms": round(cycle_duration, 2),
                    "filtered_markets": len(markets),
                    "markets_analyzed": markets_analyzed,
                    "markets_fetched": fetched_count,
                    "filter_stats": filter_stats,
                    "skipped_resolved": filter_stats.get("skipped_resolved", 0),
                    "scheduler_skipped_closed": scheduler_skipped_closed,
                    "scheduler_skipped_recently_analyzed": scheduler_skipped_recently,
                    "scheduler_skipped_other": scheduler_skipped_other,
                    "markets_passed_to_grok": markets_analyzed,
                    "markets_refined": markets_refined,
                    "parallel_analysis_requested": parallel_analysis_requested,
                    "parallel_analysis_used": parallel_analysis_used,
                    "analysis_candidates": analysis_candidates_count,
                    "analysis_workers": analysis_worker_count,
                    "analysis_phase_duration_ms": analysis_phase_duration_ms,
                    "flip_precheck_skipped_refinement": flip_precheck_skipped_refinement,
                    "flip_guard_triggered": flip_guard_triggered,
                    "flip_guard_blocked": flip_guard_blocked,
                    "outcome_mismatch_blocked": outcome_mismatch_blocked,
                    "execution_candidates": execution_candidates,
                    "research_queue_size": research_queue_size,
                    "score_gate_blocked": score_gate_blocked,
                    "decisions_made": decisions_made,
                    "order_attempts": trades_attempted,
                    "orders_filled": trades_filled,
                    "orders_canceled_unfilled": trades_canceled_unfilled,
                    "total_usd_deployed": round(total_usd_deployed, 2),
                    "pre_analysis_rejection_breakdown": pre_analysis_rejection_breakdown,
                    "skipped_no_trade": trades_skipped_no_trade,
                    "skipped_confidence": trades_skipped_confidence,
                    "skipped_edge": trades_skipped_edge,
                    "skipped_kelly_sub_floor": trades_skipped_kelly_sub_floor,
                    "skipped_balance": trades_skipped_balance,
                    "skipped_position": trades_skipped_position,
                    "analysis_only_mode": analysis_only_mode,
                    "price_buckets": price_bucket_stats,
                    "api_tokens_consumed": api_tokens_consumed,
                    "api_cost_estimate_usd": round(api_cost_estimate_usd, 6),
                    "best_candidate_market_id": best_candidate_market_id,
                    "best_candidate_score": best_candidate_score,
                    "evidence_basis_breakdown": dict(sorted(evidence_basis_breakdown.items())),
                    "per_family_edge_p50": per_family_edge_p50,
                    "per_family_edge_p90": per_family_edge_p90,
                    "blocked_direct_evidence_count": blocked_direct_evidence_count,
                    "mean_raw_vs_calibrated_confidence_delta": round(
                        mean_confidence_delta,
                        4,
                    ),
                },
            )
            if trades_attempted > 0:
                consecutive_zero_order_cycles = 0
            else:
                consecutive_zero_order_cycles += 1
            dry_streak_sleep_seconds = _dry_streak_sleep_seconds(
                base_poll_interval_sec=settings.POLL_INTERVAL_SEC,
                consecutive_zero_order_cycles=consecutive_zero_order_cycles,
                enabled=settings.DRY_STREAK_SLEEP_ENABLED,
            )
            if dry_streak_sleep_seconds is not None:
                if dry_streak_sleep_seconds > sleep_seconds:
                    sleep_seconds = dry_streak_sleep_seconds
                logger.info(
                    "Dry-streak sleep applied: streak=%d sleep_seconds=%d",
                    consecutive_zero_order_cycles,
                    sleep_seconds,
                    data={
                        "consecutive_zero_order_cycles": consecutive_zero_order_cycles,
                        "dry_streak_sleep_seconds": sleep_seconds,
                    },
                )
            if markets_analyzed == 0 and scheduler_skipped_recently > 0:
                adaptive_seconds = _compute_next_wakeup_seconds(
                    markets=markets,
                    state_manager=state_manager,
                    settings=settings,
                )
                if adaptive_seconds is not None:
                    sleep_seconds = adaptive_seconds
                    logger.debug(
                        "Adaptive sleep selected: %ds (recently analyzed markets)",
                        sleep_seconds,
                        data={
                            "sleep_seconds": sleep_seconds,
                            "scheduler_skipped_recently_analyzed": scheduler_skipped_recently,
                            "cap_seconds": _ADAPTIVE_SLEEP_CAP_SECONDS,
                        },
                    )

        except Exception as exc:
            error_text = str(exc)
            if "Could not find a suitable TLS CA certificate bundle" in error_text:
                certifi_path = None
                if certifi is not None:
                    try:
                        certifi_path = certifi.where()
                    except Exception:
                        certifi_path = None
                logger.error(
                    "Bot cycle #%d TLS CA bundle error: %s",
                    cycle_count,
                    error_text,
                    data={
                        "cycle": cycle_count,
                        "error": error_text,
                        "error_type": type(exc).__name__,
                        "python_executable": sys.executable,
                        "certifi_where": certifi_path,
                    },
                )
                sleep_seconds = max(60, int(sleep_seconds))
                continue
            logger.exception(
                "Bot cycle #%d failed: %s",
                cycle_count,
                exc,
                data={"cycle": cycle_count, "error": str(exc), "error_type": type(exc).__name__},
            )

        logger.debug(
            "Sleeping for %d seconds before next cycle",
            sleep_seconds,
            data={"sleep_seconds": sleep_seconds, "cycle_id": cycle_id},
        )
        time.sleep(sleep_seconds)
        if max_cycles is not None and cycle_count >= max_cycles:
            logger.info(
                "Reached max cycles (%d/%d) - shutting down",
                cycle_count,
                max_cycles,
                data={"cycle_count": cycle_count, "max_cycles": max_cycles},
            )
            break


if __name__ == "__main__":
    main()
