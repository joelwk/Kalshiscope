from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import re
from typing import Any

from bayesian_engine import (
    BayesianState,
    initial_state,
    log_likelihood_from_ratio,
    posterior_from_state,
)
from calibration import build_counterfactual_flags, compute_adaptive_thresholds
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
from score_engine import calibrate_confidence, compute_final_score
from xai_provider import XAIProvider

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
_WEATHER_BIN_TICKER_PATTERN = re.compile(r"-B\d", re.IGNORECASE)
_PRE_ANALYSIS_HARD_REJECTION_TERMINAL_OUTCOMES = {
    "no_trade_recommended",
    "evidence_quality_below_min",
    "confidence_below_min",
    "abstain_low_evidence",
}


def _normalize_outcome_key(outcome: str | None) -> str:
    return re.sub(r"\s+", " ", (outcome or "").strip()).lower()


def _is_retriable_xai_error(error_text: str | None) -> bool:
    normalized = (error_text or "").strip().lower()
    if not normalized:
        return False
    return any(marker in normalized for marker in _XAI_RETRIABLE_ERROR_MARKERS)


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


def _edge_threshold_for_market(
    implied_prob: float,
    settings: Settings,
    edge_source: str | None = None,
    market: Market | None = None,
) -> float:
    min_edge = settings.MIN_EDGE
    is_weather_market = market is not None and market_family(market) == "weather"
    if is_weather_market:
        min_edge = max(min_edge, settings.WEATHER_MIN_EDGE)
    if implied_prob < settings.VERY_LOW_PRICE_THRESHOLD:
        min_edge = max(min_edge, settings.VERY_LOW_PRICE_MIN_EDGE)
    if implied_prob < settings.LOW_PRICE_THRESHOLD:
        min_edge = max(min_edge, settings.LOW_PRICE_MIN_EDGE)
    if settings.COINFLIP_PRICE_LOWER <= implied_prob <= settings.COINFLIP_PRICE_UPPER:
        min_edge = max(min_edge, settings.LOW_PRICE_MIN_EDGE)
    if (edge_source or "").lower() == "fallback":
        min_edge = max(min_edge, settings.FALLBACK_EDGE_MIN_EDGE)
        if is_weather_market:
            min_edge = max(min_edge, settings.WEATHER_FALLBACK_EDGE_MIN_EDGE)
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
    # Use uncapped model confidence for edge gating so category caps only affect sizing.
    effective_confidence = (
        decision.raw_confidence
        if decision.raw_confidence is not None
        else decision.confidence
    )
    edge = effective_confidence - implied_prob
    min_edge = _edge_threshold_for_market(
        implied_prob,
        settings,
        decision.edge_source,
        market=market,
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


def _historical_win_rate_at_bucket(confidence: float) -> float | None:
    rounded = round(max(0.0, min(1.0, confidence)) * 10.0) / 10.0
    return _HISTORICAL_WIN_RATE_BY_BUCKET.get(rounded)


def _min_evidence_quality_for_market(market: Market, settings: Settings) -> float:
    if market_family(market) == "weather":
        return settings.WEATHER_MIN_EVIDENCE_QUALITY
    return settings.MIN_EVIDENCE_QUALITY_FOR_TRADE


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
    if not reasoning:
        reasoning = "Previous cycle analysis."
    return TradeDecision(
        should_trade=False,
        outcome=outcome,
        confidence=max(0.0, min(1.0, confidence)),
        bet_size_pct=0.0,
        reasoning=reasoning,
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
            "min_liquidity_usdc": settings.MIN_LIQUIDITY_USDC,
            "min_volume_24h": settings.MIN_VOLUME_24H,
            "min_open_interest": settings.MIN_OPEN_INTEREST,
            "min_tradeable_implied_price": settings.MIN_TRADEABLE_IMPLIED_PRICE,
            "max_tradeable_implied_price": settings.MAX_TRADEABLE_IMPLIED_PRICE,
            "poll_interval_sec": settings.POLL_INTERVAL_SEC,
            "market_min_close_days": settings.MARKET_MIN_CLOSE_DAYS,
            "market_max_close_days": settings.MARKET_MAX_CLOSE_DAYS,
            "grok_model": settings.GROK_MODEL,
            "categories_allowlist": settings.MARKET_CATEGORIES_ALLOWLIST,
            "categories_blocklist": settings.MARKET_CATEGORIES_BLOCKLIST,
            "ticker_prefix_blocklist": settings.MARKET_TICKER_BLOCKLIST_PREFIXES,
            "skip_weather_bin_markets": settings.SKIP_WEATHER_BIN_MARKETS,
            "crypto_bin_market_blocklist_enabled": settings.CRYPTO_BIN_MARKET_BLOCKLIST_ENABLED,
            "max_weather_candidates_per_cycle": settings.MAX_WEATHER_CANDIDATES_PER_CYCLE,
            "max_crypto_candidates_per_cycle": settings.MAX_CRYPTO_CANDIDATES_PER_CYCLE,
            "max_speech_candidates_per_cycle": settings.MAX_SPEECH_CANDIDATES_PER_CYCLE,
            "max_music_candidates_per_cycle": settings.MAX_MUSIC_CANDIDATES_PER_CYCLE,
            "weather_min_evidence_quality": settings.WEATHER_MIN_EVIDENCE_QUALITY,
            "weather_fallback_edge_min_edge": settings.WEATHER_FALLBACK_EDGE_MIN_EDGE,
            "kalshi_server_side_filters_enabled": settings.KALSHI_SERVER_SIDE_FILTERS_ENABLED,
            "kalshi_max_fetch_pages": settings.KALSHI_MAX_FETCH_PAGES,
            "score_gate_mode": settings.SCORE_GATE_MODE,
            "score_gate_threshold": settings.SCORE_GATE_THRESHOLD,
            "score_computed_edge_bonus": settings.SCORE_COMPUTED_EDGE_BONUS,
            "score_repeated_analysis_penalty_base": settings.SCORE_REPEATED_ANALYSIS_PENALTY_BASE,
            "score_repeated_analysis_penalty_start_count": settings.SCORE_REPEATED_ANALYSIS_PENALTY_START_COUNT,
            "score_confidence_calibration_floor": settings.SCORE_CONFIDENCE_CALIBRATION_FLOOR,
            "score_confidence_calibration_penalty_scale": settings.SCORE_CONFIDENCE_CALIBRATION_PENALTY_SCALE,
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
) -> list[Market]:
    if not use_server_side_filters:
        return kalshi_client.get_markets()
    try:
        return kalshi_client.get_markets(
            close_time_start=fetch_window_start,
            close_time_end=fetch_window_end,
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
            },
        )
        kalshi_client.reset_session()
        try:
            return kalshi_client.get_markets(
                close_time_start=fetch_window_start,
                close_time_end=fetch_window_end,
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
                },
            )
            return kalshi_client.get_markets()


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
) -> tuple[bool, float]:
    override_min_confidence = max(
        0.0,
        min(1.0, settings.CONFIDENCE_GATE_OVERRIDE_MIN_CONFIDENCE),
    )
    allowed = (
        settings.CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED
        and override_edge is not None
        and override_edge >= settings.CONFIDENCE_GATE_MIN_EDGE
        and decision.evidence_quality >= settings.CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY
        and decision.confidence >= override_min_confidence
    )
    return allowed, override_min_confidence


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
    limit: int = 200,
) -> int:
    payload = kalshi_client.get_settlements(limit=limit)
    rows = _iter_exchange_settlement_rows(payload)
    imported = 0
    for row in rows:
        parsed = _parse_exchange_settlement_row(row)
        if parsed is None:
            continue
        state_manager.record_exchange_settlement(**parsed)
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


def _analysis_result_rank(result: dict[str, Any] | None) -> tuple[float, float, float, float]:
    if not result:
        return (0.0, 0.0, 0.0, 0.0)
    decision = result.get("decision")
    if not isinstance(decision, TradeDecision):
        return (0.0, 0.0, 0.0, 0.0)
    should_trade_rank = 1.0 if decision.should_trade and not decision.abstain else 0.0
    final_score_rank = float(result.get("pre_execution_final_score", 0.0) or 0.0)
    score_rank = final_score_rank + (0.02 * should_trade_rank)
    evidence_rank = max(0.0, min(1.0, decision.evidence_quality))
    confidence_rank = max(0.0, min(1.0, decision.confidence))
    return (score_rank, evidence_rank, confidence_rank, should_trade_rank)


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
    if decision_phase is not None:
        payload["decision_phase"] = decision_phase
    if decision_terminal is not None:
        payload["decision_terminal"] = decision_terminal
    if final_action is not None:
        payload["final_action"] = final_action
    if final_reason is not None:
        payload["final_reason"] = final_reason
        if str(final_action or "").strip().lower() == "skip":
            payload["rejection_reason"] = final_reason
    for key, value in extra.items():
        if value is not None:
            payload[key] = value
    if final_reason and "rejection_stage" not in payload:
        if str(final_reason).startswith("pre_analysis_"):
            payload["rejection_stage"] = "pre_analysis"
        elif str(final_reason) in {"no_trade_recommended", "abstain_low_evidence"}:
            payload["rejection_stage"] = "validation"
        elif str(final_reason) in {"score_gate_blocked"}:
            payload["rejection_stage"] = "score_gate"
        elif str(final_reason).endswith("_blocked") or str(final_reason).endswith("_below_min"):
            payload["rejection_stage"] = "execution_gate"
    for alias_key, canonical_key in alias_to_canonical.items():
        if alias_key in payload and canonical_key not in payload:
            payload[canonical_key] = payload[alias_key]
        payload.pop(alias_key, None)
    return payload


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
        "score_proxy_evidence_penalty": float(
            getattr(score_result, "proxy_evidence_penalty", 0.0) or 0.0
        ),
        "score_generic_bin_penalty": float(
            getattr(score_result, "generic_bin_penalty", 0.0) or 0.0
        ),
        "score_ambiguous_resolution_penalty": float(
            getattr(score_result, "ambiguous_resolution_penalty", 0.0) or 0.0
        ),
        "score_weather_penalty": weather_penalty,
        "score_bayesian_posterior": getattr(score_result, "bayesian_posterior", None),
        "score_lmsr_price": getattr(score_result, "lmsr_price", None),
        "score_inefficiency_signal": getattr(score_result, "inefficiency_signal", None),
        "score_kelly_raw": getattr(score_result, "kelly_raw", None),
        "score_rejection_reasons": list(getattr(score_result, "rejection_reasons", ()) or ()),
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
        "evidence_basis_class": evidence_basis_class,
        "edge_source": edge_source,
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
) -> int | None:
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


def _pre_analysis_opportunity_score(
    market: Market,
    state: MarketState | None,
    settings: Settings,
    traded_before: bool,
    fallback_family_edge_rate: float | None = None,
    fallback_family_sample_size: int = 0,
    historical_family_stats: dict[str, float | int] | None = None,
) -> tuple[float, dict[str, float]]:
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
            historical_pnl_penalty_base = max(
                0.0,
                settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_PENALTY,
            )
            pnl_threshold_abs = max(0.01, abs(historical_pnl_threshold))
            historical_family_pnl_ratio = max(
                1.0,
                abs(historical_family_pnl) / pnl_threshold_abs,
            )
            historical_family_pnl_penalty = min(
                0.25,
                historical_pnl_penalty_base * historical_family_pnl_ratio,
            )
            severe_pnl_threshold = float(
                settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_SEVERE_THRESHOLD
            )
            if historical_family_pnl <= severe_pnl_threshold:
                historical_family_pnl_penalty = max(
                    historical_family_pnl_penalty,
                    max(0.0, settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PNL_SEVERE_PENALTY),
                )
                historical_family_pnl_penalty = min(0.25, historical_family_pnl_penalty)
    if (
        fallback_family_penalty > 0.0
        and historical_family_pnl > 0.0
        and historical_family_win_rate > 0.55
    ):
        fallback_family_penalty *= 0.5
        fallback_family_penalty_scale = 0.5
    ambiguous_resolution_penalty = 0.0
    if not (market.resolution_criteria or "").strip():
        ambiguous_resolution_penalty = 0.08
    churn_penalty = 0.0
    if analysis_count >= max(6, settings.PRE_ANALYSIS_HARD_REJECTION_MIN_ANALYSES // 2):
        churn_penalty = 0.05
        if not traded_before:
            churn_penalty += 0.03
    score = (
        (0.40 * price_center_score)
        + (0.35 * liquidity_score)
        + (0.25 * horizon_score)
        + post_event_bonus
        - repeated_analysis_penalty
        - non_actionable_penalty
        - family_penalty
        - generic_bin_penalty
        - crypto_bin_penalty
        - fallback_family_penalty
        - historical_family_penalty
        - historical_family_pnl_penalty
        - ambiguous_resolution_penalty
        - churn_penalty
        - coinflip_penalty
    )
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
        "pre_score_ambiguous_resolution_penalty": ambiguous_resolution_penalty,
        "pre_score_churn_penalty": churn_penalty,
        "pre_score_coinflip_penalty": coinflip_penalty,
        "pre_score_analysis_count": float(analysis_count),
        "pre_score_non_actionable_streak": float(non_actionable_streak),
        "pre_score_traded_before": 1.0 if traded_before else 0.0,
        "pre_score_hours_to_close": (
            float(raw_hours_to_close) if raw_hours_to_close is not None else 0.0
        ),
    }


def _pre_analysis_hard_rejection(
    *,
    market: Market,
    state: MarketState | None,
    settings: Settings,
    traded_before: bool,
    had_recent_fallback_edge: bool = False,
    family_action_stats: dict[str, float | int] | None = None,
    historical_family_stats: dict[str, float | int] | None = None,
    fallback_family_edge_rate: float | None = None,
    fallback_family_sample_size: int = 0,
) -> tuple[bool, str | None, dict[str, Any]]:
    if not settings.PRE_ANALYSIS_HARD_REJECTION_ENABLED or state is None:
        return False, None, {}
    analysis_count = int(state.analysis_count or 0)
    non_actionable_streak = int(state.non_actionable_streak or 0)
    if (
        non_actionable_streak >= 3
        and had_recent_fallback_edge
        and not traded_before
    ):
        metadata = {
            "pre_analysis_hard_reject": True,
            "pre_analysis_hard_reject_reason": "fallback_edge_high_churn",
            "pre_analysis_hard_reject_non_actionable_streak": non_actionable_streak,
            "pre_analysis_hard_reject_analysis_count": analysis_count,
            "pre_analysis_hard_reject_traded_before": traded_before,
            "pre_analysis_hard_reject_had_recent_fallback_edge": had_recent_fallback_edge,
        }
        return True, "pre_analysis_fallback_edge_high_churn", metadata
    family = market_family(market)
    rejection_families = {
        str(name or "").strip().lower()
        for name in settings.PRE_ANALYSIS_HARD_REJECTION_FAMILIES
    }
    if family not in rejection_families:
        return False, None, {}
    if settings.PRE_ANALYSIS_ZERO_ACTION_FAMILY_BLOCK_ENABLED:
        action_stats = family_action_stats or {}
        family_sample_size = int(action_stats.get("sample_size", 0) or 0)
        family_action_rate = float(action_stats.get("action_rate", 0.0) or 0.0)
        if (
            family_sample_size >= max(1, settings.PRE_ANALYSIS_ZERO_ACTION_FAMILY_MIN_SAMPLES)
            and family_action_rate <= 0.0
            and not traded_before
        ):
            metadata = {
                "pre_analysis_hard_reject": True,
                "pre_analysis_hard_reject_reason": "zero_action_family",
                "pre_analysis_hard_reject_family": family,
                "pre_analysis_hard_reject_family_sample_size": family_sample_size,
                "pre_analysis_hard_reject_family_action_rate": family_action_rate,
            }
            return True, "pre_analysis_zero_action_family", metadata
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
                "pre_analysis_hard_reject": True,
                "pre_analysis_hard_reject_reason": "crypto_historically_unprofitable",
                "pre_analysis_hard_reject_family": family,
                "pre_analysis_hard_reject_historical_pnl": historical_pnl_total,
                "pre_analysis_hard_reject_historical_samples": historical_sample_size,
                "pre_analysis_hard_reject_fallback_rate": fallback_rate,
                "pre_analysis_hard_reject_fallback_samples": fallback_samples,
            }
            return True, "pre_analysis_crypto_historically_unprofitable", metadata
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
        hard_reject_reason = "repeated_non_actionable_market"
        if family in {"generic", "crypto"} and _WEATHER_BIN_TICKER_PATTERN.search(market.id or ""):
            hard_reject_reason = "repeated_non_actionable_bin_market"
        metadata = {
            "pre_analysis_hard_reject": True,
            "pre_analysis_hard_reject_reason": hard_reject_reason,
            "pre_analysis_hard_reject_family": family,
            "pre_analysis_hard_reject_terminal_outcome": terminal_outcome,
            "pre_analysis_hard_reject_non_actionable_streak": non_actionable_streak,
            "pre_analysis_hard_reject_analysis_count": analysis_count,
            "pre_analysis_hard_reject_traded_before": traded_before,
        }
        return True, f"pre_analysis_{hard_reject_reason}", metadata
    if analysis_count >= 4 and non_actionable_streak >= 3 and not traded_before:
        metadata = {
            "pre_analysis_hard_reject": True,
            "pre_analysis_hard_reject_reason": "repeated_churn_market",
            "pre_analysis_hard_reject_non_actionable_streak": non_actionable_streak,
            "pre_analysis_hard_reject_analysis_count": analysis_count,
            "pre_analysis_hard_reject_traded_before": traded_before,
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
    pre_scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Apply a hard cap to per-cycle candidates with family stratification."""
    if max_markets_per_cycle <= 0:
        return []
    if len(analysis_candidates) <= max_markets_per_cycle:
        return analysis_candidates

    grouped: dict[str, list[dict[str, Any]]] = {}
    family_order: list[str] = []
    for candidate in analysis_candidates:
        market = candidate.get("market")
        if not isinstance(market, Market):
            continue
        family = market_family(market)
        if family not in grouped:
            grouped[family] = []
            family_order.append(family)
        grouped[family].append(candidate)

    for family, family_candidates in grouped.items():
        family_candidates.sort(
            key=lambda candidate: (
                -float(
                    (pre_scores or {}).get(
                        str(getattr(candidate.get("market"), "id", "")),
                        candidate.get("pre_analysis_score") or 0.0,
                    )
                ),
                int(candidate.get("non_actionable_streak", 0)),
                str(getattr(candidate.get("market"), "id", "")),
            )
        )

    if not grouped:
        return analysis_candidates[:max_markets_per_cycle]

    selected: list[dict[str, Any]] = []
    selected_weather_count = 0
    selected_crypto_count = 0
    selected_speech_count = 0
    selected_music_count = 0
    while len(selected) < max_markets_per_cycle:
        progressed = False
        for family in family_order:
            family_candidates = grouped.get(family)
            if not family_candidates:
                continue
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
            selected.append(family_candidates.pop(0))
            if family == "weather":
                selected_weather_count += 1
            elif family == "crypto":
                selected_crypto_count += 1
            elif family == "speech":
                selected_speech_count += 1
            elif family == "music":
                selected_music_count += 1
            progressed = True
            if len(selected) >= max_markets_per_cycle:
                break
        if not progressed:
            break
    return selected


def _resolve_dynamic_analysis_candidate_cap(
    *,
    settings: Settings,
    best_pre_analysis_score: float,
) -> tuple[int, bool]:
    dynamic_max_markets_per_cycle = settings.MAX_MARKETS_PER_CYCLE
    reduced_candidate_cap_applied = False
    if best_pre_analysis_score < settings.PRE_ANALYSIS_MUST_ANALYZE_THRESHOLD:
        dynamic_max_markets_per_cycle = min(
            dynamic_max_markets_per_cycle,
            max(1, settings.PRE_ANALYSIS_REDUCED_MAX_CANDIDATES),
        )
        reduced_candidate_cap_applied = True
    return dynamic_max_markets_per_cycle, reduced_candidate_cap_applied


def _build_speech_reanalysis_search_config(base_config: SearchConfig) -> SearchConfig:
    """Expand lookback and rotate sources for low-evidence speech reanalysis."""
    now = datetime.now(timezone.utc)
    base_lookback_hours = base_config.lookback_hours or 24
    expanded_lookback_hours = max(base_lookback_hours, base_lookback_hours * 2)
    rotated_domains = (
        [*base_config.allowed_domains[1:], base_config.allowed_domains[0]]
        if len(base_config.allowed_domains) > 1
        else list(base_config.allowed_domains)
    )
    rotated_handles = (
        [*base_config.allowed_x_handles[1:], base_config.allowed_x_handles[0]]
        if len(base_config.allowed_x_handles) > 1
        else list(base_config.allowed_x_handles)
    )
    return SearchConfig(
        from_date=now - timedelta(hours=expanded_lookback_hours),
        to_date=now,
        allowed_domains=rotated_domains,
        allowed_x_handles=rotated_handles,
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
) -> dict[str, Any]:
    """Run analysis/refinement/guardrails for a market candidate."""
    previous_analysis = _build_previous_analysis(anchor_analysis)
    search_config = build_market_search_config(settings, market)
    try:
        decision = grok_client.analyze_market(
            market,
            search_config=search_config,
            previous_analysis=previous_analysis,
        )
    except Exception as exc:
        error_text = str(exc)
        logger.error(
            "Initial market analysis failed for %s: %s",
            market.id,
            exc,
            data={
                "market_id": market.id,
                "error": error_text,
                "error_type": type(exc).__name__,
                "analysis_phase": "initial",
            },
        )
        return {
            "analysis_failed": True,
            "analysis_error": error_text,
            "analysis_error_type": type(exc).__name__,
            "analysis_error_retriable_xai": _is_retriable_xai_error(error_text),
            "was_refined": False,
            "refinement_reason_text": None,
            "flip_triggered": False,
            "flip_blocked": False,
            "refinement_skipped_by_flip_precheck": False,
            "flip_precheck_reason": None,
            "market_outcome_mismatch_counted": False,
        }

    anchor_outcome: str | None = None
    if anchor_analysis and anchor_analysis.get("outcome") is not None:
        anchor_outcome = str(anchor_analysis["outcome"]).strip() or None

    refinement = RefinementStrategy(
        market=market,
        urgent_days_before_close=settings.URGENT_REANALYSIS_DAYS_BEFORE_CLOSE,
    )
    was_refined = False
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
    refinement_reason_text = ",".join(refinement_reasons) if refinement_reasons else None
    if refinement_reasons:
        refinement_search_config = search_config
        if (
            search_config.profile_name == "speech"
            and decision.evidence_quality < 0.5
        ):
            refinement_search_config = _build_speech_reanalysis_search_config(
                search_config,
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
                market,
                decision,
                search_config=refinement_search_config,
            )
            was_refined = True

    decision = _cap_confidence_for_category(decision, market, settings)
    confidence_before_calibration = decision.confidence
    evidence_basis_for_calibration = _decision_evidence_basis(decision)
    definitive_outcome_for_calibration = bool(
        getattr(decision, "definitive_outcome_detected", False)
    )
    calibrated_confidence = calibrate_confidence(
        decision.confidence,
        shrinkage_floor=settings.CONFIDENCE_SHRINKAGE_FLOOR,
        shrinkage_factor=settings.CONFIDENCE_SHRINKAGE_FACTOR,
        family_shrinkage_override=_confidence_shrinkage_override_for_market(market),
        evidence_basis_class=evidence_basis_for_calibration,
        definitive_outcome=definitive_outcome_for_calibration,
    )
    calibration_delta = confidence_before_calibration - calibrated_confidence
    historical_win_rate_at_bucket = _historical_win_rate_at_bucket(confidence_before_calibration)
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
    decision, flip_triggered, flip_blocked = _apply_flip_guard(
        market,
        decision,
        anchor_analysis,
        settings,
    )
    market_outcome_mismatch_counted = "[Outcome mismatch]" in (decision.reasoning or "")
    return {
        "decision": decision,
        "was_refined": was_refined,
        "refinement_reason_text": refinement_reason_text,
        "flip_triggered": flip_triggered,
        "flip_blocked": flip_blocked,
        "refinement_skipped_by_flip_precheck": refinement_skipped_by_flip_precheck,
        "flip_precheck_reason": flip_precheck_reason,
        "market_outcome_mismatch_counted": market_outcome_mismatch_counted,
        "confidence_before_calibration": confidence_before_calibration,
        "confidence_after_calibration": decision.confidence,
        "confidence_calibration_applied": confidence_calibration_applied,
        "raw_vs_calibrated_delta": calibration_delta,
        "historical_win_rate_at_bucket": historical_win_rate_at_bucket,
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
    scheduler = MarketScheduler(
        reanalysis_cooldown_hours=settings.REANALYSIS_COOLDOWN_HOURS,
        urgent_days_before_close=settings.URGENT_REANALYSIS_DAYS_BEFORE_CLOSE,
        urgent_reanalysis_cooldown_hours=settings.URGENT_REANALYSIS_COOLDOWN_HOURS,
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
            )
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

            markets = _collapse_event_ladders(
                markets,
                ladder_collapse_threshold=settings.LADDER_COLLAPSE_THRESHOLD,
                max_brackets_per_event=settings.MAX_BRACKETS_PER_EVENT,
            )
            markets = _dedupe_markets_by_matchup(markets)

            markets = scheduler.prioritize_markets(markets, state_manager)

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
                        _update_resolved_markets(markets, state_manager, kalshi_client)
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
            trades_skipped_kelly_sub_floor = 0
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
            rejection_funnel_summary: list[dict[str, Any]] = []
            pre_analysis_rejection_breakdown: dict[str, int] = {}
            execution_family_stats: dict[str, dict[str, float]] = {}
            evidence_basis_breakdown: dict[str, int] = {}
            pre_analysis_blocked = 0
            should_trade_but_blocked = 0
            should_trade_blocked_breakdown: dict[str, int] = {}
            cycle_prompt_tokens = 0
            cycle_completion_tokens = 0
            cycle_reasoning_tokens = 0
            cycle_cached_tokens = 0
            event_cycle_traded_market_ids: dict[str, set[str]] = {}
            event_cycle_traded_outcomes: dict[str, set[str]] = {}
            confidence_calibration_applied_count = 0
            confidence_calibration_delta_sum = 0.0
            confidence_calibration_historical_win_rates: list[float] = []
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
                _base_log_trade_decision(
                    market_id=market_id,
                    question=question,
                    decision=decision,
                    order=order,
                    execution_audit=execution_audit,
                )
                normalized_score_breakdown = _score_breakdown_from_execution_audit(
                    execution_audit=execution_audit,
                    explicit_score_breakdown=score_breakdown,
                )
                try:
                    state_manager.record_decision_receipt(
                        cycle_id=cycle_id,
                        market_id=market_id,
                        decision=decision,
                        order=order,
                        execution_audit=execution_audit,
                        score_breakdown=normalized_score_breakdown,
                    )
                except Exception as receipt_exc:
                    logger.debug(
                        "Decision receipt persistence failed: market=%s error=%s",
                        market_id,
                        receipt_exc,
                        data={"market_id": market_id, "error": str(receipt_exc)},
                    )
                audit = execution_audit if isinstance(execution_audit, dict) else {}
                if str(audit.get("final_action") or "").strip().lower() == "skip":
                    rejection_funnel_summary.append(
                        {
                            "market_id": market_id,
                            "market_family": audit.get("market_family"),
                            "evidence_basis": audit.get("evidence_basis_class"),
                            "score": audit.get("pre_execution_final_score"),
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
            family_action_snapshot: dict[str, dict[str, float | int]] = {}

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
                family_action_snapshot = state_manager.get_family_action_snapshot(
                    lookback=max(100, settings.PRE_ANALYSIS_ZERO_ACTION_FAMILY_MIN_SAMPLES * 20),
                )
            except Exception as exc:
                logger.debug(
                    "Family action snapshot lookup failed: %s",
                    exc,
                    data={"error": str(exc)},
                )
                family_action_snapshot = {}

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

            def _get_family_action_stats(family_name: str) -> dict[str, float | int]:
                normalized_family = str(family_name or "").strip().lower()
                if not normalized_family:
                    return {}
                return dict(family_action_snapshot.get(normalized_family, {}))

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

                should_skip, skip_reason = scheduler.should_skip(market, state)
                if should_skip:
                    if skip_reason == "market closed":
                        scheduler_skipped_closed += 1
                    elif skip_reason == "recently analyzed":
                        scheduler_skipped_recently += 1
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
                pre_analysis_hard_reject, pre_analysis_hard_reject_reason, pre_analysis_hard_reject_data = (
                    _pre_analysis_hard_rejection(
                        market=market,
                        state=state if isinstance(state, MarketState) else None,
                        settings=settings,
                        traded_before=traded_before,
                        had_recent_fallback_edge=had_recent_fallback_edge,
                        family_action_stats=_get_family_action_stats(family_name),
                        historical_family_stats=historical_family_stats,
                        fallback_family_edge_rate=family_fallback_rate,
                        fallback_family_sample_size=family_fallback_samples,
                    )
                )
                if pre_analysis_hard_reject:
                    pre_analysis_blocked += 1
                    if pre_analysis_hard_reject_reason:
                        _record_rejection_reason(
                            pre_analysis_rejection_breakdown,
                            pre_analysis_hard_reject_reason,
                        )
                    logger.debug(
                        "Skipping %s: pre-analysis hard rejection (%s)",
                        market.id,
                        pre_analysis_hard_reject_reason or "pre_analysis_hard_reject",
                        data={
                            "market_id": market.id,
                            **pre_analysis_hard_reject_data,
                        },
                    )
                    continue
                pre_analysis_score = None
                pre_analysis_breakdown: dict[str, float] | None = None
                if settings.PRE_ANALYSIS_OPPORTUNITY_ENABLED:
                    pre_analysis_score, pre_analysis_breakdown = _pre_analysis_opportunity_score(
                        market,
                        state if isinstance(state, MarketState) else None,
                        settings,
                        traded_before=traded_before,
                        fallback_family_edge_rate=family_fallback_rate,
                        fallback_family_sample_size=family_fallback_samples,
                        historical_family_stats=historical_family_stats,
                    )
                    if pre_analysis_score < settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE:
                        pre_analysis_blocked += 1
                        _record_rejection_reason(
                            pre_analysis_rejection_breakdown,
                            "pre_analysis_score_below_min",
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
                        "market_snapshot_monotonic": time.monotonic(),
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
            ) = _resolve_dynamic_analysis_candidate_cap(
                settings=settings,
                best_pre_analysis_score=best_pre_analysis_score,
            )
            analysis_candidate_attempt_limit = dynamic_max_markets_per_cycle + max(
                0,
                settings.XAI_CIRCUIT_BREAKER_MAX_FAILURES,
            )
            analysis_candidates = _cap_analysis_candidates(
                analysis_candidates,
                analysis_candidate_attempt_limit,
                max_weather_candidates_per_cycle=settings.MAX_WEATHER_CANDIDATES_PER_CYCLE,
                max_crypto_candidates_per_cycle=settings.MAX_CRYPTO_CANDIDATES_PER_CYCLE,
                max_speech_candidates_per_cycle=settings.MAX_SPEECH_CANDIDATES_PER_CYCLE,
                max_music_candidates_per_cycle=settings.MAX_MUSIC_CANDIDATES_PER_CYCLE,
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
                    with ThreadPoolExecutor(max_workers=analysis_worker_count) as executor:
                        parallel_analysis_used = True
                        future_to_market = {}
                        for candidate in analysis_candidates:
                            worker_client = _build_grok_client_for_worker(
                                settings,
                                provider=shared_xai_provider,
                            )
                            future = executor.submit(
                                _analyze_market_candidate,
                                candidate["market"],
                                candidate["state"],
                                candidate["anchor_analysis"],
                                settings,
                                worker_client,
                            )
                            future_to_market[future] = candidate["market"]

                        for future in as_completed(future_to_market):
                            market = future_to_market[future]
                            try:
                                analysis_results[market.id] = future.result()
                            except Exception as exc:
                                logger.error(
                                    "Failed to analyze market %s: %s",
                                    market.id,
                                    exc,
                                    data={"market_id": market.id, "error": str(exc)},
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
                        )
                        analysis_results[market.id] = result
                        if result.get("analysis_failed"):
                            if result.get("analysis_error_retriable_xai"):
                                consecutive_xai_failures += 1
                            else:
                                consecutive_xai_failures = 0
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
                implied_prob_for_rank = _get_implied_probability(market, decision.outcome)
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
                    ),
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
                    "liquidity_penalty": rank_score.liquidity_penalty,
                    "staleness_penalty": rank_score.staleness_penalty,
                    "evidence_basis_bonus": rank_score.evidence_basis_bonus,
                    "generic_bin_penalty": rank_score.generic_bin_penalty,
                    "ambiguous_resolution_penalty": rank_score.ambiguous_resolution_penalty,
                }

            analysis_candidates = sorted(
                analysis_candidates,
                key=lambda candidate: _analysis_result_rank(
                    analysis_results.get(candidate["market"].id)
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

            for candidate in analysis_candidates:
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
                    _record_terminal_outcome(
                        state_manager,
                        market.id,
                        "analysis_failure",
                    )
                    logger.info(
                        "Market %s skipped due to initial analysis failure: %s",
                        market.id,
                        analysis_result.get("analysis_error"),
                        data={
                            "market_id": market.id,
                            "final_action": "skip",
                            "final_reason": "analysis_failure_after_retries",
                            "error_type": analysis_result.get("analysis_error_type"),
                        },
                    )
                    continue
                markets_analyzed += 1
                decision = analysis_result["decision"]
                decisions_made += 1
                evidence_basis = _decision_evidence_basis(decision)
                evidence_basis_breakdown[evidence_basis] = (
                    evidence_basis_breakdown.get(evidence_basis, 0) + 1
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
                audit_context: dict[str, Any] = {
                    "market_family": market_family_name,
                    "pre_analysis_score": candidate.get("pre_analysis_score"),
                    "pre_analysis_breakdown": candidate.get("pre_analysis_breakdown"),
                    "analysis_count": analysis_count_for_market,
                    "non_actionable_streak": non_actionable_streak_for_market,
                    "traded_before": bool(candidate.get("traded_before", False)),
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
                    "event_ticker_prefix": event_ticker_prefix,
                    "correlated_positions_count": correlated_positions_count,
                    "correlated_position_outcomes": sorted(correlated_position_outcomes),
                    "daily_trade_count": daily_trade_count,
                    "daily_pnl_estimate": daily_pnl_estimate,
                }
                audit_context.update(score_receipt_fields)
                if analysis_result.get("confidence_calibration_applied"):
                    confidence_calibration_applied_count += 1
                confidence_calibration_delta_sum += float(
                    analysis_result.get("raw_vs_calibrated_delta", 0.0) or 0.0
                )
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
                    ) = _is_confidence_override_allowed(
                        settings=settings,
                        decision=decision,
                        override_edge=override_edge,
                    )
                    if confidence_override_allowed:
                        logger.info(
                            "Confidence gate override [%s]: conf %.2f < min %.2f but edge %.3f and evidence %.2f meet override thresholds",
                            market.id,
                            decision.confidence,
                            settings.MIN_CONFIDENCE,
                            override_edge,
                            decision.evidence_quality,
                            data={
                                "market_id": market.id,
                                "confidence": decision.confidence,
                                "min_confidence": settings.MIN_CONFIDENCE,
                                "override_edge": override_edge,
                                "market_edge": market_edge,
                                "model_edge": decision.edge_external,
                                "evidence_quality": decision.evidence_quality,
                                "override_min_confidence": override_min_confidence,
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
                                override_edge=override_edge,
                                market_edge=market_edge,
                                override_min_confidence=override_min_confidence,
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
                if (
                    decision_for_edge.evidence_quality
                    < _min_evidence_quality_for_market(market, settings)
                ):
                    min_evidence_quality = _min_evidence_quality_for_market(
                        market, settings
                    )
                    evidence_rejection_reason = (
                        "weather_evidence_quality_below_min"
                        if market_family(market) == "weather"
                        else "evidence_quality_below_min"
                    )
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
                            decision_terminal=True,
                            final_action="skip",
                            final_reason=evidence_rejection_reason,
                            evidence_quality=decision_for_edge.evidence_quality,
                            min_evidence_quality=min_evidence_quality,
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(
                        state_manager,
                        market.id,
                        evidence_rejection_reason,
                    )
                    logger.warning(
                        "SKIP [%s] -> %s after should_trade=True",
                        market.id,
                        evidence_rejection_reason,
                        data={
                            "market_id": market.id,
                            "final_reason": evidence_rejection_reason,
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
                edge_ok, edge_value, edge_reason = _passes_edge_threshold(
                    implied_prob,
                    decision_for_edge,
                    settings,
                    market=market,
                )
                if settings.CALIBRATION_MODE_ENABLED:
                    calibration_payload = {
                        "market_id": market.id,
                        "cycle": cycle_count,
                        "edge_market": edge_value,
                        "implied_prob_market": implied_prob,
                        "confidence": decision_for_edge.confidence,
                        "evidence_quality": decision.evidence_quality,
                        "liquidity_usdc": market.liquidity_usdc,
                        "analysis_duration_ms": round((time.monotonic() - market_start) * 1000, 2),
                        "edge_gate_pass": edge_ok,
                        "gate_edge_required": required_edge_threshold,
                        "gate_edge_actual": edge_value,
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
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="edge_gate_blocked",
                            gate_edge_required=required_edge_threshold,
                            gate_edge_actual=edge_value,
                            gate_edge_reason=edge_reason,
                            **audit_context,
                        ),
                    )
                    _record_terminal_outcome(state_manager, market.id, "edge_gate_blocked")
                    question_short = market.question[:40] + "..." if len(market.question) > 40 else market.question
                    logger.warning(
                        "SKIP [%s] '%s' -> edge gate (%s)",
                        market.id,
                        question_short,
                        edge_reason,
                        data={
                            "market_id": market.id,
                            "final_reason": "edge_gate_blocked",
                            "implied_prob": implied_prob,
                            "entry_price": entry_price,
                            "confidence": decision_for_edge.confidence,
                            "edge": edge_value,
                            "gate_edge_required": required_edge_threshold,
                            "gate_edge_actual": edge_value,
                            "gate_edge_reason": edge_reason,
                            "score_breakdown": score_receipt_fields,
                        },
                    )
                    continue
                edge_gate_passed += 1

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

                score_result = analysis_result.get("pre_execution_score_result")
                if score_result is None:
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
                        ),
                    )
                score_receipt_fields = _score_receipt_fields(score_result)
                score_mode = settings.SCORE_GATE_MODE
                score_threshold_effective = _effective_score_gate_threshold(
                    settings=settings,
                    market=market,
                    evidence_basis_class=evidence_basis,
                    evidence_quality=decision_for_edge.evidence_quality,
                )
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
                        "confidence_alignment_bonus": score_result.confidence_alignment_bonus,
                        "evidence_basis_bonus": score_result.evidence_basis_bonus,
                        "observed_data_bonus": score_result.observed_data_bonus,
                        "low_information_penalty": score_result.low_information_penalty,
                        "no_external_odds_penalty": score_result.no_external_odds_penalty,
                        "repeated_analysis_penalty": score_result.repeated_analysis_penalty,
                        "mention_market_penalty": score_result.mention_market_penalty,
                        "confidence_calibration_penalty": score_result.confidence_calibration_penalty,
                        "fallback_edge_penalty": score_result.fallback_edge_penalty,
                        "proxy_evidence_penalty": score_result.proxy_evidence_penalty,
                        "liquidity_penalty": score_result.liquidity_penalty,
                        "staleness_penalty": score_result.staleness_penalty,
                        "rejection_reasons": list(score_result.rejection_reasons),
                        "generic_bin_penalty": score_result.generic_bin_penalty,
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
                    elif score_result.final_score < score_threshold_effective:
                        score_gate_blocked += 1
                        trades_skipped_edge += 1
                        score_near_misses.append(
                            {
                                "market_id": market.id,
                                "final_score": float(score_result.final_score),
                                "score_threshold": float(score_threshold_effective),
                                "score_gap": float(score_threshold_effective - score_result.final_score),
                                "rejection_reasons": list(score_result.rejection_reasons),
                            }
                        )
                        _record_should_trade_blocked("score_gate_blocked")
                        _record_rejection_reason(rejection_breakdown, "score_gate_blocked")
                        log_trade_decision(
                            market_id=market.id,
                            question=market.question,
                            decision=decision_for_edge.model_dump(),
                            execution_audit=_build_execution_audit(
                                decision_terminal=True,
                                final_action="skip",
                                final_reason="score_gate_blocked",
                                score_threshold=score_threshold_effective,
                                **audit_context,
                            ),
                        )
                        _record_terminal_outcome(state_manager, market.id, "score_gate_blocked")
                        question_short = market.question[:40] + "..." if len(market.question) > 40 else market.question
                        logger.warning(
                            "SKIP [%s] '%s' -> score gate (%.4f < %.4f)",
                            market.id,
                            question_short,
                            score_result.final_score,
                            score_threshold_effective,
                            data=score_payload,
                        )
                        continue
                    score_gate_passed += 1
                else:
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
                    )
                else:
                    adjusted_bet_pct = edge_scaling_bet_pct
                if adjusted_bet_pct <= 0:
                    trades_skipped_edge += 1
                    _record_should_trade_blocked("zero_bet_after_sizing")
                    _record_rejection_reason(rejection_breakdown, "zero_bet_after_sizing")
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
                            edge_market=edge_value,
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
                            edge_market=edge_value,
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
                    if not refreshed_edge_ok:
                        trades_skipped_edge += 1
                        _record_should_trade_blocked("refreshed_edge_gate_blocked")
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
                                edge_market=refreshed_edge_value,
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
            cycle_receipt = {
                "cycle": cycle_count,
                "cycle_id": cycle_id,
                "duration_ms": round(cycle_duration, 2),
                "fetched_markets": fetched_count,
                "eligible_markets": len(markets),
                "analysis_candidates": analysis_candidates_count,
                "pre_analysis_passed": pre_analysis_passed,
                "analyzed_markets": markets_analyzed,
                "decisions_made": decisions_made,
                "validation_passed": validation_passed,
                "refined_markets": markets_refined,
                "execution_candidates": execution_candidates,
                "edge_gate_passed": edge_gate_passed,
                "score_gate_passed": score_gate_passed,
                "order_attempted": trades_attempted,
                "order_attempts": trades_attempted,
                "orders_filled": trades_filled,
                "orders_canceled_unfilled": trades_canceled_unfilled,
                "total_usd_deployed": round(total_usd_deployed, 2),
                "execution_family_breakdown": execution_family_breakdown,
                "should_trade_but_blocked": should_trade_but_blocked,
                "should_trade_blocked_breakdown": dict(
                    sorted(should_trade_blocked_breakdown.items())
                ),
                "skip_counts": {
                    "no_trade": trades_skipped_no_trade,
                    "confidence": trades_skipped_confidence,
                    "edge": trades_skipped_edge,
                    "position": trades_skipped_position,
                    "balance": trades_skipped_balance,
                    "kelly_sub_floor": trades_skipped_kelly_sub_floor,
                    "pre_analysis": pre_analysis_blocked,
                },
                "rejection_breakdown": rejection_breakdown,
                "pre_analysis_rejection_breakdown": pre_analysis_rejection_breakdown,
                "score_rejection_reason_breakdown": score_rejection_reason_breakdown,
                "api_tokens_consumed": api_tokens_consumed,
                "api_prompt_tokens": cycle_prompt_tokens,
                "api_completion_tokens": cycle_completion_tokens,
                "api_reasoning_tokens": cycle_reasoning_tokens,
                "api_cached_tokens": cycle_cached_tokens,
                "api_cost_estimate_usd": round(api_cost_estimate_usd, 6),
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
                "raw_vs_calibrated_delta": round(
                    confidence_calibration_delta_sum / max(1, markets_analyzed),
                    4,
                ),
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
                "flip_guard_blocked=%d execution_candidates=%d should_trade_blocked=%d order_attempts=%d "
                "skipped_kelly_sub_floor=%d",
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
                should_trade_but_blocked,
                trades_attempted,
                trades_skipped_kelly_sub_floor,
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
                    "should_trade_but_blocked": should_trade_but_blocked,
                    "should_trade_blocked_breakdown": should_trade_blocked_breakdown,
                    "order_attempts": trades_attempted,
                    "orders_filled": trades_filled,
                    "orders_canceled_unfilled": trades_canceled_unfilled,
                    "total_usd_deployed": round(total_usd_deployed, 2),
                    "execution_family_breakdown": execution_family_breakdown,
                    "skipped_kelly_sub_floor": trades_skipped_kelly_sub_floor,
                    "pre_analysis_blocked": pre_analysis_blocked,
                    "pre_analysis_rejection_breakdown": pre_analysis_rejection_breakdown,
                    "api_tokens_consumed": api_tokens_consumed,
                    "api_cost_estimate_usd": round(api_cost_estimate_usd, 6),
                    "best_candidate_market_id": best_candidate_market_id,
                    "best_candidate_score": best_candidate_score,
                    "evidence_basis_breakdown": dict(sorted(evidence_basis_breakdown.items())),
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
                },
            )
            if trades_attempted > 0:
                consecutive_zero_order_cycles = 0
            else:
                consecutive_zero_order_cycles += 1
            dry_streak_sleep_seconds = _dry_streak_sleep_seconds(
                base_poll_interval_sec=settings.POLL_INTERVAL_SEC,
                consecutive_zero_order_cycles=consecutive_zero_order_cycles,
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
