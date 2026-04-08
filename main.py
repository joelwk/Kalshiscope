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
from config import Settings, load_settings
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
    log_trade_decision,
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
from score_engine import compute_final_score
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
_MAX_CONFIDENCE = 1.0
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
    extreme_yes_price_lower: float | None = None,
    extreme_yes_price_upper: float | None = None,
    min_tradeable_yes_price: float | None = None,
    max_tradeable_yes_price: float | None = None,
    skip_weather_bin_markets: bool = False,
):
    """Filter markets based on liquidity, category, and close date constraints."""
    filtered = []
    skipped_liquidity = 0
    skipped_volume_24h = 0
    skipped_extreme_price = 0
    skipped_untradeable_price = 0
    skipped_allowlist = 0
    skipped_blocklist = 0
    skipped_close_too_soon = 0
    skipped_close_too_far = 0
    skipped_closed_now = 0
    skipped_resolved = 0
    skipped_ticker_prefix_blocklist = 0
    skipped_weather_bin_markets = 0
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
        effective_liquidity = (
            market.liquidity_usdc if market.liquidity_usdc is not None else 0.0
        )
        if effective_liquidity < min_liquidity:
            skipped_liquidity += 1
            continue
        if min_volume_24h > 0.0:
            effective_volume_24h = (
                market.volume_24h if market.volume_24h is not None else 0.0
            )
            if effective_volume_24h < min_volume_24h:
                skipped_volume_24h += 1
                continue
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
        if blocklist and (market.category in blocklist):
            skipped_blocklist += 1
            continue
        if ticker_prefix_blocklist:
            market_id = (market.id or "").upper()
            if any(market_id.startswith(prefix.upper()) for prefix in ticker_prefix_blocklist):
                skipped_ticker_prefix_blocklist += 1
                continue
        if skip_weather_bin_markets and _is_weather_bin_market((market.id or "").upper()):
            skipped_weather_bin_markets += 1
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
        "skipped_blocklist=%d, skipped_ticker_prefix_blocklist=%d, skipped_resolved=%d, skipped_close_too_soon=%d, "
        "skipped_close_too_far=%d, skipped_closed_now=%d, skipped_weather_bin_markets=%d, "
        "skipped_likely_resolved_by_ticker=%d",
        len(filtered),
        skipped_liquidity,
        skipped_volume_24h,
        skipped_untradeable_price,
        skipped_extreme_price,
        skipped_allowlist,
        skipped_blocklist,
        skipped_ticker_prefix_blocklist,
        skipped_resolved,
        skipped_close_too_soon,
        skipped_close_too_far,
        skipped_closed_now,
        skipped_weather_bin_markets,
        skipped_likely_resolved_by_ticker,
        data={
            "kept": len(filtered),
            "skipped_liquidity": skipped_liquidity,
            "skipped_volume_24h": skipped_volume_24h,
            "skipped_untradeable_price": skipped_untradeable_price,
            "skipped_extreme_price": skipped_extreme_price,
            "skipped_allowlist": skipped_allowlist,
            "skipped_blocklist": skipped_blocklist,
            "skipped_ticker_prefix_blocklist": skipped_ticker_prefix_blocklist,
            "skipped_resolved": skipped_resolved,
            "skipped_close_too_soon": skipped_close_too_soon,
            "skipped_close_too_far": skipped_close_too_far,
            "skipped_closed_now": skipped_closed_now,
            "skipped_weather_bin_markets": skipped_weather_bin_markets,
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
                "skipped_ticker_prefix_blocklist": skipped_ticker_prefix_blocklist,
                "skipped_resolved": skipped_resolved,
                "skipped_close_too_soon": skipped_close_too_soon,
                "skipped_close_too_far": skipped_close_too_far,
                "skipped_closed_now": skipped_closed_now,
                "skipped_weather_bin_markets": skipped_weather_bin_markets,
                "skipped_likely_resolved_by_ticker": skipped_likely_resolved_by_ticker,
            }
        )
    return filtered


def _is_weather_bin_market(market_id: str) -> bool:
    return bool(re.match(r"^KX(?:LOWT|HIGHT|TEMP)[A-Z]+-.*-B[0-9]", market_id))


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
    if market is not None and market_family(market) == "weather":
        min_edge = max(min_edge, settings.WEATHER_MIN_EDGE)
    if implied_prob < settings.LOW_PRICE_THRESHOLD:
        min_edge = max(min_edge, settings.LOW_PRICE_MIN_EDGE)
    if settings.COINFLIP_PRICE_LOWER <= implied_prob <= settings.COINFLIP_PRICE_UPPER:
        min_edge = max(min_edge, settings.LOW_PRICE_MIN_EDGE)
    if (edge_source or "").lower() == "fallback":
        min_edge = max(min_edge, settings.FALLBACK_EDGE_MIN_EDGE)
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
    edge = decision.confidence - implied_prob
    min_edge = _edge_threshold_for_market(
        implied_prob,
        settings,
        decision.edge_source,
        market=market,
    )
    if edge < min_edge:
        return False, edge, f"edge {edge:.2f} below min {min_edge:.2f}"
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
    return max(0.0, min(1.0, bet_pct))


def _is_within_order_submission_band(
    price: float | None,
    settings: Settings,
) -> bool:
    if price is None:
        return False
    return settings.ORDER_SUBMISSION_MIN_PRICE <= price <= settings.ORDER_SUBMISSION_MAX_PRICE


def _max_confidence_for_market(market: Market | None, settings: Settings) -> float:
    if not market:
        return 1.0
    is_sports, is_esports = market_category_flags(market)
    if is_sports:
        return settings.MAX_SPORTS_CONFIDENCE
    if is_esports:
        return settings.MAX_ESPORTS_CONFIDENCE
    if market_family(market) == "weather":
        return settings.MAX_WEATHER_CONFIDENCE
    return 1.0


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
            return value
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
            },
        )
        return decision, False, False

    confidence_delta = decision.confidence - anchor_confidence
    new_edge = _decision_edge_for_outcome(market, decision.outcome, decision.confidence)
    anchor_edge = _decision_edge_for_outcome(market, anchor_outcome, anchor_confidence)
    edge_delta = None
    edge_gain_ok = True
    if new_edge is not None and anchor_edge is not None:
        edge_delta = abs(new_edge) - abs(anchor_edge)
        edge_gain_ok = edge_delta >= settings.FLIP_GUARD_MIN_EDGE_GAIN

    abs_conf_ok = decision.confidence >= settings.FLIP_GUARD_MIN_ABS_CONFIDENCE
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
            f"abs_conf {decision.confidence:.2f} < {settings.FLIP_GUARD_MIN_ABS_CONFIDENCE:.2f}"
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
) -> None:
    traded_ids = state_manager.get_traded_market_ids()
    if not traded_ids:
        return
    market_map = {market.id: market for market in markets}
    resolved_count = 0
    for market_id in traded_ids:
        market = market_map.get(market_id)
        if not market:
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
            "Resolved markets updated: count=%d",
            resolved_count,
            data={"resolved_count": resolved_count},
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
        cap_reason = "general"

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
            "min_evidence_quality_for_trade": settings.MIN_EVIDENCE_QUALITY_FOR_TRADE,
            "min_liquidity_usdc": settings.MIN_LIQUIDITY_USDC,
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
            "kalshi_server_side_filters_enabled": settings.KALSHI_SERVER_SIDE_FILTERS_ENABLED,
            "kalshi_max_fetch_pages": settings.KALSHI_MAX_FETCH_PAGES,
            "score_gate_mode": settings.SCORE_GATE_MODE,
            "score_gate_threshold": settings.SCORE_GATE_THRESHOLD,
            "max_markets_per_cycle": settings.MAX_MARKETS_PER_CYCLE,
            "max_trades_per_cycle": settings.MAX_TRADES_PER_CYCLE,
            "bayesian_enabled": settings.BAYESIAN_ENABLED,
            "bayesian_skip_stale_updates": settings.BAYESIAN_SKIP_STALE_UPDATES,
            "bayesian_max_posterior": settings.BAYESIAN_MAX_POSTERIOR,
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
            "order_submission_min_price": settings.ORDER_SUBMISSION_MIN_PRICE,
            "order_submission_max_price": settings.ORDER_SUBMISSION_MAX_PRICE,
            "order_fallback_to_market": settings.ORDER_FALLBACK_TO_MARKET,
            "order_fallback_min_confidence": settings.ORDER_FALLBACK_MIN_CONFIDENCE,
            "calibration_mode_enabled": settings.CALIBRATION_MODE_ENABLED,
            "calibration_min_samples": settings.CALIBRATION_MIN_SAMPLES,
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


def _analysis_result_rank(result: dict[str, Any] | None) -> tuple[float, float, float]:
    if not result:
        return (0.0, 0.0, 0.0)
    decision = result.get("decision")
    if not isinstance(decision, TradeDecision):
        return (0.0, 0.0, 0.0)
    should_trade_rank = 1.0 if decision.should_trade and not decision.abstain else 0.0
    evidence_rank = max(0.0, min(1.0, decision.evidence_quality))
    confidence_rank = max(0.0, min(1.0, decision.confidence))
    return (should_trade_rank, evidence_rank, confidence_rank)


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
    for key, value in extra.items():
        if value is not None:
            payload[key] = value
    for alias_key, canonical_key in alias_to_canonical.items():
        if alias_key in payload and canonical_key not in payload:
            payload[canonical_key] = payload[alias_key]
        payload.pop(alias_key, None)
    return payload


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


def _cap_analysis_candidates(
    analysis_candidates: list[dict[str, Any]],
    max_markets_per_cycle: int,
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

    if not grouped:
        return analysis_candidates[:max_markets_per_cycle]

    selected: list[dict[str, Any]] = []
    while len(selected) < max_markets_per_cycle:
        progressed = False
        for family in family_order:
            family_candidates = grouped.get(family)
            if not family_candidates:
                continue
            selected.append(family_candidates.pop(0))
            progressed = True
            if len(selected) >= max_markets_per_cycle:
                break
        if not progressed:
            break
    return selected


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
                search_config=search_config,
            )
            was_refined = True

    decision = _cap_confidence_for_category(decision, market, settings)
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
    }


def main() -> None:
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
        max_fetch_pages=settings.KALSHI_MAX_FETCH_PAGES,
    )
    logger.debug("Kalshi client initialized with base_url=%s", settings.KALSHI_API_BASE_URL)

    logger.info("PredictBot started (dry_run=%s)", settings.DRY_RUN)
    cycle_count = 0

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

            filter_stats: dict[str, int] = {}
            markets = _filter_markets(
                markets,
                settings.MIN_LIQUIDITY_USDC,
                settings.MARKET_CATEGORIES_ALLOWLIST,
                settings.MARKET_CATEGORIES_BLOCKLIST,
                ticker_prefix_blocklist=settings.MARKET_TICKER_BLOCKLIST_PREFIXES,
                skip_weather_bin_markets=settings.SKIP_WEATHER_BIN_MARKETS,
                min_close_days=settings.MARKET_MIN_CLOSE_DAYS,
                max_close_days=settings.MARKET_MAX_CLOSE_DAYS,
                stats=filter_stats,
                min_volume_24h=settings.MIN_VOLUME_24H,
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
            try:
                cycle_bankroll = kalshi_client.get_balance()
            except Exception as exc:
                logger.debug(
                    "Kalshi balance lookup failed for position cap: %s",
                    exc,
                    data={"error": str(exc)},
                )

            if settings.RESOLUTION_SYNC_INTERVAL_CYCLES > 0:
                if cycle_count % settings.RESOLUTION_SYNC_INTERVAL_CYCLES == 0:
                    try:
                        _update_resolved_markets(markets, state_manager)
                    except Exception as exc:
                        logger.warning(
                            "Resolution sync failed: %s",
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

            analysis_candidates: list[dict[str, Any]] = []
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
                analysis_candidates.append(
                    {
                        "market": market,
                        "state": state,
                        "anchor_analysis": anchor_analysis,
                        "market_snapshot_monotonic": time.monotonic(),
                    }
                )

            original_analysis_candidates_count = len(analysis_candidates)
            available_family_distribution = _analysis_candidate_family_counts(
                analysis_candidates
            )
            analysis_candidate_attempt_limit = settings.MAX_MARKETS_PER_CYCLE + max(
                0,
                settings.XAI_CIRCUIT_BREAKER_MAX_FAILURES,
            )
            analysis_candidates = _cap_analysis_candidates(
                analysis_candidates,
                analysis_candidate_attempt_limit,
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
                        "max_markets_per_cycle": settings.MAX_MARKETS_PER_CYCLE,
                        "analysis_candidate_attempt_limit": analysis_candidate_attempt_limit,
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
                    "max_markets_per_cycle": settings.MAX_MARKETS_PER_CYCLE,
                    "analysis_candidate_attempt_limit": analysis_candidate_attempt_limit,
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

            analysis_candidates = sorted(
                analysis_candidates,
                key=lambda candidate: _analysis_result_rank(
                    analysis_results.get(candidate["market"].id)
                ),
                reverse=True,
            )

            for candidate in analysis_candidates:
                if markets_analyzed >= settings.MAX_MARKETS_PER_CYCLE:
                    break
                market = candidate["market"]
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

                if decision.confidence < settings.MIN_CONFIDENCE:
                    override_edge, market_edge = _confidence_gate_override_metrics(market, decision)
                    confidence_override_allowed = (
                        settings.CONFIDENCE_GATE_EDGE_OVERRIDE_ENABLED
                        and override_edge is not None
                        and override_edge >= settings.CONFIDENCE_GATE_MIN_EDGE
                        and decision.evidence_quality
                        >= settings.CONFIDENCE_GATE_MIN_EVIDENCE_QUALITY
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
                            },
                        )
                    else:
                        trades_skipped_confidence += 1
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
                            ),
                        )
                        _record_terminal_outcome(state_manager, market.id, "confidence_below_min")
                        question_short = market.question[:40] + "..." if len(market.question) > 40 else market.question
                        logger.info(
                            "SKIP [%s] '%s' -> conf %.2f < min %.2f",
                            market.id,
                            question_short,
                            decision.confidence,
                            settings.MIN_CONFIDENCE,
                        )
                        continue

                entry_price = _get_outcome_entry_price(market, decision.outcome)
                implied_prob = _get_implied_probability(market, decision.outcome)
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
                    < settings.MIN_EVIDENCE_QUALITY_FOR_TRADE
                ):
                    trades_skipped_no_trade += 1
                    _record_rejection_reason(
                        rejection_breakdown,
                        "evidence_quality_below_min",
                    )
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="evidence_quality_below_min",
                            evidence_quality=decision_for_edge.evidence_quality,
                            min_evidence_quality=settings.MIN_EVIDENCE_QUALITY_FOR_TRADE,
                        ),
                    )
                    _record_terminal_outcome(
                        state_manager,
                        market.id,
                        "evidence_quality_below_min",
                    )
                    continue
                bucket = _price_bucket(implied_prob, settings)
                price_bucket_stats[bucket] += 1
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
                    _record_rejection_reason(rejection_breakdown, "edge_gate_blocked")
                    log_trade_decision(
                        market_id=market.id,
                        question=market.question,
                        decision=decision_for_edge.model_dump(),
                        execution_audit=_build_execution_audit(
                            decision_terminal=True,
                            final_action="skip",
                            final_reason="edge_gate_blocked",
                        ),
                    )
                    _record_terminal_outcome(state_manager, market.id, "edge_gate_blocked")
                    question_short = market.question[:40] + "..." if len(market.question) > 40 else market.question
                    logger.info(
                        "SKIP [%s] '%s' -> edge gate (%s)",
                        market.id,
                        question_short,
                        edge_reason,
                        data={
                            "market_id": market.id,
                            "implied_prob": implied_prob,
                            "entry_price": entry_price,
                            "confidence": decision_for_edge.confidence,
                            "edge": edge_value,
                        },
                    )
                    continue

                if _is_uniform_implied_probability(implied_prob, market.outcomes):
                    uniform_implied = 1.0 / len(market.outcomes)
                    trades_skipped_edge += 1
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
                        ),
                    )
                    _record_terminal_outcome(
                        state_manager,
                        market.id,
                        "uniform_implied_probability",
                    )
                    question_short = market.question[:40] + "..." if len(market.question) > 40 else market.question
                    logger.info(
                        "SKIP [%s] '%s' -> uniform implied probability detected (%d outcomes, implied=%.3f)",
                        market.id,
                        question_short,
                        len(market.outcomes),
                        implied_prob,
                        data={
                            "market_id": market.id,
                            "implied_prob": implied_prob,
                            "uniform_implied": uniform_implied,
                            "outcomes": [outcome.name for outcome in market.outcomes],
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

                score_result = compute_final_score(
                    market=market,
                    decision=decision_for_edge,
                    implied_prob_market=implied_prob,
                    bayesian_posterior=bayesian_posterior_applied,
                    lmsr_price=lmsr_execution_price,
                    inefficiency_signal=ineff_signal,
                    kelly_raw=kelly_raw_value,
                    is_weather_market=(market_family(market) == "weather"),
                    weather_score_penalty=settings.WEATHER_SCORE_PENALTY,
                )
                score_mode = settings.SCORE_GATE_MODE
                if score_mode != "off":
                    score_payload = {
                        "market_id": market.id,
                        "score_mode": score_mode,
                        "score_threshold": settings.SCORE_GATE_THRESHOLD,
                        "final_score": score_result.final_score,
                        "edge_market": score_result.edge_market,
                        "edge_external": score_result.edge_external,
                        "evidence_quality": score_result.evidence_quality,
                        "evidence_component": score_result.evidence_component,
                        "bayesian_component": score_result.bayesian_component,
                        "inefficiency_component": score_result.inefficiency_component,
                        "kelly_component": score_result.kelly_component,
                        "confidence_alignment_bonus": score_result.confidence_alignment_bonus,
                        "liquidity_penalty": score_result.liquidity_penalty,
                        "staleness_penalty": score_result.staleness_penalty,
                        "bayesian_posterior": score_result.bayesian_posterior,
                        "lmsr_price": score_result.lmsr_price,
                        "inefficiency_signal": score_result.inefficiency_signal,
                        "kelly_raw": score_result.kelly_raw,
                        "bayesian_posterior_raw": bayesian_posterior_raw,
                        "bayesian_posterior_applied": bayesian_posterior_applied,
                        "bayesian_applied": bayesian_posterior_applied is not None,
                        "bayesian_update_count": bayesian_update_count,
                        "bayesian_min_updates": settings.BAYESIAN_MIN_UPDATES_FOR_TRADE,
                        "likelihood_ratio": likelihood_ratio,
                    }
                    if score_mode == "shadow":
                        logger.debug(
                            "Score gate shadow: market=%s final_score=%.4f threshold=%.4f",
                            market.id,
                            score_result.final_score,
                            settings.SCORE_GATE_THRESHOLD,
                            data=score_payload,
                        )
                    elif score_result.final_score < settings.SCORE_GATE_THRESHOLD:
                        score_gate_blocked += 1
                        trades_skipped_edge += 1
                        _record_rejection_reason(rejection_breakdown, "score_gate_blocked")
                        log_trade_decision(
                            market_id=market.id,
                            question=market.question,
                            decision=decision_for_edge.model_dump(),
                            execution_audit=_build_execution_audit(
                                decision_terminal=True,
                                final_action="skip",
                                final_reason="score_gate_blocked",
                                score_threshold=settings.SCORE_GATE_THRESHOLD,
                                score_final=score_result.final_score,
                                score_breakdown=score_payload,
                            ),
                        )
                        _record_terminal_outcome(state_manager, market.id, "score_gate_blocked")
                        question_short = market.question[:40] + "..." if len(market.question) > 40 else market.question
                        logger.info(
                            "SKIP [%s] '%s' -> score gate (%.4f < %.4f)",
                            market.id,
                            question_short,
                            score_result.final_score,
                            settings.SCORE_GATE_THRESHOLD,
                            data=score_payload,
                        )
                        continue

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
                    _record_rejection_reason(rejection_breakdown, "zero_bet_after_sizing")
                    question_short = market.question[:40] + "..." if len(market.question) > 40 else market.question
                    skip_reason = _zero_bet_skip_message(sizing_mode)
                    logger.info(
                        "SKIP [%s] '%s' -> %s",
                        market.id,
                        question_short,
                        skip_reason,
                        data={
                            "market_id": market.id,
                            "sizing_mode": sizing_mode,
                            "implied_prob": implied_prob,
                            "entry_price": entry_price,
                            "confidence": decision_for_edge.confidence,
                            "edge": edge_value,
                            "kelly_raw": kelly_raw_value,
                            "kelly_fraction_value": kelly_fraction_value,
                            "posterior_for_kelly": posterior_for_kelly,
                            "min_edge_for_kelly": min_edge_for_kelly,
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
                        _record_rejection_reason(rejection_breakdown, "lmsr_gate_blocked")
                        question_short = market.question[:40] + "..." if len(market.question) > 40 else market.question
                        logger.info(
                            "SKIP [%s] '%s' -> LMSR inefficiency too small (|%.4f| < %.4f)",
                            market.id,
                            question_short,
                            ineff_signal,
                            settings.LMSR_MIN_INEFFICIENCY,
                            data={
                                "market_id": market.id,
                                "inefficiency_signal": ineff_signal,
                                "lmsr_execution_price": lmsr_execution_price,
                                "proposed_bet_amount_usdc": proposed_bet_amount,
                                "lmsr_liquidity_param_b": settings.LMSR_LIQUIDITY_PARAM_B,
                                "bayesian_posterior_raw": bayesian_posterior_raw,
                                "bayesian_posterior_applied": bayesian_posterior_applied,
                                "bayesian_update_count": bayesian_update_count,
                                "bayesian_min_updates": settings.BAYESIAN_MIN_UPDATES_FOR_TRADE,
                                "likelihood_ratio": likelihood_ratio,
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
                        ),
                    )
                    trades_skipped_position += 1
                    _record_terminal_outcome(
                        state_manager,
                        market.id,
                        "position_adjustment_blocked",
                    )
                    question_short = market.question[:40] + "..." if len(market.question) > 40 else market.question
                    logger.info(
                        "SKIP [%s] '%s' -> position adjustment blocked",
                        market.id,
                        question_short,
                        data={
                            "market_id": market.id,
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
                        },
                    )
                    continue

                bet_amount = _calculate_bet(settings.MAX_BET_USDC, bet_pct)
                if bet_amount <= 0:
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
                        ),
                    )
                    _record_terminal_outcome(state_manager, market.id, "bet_amount_zero")
                    logger.debug("Skipping market %s: bet_amount=0", market.id)
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
                    _record_rejection_reason(rejection_breakdown, "kelly_sub_floor_skip")
                    question_short = market.question[:40] + "..." if len(market.question) > 40 else market.question
                    logger.info(
                        "SKIP [%s] '%s' -> Kelly bet below min bet floor (raw=$%.2f < min=$%.2f)",
                        market.id,
                        question_short,
                        raw_bet_amount,
                        settings.MIN_BET_USDC,
                        data={
                            "market_id": market.id,
                            "sizing_mode": sizing_mode,
                            "raw_bet_amount_usdc": raw_bet_amount,
                            "min_bet_usdc": settings.MIN_BET_USDC,
                            "kelly_sub_floor_skipped": True,
                            "min_bet_floor_applied": False,
                            "kelly_min_bet_policy": settings.KELLY_MIN_BET_POLICY,
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
                        ),
                    )
                    _record_terminal_outcome(state_manager, market.id, "dry_run")
                    continue

                question_short = market.question[:50] + "..." if len(market.question) > 50 else market.question
                active_market = market
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
                    try:
                        refreshed = kalshi_client.get_market(market.id)
                        if refreshed.outcomes:
                            active_market = refreshed
                            logger.debug(
                                "Using refreshed market snapshot for execution: market=%s",
                                market.id,
                                data={
                                    "market_id": market.id,
                                    "market_data_age_seconds": market_data_age_seconds,
                                    "force_refresh_for_staleness": force_refresh_for_staleness,
                                },
                            )
                    except Exception as exc:
                        if force_refresh_for_staleness:
                            logger.warning(
                                "SKIP [%s] '%s' -> stale market data and refresh failed",
                                market.id,
                                question_short,
                                data={
                                    "market_id": market.id,
                                    "error": str(exc),
                                    "market_data_age_seconds": market_data_age_seconds,
                                    "max_market_data_age_seconds": settings.MAX_MARKET_DATA_AGE_SECONDS,
                                },
                            )
                            trades_skipped_edge += 1
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
                                ),
                            )
                            _record_terminal_outcome(
                                state_manager,
                                market.id,
                                "stale_market_data_refresh_failed",
                            )
                            continue
                        logger.warning(
                            "Pre-order market refresh failed; using scheduled snapshot: market=%s error=%s",
                            market.id,
                            exc,
                            data={
                                "market_id": market.id,
                                "error": str(exc),
                                "market_data_age_seconds": market_data_age_seconds,
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
                        logger.info(
                            "SKIP [%s] '%s' -> refreshed edge gate (%s)",
                            market.id,
                            question_short,
                            refreshed_edge_reason,
                            data={
                                "market_id": market.id,
                                "implied_prob": implied_prob,
                                "confidence": decision_for_edge.confidence,
                                "edge": refreshed_edge_value,
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
                                    ),
                                )
                                _record_terminal_outcome(
                                    state_manager,
                                    market.id,
                                    "orderbook_spread_too_wide",
                                )
                                logger.info(
                                    "SKIP [%s] '%s' -> orderbook precheck failed (best_sell=%.3f > entry=%.3f)",
                                    market.id,
                                    question_short,
                                    best_sell,
                                    entry_price_for_check,
                                    data={
                                        "market_id": market.id,
                                        "best_sell_price": best_sell,
                                        "entry_price": entry_price_for_check,
                                        "option_index": option_index,
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
                        ),
                    )
                    _record_terminal_outcome(
                        state_manager,
                        market.id,
                        "order_price_outside_submission_band",
                    )
                    logger.info(
                        "SKIP [%s] '%s' -> entry price outside submission band (price=%s, min=%.2f, max=%.2f)",
                        market.id,
                        question_short,
                        f"{execution_entry_price:.3f}" if execution_entry_price is not None else "n/a",
                        settings.ORDER_SUBMISSION_MIN_PRICE,
                        settings.ORDER_SUBMISSION_MAX_PRICE,
                        data={
                            "market_id": market.id,
                            "entry_price": execution_entry_price,
                            "min_submission_price": settings.ORDER_SUBMISSION_MIN_PRICE,
                            "max_submission_price": settings.ORDER_SUBMISSION_MAX_PRICE,
                        },
                    )
                    continue
                if trades_attempted >= settings.MAX_TRADES_PER_CYCLE:
                    trades_skipped_edge += 1
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
                    logger.info(
                        "SKIP [%s] '%s' -> market closed before submission (close_time=%s)",
                        market.id,
                        question_short,
                        close_time_for_submission.isoformat(),
                        data={
                            "market_id": market.id,
                            "close_time": close_time_for_submission.isoformat(),
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
                    trades_skipped_balance += 1
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
                        ),
                    )
                    _record_terminal_outcome(state_manager, market.id, "market_closed")
                    continue
                except Exception as order_exc:
                    error_msg = str(order_exc)
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
                            final_reason="order_submission_failed",
                            bet_amount_usdc=bet_amount,
                            order_error=error_msg,
                            market_data_age_seconds=market_data_age_seconds,
                        ),
                    )
                    _record_terminal_outcome(state_manager, market.id, "order_submission_failed")
                    continue  # Continue to next market for other errors

                trades_attempted += 1

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
                    final_reason = "order_canceled_unfilled"
                    terminal_outcome = "order_canceled_unfilled"
                else:
                    trades_filled += 1
                    total_usd_deployed += bet_amount
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
                        market_data_age_seconds=market_data_age_seconds,
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
            cycle_receipt = {
                "cycle": cycle_count,
                "cycle_id": cycle_id,
                "duration_ms": round(cycle_duration, 2),
                "fetched_markets": fetched_count,
                "eligible_markets": len(markets),
                "analysis_candidates": analysis_candidates_count,
                "analyzed_markets": markets_analyzed,
                "decisions_made": decisions_made,
                "refined_markets": markets_refined,
                "execution_candidates": execution_candidates,
                "order_attempts": trades_attempted,
                "orders_filled": trades_filled,
                "orders_canceled_unfilled": trades_canceled_unfilled,
                "total_usd_deployed": round(total_usd_deployed, 2),
                "skip_counts": {
                    "no_trade": trades_skipped_no_trade,
                    "confidence": trades_skipped_confidence,
                    "edge": trades_skipped_edge,
                    "position": trades_skipped_position,
                    "balance": trades_skipped_balance,
                    "kelly_sub_floor": trades_skipped_kelly_sub_floor,
                },
                "rejection_breakdown": rejection_breakdown,
                "analysis_only_mode": analysis_only_mode,
            }
            logger.info(
                "Cycle receipt",
                data={"cycle_receipt": cycle_receipt},
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
                "Rejections summary: %s",
                ", ".join(
                    f"{reason}={count}"
                    for reason, count in sorted(rejection_breakdown.items())
                )
                if rejection_breakdown
                else "none",
                data={"rejection_breakdown": rejection_breakdown},
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
                "flip_guard_blocked=%d execution_candidates=%d order_attempts=%d "
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
                    "order_attempts": trades_attempted,
                    "orders_filled": trades_filled,
                    "orders_canceled_unfilled": trades_canceled_unfilled,
                    "total_usd_deployed": round(total_usd_deployed, 2),
                    "skipped_kelly_sub_floor": trades_skipped_kelly_sub_floor,
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
                    "skipped_no_trade": trades_skipped_no_trade,
                    "skipped_confidence": trades_skipped_confidence,
                    "skipped_edge": trades_skipped_edge,
                    "skipped_kelly_sub_floor": trades_skipped_kelly_sub_floor,
                    "skipped_balance": trades_skipped_balance,
                    "skipped_position": trades_skipped_position,
                    "analysis_only_mode": analysis_only_mode,
                    "price_buckets": price_bucket_stats,
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


if __name__ == "__main__":
    main()
