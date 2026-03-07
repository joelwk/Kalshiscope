from __future__ import annotations

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
from market_scheduler import MarketScheduler
from market_state import MarketStateManager
from models import (
    InsufficientBalanceError,
    Market,
    MarketOutcome,
    MarketState,
    OrderRequest,
    Position,
    TradeDecision,
)
from predictbase_client import PredictBaseClient
from refinement import RefinementStrategy
from research_profiles import build_market_search_config, market_category_flags
from score_engine import compute_final_score
from web3_client import Web3Client

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


def _normalize_outcome_key(outcome: str | None) -> str:
    return re.sub(r"\s+", " ", (outcome or "").strip()).lower()


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
    min_close_days=None,
    max_close_days=None,
    stats: dict[str, int] | None = None,
):
    """Filter markets based on liquidity, category, and close date constraints."""
    filtered = []
    skipped_liquidity = 0
    skipped_allowlist = 0
    skipped_blocklist = 0
    skipped_close_too_soon = 0
    skipped_close_too_far = 0
    skipped_closed_now = 0
    skipped_resolved = 0

    now = datetime.now(timezone.utc)
    min_close_date = now + timedelta(days=min_close_days) if min_close_days else None
    max_close_date = now + timedelta(days=max_close_days) if max_close_days else None

    for market in markets:
        close_time = market.close_time
        if close_time and close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=timezone.utc)
        if market.liquidity_usdc is not None and market.liquidity_usdc < min_liquidity:
            skipped_liquidity += 1
            continue
        if allowlist and (market.category not in allowlist):
            skipped_allowlist += 1
            continue
        if blocklist and (market.category in blocklist):
            skipped_blocklist += 1
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
        "Market filtering complete: kept=%d, skipped_liquidity=%d, skipped_allowlist=%d, "
        "skipped_blocklist=%d, skipped_resolved=%d, skipped_close_too_soon=%d, "
        "skipped_close_too_far=%d, skipped_closed_now=%d",
        len(filtered),
        skipped_liquidity,
        skipped_allowlist,
        skipped_blocklist,
        skipped_resolved,
        skipped_close_too_soon,
        skipped_close_too_far,
        skipped_closed_now,
        data={
            "kept": len(filtered),
            "skipped_liquidity": skipped_liquidity,
            "skipped_allowlist": skipped_allowlist,
            "skipped_blocklist": skipped_blocklist,
            "skipped_resolved": skipped_resolved,
            "skipped_close_too_soon": skipped_close_too_soon,
            "skipped_close_too_far": skipped_close_too_far,
            "skipped_closed_now": skipped_closed_now,
        },
    )
    if stats is not None:
        stats.update(
            {
                "kept": len(filtered),
                "skipped_liquidity": skipped_liquidity,
                "skipped_allowlist": skipped_allowlist,
                "skipped_blocklist": skipped_blocklist,
                "skipped_resolved": skipped_resolved,
                "skipped_close_too_soon": skipped_close_too_soon,
                "skipped_close_too_far": skipped_close_too_far,
                "skipped_closed_now": skipped_closed_now,
            }
        )
    return filtered


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
) -> float:
    min_edge = settings.MIN_EDGE
    if implied_prob < settings.LOW_PRICE_THRESHOLD:
        min_edge = max(min_edge, settings.LOW_PRICE_MIN_EDGE)
    return min_edge


def _passes_edge_threshold(
    implied_prob: float | None,
    decision: TradeDecision,
    settings: Settings,
) -> tuple[bool, float | None, str]:
    if implied_prob is None:
        if settings.REQUIRE_IMPLIED_PRICE:
            return False, None, "missing implied probability"
        return True, None, ""
    edge = decision.confidence - implied_prob
    min_edge = _edge_threshold_for_market(implied_prob, settings)
    if edge < min_edge:
        return False, edge, f"edge {edge:.2f} below min {min_edge:.2f}"
    return True, edge, ""


def _adjust_bet_size_for_edge(
    decision: TradeDecision,
    implied_prob: float | None,
    edge: float | None,
    settings: Settings,
) -> float:
    if edge is None or implied_prob is None:
        return decision.bet_size_pct
    min_edge = _edge_threshold_for_market(implied_prob, settings)
    edge_over = edge - min_edge
    if edge_over <= 0:
        return 0.0
    scaling_range = max(settings.EDGE_SCALING_RANGE, 0.01)
    scale = min(1.0, edge_over / scaling_range)
    bet_pct = decision.bet_size_pct * scale
    if implied_prob < settings.LOW_PRICE_THRESHOLD:
        bet_pct *= settings.LOW_PRICE_BET_PENALTY
    return max(0.0, min(1.0, bet_pct))


def _max_confidence_for_market(market: Market | None, settings: Settings) -> float:
    if not market:
        return 1.0
    is_sports, is_esports = market_category_flags(market)
    if is_sports:
        return settings.MAX_SPORTS_CONFIDENCE
    if is_esports:
        return settings.MAX_ESPORTS_CONFIDENCE
    return 1.0


def _effective_position_override_threshold(
    market: Market | None,
    settings: Settings,
) -> float:
    cap = _max_confidence_for_market(market, settings)
    return min(settings.HIGH_CONFIDENCE_POSITION_OVERRIDE, cap)


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
    if market.close_time is None:
        return settings.KELLY_FRACTION_DEFAULT
    close_time = market.close_time
    if close_time.tzinfo is None:
        close_time = close_time.replace(tzinfo=timezone.utc)
    horizon_seconds = (close_time - datetime.now(timezone.utc)).total_seconds()
    short_horizon_seconds = max(0, settings.KELLY_FRACTION_SHORT_HORIZON_HOURS) * 3600
    if short_horizon_seconds > 0 and horizon_seconds <= short_horizon_seconds:
        return settings.KELLY_FRACTION_SHORT_HORIZON
    return settings.KELLY_FRACTION_DEFAULT


def _sizing_mode_label(kelly_enabled: bool) -> str:
    return "kelly" if kelly_enabled else "edge_scaling"


def _zero_bet_skip_message(sizing_mode: str) -> str:
    if sizing_mode == "kelly":
        return "bet size reduced to zero by Kelly sizing"
    return "bet size reduced to zero by edge scaling"


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
        edge_delta = new_edge - anchor_edge
        edge_gain_ok = edge_delta >= settings.FLIP_GUARD_MIN_EDGE_GAIN

    abs_conf_ok = decision.confidence >= settings.FLIP_GUARD_MIN_ABS_CONFIDENCE
    conf_gain_ok = confidence_delta >= settings.FLIP_GUARD_MIN_CONF_GAIN
    evidence_quality = decision.evidence_quality or 0.0
    evidence_ok = evidence_quality >= settings.FLIP_GUARD_MIN_EVIDENCE_QUALITY

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
    }

    if abs_conf_ok and conf_gain_ok and edge_gain_ok and evidence_ok:
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
    cap_reason = "sports" if is_sports else ("esports" if is_esports else None)

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

    if existing_position.total_amount_usdc >= settings.MAX_POSITION_PER_MARKET_USDC:
        return False, 0.0, "max_position_reached"

    remaining = (
        settings.MAX_POSITION_PER_MARKET_USDC
        - existing_position.total_amount_usdc
    )
    if remaining <= 0:
        return False, 0.0, "no_remaining_capacity"

    override_threshold = _effective_position_override_threshold(market, settings)
    is_high_confidence = decision.confidence >= override_threshold

    # Otherwise, require minimum confidence increase over existing position
    confidence_increase = decision.confidence - existing_position.avg_confidence
    position_fill_ratio = (
        existing_position.total_amount_usdc / settings.MAX_POSITION_PER_MARKET_USDC
    )
    scaled_increase_threshold = settings.MIN_CONFIDENCE_INCREASE_FOR_ADD * max(
        0.25,
        min(1.0, position_fill_ratio),
    )
    meets_increase_threshold = confidence_increase >= scaled_increase_threshold

    if not is_high_confidence and not meets_increase_threshold:
        return False, 0.0, "insufficient_confidence_increase"

    reason = (
        "high_confidence_override"
        if is_high_confidence
        else "confidence_increase_threshold_met"
    )
    return True, min(decision.bet_size_pct, remaining / settings.MAX_BET_USDC), reason


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
            "min_liquidity_usdc": settings.MIN_LIQUIDITY_USDC,
            "poll_interval_sec": settings.POLL_INTERVAL_SEC,
            "market_min_close_days": settings.MARKET_MIN_CLOSE_DAYS,
            "market_max_close_days": settings.MARKET_MAX_CLOSE_DAYS,
            "execute_onchain": settings.EXECUTE_ONCHAIN,
            "auto_approve_usdc": settings.AUTO_APPROVE_USDC,
            "grok_model": settings.GROK_MODEL,
            "categories_allowlist": settings.MARKET_CATEGORIES_ALLOWLIST,
            "categories_blocklist": settings.MARKET_CATEGORIES_BLOCKLIST,
            "score_gate_mode": settings.SCORE_GATE_MODE,
            "score_gate_threshold": settings.SCORE_GATE_THRESHOLD,
            "bayesian_enabled": settings.BAYESIAN_ENABLED,
            "lmsr_enabled": settings.LMSR_ENABLED,
            "kelly_sizing_enabled": settings.KELLY_SIZING_ENABLED,
            "kelly_fraction_default": settings.KELLY_FRACTION_DEFAULT,
            "kelly_fraction_short_horizon_hours": settings.KELLY_FRACTION_SHORT_HORIZON_HOURS,
            "kelly_fraction_short_horizon": settings.KELLY_FRACTION_SHORT_HORIZON,
            "parallel_analysis_enabled": settings.PARALLEL_ANALYSIS_ENABLED,
            "analysis_max_workers": settings.ANALYSIS_MAX_WORKERS,
            "pre_order_market_refresh": settings.PRE_ORDER_MARKET_REFRESH,
            "orderbook_precheck_enabled": settings.ORDERBOOK_PRECHECK_ENABLED,
            "orderbook_precheck_min_confidence": settings.ORDERBOOK_PRECHECK_MIN_CONFIDENCE,
            "calibration_mode_enabled": settings.CALIBRATION_MODE_ENABLED,
            "calibration_min_samples": settings.CALIBRATION_MIN_SAMPLES,
            "opposite_outcome_strategy": settings.OPPOSITE_OUTCOME_STRATEGY,
            "flip_guard_enabled": settings.FLIP_GUARD_ENABLED,
            "flip_guard_min_abs_confidence": settings.FLIP_GUARD_MIN_ABS_CONFIDENCE,
            "flip_guard_min_conf_gain": settings.FLIP_GUARD_MIN_CONF_GAIN,
            "flip_guard_min_edge_gain": settings.FLIP_GUARD_MIN_EDGE_GAIN,
            "flip_guard_min_evidence_quality": settings.FLIP_GUARD_MIN_EVIDENCE_QUALITY,
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

        last_analysis = state.last_analysis
        if last_analysis.tzinfo is None:
            last_analysis = last_analysis.replace(tzinfo=timezone.utc)

        urgent_cutoff = now_utc + timedelta(days=settings.URGENT_REANALYSIS_DAYS_BEFORE_CLOSE)
        is_urgent = bool(close_time and close_time <= urgent_cutoff)
        cooldown_hours = (
            settings.URGENT_REANALYSIS_COOLDOWN_HOURS
            if is_urgent
            else settings.REANALYSIS_COOLDOWN_HOURS
        )
        next_allowed = last_analysis + timedelta(hours=cooldown_hours)
        remaining_seconds = (next_allowed - now_utc).total_seconds()
        if remaining_seconds <= 0:
            return 1
        if earliest_remaining is None or remaining_seconds < earliest_remaining:
            earliest_remaining = remaining_seconds

    if earliest_remaining is None:
        return None
    capped = min(earliest_remaining, float(_ADAPTIVE_SLEEP_CAP_SECONDS))
    return max(1, int(capped))


def _build_grok_client_for_worker(settings: Settings) -> GrokClient:
    """Create an isolated Grok client for threaded analysis workers."""
    return GrokClient(
        api_key=settings.XAI_API_KEY,
        model=settings.GROK_MODEL,
        min_bet_usdc=settings.MIN_BET_USDC,
        max_bet_usdc=settings.MAX_BET_USDC,
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
    decision = grok_client.analyze_market(
        market,
        search_config=search_config,
        previous_analysis=previous_analysis,
    )

    anchor_outcome: str | None = None
    if anchor_analysis and anchor_analysis.get("outcome") is not None:
        anchor_outcome = str(anchor_analysis["outcome"]).strip() or None

    refinement = RefinementStrategy(
        market=market,
        urgent_days_before_close=settings.URGENT_REANALYSIS_DAYS_BEFORE_CLOSE,
    )
    was_refined = False
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
    refinement = RefinementStrategy(
        urgent_days_before_close=settings.URGENT_REANALYSIS_DAYS_BEFORE_CLOSE,
    )

    grok_client = GrokClient(
        api_key=settings.XAI_API_KEY,
        model=settings.GROK_MODEL,
        min_bet_usdc=settings.MIN_BET_USDC,
        max_bet_usdc=settings.MAX_BET_USDC,
    )
    logger.debug("Grok client initialized with model=%s", settings.GROK_MODEL)

    # Initialize Web3Client first to get wallet address for PredictBase
    web3_client = None
    wallet_address = None
    if settings.EXECUTE_ONCHAIN or settings.AUTO_APPROVE_USDC:
        if not settings.USDC_TOKEN_ADDRESS:
            logger.error("USDC_TOKEN_ADDRESS is required for on-chain execution")
            raise ValueError("USDC_TOKEN_ADDRESS is required for on-chain execution")
        web3_client = Web3Client(
            rpc_url=settings.ALCHEMY_RPC_URL,
            private_key=settings.WALLET_PRIVATE_KEY,
            usdc_token_address=settings.USDC_TOKEN_ADDRESS,
            chain_id=settings.CHAIN_ID,
        )
        wallet_address = web3_client.address
        logger.info(
            "Web3 client initialized: address=%s, chain_id=%s",
            web3_client.address,
            web3_client.chain_id,
        )

    predictbase_client = PredictBaseClient(
        base_url=settings.PREDICTBASE_API_BASE_URL,
        api_key=settings.PREDICTBASE_API_KEY,
        api_key_header=settings.PREDICTBASE_API_KEY_HEADER,
        api_key_prefix=settings.PREDICTBASE_API_KEY_PREFIX,
        wallet_address=wallet_address,
        slippage_confidence_threshold=settings.SLIPPAGE_CONFIDENCE_THRESHOLD,
    )
    logger.debug("PredictBase client initialized with base_url=%s", settings.PREDICTBASE_API_BASE_URL)

    if web3_client and settings.AUTO_APPROVE_USDC:
        if not settings.PREDICTBASE_CONTRACT_ADDRESS:
            logger.error("PREDICTBASE_CONTRACT_ADDRESS is required for approvals")
            raise ValueError("PREDICTBASE_CONTRACT_ADDRESS is required for approvals")

        allowance = web3_client.get_allowance(settings.PREDICTBASE_CONTRACT_ADDRESS)
        target_allowance = int(settings.MAX_BET_USDC * (10**settings.USDC_DECIMALS))
        logger.debug(
            "Checking USDC allowance: current=%d, target=%d",
            allowance,
            target_allowance,
        )

        if allowance < target_allowance:
            if settings.DRY_RUN:
                logger.info(
                    "DRY_RUN: would approve USDC for contract=%s",
                    settings.PREDICTBASE_CONTRACT_ADDRESS,
                )
            else:
                logger.info(
                    "Approving USDC for contract=%s, amount=%.2f",
                    settings.PREDICTBASE_CONTRACT_ADDRESS,
                    settings.MAX_BET_USDC,
                )
                web3_client.approve_usdc(
                    settings.PREDICTBASE_CONTRACT_ADDRESS,
                    settings.MAX_BET_USDC,
                    decimals=settings.USDC_DECIMALS,
                )

    logger.info("PredictBot started (dry_run=%s)", settings.DRY_RUN)
    cycle_count = 0

    while True:
        cycle_count += 1
        cycle_id = set_correlation_id()
        cycle_start = time.monotonic()
        sleep_seconds = settings.POLL_INTERVAL_SEC

        logger.info("Starting bot cycle #%d", cycle_count)

        try:
            markets = predictbase_client.get_markets()
            fetched_count = len(markets)
            logger.info("Fetched %d raw markets", fetched_count)

            filter_stats: dict[str, int] = {}
            markets = _filter_markets(
                markets,
                settings.MIN_LIQUIDITY_USDC,
                settings.MARKET_CATEGORIES_ALLOWLIST,
                settings.MARKET_CATEGORIES_BLOCKLIST,
                min_close_days=settings.MARKET_MIN_CLOSE_DAYS,
                max_close_days=settings.MARKET_MAX_CLOSE_DAYS,
                stats=filter_stats,
            )
            logger.info("Filtered to %d eligible markets", len(markets))

            markets = _dedupe_markets_by_matchup(markets)

            markets = scheduler.prioritize_markets(markets, state_manager)

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
            trades_skipped_confidence = 0
            trades_skipped_balance = 0
            trades_skipped_no_trade = 0
            trades_skipped_edge = 0
            trades_skipped_position = 0
            scheduler_skipped_closed = 0
            scheduler_skipped_recently = 0
            scheduler_skipped_other = 0
            markets_analyzed = 0
            markets_refined = 0
            markets_passed_edge = 0
            score_gate_blocked = 0
            flip_guard_triggered = 0
            flip_guard_blocked = 0
            outcome_mismatch_blocked = 0
            analysis_only_mode = False  # Set True when balance is insufficient
            price_bucket_stats = {
                _PRICE_BUCKET_LOW: 0,
                _PRICE_BUCKET_MID: 0,
                _PRICE_BUCKET_HIGH: 0,
            }
            calibration_samples: list[dict[str, Any]] = []

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
                analysis_candidates.append(
                    {
                        "market": market,
                        "state": state,
                        "anchor_analysis": anchor_analysis,
                    }
                )

            analysis_results: dict[str, dict[str, Any]] = {}
            if settings.PARALLEL_ANALYSIS_ENABLED and len(analysis_candidates) > 1:
                max_workers = max(
                    1,
                    min(settings.ANALYSIS_MAX_WORKERS, len(analysis_candidates)),
                )
                logger.info(
                    "Parallel analysis enabled: candidates=%d workers=%d",
                    len(analysis_candidates),
                    max_workers,
                    data={
                        "candidates": len(analysis_candidates),
                        "workers": max_workers,
                    },
                )
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_market = {}
                    for candidate in analysis_candidates:
                        worker_client = _build_grok_client_for_worker(settings)
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
            else:
                for candidate in analysis_candidates:
                    market = candidate["market"]
                    try:
                        analysis_results[market.id] = _analyze_market_candidate(
                            market=market,
                            state=candidate["state"],
                            anchor_analysis=candidate["anchor_analysis"],
                            settings=settings,
                            grok_client=grok_client,
                        )
                    except Exception as exc:
                        logger.error(
                            "Failed to analyze market %s: %s",
                            market.id,
                            exc,
                            data={"market_id": market.id, "error": str(exc)},
                        )

            for candidate in analysis_candidates:
                market = candidate["market"]
                state = candidate["state"]
                market_start = time.monotonic()
                analysis_result = analysis_results.get(market.id)
                if analysis_result is None:
                    continue
                markets_analyzed += 1
                decision = analysis_result["decision"]
                was_refined = analysis_result["was_refined"]
                if was_refined:
                    markets_refined += 1
                refinement_reason_text = analysis_result["refinement_reason_text"]
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

                log_trade_decision(
                    market_id=market.id,
                    question=market.question,
                    decision=decision.model_dump(),
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

                if not decision.should_trade:
                    trades_skipped_no_trade += 1
                    question_short = market.question[:40] + "..." if len(market.question) > 40 else market.question
                    logger.info(
                        "SKIP [%s] '%s' -> no trade recommended",
                        market.id,
                        question_short,
                    )
                    continue

                if decision.confidence < settings.MIN_CONFIDENCE:
                    trades_skipped_confidence += 1
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
                                effective_confidence = bayesian_posterior_applied
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
                bucket = _price_bucket(implied_prob, settings)
                price_bucket_stats[bucket] += 1
                edge_ok, edge_value, edge_reason = _passes_edge_threshold(
                    implied_prob,
                    decision_for_edge,
                    settings,
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
                kelly_path_active = settings.KELLY_SIZING_ENABLED and implied_prob is not None
                sizing_mode = _sizing_mode_label(kelly_path_active)
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

                if settings.KELLY_SIZING_ENABLED and implied_prob is not None:
                    if posterior_for_kelly is None:
                        posterior_for_kelly = (
                            bayesian_posterior_applied
                            if bayesian_posterior_applied is not None
                            else effective_confidence
                        )
                    if kelly_fraction_value is None:
                        kelly_fraction_value = _kelly_fraction_for_market_horizon(market, settings)
                    min_edge_for_kelly = _edge_threshold_for_market(implied_prob, settings)
                    adjusted_bet_pct = kelly_bet_pct(
                        posterior=posterior_for_kelly,
                        market_price=implied_prob,
                        fraction=kelly_fraction_value,
                        min_edge=min_edge_for_kelly,
                    )
                else:
                    adjusted_bet_pct = _adjust_bet_size_for_edge(
                        decision_for_edge,
                        implied_prob,
                        edge_value,
                        settings,
                    )
                if adjusted_bet_pct <= 0:
                    trades_skipped_edge += 1
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
                        execution_audit={
                            "decision_phase": "post_sizing",
                            "sizing_mode": sizing_mode,
                            "adjusted_bet_pct": adjusted_bet_pct,
                            "bet_amount_usdc": 0.0,
                            "kelly_raw": kelly_raw_value,
                            "kelly_fraction_value": kelly_fraction_value,
                            "posterior_for_kelly": posterior_for_kelly,
                            "bayesian_posterior_raw": bayesian_posterior_raw,
                            "bayesian_posterior_applied": bayesian_posterior_applied,
                            "bayesian_applied": bayesian_posterior_applied is not None,
                            "bayesian_update_count": bayesian_update_count,
                            "bayesian_min_updates": settings.BAYESIAN_MIN_UPDATES_FOR_TRADE,
                            "likelihood_ratio": likelihood_ratio,
                            "implied_prob_market": implied_prob,
                            "min_edge_for_kelly": min_edge_for_kelly,
                            "edge_market": edge_value,
                            "lmsr_execution_price": lmsr_execution_price,
                            "lmsr_inefficiency_signal": ineff_signal,
                            "lmsr_liquidity_param_b": settings.LMSR_LIQUIDITY_PARAM_B,
                        },
                    )
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
                            execution_audit={
                                "decision_phase": "post_lmsr_gate",
                                "lmsr_gate_decision": "blocked",
                                "lmsr_execution_price": lmsr_execution_price,
                                "lmsr_inefficiency_signal": ineff_signal,
                                "lmsr_min_inefficiency": settings.LMSR_MIN_INEFFICIENCY,
                                "proposed_bet_amount_usdc": proposed_bet_amount,
                                "lmsr_liquidity_param_b": settings.LMSR_LIQUIDITY_PARAM_B,
                                "bayesian_posterior_raw": bayesian_posterior_raw,
                                "bayesian_posterior_applied": bayesian_posterior_applied,
                                "bayesian_applied": bayesian_posterior_applied is not None,
                                "bayesian_update_count": bayesian_update_count,
                                "bayesian_min_updates": settings.BAYESIAN_MIN_UPDATES_FOR_TRADE,
                                "likelihood_ratio": likelihood_ratio,
                            },
                        )
                        continue

                markets_passed_edge += 1

                log_trade_decision(
                    market_id=market.id,
                    question=market.question,
                    decision=decision_for_edge.model_copy(
                        update={"bet_size_pct": adjusted_bet_pct}
                    ).model_dump(),
                    execution_audit={
                        "decision_phase": "post_sizing",
                        "sizing_mode": sizing_mode,
                        "position_decision": "pending",
                        "adjusted_bet_pct": adjusted_bet_pct,
                        "bet_amount_usdc": None,
                        "proposed_bet_amount_usdc": proposed_bet_amount,
                        "kelly_raw": kelly_raw_value,
                        "kelly_fraction_value": kelly_fraction_value,
                        "posterior_for_kelly": posterior_for_kelly,
                        "bayesian_posterior_raw": bayesian_posterior_raw,
                        "bayesian_posterior_applied": bayesian_posterior_applied,
                        "bayesian_applied": bayesian_posterior_applied is not None,
                        "bayesian_update_count": bayesian_update_count,
                        "bayesian_min_updates": settings.BAYESIAN_MIN_UPDATES_FOR_TRADE,
                        "likelihood_ratio": likelihood_ratio,
                        "implied_prob_market": implied_prob,
                        "min_edge_for_kelly": min_edge_for_kelly,
                        "edge_market": edge_value,
                        "lmsr_execution_price": lmsr_execution_price,
                        "lmsr_inefficiency_signal": ineff_signal,
                        "lmsr_liquidity_param_b": settings.LMSR_LIQUIDITY_PARAM_B,
                    },
                )

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

                should_add, bet_pct, position_reason = _should_adjust_position(
                    decision_for_edge.model_copy(update={"bet_size_pct": adjusted_bet_pct}),
                    market,
                    existing_position,
                    state,
                    settings,
                )
                log_trade_decision(
                    market_id=market.id,
                    question=market.question,
                    decision=decision_for_edge.model_copy(
                        update={"bet_size_pct": adjusted_bet_pct}
                    ).model_dump(),
                    execution_audit={
                        "decision_phase": "post_position_gate",
                        "sizing_mode": sizing_mode,
                        "position_decision": "allowed" if should_add else "blocked",
                        "position_decision_reason": (
                            position_reason if not should_add else None
                        ),
                        "adjusted_bet_pct": adjusted_bet_pct,
                        "post_position_bet_pct": bet_pct,
                        "bet_amount_usdc": None,
                        "proposed_bet_amount_usdc": proposed_bet_amount,
                        "kelly_raw": kelly_raw_value,
                        "kelly_fraction_value": kelly_fraction_value,
                        "posterior_for_kelly": posterior_for_kelly,
                        "bayesian_posterior_raw": bayesian_posterior_raw,
                        "bayesian_posterior_applied": bayesian_posterior_applied,
                        "bayesian_applied": bayesian_posterior_applied is not None,
                        "bayesian_update_count": bayesian_update_count,
                        "bayesian_min_updates": settings.BAYESIAN_MIN_UPDATES_FOR_TRADE,
                        "likelihood_ratio": likelihood_ratio,
                        "implied_prob_market": implied_prob,
                        "min_edge_for_kelly": min_edge_for_kelly,
                        "edge_market": edge_value,
                        "lmsr_execution_price": lmsr_execution_price,
                        "lmsr_inefficiency_signal": ineff_signal,
                        "lmsr_liquidity_param_b": settings.LMSR_LIQUIDITY_PARAM_B,
                    },
                )
                if not should_add:
                    trades_skipped_position += 1
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
                    logger.debug("Skipping market %s: bet_amount=0", market.id)
                    continue

                # Apply bet floor: ensure at least MIN_BET_USDC for any trade recommendation
                if bet_amount < settings.MIN_BET_USDC:
                    original_bet = bet_amount
                    bet_amount = settings.MIN_BET_USDC
                    logger.debug(
                        "Applied bet floor: market=%s, original=$%.2f, adjusted=$%.2f",
                        market.id,
                        original_bet,
                        bet_amount,
                    )

                order = OrderRequest(
                    market_id=market.id,
                    outcome=decision.outcome,
                    amount_usdc=bet_amount,
                    confidence=decision_for_edge.confidence,
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
                    )
                    trades_skipped_balance += 1
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
                            "amount_usdc": bet_amount,
                            "confidence": decision_for_edge.confidence,
                            "reasoning": decision.reasoning,
                        },
                    )
                    trades_attempted += 1
                    continue

                question_short = market.question[:50] + "..." if len(market.question) > 50 else market.question
                active_market = market
                if settings.PRE_ORDER_MARKET_REFRESH:
                    try:
                        refreshed = predictbase_client.get_market(market.id)
                        if refreshed.outcomes:
                            active_market = refreshed
                            logger.debug(
                                "Using refreshed market snapshot for execution: market=%s",
                                market.id,
                                data={"market_id": market.id},
                            )
                    except Exception as exc:
                        logger.warning(
                            "Pre-order market refresh failed; using scheduled snapshot: market=%s error=%s",
                            market.id,
                            exc,
                            data={"market_id": market.id, "error": str(exc)},
                        )
                if active_market is not market:
                    entry_price = _get_outcome_entry_price(active_market, decision.outcome)
                    implied_prob = _get_implied_probability(active_market, decision.outcome)

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
                            orderbook = predictbase_client.get_market_orderbook(active_market.id)
                            best_sell = _best_orderbook_sell_price(orderbook, option_index)
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
                logger.info(
                    "TRADE: [%s] '%s' -> %s @ $%.2f (conf=%.2f)",
                    market.id,
                    question_short,
                    decision.outcome,
                    bet_amount,
                    decision_for_edge.confidence,
                )

                slippage_pct = (
                    settings.SLIPPAGE_PCT
                    if decision_for_edge.confidence >= settings.SLIPPAGE_CONFIDENCE_THRESHOLD
                    else 0.0
                )

                try:
                    order_response = predictbase_client.submit_order(
                        order,
                        market=active_market,
                        slippage_pct=slippage_pct,
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
                    continue  # Continue analyzing remaining markets
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

                if settings.EXECUTE_ONCHAIN and web3_client:
                    if order_response.onchain_payload:
                        # Only validate wallet balance when an on-chain payload is actually present.
                        # If PredictBase fulfills the order using deposited funds, the on-chain
                        # wallet USDC balance is irrelevant and should not trigger warnings.
                        if not web3_client.has_sufficient_balance(
                            bet_amount, decimals=settings.USDC_DECIMALS
                        ):
                            logger.warning(
                                "Skipping on-chain execution due to low wallet USDC balance: "
                                "order=%s market=%s needed=%.2f",
                                order_response.id,
                                market.id,
                                bet_amount,
                                data={
                                    "order_id": order_response.id,
                                    "market_id": market.id,
                                    "required_usdc": bet_amount,
                                },
                            )
                            continue
                        logger.info(
                            "Executing on-chain trade for order=%s",
                            order_response.id,
                        )
                        tx_hash = web3_client.send_onchain_payload(
                            order_response.onchain_payload
                        )
                        logger.info(
                            "On-chain trade submitted: order=%s, tx_hash=%s",
                            order_response.id,
                            tx_hash,
                        )
                    else:
                        logger.warning(
                            "No on-chain payload found for order=%s",
                            order_response.id,
                        )

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
                "Cycle funnel: fetched=%d filtered=%d skipped_resolved=%d scheduler_skips=%d "
                "analyzed=%d refined=%d flip_guard_triggered=%d flip_guard_blocked=%d edge_pass=%d traded=%d",
                fetched_count,
                len(markets),
                filter_stats.get("skipped_resolved", 0),
                scheduler_skipped_closed + scheduler_skipped_recently + scheduler_skipped_other,
                markets_analyzed,
                markets_refined,
                flip_guard_triggered,
                flip_guard_blocked,
                markets_passed_edge,
                trades_attempted,
                data={
                    "fetched": fetched_count,
                    "filtered": len(markets),
                    "skipped_resolved": filter_stats.get("skipped_resolved", 0),
                    "scheduler_skipped_closed": scheduler_skipped_closed,
                    "scheduler_skipped_recently_analyzed": scheduler_skipped_recently,
                    "scheduler_skipped_other": scheduler_skipped_other,
                    "analyzed": markets_analyzed,
                    "refined": markets_refined,
                    "flip_guard_triggered": flip_guard_triggered,
                    "flip_guard_blocked": flip_guard_blocked,
                    "outcome_mismatch_blocked": outcome_mismatch_blocked,
                    "edge_passed": markets_passed_edge,
                    "traded": trades_attempted,
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
                    "markets_analyzed": len(markets),
                    "markets_fetched": fetched_count,
                    "filter_stats": filter_stats,
                    "skipped_resolved": filter_stats.get("skipped_resolved", 0),
                    "scheduler_skipped_closed": scheduler_skipped_closed,
                    "scheduler_skipped_recently_analyzed": scheduler_skipped_recently,
                    "scheduler_skipped_other": scheduler_skipped_other,
                    "markets_passed_to_grok": markets_analyzed,
                    "markets_refined": markets_refined,
                    "flip_guard_triggered": flip_guard_triggered,
                    "flip_guard_blocked": flip_guard_blocked,
                    "outcome_mismatch_blocked": outcome_mismatch_blocked,
                    "markets_passed_edge": markets_passed_edge,
                    "score_gate_blocked": score_gate_blocked,
                    "trades_attempted": trades_attempted,
                    "skipped_no_trade": trades_skipped_no_trade,
                    "skipped_confidence": trades_skipped_confidence,
                    "skipped_edge": trades_skipped_edge,
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
