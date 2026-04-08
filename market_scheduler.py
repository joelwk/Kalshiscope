from __future__ import annotations

from datetime import datetime, timedelta, timezone

from logging_config import get_logger
from market_state import MarketStateManager
from models import Market, MarketState

logger = get_logger(__name__)

_NON_ACTIONABLE_TERMINAL_COOLDOWN_HOURS = 0.25
_NON_ACTIONABLE_TERMINAL_COOLDOWN_HOURS_SECOND = 2.0
_NON_ACTIONABLE_TERMINAL_COOLDOWN_HOURS_THIRD = 4.0
_NON_ACTIONABLE_TERMINAL_OUTCOMES = {
    "abstain_low_evidence",
    "analysis_failure",
    "analysis_only_insufficient_balance",
    "bet_amount_zero",
    "confidence_below_min",
    "evidence_quality_below_min",
    "edge_gate_blocked",
    "kelly_sub_floor_skip",
    "lmsr_gate_blocked",
    "max_trades_per_cycle_reached",
    "no_trade_recommended",
    "orderbook_spread_too_wide",
    "order_price_outside_submission_band",
    "position_adjustment_blocked",
    "score_gate_blocked",
    "stale_market_data_refresh_failed",
    "uniform_implied_probability",
    "zero_bet_after_sizing",
}
_HARD_BACKOFF_TERMINAL_OUTCOMES = {
    "abstain_low_evidence",
    "analysis_failure",
}
_PRIORITY_STALENESS_WEIGHT = 2.0
_PRIORITY_PRICE_OPPORTUNITY_WEIGHT = 1.5
_PRIORITY_LIQUIDITY_WEIGHT = 0.75
_PRIORITY_URGENT_BONUS = 1.0
_PRIORITY_NON_ACTIONABLE_PENALTY = 0.5
_LIQUIDITY_NORMALIZATION_CAP_USDC = 500.0
_MAX_STALENESS_HOURS = 24.0 * 7.0
_FILL_FAILURE_COOLDOWN_THRESHOLD = 3
_FILL_FAILURE_COOLDOWN_MULTIPLIER = 2.0


def _normalize_terminal_outcome(value: str | None) -> str:
    return (value or "").strip().lower()


def _base_reanalysis_cooldown_hours(
    *,
    is_urgent: bool,
    reanalysis_cooldown_hours: int,
    urgent_reanalysis_cooldown_hours: int,
) -> float:
    return float(
        urgent_reanalysis_cooldown_hours
        if is_urgent
        else reanalysis_cooldown_hours
    )


def resolve_reanalysis_cooldown_hours(
    market: Market,
    state: MarketState | None,
    *,
    reanalysis_cooldown_hours: int,
    urgent_days_before_close: int,
    urgent_reanalysis_cooldown_hours: int,
    now: datetime | None = None,
) -> float:
    now_utc = now or datetime.now(timezone.utc)
    is_urgent = False
    if market.close_time:
        close_time = _normalize_timestamp(market.close_time)
        urgent_cutoff = now_utc + timedelta(days=urgent_days_before_close)
        is_urgent = close_time <= urgent_cutoff
    base_cooldown = _base_reanalysis_cooldown_hours(
        is_urgent=is_urgent,
        reanalysis_cooldown_hours=reanalysis_cooldown_hours,
        urgent_reanalysis_cooldown_hours=urgent_reanalysis_cooldown_hours,
    )
    terminal_outcome = _normalize_terminal_outcome(
        state.last_terminal_outcome if state else None
    )
    if terminal_outcome in _NON_ACTIONABLE_TERMINAL_OUTCOMES:
        non_actionable_streak = _effective_non_actionable_streak(state)
        if non_actionable_streak <= 1:
            cooldown_hours = min(base_cooldown, _NON_ACTIONABLE_TERMINAL_COOLDOWN_HOURS)
        elif non_actionable_streak == 2:
            cooldown_hours = min(base_cooldown, _NON_ACTIONABLE_TERMINAL_COOLDOWN_HOURS_SECOND)
        elif non_actionable_streak >= 3:
            cooldown_hours = min(base_cooldown, _NON_ACTIONABLE_TERMINAL_COOLDOWN_HOURS_THIRD)
        else:
            cooldown_hours = base_cooldown
        if terminal_outcome in _HARD_BACKOFF_TERMINAL_OUTCOMES:
            cooldown_hours = min(base_cooldown, cooldown_hours * 2.0)
    else:
        cooldown_hours = base_cooldown
    fill_failure_count = int(getattr(state, "fill_failure_count", 0) or 0) if state else 0
    if fill_failure_count >= _FILL_FAILURE_COOLDOWN_THRESHOLD:
        cooldown_hours *= _FILL_FAILURE_COOLDOWN_MULTIPLIER
    return cooldown_hours


def next_eligible_reanalysis_at(
    market: Market,
    state: MarketState | None,
    *,
    reanalysis_cooldown_hours: int,
    urgent_days_before_close: int,
    urgent_reanalysis_cooldown_hours: int,
    now: datetime | None = None,
) -> datetime | None:
    if not state or not state.last_analysis:
        return None
    last_analysis = _normalize_timestamp(state.last_analysis)
    cooldown_hours = resolve_reanalysis_cooldown_hours(
        market,
        state,
        reanalysis_cooldown_hours=reanalysis_cooldown_hours,
        urgent_days_before_close=urgent_days_before_close,
        urgent_reanalysis_cooldown_hours=urgent_reanalysis_cooldown_hours,
        now=now,
    )
    return last_analysis + timedelta(hours=max(0.0, cooldown_hours))


def remaining_reanalysis_cooldown_seconds(
    market: Market,
    state: MarketState | None,
    *,
    reanalysis_cooldown_hours: int,
    urgent_days_before_close: int,
    urgent_reanalysis_cooldown_hours: int,
    now: datetime | None = None,
) -> float | None:
    now_utc = now or datetime.now(timezone.utc)
    next_eligible_at = next_eligible_reanalysis_at(
        market,
        state,
        reanalysis_cooldown_hours=reanalysis_cooldown_hours,
        urgent_days_before_close=urgent_days_before_close,
        urgent_reanalysis_cooldown_hours=urgent_reanalysis_cooldown_hours,
        now=now_utc,
    )
    if next_eligible_at is None:
        return None
    return max(0.0, (next_eligible_at - now_utc).total_seconds())


class MarketScheduler:
    """Prioritize markets and determine skip conditions based on state."""

    def __init__(
        self,
        reanalysis_cooldown_hours: int = 6,
        urgent_days_before_close: int = 2,
        urgent_reanalysis_cooldown_hours: int = 1,
    ) -> None:
        self.reanalysis_cooldown_hours = reanalysis_cooldown_hours
        self.urgent_days_before_close = urgent_days_before_close
        self.urgent_reanalysis_cooldown_hours = urgent_reanalysis_cooldown_hours

    def prioritize_markets(
        self,
        markets: list[Market],
        state_manager: MarketStateManager,
    ) -> list[Market]:
        """Sort markets by urgency, staleness, opportunity, and liquidity."""
        now = datetime.now(timezone.utc)

        def sort_key(market: Market) -> tuple[float, datetime, str]:
            state = state_manager.get_market_state(market.id)
            last_analysis = state.last_analysis if state else None
            if last_analysis and last_analysis.tzinfo is None:
                last_analysis = last_analysis.replace(tzinfo=timezone.utc)

            urgent = self._is_urgent_close(market, now)
            close_time = _normalize_timestamp(market.close_time)
            staleness_hours = _staleness_hours(last_analysis, now)
            staleness_score = min(
                1.0,
                staleness_hours / max(_MAX_STALENESS_HOURS, 1.0),
            )
            yes_price = _extract_yes_price(market)
            price_opportunity = _price_opportunity_score(yes_price)
            liquidity_score = min(
                1.0,
                (market.liquidity_usdc or 0.0) / _LIQUIDITY_NORMALIZATION_CAP_USDC,
            )
            non_actionable_penalty = (
                _PRIORITY_NON_ACTIONABLE_PENALTY
                if _normalize_terminal_outcome(
                    state.last_terminal_outcome if state else None
                )
                in _NON_ACTIONABLE_TERMINAL_OUTCOMES
                else 0.0
            )
            priority_score = (
                (_PRIORITY_URGENT_BONUS if urgent else 0.0)
                + (_PRIORITY_STALENESS_WEIGHT * staleness_score)
                + (_PRIORITY_PRICE_OPPORTUNITY_WEIGHT * price_opportunity)
                + (_PRIORITY_LIQUIDITY_WEIGHT * liquidity_score)
                - non_actionable_penalty
            )
            return (-priority_score, close_time, market.id)

        try:
            return sorted(markets, key=sort_key)
        except Exception as exc:
            logger.warning(
                "Market prioritization failed: %s",
                exc,
                data={"error": str(exc)},
            )
            return markets

    def should_skip(
        self,
        market: Market,
        state: MarketState | None,
    ) -> tuple[bool, str]:
        """Determine if market should be skipped based on cooldown and close time."""
        now = datetime.now(timezone.utc)
        if market.close_time:
            close_time = _normalize_timestamp(market.close_time)
            if close_time <= now:
                return True, "market closed"

        remaining_seconds = remaining_reanalysis_cooldown_seconds(
            market,
            state,
            reanalysis_cooldown_hours=self.reanalysis_cooldown_hours,
            urgent_days_before_close=self.urgent_days_before_close,
            urgent_reanalysis_cooldown_hours=self.urgent_reanalysis_cooldown_hours,
            now=now,
        )
        if remaining_seconds is not None and remaining_seconds > 0:
            fill_failure_count = int(getattr(state, "fill_failure_count", 0) or 0) if state else 0
            if fill_failure_count >= _FILL_FAILURE_COOLDOWN_THRESHOLD:
                logger.debug(
                    "Applied fill-failure cooldown backoff: market=%s fill_failure_count=%d remaining_seconds=%.1f",
                    market.id,
                    fill_failure_count,
                    remaining_seconds,
                    data={
                        "market_id": market.id,
                        "fill_failure_count": fill_failure_count,
                        "remaining_reanalysis_cooldown_seconds": remaining_seconds,
                        "fill_failure_cooldown_multiplier": _FILL_FAILURE_COOLDOWN_MULTIPLIER,
                    },
                )
            return True, "recently analyzed"

        return False, ""

    def _is_urgent_close(self, market: Market, now: datetime | None = None) -> bool:
        if not market.close_time:
            return False
        now = now or datetime.now(timezone.utc)
        close_time = market.close_time
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=timezone.utc)
        urgent_cutoff = now + timedelta(days=self.urgent_days_before_close)
        return close_time <= urgent_cutoff


def _normalize_timestamp(
    value: datetime | None,
    fallback: datetime | None = None,
) -> datetime:
    if value is None:
        return fallback or datetime.max.replace(tzinfo=timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _effective_non_actionable_streak(state: MarketState | None) -> int:
    if not state:
        return 0
    raw_streak = max(0, int(getattr(state, "non_actionable_streak", 0) or 0))
    terminal_outcome = _normalize_terminal_outcome(state.last_terminal_outcome)
    if raw_streak == 0 and terminal_outcome in _NON_ACTIONABLE_TERMINAL_OUTCOMES:
        return 1
    return raw_streak


def _extract_yes_price(market: Market) -> float | None:
    if market.yes_price is not None:
        return float(market.yes_price)
    for outcome in market.outcomes or []:
        if (outcome.name or "").strip().upper() != "YES":
            continue
        if outcome.price is None:
            continue
        return float(outcome.price)
    return None


def _price_opportunity_score(yes_price: float | None) -> float:
    if yes_price is None:
        return 0.0
    distance_from_coinflip = abs(yes_price - 0.5)
    return max(0.0, 1.0 - min(1.0, distance_from_coinflip / 0.5))


def _staleness_hours(last_analysis: datetime | None, now: datetime) -> float:
    if last_analysis is None:
        return _MAX_STALENESS_HOURS
    return max(0.0, (now - last_analysis).total_seconds() / 3600.0)
