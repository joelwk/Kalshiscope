from __future__ import annotations

from datetime import datetime, timedelta, timezone

from logging_config import get_logger
from market_state import MarketStateManager
from models import Market, MarketState

logger = get_logger(__name__)

_NON_ACTIONABLE_TERMINAL_COOLDOWN_HOURS = 0.25
_NON_ACTIONABLE_TERMINAL_OUTCOMES = {
    "analysis_only_insufficient_balance",
    "bet_amount_zero",
    "confidence_below_min",
    "edge_gate_blocked",
    "kelly_sub_floor_skip",
    "lmsr_gate_blocked",
    "no_trade_recommended",
    "position_adjustment_blocked",
    "score_gate_blocked",
    "uniform_implied_probability",
    "zero_bet_after_sizing",
}


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
        return min(base_cooldown, _NON_ACTIONABLE_TERMINAL_COOLDOWN_HOURS)
    return base_cooldown


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
        """Sort markets by urgency and analysis recency."""
        now = datetime.now(timezone.utc)

        def sort_key(market: Market) -> tuple[int, datetime, datetime]:
            state = state_manager.get_market_state(market.id)
            last_analysis = state.last_analysis if state else None
            if last_analysis and last_analysis.tzinfo is None:
                last_analysis = last_analysis.replace(tzinfo=timezone.utc)

            urgent = self._is_urgent_close(market, now)
            close_time = _normalize_timestamp(market.close_time)

            last_analysis_key = last_analysis or datetime.min.replace(tzinfo=timezone.utc)
            return (
                0 if urgent else 1,
                last_analysis_key,
                close_time,
            )

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
