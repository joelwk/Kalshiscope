from __future__ import annotations

from datetime import datetime, timedelta, timezone

from logging_config import get_logger
from market_state import MarketStateManager
from models import Market, MarketState

logger = get_logger(__name__)


class MarketScheduler:
    """Prioritize markets and determine skip conditions based on state."""

    def __init__(
        self,
        reanalysis_cooldown_hours: int = 6,
        urgent_days_before_close: int = 2,
    ) -> None:
        self.reanalysis_cooldown_hours = reanalysis_cooldown_hours
        self.urgent_days_before_close = urgent_days_before_close

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

        urgent = self._is_urgent_close(market, now)
        if state and state.last_analysis:
            last_analysis = state.last_analysis
            if last_analysis.tzinfo is None:
                last_analysis = last_analysis.replace(tzinfo=timezone.utc)
            cooldown = timedelta(hours=self.reanalysis_cooldown_hours)
            if (now - last_analysis) < cooldown and not urgent:
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
