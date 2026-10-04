from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from config import Settings
from market_state import MarketStateManager


class XAIBudgetExhaustedError(RuntimeError):
    pass


def estimate_xai_cost_usd(
    *,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    server_tool_calls: int,
    settings: Settings,
) -> float:
    cached = min(max(0, int(cached_tokens)), max(0, int(prompt_tokens)))
    uncached = max(0, int(prompt_tokens)) - cached
    long_context = (
        settings.API_COST_LONG_CONTEXT_THRESHOLD_TOKENS > 0
        and max(0, int(prompt_tokens))
        >= settings.API_COST_LONG_CONTEXT_THRESHOLD_TOKENS
    )
    input_rate = (
        settings.API_COST_LONG_CONTEXT_INPUT_PER_1K_TOKENS_USD
        if long_context
        else settings.API_COST_INPUT_PER_1K_TOKENS_USD
    )
    cached_input_rate = (
        settings.API_COST_LONG_CONTEXT_CACHED_INPUT_PER_1K_TOKENS_USD
        if long_context
        else settings.API_COST_CACHED_INPUT_PER_1K_TOKENS_USD
    )
    output_rate = (
        settings.API_COST_LONG_CONTEXT_OUTPUT_PER_1K_TOKENS_USD
        if long_context
        else settings.API_COST_OUTPUT_PER_1K_TOKENS_USD
    )
    return (
        (uncached / 1000.0) * max(0.0, input_rate)
        + (cached / 1000.0) * max(0.0, cached_input_rate)
        + (max(0, int(completion_tokens)) / 1000.0)
        * max(0.0, output_rate)
        + max(0, int(server_tool_calls))
        * max(0.0, settings.API_COST_SERVER_TOOL_PER_CALL_USD)
    )


class XAIUsageTracker:
    """Thread-safe, persistent run/cycle accounting and admission control."""

    def __init__(
        self,
        *,
        state_manager: MarketStateManager,
        settings: Settings,
        run_id: str,
    ) -> None:
        self.state_manager = state_manager
        self.settings = settings
        self.run_id = run_id
        self._lock = threading.Lock()
        self._cycle_id = "startup"
        self._cycle_number = 0
        self._reservations: dict[str, tuple[str, int, float]] = {}

    def begin_cycle(self, cycle_id: str, cycle_number: int) -> None:
        with self._lock:
            self._cycle_id = str(cycle_id)
            self._cycle_number = int(cycle_number)

    def record(self, event: dict[str, Any]) -> None:
        with self._lock:
            payload = dict(event)
            reservation_id = str(payload.pop("_reservation_id", "") or "")
            reservation = self._reservations.get(reservation_id)
            if reservation is None:
                cycle_id = self._cycle_id
                cycle_number = self._cycle_number
            else:
                cycle_id, cycle_number, _reserved_cost = reservation
            prompt_tokens = int(payload.get("prompt_tokens") or 0)
            cached_tokens = int(payload.get("cached_tokens") or 0)
            completion_tokens = int(payload.get("completion_tokens") or 0)
            server_tool_calls = int(payload.get("server_tool_calls") or 0)
            payload.update(
                {
                    "event_id": uuid.uuid4().hex,
                    "run_id": self.run_id,
                    "cycle_id": cycle_id,
                    "cycle_number": cycle_number,
                    "cost_usd": estimate_xai_cost_usd(
                        prompt_tokens=prompt_tokens,
                        cached_tokens=cached_tokens,
                        completion_tokens=completion_tokens,
                        server_tool_calls=server_tool_calls,
                        settings=self.settings,
                    ),
                    "pricing_version": self.settings.API_COST_PRICING_VERSION,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            self.state_manager.record_xai_usage(payload)
            if reservation_id:
                self._reservations.pop(reservation_id, None)

    def totals(self, *, cycle_id: str | None = None) -> dict[str, int | float]:
        with self._lock:
            return self.state_manager.get_xai_usage_totals(
                run_id=self.run_id,
                cycle_id=cycle_id,
            )

    def budget_exhausted(self) -> tuple[bool, str | None]:
        with self._lock:
            return self._budget_exhausted_locked()

    def _budget_exhausted_locked(
        self,
        *,
        additional_reserved_cost: float = 0.0,
    ) -> tuple[bool, str | None]:
        cycle_id = self._cycle_id
        run_total = self.state_manager.get_xai_usage_totals(run_id=self.run_id)
        cycle_total = self.state_manager.get_xai_usage_totals(
            run_id=self.run_id,
            cycle_id=cycle_id,
        )
        run_reserved = sum(reservation[2] for reservation in self._reservations.values())
        cycle_reserved = sum(
            reservation[2]
            for reservation in self._reservations.values()
            if reservation[0] == cycle_id
        )
        projected_run_cost = (
            float(run_total["cost_usd"])
            + run_reserved
            + max(0.0, float(additional_reserved_cost))
        )
        projected_cycle_cost = (
            float(cycle_total["cost_usd"])
            + cycle_reserved
            + max(0.0, float(additional_reserved_cost))
        )
        reject_at_cap = additional_reserved_cost <= 0
        run_cap_reached = (
            projected_run_cost >= self.settings.MAX_XAI_COST_PER_RUN_USD
            if reject_at_cap
            else projected_run_cost > self.settings.MAX_XAI_COST_PER_RUN_USD
        )
        cycle_cap_reached = (
            projected_cycle_cost >= self.settings.MAX_XAI_COST_PER_CYCLE_USD
            if reject_at_cap
            else projected_cycle_cost > self.settings.MAX_XAI_COST_PER_CYCLE_USD
        )
        if (
            self.settings.MAX_XAI_COST_PER_RUN_USD > 0
            and run_cap_reached
        ):
            return True, "run_cost_cap"
        if (
            self.settings.MAX_XAI_COST_PER_CYCLE_USD > 0
            and cycle_cap_reached
        ):
            return True, "cycle_cost_cap"
        return False, None

    def ensure_call_allowed(self) -> None:
        exhausted, reason = self.budget_exhausted()
        if exhausted:
            raise XAIBudgetExhaustedError(f"api_budget_exhausted:{reason}")

    def reserve_call(self) -> str:
        """Atomically reserve projected spend before one provider call."""
        with self._lock:
            reservation_cost = max(
                0.0,
                float(self.settings.API_COST_RESERVATION_PER_CALL_USD),
            )
            positive_caps = [
                float(cap)
                for cap in (
                    self.settings.MAX_XAI_COST_PER_RUN_USD,
                    self.settings.MAX_XAI_COST_PER_CYCLE_USD,
                )
                if float(cap) > 0
            ]
            if positive_caps:
                reservation_cost = min(reservation_cost, min(positive_caps))
            exhausted, reason = self._budget_exhausted_locked(
                additional_reserved_cost=reservation_cost,
            )
            if exhausted:
                raise XAIBudgetExhaustedError(f"api_budget_exhausted:{reason}")
            reservation_id = uuid.uuid4().hex
            self._reservations[reservation_id] = (
                self._cycle_id,
                self._cycle_number,
                reservation_cost,
            )
            return reservation_id

    def release_reservation(self, reservation_id: str) -> None:
        """Release an unfinished or already-reconciled provider-call reservation."""
        normalized_id = str(reservation_id or "").strip()
        if not normalized_id:
            return
        with self._lock:
            self._reservations.pop(normalized_id, None)
