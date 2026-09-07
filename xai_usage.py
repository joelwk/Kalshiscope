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
    return (
        (uncached / 1000.0) * max(0.0, settings.API_COST_INPUT_PER_1K_TOKENS_USD)
        + (cached / 1000.0)
        * max(0.0, settings.API_COST_CACHED_INPUT_PER_1K_TOKENS_USD)
        + (max(0, int(completion_tokens)) / 1000.0)
        * max(0.0, settings.API_COST_OUTPUT_PER_1K_TOKENS_USD)
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

    def begin_cycle(self, cycle_id: str, cycle_number: int) -> None:
        with self._lock:
            self._cycle_id = str(cycle_id)
            self._cycle_number = int(cycle_number)

    def record(self, event: dict[str, Any]) -> None:
        with self._lock:
            cycle_id = self._cycle_id
            cycle_number = self._cycle_number
            payload = dict(event)
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

    def totals(self, *, cycle_id: str | None = None) -> dict[str, int | float]:
        with self._lock:
            return self.state_manager.get_xai_usage_totals(
                run_id=self.run_id,
                cycle_id=cycle_id,
            )

    def budget_exhausted(self) -> tuple[bool, str | None]:
        with self._lock:
            cycle_id = self._cycle_id
            run_total = self.state_manager.get_xai_usage_totals(run_id=self.run_id)
            cycle_total = self.state_manager.get_xai_usage_totals(
                run_id=self.run_id,
                cycle_id=cycle_id,
            )
        if (
            self.settings.MAX_XAI_COST_PER_RUN_USD > 0
            and float(run_total["cost_usd"]) >= self.settings.MAX_XAI_COST_PER_RUN_USD
        ):
            return True, "run_cost_cap"
        if (
            self.settings.MAX_XAI_COST_PER_CYCLE_USD > 0
            and float(cycle_total["cost_usd"])
            >= self.settings.MAX_XAI_COST_PER_CYCLE_USD
        ):
            return True, "cycle_cost_cap"
        return False, None

    def ensure_call_allowed(self) -> None:
        exhausted, reason = self.budget_exhausted()
        if exhausted:
            raise XAIBudgetExhaustedError(f"api_budget_exhausted:{reason}")
