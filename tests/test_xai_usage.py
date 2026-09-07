from __future__ import annotations

from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest

from config import SearchConfig, Settings
from grok_client import GrokClient
from market_state import MarketStateManager
from models import Market, MarketOutcome
from xai_usage import XAIBudgetExhaustedError, XAIUsageTracker, estimate_xai_cost_usd


def test_cost_includes_cached_input_output_and_server_tools() -> None:
    settings = Settings()
    cost = estimate_xai_cost_usd(
        prompt_tokens=1_000_000,
        cached_tokens=250_000,
        completion_tokens=100_000,
        server_tool_calls=3,
        settings=settings,
    )
    assert cost == pytest.approx(1.2525)


def test_usage_ledger_is_persistent_and_budget_survives_restart(tmp_path) -> None:
    db_path = str(tmp_path / "state.db")
    settings = Settings(
        MAX_XAI_COST_PER_RUN_USD=0.001,
        MAX_XAI_COST_PER_CYCLE_USD=10.0,
    )
    manager = MarketStateManager(db_path)
    tracker = XAIUsageTracker(
        state_manager=manager,
        settings=settings,
        run_id="run-1",
    )
    tracker.begin_cycle("cycle-1", 1)
    tracker.record(
        {
            "market_id": "MKT",
            "phase": "initial",
            "model": "grok-4.3",
            "prompt_tokens": 1_000,
            "cached_tokens": 0,
            "completion_tokens": 100,
            "reasoning_tokens": 50,
            "server_tool_calls": 1,
            "server_side_tool_usage": {"web_search": 1},
        }
    )
    manager.close()

    restarted_manager = MarketStateManager(db_path)
    restarted = XAIUsageTracker(
        state_manager=restarted_manager,
        settings=settings,
        run_id="run-1",
    )
    restarted.begin_cycle("cycle-2", 2)
    totals = restarted.totals()
    assert totals == {
        "calls": 1,
        "prompt_tokens": 1_000,
        "cached_tokens": 0,
        "completion_tokens": 100,
        "reasoning_tokens": 50,
        "server_tool_calls": 1,
        "cost_usd": pytest.approx(0.0065),
    }
    with pytest.raises(XAIBudgetExhaustedError, match="run_cost_cap"):
        restarted.ensure_call_allowed()
    restarted_manager.close()


def test_parallel_usage_events_reconcile_exactly(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    tracker = XAIUsageTracker(
        state_manager=manager,
        settings=Settings(),
        run_id="parallel-run",
    )
    tracker.begin_cycle("cycle-1", 1)

    def record(index: int) -> None:
        tracker.record(
            {
                "market_id": f"MKT-{index}",
                "phase": "initial",
                "model": "grok-4.3",
                "prompt_tokens": 100 + index,
                "cached_tokens": index,
                "completion_tokens": 10,
                "reasoning_tokens": 5,
                "server_tool_calls": 1,
                "server_side_tool_usage": {"web_search": 1},
            }
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(record, range(12)))

    totals = tracker.totals(cycle_id="cycle-1")
    assert totals["calls"] == 12
    assert totals["prompt_tokens"] == sum(100 + index for index in range(12))
    assert totals["cached_tokens"] == sum(range(12))
    assert totals["completion_tokens"] == 120
    assert totals["server_tool_calls"] == 12
    manager.close()


def test_parallel_budget_admission_uses_thread_safe_ledger_reads(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    tracker = XAIUsageTracker(
        state_manager=manager,
        settings=Settings(),
        run_id="parallel-admission-run",
    )
    tracker.begin_cycle("cycle-1", 1)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: tracker.budget_exhausted(), range(8)))

    assert results == [(False, None)] * 8
    manager.close()


class _UsageChat:
    def append(self, message) -> None:
        return None

    def stream(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=120,
                completion_tokens=30,
                prompt_tokens_details=SimpleNamespace(cached_tokens=20),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=10),
            ),
            server_side_tool_usage={"web_search": 2, "x_search": 1},
        )
        yield response, SimpleNamespace(content="not-json")


class _UsageClient:
    chat = SimpleNamespace(create=lambda **kwargs: _UsageChat())


def test_completed_provider_call_is_recorded_before_parse_failure() -> None:
    events: list[dict] = []
    client = GrokClient(
        api_key="x",
        search_config=SearchConfig(profile_name="generic"),
        usage_recorder=events.append,
    )
    client.client = _UsageClient()
    market = Market(
        id="MKT",
        question="Will it happen?",
        outcomes=[MarketOutcome(name="YES", price=0.5), MarketOutcome(name="NO", price=0.5)],
    )

    with pytest.raises(ValueError):
        client.analyze_market(market, usage_phase="repair")

    assert len(events) == 1
    assert events[0]["phase"] == "repair"
    assert events[0]["prompt_tokens"] == 120
    assert events[0]["cached_tokens"] == 20
    assert events[0]["server_tool_calls"] == 3


def test_guaranteed_plan_round_trip_preserves_accepted_slot_and_spend() -> None:
    import main

    market = Market(
        id="MKT",
        question="Will it happen?",
        outcomes=[MarketOutcome(name="YES", price=0.5), MarketOutcome(name="NO", price=0.5)],
    )
    plan = main.GuaranteedOrderPlan(
        target=1,
        run_id="resume-run",
        slots=[
            main.GuaranteedOrderSlot(
                slot_number=1,
                market_id=market.id,
                market=market,
                locked_cycle=1,
                client_order_id="client-1",
                completed=True,
                order_id="order-1",
            )
        ],
        research_cost_usd=1.25,
    )

    restored = main.GuaranteedOrderPlan.from_json(plan.to_json())
    assert restored.run_id == "resume-run"
    assert restored.completed_count == 1
    assert restored.slots[0].client_order_id == "client-1"
    assert restored.slots[0].order_id == "order-1"
    assert restored.research_cost_usd == 1.25
