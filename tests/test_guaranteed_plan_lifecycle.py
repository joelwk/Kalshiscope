from __future__ import annotations

import json
import sys

import pytest

import main
from config import Settings
from kalshi import cli
from market_state import MarketStateManager
from xai_usage import XAIUsageTracker


def _seed_plan(state: MarketStateManager, *, run_id: str = "old-run") -> main.GuaranteedOrderPlan:
    plan = main.GuaranteedOrderPlan(target=2, run_id=run_id)
    main._persist_guaranteed_order_plan(state, plan)
    return plan


def test_plan_lifecycle_requires_an_active_plan(tmp_path) -> None:
    state = MarketStateManager(str(tmp_path / "state.db"))
    try:
        with pytest.raises(main.GuaranteedPlanLifecycleError, match="No active"):
            main._apply_guaranteed_plan_lifecycle_action(
                state_manager=state,
                resumed_plan=None,
                configured_target=2,
                abandon_requested=False,
                new_requested=True,
            )
        receipt_count = state._conn.execute(
            "SELECT COUNT(*) AS count FROM cycle_receipts"
        ).fetchone()["count"]
    finally:
        state.close()

    assert receipt_count == 0


@pytest.mark.parametrize(
    ("action", "expected_outcome", "should_exit"),
    [
        ("abandon", "abandoned_by_operator", True),
        ("new", "replaced_by_operator", False),
    ],
)
def test_plan_lifecycle_clears_active_pointer_and_records_audit(
    tmp_path,
    action: str,
    expected_outcome: str,
    should_exit: bool,
) -> None:
    state = MarketStateManager(str(tmp_path / f"{action}.db"))
    plan = _seed_plan(state)
    try:
        resumed, exit_requested = main._apply_guaranteed_plan_lifecycle_action(
            state_manager=state,
            resumed_plan=plan,
            configured_target=2,
            abandon_requested=action == "abandon",
            new_requested=action == "new",
        )
        active = state.get_runtime_flag(main._GUARANTEED_PLAN_RUNTIME_FLAG)
        row = state._conn.execute(
            "SELECT payload_json FROM cycle_receipts ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        state.close()

    assert resumed is None
    assert exit_requested is should_exit
    assert active is None
    assert row is not None
    assert json.loads(row["payload_json"])["guaranteed_run_outcome"] == expected_outcome


def test_new_plan_requires_positive_configured_target(tmp_path) -> None:
    state = MarketStateManager(str(tmp_path / "state.db"))
    plan = _seed_plan(state)
    try:
        with pytest.raises(
            main.GuaranteedPlanLifecycleError,
            match="GUARANTEED_ORDERS_N",
        ):
            main._apply_guaranteed_plan_lifecycle_action(
                state_manager=state,
                resumed_plan=plan,
                configured_target=0,
                abandon_requested=False,
                new_requested=True,
            )
        assert state.get_runtime_flag(main._GUARANTEED_PLAN_RUNTIME_FLAG) is not None
    finally:
        state.close()


def test_main_abandon_exits_before_api_client_initialization(
    monkeypatch,
    tmp_path,
) -> None:
    db_path = str(tmp_path / "state.db")
    settings = Settings(
        GUARANTEED_ORDERS_N=2,
        STATE_DB_PATH=db_path,
        STATE_JSON_EXPORT_PATH=str(tmp_path / "state.json"),
        EXPORT_STATE_JSON=False,
        LOG_DIR=str(tmp_path / "logs"),
        ENABLE_FILE_LOGGING=False,
    )
    seed = MarketStateManager(db_path)
    _seed_plan(seed)
    seed.close()

    monkeypatch.setattr(main, "load_settings", lambda: settings)
    monkeypatch.setattr(
        main,
        "XAIProvider",
        lambda *args, **kwargs: pytest.fail("xAI provider initialized"),
    )
    monkeypatch.setattr(
        main,
        "KalshiClient",
        lambda *args, **kwargs: pytest.fail("Kalshi client initialized"),
    )

    main.main(abandon_guaranteed_plan=True)

    verifier = MarketStateManager(db_path)
    try:
        assert verifier.get_runtime_flag(main._GUARANTEED_PLAN_RUNTIME_FLAG) is None
    finally:
        verifier.close()


def test_live_exhausted_resumed_plan_fails_before_first_cycle_and_records_once(
    monkeypatch,
    tmp_path,
) -> None:
    db_path = str(tmp_path / "state.db")
    settings = Settings(
        XAI_API_KEY="xai-key",
        KALSHI_API_BASE_URL="https://api.example/trade-api/v2",
        KALSHI_API_KEY_ID="kalshi-key-id",
        KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        DRY_RUN=False,
        GUARANTEED_ORDERS_N=1,
        MAX_XAI_COST_PER_RUN_USD=0.001,
        MAX_XAI_COST_PER_CYCLE_USD=3.0,
        STATE_DB_PATH=db_path,
        STATE_JSON_EXPORT_PATH=str(tmp_path / "state.json"),
        EXPORT_STATE_JSON=False,
        LOG_DIR=str(tmp_path / "logs"),
        ENABLE_FILE_LOGGING=False,
    )
    seed = MarketStateManager(db_path)
    plan = main.GuaranteedOrderPlan(target=1, run_id="spent-run")
    main._persist_guaranteed_order_plan(seed, plan)
    tracker = XAIUsageTracker(state_manager=seed, settings=settings, run_id=plan.run_id)
    tracker.begin_cycle("old-cycle", 1)
    tracker.record(
        {
            "market_id": "MKT",
            "phase": "initial",
            "model": "grok-4.3",
            "prompt_tokens": 1_000,
            "cached_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "server_tool_calls": 0,
            "server_side_tool_usage": {},
        }
    )
    seed.close()

    client_initializations = {"grok": 0, "kalshi": 0}

    def _grok(*args, **kwargs):
        client_initializations["grok"] += 1
        return object()

    def _kalshi(*args, **kwargs):
        client_initializations["kalshi"] += 1
        return object()

    monkeypatch.setattr(main, "load_settings", lambda: settings)
    monkeypatch.setattr(main, "XAIProvider", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "GrokClient", _grok)
    monkeypatch.setattr(main, "KalshiClient", _kalshi)
    monkeypatch.setattr(main, "run_bootstrap_checks", lambda **kwargs: None)
    sync_calls: list[object] = []

    def _sync(**kwargs):
        sync_calls.append(kwargs["kalshi_client"])
        return main.OrderSyncMetrics(complete=True)

    monkeypatch.setattr(main, "_sync_orders_from_exchange", _sync)

    with pytest.raises(
        main.GuaranteedOrdersIncompleteError,
        match="cost cap exhausted before cycle execution",
    ):
        main.main(max_cycles=5)

    verifier = MarketStateManager(db_path)
    try:
        receipts = verifier._conn.execute(
            "SELECT cycle_number, payload_json FROM cycle_receipts ORDER BY id"
        ).fetchall()
        usage = verifier.get_xai_usage_totals(run_id=plan.run_id)
    finally:
        verifier.close()

    assert client_initializations == {"grok": 1, "kalshi": 1}
    assert len(receipts) == 1
    assert receipts[0]["cycle_number"] == 0
    receipt = json.loads(receipts[0]["payload_json"])
    assert receipt["startup_fail_fast"] is True
    assert receipt["api_budget_exhausted_reason"] == "run_cost_cap"
    assert receipt["fetched_markets"] == 0
    assert receipt["analyzed_markets"] == 0
    assert usage["calls"] == 1
    assert len(sync_calls) == 1


def test_normal_dry_run_command_replaces_exhausted_plan_and_starts_cycle(
    monkeypatch,
    tmp_path,
) -> None:
    db_path = str(tmp_path / "state.db")
    settings = Settings(
        XAI_API_KEY="xai-key",
        KALSHI_API_BASE_URL="https://api.example/trade-api/v2",
        KALSHI_API_KEY_ID="kalshi-key-id",
        KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        DRY_RUN=True,
        GUARANTEED_ORDERS_N=1,
        MAX_XAI_COST_PER_RUN_USD=0.001,
        STATE_DB_PATH=db_path,
        STATE_JSON_EXPORT_PATH=str(tmp_path / "state.json"),
        EXPORT_STATE_JSON=False,
        LOG_DIR=str(tmp_path / "logs"),
        ENABLE_FILE_LOGGING=False,
        POLL_INTERVAL_SEC=0,
    )
    seed = MarketStateManager(db_path)
    old_plan = main.GuaranteedOrderPlan(target=1, run_id="spent-dry-run")
    main._persist_guaranteed_order_plan(seed, old_plan)
    tracker = XAIUsageTracker(
        state_manager=seed,
        settings=settings,
        run_id=old_plan.run_id,
    )
    tracker.begin_cycle("old-cycle", 1)
    tracker.record(
        {
            "market_id": "MKT",
            "phase": "initial",
            "model": "grok-4.3",
            "prompt_tokens": 1_000,
            "cached_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "server_tool_calls": 0,
            "server_side_tool_usage": {},
        }
    )
    seed.close()

    class _Kalshi:
        last_fetch_pages = 1
        last_fetch_cap_hit = False
        last_fetch_mve_filter = None
        min_bet_usdc = 0.0
        max_bet_usdc = 0.0

        def __init__(self) -> None:
            self.fetches = 0

        def get_markets(self, **kwargs):
            self.fetches += 1
            return []

        def reset_session(self) -> None:
            pass

    class _Grok:
        min_bet_usdc = 0.0
        max_bet_usdc = 0.0

    kalshi = _Kalshi()
    monkeypatch.setattr(main, "load_settings", lambda: settings)
    monkeypatch.setattr(main, "XAIProvider", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "GrokClient", lambda *args, **kwargs: _Grok())
    monkeypatch.setattr(main, "KalshiClient", lambda *args, **kwargs: kalshi)
    monkeypatch.setattr(main, "run_bootstrap_checks", lambda **kwargs: None)

    with pytest.raises(main.GuaranteedOrdersIncompleteError):
        main.main(max_cycles=1)

    verifier = MarketStateManager(db_path)
    try:
        active_raw = verifier.get_runtime_flag(main._GUARANTEED_PLAN_RUNTIME_FLAG)
        receipt_rows = verifier._conn.execute(
            "SELECT payload_json FROM cycle_receipts ORDER BY id"
        ).fetchall()
    finally:
        verifier.close()

    assert kalshi.fetches == 1
    assert active_raw is not None
    assert main.GuaranteedOrderPlan.from_json(active_raw).run_id != old_plan.run_id
    receipts = [json.loads(row["payload_json"]) for row in receipt_rows]
    assert receipts[0]["guaranteed_run_outcome"] == "replaced_after_dry_run_cost_cap"
    assert receipts[1]["cycle"] == 1


def test_run_cost_cap_reached_in_cycle_stops_before_next_catalog_fetch(
    monkeypatch,
    tmp_path,
) -> None:
    settings = Settings(
        XAI_API_KEY="xai-key",
        KALSHI_API_BASE_URL="https://api.example/trade-api/v2",
        KALSHI_API_KEY_ID="kalshi-key-id",
        KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        DRY_RUN=True,
        GUARANTEED_ORDERS_N=1,
        STATE_DB_PATH=str(tmp_path / "state.db"),
        STATE_JSON_EXPORT_PATH=str(tmp_path / "state.json"),
        EXPORT_STATE_JSON=False,
        LOG_DIR=str(tmp_path / "logs"),
        ENABLE_FILE_LOGGING=False,
        POLL_INTERVAL_SEC=0,
    )

    class _Tracker:
        budget_checks = 0

        def __init__(self, **kwargs) -> None:
            pass

        def begin_cycle(self, cycle_id: str, cycle_number: int) -> None:
            pass

        def record(self, event: dict) -> None:
            pass

        def totals(self, *, cycle_id: str | None = None) -> dict[str, int | float]:
            return {
                "calls": 1,
                "prompt_tokens": 1_000,
                "cached_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "server_tool_calls": 0,
                "cost_usd": 10.0,
            }

        def budget_exhausted(self) -> tuple[bool, str | None]:
            self.budget_checks += 1
            if self.budget_checks <= 2:
                return False, None
            return True, "run_cost_cap"

        def ensure_call_allowed(self) -> None:
            pass

    class _Kalshi:
        last_fetch_pages = 1
        last_fetch_cap_hit = False
        last_fetch_mve_filter = None

        def __init__(self) -> None:
            self.fetches = 0

        def get_markets(self, **kwargs):
            self.fetches += 1
            return []

        def reset_session(self) -> None:
            pass

    class _Grok:
        min_bet_usdc = 0.0
        max_bet_usdc = 0.0

    kalshi = _Kalshi()
    monkeypatch.setattr(main, "load_settings", lambda: settings)
    monkeypatch.setattr(main, "XAIUsageTracker", _Tracker)
    monkeypatch.setattr(main, "XAIProvider", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "GrokClient", lambda *args, **kwargs: _Grok())
    monkeypatch.setattr(main, "KalshiClient", lambda *args, **kwargs: kalshi)
    monkeypatch.setattr(main, "run_bootstrap_checks", lambda **kwargs: None)

    with pytest.raises(main.GuaranteedOrdersIncompleteError):
        main.main(max_cycles=5)

    assert kalshi.fetches == 1


@pytest.mark.parametrize(
    ("flag", "argument_name"),
    [
        ("--abandon-guaranteed-plan", "abandon_guaranteed_plan"),
        ("--new-guaranteed-run", "new_guaranteed_run"),
    ],
)
def test_cli_forwards_plan_action(
    monkeypatch,
    flag: str,
    argument_name: str,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(cli, "run_main", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(sys, "argv", ["kalshi", "--cycles", "2", flag])

    cli.main()

    assert captured["max_cycles"] == 2
    assert captured[argument_name] is True
    other = (
        "new_guaranteed_run"
        if argument_name == "abandon_guaranteed_plan"
        else "abandon_guaranteed_plan"
    )
    assert captured[other] is False


def test_cli_prints_expected_plan_stop_without_traceback(monkeypatch, capsys) -> None:
    def _stop(**kwargs) -> None:
        raise main.GuaranteedOrdersIncompleteError("run cost cap exhausted")

    monkeypatch.setattr(cli, "run_main", _stop)
    monkeypatch.setattr(sys, "argv", ["kalshi", "--cycles", "5"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert error == "PredictBot stopped: run cost cap exhausted\n"
    assert "Traceback" not in error
