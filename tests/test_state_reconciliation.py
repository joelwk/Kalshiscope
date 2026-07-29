from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

import market_state
from main import (
    _reconciliation_execution_gate,
    _sync_exchange_fills,
    _sync_orders_from_exchange,
    _sync_positions_from_exchange,
    _sync_settlements_from_exchange,
)
from market_state import MarketStateManager
from models import Market, OrderResponse


def _pending(
    manager: MarketStateManager,
    order_id: str,
    *,
    market_id: str | None = None,
    status: str = "resting",
) -> None:
    manager.record_pending_order(
        order_id=order_id,
        market_id=market_id or f"MKT-{order_id}",
        outcome="YES",
        submitted_amount_usdc=2.0,
        requested_shares=5.0,
        limit_price=0.40,
        confidence=0.75,
        implied_prob=0.40,
        status=status,
        raw={"status": status},
    )


def test_reconciliation_gate_blocks_live_but_not_dry_run_analysis() -> None:
    live_blocked, reasons = _reconciliation_execution_gate(
        dry_run=False,
        orders_ready=False,
        positions_ready=True,
        unknown_exchange_orders=("external",),
    )
    dry_run_blocked, dry_run_reasons = _reconciliation_execution_gate(
        dry_run=True,
        orders_ready=False,
        positions_ready=True,
        unknown_exchange_orders=("external",),
    )
    assert live_blocked is True
    assert reasons == ["orders_incomplete", "untracked_resting_orders"]
    assert dry_run_blocked is False
    assert dry_run_reasons == reasons


def test_complete_position_snapshot_closes_missing_and_preserves_history(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    manager.upsert_position_snapshot(
        market_id="MKT-SEEN",
        outcome="YES",
        total_amount_usdc=4.0,
        contracts=8.0,
    )
    manager.upsert_position_snapshot(
        market_id="MKT-MISSING",
        outcome="NO",
        total_amount_usdc=3.0,
        contracts=6.0,
    )

    class IncompleteClient:
        @staticmethod
        def get_positions(**kwargs) -> dict:
            return {
                "market_positions": [
                    {
                        "ticker": "MKT-SEEN",
                        "position_fp": "7.50",
                        "market_exposure_dollars": "3.25",
                    }
                ],
                "cursor": "more",
            }

    class CompleteClient:
        @staticmethod
        def get_positions(**kwargs) -> dict:
            assert kwargs["count_filter"] == "position"
            assert kwargs["subaccount"] == 0
            return {
                "market_positions": [
                    {
                        "ticker": "MKT-SEEN",
                        "position_fp": "7.50",
                        "market_exposure_dollars": "3.25",
                    }
                ],
                "cursor": "",
            }

    try:
        incomplete = _sync_positions_from_exchange(
            state_manager=manager,
            kalshi_client=IncompleteClient(),
            max_pages=1,
        )
        assert incomplete.complete is False
        assert manager.get_position("MKT-MISSING") is not None

        complete = _sync_positions_from_exchange(
            state_manager=manager,
            kalshi_client=CompleteClient(),
            max_pages=2,
        )
        assert complete.complete is True
        assert complete.closed_positions == 1
        seen = manager.get_position("MKT-SEEN")
        assert seen is not None
        assert seen.contracts == 7.5
        assert manager.get_position("MKT-MISSING") is None
        closed = manager.get_position("MKT-MISSING", include_closed=True)
        assert closed is not None
        assert closed.status == "closed"
        assert closed.total_amount_usdc == 3.0
        assert manager.get_open_position_market_ids_for_event("MKT-") == ["MKT-SEEN"]

        manager.record_trade(
            "MKT-MISSING",
            OrderResponse(id="reopen-order", raw={"outcome": "NO"}),
            0.5,
            outcome="NO",
            entry_price=0.50,
            shares=1.0,
        )
        reopened = manager.get_position("MKT-MISSING")
        assert reopened is not None
        assert reopened.status == "open"
        assert reopened.closed_at is None
        assert reopened.total_amount_usdc == 0.5
    finally:
        manager.close()


def test_order_reconciliation_maps_lifecycle_and_blocks_unknown_resting(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    for order_id in ("resting", "filled", "partial-cancel", "cancel", "reject"):
        _pending(manager, order_id)

    lifecycle = {
        "filled": {
            "order_id": "filled",
            "status": "executed",
            "initial_count_fp": "5.00",
            "fill_count_fp": "5.00",
            "maker_fill_cost_dollars": "2.00",
            "yes_price_dollars": "0.40",
        },
        "partial-cancel": {
            "order_id": "partial-cancel",
            "status": "canceled",
            "initial_count_fp": "5.00",
            "fill_count_fp": "2.00",
            "taker_fill_cost_dollars": "0.80",
            "yes_price_dollars": "0.40",
        },
        "cancel": {
            "order_id": "cancel",
            "status": "canceled",
            "initial_count_fp": "5.00",
            "fill_count_fp": "0.00",
        },
        "reject": {
            "order_id": "reject",
            "status": "rejected",
            "initial_count_fp": "5.00",
            "fill_count_fp": "0.00",
        },
    }

    class Client:
        @staticmethod
        def get_orders(**kwargs) -> dict:
            assert kwargs["status"] == "resting"
            if kwargs["cursor"] is None:
                return {
                    "orders": [
                        {
                            "order_id": "resting",
                            "status": "resting",
                            "initial_count_fp": "5.00",
                            "fill_count_fp": "0.00",
                            "remaining_count_fp": "5.00",
                        }
                    ],
                    "cursor": "next",
                }
            return {
                "orders": [
                    {
                        "order_id": "external-order",
                        "status": "resting",
                        "initial_count_fp": "1.00",
                        "fill_count_fp": "0.00",
                    },
                ],
                "cursor": "",
            }

        @staticmethod
        def get_order(order_id: str, **kwargs) -> dict:
            return {"order": lifecycle[order_id]}

    try:
        metrics = _sync_orders_from_exchange(
            state_manager=manager,
            kalshi_client=Client(),
            max_pages=2,
        )
        assert metrics.complete is True
        assert metrics.pages_fetched == 2
        assert metrics.live_submission_blocked is True
        assert metrics.unknown_exchange_orders == ("external-order",)
        assert manager.get_pending_order("resting")["status"] == "resting"
        assert manager.get_pending_order("resting")["remaining_shares"] == 5.0
        assert manager.get_pending_order("filled")["status"] == "filled"
        assert (
            manager.get_pending_order("partial-cancel")["status"]
            == "canceled_partially_filled"
        )
        assert manager.get_pending_order("cancel")["status"] == "canceled"
        assert manager.get_pending_order("reject")["status"] == "rejected"
        assert manager.get_position("MKT-filled").total_amount_usdc == 2.0
        assert manager.get_position("MKT-partial-cancel").total_amount_usdc == 0.8
        assert len(manager.get_pending_orders(active_only=True)) == 1
        assert len(manager.get_order_history()) == 5
    finally:
        manager.close()


def test_order_reconciliation_uses_historical_fallback(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    _pending(manager, "old-order", market_id="MKT-OLD")

    class Client:
        @staticmethod
        def get_orders(**kwargs) -> dict:
            return {"orders": [], "cursor": ""}

        @staticmethod
        def get_order(order_id: str, **kwargs) -> dict:
            raise LookupError("past live cutoff")

        @staticmethod
        def get_historical_orders(**kwargs) -> dict:
            assert kwargs["ticker"] == "MKT-OLD"
            return {
                "orders": [
                    {
                        "order_id": "old-order",
                        "status": "executed",
                        "initial_count_fp": "5.00",
                        "fill_count_fp": "5.00",
                        "maker_fill_cost_dollars": "2.00",
                    }
                ],
                "cursor": "",
            }

    try:
        metrics = _sync_orders_from_exchange(
            state_manager=manager,
            kalshi_client=Client(),
            max_pages=2,
        )
        assert metrics.complete is True
        assert manager.get_pending_order("old-order")["status"] == "filled"
    finally:
        manager.close()


def test_terminal_order_status_never_regresses_to_active(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    _pending(manager, "terminal", status="filled")

    class Client:
        @staticmethod
        def get_orders(**kwargs) -> dict:
            return {
                "orders": [
                    {
                        "order_id": "terminal",
                        "status": "resting",
                        "initial_count_fp": "5.00",
                        "fill_count_fp": "0.00",
                    }
                ],
                "cursor": "",
            }

    try:
        before = manager.get_pending_order("terminal")
        _sync_orders_from_exchange(
            state_manager=manager,
            kalshi_client=Client(),
        )
        after = manager.get_pending_order("terminal")
        assert after["status"] == "filled"
        assert after["updated_at"] == before["updated_at"]
        assert after["last_synced_at"] == before["last_synced_at"]
    finally:
        manager.close()


def test_fill_pagination_and_restart_replay_are_idempotent(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    _pending(manager, "fill-order", market_id="MKT-FILL")

    class Client:
        @staticmethod
        def get_fills(**kwargs) -> dict:
            if kwargs["cursor"] is None:
                return {
                    "fills": [
                        {
                            "fill_id": "fill-1",
                            "order_id": "fill-order",
                            "ticker": "MKT-FILL",
                            "count_fp": "2.00",
                            "yes_price_dollars": "0.40",
                            "created_time": "2026-07-29T10:00:00Z",
                        }
                    ],
                    "cursor": "next",
                }
            return {
                "fills": [
                    {
                        "fill_id": "fill-2",
                        "order_id": "fill-order",
                        "ticker": "MKT-FILL",
                        "count_fp": "3.00",
                        "yes_price_dollars": "0.40",
                        "created_time": "2026-07-29T10:01:00Z",
                    }
                ],
                "cursor": "",
            }

    try:
        first = _sync_exchange_fills(
            state_manager=manager,
            kalshi_client=Client(),
            max_pages=3,
        )
        replay = _sync_exchange_fills(
            state_manager=manager,
            kalshi_client=Client(),
            max_pages=3,
        )
        assert first.complete is True
        assert first.pages_fetched == 2
        assert first.new_fill_events == 2
        assert replay.new_fill_events == 0
        assert manager.get_pending_order("fill-order")["status"] == "filled"
        count = manager._conn.execute(
            "SELECT COUNT(*) FROM order_fill_events"
        ).fetchone()[0]
        assert count == 2
    finally:
        manager.close()


def test_settlement_sync_resolves_market_outcome_and_position_atomically(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    manager.record_trade(
        "MKT-SETTLE",
        OrderResponse(id="order-settle", raw={"outcome": "YES"}),
        2.0,
        outcome="YES",
        entry_price=0.40,
        shares=5.0,
    )

    class Client:
        @staticmethod
        def get_settlements(**kwargs) -> dict:
            return {
                "settlements": [
                    {
                        "settlement_id": "settle-1",
                        "ticker": "MKT-SETTLE",
                        "market_result": "YES",
                        "yes_count_fp": "5.00",
                        "yes_total_cost_dollars": "2.00",
                        "profit": "3.00",
                        "settled_time": "2026-07-29T11:00:00Z",
                    }
                ],
                "cursor": "",
            }

    try:
        imported = _sync_settlements_from_exchange(
            state_manager=manager,
            kalshi_client=Client(),
            max_pages=2,
        )
        assert imported == 1
        assert manager.get_position("MKT-SETTLE") is None
        assert manager.get_position("MKT-SETTLE", include_closed=True).status == "closed"
        outcome = manager._conn.execute(
            "SELECT * FROM trade_outcomes WHERE market_id = 'MKT-SETTLE'"
        ).fetchone()
        market = manager._conn.execute(
            "SELECT * FROM markets WHERE id = 'MKT-SETTLE'"
        ).fetchone()
        assert outcome["resolution_state"] == "resolved_exchange"
        assert outcome["won"] == 1
        assert market["resolved_winning_outcome"] == "YES"
        assert market["market_status"] == "finalized"
    finally:
        manager.close()


def test_compact_export_is_current_cycle_bounded_and_atomic(tmp_path, monkeypatch) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    market = Market(id="MKT-CURRENT", question="Current?", category="test")
    manager.upsert_market_snapshots([market], cycle_id="cycle-current")
    manager.record_cycle_receipt(
        cycle_id="cycle-current",
        cycle_number=7,
        payload={"complete": True},
    )
    for index in range(4):
        manager.record_decision_receipt(
            cycle_id="cycle-current",
            market_id=f"MKT-{index}",
            decision={"should_trade": False},
            execution_audit={"final_action": "skip"},
        )
    export_path = tmp_path / "state.json"
    try:
        manager.export_to_json(
            str(export_path),
            cycle_id="cycle-current",
            recent_decisions_limit=2,
        )
        payload = json.loads(export_path.read_text(encoding="utf-8"))
        assert payload["cycle_id"] == "cycle-current"
        assert payload["latest_cycle_receipt"]["payload"] == {"complete": True}
        assert len(payload["decision_receipts"]) == 2
        assert [row["id"] for row in payload["markets"]] == ["MKT-CURRENT"]
        previous = export_path.read_text(encoding="utf-8")

        def fail_replace(*args, **kwargs) -> None:
            raise OSError("simulated replace failure")

        monkeypatch.setattr(market_state.os, "replace", fail_replace)
        with pytest.raises(OSError, match="simulated replace failure"):
            manager.export_to_json(str(export_path), cycle_id="cycle-current")
        assert export_path.read_text(encoding="utf-8") == previous
    finally:
        manager.close()


def test_reconciliation_migration_is_idempotent_and_preserves_audit_rows(tmp_path) -> None:
    db_path = str(tmp_path / "state.db")
    manager = MarketStateManager(db_path)
    _pending(manager, "history-order", market_id="MKT-HISTORY", status="filled")
    manager.record_trade(
        "MKT-HISTORY",
        OrderResponse(id="history-order", raw={"outcome": "YES"}),
        2.0,
        outcome="YES",
    )
    manager.record_trade(
        "MKT-RESOLVED-ONLY",
        OrderResponse(id="resolved-only-order", raw={"outcome": "NO"}),
        1.0,
        outcome="NO",
    )
    manager.record_cycle_receipt(cycle_id="c1", cycle_number=1, payload={})
    settled_at = datetime.now(timezone.utc).isoformat()
    with manager._conn:
        manager._conn.execute(
            "UPDATE pending_orders SET terminal_at = NULL WHERE order_id = 'history-order'"
        )
        manager._conn.execute(
            """
            INSERT INTO exchange_settlements (
                settlement_id, market_id, winning_outcome, settled_at, raw_json
            ) VALUES ('s-history', 'MKT-HISTORY', 'YES', ?, '{}')
            """,
            (settled_at,),
        )
        manager._conn.execute(
            "DELETE FROM runtime_flags WHERE key = 'state_reconciliation_schema_version'"
        )
        manager._conn.execute(
            """
            UPDATE trade_outcomes
            SET resolved_winning_outcome = 'NO', won = 1,
                resolved_at = ?, resolution_state = 'unresolved'
            WHERE market_id = 'MKT-RESOLVED-ONLY'
            """,
            (settled_at,),
        )
    manager.close()

    for _ in range(2):
        manager = MarketStateManager(db_path)
        try:
            assert len(manager.get_order_history()) == 1
            assert manager._conn.execute("SELECT COUNT(*) FROM trade_log").fetchone()[0] == 2
            assert manager._conn.execute("SELECT COUNT(*) FROM cycle_receipts").fetchone()[0] == 1
            order = manager.get_pending_order("history-order")
            assert order["terminal_at"] is not None
            assert manager.get_position("MKT-HISTORY") is None
            assert manager.get_position("MKT-RESOLVED-ONLY") is None
            market = manager._conn.execute(
                "SELECT resolved_winning_outcome FROM markets WHERE id = 'MKT-HISTORY'"
            ).fetchone()
            assert market["resolved_winning_outcome"] == "YES"
        finally:
            manager.close()
