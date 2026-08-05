from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from market_state import MarketStateManager
from models import OrderResponse, TradeDecision


def _decision(confidence: float, outcome: str = "YES") -> TradeDecision:
    return TradeDecision(
        should_trade=True,
        outcome=outcome,
        confidence=confidence,
        bet_size_pct=0.5,
        reasoning="test",
    )


def test_market_state_trend_and_counts(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m1"
        confidences = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        for confidence in confidences:
            manager.record_analysis(market_id, _decision(confidence), is_refined=False)

        state = manager.get_market_state(market_id)
        assert state is not None
        assert state.analysis_count == len(confidences)
        assert state.last_confidence == confidences[-1]
        assert state.last_analysis is not None
        assert state.confidence_trend == confidences[-5:]
    finally:
        manager.close()


def test_record_trade_updates_position_and_avg_confidence(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m2"
        manager.record_analysis(market_id, _decision(0.8), is_refined=False)
        order = OrderResponse(id="o1", raw={"outcome": "YES"})
        manager.record_trade(market_id, order, 25.0)

        position = manager.get_position(market_id)
        assert position is not None
        assert position.total_amount_usdc == 25.0
        assert position.avg_confidence == 0.8
        assert position.trade_count == 1

        manager.record_analysis(market_id, _decision(0.6), is_refined=True)
        order2 = OrderResponse(id="o2", raw={"outcome": "YES"})
        manager.record_trade(market_id, order2, 25.0)

        position = manager.get_position(market_id)
        assert position is not None
        assert position.total_amount_usdc == 50.0
        assert round(position.avg_confidence, 4) == 0.7
        assert position.trade_count == 2
    finally:
        manager.close()


def test_pending_order_does_not_create_trade_or_position(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        manager.record_pending_order(
            order_id="pending-1",
            market_id="m-pending",
            outcome="NO",
            submitted_amount_usdc=2.0,
            requested_shares=5.0,
            limit_price=0.40,
            confidence=0.70,
            implied_prob=0.40,
            status="resting",
            raw={"status": "resting"},
        )

        pending = manager.get_pending_order("pending-1")
        assert pending is not None
        assert pending["filled_shares"] == 0.0
        assert manager.get_position("m-pending") is None
        assert "m-pending" not in manager.get_traded_market_ids()
        assert "m-pending" in manager.get_pending_market_ids()
        assert "pending-1" in manager.get_known_order_ids()
    finally:
        manager.close()


def test_pending_fill_updates_are_idempotent_and_delta_based(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        manager.record_pending_order(
            order_id="pending-2",
            market_id="m-partial",
            outcome="YES",
            submitted_amount_usdc=2.5,
            requested_shares=5.0,
            limit_price=0.50,
            confidence=0.75,
            implied_prob=0.50,
            status="resting",
            raw={},
        )

        first = manager.apply_pending_order_fill(
            order_id="pending-2",
            cumulative_filled_shares=2.0,
            fill_price=0.48,
            status="partially_filled",
            raw={"fill_id": "fill-1"},
        )
        duplicate = manager.apply_pending_order_fill(
            order_id="pending-2",
            cumulative_filled_shares=2.0,
            fill_price=0.48,
            status="partially_filled",
            raw={"fill_id": "fill-1"},
        )
        final = manager.apply_pending_order_fill(
            order_id="pending-2",
            cumulative_filled_shares=5.0,
            fill_price=0.49,
            status="filled",
            raw={"fill_id": "fill-2"},
        )

        assert first is not None
        assert first["delta_filled_shares"] == 2.0
        assert first["delta_filled_amount_usdc"] == 0.96
        assert duplicate is not None
        assert duplicate["delta_filled_shares"] == 0.0
        assert final is not None
        assert final["delta_filled_shares"] == 3.0
        assert final["delta_filled_amount_usdc"] == 1.47
        assert final["status"] == "filled"
        assert "m-partial" not in manager.get_pending_market_ids()
    finally:
        manager.close()


def test_pending_fill_rolls_back_when_trade_recording_fails(
    tmp_path,
    monkeypatch,
) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        manager.record_pending_order(
            order_id="pending-rollback",
            market_id="m-rollback",
            outcome="YES",
            submitted_amount_usdc=2.0,
            requested_shares=4.0,
            limit_price=0.50,
            confidence=0.70,
            implied_prob=0.50,
            status="resting",
            raw={},
        )

        def fail_record_trade(*args, **kwargs) -> None:
            raise RuntimeError("simulated trade persistence failure")

        monkeypatch.setattr(manager, "record_trade", fail_record_trade)
        with pytest.raises(RuntimeError, match="simulated trade persistence failure"):
            manager.apply_pending_order_fill(
                order_id="pending-rollback",
                cumulative_filled_shares=4.0,
                fill_price=0.50,
                status="filled",
                record_trade_order=OrderResponse(id="pending-rollback"),
            )

        pending = manager.get_pending_order("pending-rollback")
        assert pending is not None
        assert pending["filled_shares"] == 0.0
        assert pending["status"] == "resting"
        assert manager.get_position("m-rollback") is None
    finally:
        manager.close()


def test_multiple_fill_deltas_for_one_order_count_as_one_trade(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        manager.record_analysis("m-delta", _decision(0.8), is_refined=False)
        order = OrderResponse(id="order-delta", raw={"outcome": "YES"})

        manager.record_trade(
            "m-delta",
            order,
            0.96,
            outcome="YES",
            entry_price=0.48,
            shares=2.0,
        )
        manager.record_trade(
            "m-delta",
            order,
            1.47,
            outcome="YES",
            entry_price=0.49,
            shares=3.0,
        )

        position = manager.get_position("m-delta")
        assert position is not None
        assert position.total_amount_usdc == pytest.approx(2.43)
        assert position.trade_count == 1
        event = manager._conn.execute(
            """
            SELECT amount_usdc, shares
            FROM trade_outcome_events
            WHERE market_id = ? AND order_id = ?
            """,
            ("m-delta", "order-delta"),
        ).fetchone()
        assert event is not None
        assert event["amount_usdc"] == pytest.approx(2.43)
        assert event["shares"] == 5.0
    finally:
        manager.close()


def test_get_markets_needing_reanalysis(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m3"
        manager.record_analysis(market_id, _decision(0.7), is_refined=False)
        old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()
        manager._conn.execute(
            "UPDATE analyses SET timestamp = ? WHERE market_id = ?",
            (old_timestamp, market_id),
        )
        manager._conn.commit()

        needs_reanalysis = manager.get_markets_needing_reanalysis(6)
        assert market_id in needs_reanalysis

        fresh_only = manager.get_markets_needing_reanalysis(12)
        assert market_id not in fresh_only
    finally:
        manager.close()


def test_export_to_json(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m4"
        manager.record_analysis(market_id, _decision(0.9), is_refined=False)
        order = OrderResponse(id="o9", raw={"outcome": "YES"})
        manager.record_trade(market_id, order, 15.0)

        export_path = tmp_path / "state.json"
        manager.export_to_json(str(export_path))

        payload = json.loads(export_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 2
        assert payload["source_checkpoint"] == {
            "cycle_receipt_id": None,
            "decision_receipt_id": None,
        }
        assert payload["open_positions"][0]["order_ids"] == ["o9"]
        assert payload["active_pending_orders"] == []
        assert payload["historical_counts"]["trade_log"] == 1
        assert payload["historical_counts"]["trade_outcomes"] == 1
        assert "analyses" not in payload
        assert "trade_log" not in payload
    finally:
        manager.close()


def test_receipt_persistence_tables(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        manager.record_cycle_receipt(
            cycle_id="cycle-1",
            cycle_number=1,
            payload={"analyzed_markets": 3, "order_attempts": 1},
        )
        manager.record_decision_receipt(
            cycle_id="cycle-1",
            market_id="m-receipt",
            decision={"should_trade": False, "confidence": 0.61},
            execution_audit={
                "final_action": "skip",
                "final_reason": "score_gate_blocked",
                "score_breakdown": {"final_score": 0.28, "score_threshold": 0.38},
            },
        )

        cycle_row = manager._conn.execute(
            "SELECT cycle_id, cycle_number, payload_json FROM cycle_receipts LIMIT 1"
        ).fetchone()
        assert cycle_row is not None
        assert cycle_row["cycle_id"] == "cycle-1"
        assert cycle_row["cycle_number"] == 1

        decision_row = manager._conn.execute(
            "SELECT market_id, final_action, final_reason, score_json FROM decision_receipts LIMIT 1"
        ).fetchone()
        assert decision_row is not None
        assert decision_row["market_id"] == "m-receipt"
        assert decision_row["final_action"] == "skip"
        assert decision_row["final_reason"] == "score_gate_blocked"
        assert decision_row["score_json"] is not None
        score_payload = json.loads(decision_row["score_json"])
        assert score_payload["final_score"] == 0.28
        assert score_payload["score_threshold"] == 0.38
    finally:
        manager.close()


def test_decision_receipt_infers_order_summary_from_audit(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        manager.record_decision_receipt(
            cycle_id="cycle-order",
            market_id="m-order",
            decision={"should_trade": True, "confidence": 0.72},
            execution_audit={
                "final_action": "order_attempt",
                "final_reason": "order_submitted",
                "order_id": "order-123",
                "order_status": "filled",
                "order_fill_count": 4,
            },
        )

        row = manager._conn.execute(
            "SELECT order_json FROM decision_receipts WHERE market_id = 'm-order'"
        ).fetchone()
        assert row is not None
        order_payload = json.loads(row["order_json"])
        assert order_payload["order_id"] == "order-123"
        assert order_payload["order_status"] == "filled"
        assert order_payload["order_fill_count"] == 4
    finally:
        manager.close()


def test_daily_order_attempt_summary_restores_count_and_expected_value(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        manager.record_decision_receipt(
            cycle_id="cycle-order-1",
            market_id="m-order-1",
            decision={"should_trade": True},
            execution_audit={
                "final_action": "order_attempt",
                "final_reason": "order_submitted",
                "expected_value_usdc": 1.25,
            },
        )
        manager.record_decision_receipt(
            cycle_id="cycle-order-2",
            market_id="m-order-2",
            decision={"should_trade": True},
            execution_audit={
                "final_action": "order_attempt",
                "final_reason": "dry_run",
                "expected_value_usdc": 0.75,
            },
        )
        manager.record_decision_receipt(
            cycle_id="cycle-skip",
            market_id="m-skip",
            decision={"should_trade": False},
            execution_audit={
                "final_action": "skip",
                "final_reason": "score_gate_blocked",
                "expected_value_usdc": 100.0,
            },
        )
        manager.record_decision_receipt(
            cycle_id="cycle-failed-attempt",
            market_id="m-failed-attempt",
            decision={"should_trade": True},
            execution_audit={
                "final_action": "order_attempt",
                "final_reason": "order_submission_failed",
                "expected_value_usdc": 20.0,
                "daily_expectancy_ev_credited": False,
            },
        )

        attempts, exposures, projected_ev = manager.get_daily_order_attempt_summary(
            since=datetime.now(timezone.utc) - timedelta(hours=1)
        )
        live_attempts, live_exposures, live_projected_ev = manager.get_daily_order_attempt_summary(
            since=datetime.now(timezone.utc) - timedelta(hours=1),
            include_dry_run=False,
        )

        assert attempts == 3
        assert exposures == 2
        assert projected_ev == 2.0
        assert live_attempts == 2
        assert live_exposures == 1
        assert live_projected_ev == 1.25
    finally:
        manager.close()


def test_decision_receipt_audit_json_populated_on_abstain_and_skip(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        manager.record_decision_receipt(
            cycle_id="cycle-audit",
            market_id="m-abstain",
            decision={"should_trade": False, "abstain": True, "confidence": 0.41},
            execution_audit={
                "final_action": "skip",
                "final_reason": "abstain_low_evidence",
                "score_breakdown": {"score_final": 0.05},
            },
        )
        manager.record_decision_receipt(
            cycle_id="cycle-audit",
            market_id="m-research",
            decision={"should_trade": False, "abstain": False, "confidence": 0.55},
            execution_audit={
                "final_action": "research_queued",
                "final_reason": "hallucinated_edge",
                "score_breakdown": {"score_final": 0.21},
            },
        )
        rows = manager._conn.execute(
            """
            SELECT market_id, audit_json
            FROM decision_receipts
            WHERE cycle_id = 'cycle-audit'
            ORDER BY market_id
            """
        ).fetchall()
        assert len(rows) == 2
        for row in rows:
            payload = json.loads(row["audit_json"])
            assert payload.get("final_action") in {"skip", "research_queued"}
            assert payload.get("final_reason")
            assert "score_breakdown" in payload
    finally:
        manager.close()


def test_load_confidence_calibration_buckets_groups_all_and_family(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        resolved_at = datetime.now(timezone.utc).isoformat()
        with manager._conn:
            manager._conn.execute(
                "INSERT OR REPLACE INTO markets (id, question, close_time, category) VALUES (?, ?, ?, ?)",
                ("KXRAINNYC-TEST", "Will NYC rain tomorrow?", "", "weather"),
            )
            manager._conn.execute(
                "INSERT OR REPLACE INTO markets (id, question, close_time, category) VALUES (?, ?, ?, ?)",
                ("KXSAMPLEGAME-TEST", "Will Team A beat Team B?", "", "sports"),
            )
            manager._conn.execute(
                """
                INSERT OR REPLACE INTO trade_outcomes (
                    market_id, confidence, won, resolved_at, last_updated
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                ("KXRAINNYC-TEST", 0.84, 0, resolved_at, resolved_at),
            )
            manager._conn.execute(
                """
                INSERT OR REPLACE INTO trade_outcomes (
                    market_id, confidence, won, resolved_at, last_updated
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                ("KXSAMPLEGAME-TEST", 0.86, 1, resolved_at, resolved_at),
            )
        buckets = manager.load_confidence_calibration_buckets(days=30)
        assert "all" in buckets
        assert 0.8 in buckets["all"]
        assert buckets["all"][0.8]["sample_size"] == 2
        assert "weather" in buckets
        assert buckets["weather"][0.8]["sample_size"] == 1
    finally:
        manager.close()


def test_online_confidence_calibration_ema_and_sample_cap(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        manager.record_online_confidence_calibration(
            market_id="KXRAINNYC-TEST",
            confidence=0.84,
            won=True,
            question="Will NYC rain tomorrow?",
            category="weather",
            alpha=0.25,
            max_samples_per_bucket=2,
        )
        manager.record_online_confidence_calibration(
            market_id="KXRAINNYC-TEST",
            confidence=0.84,
            won=False,
            question="Will NYC rain tomorrow?",
            category="weather",
            alpha=0.25,
            max_samples_per_bucket=2,
        )
        manager.record_online_confidence_calibration(
            market_id="KXRAINNYC-TEST",
            confidence=0.84,
            won=False,
            question="Will NYC rain tomorrow?",
            category="weather",
            alpha=0.25,
            max_samples_per_bucket=2,
        )

        row = manager._conn.execute(
            """
            SELECT win_rate, sample_size
            FROM confidence_calibration_online
            WHERE family = 'weather' AND bucket = 0.8
            """
        ).fetchone()
        assert row is not None
        assert row["sample_size"] == 2
        assert round(float(row["win_rate"]), 4) == 0.5625
        fallback = manager.load_confidence_calibration_buckets(days=30)
        assert fallback["all"][0.8]["sample_size"] == 2
        assert round(float(fallback["all"][0.8]["win_rate"]), 4) == 0.5625
    finally:
        manager.close()


def test_record_resolution_can_update_online_calibration(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "KXRAINNYC-RES"
        manager._conn.execute(
            "INSERT OR REPLACE INTO markets (id, question, close_time, category) VALUES (?, ?, ?, ?)",
            (market_id, "Will NYC rain tomorrow?", "", "weather"),
        )
        manager.record_analysis(market_id, _decision(0.84), is_refined=False)
        manager.record_trade(
            market_id,
            OrderResponse(id="o-online", raw={"outcome": "YES"}),
            10.0,
            outcome="YES",
        )
        updated = manager.record_resolution(
            market_id,
            "YES",
            datetime.now(timezone.utc),
            online_calibration_enabled=True,
            online_calibration_alpha=0.15,
            online_calibration_max_samples_per_bucket=500,
        )
        assert updated is True

        buckets = manager.load_confidence_calibration_buckets(days=30)
        assert "weather" in buckets
        assert buckets["weather"][0.8]["sample_size"] == 1
        row = manager._conn.execute(
            "SELECT sample_size FROM confidence_calibration_online WHERE family = 'weather' AND bucket = 0.8"
        ).fetchone()
        assert row is not None
        assert row["sample_size"] == 1
    finally:
        manager.close()


def test_record_market_winning_outcome_is_update_only(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        # Unknown market: no row is created (writes stay scoped to markets
        # with local decision history).
        assert manager.record_market_winning_outcome("KX-UNKNOWN", "YES") is False
        # A blocked market with a decision-reason terminal outcome later gets
        # the actual exchange winner recorded alongside it.
        manager.record_terminal_outcome("KX-BLOCKED", "edge_gate_blocked")
        assert manager.record_market_winning_outcome("KX-BLOCKED", "YES") is True
        assert manager.record_market_winning_outcome("KX-BLOCKED", "YES") is False
        row = manager._conn.execute(
            """
            SELECT last_terminal_outcome, resolved_winning_outcome,
                   resolved_winning_outcome_at
            FROM markets WHERE id = ?
            """,
            ("KX-BLOCKED",),
        ).fetchone()
        assert row["last_terminal_outcome"] == "edge_gate_blocked"
        assert row["resolved_winning_outcome"] == "YES"
        assert row["resolved_winning_outcome_at"] is not None
    finally:
        manager.close()


def test_participation_tier_repr_leak_migration_normalizes_receipts(tmp_path) -> None:
    db_path = str(tmp_path / "state.db")
    manager = MarketStateManager(db_path)
    try:
        leaked_audit = (
            '{"participation_tier": "ParticipationTier.SKIP_FOR_NOW_WITH_REASON"}'
        )
        clean_decision = '{"should_trade": false}'
        manager._conn.execute(
            """
            INSERT INTO decision_receipts
                (cycle_id, market_id, final_action, timestamp,
                 decision_json, audit_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "c1",
                "KX-LEAK",
                "skip",
                datetime.now(timezone.utc).isoformat(),
                clean_decision,
                leaked_audit,
            ),
        )
        # Re-arm the one-shot migration so the reopened manager repairs the row.
        manager._conn.execute(
            "DELETE FROM runtime_flags WHERE key = 'participation_tier_repr_normalized'"
        )
        manager._conn.commit()
    finally:
        manager.close()

    reopened = MarketStateManager(db_path)
    try:
        row = reopened._conn.execute(
            "SELECT audit_json FROM decision_receipts WHERE market_id = ?",
            ("KX-LEAK",),
        ).fetchone()
        assert row["audit_json"] == (
            '{"participation_tier": "skip_for_now_with_reason"}'
        )
        flag = reopened.get_runtime_flag("participation_tier_repr_normalized")
        assert flag == "v1"
    finally:
        reopened.close()


def test_runtime_flags_persist_across_manager_instances(tmp_path) -> None:
    db_path = str(tmp_path / "runtime_flags.db")
    manager = MarketStateManager(db_path)
    try:
        assert manager.get_runtime_flag("sports_jurisdiction_blocked") is None
        manager.set_runtime_flag("sports_jurisdiction_blocked", "1")
        assert manager.get_runtime_flag("sports_jurisdiction_blocked") == "1"
    finally:
        manager.close()

    restored = MarketStateManager(db_path)
    try:
        assert restored.get_runtime_flag("sports_jurisdiction_blocked") == "1"
        assert restored.clear_runtime_flag("sports_jurisdiction_blocked") is True
        assert restored.get_runtime_flag("sports_jurisdiction_blocked") is None
    finally:
        restored.close()


def test_neutralize_pathological_online_calibration(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "calib.db"))
    try:
        now = datetime.now(timezone.utc).isoformat()
        manager._conn.execute(
            """
            INSERT INTO confidence_calibration_online
                (family, bucket, win_rate, sample_size, updated_at)
            VALUES
                ('sports', 0.7, 0.0, 500, ?),
                ('sports', 0.5, 1.0, 52, ?),
                ('sports', 0.6, 0.55, 40, ?),
                ('crypto', 0.7, 0.0, 500, ?)
            """,
            (now, now, now, now),
        )
        manager._conn.commit()
        changed = manager.neutralize_pathological_online_calibration(family="sports")
        assert changed == 2
        rows = {
            float(row["bucket"]): float(row["win_rate"])
            for row in manager._conn.execute(
                "SELECT bucket, win_rate FROM confidence_calibration_online WHERE family = 'sports'"
            ).fetchall()
        }
        assert rows[0.7] == 0.50
        assert rows[0.5] == 0.50
        assert rows[0.6] == 0.55
        crypto = manager._conn.execute(
            "SELECT win_rate FROM confidence_calibration_online WHERE family = 'crypto' AND bucket = 0.7"
        ).fetchone()
        assert float(crypto["win_rate"]) == 0.0
    finally:
        manager.close()

    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m5"
        manager.record_analysis(market_id, _decision(0.7), is_refined=False)
        manager.record_trade(market_id, OrderResponse(id="o1", raw={"outcome": "YES"}), 10.0, outcome="YES")
        updated = manager.record_resolution(market_id, "YES", datetime.now(timezone.utc))
        assert updated is True
        updated_again = manager.record_resolution(market_id, "YES", datetime.now(timezone.utc))
        assert updated_again is False
    finally:
        manager.close()


def test_upsert_position_snapshot_updates_existing_row(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m-sync"
        manager.upsert_position_snapshot(
            market_id=market_id,
            outcome="YES",
            total_amount_usdc=7.5,
        )
        position = manager.get_position(market_id)
        assert position is not None
        assert position.outcome == "YES"
        assert position.total_amount_usdc == 7.5

        manager.upsert_position_snapshot(
            market_id=market_id,
            outcome="NO",
            total_amount_usdc=3.25,
        )
        position = manager.get_position(market_id)
        assert position is not None
        assert position.outcome == "NO"
        assert position.total_amount_usdc == 3.25
    finally:
        manager.close()


def test_backfill_sentinel_resolution_to_unresolved(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m6"
        manager.record_analysis(market_id, _decision(0.7), is_refined=False)
        manager.record_trade(market_id, OrderResponse(id="o2", raw={"outcome": "YES"}), 5.0, outcome="YES")
        manager._conn.execute(
            """
            UPDATE trade_outcomes
            SET resolved_winning_outcome = '18446744073709551615', won = 0, pnl_estimate = -1.0, resolved_at = ?
            WHERE market_id = ?
            """,
            (datetime.now(timezone.utc).isoformat(), market_id),
        )
        manager._conn.commit()
        manager._backfill_resolution_state()
        row = manager._conn.execute(
            "SELECT resolved_winning_outcome, won, pnl_estimate, resolved_at, resolution_state FROM trade_outcomes WHERE market_id = ?",
            (market_id,),
        ).fetchone()
        assert row["resolved_winning_outcome"] is None
        assert row["won"] is None
        assert row["pnl_estimate"] is None
        assert row["resolved_at"] is None
        assert row["resolution_state"] == "unresolved"
    finally:
        manager.close()


def test_get_anchor_analysis_prefers_high_confidence(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m7"
        manager.record_analysis(market_id, _decision(0.55, outcome="YES"), is_refined=False)
        manager.record_analysis(market_id, _decision(0.72, outcome="NO"), is_refined=False)
        manager.record_analysis(market_id, _decision(0.61, outcome="YES"), is_refined=False)

        anchor = manager.get_anchor_analysis(market_id, min_confidence=0.65)
        assert anchor is not None
        assert anchor["outcome"] == "NO"
        assert round(float(anchor["confidence"]), 2) == 0.72
    finally:
        manager.close()


def test_get_anchor_analysis_falls_back_to_latest(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m8"
        manager.record_analysis(market_id, _decision(0.50, outcome="YES"), is_refined=False)
        manager.record_analysis(market_id, _decision(0.58, outcome="NO"), is_refined=False)

        anchor = manager.get_anchor_analysis(market_id, min_confidence=0.65)
        assert anchor is not None
        assert anchor["outcome"] == "NO"
        assert round(float(anchor["confidence"]), 2) == 0.58
    finally:
        manager.close()


def test_record_terminal_outcome_persists_on_market_state(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m9"
        manager.record_analysis(market_id, _decision(0.61), is_refined=False)
        manager.record_terminal_outcome(market_id, "no_trade_recommended")
        state = manager.get_market_state(market_id)
        assert state is not None
        assert state.last_terminal_outcome == "no_trade_recommended"
        assert state.non_actionable_streak == 1

        manager.record_terminal_outcome(market_id, "score_gate_blocked")
        state = manager.get_market_state(market_id)
        assert state is not None
        assert state.non_actionable_streak == 2

        manager.record_terminal_outcome(market_id, "analysis_failure")
        state = manager.get_market_state(market_id)
        assert state is not None
        assert state.non_actionable_streak == 3

        manager.record_terminal_outcome(market_id, "order_submitted")
        state = manager.get_market_state(market_id)
        assert state is not None
        assert state.non_actionable_streak == 0
    finally:
        manager.close()


def test_reasoning_hash_and_stale_bayesian_update(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m-stale"
        decision = _decision(0.66, outcome="YES")
        manager.record_analysis(market_id, decision, is_refined=False)
        first_hash = manager.get_last_reasoning_hash(market_id)
        assert first_hash is not None

        manager.record_analysis(market_id, decision, is_refined=False)
        second_hash = manager.get_last_reasoning_hash(market_id)
        assert second_hash == first_hash

        manager.update_bayesian_state(
            market_id=market_id,
            outcome="YES",
            log_prior=0.0,
            log_likelihood=0.2,
            count_as_update=True,
        )
        manager.update_bayesian_state(
            market_id=market_id,
            outcome="YES",
            log_prior=0.0,
            log_likelihood=0.2,
            count_as_update=False,
        )
        state = manager.get_bayesian_state(market_id)["YES"]
        assert state.update_count == 1
    finally:
        manager.close()


def test_record_exchange_settlement_upserts_trade_outcome_and_pnl(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        manager.record_exchange_settlement(
            settlement_id="settle-1",
            market_id="KXWTI-26APR13-T94.99",
            winning_outcome="YES",
            predicted_outcome="YES",
            pnl_realized=2.15,
            contracts=5,
            avg_price=0.57,
            settled_at=datetime.now(timezone.utc),
            raw={"market_result": "yes"},
        )
        row = manager._conn.execute(
            """
            SELECT resolved_winning_outcome, won, pnl_estimate, resolution_state
            FROM trade_outcomes
            WHERE market_id = ?
            """,
            ("KXWTI-26APR13-T94.99",),
        ).fetchone()
        assert row is not None
        assert row["resolved_winning_outcome"] == "YES"
        assert row["won"] == 1
        assert round(float(row["pnl_estimate"]), 2) == 2.15
        assert row["resolution_state"] == "resolved_exchange"
        assert round(manager.get_exchange_realized_pnl_total(), 2) == 2.15
    finally:
        manager.close()


def test_get_exchange_realized_pnl_since_hours_filters_by_settled_at(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        old_time = datetime.now(timezone.utc) - timedelta(days=10)
        recent_time = datetime.now(timezone.utc) - timedelta(hours=2)
        manager.record_exchange_settlement(
            settlement_id="old-settlement",
            market_id="OLD-1",
            predicted_outcome="YES",
            winning_outcome="YES",
            pnl_realized=-5.0,
            contracts=10,
            avg_price=0.5,
            settled_at=old_time,
            raw={"settled_time": old_time.isoformat()},
        )
        manager.record_exchange_settlement(
            settlement_id="recent-settlement",
            market_id="NEW-1",
            predicted_outcome="YES",
            winning_outcome="YES",
            pnl_realized=3.0,
            contracts=10,
            avg_price=0.5,
            settled_at=recent_time,
            raw={"settled_time": recent_time.isoformat()},
        )
        assert round(manager.get_exchange_realized_pnl_since_hours(24.0), 2) == 3.0
        assert round(manager.get_exchange_realized_pnl_total(), 2) == -2.0
    finally:
        manager.close()


def test_record_exchange_settlement_can_update_online_calibration(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "KXWTI-26APR13-T94.99"
        now = datetime.now(timezone.utc).isoformat()
        with manager._conn:
            manager._conn.execute(
                "INSERT OR REPLACE INTO markets (id, question, close_time, category) VALUES (?, ?, ?, ?)",
                (market_id, "Will WTI settle above threshold?", "", "commodities"),
            )
            manager._conn.execute(
                """
                INSERT OR REPLACE INTO trade_outcomes (
                    market_id, predicted_outcome, confidence, last_updated
                )
                VALUES (?, ?, ?, ?)
                """,
                (market_id, "YES", 0.76, now),
            )
        manager.record_exchange_settlement(
            settlement_id="settle-online",
            market_id=market_id,
            winning_outcome="YES",
            predicted_outcome="YES",
            pnl_realized=2.15,
            contracts=5,
            avg_price=0.57,
            settled_at=datetime.now(timezone.utc),
            raw={"market_result": "yes"},
            online_calibration_enabled=True,
            online_calibration_alpha=0.15,
            online_calibration_max_samples_per_bucket=500,
        )
        manager.record_exchange_settlement(
            settlement_id="settle-online",
            market_id=market_id,
            winning_outcome="YES",
            predicted_outcome="YES",
            pnl_realized=2.15,
            contracts=5,
            avg_price=0.57,
            settled_at=datetime.now(timezone.utc),
            raw={"market_result": "yes"},
            online_calibration_enabled=True,
            online_calibration_alpha=0.15,
            online_calibration_max_samples_per_bucket=500,
        )

        row = manager._conn.execute(
            "SELECT sample_size, win_rate FROM confidence_calibration_online WHERE family = 'all' AND bucket = 0.7"
        ).fetchone()
        assert row is not None
        assert row["sample_size"] == 1
        assert row["win_rate"] == 1.0
    finally:
        manager.close()


def test_binary_bayesian_likelihood_migration_halves_legacy_sums_once(tmp_path) -> None:
    db_path = str(tmp_path / "state.db")
    manager = MarketStateManager(db_path)
    try:
        with manager._conn:
            manager._conn.execute(
                "DELETE FROM runtime_flags WHERE key = 'bayesian_lr_semantics_version'"
            )
            manager._conn.execute(
                """
                INSERT INTO bayesian_state (
                    market_id, outcome, log_prior, log_likelihood_sum,
                    update_count, last_updated
                )
                VALUES
                    ('legacy-binary', 'YES', ?, ?, 1, ?),
                    ('legacy-binary', 'NO', ?, ?, 1, ?)
                """,
                (
                    math.log(0.5),
                    math.log(2.0),
                    datetime.now(timezone.utc).isoformat(),
                    math.log(0.5),
                    -math.log(2.0),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    finally:
        manager.close()

    migrated = MarketStateManager(db_path)
    try:
        states = migrated.get_bayesian_state("legacy-binary")
        assert states["YES"].log_likelihoods[0] == pytest.approx(0.5 * math.log(2.0))
        assert states["NO"].log_likelihoods[0] == pytest.approx(-0.5 * math.log(2.0))
    finally:
        migrated.close()

    reopened = MarketStateManager(db_path)
    try:
        states = reopened.get_bayesian_state("legacy-binary")
        assert states["YES"].log_likelihoods[0] == pytest.approx(0.5 * math.log(2.0))
        assert states["NO"].log_likelihoods[0] == pytest.approx(-0.5 * math.log(2.0))
    finally:
        reopened.close()


def test_exchange_settlement_calibrates_once_when_resolution_arrives_later(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "KXWTI-UNRESOLVED-FIRST"
        now = datetime.now(timezone.utc).isoformat()
        with manager._conn:
            manager._conn.execute(
                "INSERT OR REPLACE INTO markets (id, question, close_time, category) VALUES (?, ?, ?, ?)",
                (market_id, "Will WTI settle above threshold?", "", "commodities"),
            )
            manager._conn.execute(
                """
                INSERT OR REPLACE INTO trade_outcomes (
                    market_id, predicted_outcome, confidence, last_updated
                )
                VALUES (?, ?, ?, ?)
                """,
                (market_id, "YES", 0.76, now),
            )

        common = {
            "settlement_id": "settle-late-resolution",
            "market_id": market_id,
            "predicted_outcome": "YES",
            "pnl_realized": 0.0,
            "contracts": 5,
            "avg_price": 0.57,
            "settled_at": datetime.now(timezone.utc),
            "online_calibration_enabled": True,
        }
        manager.record_exchange_settlement(
            **common,
            winning_outcome=None,
            raw={"status": "pending"},
        )
        assert manager._conn.execute(
            "SELECT COUNT(*) FROM confidence_calibration_online"
        ).fetchone()[0] == 0

        manager.record_exchange_settlement(
            **common,
            winning_outcome="YES",
            raw={"market_result": "yes"},
        )
        manager.record_exchange_settlement(
            **common,
            winning_outcome="YES",
            raw={"market_result": "yes"},
        )
        row = manager._conn.execute(
            "SELECT sample_size FROM confidence_calibration_online WHERE family = 'all' AND bucket = 0.7"
        ).fetchone()
        assert row is not None
        assert row["sample_size"] == 1
    finally:
        manager.close()


def test_get_known_order_ids_returns_logged_orders(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        manager.record_trade(
            "m-known-order",
            OrderResponse(id="ord-known", raw={"outcome": "YES"}),
            3.0,
            outcome="YES",
        )
        known_order_ids = manager.get_known_order_ids()
        assert "ord-known" in known_order_ids
    finally:
        manager.close()


def test_get_confidence_tier_outcomes_returns_bucketed_stats(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        manager._conn.execute(
            """
            INSERT INTO trade_outcomes (
                market_id, predicted_outcome, confidence, won, pnl_estimate, resolution_state
            )
            VALUES
                ('m-tier-1', 'YES', 0.91, 1, 3.2, 'resolved_exchange'),
                ('m-tier-2', 'YES', 0.83, 0, -1.7, 'resolved_exchange'),
                ('m-tier-3', 'NO', 0.74, 1, 2.1, 'resolved_exchange'),
                ('m-tier-4', 'NO', 0.62, 0, -0.9, 'resolved_exchange')
            """
        )
        manager._conn.commit()

        tiers = manager.get_confidence_tier_outcomes()
        tier_names = [row["tier"] for row in tiers]
        assert tier_names == ["0.90+", "0.80-0.89", "0.70-0.79", "0.60-0.69"]
        assert tiers[0]["wins"] == 1
        assert tiers[1]["losses"] == 1
    finally:
        manager.close()


def test_reasoning_hash_ignores_validated_prefix_variation(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m-hash"
        manager.record_analysis(
            market_id,
            TradeDecision(
                should_trade=False,
                outcome="YES",
                confidence=0.66,
                bet_size_pct=0.0,
                reasoning="[Validated eq=1.00 edge_market=0.031] thesis text",
            ),
            is_refined=False,
        )
        first_hash = manager.get_last_reasoning_hash(market_id)
        manager.record_analysis(
            market_id,
            TradeDecision(
                should_trade=False,
                outcome="YES",
                confidence=0.66,
                bet_size_pct=0.0,
                reasoning="[Validated eq=0.95 edge_market=0.028] thesis text",
            ),
            is_refined=False,
        )
        second_hash = manager.get_last_reasoning_hash(market_id)
        assert first_hash == second_hash
    finally:
        manager.close()


def test_get_outcome_flip_count_counts_transitions(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m-flips"
        manager.record_analysis(market_id, _decision(0.60, outcome="YES"), is_refined=False)
        manager.record_analysis(market_id, _decision(0.62, outcome="NO"), is_refined=False)
        manager.record_analysis(market_id, _decision(0.64, outcome="YES"), is_refined=False)
        manager.record_analysis(market_id, _decision(0.66, outcome="NO"), is_refined=False)
        assert manager.get_outcome_flip_count(market_id) == 3
    finally:
        manager.close()


def test_get_last_trade_entry_price_returns_latest(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m-last-entry"
        manager.record_analysis(market_id, _decision(0.60, outcome="YES"), is_refined=False)
        manager.record_trade(
            market_id,
            OrderResponse(id="o-entry-1", raw={"outcome": "YES"}),
            5.0,
            outcome="YES",
            entry_price=0.44,
        )
        manager.record_trade(
            market_id,
            OrderResponse(id="o-entry-2", raw={"outcome": "YES"}),
            5.0,
            outcome="YES",
            entry_price=0.51,
        )
        assert manager.get_last_trade_entry_price(market_id) == 0.51
    finally:
        manager.close()


def test_fill_failure_count_tracking(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m-fill-failure"
        manager.increment_fill_failure_count(market_id)
        manager.increment_fill_failure_count(market_id)
        manager.increment_fill_failure_count(market_id)
        state = manager.get_market_state(market_id)
        assert state is not None
        assert state.fill_failure_count == 3

        manager.reset_fill_failure_count(market_id)
        state = manager.get_market_state(market_id)
        assert state is not None
        assert state.fill_failure_count == 0
    finally:
        manager.close()


def test_market_cooldown_cycle_tracking(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m-cooldown"
        manager.set_market_cooldown_cycle(market_id, 11)
        state = manager.get_market_state(market_id)
        assert state is not None
        assert state.next_eligible_cycle == 11

        manager.record_terminal_outcome(market_id, "no_trade_recommended")
        state = manager.get_market_state(market_id)
        assert state is not None
        assert state.next_eligible_cycle == 11

        manager.set_market_cooldown_cycle(market_id, 0)
        state = manager.get_market_state(market_id)
        assert state is not None
        assert state.next_eligible_cycle == 0
    finally:
        manager.close()


def test_backfill_outcomes_from_settlements(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        manager._conn.execute(
            """INSERT INTO trade_outcomes (market_id, predicted_outcome, confidence, last_updated)
               VALUES (?, ?, ?, ?)""",
            ("MARKET_A", "YES", 0.75, now_iso),
        )
        manager._conn.execute(
            """INSERT INTO exchange_settlements
               (settlement_id, market_id, won, pnl_realized, contracts, avg_price, settled_at, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("S1", "MARKET_A", 1, 5.50, 10, 0.55, now_iso, "{}"),
        )
        manager._conn.commit()

        row_before = manager._conn.execute(
            "SELECT won, pnl_estimate FROM trade_outcomes WHERE market_id = 'MARKET_A'"
        ).fetchone()
        assert row_before["won"] is None

        updated = manager.backfill_outcomes_from_settlements()
        assert updated == 1

        row_after = manager._conn.execute(
            "SELECT won, pnl_estimate, resolution_state FROM trade_outcomes WHERE market_id = 'MARKET_A'"
        ).fetchone()
        assert row_after["won"] == 1
        assert row_after["pnl_estimate"] == 5.50
        assert row_after["resolution_state"] == "resolved_valid"

        updated_again = manager.backfill_outcomes_from_settlements()
        assert updated_again == 0
    finally:
        manager.close()


def test_attributed_daily_realized_pnl_excludes_legacy_entries(tmp_path) -> None:
    """Drawdown attribution: only settlements of same-window entries count."""
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        now = datetime.now(timezone.utc)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        manager.record_trade(
            "m-today-loss",
            OrderResponse(id="o-today-loss", raw={"outcome": "YES"}),
            5.0,
            outcome="YES",
        )
        manager.record_trade(
            "m-today-win",
            OrderResponse(id="o-today-win", raw={"outcome": "YES"}),
            5.0,
            outcome="YES",
        )
        manager.record_trade(
            "m-legacy",
            OrderResponse(id="o-legacy", raw={"outcome": "YES"}),
            5.0,
            outcome="YES",
        )
        legacy_entry_time = (day_start - timedelta(days=20)).isoformat()
        manager._conn.execute(
            "UPDATE trade_log SET timestamp = ? WHERE market_id = 'm-legacy'",
            (legacy_entry_time,),
        )

        manager.record_exchange_settlement(
            settlement_id="s-today-loss",
            market_id="m-today-loss",
            winning_outcome="NO",
            predicted_outcome="YES",
            pnl_realized=-4.5,
            contracts=10,
            avg_price=0.45,
            settled_at=now,
            raw={},
        )
        manager.record_exchange_settlement(
            settlement_id="s-today-win",
            market_id="m-today-win",
            winning_outcome="YES",
            predicted_outcome="YES",
            pnl_realized=2.0,
            contracts=5,
            avg_price=0.60,
            settled_at=now,
            raw={},
        )
        # Legacy position (entered 20 days ago) settling today must not count.
        manager.record_exchange_settlement(
            settlement_id="s-legacy",
            market_id="m-legacy",
            winning_outcome="NO",
            predicted_outcome="YES",
            pnl_realized=-8.25,
            contracts=20,
            avg_price=0.41,
            settled_at=now,
            raw={},
        )
        # Settlement for a market never traded locally must not count either.
        manager.record_exchange_settlement(
            settlement_id="s-untraded",
            market_id="m-untraded",
            winning_outcome="NO",
            predicted_outcome="YES",
            pnl_realized=-3.0,
            contracts=6,
            avg_price=0.50,
            settled_at=now,
            raw={},
        )

        attributed = manager.get_attributed_daily_realized_pnl(day_start)
        assert round(attributed, 2) == -2.5
    finally:
        manager.close()
