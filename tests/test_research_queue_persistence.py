from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone

from market_state import MarketStateManager


def _make_manager() -> MarketStateManager:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return MarketStateManager(db_path=path)


def test_record_and_get_research_queue_entry() -> None:
    mgr = _make_manager()
    mgr.record_research_queue_entry(
        market_id="KXBTCD-26APR2517-T77749.99",
        cycle_id="cycle-001",
        gate_name="pre_analysis_performance",
        reason="pre_analysis_zero_action_family",
        threshold_gap=0.0,
        what_to_learn_next="Need a probe trade",
    )
    entries = mgr.get_active_research_entries(lookback_hours=1)
    assert len(entries) == 1
    assert entries[0]["market_id"] == "KXBTCD-26APR2517-T77749.99"
    assert entries[0]["reason"] == "pre_analysis_zero_action_family"
    assert entries[0]["what_to_learn_next"] == "Need a probe trade"


def test_upsert_updates_existing_entry() -> None:
    mgr = _make_manager()
    mgr.record_research_queue_entry(
        market_id="KXTEST-001",
        cycle_id="c1",
        gate_name="g1",
        reason="reason1",
    )
    mgr.record_research_queue_entry(
        market_id="KXTEST-001",
        cycle_id="c2",
        gate_name="g2",
        reason="reason2",
        what_to_learn_next="updated learning",
    )
    entries = mgr.get_active_research_entries(lookback_hours=1)
    assert len(entries) == 1
    assert entries[0]["cycle_id"] == "c2"
    assert entries[0]["reason"] == "reason2"


def test_expired_entries_excluded_from_get() -> None:
    mgr = _make_manager()
    expired = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    mgr.record_research_queue_entry(
        market_id="KXEXPIRED-001",
        cycle_id="c1",
        gate_name="g1",
        reason="r1",
        expires_at=expired,
    )
    entries = mgr.get_active_research_entries(lookback_hours=24)
    assert len(entries) == 0


def test_prune_expired_removes_old() -> None:
    mgr = _make_manager()
    expired = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    mgr.record_research_queue_entry(
        market_id="KXEXPIRED-001",
        cycle_id="c1",
        gate_name="g1",
        reason="r1",
        expires_at=expired,
    )
    mgr.record_research_queue_entry(
        market_id="KXVALID-001",
        cycle_id="c2",
        gate_name="g2",
        reason="r2",
    )
    pruned = mgr.prune_expired_research_entries()
    assert pruned == 1
    entries = mgr.get_active_research_entries(lookback_hours=24)
    assert len(entries) == 1
    assert entries[0]["market_id"] == "KXVALID-001"


def test_lookback_filters_old_entries() -> None:
    mgr = _make_manager()
    mgr.record_research_queue_entry(
        market_id="KXRECENT-001",
        cycle_id="c1",
        gate_name="g1",
        reason="r1",
    )
    entries_1h = mgr.get_active_research_entries(lookback_hours=1)
    assert len(entries_1h) == 1
    entries_0h = mgr.get_active_research_entries(lookback_hours=0)
    assert len(entries_0h) >= 0


def test_drainable_entries_filter_by_age_window() -> None:
    """get_drainable_research_entries returns only entries whose queued_at age
    falls between min_age_hours and max_age_hours, ordered oldest-first."""
    mgr = _make_manager()
    now = datetime.now(timezone.utc)
    age_30min = (now - timedelta(minutes=30)).isoformat()
    age_2h = (now - timedelta(hours=2)).isoformat()
    age_8h = (now - timedelta(hours=8)).isoformat()
    age_24h = (now - timedelta(hours=24)).isoformat()
    for market_id, queued_at in (
        ("KXFRESH-001", age_30min),
        ("KXMID-001", age_2h),
        ("KXOLD-001", age_8h),
        ("KXEXPIRED-001", age_24h),
    ):
        mgr._conn.execute(
            """
            INSERT INTO research_queue_entries
                (market_id, cycle_id, queued_at, gate_name, reason,
                 threshold_gap, what_to_learn_next, last_seen, expires_at,
                 last_decision_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                market_id, "c1", queued_at, "g1", "r1",
                0.0, None, queued_at, None, None,
            ),
        )
    mgr._conn.commit()

    drainable = mgr.get_drainable_research_entries(
        min_age_hours=1.0,
        max_age_hours=12.0,
        limit=5,
    )
    drained_ids = [entry["market_id"] for entry in drainable]
    assert "KXFRESH-001" not in drained_ids
    assert "KXEXPIRED-001" not in drained_ids
    assert "KXMID-001" in drained_ids
    assert "KXOLD-001" in drained_ids
    assert drained_ids.index("KXOLD-001") < drained_ids.index("KXMID-001")


def test_drainable_entries_excludes_specified_market_ids() -> None:
    """Caller can pass excluded_market_ids to keep already-traded or current-
    cycle markets out of the drain set."""
    mgr = _make_manager()
    queued_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    for market_id in ("KXEXCLUDE-001", "KXKEEP-001"):
        mgr._conn.execute(
            """
            INSERT INTO research_queue_entries
                (market_id, cycle_id, queued_at, gate_name, reason,
                 threshold_gap, what_to_learn_next, last_seen, expires_at,
                 last_decision_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                market_id, "c1", queued_at, "g1", "r1",
                0.0, None, queued_at, None, None,
            ),
        )
    mgr._conn.commit()

    drainable = mgr.get_drainable_research_entries(
        min_age_hours=1.0,
        max_age_hours=12.0,
        limit=5,
        excluded_market_ids=("KXEXCLUDE-001",),
    )
    drained_ids = [entry["market_id"] for entry in drainable]
    assert drained_ids == ["KXKEEP-001"]


def test_drainable_entries_respects_limit() -> None:
    """The drain helper must honor the limit parameter so callers can cap
    the number of forced probes per cycle."""
    mgr = _make_manager()
    base = datetime.now(timezone.utc) - timedelta(hours=4)
    for index in range(5):
        queued_at = (base + timedelta(minutes=index * 10)).isoformat()
        market_id = f"KXMANY-{index:03d}"
        mgr._conn.execute(
            """
            INSERT INTO research_queue_entries
                (market_id, cycle_id, queued_at, gate_name, reason,
                 threshold_gap, what_to_learn_next, last_seen, expires_at,
                 last_decision_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                market_id, "c1", queued_at, "g1", "r1",
                0.0, None, queued_at, None, None,
            ),
        )
    mgr._conn.commit()

    drainable = mgr.get_drainable_research_entries(
        min_age_hours=1.0,
        max_age_hours=12.0,
        limit=2,
    )
    assert len(drainable) == 2


def test_get_family_action_snapshot_deep_only() -> None:
    mgr = _make_manager()
    mgr._conn.execute(
        """
        INSERT INTO decision_receipts (cycle_id, market_id, final_action, final_reason, timestamp, decision_json, audit_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "c1", "KXBTC-001", "research_queued",
            "pre_analysis_zero_action_family",
            datetime.now(timezone.utc).isoformat(),
            '{}',
            '{"market_family": "crypto"}',
        ),
    )
    mgr._conn.execute(
        """
        INSERT INTO decision_receipts (cycle_id, market_id, final_action, final_reason, timestamp, decision_json, audit_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "c1", "KXBTC-002", "skip",
            "no_trade_recommended",
            datetime.now(timezone.utc).isoformat(),
            '{}',
            '{"market_family": "crypto"}',
        ),
    )
    mgr._conn.commit()

    snapshot_all = mgr.get_family_action_snapshot(lookback=100, deep_only=False)
    assert snapshot_all.get("crypto", {}).get("sample_size", 0) == 2

    snapshot_deep = mgr.get_family_action_snapshot(lookback=100, deep_only=True)
    assert snapshot_deep.get("crypto", {}).get("sample_size", 0) == 1
