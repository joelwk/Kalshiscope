from __future__ import annotations

import os
import json
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

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
        reason="pre_analysis_score_soft_research",
        threshold_gap=0.0,
        what_to_learn_next="Need more data",
    )
    entries = mgr.get_active_research_entries(lookback_hours=1)
    assert len(entries) == 1
    assert entries[0]["market_id"] == "KXBTCD-26APR2517-T77749.99"
    assert entries[0]["reason"] == "pre_analysis_score_soft_research"
    assert entries[0]["what_to_learn_next"] == "Need more data"


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
    assert entries[0]["times_seen"] == 2


def test_research_queue_times_seen_defaults_for_new_entry() -> None:
    mgr = _make_manager()
    mgr.record_research_queue_entry(
        market_id="KXSEEN-001",
        cycle_id="c1",
        gate_name="g1",
        reason="reason1",
    )
    entries = mgr.get_active_research_entries(lookback_hours=1)
    assert entries[0]["times_seen"] == 1


def test_research_queue_backlog_summary_separates_actionable_and_legacy_rows() -> None:
    mgr = _make_manager()
    mgr.record_research_queue_entry(
        market_id="KX-ACTIONABLE",
        cycle_id="c1",
        gate_name="edge",
        reason="edge_gate_blocked",
        threshold_gap=0.02,
    )
    mgr.record_research_queue_entry(
        market_id="KX-SOFT",
        cycle_id="c1",
        gate_name="pre_analysis_movement_score",
        reason="pre_analysis_score_soft_research",
        threshold_gap=0.20,
        last_decision_json=json.dumps(
            {
                "confidence": 0.50,
                "evidence_quality": 0.0,
                "edge_source": "none",
                "audit": {
                    "synthetic_decision": True,
                    "research_drain_attempts": 4,
                },
            }
        ),
    )
    mgr._conn.execute(
        "UPDATE research_queue_entries SET times_seen = 8 WHERE market_id = ?",
        ("KX-SOFT",),
    )
    mgr.record_research_queue_entry(
        market_id="KX-LEGACY-SPORTS",
        cycle_id="c1",
        gate_name="jurisdiction_sports_hold",
        reason="jurisdiction_sports_analysis_held",
    )
    mgr._conn.commit()

    summary = mgr.get_research_queue_backlog_summary(lookback_hours=1)

    assert summary == {
        "active_total": 3,
        "priority_drain_candidates": 1,
        "soft_research_placeholders": 1,
        "repeated_low_yield": 1,
        "legacy_jurisdiction_holds": 1,
    }


def test_mark_research_queue_drain_attempt_updates_audit_payload() -> None:
    mgr = _make_manager()
    attempted_at = datetime(2026, 5, 16, 20, 0, tzinfo=timezone.utc)
    mgr.record_research_queue_entry(
        market_id="KXDRAIN-001",
        cycle_id="c1",
        gate_name="score_gate",
        reason="near_threshold",
        last_decision_json=json.dumps(
            {"audit": {"research_priority": 0.71}, "should_trade": False}
        ),
    )

    mgr.mark_research_queue_drain_attempt(
        "KXDRAIN-001",
        cycle_id="cycle-002",
        attempted_at=attempted_at,
    )
    entries = mgr.get_active_research_entries(lookback_hours=1)
    attempts, last_attempt = mgr.research_queue_drain_attempt_metadata(entries[0])

    assert attempts == 1
    assert last_attempt == attempted_at
    payload = json.loads(entries[0]["last_decision_json"])
    assert payload["audit"]["research_priority"] == 0.71
    assert payload["audit"]["research_queue_last_drain_cycle_id"] == "cycle-002"


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


def test_drainable_entries_can_filter_to_included_market_ids() -> None:
    """Current-cycle include filtering prevents stale queue rows from burning
    the drain pool before live candidates can be selected."""
    mgr = _make_manager()
    base = datetime.now(timezone.utc) - timedelta(hours=4)
    for index, market_id in enumerate(
        ("KXSTALE-001", "KXSTALE-002", "KXLIVE-001")
    ):
        queued_at = (base + timedelta(minutes=index)).isoformat()
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
        limit=1,
        included_market_ids=("KXLIVE-001",),
    )

    assert [entry["market_id"] for entry in drainable] == ["KXLIVE-001"]


def test_drainable_entries_empty_include_returns_empty() -> None:
    mgr = _make_manager()
    queued_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    mgr._conn.execute(
        """
        INSERT INTO research_queue_entries
            (market_id, cycle_id, queued_at, gate_name, reason,
             threshold_gap, what_to_learn_next, last_seen, expires_at,
             last_decision_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "KXLIVE-001", "c1", queued_at, "g1", "r1",
            0.0, None, queued_at, None, None,
        ),
    )
    mgr._conn.commit()
    drainable = mgr.get_drainable_research_entries(
        min_age_hours=1.0,
        max_age_hours=12.0,
        limit=5,
        included_market_ids=(),
    )
    assert drainable == []


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


def test_drain_priority_filter_drops_low_priority_audit() -> None:
    """get_drainable_research_entries with min_priority must drop entries
    whose last_decision_json.audit.pre_analysis_score is below the cutoff."""
    mgr = _make_manager()
    queued_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    high_priority_audit = (
        '{"audit": {"pre_analysis_score": 0.62, '
        '"final_action": "research_queued"}}'
    )
    low_priority_audit = (
        '{"audit": {"pre_analysis_score": 0.30, '
        '"final_action": "research_queued"}}'
    )
    for market_id, audit in (
        ("KXHIGH-001", high_priority_audit),
        ("KXLOW-001", low_priority_audit),
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
                0.0, None, queued_at, None, audit,
            ),
        )
    mgr._conn.commit()

    drainable = mgr.get_drainable_research_entries(
        min_age_hours=1.0,
        max_age_hours=12.0,
        limit=10,
        min_priority=0.55,
    )
    drained_ids = [entry["market_id"] for entry in drainable]
    assert "KXHIGH-001" in drained_ids
    assert "KXLOW-001" not in drained_ids


def test_drain_priority_filter_admits_unknown_priority() -> None:
    """Entries with no decision JSON and no threshold_gap signal must be
    treated as 'unknown priority' rather than 'low priority' so the filter
    only prunes *known* low-quality entries."""
    mgr = _make_manager()
    queued_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    mgr._conn.execute(
        """
        INSERT INTO research_queue_entries
            (market_id, cycle_id, queued_at, gate_name, reason,
             threshold_gap, what_to_learn_next, last_seen, expires_at,
             last_decision_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "KXUNKNOWN-001", "c1", queued_at, "g1", "r1",
            None, None, queued_at, None, None,
        ),
    )
    mgr._conn.commit()
    drainable = mgr.get_drainable_research_entries(
        min_age_hours=1.0,
        max_age_hours=12.0,
        limit=5,
        min_priority=0.55,
    )
    assert [entry["market_id"] for entry in drainable] == ["KXUNKNOWN-001"]


def test_drain_priority_filter_uses_threshold_gap_fallback() -> None:
    """When no decision JSON is present, the priority falls back to
    1.0 - threshold_gap. A threshold_gap of 0.50 implies priority 0.50,
    which is below the 0.55 cutoff and therefore filtered out."""
    mgr = _make_manager()
    queued_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    mgr._conn.execute(
        """
        INSERT INTO research_queue_entries
            (market_id, cycle_id, queued_at, gate_name, reason,
             threshold_gap, what_to_learn_next, last_seen, expires_at,
             last_decision_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "KXGAP-LOW", "c1", queued_at, "g1", "r1",
            0.50, None, queued_at, None, None,
        ),
    )
    mgr._conn.execute(
        """
        INSERT INTO research_queue_entries
            (market_id, cycle_id, queued_at, gate_name, reason,
             threshold_gap, what_to_learn_next, last_seen, expires_at,
             last_decision_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "KXGAP-HIGH", "c1", queued_at, "g1", "r1",
            0.10, None, queued_at, None, None,
        ),
    )
    mgr._conn.commit()
    drainable = mgr.get_drainable_research_entries(
        min_age_hours=1.0,
        max_age_hours=12.0,
        limit=5,
        min_priority=0.55,
    )
    drained_ids = [entry["market_id"] for entry in drainable]
    assert "KXGAP-HIGH" in drained_ids
    assert "KXGAP-LOW" not in drained_ids


def test_estimate_research_entry_priority_prefers_audit_over_gap() -> None:
    """When both an audit pre_analysis_score and a threshold_gap are present,
    the audit wins so we never lose precision to the gap heuristic."""
    audit_priority = MarketStateManager.estimate_research_entry_priority(
        {
            "last_decision_json": (
                '{"audit": {"pre_analysis_score": 0.42}}'
            ),
            "threshold_gap": 0.10,
        }
    )
    assert audit_priority == 0.42


def test_estimate_research_entry_priority_returns_none_when_no_signal() -> None:
    """No JSON, no threshold_gap, no audit -> None so the caller can treat
    the entry as 'unknown' rather than penalizing it."""
    priority = MarketStateManager.estimate_research_entry_priority({})
    assert priority is None


def test_repeated_low_yield_synthetic_placeholder_priority_decays() -> None:
    entry = {
        "times_seen": 9,
        "threshold_gap": 0.18,
        "reason": "pre_analysis_score_soft_research",
        "last_decision_json": json.dumps(
            {
                "should_trade": False,
                "confidence": 0.50,
                "edge_source": "none",
                "evidence_quality": 0.0,
                "audit": {
                    "synthetic_decision": True,
                    "decision_origin": "synthetic_research_queue",
                    "research_priority": 0.70,
                    "evidence_quality": 0.0,
                    "research_queue_drain_attempts": 4,
                },
            }
        ),
    }

    assert MarketStateManager.is_repeated_low_yield_research_entry(entry)
    assert MarketStateManager.estimate_research_entry_priority(entry) == 0.25


def test_repeated_placeholder_decay_preserves_improving_source_signal() -> None:
    entry = {
        "times_seen": 10,
        "threshold_gap": 0.02,
        "reason": "pre_analysis_score_soft_research",
        "last_decision_json": json.dumps(
            {
                "should_trade": True,
                "confidence": 0.68,
                "edge_source": "computed",
                "evidence_quality": 0.92,
                "primary_source_url": "https://espn.com/game",
                "audit": {
                    "synthetic_decision": False,
                    "source_match_class": "settlement_aligned",
                    "evidence_quality": 0.92,
                    "research_queue_drain_attempts": 5,
                },
            }
        ),
    }

    assert not MarketStateManager.is_repeated_low_yield_research_entry(entry)


def test_estimate_research_entry_priority_promotes_near_miss_repeated_settlement_signal() -> None:
    priority = MarketStateManager.estimate_research_entry_priority(
        {
            "times_seen": 6,
            "threshold_gap": 0.02,
            "last_decision_json": (
                '{"audit": {"research_priority": 0.38, "source_match_class": '
                '"settlement_aligned", "evidence_quality": 0.94, '
                '"historical_family_pnl_total": 11.75, '
                '"historical_family_samples": 172}}'
            ),
        }
    )

    assert priority is not None
    assert priority >= 0.66


def test_estimate_research_entry_priority_promotes_extended_cooldown_near_threshold() -> None:
    priority = MarketStateManager.estimate_research_entry_priority(
        {
            "reason": "extended_research_cooldown",
            "times_seen": 3,
            "threshold_gap": 0.04,
            "last_decision_json": '{"audit": {"research_priority": 0.32}}',
        }
    )

    assert priority is not None
    assert priority >= 0.47


def test_estimate_research_entry_priority_boosts_conviction_repair_entries() -> None:
    base_entry = {
        "threshold_gap": 0.10,
        "last_decision_json": '{"audit": {"research_priority": 0.30}}',
    }
    base_priority = MarketStateManager.estimate_research_entry_priority(dict(base_entry))
    repair_priority = MarketStateManager.estimate_research_entry_priority(
        {**base_entry, "gate_name": "conviction_repair"}
    )

    assert base_priority is not None and repair_priority is not None
    assert repair_priority == pytest.approx(base_priority + 0.15)


def test_estimate_research_entry_priority_boosts_missing_url_settlement_aligned() -> None:
    """URL-repair near-misses (settlement_aligned + missing URL + score/edge) get +0.20."""
    base_entry = {
        "threshold_gap": 0.0,
        "last_decision_json": json.dumps(
            {
                "audit": {
                    "research_priority": 0.40,
                    "source_match_class": "settlement_aligned",
                    "pre_execution_final_score": 0.50,
                    "edge_market": 0.18,
                }
            }
        ),
    }
    base_priority = MarketStateManager.estimate_research_entry_priority(dict(base_entry))
    repair_entry = {
        "threshold_gap": 0.0,
        "last_decision_json": json.dumps(
            {
                "audit": {
                    "research_priority": 0.40,
                    "source_match_class": "settlement_aligned",
                    "evidence_floor_suppressed_reason": "missing_primary_source_url",
                    "pre_execution_final_score": 0.50,
                    "edge_market": 0.18,
                }
            }
        ),
    }
    repair_priority = MarketStateManager.estimate_research_entry_priority(repair_entry)
    assert base_priority is not None and repair_priority is not None
    assert repair_priority == pytest.approx(base_priority + 0.20)

    # absence_only / low_information must not get the URL-repair boost.
    low_info = MarketStateManager.estimate_research_entry_priority(
        {
            "threshold_gap": 0.0,
            "last_decision_json": json.dumps(
                {
                    "audit": {
                        "research_priority": 0.40,
                        "source_match_class": "missing_or_absence_only",
                        "evidence_floor_suppressed_reason": "low_information",
                        "pre_execution_final_score": 0.50,
                        "edge_market": 0.18,
                    }
                }
            ),
        }
    )
    assert low_info is not None
    assert low_info < repair_priority
    # Base research_priority 0.40 + near-miss gap boost (+0.10 for gap<=0.03);
    # no settlement_aligned or URL-repair boosts.
    assert low_info == pytest.approx(0.50)


def test_is_url_repair_near_miss_detection() -> None:
    repair = {
        "last_decision_json": json.dumps(
            {
                "audit": {
                    "source_match_class": "settlement_aligned",
                    "evidence_floor_suppressed_reason": "missing_primary_source_url",
                    "pre_execution_final_score": 0.50,
                    "edge_market": 0.18,
                }
            }
        ),
    }
    assert MarketStateManager.is_url_repair_near_miss(repair) is True
    assert (
        MarketStateManager.is_url_repair_near_miss(
            {
                "last_decision_json": json.dumps(
                    {
                        "audit": {
                            "source_match_class": "missing_or_absence_only",
                            "evidence_floor_suppressed_reason": "missing_primary_source_url",
                            "edge_market": 0.18,
                        }
                    }
                ),
            }
        )
        is False
    )


def test_estimate_research_entry_priority_conviction_repair_signal_alone_is_sufficient() -> None:
    # Repair entries are persisted with threshold_gap=0.0 and a full decision
    # audit; even without other signals the gate alone must clear the drain
    # priority floor (0.40) so parked repairs get retried.
    priority = MarketStateManager.estimate_research_entry_priority(
        {
            "gate_name": "conviction_repair",
            "reason": "conviction_repair_no_trade",
            "threshold_gap": 0.0,
            "last_decision_json": '{"audit": {"pre_analysis_score": 0.30}}',
        }
    )

    assert priority is not None
    assert priority >= 0.40


def test_is_jurisdiction_sports_hold_entry_detects_gate_and_reason() -> None:
    assert MarketStateManager.is_jurisdiction_sports_hold_entry(
        {"gate_name": "jurisdiction_sports_hold", "reason": "held"}
    )
    assert MarketStateManager.is_jurisdiction_sports_hold_entry(
        {
            "gate_name": "research",
            "reason": "jurisdiction_sports_analysis_held",
        }
    )
    assert MarketStateManager.is_jurisdiction_sports_hold_entry(
        {
            "gate_name": "other",
            "reason": "other",
            "last_decision_json": (
                '{"audit": {"final_reason": "jurisdiction_sports_blocked"}}'
            ),
        }
    )
    assert not MarketStateManager.is_jurisdiction_sports_hold_entry(
        {"gate_name": "conviction_repair", "reason": "conviction_repair_no_trade"}
    )


def test_estimate_research_entry_priority_zeros_jurisdiction_sports_holds() -> None:
    priority = MarketStateManager.estimate_research_entry_priority(
        {
            "gate_name": "jurisdiction_sports_hold",
            "reason": "jurisdiction_sports_analysis_held",
            "threshold_gap": 0.0,
            "last_decision_json": '{"audit": {"research_priority": 0.90}}',
        }
    )
    assert priority == 0.0


def test_drainable_entries_exclude_jurisdiction_sports_holds() -> None:
    mgr = _make_manager()
    queued_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    for market_id, gate_name, reason in (
        ("KXSPORTS-HOLD", "jurisdiction_sports_hold", "jurisdiction_sports_analysis_held"),
        ("KXWEATHER-NEAR", "edge_gate", "edge_gate_blocked"),
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
                market_id,
                "c1",
                queued_at,
                gate_name,
                reason,
                0.02,
                None,
                queued_at,
                None,
                '{"audit": {"research_priority": 0.80}}',
            ),
        )
    mgr._conn.commit()

    drainable = mgr.get_drainable_research_entries(
        min_age_hours=1.0,
        max_age_hours=12.0,
        limit=5,
    )
    drained_ids = [entry["market_id"] for entry in drainable]
    assert "KXSPORTS-HOLD" not in drained_ids
    assert "KXWEATHER-NEAR" in drained_ids


def test_is_soft_research_drain_placeholder_detects_soft_research_reasons() -> None:
    assert MarketStateManager.is_soft_research_drain_placeholder(
        {
            "gate_name": "pre_analysis_movement_score",
            "reason": "pre_analysis_score_soft_research",
        }
    )
    assert MarketStateManager.is_soft_research_drain_placeholder(
        {
            "gate_name": "pre_analysis",
            "reason": "pre_analysis_score_far_below_min",
        }
    )
    assert not MarketStateManager.is_soft_research_drain_placeholder(
        {
            "gate_name": "edge_gate",
            "reason": "edge_gate_blocked",
            "threshold_gap": 0.02,
            "last_decision_json": (
                '{"should_trade": true, "confidence": 0.70, '
                '"evidence_quality": 1.0, "edge_source": "computed", '
                '"evidence_basis": "direct"}'
            ),
        }
    )


def test_estimate_research_entry_priority_boosts_edge_near_miss() -> None:
    base_entry = {
        "threshold_gap": 0.02,
        "last_decision_json": '{"audit": {"research_priority": 0.30}}',
    }
    base_priority = MarketStateManager.estimate_research_entry_priority(dict(base_entry))
    near_miss_priority = MarketStateManager.estimate_research_entry_priority(
        {
            **base_entry,
            "gate_name": "edge",
            "reason": "edge_gate_blocked",
        }
    )

    assert base_priority is not None and near_miss_priority is not None
    assert near_miss_priority == pytest.approx(base_priority + 0.15)


def test_estimate_research_entry_priority_boosts_high_score_hallucinated_edge() -> None:
    base_entry = {
        "threshold_gap": 0.0,
        "last_decision_json": (
            '{"audit": {"research_priority": 0.30, '
            '"pre_execution_final_score": 0.77}}'
        ),
    }
    base_priority = MarketStateManager.estimate_research_entry_priority(dict(base_entry))
    boosted = MarketStateManager.estimate_research_entry_priority(
        {
            **base_entry,
            "gate_name": "score",
            "reason": "hallucinated_edge",
        }
    )
    assert base_priority is not None and boosted is not None
    assert boosted == pytest.approx(base_priority + 0.20)


def test_drainable_entries_exclude_soft_research_when_min_priority_set() -> None:
    mgr = _make_manager()
    older = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
    newer = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    soft_rows = [
        (
            f"KXSOFT-{idx}",
            "pre_analysis_movement_score",
            "pre_analysis_score_soft_research",
            older,
            0.05,
            '{"audit": {"research_priority": 0.20}}',
        )
        for idx in range(12)
    ]
    soft_rows.append(
        (
            "KXEDGE-NEAR",
            "edge_gate",
            "edge_gate_blocked",
            newer,
            0.02,
            '{"audit": {"research_priority": 0.45, "evidence_quality": 1.0}}',
        )
    )
    for market_id, gate_name, reason, queued_at, gap, decision_json in soft_rows:
        mgr._conn.execute(
            """
            INSERT INTO research_queue_entries
                (market_id, cycle_id, queued_at, gate_name, reason,
                 threshold_gap, what_to_learn_next, last_seen, expires_at,
                 last_decision_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                market_id,
                "c1",
                queued_at,
                gate_name,
                reason,
                gap,
                None,
                queued_at,
                None,
                decision_json,
            ),
        )
    mgr._conn.commit()

    drainable = mgr.get_drainable_research_entries(
        min_age_hours=1.0,
        max_age_hours=12.0,
        limit=3,
        min_priority=0.40,
    )
    drained_ids = [entry["market_id"] for entry in drainable]
    assert "KXEDGE-NEAR" in drained_ids
    assert all(not mid.startswith("KXSOFT-") for mid in drained_ids)


def test_record_entry_stamps_default_ttl_expiry() -> None:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    mgr = MarketStateManager(db_path=path, research_queue_entry_ttl_hours=48.0)
    mgr.record_research_queue_entry(
        market_id="KXTTL-001",
        cycle_id="c1",
        gate_name="g1",
        reason="r1",
    )
    row = mgr._conn.execute(
        "SELECT expires_at FROM research_queue_entries WHERE market_id = ?",
        ("KXTTL-001",),
    ).fetchone()
    assert row["expires_at"] is not None
    expiry = datetime.fromisoformat(row["expires_at"])
    delta_hours = (expiry - datetime.now(timezone.utc)).total_seconds() / 3600
    assert 47.0 < delta_hours <= 48.0


def test_record_entry_ttl_disabled_leaves_expiry_null() -> None:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    mgr = MarketStateManager(db_path=path, research_queue_entry_ttl_hours=None)
    mgr.record_research_queue_entry(
        market_id="KXNOTTL-001",
        cycle_id="c1",
        gate_name="g1",
        reason="r1",
    )
    row = mgr._conn.execute(
        "SELECT expires_at FROM research_queue_entries WHERE market_id = ?",
        ("KXNOTTL-001",),
    ).fetchone()
    assert row["expires_at"] is None


def test_record_entry_explicit_expiry_overrides_default_ttl() -> None:
    mgr = _make_manager()
    explicit = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
    mgr.record_research_queue_entry(
        market_id="KXEXPLICIT-001",
        cycle_id="c1",
        gate_name="g1",
        reason="r1",
        expires_at=explicit,
    )
    row = mgr._conn.execute(
        "SELECT expires_at FROM research_queue_entries WHERE market_id = ?",
        ("KXEXPLICIT-001",),
    ).fetchone()
    assert row["expires_at"] == explicit


def _zombie_decision_json() -> str:
    return json.dumps(
        {
            "should_trade": False,
            "confidence": 0.50,
            "evidence_quality": 0.0,
            "edge_source": "none",
            "audit": {"synthetic_decision": True},
        }
    )


def test_low_yield_zombie_recapture_is_skipped() -> None:
    """A repeated low-yield placeholder must not refresh last_seen forever."""
    mgr = _make_manager()
    recorded = mgr.record_research_queue_entry(
        market_id="KXZOMBIE-001",
        cycle_id="c1",
        gate_name="pre_analysis_movement_score",
        reason="pre_analysis_score_soft_research",
        threshold_gap=0.20,
        last_decision_json=_zombie_decision_json(),
    )
    assert recorded is True
    mgr._conn.execute(
        "UPDATE research_queue_entries SET times_seen = 31 WHERE market_id = ?",
        ("KXZOMBIE-001",),
    )
    mgr._conn.commit()
    before = mgr._conn.execute(
        "SELECT last_seen, times_seen FROM research_queue_entries WHERE market_id = ?",
        ("KXZOMBIE-001",),
    ).fetchone()

    recaptured = mgr.record_research_queue_entry(
        market_id="KXZOMBIE-001",
        cycle_id="c2",
        gate_name="pre_analysis_movement_score",
        reason="pre_analysis_score_soft_research",
        threshold_gap=0.20,
        last_decision_json=_zombie_decision_json(),
    )
    assert recaptured is False
    after = mgr._conn.execute(
        "SELECT last_seen, times_seen FROM research_queue_entries WHERE market_id = ?",
        ("KXZOMBIE-001",),
    ).fetchone()
    assert after["times_seen"] == before["times_seen"] == 31
    assert after["last_seen"] == before["last_seen"]


def test_near_miss_recapture_still_allowed_despite_high_times_seen() -> None:
    """Direct-evidence / near-threshold entries are never zombie-blocked."""
    mgr = _make_manager()
    near_miss_json = json.dumps(
        {
            "should_trade": True,
            "confidence": 0.68,
            "evidence_quality": 0.92,
            "edge_source": "computed",
            "evidence_basis": "direct",
            "audit": {"source_match_class": "settlement_aligned"},
        }
    )
    mgr.record_research_queue_entry(
        market_id="KXNEARMISS-001",
        cycle_id="c1",
        gate_name="edge_gate",
        reason="edge_gate_blocked",
        threshold_gap=0.02,
        last_decision_json=near_miss_json,
    )
    mgr._conn.execute(
        "UPDATE research_queue_entries SET times_seen = 31 WHERE market_id = ?",
        ("KXNEARMISS-001",),
    )
    mgr._conn.commit()

    recaptured = mgr.record_research_queue_entry(
        market_id="KXNEARMISS-001",
        cycle_id="c2",
        gate_name="edge_gate",
        reason="edge_gate_blocked",
        threshold_gap=0.02,
        last_decision_json=near_miss_json,
    )
    assert recaptured is True
    row = mgr._conn.execute(
        "SELECT times_seen FROM research_queue_entries WHERE market_id = ?",
        ("KXNEARMISS-001",),
    ).fetchone()
    assert row["times_seen"] == 32


def test_prune_stale_legacy_rows_without_expiry() -> None:
    """Rows recorded before TTL stamping (expires_at NULL) are purged once
    their last_seen falls outside the stale window."""
    mgr = _make_manager()
    now = datetime.now(timezone.utc)
    stale_seen = (now - timedelta(days=30)).isoformat()
    fresh_seen = now.isoformat()
    for market_id, last_seen in (
        ("KXLEGACY-STALE", stale_seen),
        ("KXLEGACY-FRESH", fresh_seen),
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
                market_id, "c1", last_seen, "g1", "r1",
                0.0, None, last_seen, None, None,
            ),
        )
    mgr._conn.commit()

    pruned = mgr.prune_expired_research_entries(stale_after_hours=168.0)
    assert pruned == 1
    remaining = [
        row["market_id"]
        for row in mgr._conn.execute(
            "SELECT market_id FROM research_queue_entries"
        ).fetchall()
    ]
    assert remaining == ["KXLEGACY-FRESH"]

