from __future__ import annotations

import io
import json
import sqlite3
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone

from scripts.pnl_report import (
    _confidence_tier_label,
    _print_confidence_tier_breakdown,
    _print_conversion_funnel,
)
from scripts.inspect_participation_quality import (
    _prepare_reporting_views,
    section_cycle_funnel,
    section_legacy_receipts,
)


def test_conversion_funnel_section_renders(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    try:
        now = datetime.now(timezone.utc)
        recent = (now - timedelta(days=1)).isoformat()
        conn.execute(
            """
            CREATE TABLE decision_receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id TEXT,
                market_id TEXT NOT NULL,
                final_action TEXT,
                final_reason TEXT,
                timestamp TEXT NOT NULL,
                decision_json TEXT NOT NULL,
                order_json TEXT,
                audit_json TEXT,
                score_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE exchange_settlements (
                settlement_id TEXT PRIMARY KEY,
                market_id TEXT NOT NULL,
                predicted_outcome TEXT,
                winning_outcome TEXT,
                won INTEGER,
                pnl_realized REAL,
                contracts INTEGER,
                avg_price REAL,
                settled_at TEXT,
                raw_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE markets (
                id TEXT PRIMARY KEY,
                question TEXT,
                category TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO markets (id, question, category) VALUES ('SPORT-1', 'game', 'sports')"
        )
        conn.execute(
            """
            INSERT INTO decision_receipts (
                market_id, timestamp, decision_json, audit_json, final_action, final_reason
            )
            VALUES (
                'SPORT-1',
                ?,
                '{"should_trade": true}',
                '{"market_family":"sports","synthetic_decision": false}',
                'order_attempt',
                'dry_run'
            )
            """,
            (recent,),
        )
        conn.execute(
            """
            INSERT INTO decision_receipts (
                market_id, timestamp, decision_json, audit_json, final_action, final_reason
            )
            VALUES (
                'SPORT-HELD',
                ?,
                '{"should_trade": false, "prompt_tokens": null}',
                '{"market_family":"sports","synthetic_decision": false}',
                'research_queued',
                'jurisdiction_sports_analysis_held'
            )
            """,
            (recent,),
        )
        conn.execute(
            """
            INSERT INTO exchange_settlements (
                settlement_id, market_id, won, pnl_realized, contracts, avg_price,
                settled_at, raw_json
            )
            VALUES ('s1', 'SPORT-1', 1, 2.5, 5, 0.5, ?, '{}')
            """,
            (recent,),
        )
        conn.commit()
    finally:
        conn.close()

    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        _print_conversion_funnel(str(db_path), lookback_days=7)
    output = output_buffer.getvalue()
    assert "Analyzed -> Executed Conversion" in output
    assert "sports" in output
    sports_line = next(line for line in output.splitlines() if line.startswith("sports"))
    assert int(sports_line.split()[1]) == 1


def test_participation_report_separates_legacy_jurisdiction_receipts() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    recent = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        CREATE TABLE decision_receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            final_reason TEXT,
            timestamp TEXT NOT NULL
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO decision_receipts (final_reason, timestamp)
        VALUES (?, ?)
        """,
        [
            ("edge_gate_blocked", recent),
            ("jurisdiction_sports_analysis_held", recent),
        ],
    )

    _prepare_reporting_views(conn)

    actionable_count = conn.execute(
        "SELECT COUNT(*) FROM decision_receipts_report"
    ).fetchone()[0]
    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        section_legacy_receipts(conn, window_days=7)

    assert actionable_count == 1
    assert "recent=1 all_time=1" in output_buffer.getvalue()


def test_cycle_funnel_reports_actionable_queue_without_legacy_holds() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE cycle_receipts (
            timestamp TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    payload = {
        "analyzed": 1,
        "execution_candidates": 0,
        "research_queue_size": 5,
        "rejection_breakdown": {
            "jurisdiction_sports_analysis_held": 4,
            "edge_gate_blocked": 1,
        },
    }
    conn.execute(
        "INSERT INTO cycle_receipts (timestamp, payload_json) VALUES (?, ?)",
        (datetime.now(timezone.utc).isoformat(), json.dumps(payload)),
    )

    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        section_cycle_funnel(conn, window_days=7)
    output = output_buffer.getvalue()

    actionable_line = next(
        line
        for line in output.splitlines()
        if line.startswith("research_queue_actionable_size")
    )
    assert float(actionable_line.split()[1]) == 1.0
    assert "legacy jurisdiction holds excluded from rejection mix: 4" in output
    rejection_section = output.split("rejection_breakdown totals", 1)[1]
    assert "jurisdiction_sports_analysis_held" not in rejection_section


def test_confidence_tier_label_bands_are_contiguous() -> None:
    assert _confidence_tier_label(0.95) == "0.70+"
    assert _confidence_tier_label(0.70) == "0.70+"
    assert _confidence_tier_label(0.69) == "0.62-0.69"
    assert _confidence_tier_label(0.61) == "0.55-0.61"
    assert _confidence_tier_label(0.54) == "0.50-0.54"
    assert _confidence_tier_label(0.49) == "<0.50"
    assert _confidence_tier_label(0.0) == "<0.50"


def test_confidence_tier_section_renders(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE trade_outcomes (
                market_id TEXT PRIMARY KEY,
                confidence REAL,
                won INTEGER,
                pnl_estimate REAL
            )
            """
        )
        conn.executemany(
            "INSERT INTO trade_outcomes (market_id, confidence, won, pnl_estimate) VALUES (?, ?, ?, ?)",
            [
                ("HI-WIN", 0.75, 1, 3.0),
                ("HI-LOSS", 0.72, 0, -5.0),
                ("MID-WIN", 0.58, 1, 4.0),
                ("LOW", 0.45, 0, -1.0),
                # NULL confidence must be excluded from every tier.
                ("NULL-CONF", None, 1, 9.0),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        _print_confidence_tier_breakdown(str(db_path))
    output = output_buffer.getvalue()
    assert "Win Rate by Confidence Tier" in output
    assert "0.70+" in output
    assert "0.55-0.61" in output
    # The NULL-confidence row's PnL (9.0) must not appear in any tier total.
    assert "9.00" not in output


def test_confidence_tier_section_handles_missing_table(tmp_path) -> None:
    db_path = tmp_path / "empty.db"
    conn = sqlite3.connect(db_path)
    conn.close()
    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        _print_confidence_tier_breakdown(str(db_path))
    assert "No trade_outcomes table found." in output_buffer.getvalue()
