from __future__ import annotations

import io
import sqlite3
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone

from scripts.pnl_report import (
    _confidence_tier_label,
    _print_confidence_tier_breakdown,
    _print_conversion_funnel,
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
