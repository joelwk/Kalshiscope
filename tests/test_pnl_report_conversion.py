from __future__ import annotations

import io
import sqlite3
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone

from scripts.pnl_report import _print_conversion_funnel


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
