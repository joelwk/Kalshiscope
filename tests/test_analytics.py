from __future__ import annotations

import io
import sqlite3
from contextlib import redirect_stdout

from analytics import run


def test_analytics_reports_cycle_api_and_score_gate_metrics(tmp_path) -> None:
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE trade_outcomes (
                confidence REAL,
                implied_prob REAL,
                won INTEGER,
                resolution_state TEXT,
                amount_usdc REAL,
                pnl_estimate REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE markets (
                last_terminal_outcome TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE decision_receipts (
                audit_json TEXT,
                decision_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE cycle_receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                payload_json TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO trade_outcomes (confidence, implied_prob, won, resolution_state, amount_usdc, pnl_estimate)
            VALUES (0.72, 0.60, 1, 'resolved_valid', 2.0, 1.0)
            """
        )
        conn.execute(
            """
            INSERT INTO markets (last_terminal_outcome)
            VALUES ('no_trade_recommended')
            """
        )
        conn.execute(
            """
            INSERT INTO decision_receipts (audit_json, decision_json)
            VALUES (
                '{"market_family":"crypto","evidence_basis_class":"proxy","final_action":"skip","final_reason":"score_gate_blocked","score_liquidity_penalty":0.04,"score_weather_penalty":0.00,"score_proxy_evidence_penalty":0.08,"score_repeated_penalty":0.02,"score_generic_bin_penalty":0.01,"score_ambiguous_resolution_penalty":0.00}',
                '{"edge_source":"fallback","should_trade":true}'
            )
            """
        )
        conn.execute(
            """
            INSERT INTO decision_receipts (audit_json, decision_json)
            VALUES (
                '{"market_family":"weather","evidence_basis_class":"direct","final_action":"order_attempt","final_reason":"dry_run","score_liquidity_penalty":0.01,"score_weather_penalty":0.02,"score_proxy_evidence_penalty":0.00,"score_repeated_penalty":0.00,"score_generic_bin_penalty":0.00,"score_ambiguous_resolution_penalty":0.00}',
                '{"edge_source":"computed","should_trade":true}'
            )
            """
        )
        conn.execute(
            """
            INSERT INTO cycle_receipts (payload_json)
            VALUES ('{"api_tokens_consumed":1200,"api_cost_estimate_usd":0.03,"order_attempts":1,"decisions_made":4,"rejection_breakdown":{"score_gate_blocked":2},"evidence_basis_breakdown":{"proxy":3,"direct":1}}')
            """
        )
        conn.commit()
    finally:
        conn.close()

    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        run(str(db_path))
    output = output_buffer.getvalue()

    assert "Cycle API/score-gate summary" in output
    assert "score_gate_block_rate=50.00%" in output
    assert "api_cost_per_trade_attempt_usd=0.030000" in output
    assert "evidence_basis_breakdown" in output
    assert "Should-trade block rate" in output
    assert "should_trade_block_rate=50.00%" in output
    assert "Average score penalties (decision receipts)" in output
    assert "Should-trade block rate by family" in output
