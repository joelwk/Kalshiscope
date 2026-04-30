"""Read-only diagnostic: what prefix evidence is the historical gate using?

For markets recently rejected with reason `pre_analysis_historical_prefix_pnl_block`
(or related), shows the 12-char prefix it evaluated, and the historical settled
sample size + win rate + PnL that triggered the block. Read-only.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_DB_PATH = "data/market_state.db"
WINDOW_DAYS = 7
PREFIX_LEN = 12
LOOKBACK_DAYS_FOR_GATE = 30


def _safe_json(raw):
    if raw is None:
        return None
    try:
        return json.loads(str(raw))
    except (TypeError, ValueError):
        return None


def main() -> None:
    cutoff_window = (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).isoformat()
    cutoff_lookback = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS_FOR_GATE)).isoformat()

    uri = f"file:{DEFAULT_DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT market_id, audit_json, final_reason
        FROM decision_receipts
        WHERE timestamp >= ?
          AND final_action = 'research_queued'
          AND final_reason LIKE 'pre_analysis_historical%'
        """,
        (cutoff_window,),
    ).fetchall()

    print(f"recent (last {WINDOW_DAYS}d) historical-prefix-gate research_queued rows: {len(rows)}")

    prefix_counts: Counter = Counter()
    sample_size_obs: Counter = Counter()
    pnl_per_prefix: dict[str, float] = defaultdict(float)
    prefix_metadata: dict[str, dict] = {}
    reasons: Counter = Counter()

    for row in rows:
        audit = _safe_json(row["audit_json"]) or {}
        prefix = (
            audit.get("historical_gate_market_prefix")
            or audit.get("historical_gate_prefix")
            or (str(row["market_id"] or "")[:PREFIX_LEN]).upper()
        )
        prefix_counts[prefix] += 1
        reason = str(row["final_reason"] or "unknown")
        reasons[reason] += 1
        sample_size = audit.get("historical_gate_prefix_sample_size")
        pnl = audit.get("historical_gate_prefix_pnl_total")
        wr = audit.get("historical_gate_prefix_win_rate")
        wlb = audit.get("historical_gate_prefix_wilson_lb")
        if sample_size is not None:
            try:
                sample_size_obs[int(sample_size)] += 1
            except (TypeError, ValueError):
                pass
        if pnl is not None:
            try:
                pnl_per_prefix[prefix] = float(pnl)
            except (TypeError, ValueError):
                pass
        prefix_metadata[prefix] = {
            "sample_size": sample_size,
            "pnl_total": pnl,
            "win_rate": wr,
            "wilson_lb": wlb,
        }

    print()
    print("Reason breakdown:")
    for reason, n in reasons.most_common():
        print(f"  {reason:<60} n={n}")

    print()
    print(f"Top blocked prefixes (last {WINDOW_DAYS}d window):")
    print(
        f"{'prefix':<14} {'block_count':>11} {'gate_n':>7} "
        f"{'gate_wr':>8} {'gate_pnl':>10} {'gate_wlb':>9}"
    )
    for prefix, n in prefix_counts.most_common(40):
        meta = prefix_metadata.get(prefix, {})
        sample_size = meta.get("sample_size")
        pnl = meta.get("pnl_total")
        wr = meta.get("win_rate")
        wlb = meta.get("wilson_lb")
        sample_str = f"{int(sample_size)}" if sample_size is not None else "?"
        wr_str = f"{float(wr):.2%}" if wr is not None else "?"
        pnl_str = f"{float(pnl):+.2f}" if pnl is not None else "?"
        wlb_str = f"{float(wlb):.4f}" if wlb is not None else "?"
        print(
            f"{prefix:<14} {n:>11} {sample_str:>7} "
            f"{wr_str:>8} {pnl_str:>10} {wlb_str:>9}"
        )

    print()
    print("Distribution of gate sample_size (smaller = weaker evidence):")
    for size, n in sorted(sample_size_obs.items()):
        print(f"  n_samples={size:>3}  rows_blocked={n}")
    if not sample_size_obs:
        print("  (no sample_size metadata recorded)")

    rows_recent = conn.execute(
        """
        SELECT
          COUNT(*) AS cycles,
          SUM(json_extract(payload_json,'$.api_tokens_consumed')) AS tokens,
          SUM(json_extract(payload_json,'$.api_cost_estimate_usd')) AS cost_usd,
          SUM(json_extract(payload_json,'$.order_attempts')) AS orders
        FROM cycle_receipts
        WHERE timestamp >= ?
        """,
        (cutoff_window,),
    ).fetchone()
    print()
    print(f"7-day cycle cost / activity:")
    print(f"  cycles:              {int(rows_recent['cycles'] or 0)}")
    print(f"  api_tokens_consumed: {int(rows_recent['tokens'] or 0)}")
    print(f"  api_cost_usd:        {float(rows_recent['cost_usd'] or 0.0):.2f}")
    print(f"  order_attempts:      {int(rows_recent['orders'] or 0)}")
    orders_recent = int(rows_recent["orders"] or 0)
    if orders_recent > 0:
        cost_per_order = float(rows_recent["cost_usd"] or 0.0) / orders_recent
        print(f"  cost_per_order:      {cost_per_order:.4f}")

    settled_recent = conn.execute(
        """
        SELECT
          COUNT(*) AS n,
          SUM(CASE WHEN won=1 THEN 1 ELSE 0 END) AS wins,
          SUM(COALESCE(pnl_realized,0.0)) AS pnl
        FROM exchange_settlements
        WHERE settled_at >= ?
        """,
        (cutoff_window,),
    ).fetchone()
    print()
    print(f"7-day settlements (exchange_settlements):")
    print(f"  settled trades: {int(settled_recent['n'] or 0)}")
    print(f"  wins:           {int(settled_recent['wins'] or 0)}")
    print(f"  realized pnl:   {float(settled_recent['pnl'] or 0.0):+.2f}")

    conn.close()


if __name__ == "__main__":
    main()
