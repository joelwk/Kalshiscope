"""Recent cost trend report (read-only)."""
import json
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_profiles import family_from_text

conn = sqlite3.connect('data/market_state.db')
conn.row_factory = sqlite3.Row

cutoff_3d = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
cutoff_7d = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

print("=== Cost trend (last 3 days vs last 7 days) ===")
for label, cutoff in [("last 3d", cutoff_3d), ("last 7d", cutoff_7d), ("lifetime", "1900-01-01")]:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS cycles,
            ROUND(SUM(CAST(json_extract(payload_json,'$.api_cost_estimate_usd') AS REAL)),4) AS total_cost,
            SUM(CAST(json_extract(payload_json,'$.order_attempts') AS INTEGER)) AS orders,
            SUM(CAST(json_extract(payload_json,'$.decisions_made') AS INTEGER)) AS decisions
        FROM cycle_receipts
        WHERE timestamp >= ?
        """,
        (cutoff,),
    ).fetchone()
    cycles = int(row['cycles'] or 0)
    total = float(row['total_cost'] or 0)
    orders = int(row['orders'] or 0)
    decisions = int(row['decisions'] or 0)
    print(
        f"  {label:<10} cycles={cycles} total_cost=${total:.2f} "
        f"orders={orders} decisions={decisions} "
        f"per_cycle=${(total/cycles if cycles else 0):.4f} "
        f"per_order_attempt=${(total/orders if orders else 0):.4f}"
    )

print()
print("=== Recent settlement losses by family (last 7 days) ===")
cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
rows = conn.execute(
    """
    SELECT market_id, won, pnl_realized, settled_at
    FROM exchange_settlements
    WHERE won IS NOT NULL AND settled_at >= ?
    ORDER BY settled_at DESC
    """,
    (cutoff,),
).fetchall()


def family(mid: str) -> str:
    return family_from_text(mid)


totals = {}
for r in rows:
    fam = family(r['market_id'] or '')
    totals.setdefault(fam, {"n": 0, "wins": 0, "pnl": 0.0})
    totals[fam]["n"] += 1
    totals[fam]["wins"] += 1 if int(r['won'] or 0) == 1 else 0
    totals[fam]["pnl"] += float(r['pnl_realized'] or 0)

for fam, t in sorted(totals.items(), key=lambda x: x[1]['pnl']):
    n = t['n']
    wr = (t['wins'] / n) if n else 0
    print(f"  {fam:<10} n={n:>3} wr={wr:.4f} pnl=${t['pnl']:+.2f}")

print()
print("=== order_attempt confidence histogram (resolved trades only) ===")
rows = conn.execute(
    """
    SELECT
        CAST(json_extract(dr.decision_json,'$.confidence') AS REAL) AS conf,
        es.won,
        es.pnl_realized,
        dr.market_id
    FROM decision_receipts dr
    JOIN exchange_settlements es ON es.market_id = dr.market_id AND es.won IS NOT NULL
    WHERE dr.final_action = 'order_attempt'
    """
).fetchall()

buckets = {"<0.62": [0,0,0.0], "0.62-0.69": [0,0,0.0], "0.70-0.79": [0,0,0.0], "0.80-0.89": [0,0,0.0], "0.90+": [0,0,0.0]}
for r in rows:
    c = float(r['conf'] or 0)
    if c < 0.62:
        b = "<0.62"
    elif c < 0.70:
        b = "0.62-0.69"
    elif c < 0.80:
        b = "0.70-0.79"
    elif c < 0.90:
        b = "0.80-0.89"
    else:
        b = "0.90+"
    buckets[b][0] += 1
    if int(r['won'] or 0) == 1:
        buckets[b][1] += 1
    buckets[b][2] += float(r['pnl_realized'] or 0)

print(f"{'tier':<12} {'n':>4} {'wins':>5} {'wr':>7} {'pnl':>8}")
for b, (n, w, p) in buckets.items():
    wr = (w / n) if n else 0
    print(f"{b:<12} {n:>4} {w:>5} {wr:>7.4f} {p:>+8.2f}")

print()
print("=== Family confidence calibration (>=0.85 confidence trades that lost) ===")
rows = conn.execute(
    """
    SELECT market_id, confidence, implied_prob, won, pnl_estimate
    FROM trade_outcomes
    WHERE won IS NOT NULL AND confidence >= 0.85
    """
).fetchall()

print(f"{'market':<42} {'conf':>5} {'impl':>5} {'won':>3} {'pnl':>8} {'fam':<10}")
loss_count = 0
win_count = 0
for r in rows:
    fam = family(r['market_id'] or '')
    if int(r['won'] or 0) == 0:
        loss_count += 1
    else:
        win_count += 1
    print(
        f"{(r['market_id'] or '')[:42]:<42} "
        f"{float(r['confidence'] or 0):>5.2f} "
        f"{float(r['implied_prob'] or 0):>5.2f} "
        f"{int(r['won'] or 0):>3} "
        f"{float(r['pnl_estimate'] or 0):>+8.2f} "
        f"{fam:<10}"
    )
print(f"\nTotal high-conf (>=0.85) trades: wins={win_count} losses={loss_count} "
      f"win_rate={(win_count/(win_count+loss_count) if (win_count+loss_count) else 0):.4f}")

print()
print("=== Submarket breakout by event-prefix (top 25 by sample size) ===")
rows = conn.execute(
    """
    SELECT market_id, confidence, won, pnl_estimate
    FROM trade_outcomes
    WHERE won IS NOT NULL
    """
).fetchall()
sub_totals = {}
for r in rows:
    mid = r['market_id'] or ''
    parts = mid.split('-')
    sub = parts[0] if parts else mid
    sub_totals.setdefault(sub, {"n": 0, "wins": 0, "pnl": 0.0})
    sub_totals[sub]["n"] += 1
    sub_totals[sub]["wins"] += 1 if int(r['won'] or 0) == 1 else 0
    sub_totals[sub]["pnl"] += float(r['pnl_estimate'] or 0)

ranked = sorted(sub_totals.items(), key=lambda x: x[1]['n'], reverse=True)[:25]
for sub, t in sorted(ranked, key=lambda x: x[1]['pnl']):
    n = t['n']
    wr = (t['wins'] / n) if n else 0
    print(f"  {sub:<24} n={n:>3} wr={wr:.4f} pnl=${t['pnl']:+.2f}")
