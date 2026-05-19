"""Read-only performance review for the current session.

Pulls signals straight from `data/market_state.db` (stdlib sqlite3 only) so we
don't depend on the project's broken venv during a live run.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path("data/market_state.db")
SESSION_START_HOUR_UTC = 18  # the four-cycle session began around 18:00 UTC today
TODAY_UTC = datetime.now(timezone.utc).date()


def open_ro(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    return con


def safe_json(raw):
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return None


def section(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def row_dict(r: sqlite3.Row) -> dict:
    return {k: r[k] for k in r.keys()}


def main() -> None:
    con = open_ro(DB_PATH)
    cur = con.cursor()

    section("0. SCHEMA SNAPSHOT")
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    print("tables:", tables)

    for tbl in [
        "exchange_settlements",
        "trade_outcomes",
        "decision_receipts",
        "cycle_receipts",
        "trade_log",
        "markets",
        "analyses",
        "bayesian_state",
    ]:
        if tbl in tables:
            cur.execute(f"SELECT COUNT(*) FROM {tbl}")
            print(f"  {tbl:<22} rows = {cur.fetchone()[0]}")

    # --- 1. Account snapshot from settlements -------------------------------
    section("1. EXCHANGE SETTLEMENTS — LIFETIME PnL")
    if "exchange_settlements" in tables:
        cur.execute(
            """
            SELECT COUNT(*) AS n,
                   SUM(CASE WHEN won=1 THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN won=0 THEN 1 ELSE 0 END) AS losses,
                   ROUND(SUM(pnl_realized), 2) AS pnl,
                   MIN(settled_at) AS first_settle,
                   MAX(settled_at) AS last_settle
            FROM exchange_settlements
            """
        )
        r = cur.fetchone()
        print(f"  total settlements:     {r['n']}")
        print(f"  wins / losses:         {r['wins']} / {r['losses']}")
        print(f"  realized PnL (USD):    {r['pnl']}")
        print(f"  first settlement:      {r['first_settle']}")
        print(f"  last  settlement:      {r['last_settle']}")

        # Last 30 days roll-up.
        cur.execute(
            """
            SELECT
                substr(settled_at, 1, 10) AS day,
                COUNT(*) AS n,
                SUM(CASE WHEN won=1 THEN 1 ELSE 0 END) AS wins,
                ROUND(SUM(pnl_realized), 2) AS pnl
            FROM exchange_settlements
            GROUP BY substr(settled_at, 1, 10)
            ORDER BY day DESC
            LIMIT 14
            """
        )
        print("\n  last 14 settlement days (most recent first):")
        print("    day        n  wins   pnl")
        for r in cur.fetchall():
            print(f"    {r['day']}  {r['n']:>3}  {r['wins']:>4}  {r['pnl']:>7}")

    # --- 2. Cycle receipts: this session funnel -----------------------------
    section("2. CYCLE FUNNEL — RECENT CYCLES")
    if "cycle_receipts" in tables:
        cur.execute("PRAGMA table_info(cycle_receipts)")
        cols = [r[1] for r in cur.fetchall()]
        print("  cycle_receipts columns:", cols)

        # First, dump one full payload so we know the exact field names.
        cur.execute("SELECT id, cycle_id, cycle_number, timestamp, payload_json FROM cycle_receipts ORDER BY id DESC LIMIT 1")
        latest = cur.fetchone()
        if latest is not None:
            print(f"\n  LATEST cycle_receipts row:")
            print(f"    id={latest['id']} cycle_id={latest['cycle_id']} cycle_number={latest['cycle_number']} ts={latest['timestamp']}")
            payload = safe_json(latest["payload_json"]) or {}
            print(f"    payload top-level keys: {sorted(payload.keys())}")
            print(f"    payload (truncated 4000 chars):\n{json.dumps(payload, indent=2)[:4000]}")

        cur.execute(
            "SELECT id, cycle_id, cycle_number, timestamp, payload_json FROM cycle_receipts ORDER BY id DESC LIMIT 14"
        )
        rows = cur.fetchall()
        print(f"\n  showing last {len(rows)} cycles (most recent first)\n")
        header = (
            "id    cycle_no  ts                          | fetched filt  pre  ana  st=1 ord  fill | tokens   $cost   | top tier_breakdown"
        )
        print(header)
        for r in rows:
            payload = safe_json(r["payload_json"]) or {}
            ts = r["timestamp"]
            fetched = payload.get("markets_fetched") or payload.get("fetched") or payload.get("candidates_fetched")
            filt = payload.get("filtered") or payload.get("markets_filtered") or payload.get("candidates_after_filter") or payload.get("after_filter")
            pre = payload.get("pre_scored") or payload.get("pre_scored_count") or payload.get("opportunity_admitted") or payload.get("preanalysis_admitted")
            ana = payload.get("analyzed") or payload.get("decisions_made") or payload.get("analyses_run") or payload.get("analyses_count")
            should = payload.get("should_trade_count") or payload.get("trade_decisions") or payload.get("should_trade")
            ord_attempts = payload.get("order_attempts")
            fills = payload.get("orders_filled") or payload.get("fills") or payload.get("orders_submitted")
            tokens = payload.get("api_tokens_consumed")
            cost = payload.get("api_cost_estimate_usd")
            tiers = payload.get("participation_tier_breakdown") or payload.get("tier_breakdown") or payload.get("rejection_breakdown")
            tier_str = ""
            if isinstance(tiers, dict):
                tier_str = ",".join(f"{k}={v}" for k, v in tiers.items())
            print(
                f"{r['id']:>5}  {str(r['cycle_number']):>8}  {str(ts)[:26]} | {str(fetched):>7} {str(filt):>4} {str(pre):>4} {str(ana):>4} {str(should):>4} {str(ord_attempts):>3}  {str(fills):>4} | {str(tokens):>7} {str(cost):>7}  | {tier_str[:90]}"
            )

        print("\n  rejection_breakdown / tier_breakdown for recent cycles:")
        for r in rows[:8]:
            payload = safe_json(r["payload_json"]) or {}
            print(f"  -- id {r['id']} cycle {r['cycle_number']} ts {r['timestamp']} --")
            for key in (
                "participation_tier_breakdown",
                "tier_breakdown",
                "rejection_breakdown",
                "evidence_basis_breakdown",
                "skip_reason_counts",
                "consecutive_zero_execution_yield_cycles",
                "research_queue_size",
                "cycle_yield_alert",
                "markets_fetched",
                "after_filter",
                "preanalysis_admitted",
                "analyses_count",
                "should_trade_count",
                "order_attempts",
                "orders_submitted",
                "api_tokens_consumed",
                "api_cost_estimate_usd",
            ):
                if key in payload:
                    print(f"    {key}: {json.dumps(payload[key])}")

    # --- 3. Decision receipts in the current session ------------------------
    section("3. DECISION RECEIPTS — TODAY")
    if "decision_receipts" in tables:
        cur.execute("PRAGMA table_info(decision_receipts)")
        dr_cols = [r[1] for r in cur.fetchall()]
        print("  decision_receipts columns:", dr_cols)

        ts_col = None
        for cand in ("created_at", "decided_at", "timestamp"):
            if cand in dr_cols:
                ts_col = cand
                break
        if ts_col is None:
            print("  no timestamp column — falling back to last 400 rows")
            cur.execute("SELECT * FROM decision_receipts ORDER BY id DESC LIMIT 400")
        else:
            today = TODAY_UTC.isoformat()
            cur.execute(
                f"SELECT * FROM decision_receipts WHERE substr({ts_col},1,10) = ? ORDER BY id DESC",
                (today,),
            )
        recs = cur.fetchall()
        print(f"  decisions today: {len(recs)}")

        action_counter: Counter = Counter()
        edge_source_counter: Counter = Counter()
        evidence_basis_counter: Counter = Counter()
        family_counter: Counter = Counter()
        should_trade_count = 0
        should_trade_blocked = []
        sample_decisions = []

        for r in recs:
            decision = safe_json(r["decision_json"]) or {}
            audit = safe_json(r["audit_json"]) or {}
            score = safe_json(r["score_json"]) or {}
            row_identifier = r["rowid"] if "rowid" in r.keys() else r["id"]
            action_counter[r["final_action"]] += 1
            edge_source_counter[(decision.get("edge_source") or "unknown").lower()] += 1
            ev = decision.get("evidence_basis") or audit.get("evidence_basis") or "unknown"
            evidence_basis_counter[str(ev).lower()] += 1
            family = audit.get("market_family") or audit.get("family") or "unknown"
            family_counter[family] += 1
            if decision.get("should_trade"):
                should_trade_count += 1
                if r["final_action"] not in ("order_submitted", "dry_run", "fill", "ordered"):
                    should_trade_blocked.append(
                        {
                            "rowid": row_identifier,
                            "market_id": r["market_id"],
                            "final_action": r["final_action"],
                            "family": family,
                            "edge": decision.get("edge"),
                            "edge_source": decision.get("edge_source"),
                            "confidence": decision.get("confidence"),
                            "implied_prob": decision.get("implied_prob"),
                            "evidence_basis": ev,
                            "rejection_reason": audit.get("rejection_reason")
                            or audit.get("skip_reason")
                            or audit.get("block_reason"),
                            "pre_execution_final_score": audit.get("pre_execution_final_score"),
                            "score_gate_threshold": audit.get("score_gate_threshold"),
                        }
                    )
            if len(sample_decisions) < 25:
                sample_decisions.append(
                    {
                        "rowid": row_identifier,
                        "market": r["market_id"],
                        "final_action": r["final_action"],
                        "family": family,
                        "should_trade": decision.get("should_trade"),
                        "confidence": decision.get("confidence"),
                        "implied_prob": decision.get("implied_prob"),
                        "edge": decision.get("edge"),
                        "edge_source": decision.get("edge_source"),
                        "evidence_basis": ev,
                        "rejection_reason": audit.get("rejection_reason")
                        or audit.get("skip_reason")
                        or audit.get("block_reason"),
                        "pre_score": audit.get("pre_execution_final_score"),
                    }
                )

        print("  final_action mix:", dict(action_counter))
        print("  edge_source mix:", dict(edge_source_counter))
        print("  evidence_basis mix:", dict(evidence_basis_counter))
        print("  family mix:", dict(family_counter.most_common(10)))
        print(f"  should_trade=True today: {should_trade_count}")
        print(f"  should_trade=True but blocked: {len(should_trade_blocked)}")

        print("\n  sample of today's decisions (most recent 25):")
        for s in sample_decisions:
            print(
                f"    [{s['rowid']}] {str(s['market'])[:40]:<40} action={s['final_action']:<22} fam={s['family']:<22} st={s['should_trade']!s:<5} edge={s['edge']!s:<6} es={s['edge_source']!s:<10} conf={s['confidence']!s:<6} pre={s['pre_score']!s:<6} reason={str(s['rejection_reason'])[:30]}"
            )

        if should_trade_blocked:
            print("\n  should_trade=True but blocked (full detail):")
            for s in should_trade_blocked:
                print("   ", json.dumps(s, default=str))
        else:
            print("\n  no should_trade=True trades got blocked today (= the funnel stops earlier).")

    # --- 4. Calibration / win rate by confidence ---------------------------
    section("4. CALIBRATION — WIN RATE BY CONFIDENCE TIER")
    if "trade_outcomes" in tables:
        cur.execute(
            """
            SELECT
                CASE
                    WHEN confidence >= 0.90 THEN '0.90+'
                    WHEN confidence >= 0.80 THEN '0.80-0.89'
                    WHEN confidence >= 0.70 THEN '0.70-0.79'
                    WHEN confidence >= 0.60 THEN '0.60-0.69'
                    ELSE '<0.60'
                END AS tier,
                COUNT(*) AS n,
                SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END) AS wins,
                ROUND(AVG(CASE WHEN won IS NOT NULL THEN won END), 4) AS win_rate,
                ROUND(SUM(pnl_estimate), 2) AS pnl
            FROM trade_outcomes
            WHERE won IS NOT NULL
            GROUP BY tier
            ORDER BY tier DESC
            """
        )
        print("  tier         n   wins   win_rate    pnl_estimate")
        for r in cur.fetchall():
            print(f"  {r['tier']:<10} {r['n']:>4} {r['wins']:>5}   {r['win_rate']!s:<10} {r['pnl']!s:>10}")

        section("4b. WIN RATE BY EDGE BUCKET")
        cur.execute(
            """
            SELECT
                CASE
                    WHEN implied_prob IS NULL OR confidence IS NULL THEN 'na'
                    WHEN (confidence - implied_prob) >= 0.20 THEN '0.20+'
                    WHEN (confidence - implied_prob) >= 0.15 THEN '0.15-0.20'
                    WHEN (confidence - implied_prob) >= 0.10 THEN '0.10-0.15'
                    WHEN (confidence - implied_prob) >= 0.05 THEN '0.05-0.10'
                    ELSE '<0.05'
                END AS edge_bucket,
                COUNT(*) AS n,
                ROUND(AVG(CASE WHEN won IS NOT NULL THEN won END), 4) AS win_rate,
                ROUND(SUM(pnl_estimate), 2) AS pnl
            FROM trade_outcomes
            WHERE won IS NOT NULL
            GROUP BY edge_bucket
            ORDER BY edge_bucket DESC
            """
        )
        print("  edge_bucket    n   win_rate   pnl_estimate")
        for r in cur.fetchall():
            print(f"  {r['edge_bucket']:<12} {r['n']:>4}  {r['win_rate']!s:<10} {r['pnl']!s:>10}")

    # --- 5. Family attribution ---------------------------------------------
    section("5. FAMILY-LEVEL PnL (from decision_receipts joined to outcomes)")
    if "decision_receipts" in tables:
        # Try a join on market_id with trade_outcomes for realized signal.
        try:
            cur.execute(
                """
                SELECT
                    COALESCE(json_extract(d.audit_json, '$.market_family'), 'unknown') AS family,
                    COUNT(DISTINCT d.market_id) AS markets,
                    SUM(CASE WHEN d.final_action='order_submitted' THEN 1 ELSE 0 END) AS submitted,
                    ROUND(AVG(CAST(json_extract(d.decision_json, '$.confidence') AS REAL)), 4) AS avg_conf,
                    ROUND(AVG(CAST(json_extract(d.audit_json, '$.pre_execution_final_score') AS REAL)), 4) AS avg_pre_score
                FROM decision_receipts d
                GROUP BY family
                ORDER BY markets DESC
                """
            )
            print("  family                       markets submitted avg_conf avg_pre_score")
            for r in cur.fetchall():
                print(
                    f"  {str(r['family']):<28} {str(r['markets']):>6} {str(r['submitted']):>9} {str(r['avg_conf']):>8} {str(r['avg_pre_score']):>13}"
                )
        except sqlite3.Error as exc:
            print(f"  family rollup failed: {exc}")

        section("5b. FAMILY-LEVEL REALIZED PnL (settlement join)")
        try:
            cur.execute(
                """
                WITH famtag AS (
                    SELECT DISTINCT
                        market_id,
                        COALESCE(json_extract(audit_json, '$.market_family'), 'unknown') AS family
                    FROM decision_receipts
                )
                SELECT
                    f.family AS family,
                    COUNT(*) AS n_settle,
                    SUM(CASE WHEN s.won=1 THEN 1 ELSE 0 END) AS wins,
                    ROUND(SUM(s.pnl_realized), 2) AS pnl
                FROM exchange_settlements s
                LEFT JOIN famtag f ON f.market_id = s.market_id
                GROUP BY f.family
                ORDER BY pnl ASC
                """
            )
            print("  family                       n_settle  wins      pnl")
            for r in cur.fetchall():
                print(f"  {str(r['family']):<28} {str(r['n_settle']):>8} {str(r['wins']):>5} {str(r['pnl']):>9}")
        except sqlite3.Error as exc:
            print(f"  family x settlement join failed: {exc}")

    # --- 6. API cost / cycle ------------------------------------------------
    section("6. API COST & THROUGHPUT — RECENT 50 CYCLES")
    if "cycle_receipts" in tables:
        cur.execute(
            "SELECT * FROM cycle_receipts ORDER BY rowid DESC LIMIT 50"
        )
        rows = cur.fetchall()
        total_tokens = 0
        total_cost = 0.0
        total_orders = 0
        total_fills = 0
        total_decisions = 0
        for r in rows:
            payload = safe_json(r["payload_json"]) or {}
            total_tokens += payload.get("api_tokens_consumed") or 0
            total_cost += payload.get("api_cost_estimate_usd") or 0.0
            total_orders += payload.get("order_attempts") or 0
            total_fills += payload.get("orders_filled") or payload.get("orders_submitted") or 0
            total_decisions += payload.get("decisions_made") or payload.get("analyzed") or 0
        n = len(rows) or 1
        print(f"  cycles examined:        {n}")
        print(f"  total tokens:           {total_tokens}")
        print(f"  total API cost (USD):   {total_cost:.4f}")
        print(f"  total order attempts:   {total_orders}")
        print(f"  total orders submitted/filled: {total_fills}")
        print(f"  total decisions made:   {total_decisions}")
        print(f"  avg cost / cycle:       {total_cost / n:.4f}")
        if total_orders:
            print(f"  avg cost / order_attempt: {total_cost / total_orders:.4f}")
        else:
            print("  avg cost / order_attempt: N/A (zero attempts)")

    # --- 7. Last successful order -----------------------------------------
    section("7. LAST EXECUTED TRADE (trade_log + settlements)")
    if "trade_log" in tables:
        cur.execute("PRAGMA table_info(trade_log)")
        tl_cols = [r[1] for r in cur.fetchall()]
        print("  trade_log columns:", tl_cols)
        cur.execute("SELECT * FROM trade_log ORDER BY rowid DESC LIMIT 5")
        for r in cur.fetchall():
            print("   ", row_dict(r))
    if "exchange_settlements" in tables:
        print("\n  most recent 5 settlements:")
        cur.execute(
            "SELECT settlement_id, market_id, won, pnl_realized, contracts, avg_price, settled_at FROM exchange_settlements ORDER BY settled_at DESC LIMIT 5"
        )
        for r in cur.fetchall():
            print("   ", row_dict(r))

    # --- 8. Score gate effectiveness --------------------------------------
    section("8. SCORE GATE — should_trade=True with skip action")
    if "decision_receipts" in tables:
        try:
            cur.execute(
                """
                SELECT
                    COALESCE(json_extract(audit_json, '$.market_family'), 'unknown') AS family,
                    COUNT(*) AS blocked,
                    ROUND(AVG(CAST(json_extract(decision_json, '$.confidence') AS REAL)), 4) AS avg_conf,
                    ROUND(AVG(CAST(json_extract(audit_json, '$.pre_execution_final_score') AS REAL)), 4) AS avg_score
                FROM decision_receipts
                WHERE COALESCE(json_extract(decision_json, '$.should_trade'), 0) = 1
                  AND final_action NOT IN ('order_submitted', 'dry_run')
                GROUP BY family
                ORDER BY blocked DESC
                """
            )
            print("  family                       blocked  avg_conf  avg_pre_score")
            for r in cur.fetchall():
                print(
                    f"  {str(r['family']):<28} {str(r['blocked']):>7}  {str(r['avg_conf']):>7}  {str(r['avg_score']):>13}"
                )
        except sqlite3.Error as exc:
            print(f"  query failed: {exc}")


if __name__ == "__main__":
    main()
