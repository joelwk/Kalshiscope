"""Read-only participation-quality and funnel analysis.

Computes the participation/funnel metrics requested by the performance-review
workflow without modifying the database.

Run with:
    poetry run python scripts/inspect_participation_quality.py

Optional flags:
    --db PATH         Override the database path.
    --window-days N   Recent window in days (default: 7).
    --logs PATH...    Override log files for Grok timeout scan.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_DB_PATH = "data/market_state.db"
DEFAULT_WINDOW_DAYS = 7
DEFAULT_LOG_FILES = ("logs/predictbot.log", "logs/predictbot.log.1")
SECTION_RULE = "=" * 78
SUBSECTION_RULE = "-" * 78


def _open_readonly(db_path: str) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _fmt_pct(numerator: float, denominator: float) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{(numerator / denominator):.2%}"


def _fmt_signed(value: float) -> str:
    return f"{value:+,.2f}"


def _print_header(title: str) -> None:
    print()
    print(SECTION_RULE)
    print(title)
    print(SECTION_RULE)


def _print_subheader(title: str) -> None:
    print()
    print(title)
    print(SUBSECTION_RULE)


def _cutoff_iso(window_days: int) -> str:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(window_days)))
    return cutoff.isoformat()


def _coerce_dt(text: object) -> datetime | None:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        match = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.(\d+)(.*)", raw)
        if not match:
            return None
        base, frac, suffix = match.groups()
        try:
            parsed = datetime.fromisoformat(
                f"{base}.{frac[:6].ljust(6, '0')}{suffix}"
            )
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _market_family_from_id(market_id: str) -> str:
    normalized = (market_id or "").upper()
    if "BTC" in normalized or "ETH" in normalized:
        return "crypto"
    if normalized.startswith(("KXNASDAQ100U-", "KXINXU-")):
        return "index"
    if "MENTION" in normalized or "LASTWORDCOUNT" in normalized:
        return "speech"
    if any(
        token in normalized
        for token in (
            "GOLD", "SILVER", "WTI", "NATGAS", "COPPER", "CORN",
            "SOY", "WHEAT", "AAA",
        )
    ):
        return "commodity"
    if any(token in normalized for token in ("LOWT", "HIGHT", "TEMPNYC")):
        return "weather"
    return "generic"


def _safe_json_loads(raw: object) -> dict[str, Any] | None:
    if not raw:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def section_decision_outcome_mix(
    conn: sqlite3.Connection,
    *,
    window_days: int,
) -> None:
    if not _table_exists(conn, "decision_receipts"):
        print("decision_receipts table missing; skipping.")
        return
    cutoff = _cutoff_iso(window_days)

    _print_header(f"DECISION OUTCOME MIX (last {window_days}d and all-time)")

    rows_recent = conn.execute(
        """
        SELECT COALESCE(final_action, 'unknown') AS final_action, COUNT(*) AS n
        FROM decision_receipts
        WHERE timestamp >= ?
        GROUP BY final_action
        ORDER BY n DESC
        """,
        (cutoff,),
    ).fetchall()
    rows_all = conn.execute(
        """
        SELECT COALESCE(final_action, 'unknown') AS final_action, COUNT(*) AS n
        FROM decision_receipts
        GROUP BY final_action
        ORDER BY n DESC
        """,
    ).fetchall()

    total_recent = sum(int(r["n"]) for r in rows_recent)
    total_all = sum(int(r["n"]) for r in rows_all)

    print(f"recent total decisions: {total_recent}")
    print(f"all-time total decisions: {total_all}")

    print()
    print(f"{'final_action':<32} {'recent_n':>10} {'recent_%':>10} {'all_n':>10} {'all_%':>10}")
    by_action: dict[str, dict[str, int]] = defaultdict(lambda: {"recent": 0, "all": 0})
    for row in rows_recent:
        by_action[str(row["final_action"])]["recent"] = int(row["n"])
    for row in rows_all:
        by_action[str(row["final_action"])]["all"] = int(row["n"])
    for action in sorted(by_action, key=lambda a: by_action[a]["all"], reverse=True):
        recent_n = by_action[action]["recent"]
        all_n = by_action[action]["all"]
        print(
            f"{action:<32} {recent_n:>10} {_fmt_pct(recent_n, total_recent):>10} "
            f"{all_n:>10} {_fmt_pct(all_n, total_all):>10}"
        )


def section_blocked_conviction(
    conn: sqlite3.Connection,
    *,
    window_days: int,
) -> None:
    if not _table_exists(conn, "decision_receipts"):
        return
    cutoff = _cutoff_iso(window_days)

    _print_header(f"BLOCKED CONVICTION ANALYSIS (last {window_days}d)")

    summary = conn.execute(
        """
        SELECT
          COUNT(*) AS total_should_trade,
          SUM(CASE WHEN COALESCE(final_action,'') NOT IN ('order_submitted','dry_run','order_attempt')
                   THEN 1 ELSE 0 END) AS blocked,
          SUM(CASE WHEN COALESCE(final_action,'') = 'order_submitted' THEN 1 ELSE 0 END) AS submitted,
          SUM(CASE WHEN COALESCE(final_action,'') = 'order_attempt' THEN 1 ELSE 0 END) AS attempted,
          SUM(CASE WHEN COALESCE(final_action,'') = 'dry_run' THEN 1 ELSE 0 END) AS dry_run
        FROM decision_receipts
        WHERE timestamp >= ?
          AND COALESCE(json_extract(decision_json,'$.should_trade'), 0) = 1
        """,
        (cutoff,),
    ).fetchone()

    total = int(summary["total_should_trade"] or 0)
    blocked = int(summary["blocked"] or 0)
    submitted = int(summary["submitted"] or 0)
    attempted = int(summary["attempted"] or 0)
    dry_run = int(summary["dry_run"] or 0)

    print(f"should_trade=true decisions: {total}")
    print(f"  -> order_submitted:          {submitted}")
    print(f"  -> order_attempt (no submit): {attempted}")
    print(f"  -> dry_run:                  {dry_run}")
    print(f"  -> blocked (anything else):  {blocked}")
    if total > 0:
        print(f"  blocked rate:               {_fmt_pct(blocked, total)}")

    rows = conn.execute(
        """
        SELECT
          COALESCE(final_action,'unknown') AS final_action,
          COALESCE(final_reason,'unknown') AS final_reason,
          COUNT(*) AS n
        FROM decision_receipts
        WHERE timestamp >= ?
          AND COALESCE(json_extract(decision_json,'$.should_trade'), 0) = 1
          AND COALESCE(final_action,'') NOT IN ('order_submitted','dry_run','order_attempt')
        GROUP BY final_action, final_reason
        ORDER BY n DESC
        LIMIT 30
        """,
        (cutoff,),
    ).fetchall()
    if rows:
        _print_subheader("Reason breakdown for blocked conviction")
        print(f"{'final_action':<22} {'final_reason':<42} {'n':>6}")
        for row in rows:
            print(
                f"{row['final_action']:<22} {row['final_reason']:<42} "
                f"{int(row['n']):>6}"
            )

    family_rows = conn.execute(
        """
        SELECT
          COALESCE(json_extract(audit_json,'$.market_family'),'unknown') AS family,
          COUNT(*) AS n_total,
          SUM(CASE WHEN COALESCE(final_action,'') NOT IN ('order_submitted','dry_run','order_attempt')
                   THEN 1 ELSE 0 END) AS n_blocked,
          SUM(CASE WHEN COALESCE(final_action,'') = 'order_submitted' THEN 1 ELSE 0 END) AS n_submitted
        FROM decision_receipts
        WHERE timestamp >= ?
          AND COALESCE(json_extract(decision_json,'$.should_trade'), 0) = 1
        GROUP BY family
        ORDER BY n_total DESC
        """,
        (cutoff,),
    ).fetchall()
    if family_rows:
        _print_subheader("Blocked conviction by family")
        print(
            f"{'family':<14} {'n_total':>8} {'submitted':>10} {'blocked':>8} "
            f"{'block_rate':>10}"
        )
        for row in family_rows:
            n_total = int(row["n_total"] or 0)
            n_blocked = int(row["n_blocked"] or 0)
            n_submitted = int(row["n_submitted"] or 0)
            print(
                f"{str(row['family']):<14} {n_total:>8} {n_submitted:>10} "
                f"{n_blocked:>8} {_fmt_pct(n_blocked, n_total):>10}"
            )


def section_naming_mismatch(
    conn: sqlite3.Connection,
    *,
    window_days: int,
) -> None:
    if not _table_exists(conn, "decision_receipts"):
        return
    cutoff = _cutoff_iso(window_days)

    _print_header("PRE-ANALYSIS HARD-REJECT NAMING MISMATCH")

    queries = [
        (
            f"recent (last {window_days}d): pre_analysis_hard_reject=true with final_action=research_queued",
            cutoff,
        ),
        (
            "all-time: pre_analysis_hard_reject=true with final_action=research_queued",
            None,
        ),
    ]
    for label, cutoff_value in queries:
        if cutoff_value is not None:
            row = conn.execute(
                """
                SELECT
                  COUNT(*) AS n,
                  SUM(CASE WHEN COALESCE(json_extract(audit_json,'$.legacy_pre_analysis_hard_reject'),
                                         json_extract(audit_json,'$.pre_analysis_hard_reject'),0)=1
                            AND COALESCE(final_action,'') = 'research_queued'
                           THEN 1 ELSE 0 END) AS mismatch,
                  SUM(CASE WHEN COALESCE(json_extract(audit_json,'$.legacy_pre_analysis_hard_reject'),
                                         json_extract(audit_json,'$.pre_analysis_hard_reject'),0)=1
                            AND COALESCE(final_action,'') = 'skip'
                           THEN 1 ELSE 0 END) AS skip_aligned,
                  SUM(CASE WHEN COALESCE(json_extract(audit_json,'$.legacy_pre_analysis_hard_reject'),
                                         json_extract(audit_json,'$.pre_analysis_hard_reject'),0)=1
                           THEN 1 ELSE 0 END) AS hard_reject_total
                FROM decision_receipts
                WHERE timestamp >= ?
                """,
                (cutoff_value,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT
                  COUNT(*) AS n,
                  SUM(CASE WHEN COALESCE(json_extract(audit_json,'$.legacy_pre_analysis_hard_reject'),
                                         json_extract(audit_json,'$.pre_analysis_hard_reject'),0)=1
                            AND COALESCE(final_action,'') = 'research_queued'
                           THEN 1 ELSE 0 END) AS mismatch,
                  SUM(CASE WHEN COALESCE(json_extract(audit_json,'$.legacy_pre_analysis_hard_reject'),
                                         json_extract(audit_json,'$.pre_analysis_hard_reject'),0)=1
                            AND COALESCE(final_action,'') = 'skip'
                           THEN 1 ELSE 0 END) AS skip_aligned,
                  SUM(CASE WHEN COALESCE(json_extract(audit_json,'$.legacy_pre_analysis_hard_reject'),
                                         json_extract(audit_json,'$.pre_analysis_hard_reject'),0)=1
                           THEN 1 ELSE 0 END) AS hard_reject_total
                FROM decision_receipts
                """,
            ).fetchone()
        total = int(row["n"] or 0)
        mismatch = int(row["mismatch"] or 0)
        skip_aligned = int(row["skip_aligned"] or 0)
        hard_reject_total = int(row["hard_reject_total"] or 0)
        print(f"{label}")
        print(f"  decisions seen:                         {total}")
        print(f"  pre_analysis_hard_reject=true:          {hard_reject_total}")
        print(
            f"    -> with final_action=research_queued: {mismatch} "
            f"({_fmt_pct(mismatch, hard_reject_total)})"
        )
        print(
            f"    -> with final_action=skip:            {skip_aligned} "
            f"({_fmt_pct(skip_aligned, hard_reject_total)})"
        )
        print()

    rows = conn.execute(
        """
        SELECT
          COALESCE(json_extract(audit_json,'$.pre_analysis_hard_reject_reason'),'unknown') AS reason,
          COUNT(*) AS n
        FROM decision_receipts
        WHERE COALESCE(json_extract(audit_json,'$.legacy_pre_analysis_hard_reject'),
                       json_extract(audit_json,'$.pre_analysis_hard_reject'),0)=1
          AND COALESCE(final_action,'') = 'research_queued'
        GROUP BY reason
        ORDER BY n DESC
        LIMIT 20
        """,
    ).fetchall()
    if rows:
        _print_subheader("Top reasons among hard-reject -> research_queued (all-time)")
        print(f"{'reason':<48} {'n':>8}")
        for row in rows:
            print(f"{str(row['reason']):<48} {int(row['n']):>8}")


def section_decision_field_distribution(
    conn: sqlite3.Connection,
    *,
    window_days: int,
) -> None:
    if not _table_exists(conn, "decision_receipts"):
        return
    cutoff = _cutoff_iso(window_days)

    _print_header(
        f"DECISION FIELD DISTRIBUTIONS (last {window_days}d)"
    )

    rows = conn.execute(
        """
        SELECT decision_json
        FROM decision_receipts
        WHERE timestamp >= ?
        """,
        (cutoff,),
    ).fetchall()
    total = len(rows)
    print(f"sample size: {total}")

    eq_buckets = Counter()
    eq_exact_zero = 0
    eq_missing = 0
    edge_source_counter = Counter()
    evidence_basis_counter = Counter()
    confidence_buckets = Counter()
    confidence_exact_half = 0
    confidence_missing = 0
    abstain_count = 0
    should_trade_true = 0
    should_trade_false = 0

    for row in rows:
        decision = _safe_json_loads(row["decision_json"]) or {}
        eq_raw = decision.get("evidence_quality")
        if eq_raw is None:
            eq_missing += 1
        else:
            try:
                eq = float(eq_raw)
            except (TypeError, ValueError):
                eq = None
            if eq is None:
                eq_missing += 1
            else:
                if abs(eq) < 1e-9:
                    eq_exact_zero += 1
                left = max(0.0, min(1.0, eq))
                bucket_lo = int(left * 10) / 10
                eq_buckets[f"{bucket_lo:.1f}-{bucket_lo + 0.1:.1f}"] += 1

        edge_source = str(decision.get("edge_source") or "missing").strip().lower() or "missing"
        edge_source_counter[edge_source] += 1

        evidence_basis = (
            str(decision.get("evidence_basis") or "missing").strip().lower() or "missing"
        )
        evidence_basis_counter[evidence_basis] += 1

        conf_raw = decision.get("confidence")
        if conf_raw is None:
            confidence_missing += 1
        else:
            try:
                conf = float(conf_raw)
            except (TypeError, ValueError):
                conf = None
            if conf is None:
                confidence_missing += 1
            else:
                if abs(conf - 0.5) < 1e-9:
                    confidence_exact_half += 1
                left = max(0.0, min(1.0, conf))
                bucket_lo = int(left * 10) / 10
                confidence_buckets[f"{bucket_lo:.1f}-{bucket_lo + 0.1:.1f}"] += 1

        if decision.get("abstain") is True:
            abstain_count += 1
        st = decision.get("should_trade")
        if st is True:
            should_trade_true += 1
        elif st is False:
            should_trade_false += 1

    _print_subheader("evidence_quality distribution")
    print(f"  exact 0.0:   {eq_exact_zero:>6}  ({_fmt_pct(eq_exact_zero, total)})")
    print(f"  missing:     {eq_missing:>6}  ({_fmt_pct(eq_missing, total)})")
    for bucket in sorted(eq_buckets):
        n = eq_buckets[bucket]
        print(f"  {bucket}:   {n:>6}  ({_fmt_pct(n, total)})")

    _print_subheader("edge_source distribution (top 12)")
    for source, n in edge_source_counter.most_common(12):
        print(f"  {source:<24} n={n:>6} ({_fmt_pct(n, total)})")

    _print_subheader("evidence_basis distribution")
    for basis, n in evidence_basis_counter.most_common(12):
        print(f"  {basis:<20} n={n:>6} ({_fmt_pct(n, total)})")

    _print_subheader("confidence distribution")
    print(f"  exact 0.50: {confidence_exact_half:>6} ({_fmt_pct(confidence_exact_half, total)})")
    print(f"  missing:    {confidence_missing:>6} ({_fmt_pct(confidence_missing, total)})")
    for bucket in sorted(confidence_buckets):
        n = confidence_buckets[bucket]
        print(f"  {bucket}:   {n:>6}  ({_fmt_pct(n, total)})")

    _print_subheader("should_trade / abstain summary")
    print(f"  should_trade=true:  {should_trade_true:>6} ({_fmt_pct(should_trade_true, total)})")
    print(f"  should_trade=false: {should_trade_false:>6} ({_fmt_pct(should_trade_false, total)})")
    print(f"  abstain=true:       {abstain_count:>6} ({_fmt_pct(abstain_count, total)})")


def section_per_family_prefix(
    conn: sqlite3.Connection,
    *,
    window_days: int,
    prefix_len: int = 12,
) -> None:
    if not _table_exists(conn, "decision_receipts"):
        return
    cutoff = _cutoff_iso(window_days)

    _print_header(f"PARTICIPATION BY FAMILY (decision_receipts, last {window_days}d)")

    rows = conn.execute(
        """
        SELECT
          COALESCE(json_extract(audit_json,'$.market_family'),'unknown') AS family,
          COALESCE(final_action,'unknown') AS final_action,
          CAST(json_extract(audit_json,'$.pre_execution_final_score') AS REAL) AS score,
          market_id
        FROM decision_receipts
        WHERE timestamp >= ?
        """,
        (cutoff,),
    ).fetchall()

    family_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0, "submitted": 0, "skip": 0, "research_queued": 0,
            "score_sum": 0.0, "score_n": 0,
        }
    )
    for row in rows:
        fam = str(row["family"] or "unknown")
        action = str(row["final_action"] or "unknown")
        st = family_stats[fam]
        st["n"] += 1
        if action == "order_submitted":
            st["submitted"] += 1
        elif action == "skip":
            st["skip"] += 1
        elif action == "research_queued":
            st["research_queued"] += 1
        score = row["score"]
        if score is not None:
            try:
                st["score_sum"] += float(score)
                st["score_n"] += 1
            except (TypeError, ValueError):
                pass

    print(
        f"{'family':<14} {'n':>8} {'submitted':>10} {'skip':>8} "
        f"{'research_q':>10} {'avg_score':>10} {'submit_rate':>11}"
    )
    for fam in sorted(family_stats, key=lambda f: family_stats[f]["n"], reverse=True):
        st = family_stats[fam]
        avg_score = (st["score_sum"] / st["score_n"]) if st["score_n"] > 0 else 0.0
        print(
            f"{fam:<14} {st['n']:>8} {st['submitted']:>10} {st['skip']:>8} "
            f"{st['research_queued']:>10} {avg_score:>10.4f} "
            f"{_fmt_pct(st['submitted'], st['n']):>11}"
        )

    if not _table_exists(conn, "trade_outcomes"):
        return

    _print_header(
        f"EXECUTED + SETTLED P&L BY FAMILY (trade_outcomes, last {window_days}d)"
    )
    rows = conn.execute(
        """
        SELECT
          t.market_id AS market_id,
          t.confidence AS confidence,
          t.won AS won,
          t.pnl_estimate AS pnl_estimate,
          t.amount_usdc AS amount_usdc,
          t.resolved_at AS resolved_at,
          t.last_updated AS last_updated,
          COALESCE(m.question,'') AS question,
          COALESCE(m.category,'') AS category
        FROM trade_outcomes t
        LEFT JOIN markets m ON m.id = t.market_id
        WHERE COALESCE(t.last_updated, t.resolved_at, '') >= ?
        """,
        (cutoff,),
    ).fetchall()
    by_family: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "n": 0.0, "wins": 0.0, "settled": 0.0, "pnl": 0.0, "deployed": 0.0,
        }
    )
    for row in rows:
        fam = _market_family_from_id(str(row["market_id"] or ""))
        bucket = by_family[fam]
        bucket["n"] += 1
        won = row["won"]
        if won is not None:
            bucket["settled"] += 1
            if int(won) == 1:
                bucket["wins"] += 1
            bucket["pnl"] += float(row["pnl_estimate"] or 0.0)
        bucket["deployed"] += float(row["amount_usdc"] or 0.0)

    print(
        f"{'family':<14} {'executed_n':>11} {'settled_n':>10} {'wins':>6} "
        f"{'win_rate':>9} {'pnl':>10} {'deployed':>10}"
    )
    for fam in sorted(by_family, key=lambda f: by_family[f]["pnl"]):
        s = by_family[fam]
        n = int(s["n"])
        settled = int(s["settled"])
        wins = int(s["wins"])
        wr = (wins / settled) if settled > 0 else 0.0
        print(
            f"{fam:<14} {n:>11} {settled:>10} {wins:>6} {wr:>9.2%} "
            f"{s['pnl']:>+10.2f} {s['deployed']:>10.2f}"
        )

    _print_header(
        f"TICKER PREFIX (len={prefix_len}) PERFORMANCE (last {window_days}d, settled only)"
    )
    rows = conn.execute(
        """
        SELECT
          SUBSTR(UPPER(COALESCE(market_id,'')), 1, ?) AS prefix,
          COUNT(*) AS n,
          SUM(CASE WHEN won=1 THEN 1 ELSE 0 END) AS wins,
          SUM(COALESCE(pnl_estimate,0.0)) AS pnl_total
        FROM trade_outcomes
        WHERE won IS NOT NULL
          AND COALESCE(resolved_at,last_updated,'') >= ?
          AND COALESCE(market_id,'') <> ''
        GROUP BY prefix
        HAVING n >= 1
        ORDER BY pnl_total ASC
        """,
        (prefix_len, cutoff),
    ).fetchall()
    if not rows:
        print("No settled outcomes in window.")
        return

    print(
        f"{'prefix':<14} {'n':>5} {'wins':>5} {'win_rate':>9} "
        f"{'pnl_total':>10} {'wilson_lb':>10} "
        f"{'shrunk_pnl_pt':>14} {'gate_block':>10}"
    )
    for row in rows:
        n = int(row["n"] or 0)
        wins = int(row["wins"] or 0)
        pnl_total = float(row["pnl_total"] or 0.0)
        wr = (wins / n) if n > 0 else 0.0
        wlb = _wilson_lower_bound(wins, n)
        shrunk = _bayesian_shrunk_pnl(pnl_total, n)
        block = _historical_prefix_gate_predicate(n, wlb, pnl_total, shrunk)
        print(
            f"{str(row['prefix']):<14} {n:>5} {wins:>5} {wr:>9.2%} "
            f"{pnl_total:>+10.2f} {wlb:>10.4f} "
            f"{shrunk:>14.4f} {block:>10}"
        )


def _wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p_hat = wins / n
    denominator = 1.0 + z * z / n
    center = p_hat + z * z / (2.0 * n)
    spread = (
        z * ((p_hat * (1.0 - p_hat) + z * z / (4.0 * n)) / n) ** 0.5
    )
    return max(0.0, (center - spread) / denominator)


def _bayesian_shrunk_pnl(
    pnl: float,
    n: int,
    prior_pnl_per_trade: float = 0.0,
    prior_strength: float = 10.0,
) -> float:
    if n <= 0:
        return prior_pnl_per_trade
    observed = pnl / n
    weight = n / (n + prior_strength)
    return weight * observed + (1.0 - weight) * prior_pnl_per_trade


def _historical_prefix_gate_predicate(
    n: int,
    wlb: float,
    pnl_total: float,
    shrunk_pnl_per_trade: float = 0.0,
    *,
    pnl_cutoff: float = -3.0,
    win_rate_cutoff: float = 0.40,
    hard_block_min_samples: int = 10,
    soft_min_samples: int = 3,
    shrunk_pnl_cutoff: float = -0.50,
) -> str:
    if n >= hard_block_min_samples and wlb <= win_rate_cutoff and pnl_total <= pnl_cutoff:
        return "HARD"
    if n >= soft_min_samples and wlb <= win_rate_cutoff and shrunk_pnl_per_trade <= shrunk_pnl_cutoff:
        return "SOFT"
    return "-"


def section_cycle_funnel(
    conn: sqlite3.Connection,
    *,
    window_days: int,
) -> None:
    if not _table_exists(conn, "cycle_receipts"):
        return
    cutoff = _cutoff_iso(window_days)

    _print_header(f"CYCLE FUNNEL (last {window_days}d)")

    rows = conn.execute(
        """
        SELECT timestamp, payload_json
        FROM cycle_receipts
        WHERE timestamp >= ?
        ORDER BY timestamp ASC
        """,
        (cutoff,),
    ).fetchall()
    if not rows:
        rows = conn.execute(
            "SELECT timestamp, payload_json FROM cycle_receipts ORDER BY timestamp ASC LIMIT 200",
        ).fetchall()
        print("(no cycles inside window — falling back to most recent 200 cycles)")

    fields = [
        "fetched", "filtered", "analyzed", "decisions_made",
        "execution_candidates", "research_queue_size",
        "order_attempts", "api_tokens_consumed", "api_cost_estimate_usd",
    ]
    series: dict[str, list[float]] = {field: [] for field in fields}
    rejection_total = Counter()
    evidence_total = Counter()
    cycles_analyzed_pos_exec_zero_with_research_pos = 0
    cycles_analyzed_pos = 0
    cycles_total = 0

    for row in rows:
        payload = _safe_json_loads(row["payload_json"]) or {}
        cycles_total += 1

        funnel = (
            payload.get("funnel_stage_counts")
            or {}
        )
        for field in fields:
            value = (
                funnel.get(field)
                if field in funnel
                else payload.get(field)
            )
            try:
                series[field].append(float(value if value is not None else 0.0))
            except (TypeError, ValueError):
                pass

        rejection = payload.get("rejection_breakdown")
        if isinstance(rejection, dict):
            for key, val in rejection.items():
                try:
                    rejection_total[str(key)] += int(val)
                except (TypeError, ValueError):
                    continue

        evidence = payload.get("evidence_basis_breakdown")
        if isinstance(evidence, dict):
            for key, val in evidence.items():
                try:
                    evidence_total[str(key)] += int(val)
                except (TypeError, ValueError):
                    continue

        analyzed = (
            funnel.get("analyzed")
            if isinstance(funnel, dict) and funnel.get("analyzed") is not None
            else payload.get("analyzed", 0)
        )
        execution_candidates = (
            payload.get("execution_candidates", 0)
        )
        research_queue_size = (
            payload.get("research_queue_size", 0)
        )
        try:
            analyzed_int = int(float(analyzed or 0))
            exec_int = int(float(execution_candidates or 0))
            rq_int = int(float(research_queue_size or 0))
        except (TypeError, ValueError):
            continue
        if analyzed_int > 0:
            cycles_analyzed_pos += 1
            if exec_int == 0 and rq_int > 0:
                cycles_analyzed_pos_exec_zero_with_research_pos += 1

    print(f"cycles in sample: {cycles_total}")
    print()
    print(f"{'field':<28} {'mean':>10} {'median':>10} {'p10':>8} {'p90':>8} {'min':>8} {'max':>8}")
    for field in fields:
        values = series[field]
        if not values:
            continue
        ordered = sorted(values)

        def _percentile(p: float) -> float:
            if not ordered:
                return 0.0
            idx = max(0, min(len(ordered) - 1, int(p * (len(ordered) - 1))))
            return ordered[idx]

        mean = sum(values) / len(values)
        median = _percentile(0.5)
        p10 = _percentile(0.1)
        p90 = _percentile(0.9)
        print(
            f"{field:<28} {mean:>10.2f} {median:>10.2f} {p10:>8.2f} {p90:>8.2f} "
            f"{min(values):>8.2f} {max(values):>8.2f}"
        )

    print()
    print("Funnel pathology check:")
    print(f"  cycles with analyzed>0:                      {cycles_analyzed_pos}")
    print(
        "  cycles with analyzed>0 AND execution_candidates=0 AND research_queue_size>0: "
        f"{cycles_analyzed_pos_exec_zero_with_research_pos} "
        f"({_fmt_pct(cycles_analyzed_pos_exec_zero_with_research_pos, cycles_analyzed_pos)})"
    )

    if rejection_total:
        _print_subheader("rejection_breakdown totals (top 20)")
        total_rejections = sum(rejection_total.values())
        for key, n in rejection_total.most_common(20):
            print(f"  {key:<40} n={n:>6} ({_fmt_pct(n, total_rejections)})")

    if evidence_total:
        _print_subheader("evidence_basis_breakdown totals")
        total_evidence = sum(evidence_total.values())
        for key, n in evidence_total.most_common():
            print(f"  {key:<24} n={n:>6} ({_fmt_pct(n, total_evidence)})")


_GROK_TIMEOUT_PATTERNS = (
    re.compile(r"Grok stream exceeded", re.IGNORECASE),
    re.compile(r"Grok stream timed out", re.IGNORECASE),
    re.compile(r"Deep market analysis failed", re.IGNORECASE),
    re.compile(r"grok_stream_timeout", re.IGNORECASE),
    re.compile(r"analysis_failure_after_retries", re.IGNORECASE),
)


def section_grok_failures(log_paths: Iterable[str]) -> None:
    _print_header("GROK TIMEOUT / FAILURE SCAN (log files)")

    counters: dict[str, Counter] = {}
    line_counts: dict[str, int] = {}
    deep_total = Counter()

    for path_str in log_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"  (missing: {path})")
            continue
        ctr: Counter = Counter()
        line_counts[str(path)] = 0
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    line_counts[str(path)] += 1
                    for pattern in _GROK_TIMEOUT_PATTERNS:
                        if pattern.search(line):
                            ctr[pattern.pattern] += 1
                            deep_total[pattern.pattern] += 1
                            break
        except OSError as exc:
            print(f"  (read error on {path}: {exc})")
            continue
        counters[str(path)] = ctr

    print(f"{'log_file':<40} {'lines':>10}")
    for path_str, n_lines in line_counts.items():
        print(f"  {path_str:<38} {n_lines:>10}")

    print()
    print("Per-pattern hits:")
    for path_str, ctr in counters.items():
        print(f"  {path_str}")
        if not ctr:
            print("    (no matches)")
            continue
        for pattern, n in ctr.most_common():
            print(f"    {pattern:<40} n={n}")

    print()
    print("Aggregate:")
    if not deep_total:
        print("  no timeout/failure markers found")
    else:
        for pattern, n in deep_total.most_common():
            print(f"  {pattern:<40} n={n}")


def section_research_queued_settlement_review(
    conn: sqlite3.Connection,
    *,
    window_days: int,
) -> None:
    if not _table_exists(conn, "decision_receipts"):
        return
    cutoff = _cutoff_iso(window_days)

    _print_header(
        f"SETTLEMENT-ANCHORED REVIEW OF research_queued (last {window_days}d)"
    )

    rows = conn.execute(
        """
        SELECT
          dr.market_id,
          dr.timestamp,
          dr.decision_json,
          dr.audit_json,
          es.won AS exch_won,
          es.winning_outcome AS exch_winning_outcome,
          es.pnl_realized AS exch_pnl_realized,
          es.contracts AS exch_contracts,
          es.avg_price AS exch_avg_price,
          tox.won AS internal_won,
          tox.resolved_winning_outcome AS internal_winning,
          tox.implied_prob AS internal_implied_prob
        FROM decision_receipts dr
        LEFT JOIN exchange_settlements es ON es.market_id = dr.market_id
        LEFT JOIN trade_outcomes tox ON tox.market_id = dr.market_id
        WHERE dr.timestamp >= ?
          AND COALESCE(dr.final_action,'') = 'research_queued'
        """,
        (cutoff,),
    ).fetchall()

    total = len(rows)
    print(f"research_queued decisions in window: {total}")
    if total == 0:
        return

    settled = 0
    settled_yes = 0
    settled_no = 0
    proposed_with_side = 0
    profitable_if_traded = 0
    losing_if_traded = 0
    coverage_total = 0
    coverage_with_implied = 0
    pnl_simulated_total = 0.0
    samples_for_pnl = []

    for row in rows:
        decision = _safe_json_loads(row["decision_json"]) or {}
        proposed_outcome = (
            str(decision.get("outcome") or "").strip().upper()
            if decision.get("outcome") is not None
            else None
        )
        proposed_implied = decision.get("implied_prob_external")
        if proposed_implied is None:
            proposed_implied = decision.get("my_prob")
        try:
            proposed_implied_f = (
                float(proposed_implied) if proposed_implied is not None else None
            )
        except (TypeError, ValueError):
            proposed_implied_f = None

        if proposed_outcome in {"YES", "NO"}:
            proposed_with_side += 1

        winning = (
            str(row["exch_winning_outcome"] or row["internal_winning"] or "").strip().upper()
            or None
        )
        won_int = row["exch_won"]
        if won_int is None:
            won_int = row["internal_won"]
        if winning in {"YES", "NO"} or won_int is not None:
            settled += 1
            if winning == "YES":
                settled_yes += 1
            elif winning == "NO":
                settled_no += 1

            if proposed_outcome in {"YES", "NO"}:
                coverage_total += 1
                if proposed_implied_f is not None:
                    coverage_with_implied += 1
                    payoff = 1.0 if proposed_outcome == winning else 0.0
                    pnl_per_share = payoff - proposed_implied_f
                    pnl_simulated_total += pnl_per_share
                    samples_for_pnl.append(pnl_per_share)
                    if proposed_outcome == winning:
                        profitable_if_traded += 1
                    else:
                        losing_if_traded += 1

    print(
        f"  with proposed side (decision.outcome in YES/NO): {proposed_with_side}"
    )
    print(f"  settled: {settled} ({_fmt_pct(settled, total)})")
    if settled > 0:
        print(
            f"    settled YES: {settled_yes} ({_fmt_pct(settled_yes, settled)}); "
            f"settled NO: {settled_no} ({_fmt_pct(settled_no, settled)})"
        )

    if coverage_total > 0:
        print(
            f"  proposed-side coverage among settled: {coverage_total} "
            f"({_fmt_pct(coverage_total, settled)})"
        )
    if coverage_with_implied > 0:
        print(
            f"  with proposed implied price for sim: {coverage_with_implied}"
        )
        print(
            f"    profitable if traded: {profitable_if_traded} "
            f"({_fmt_pct(profitable_if_traded, coverage_with_implied)})"
        )
        print(
            f"    losing if traded:     {losing_if_traded} "
            f"({_fmt_pct(losing_if_traded, coverage_with_implied)})"
        )
        avg_pnl_per_share = pnl_simulated_total / coverage_with_implied
        print(
            f"    avg simulated pnl/share at proposed price: {avg_pnl_per_share:+.4f}"
        )
        print(
            f"    total simulated pnl across N shares-per-decision (1 share each): "
            f"{pnl_simulated_total:+.2f}"
        )

    _print_subheader("Recent research_queued (sample, up to 25)")
    rows = conn.execute(
        """
        SELECT
          dr.market_id,
          dr.timestamp,
          json_extract(dr.decision_json,'$.should_trade') AS should_trade,
          json_extract(dr.decision_json,'$.outcome') AS outcome,
          json_extract(dr.decision_json,'$.confidence') AS confidence,
          json_extract(dr.decision_json,'$.evidence_quality') AS evidence_quality,
          json_extract(dr.decision_json,'$.implied_prob_external') AS implied_prob_external,
          json_extract(dr.audit_json,'$.market_family') AS family,
          dr.final_reason AS final_reason,
          es.winning_outcome AS exch_winning_outcome,
          es.pnl_realized AS exch_pnl_realized
        FROM decision_receipts dr
        LEFT JOIN exchange_settlements es ON es.market_id = dr.market_id
        WHERE dr.timestamp >= ?
          AND COALESCE(dr.final_action,'') = 'research_queued'
        ORDER BY dr.timestamp DESC
        LIMIT 25
        """,
        (cutoff,),
    ).fetchall()
    if rows:
        print(
            f"{'market_id':<36} {'fam':<10} {'st':>3} {'side':<5} "
            f"{'conf':>5} {'eq':>5} {'imp':>5} {'won':<4} {'reason':<28}"
        )
        for row in rows:
            mid = (str(row["market_id"]) or "")[:36]
            fam = str(row["family"] or "")[:10]
            st = "T" if row["should_trade"] in (1, True, "1", "true") else "F"
            side = (str(row["outcome"]) or "")[:5]
            try:
                conf = float(row["confidence"] or 0.0)
                conf_str = f"{conf:.2f}"
            except Exception:
                conf_str = "?"
            try:
                eq = float(row["evidence_quality"] or 0.0)
                eq_str = f"{eq:.2f}"
            except Exception:
                eq_str = "?"
            try:
                imp = float(row["implied_prob_external"] or 0.0)
                imp_str = f"{imp:.2f}"
            except Exception:
                imp_str = "?"
            won_label = (str(row["exch_winning_outcome"]) or "")[:4]
            reason = (str(row["final_reason"]) or "")[:28]
            print(
                f"{mid:<36} {fam:<10} {st:>3} {side:<5} "
                f"{conf_str:>5} {eq_str:>5} {imp_str:>5} {won_label:<4} {reason:<28}"
            )


def section_calibration_guard_receipts(
    conn: sqlite3.Connection,
    *,
    window_days: int,
) -> None:
    if not _table_exists(conn, "decision_receipts"):
        return
    cutoff = _cutoff_iso(window_days)
    _print_header(f"CALIBRATION GUARD RECEIPTS (last {window_days}d)")

    rows = conn.execute(
        """
        SELECT
          COALESCE(json_extract(audit_json,'$.source_match_class'), 'unknown') AS source_match_class,
          COUNT(*) AS n,
          SUM(CASE WHEN COALESCE(final_action,'') IN ('order_attempt','order_submitted','dry_run') THEN 1 ELSE 0 END) AS attempted,
          SUM(CASE WHEN COALESCE(json_extract(audit_json,'$.evidence_floor_suppressed_reason'), '') <> '' THEN 1 ELSE 0 END) AS floor_suppressed
        FROM decision_receipts
        WHERE timestamp >= ?
        GROUP BY source_match_class
        ORDER BY n DESC
        LIMIT 12
        """,
        (cutoff,),
    ).fetchall()
    if rows:
        _print_subheader("Source match class")
        print(f"{'source_match_class':<28} {'n':>8} {'attempted':>10} {'floor_supp':>10}")
        for row in rows:
            print(
                f"{str(row['source_match_class'] or ''):<28} "
                f"{int(row['n'] or 0):>8} "
                f"{int(row['attempted'] or 0):>10} "
                f"{int(row['floor_suppressed'] or 0):>10}"
            )

    rows = conn.execute(
        """
        SELECT
          COALESCE(CAST(json_extract(audit_json,'$.ranking_rank') AS INTEGER), 0) AS ranking_rank,
          COUNT(*) AS n,
          SUM(CASE WHEN COALESCE(final_action,'') IN ('order_attempt','order_submitted','dry_run') THEN 1 ELSE 0 END) AS attempted,
          AVG(COALESCE(CAST(json_extract(audit_json,'$.pre_execution_final_score') AS REAL), 0.0)) AS avg_score
        FROM decision_receipts
        WHERE timestamp >= ?
          AND json_extract(audit_json,'$.ranking_rank') IS NOT NULL
        GROUP BY ranking_rank
        ORDER BY ranking_rank ASC
        LIMIT 10
        """,
        (cutoff,),
    ).fetchall()
    if rows:
        _print_subheader("Rank yield")
        print(f"{'rank':>4} {'n':>8} {'attempted':>10} {'attempt_rate':>12} {'avg_score':>10}")
        for row in rows:
            n = int(row["n"] or 0)
            attempted = int(row["attempted"] or 0)
            print(
                f"{int(row['ranking_rank'] or 0):>4} {n:>8} {attempted:>10} "
                f"{_fmt_pct(attempted, n):>12} {float(row['avg_score'] or 0.0):>10.4f}"
            )

    rows = conn.execute(
        """
        SELECT
          CASE
            WHEN MAX(
              ABS(COALESCE(CAST(json_extract(audit_json,'$.score_edge_market') AS REAL), 0.0)),
              ABS(COALESCE(CAST(json_extract(audit_json,'$.score_edge_external') AS REAL), 0.0))
            ) >= 0.45 THEN 'extreme_45_plus'
            WHEN MAX(
              ABS(COALESCE(CAST(json_extract(audit_json,'$.score_edge_market') AS REAL), 0.0)),
              ABS(COALESCE(CAST(json_extract(audit_json,'$.score_edge_external') AS REAL), 0.0))
            ) >= 0.32 THEN 'high_32_to_45'
            ELSE 'normal'
          END AS edge_band,
          COUNT(*) AS n,
          SUM(CASE WHEN COALESCE(json_extract(audit_json,'$.score_rejection_reasons'), '') LIKE '%high_edge_calibration_penalty%' THEN 1 ELSE 0 END) AS high_edge_penalty,
          SUM(CASE WHEN COALESCE(json_extract(audit_json,'$.score_rejection_reasons'), '') LIKE '%extreme_edge_learning_queue%' THEN 1 ELSE 0 END) AS learning_queue
        FROM decision_receipts
        WHERE timestamp >= ?
          AND (
            json_extract(audit_json,'$.score_edge_market') IS NOT NULL
            OR json_extract(audit_json,'$.score_edge_external') IS NOT NULL
          )
        GROUP BY edge_band
        ORDER BY n DESC
        """,
        (cutoff,),
    ).fetchall()
    if rows:
        _print_subheader("High-edge guard bands")
        print(f"{'edge_band':<18} {'n':>8} {'high_edge_pen':>14} {'learning_q':>12}")
        for row in rows:
            print(
                f"{str(row['edge_band'] or ''):<18} {int(row['n'] or 0):>8} "
                f"{int(row['high_edge_penalty'] or 0):>14} "
                f"{int(row['learning_queue'] or 0):>12}"
            )


def section_account_snapshot(conn: sqlite3.Connection) -> None:
    _print_header("ACCOUNT SNAPSHOT (DB-only — for live API run pnl_report.py --sync)")

    if _table_exists(conn, "exchange_settlements"):
        row = conn.execute(
            """
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN won=1 THEN 1 ELSE 0 END) AS wins,
              SUM(CASE WHEN won=0 THEN 1 ELSE 0 END) AS losses,
              SUM(COALESCE(pnl_realized,0.0)) AS pnl_total,
              SUM(COALESCE(contracts,0)) AS contracts
            FROM exchange_settlements
            """,
        ).fetchone()
        print(f"settled trades:    {int(row['total'] or 0)}")
        print(f"wins / losses:     {int(row['wins'] or 0)} / {int(row['losses'] or 0)}")
        wins = int(row["wins"] or 0)
        losses = int(row["losses"] or 0)
        decided = wins + losses
        print(f"win rate:          {_fmt_pct(wins, decided)}")
        print(f"contracts:         {int(row['contracts'] or 0)}")
        print(f"realized pnl:      {_fmt_signed(float(row['pnl_total'] or 0.0))}")

    if _table_exists(conn, "exchange_settlements") and _table_exists(conn, "markets"):
        rows = conn.execute(
            """
            SELECT
              s.market_id,
              s.won,
              s.pnl_realized,
              s.contracts,
              COALESCE(m.question,'') AS question,
              COALESCE(m.category,'') AS category
            FROM exchange_settlements s
            LEFT JOIN markets m ON m.id = s.market_id
            WHERE s.won IN (0, 1)
            """,
        ).fetchall()
        by_family: dict[str, dict[str, float]] = defaultdict(
            lambda: {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}
        )
        for row in rows:
            fam = _market_family_from_id(str(row["market_id"] or ""))
            bucket = by_family[fam]
            bucket["trades"] += 1
            if int(row["won"] or 0) == 1:
                bucket["wins"] += 1
            else:
                bucket["losses"] += 1
            bucket["pnl"] += float(row["pnl_realized"] or 0.0)

        _print_subheader("Realized PnL by market family (all-time)")
        print(f"{'family':<14} {'trades':>8} {'wins':>6} {'losses':>8} {'win_rate':>9} {'pnl':>12}")
        for fam in sorted(by_family, key=lambda f: by_family[f]["pnl"]):
            s = by_family[fam]
            t = int(s["trades"])
            w = int(s["wins"])
            l = int(s["losses"])
            decided = w + l
            print(
                f"{fam:<14} {t:>8} {w:>6} {l:>8} "
                f"{_fmt_pct(w, decided):>9} {s['pnl']:>+12.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only participation/funnel quality analysis.",
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--logs", nargs="+", default=list(DEFAULT_LOG_FILES))
    args = parser.parse_args()

    conn = _open_readonly(args.db)
    try:
        print(SECTION_RULE)
        print("PARTICIPATION QUALITY REPORT")
        print(f"Generated:  {datetime.now(timezone.utc).isoformat()}")
        print(f"Database:   {Path(args.db).resolve()}")
        print(f"Window:     last {args.window_days} days (and all-time where labeled)")
        print(SECTION_RULE)

        section_account_snapshot(conn)
        section_decision_outcome_mix(conn, window_days=args.window_days)
        section_blocked_conviction(conn, window_days=args.window_days)
        section_naming_mismatch(conn, window_days=args.window_days)
        section_decision_field_distribution(conn, window_days=args.window_days)
        section_per_family_prefix(conn, window_days=args.window_days)
        section_cycle_funnel(conn, window_days=args.window_days)
        section_research_queued_settlement_review(conn, window_days=args.window_days)
        section_calibration_guard_receipts(conn, window_days=args.window_days)
        section_grok_failures(args.logs)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
