from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _bucket(confidence: float) -> str:
    left = int(confidence * 10) / 10
    right = left + 0.1
    return f"{left:.1f}-{right:.1f}"


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
        for token in ("GOLD", "SILVER", "WTI", "NATGAS", "COPPER", "CORN", "SOY", "WHEAT", "AAA")
    ):
        return "commodity"
    if any(token in normalized for token in ("LOWT", "HIGHT", "TEMPNYC")):
        return "weather"
    return "generic"


def run(db_path: str) -> None:
    if not Path(db_path).exists():
        print(f"State database not found: {db_path}")
        return
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        coverage_row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_outcomes,
                SUM(CASE WHEN won IS NOT NULL THEN 1 ELSE 0 END) AS resolved_with_won,
                SUM(CASE WHEN won IS NULL THEN 1 ELSE 0 END) AS unresolved_with_won_null
            FROM trade_outcomes
            """
        ).fetchone()
        if coverage_row:
            total_outcomes = int(coverage_row["total_outcomes"] or 0)
            resolved_with_won = int(coverage_row["resolved_with_won"] or 0)
            unresolved_with_won_null = int(coverage_row["unresolved_with_won_null"] or 0)
            print("Outcome coverage:")
            print(
                "  trade_outcomes="
                f"{total_outcomes} resolved_with_won={resolved_with_won} unresolved={unresolved_with_won_null}"
            )
            if total_outcomes > 0:
                print(
                    "  resolved_coverage="
                    f"{(resolved_with_won / total_outcomes):.2%}"
                )

        resolution_rows = conn.execute(
            """
            SELECT resolution_state, COUNT(*) AS n
            FROM trade_outcomes
            GROUP BY resolution_state
            ORDER BY n DESC
            """
        ).fetchall()
        if resolution_rows:
            print("\nResolution states:")
            for row in resolution_rows:
                print(f"  {row['resolution_state']}: n={int(row['n'])}")

        rows = conn.execute(
            """
            SELECT confidence, implied_prob, won
            FROM trade_outcomes
            WHERE confidence IS NOT NULL
              AND won IS NOT NULL
            """
        ).fetchall()
        if not rows:
            unresolved_count = conn.execute(
                "SELECT COUNT(*) FROM trade_outcomes WHERE won IS NULL"
            ).fetchone()[0]
            print("No resolved outcomes with confidence found.")
            print(
                "Calibration status: insufficient resolved outcomes "
                f"(unresolved_trade_outcomes={unresolved_count})."
            )
            rows = []

        if rows:
            total = len(rows)
            wins = sum(1 for row in rows if row["won"] == 1)
            print(f"\nResolved trades: {total}")
            print(f"Win rate: {wins / total:.2%}")

            brier_sum = 0.0
            by_bucket: dict[str, list[int]] = defaultdict(list)
            edge_rows: list[tuple[float, int]] = []
            for row in rows:
                confidence = float(row["confidence"])
                outcome = int(row["won"])
                brier_sum += (confidence - outcome) ** 2
                by_bucket[_bucket(confidence)].append(outcome)
                implied = row["implied_prob"]
                if implied is not None:
                    edge_rows.append((confidence - float(implied), outcome))

            print(f"Brier score: {brier_sum / total:.4f}")
            print("\nCalibration by confidence bucket:")
            for bucket in sorted(by_bucket):
                samples = by_bucket[bucket]
                win_rate = sum(samples) / len(samples)
                print(f"  {bucket}: n={len(samples)} win_rate={win_rate:.2%}")

            trade_outcome_columns = {
                str(column["name"])
                for column in conn.execute("PRAGMA table_info(trade_outcomes)").fetchall()
            }
            required_family_columns = {"market_id", "won", "pnl_estimate", "amount_usdc", "confidence"}
            if required_family_columns.issubset(trade_outcome_columns):
                family_rows = conn.execute(
                    """
                    SELECT market_id, won, pnl_estimate, amount_usdc, confidence
                    FROM trade_outcomes
                    WHERE won IS NOT NULL
                    """
                ).fetchall()
                family_stats: dict[str, dict[str, float]] = defaultdict(
                    lambda: {
                        "n": 0.0,
                        "wins": 0.0,
                        "pnl": 0.0,
                        "deployed": 0.0,
                        "high_conf_losses": 0.0,
                    }
                )
                for family_row in family_rows:
                    family = _market_family_from_id(str(family_row["market_id"] or ""))
                    stats = family_stats[family]
                    stats["n"] += 1.0
                    stats["wins"] += 1.0 if int(family_row["won"] or 0) == 1 else 0.0
                    stats["pnl"] += float(family_row["pnl_estimate"] or 0.0)
                    stats["deployed"] += float(family_row["amount_usdc"] or 0.0)
                    if int(family_row["won"] or 0) == 0 and float(family_row["confidence"] or 0.0) >= 0.90:
                        stats["high_conf_losses"] += 1.0
                if family_stats:
                    print("\nResolved P&L by market family:")
                    for family_name in sorted(
                        family_stats,
                        key=lambda key: family_stats[key]["pnl"],
                    ):
                        stats = family_stats[family_name]
                        n = int(stats["n"])
                        wins_count = int(stats["wins"])
                        win_rate = (wins_count / n) if n > 0 else 0.0
                        print(
                            f"  {family_name}: n={n} wins={wins_count} win_rate={win_rate:.2%} "
                            f"pnl={stats['pnl']:+.2f} deployed={stats['deployed']:.2f} "
                            f"high_conf_losses={int(stats['high_conf_losses'])}"
                        )

            if edge_rows:
                edge_rows.sort(key=lambda item: item[0])
                decile_size = max(1, len(edge_rows) // 10)
                print("\nEdge deciles (lowest to highest):")
                for idx in range(0, len(edge_rows), decile_size):
                    decile = edge_rows[idx : idx + decile_size]
                    avg_edge = sum(item[0] for item in decile) / len(decile)
                    win_rate = sum(item[1] for item in decile) / len(decile)
                    print(
                        f"  decile={idx // decile_size + 1} n={len(decile)} "
                        f"avg_edge={avg_edge:.4f} win_rate={win_rate:.2%}"
                    )

        market_terminal_rows = conn.execute(
            """
            SELECT last_terminal_outcome, COUNT(*) AS n
            FROM markets
            GROUP BY last_terminal_outcome
            ORDER BY n DESC
            LIMIT 12
            """
        ).fetchall()
        if market_terminal_rows:
            print("\nRecent terminal outcome mix:")
            for row in market_terminal_rows:
                label = row["last_terminal_outcome"] or "none"
                print(f"  {label}: n={int(row['n'])}")

        if _table_exists(conn, "decision_receipts"):
            family_rows = conn.execute(
                """
                SELECT
                    COALESCE(json_extract(audit_json, '$.market_family'), 'unknown') AS market_family,
                    COUNT(*) AS n
                FROM decision_receipts
                GROUP BY market_family
                ORDER BY n DESC
                LIMIT 12
                """
            ).fetchall()
            if family_rows:
                print("\nDecision receipt family mix:")
                for row in family_rows:
                    print(f"  {row['market_family']}: n={int(row['n'])}")
            evidence_rows = conn.execute(
                """
                SELECT
                    COALESCE(json_extract(audit_json, '$.evidence_basis_class'), 'unknown') AS evidence_basis_class,
                    COUNT(*) AS n
                FROM decision_receipts
                GROUP BY evidence_basis_class
                ORDER BY n DESC
                LIMIT 12
                """
            ).fetchall()
            if evidence_rows:
                print("\nDecision receipt evidence-basis mix:")
                for row in evidence_rows:
                    print(f"  {row['evidence_basis_class']}: n={int(row['n'])}")
            outcome_rows = conn.execute(
                """
                SELECT
                    COALESCE(json_extract(audit_json, '$.final_action'), 'unknown') AS final_action,
                    COUNT(*) AS n
                FROM decision_receipts
                GROUP BY final_action
                ORDER BY n DESC
                LIMIT 12
                """
            ).fetchall()
            if outcome_rows:
                print("\nDecision outcome mix:")
                for row in outcome_rows:
                    print(f"  {row['final_action']}: n={int(row['n'])}")

            blocked_summary = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_should_trade,
                    SUM(
                        CASE
                            WHEN COALESCE(json_extract(audit_json, '$.final_action'), '') = 'skip'
                            THEN 1 ELSE 0
                        END
                    ) AS blocked_should_trade
                FROM decision_receipts
                WHERE COALESCE(json_extract(decision_json, '$.should_trade'), 0) = 1
                """
            ).fetchone()
            if blocked_summary:
                total_should_trade = int(blocked_summary["total_should_trade"] or 0)
                blocked_should_trade = int(blocked_summary["blocked_should_trade"] or 0)
                print("\nShould-trade block rate:")
                print(f"  should_trade_total={total_should_trade}")
                print(f"  should_trade_blocked={blocked_should_trade}")
                if total_should_trade > 0:
                    print(
                        "  should_trade_block_rate="
                        f"{(blocked_should_trade / total_should_trade):.2%}"
                    )

            blocked_reason_rows = conn.execute(
                """
                SELECT
                    COALESCE(json_extract(audit_json, '$.final_reason'), 'unknown') AS final_reason,
                    COUNT(*) AS n
                FROM decision_receipts
                WHERE COALESCE(json_extract(decision_json, '$.should_trade'), 0) = 1
                  AND COALESCE(json_extract(audit_json, '$.final_action'), '') = 'skip'
                GROUP BY final_reason
                ORDER BY n DESC
                LIMIT 12
                """
            ).fetchall()
            if blocked_reason_rows:
                print("  blocked_reasons:")
                for row in blocked_reason_rows:
                    print(f"    {row['final_reason']}: n={int(row['n'])}")

            family_block_rows = conn.execute(
                """
                SELECT
                    COALESCE(json_extract(audit_json, '$.market_family'), 'unknown') AS market_family,
                    COUNT(*) AS n_total,
                    SUM(
                        CASE
                            WHEN COALESCE(json_extract(audit_json, '$.final_action'), '') = 'skip'
                            THEN 1 ELSE 0
                        END
                    ) AS n_blocked
                FROM decision_receipts
                WHERE COALESCE(json_extract(decision_json, '$.should_trade'), 0) = 1
                GROUP BY market_family
                ORDER BY n_total DESC
                LIMIT 12
                """
            ).fetchall()
            if family_block_rows:
                print("\nShould-trade block rate by family:")
                for row in family_block_rows:
                    total = int(row["n_total"] or 0)
                    blocked = int(row["n_blocked"] or 0)
                    blocked_rate = (blocked / total) if total > 0 else 0.0
                    print(
                        f"  {row['market_family']}: total={total} blocked={blocked} blocked_rate={blocked_rate:.2%}"
                    )

            penalty_avg_row = conn.execute(
                """
                SELECT
                    AVG(COALESCE(CAST(json_extract(audit_json, '$.score_liquidity_penalty') AS REAL), 0.0)) AS liquidity_penalty_avg,
                    AVG(COALESCE(CAST(json_extract(audit_json, '$.score_weather_penalty') AS REAL), 0.0)) AS weather_penalty_avg,
                    AVG(COALESCE(CAST(json_extract(audit_json, '$.score_proxy_evidence_penalty') AS REAL), 0.0)) AS proxy_evidence_penalty_avg,
                    AVG(COALESCE(CAST(json_extract(audit_json, '$.score_repeated_penalty') AS REAL), 0.0)) AS repeated_penalty_avg,
                    AVG(COALESCE(CAST(json_extract(audit_json, '$.score_generic_bin_penalty') AS REAL), 0.0)) AS generic_bin_penalty_avg,
                    AVG(COALESCE(CAST(json_extract(audit_json, '$.score_ambiguous_resolution_penalty') AS REAL), 0.0)) AS ambiguous_resolution_penalty_avg
                FROM decision_receipts
                """
            ).fetchone()
            if penalty_avg_row:
                print("\nAverage score penalties (decision receipts):")
                print(
                    "  liquidity={:.4f} weather={:.4f} proxy_evidence={:.4f} repeated={:.4f} generic_bin={:.4f} ambiguous_resolution={:.4f}".format(
                        float(penalty_avg_row["liquidity_penalty_avg"] or 0.0),
                        float(penalty_avg_row["weather_penalty_avg"] or 0.0),
                        float(penalty_avg_row["proxy_evidence_penalty_avg"] or 0.0),
                        float(penalty_avg_row["repeated_penalty_avg"] or 0.0),
                        float(penalty_avg_row["generic_bin_penalty_avg"] or 0.0),
                        float(penalty_avg_row["ambiguous_resolution_penalty_avg"] or 0.0),
                    )
                )

            decision_receipt_columns = {
                str(column["name"])
                for column in conn.execute("PRAGMA table_info(decision_receipts)").fetchall()
            }
            if {"market_id", "final_action", "decision_json", "audit_json"}.issubset(
                decision_receipt_columns
            ):
                retrospective_rows = conn.execute(
                    """
                    SELECT
                        market_id,
                        COALESCE(json_extract(audit_json, '$.market_family'), 'unknown') AS market_family,
                        COALESCE(CAST(json_extract(audit_json, '$.pre_execution_final_score') AS REAL), 0.0) AS pre_execution_score,
                        COALESCE(CAST(json_extract(decision_json, '$.evidence_quality') AS REAL), 0.0) AS evidence_quality,
                        LOWER(COALESCE(json_extract(decision_json, '$.edge_source'), 'none')) AS edge_source,
                        COALESCE(CAST(json_extract(decision_json, '$.confidence') AS REAL), 0.0) AS confidence
                    FROM decision_receipts
                    WHERE COALESCE(final_action, '') IN ('order_submitted', 'dry_run')
                    """
                ).fetchall()
                if retrospective_rows:
                    reject_counters: dict[str, int] = defaultdict(int)
                    rejected_total = 0
                    for row in retrospective_rows:
                        reasons: list[str] = []
                        family = str(row["market_family"] or "unknown")
                        market_id = str(row["market_id"] or "")
                        pre_execution_score = float(row["pre_execution_score"] or 0.0)
                        evidence_quality = float(row["evidence_quality"] or 0.0)
                        edge_source = str(row["edge_source"] or "none")
                        confidence = float(row["confidence"] or 0.0)
                        if pre_execution_score < 0.35:
                            reasons.append("score_gate_0.35")
                        if family == "weather" and evidence_quality < 0.70:
                            reasons.append("weather_evidence_floor_0.70")
                        if market_id.startswith(("KXSNLMENTION-", "KXLASTWORDCOUNT-", "KXVANCEMENTION-")):
                            reasons.append("speech_blocklist_prefix")
                        if edge_source in {"fallback", "none"} and confidence - evidence_quality > 0.20:
                            reasons.append("overconfidence_gap_guard")
                        if reasons:
                            rejected_total += 1
                            for reason in reasons:
                                reject_counters[reason] += 1
                    print("\nRetrospective rejection simulation (new thresholds):")
                    print(f"  submitted_decisions={len(retrospective_rows)}")
                    print(f"  would_be_rejected={rejected_total}")
                    if retrospective_rows:
                        print(
                            "  would_be_rejected_rate="
                            f"{(rejected_total / len(retrospective_rows)):.2%}"
                        )
                    if reject_counters:
                        print("  rejection_reasons:")
                        for reason in sorted(reject_counters, key=reject_counters.get, reverse=True):
                            print(f"    {reason}: n={reject_counters[reason]}")

        if _table_exists(conn, "cycle_receipts"):
            cycle_rows = conn.execute(
                """
                SELECT payload_json
                FROM cycle_receipts
                ORDER BY id DESC
                LIMIT 200
                """
            ).fetchall()
            total_cycles = 0
            total_api_tokens = 0
            total_api_cost = 0.0
            total_order_attempts = 0
            total_score_gate_blocked = 0
            total_decisions = 0
            evidence_basis_counts: dict[str, int] = defaultdict(int)
            for row in cycle_rows:
                payload_raw = row["payload_json"]
                if not payload_raw:
                    continue
                try:
                    payload = json.loads(payload_raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                total_cycles += 1
                total_api_tokens += int(payload.get("api_tokens_consumed") or 0)
                total_api_cost += float(payload.get("api_cost_estimate_usd") or 0.0)
                total_order_attempts += int(payload.get("order_attempts") or 0)
                total_decisions += int(payload.get("decisions_made") or 0)
                rejection_breakdown = payload.get("rejection_breakdown")
                if isinstance(rejection_breakdown, dict):
                    total_score_gate_blocked += int(
                        rejection_breakdown.get("score_gate_blocked") or 0
                    )
                evidence_breakdown = payload.get("evidence_basis_breakdown")
                if isinstance(evidence_breakdown, dict):
                    for basis, count in evidence_breakdown.items():
                        evidence_basis_counts[str(basis)] += int(count or 0)

            if total_cycles > 0:
                print("\nCycle API/score-gate summary (latest 200 cycles):")
                print(f"  cycles={total_cycles}")
                print(f"  api_tokens_consumed={total_api_tokens}")
                print(f"  api_cost_estimate_usd={total_api_cost:.6f}")
                print(
                    "  avg_api_tokens_per_cycle="
                    f"{(total_api_tokens / total_cycles):.2f}"
                )
                print(
                    "  avg_api_cost_per_cycle_usd="
                    f"{(total_api_cost / total_cycles):.6f}"
                )
                if total_order_attempts > 0:
                    print(
                        "  api_cost_per_trade_attempt_usd="
                        f"{(total_api_cost / total_order_attempts):.6f}"
                    )
                else:
                    print("  api_cost_per_trade_attempt_usd=n/a (no order attempts)")
                if total_decisions > 0:
                    print(
                        "  score_gate_block_rate="
                        f"{(total_score_gate_blocked / total_decisions):.2%}"
                    )
                if evidence_basis_counts:
                    print("  evidence_basis_breakdown:")
                    for basis in sorted(evidence_basis_counts):
                        print(f"    {basis}: n={evidence_basis_counts[basis]}")

        profitable_row = conn.execute(
            """
            SELECT COUNT(*) AS profitable_count
            FROM trade_outcomes
            WHERE won = 1
            """
        ).fetchone()
        profitable_count = int(profitable_row["profitable_count"] or 0) if profitable_row else 0
        if profitable_count > 0:
            api_cost_row = conn.execute(
                """
                SELECT payload_json
                FROM cycle_receipts
                """
            ).fetchall() if _table_exists(conn, "cycle_receipts") else []
            cumulative_api_cost = 0.0
            for row in api_cost_row:
                payload_raw = row["payload_json"]
                if not payload_raw:
                    continue
                try:
                    payload = json.loads(payload_raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    cumulative_api_cost += float(payload.get("api_cost_estimate_usd") or 0.0)
            print(
                "\nAPI cost per profitable trade (est.): "
                f"{(cumulative_api_cost / profitable_count):.6f} USD"
            )
        else:
            print("\nAPI cost per profitable trade (est.): n/a (no resolved profitable trades)")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily calibration metrics report.")
    parser.add_argument(
        "--db",
        default="data/market_state.db",
        help="Path to market state sqlite database",
    )
    args = parser.parse_args()
    run(args.db)


if __name__ == "__main__":
    main()

