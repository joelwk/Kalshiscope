from __future__ import annotations

import argparse
from collections import Counter
import json
import sqlite3
from pathlib import Path
from typing import Any


def _load_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _counter_from_mapping(payloads: list[dict[str, Any]], key: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for payload in payloads:
        value = payload.get(key)
        if isinstance(value, dict):
            for subkey, count in value.items():
                try:
                    counts[str(subkey)] += int(count)
                except (TypeError, ValueError):
                    continue
        elif value:
            counts[str(value)] += 1
    return counts


def _recent_cycle_payloads(conn: sqlite3.Connection, limit: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT payload_json
        FROM cycle_receipts
        ORDER BY id DESC
        LIMIT ?
        """,
        (max(1, int(limit)),),
    ).fetchall()
    return [_load_json(row["payload_json"]) for row in rows]


def _recent_decision_payloads(conn: sqlite3.Connection, limit: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT final_action, final_reason, decision_json, audit_json
        FROM decision_receipts
        ORDER BY id DESC
        LIMIT ?
        """,
        (max(1, int(limit)),),
    ).fetchall()
    payloads: list[dict[str, Any]] = []
    for row in rows:
        decision = _load_json(row["decision_json"])
        audit = _load_json(row["audit_json"])
        payloads.append(
            {
                "final_action": row["final_action"],
                "final_reason": row["final_reason"],
                "decision": decision,
                "audit": audit,
            }
        )
    return payloads


def build_report(db_path: Path, *, cycles: int, decisions: int) -> dict[str, Any]:
    uri = f"file:{db_path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cycle_payloads = _recent_cycle_payloads(conn, cycles)
        decision_payloads = _recent_decision_payloads(conn, decisions)
    finally:
        conn.close()

    cycle_totals: Counter[str] = Counter()
    cycle_total_aliases = {
        "markets_fetched": ("markets_fetched", "fetched_markets"),
        "markets_filtered": ("markets_filtered", "eligible_markets"),
        "analysis_candidates": ("analysis_candidates",),
        "markets_analyzed": ("markets_analyzed", "analyzed_markets"),
        "execution_candidates": ("execution_candidates",),
        "research_queue_size": ("research_queue_size", "research_queue_emissions"),
        "should_trade_but_blocked_count": (
            "should_trade_but_blocked_count",
            "should_trade_but_blocked",
        ),
        "timeout_routed_to_monitor_only_count": (
            "timeout_routed_to_monitor_only_count",
        ),
        "missing_primary_source_count": (
            "missing_primary_source_count",
            "source_requirement_missing_primary_source_count",
        ),
    }
    for payload in cycle_payloads:
        for target_key, aliases in cycle_total_aliases.items():
            for alias in aliases:
                if alias in payload:
                    try:
                        cycle_totals[target_key] += int(payload.get(alias) or 0)
                    except (TypeError, ValueError):
                        pass
                    break

    final_actions: Counter[str] = Counter()
    final_reasons: Counter[str] = Counter()
    participation_tiers: Counter[str] = Counter()
    skip_due: Counter[str] = Counter()
    evidence_basis: Counter[str] = Counter()
    source_status: Counter[str] = Counter()
    edge_sources: Counter[str] = Counter()
    should_trade_blocked: Counter[str] = Counter()
    for payload in decision_payloads:
        decision = payload["decision"]
        audit = payload["audit"]
        if payload.get("final_action"):
            final_actions[str(payload["final_action"])] += 1
        if payload.get("final_reason"):
            final_reasons[str(payload["final_reason"])] += 1
        if audit.get("participation_tier"):
            participation_tiers[str(audit["participation_tier"])] += 1
        if audit.get("skip_due_to"):
            skip_due[str(audit["skip_due_to"])] += 1
        if decision.get("evidence_basis"):
            evidence_basis[str(decision["evidence_basis"])] += 1
        if decision.get("edge_source"):
            edge_sources[str(decision["edge_source"])] += 1
        status = audit.get("source_requirement_status")
        if isinstance(status, dict) and status.get("status"):
            source_status[str(status["status"])] += 1
        if decision.get("should_trade") is True and payload.get("final_action") != "order_attempt":
            should_trade_blocked[str(payload.get("final_reason") or "unknown")] += 1

    cycle_breakdowns = {
        "pre_analysis_rejection_breakdown": _counter_from_mapping(
            cycle_payloads,
            "pre_analysis_rejection_breakdown",
        ),
        "should_trade_blocked_breakdown": _counter_from_mapping(
            cycle_payloads,
            "should_trade_blocked_breakdown",
        ),
        "participation_tier_breakdown": _counter_from_mapping(
            cycle_payloads,
            "participation_tier_breakdown",
        ),
        "evidence_basis_breakdown": _counter_from_mapping(
            cycle_payloads,
            "evidence_basis_breakdown",
        ),
    }

    return {
        "db_path": str(db_path),
        "cycles_sampled": len(cycle_payloads),
        "decisions_sampled": len(decision_payloads),
        "cycle_totals": dict(cycle_totals),
        "cycle_breakdowns": {
            key: dict(value.most_common())
            for key, value in cycle_breakdowns.items()
        },
        "decision_final_actions": dict(final_actions.most_common()),
        "decision_final_reasons": dict(final_reasons.most_common()),
        "decision_participation_tiers": dict(participation_tiers.most_common()),
        "decision_skip_due": dict(skip_due.most_common()),
        "decision_should_trade_blocked": dict(should_trade_blocked.most_common()),
        "decision_evidence_basis": dict(evidence_basis.most_common()),
        "decision_edge_sources": dict(edge_sources.most_common()),
        "decision_source_requirement_status": dict(source_status.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only participation funnel report from Prediscope SQLite state."
    )
    parser.add_argument("--db", default="data/market_state.db", help="SQLite DB path")
    parser.add_argument("--cycles", type=int, default=20, help="Recent cycle receipts")
    parser.add_argument("--decisions", type=int, default=5000, help="Recent decisions")
    args = parser.parse_args()
    report = build_report(Path(args.db), cycles=args.cycles, decisions=args.decisions)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
