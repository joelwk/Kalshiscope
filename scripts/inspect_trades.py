"""Inspect post-analysis (non-pre-analysis) trade decisions."""

from __future__ import annotations

import json


def main() -> None:
    records = []
    with open("logs/trades.log", "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    deep_records = []
    should_trade_true = []
    for r in records:
        d = r.get("data", {})
        decision = d.get("decision", {})
        audit = d.get("audit", {})
        final_reason = str(audit.get("final_reason") or "")
        if not final_reason.startswith("pre_analysis_"):
            deep_records.append({
                "ts": r.get("timestamp"),
                "cid": r.get("correlation_id"),
                "mid": d.get("market_id"),
                "should_trade": decision.get("should_trade"),
                "conf": decision.get("confidence"),
                "eq": decision.get("evidence_quality"),
                "edge_external": decision.get("edge_external"),
                "my_prob": decision.get("my_prob"),
                "definitive": decision.get("definitive_outcome_detected"),
                "evidence_basis": decision.get("evidence_basis"),
                "edge_source": decision.get("edge_source"),
                "primary_source_url": decision.get("primary_source_url"),
                "final_action": audit.get("final_action"),
                "final_reason": final_reason,
                "rejection_stage": audit.get("rejection_stage"),
                "edge_market": audit.get("edge_market"),
                "evidence_quality_floor_applied": decision.get("evidence_quality_floor_applied"),
            })
        if decision.get("should_trade") is True:
            should_trade_true.append({
                "ts": r.get("timestamp"),
                "cid": r.get("correlation_id"),
                "mid": d.get("market_id"),
                "conf": decision.get("confidence"),
                "eq": decision.get("evidence_quality"),
                "my_prob": decision.get("my_prob"),
                "definitive": decision.get("definitive_outcome_detected"),
                "src": decision.get("primary_source_url"),
                "final_reason": audit.get("final_reason"),
            })

    print(f"=== Total trade-decision records: {len(records)} ===")
    print(f"=== Deep-analysis records (post pre-analysis): {len(deep_records)} ===")
    for r in deep_records:
        print(f"  cid={r['cid']} mid={r['mid']}")
        print(f"    should_trade={r['should_trade']} conf={r['conf']} eq={r['eq']} my_prob={r['my_prob']} edge_ext={r['edge_external']}")
        print(f"    edge_market={r['edge_market']} edge_source={r['edge_source']} basis={r['evidence_basis']} definitive={r['definitive']}")
        print(f"    floor_applied={r['evidence_quality_floor_applied']}")
        print(f"    src={r['primary_source_url']}")
        print(f"    final_action={r['final_action']} reason={r['final_reason']} stage={r['rejection_stage']}")
        print()

    print(f"=== should_trade=True records ({len(should_trade_true)}) ===")
    for r in should_trade_true:
        print(f"  cid={r['cid']} mid={r['mid']} conf={r['conf']} eq={r['eq']} my_prob={r['my_prob']} def={r['definitive']} reason={r['final_reason']}")
        print(f"    src={r['src']}")


if __name__ == "__main__":
    main()
