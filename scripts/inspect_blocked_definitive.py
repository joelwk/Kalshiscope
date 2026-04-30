"""Find should_trade=True markets that hit gates after definitive floor."""

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

    blocked_should_trade = []
    for r in records:
        d = r.get("data", {})
        decision = d.get("decision", {})
        audit = d.get("audit", {})
        if not decision.get("should_trade"):
            continue
        if audit.get("final_action") not in ("skip", "research_queued"):
            continue
        blocked_should_trade.append({
            "ts": r.get("timestamp"),
            "cid": r.get("correlation_id"),
            "mid": d.get("market_id"),
            "conf": decision.get("confidence"),
            "raw_conf": decision.get("raw_confidence"),
            "eq": decision.get("evidence_quality"),
            "my_prob": decision.get("my_prob"),
            "edge_external": decision.get("edge_external"),
            "edge_market": audit.get("edge_market"),
            "edge_source": decision.get("edge_source"),
            "basis": decision.get("evidence_basis"),
            "definitive": decision.get("definitive_outcome_detected"),
            "floor_applied": decision.get("evidence_quality_floor_applied"),
            "src": decision.get("primary_source_url"),
            "final_action": audit.get("final_action"),
            "final_reason": audit.get("final_reason"),
            "score_final": audit.get("pre_execution_final_score") or audit.get("score_final"),
            "score_threshold": audit.get("score_threshold_effective"),
            "score_critical_reasons": audit.get("score_gate_critical_reasons"),
            "score_rejection_reasons": audit.get("pre_execution_rejection_reasons") or audit.get("score_rejection_reasons"),
        })

    print(f"=== Blocked should_trade=True records ({len(blocked_should_trade)}) ===\n")
    for r in blocked_should_trade:
        print(f"cid={r['cid']} mid={r['mid']}")
        print(f"  decision: should_trade=True conf={r['conf']} raw_conf={r['raw_conf']} eq={r['eq']} my_prob={r['my_prob']}")
        print(f"  edge_market={r['edge_market']} edge_external={r['edge_external']} edge_source={r['edge_source']} basis={r['basis']}")
        print(f"  definitive={r['definitive']} floor_applied={r['floor_applied']}")
        print(f"  src={r['src']}")
        print(f"  --> final_action={r['final_action']} reason={r['final_reason']}")
        print(f"      score_final={r['score_final']} score_threshold={r['score_threshold']}")
        print(f"      score_critical_reasons={r['score_critical_reasons']}")
        print(f"      score_rejection_reasons={r['score_rejection_reasons']}")
        print()


if __name__ == "__main__":
    main()
