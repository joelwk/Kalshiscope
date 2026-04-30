"""Show raw audit fields for the score_gate_blocked + definitive cases."""

from __future__ import annotations

import json


def main() -> None:
    with open("logs/trades.log", "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d = r.get("data", {})
            decision = d.get("decision", {})
            audit = d.get("audit", {})
            if (
                decision.get("should_trade")
                and decision.get("definitive_outcome_detected")
                and audit.get("final_reason") == "score_gate_blocked"
            ):
                mid = d.get("market_id", "?")
                print(f"=== mid={mid} ===")
                print(f"audit keys: {sorted([k for k in audit.keys() if 'score' in k.lower() or 'reject' in k.lower() or 'def' in k.lower() or 'sup' in k.lower()])}")
                print(f"pre_execution_rejection_reasons={audit.get('pre_execution_rejection_reasons')}")
                print(f"score_rejection_reasons={audit.get('score_rejection_reasons')}")
                print(f"score_gate_critical_reasons={audit.get('score_gate_critical_reasons')}")
                print(f"pre_execution_final_score={audit.get('pre_execution_final_score')}")
                print(f"score_final={audit.get('score_final')}")
                print(f"definitive_outcome_floor_applied={audit.get('definitive_outcome_floor_applied')}")
                print(f"score_breakdown:")
                for k, v in (audit.get("score_breakdown") or {}).items():
                    print(f"  {k}={v}")
                print()


if __name__ == "__main__":
    main()
