"""Inspect executed orders and timeouts."""

from __future__ import annotations

import json


def main() -> None:
    with open("logs/trades.log", "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d = r.get("data", {})
            audit = d.get("audit", {})
            decision = d.get("decision", {})
            order = d.get("order")
            final_action = audit.get("final_action")
            if final_action in ("order_attempt", "order_submitted", "dry_run", "trade_executed"):
                print(f"=== EXECUTED: cid={r.get('correlation_id')} mid={d.get('market_id')} ===")
                print(f"  final_action={final_action}")
                print(f"  final_reason={audit.get('final_reason')}")
                print(f"  decision: should_trade={decision.get('should_trade')} conf={decision.get('confidence')} eq={decision.get('evidence_quality')} my_prob={decision.get('my_prob')} definitive={decision.get('definitive_outcome_detected')}")
                print(f"  primary_source_url={decision.get('primary_source_url')}")
                print(f"  order={order}")
                print(f"  bet_amount_usdc={audit.get('bet_amount_usdc')}")
                print(f"  score_final={audit.get('score_final')}")
                print()


if __name__ == "__main__":
    main()
