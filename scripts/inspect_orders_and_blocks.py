"""Inventory all orders placed + blocking reasons over the latest cycles."""

from __future__ import annotations

import json
from collections import Counter, defaultdict


def main() -> None:
    records = []
    with open("logs/trades.log", "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    # Limit to most recent ~10 cycles by correlation_id ordering
    cids_in_order = []
    seen = set()
    for r in reversed(records):
        cid = r.get("correlation_id")
        if cid and cid not in seen:
            seen.add(cid)
            cids_in_order.append(cid)
    last_10_cids = set(cids_in_order[:10])

    orders = []
    blocks = []
    for r in records:
        if r.get("correlation_id") not in last_10_cids:
            continue
        d = r.get("data", {})
        decision = d.get("decision", {})
        audit = d.get("audit", {})
        final_action = audit.get("final_action")
        if final_action in ("order_attempt", "order_submitted", "dry_run", "trade_executed"):
            orders.append({
                "ts": r.get("timestamp"),
                "cid": r.get("correlation_id"),
                "mid": d.get("market_id"),
                "outcome": decision.get("outcome"),
                "side": (d.get("order") or {}).get("side") or (d.get("order") or {}).get("raw", {}).get("order", {}).get("side"),
                "amount": audit.get("bet_amount_usdc"),
                "price": (d.get("order") or {}).get("client_price"),
                "definitive": decision.get("definitive_outcome_detected"),
                "src": decision.get("primary_source_url"),
                "order_id": (d.get("order") or {}).get("id"),
                "status": (d.get("order") or {}).get("status"),
            })
        elif decision.get("should_trade") and audit.get("final_action") in ("skip", "research_queued"):
            blocks.append({
                "ts": r.get("timestamp"),
                "cid": r.get("correlation_id"),
                "mid": d.get("market_id"),
                "final_reason": audit.get("final_reason"),
                "score_final_recomputed": (audit.get("score_breakdown") or {}).get("score_final"),
                "score_critical": audit.get("score_gate_critical_reasons"),
                "definitive": decision.get("definitive_outcome_detected"),
                "definitive_bonus": (audit.get("score_breakdown") or {}).get("score_definitive_outcome_bonus"),
                "edge_market": audit.get("edge_market"),
                "conf": decision.get("confidence"),
                "eq": decision.get("evidence_quality"),
                "src": decision.get("primary_source_url"),
            })

    print(f"=== Orders placed in last 10 cycles: {len(orders)} ===")
    for o in orders:
        print(
            f"  {o['ts'][:19]} cid={o['cid']} mid={o['mid'][:50]:<50} side={o['side']:<5} "
            f"price={o['price']} amount=${o['amount']} def={o['definitive']} status={o['status']}"
        )
    print()

    print(f"=== Blocked should_trade=True records: {len(blocks)} ===")
    by_reason = Counter(b["final_reason"] for b in blocks)
    print("Reason distribution:")
    for reason, count in by_reason.most_common():
        print(f"  {reason}: {count}")
    print()

    for b in blocks:
        bonus = b.get("definitive_bonus")
        print(
            f"  cid={b['cid']} mid={b['mid'][:50]:<50} reason={b['final_reason']} "
            f"def={b['definitive']} bonus={bonus} score_recomputed={b['score_final_recomputed']} "
            f"edge={b['edge_market']} conf={b['conf']} eq={b['eq']}"
        )


if __name__ == "__main__":
    main()
