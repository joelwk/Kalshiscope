"""Quick diagnostic dump of cycle funnels and key events from logs."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _rotated_log_files() -> list[str]:
    """Return all predictbot log files in chronological order (oldest first)."""
    numbered = sorted(
        glob.glob("logs/predictbot.log.[0-9]*"),
        key=lambda p: int(p.rsplit(".", 1)[-1]),
        reverse=True,
    )
    numbered.append("logs/predictbot.log")
    return numbered


def main() -> None:
    records: list[dict] = []
    for fname in _rotated_log_files():
        try:
            with open(fname, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        except FileNotFoundError:
            pass

    print(f"Total log records: {len(records)}")
    funnels = [r for r in records if "Cycle funnel" in r.get("message", "")]
    yield_alerts = [r for r in records if r.get("message", "").startswith("Cycle yield alert")]
    def_floor = [r for r in records if r.get("message", "").startswith("Definitive outcome floor")]
    monitor = [
        r for r in records
        if "MONITOR_ONLY" in r.get("message", "") or "monitor_only" in r.get("message", "")
    ]
    research_q = [r for r in records if r.get("message", "").startswith("Research queue captured")]

    print(f"Cycle funnels: {len(funnels)}")
    print(f"Yield alerts: {len(yield_alerts)}")
    print(f"Definitive floor applied logs: {len(def_floor)}")
    print(f"Monitor-only logs: {len(monitor)}")
    print(f"Research queue captured logs: {len(research_q)}")
    print()
    print("=== Per-cycle funnels ===")
    for r in funnels:
        d = r.get("data", {})
        cid = r.get("correlation_id", "?")
        print(
            f"cid={cid} fetched={d.get('fetched')} "
            f"filtered={d.get('filtered')} analyzed={d.get('analyzed')} "
            f"exec={d.get('execution_candidates')} "
            f"rq={d.get('research_queue_size')} "
            f"stb={d.get('should_trade_but_blocked')} "
            f"attempts={d.get('order_attempts')}"
        )
        print(f"    tier_breakdown={d.get('participation_tier_breakdown')}")
        print(
            f"    def_floor_count={d.get('definitive_outcome_floor_applied_count')} "
            f"timeout_monitor={d.get('timeout_routed_to_monitor_only_count')} "
            f"neg_score_skip={d.get('negative_best_score_skipped_count')}"
        )
        print(
            f"    pre_analysis_breakdown={d.get('pre_analysis_rejection_breakdown')}"
        )
        print(
            f"    should_trade_blocked_breakdown={d.get('should_trade_blocked_breakdown')}"
        )
        print()

    if def_floor:
        print("=== Definitive floor applied events ===")
        for r in def_floor:
            d = r.get("data", {})
            print(
                f"  cid={r.get('correlation_id')} mid={d.get('market_id')} "
                f"eq_after={d.get('evidence_quality_after')} "
                f"conf={d.get('confidence')} "
                f"auto={d.get('definitive_outcome_auto_detected')} "
                f"src={d.get('primary_source_url')}"
            )
        print()

    if monitor:
        print("=== Monitor-only routing events ===")
        for r in monitor:
            d = r.get("data", {})
            print(
                f"  cid={r.get('correlation_id')} mid={d.get('market_id')} "
                f"reason={d.get('final_reason')} profile={d.get('search_profile')}"
            )


if __name__ == "__main__":
    main()
