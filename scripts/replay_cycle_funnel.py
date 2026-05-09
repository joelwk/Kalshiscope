"""Replay recent trade-decision logs through the new participation tier
classifier and print a side-by-side comparison.

Usage::

    python scripts/replay_cycle_funnel.py [--log logs/trades.log]

Reads each JSON-line trade decision record, extracts audit fields, and
re-classifies using ``classify_participation``.  Prints:

- Old ``final_action / final_reason`` distribution
- New ``participation_tier`` distribution
- Reclassification counts (e.g. ``historical_prefix_pnl_block`` →
  ``RESEARCH_ONLY_LEARNING_QUEUE`` due to small sample)
- Evidence-quality floor uplift count
"""

from __future__ import annotations

import json
import sys
import argparse
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from participation import (
    HistoricalGateResult,
    ParticipationTier,
    classify_participation,
)

_DEFAULT_LOG = "logs/trades.log"


def _normalize_participation_tier(raw: object) -> str:
    text = str(raw or "").strip()
    if text.startswith("ParticipationTier."):
        text = text.split(".", 1)[1].lower()
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", default=None, help="Path to trades log")
    parser.add_argument("--log", dest="log_flag", default=None, help="Path to trades log")
    args = parser.parse_args()
    log_path = args.log_flag or args.log or _DEFAULT_LOG
    path = Path(log_path)
    if not path.exists():
        print(f"Log file not found: {path}")
        return

    old_action_counts: Counter[str] = Counter()
    old_reason_counts: Counter[str] = Counter()
    new_tier_counts: Counter[str] = Counter()
    reclassified: Counter[str] = Counter()
    definitive_floor_would_help = 0
    total = 0

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        data = record.get("data", record)
        audit = data.get("audit", data.get("execution_audit", {}))
        decision = data.get("decision", {})

        old_action = str(audit.get("final_action") or "unknown")
        old_reason = str(audit.get("final_reason") or "unknown")
        old_action_counts[old_action] += 1
        old_reason_counts[old_reason] += 1
        total += 1

        existing_tier = _normalize_participation_tier(audit.get("participation_tier"))
        if existing_tier:
            new_tier = existing_tier
            new_tier_counts[new_tier] += 1
            transition_key = f"{old_reason} -> {new_tier}"
            if old_reason != new_tier:
                reclassified[transition_key] += 1
            continue

        if old_action in {"order_attempt", "order_submitted", "dry_run"}:
            new_tier = str(ParticipationTier.EXECUTION_ELIGIBLE)
            new_tier_counts[new_tier] += 1
            transition_key = f"{old_reason} -> {new_tier}"
            if old_reason != new_tier:
                reclassified[transition_key] += 1
            continue

        hist_sample = audit.get("historical_gate_prefix_sample_size")
        hist_wr = audit.get("historical_gate_prefix_win_rate")
        hist_wlb = audit.get("historical_gate_wilson_lb")

        pre_reason = old_reason if old_reason.startswith("pre_analysis_") else None

        result = classify_participation(
            historical_gate=HistoricalGateResult(
                allowed=bool(audit.get("historical_gate_allowed", True)),
                reason=audit.get("historical_gate_reason"),
                sample_size=int(hist_sample) if hist_sample is not None else None,
                wilson_win_rate_lower_bound=float(hist_wlb) if hist_wlb is not None else None,
            ) if hist_sample is not None else None,
            pre_analysis_rejection_reason=pre_reason,
            pre_analysis_metadata=audit,
            decision_should_trade=decision.get("should_trade"),
            decision_abstain=decision.get("abstain"),
            decision_definitive_outcome=bool(decision.get("definitive_outcome_detected")),
            decision_evidence_basis=decision.get("evidence_basis"),
            decision_edge_source=decision.get("edge_source"),
            decision_evidence_quality=decision.get("evidence_quality"),
            confidence_value=decision.get("confidence"),
            confidence_threshold=audit.get("counterfactual_required_confidence"),
            edge_value=audit.get("gate_edge_actual") or audit.get("edge_market"),
            evidence_quality_threshold=0.75,
            score_gate_blocked="score_gate" in old_reason,
            score_gate_reason=old_reason if "score_gate" in old_reason else None,
            downstream_gate_reason=(
                old_reason
                if old_action in {"skip", "monitor_only"}
                and decision.get("should_trade") is True
                else None
            ),
        )

        new_tier = str(result.tier)
        new_tier_counts[new_tier] += 1

        transition_key = f"{old_reason} -> {new_tier}"
        if old_reason != new_tier:
            reclassified[transition_key] += 1

        if (
            decision.get("should_trade") is True
            and decision.get("definitive_outcome_detected")
            and old_reason in ("evidence_quality_below_min", "edge_above_reasonable_max")
        ):
            definitive_floor_would_help += 1

    print(f"\n=== Replay Cycle Funnel ({total} records from {log_path}) ===\n")
    if total <= 0:
        print("No trade-decision records found.")
        return

    print("OLD final_action distribution:")
    for action, count in old_action_counts.most_common():
        print(f"  {action}: {count} ({count / total * 100:.1f}%)")

    print("\nOLD final_reason distribution (top 15):")
    for reason, count in old_reason_counts.most_common(15):
        print(f"  {reason}: {count}")

    print(f"\nNEW participation_tier distribution:")
    for tier, count in new_tier_counts.most_common():
        print(f"  {tier}: {count} ({count / total * 100:.1f}%)")

    print(f"\nReclassifications (top 20):")
    for transition, count in reclassified.most_common(20):
        print(f"  {transition}: {count}")

    print(f"\nDefinitive-outcome floor would have helped: {definitive_floor_would_help}")
    print()


if __name__ == "__main__":
    main()
