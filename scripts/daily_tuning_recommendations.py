from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from calibration import compute_adaptive_thresholds


def _parse_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _extract_calibration_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in rows:
        message = str(row.get("message") or "")
        if not (
            message.startswith("Calibration sample:")
            or message.startswith("Calibration orderbook sample:")
        ):
            continue
        data = row.get("data")
        if isinstance(data, dict):
            samples.append(data)
    return samples


def _extract_grok_durations(rows: list[dict[str, Any]]) -> list[float]:
    durations: list[float] = []
    for row in rows:
        logger_name = str(row.get("logger") or "")
        message = str(row.get("message") or "")
        if logger_name != "grok_client":
            continue
        if "Grok decision" not in message and "Grok deep decision" not in message:
            continue
        data = row.get("data") or {}
        duration = data.get("duration_ms")
        if isinstance(duration, (int, float)):
            durations.append(float(duration))
    return durations


def _nested_number(data: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value: Any = data
        for part in key.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _extract_profit_lever_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    volume_discounts: list[float] = []
    edge_reductions: list[float] = []
    kelly_effective_fractions: list[float] = []
    for row in rows:
        data = row.get("data")
        if not isinstance(data, dict):
            continue
        volume_discount = _nested_number(
            data,
            "score_volume_amplifier_discount",
            "score_breakdown.score_volume_amplifier_discount",
        )
        if volume_discount is not None:
            volume_discounts.append(volume_discount)
        edge_reduction = _nested_number(
            data,
            "gate_edge_dynamic_reduction",
            "score_breakdown.gate_edge_dynamic_reduction",
        )
        if edge_reduction is not None:
            edge_reductions.append(edge_reduction)
        kelly_fraction = _nested_number(
            data,
            "kelly_effective_fraction",
            "audit.kelly_effective_fraction",
        )
        if kelly_fraction is not None:
            kelly_effective_fractions.append(kelly_fraction)

    reduced_count = sum(1 for value in edge_reductions if value > 1e-9)
    kelly_gt_half = sum(1 for value in kelly_effective_fractions if value > 0.50)
    return {
        "volume_discount_count": len(volume_discounts),
        "volume_discount_avg": (
            sum(volume_discounts) / len(volume_discounts)
            if volume_discounts
            else 0.0
        ),
        "edge_reduction_count": reduced_count,
        "edge_reduction_rate": (
            reduced_count / len(edge_reductions)
            if edge_reductions
            else 0.0
        ),
        "kelly_fraction_count": len(kelly_effective_fractions),
        "kelly_fraction_gt_half": kelly_gt_half,
        "kelly_fraction_gt_half_rate": (
            kelly_gt_half / len(kelly_effective_fractions)
            if kelly_effective_fractions
            else 0.0
        ),
    }


def _p(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(q * (len(ordered) - 1))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print daily threshold tuning recommendations from calibration logs."
    )
    parser.add_argument(
        "--log-file",
        default="logs/predictbot.log",
        help="Path to predictbot JSON log file.",
    )
    parser.add_argument(
        "--current-edge-threshold",
        type=float,
        default=0.05,
        help="Current market-edge threshold in use.",
    )
    parser.add_argument(
        "--current-spread-cutoff",
        type=float,
        default=0.08,
        help="Current orderbook spread cutoff in use.",
    )
    parser.add_argument(
        "--current-workers",
        type=int,
        default=2,
        help="Current analysis worker default.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=20,
        help="Minimum sample count before adaptive updates are considered reliable.",
    )
    args = parser.parse_args()

    log_base = Path(args.log_file)
    log_files = sorted(log_base.parent.glob(log_base.name + "*"))
    rows: list[dict[str, Any]] = []
    for log_path in log_files:
        rows.extend(_parse_json_lines(log_path))
    samples = _extract_calibration_samples(rows)
    recommendation = compute_adaptive_thresholds(
        samples=samples,
        current_edge_threshold=args.current_edge_threshold,
        current_spread_cutoff=args.current_spread_cutoff,
        current_workers=args.current_workers,
        min_samples=args.min_samples,
    )
    grok_durations = _extract_grok_durations(rows)
    profit_metrics = _extract_profit_lever_metrics(rows)

    print("Daily Tuning Recommendations")
    print("============================")
    scanned = [str(f.name) for f in log_files]
    print(f"log_files_scanned: {scanned}")
    print(f"calibration_samples: {recommendation['sample_count']}")
    print(
        "recommended_min_market_edge_for_trade: "
        f"{recommendation['recommended_min_market_edge_for_trade']:.4f}"
    )
    print(
        "recommended_orderbook_spread_cutoff: "
        f"{recommendation['recommended_orderbook_spread_cutoff']:.4f}"
    )
    print(
        "recommended_analysis_max_workers: "
        f"{recommendation['recommended_analysis_max_workers']}"
    )
    print(
        "insufficient_edge_data: "
        f"{recommendation['insufficient_edge_data']}"
    )
    print(
        "insufficient_spread_data: "
        f"{recommendation['insufficient_spread_data']}"
    )
    if grok_durations:
        print(f"grok_duration_p50_ms: {_p(grok_durations, 0.50):.2f}")
        print(f"grok_duration_p90_ms: {_p(grok_durations, 0.90):.2f}")
        print(f"grok_duration_max_ms: {max(grok_durations):.2f}")
    print(
        "score_volume_amplifier_discount_avg: "
        f"{profit_metrics['volume_discount_avg']:.4f} "
        f"(n={profit_metrics['volume_discount_count']})"
    )
    print(
        "dynamic_edge_reductions_applied: "
        f"{profit_metrics['edge_reduction_count']} "
        f"({profit_metrics['edge_reduction_rate'] * 100:.1f}%)"
    )
    print(
        "kelly_effective_fraction_gt_0_50: "
        f"{profit_metrics['kelly_fraction_gt_half']}/"
        f"{profit_metrics['kelly_fraction_count']} "
        f"({profit_metrics['kelly_fraction_gt_half_rate'] * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
