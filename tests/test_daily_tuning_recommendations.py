from __future__ import annotations

from scripts.daily_tuning_recommendations import _extract_profit_lever_metrics


def test_extract_profit_lever_metrics_from_structured_logs() -> None:
    rows = [
        {
            "data": {
                "score_volume_amplifier_discount": 0.02,
                "gate_edge_dynamic_reduction": 0.03,
                "kelly_effective_fraction": 0.55,
            }
        },
        {
            "data": {
                "score_breakdown": {
                    "score_volume_amplifier_discount": 0.00,
                },
                "gate_edge_dynamic_reduction": 0.00,
                "kelly_effective_fraction": 0.45,
            }
        },
    ]

    metrics = _extract_profit_lever_metrics(rows)

    assert metrics["volume_discount_count"] == 2
    assert metrics["volume_discount_avg"] == 0.01
    assert metrics["edge_reduction_count"] == 1
    assert metrics["edge_reduction_rate"] == 0.5
    assert metrics["kelly_fraction_gt_half"] == 1
    assert metrics["kelly_fraction_gt_half_rate"] == 0.5
