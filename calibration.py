from __future__ import annotations

from typing import Any

_EDGE_MIN = 0.03
_EDGE_MAX = 0.07
_SPREAD_MIN = 0.06
_SPREAD_MAX = 0.12
_DEFAULT_EDGE_THRESHOLDS = (0.03, 0.04, 0.05, 0.06)


def build_counterfactual_flags(
    edge_market: float | None,
    thresholds: tuple[float, ...] = _DEFAULT_EDGE_THRESHOLDS,
) -> dict[str, bool | None]:
    """Return would-trade flags for multiple market-edge thresholds."""
    if edge_market is None:
        return {f"would_trade_at_{int(t * 100):02d}": None for t in thresholds}
    return {
        f"would_trade_at_{int(t * 100):02d}": edge_market >= t
        for t in thresholds
    }


def compute_adaptive_thresholds(
    samples: list[dict[str, Any]],
    current_edge_threshold: float,
    current_spread_cutoff: float,
    current_workers: int,
    min_samples: int = 20,
) -> dict[str, Any]:
    """Compute conservative threshold recommendations from calibration samples."""
    market_edges = [
        float(sample["edge_market"])
        for sample in samples
        if sample.get("edge_market") is not None
        and sample.get("evidence_quality", 0.0) >= 0.45
    ]
    spread_values = [
        float(sample["orderbook_spread_abs"])
        for sample in samples
        if sample.get("orderbook_spread_abs") is not None
    ]
    analysis_durations = [
        float(sample["analysis_duration_ms"])
        for sample in samples
        if sample.get("analysis_duration_ms") is not None
    ]

    has_min_edge_data = len(market_edges) >= min_samples
    has_min_spread_data = len(spread_values) >= min_samples

    recommended_edge = current_edge_threshold
    if has_min_edge_data:
        edge_quantile = _quantile(market_edges, 0.60)
        recommended_edge = _clamp(edge_quantile, _EDGE_MIN, _EDGE_MAX)

    recommended_spread_cutoff = current_spread_cutoff
    if has_min_spread_data:
        spread_quantile = _quantile(spread_values, 0.75)
        recommended_spread_cutoff = _clamp(spread_quantile, _SPREAD_MIN, _SPREAD_MAX)

    recommended_workers = current_workers
    if len(analysis_durations) >= min_samples:
        p90_duration = _quantile(analysis_durations, 0.90)
        if p90_duration >= 60000:
            recommended_workers = 2
        elif p90_duration <= 45000:
            recommended_workers = max(2, min(4, current_workers + 1))
        else:
            recommended_workers = max(2, min(4, current_workers))

    return {
        "sample_count": len(samples),
        "edge_sample_count": len(market_edges),
        "spread_sample_count": len(spread_values),
        "duration_sample_count": len(analysis_durations),
        "recommended_min_market_edge_for_trade": round(recommended_edge, 4),
        "recommended_orderbook_spread_cutoff": round(recommended_spread_cutoff, 4),
        "recommended_analysis_max_workers": int(recommended_workers),
        "insufficient_edge_data": not has_min_edge_data,
        "insufficient_spread_data": not has_min_spread_data,
    }


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Quantile requires non-empty values")
    ordered = sorted(values)
    index = int(q * (len(ordered) - 1))
    return ordered[index]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
