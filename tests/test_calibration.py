from __future__ import annotations

from calibration import (
    build_counterfactual_flags,
    compute_adaptive_thresholds,
    historical_confidence_shrink,
)


def test_build_counterfactual_flags_with_edge() -> None:
    flags = build_counterfactual_flags(0.051)
    assert flags["would_trade_at_03"] is True
    assert flags["would_trade_at_04"] is True
    assert flags["would_trade_at_05"] is True
    assert flags["would_trade_at_06"] is False


def test_build_counterfactual_flags_without_edge() -> None:
    flags = build_counterfactual_flags(None)
    assert all(value is None for value in flags.values())


def test_compute_adaptive_thresholds_with_sufficient_samples() -> None:
    samples = []
    for i in range(30):
        samples.append(
            {
                "edge_market": 0.03 + (i * 0.001),
                "evidence_quality": 0.6,
                "analysis_duration_ms": 42000 + i * 100,
                "orderbook_spread_abs": 0.06 + (i * 0.001),
            }
        )
    recommendation = compute_adaptive_thresholds(
        samples=samples,
        current_edge_threshold=0.05,
        current_spread_cutoff=0.08,
        current_workers=2,
        min_samples=20,
    )
    assert recommendation["insufficient_edge_data"] is False
    assert recommendation["insufficient_spread_data"] is False
    assert 0.03 <= recommendation["recommended_min_market_edge_for_trade"] <= 0.07
    assert 0.06 <= recommendation["recommended_orderbook_spread_cutoff"] <= 0.12
    assert recommendation["recommended_analysis_max_workers"] >= 2


def test_compute_adaptive_thresholds_with_insufficient_samples() -> None:
    samples = [{"edge_market": 0.02, "evidence_quality": 0.8}]
    recommendation = compute_adaptive_thresholds(
        samples=samples,
        current_edge_threshold=0.05,
        current_spread_cutoff=0.08,
        current_workers=3,
        min_samples=20,
    )
    assert recommendation["insufficient_edge_data"] is True
    assert recommendation["recommended_min_market_edge_for_trade"] == 0.05
    assert recommendation["recommended_orderbook_spread_cutoff"] == 0.08
    assert recommendation["recommended_analysis_max_workers"] == 3


def test_historical_confidence_shrink_respects_min_samples() -> None:
    buckets = {
        "all": {
            0.8: {
                "sample_size": 10,
                "win_rate": 0.40,
            }
        }
    }
    result = historical_confidence_shrink(
        0.84,
        family="generic",
        calibration_buckets=buckets,
        min_samples=15,
    )
    assert result.applied is False
    assert result.sample_size == 0
    assert result.calibrated_confidence == 0.84


def test_historical_confidence_shrink_is_monotonic_down_only() -> None:
    buckets = {
        "all": {
            0.8: {
                "sample_size": 30,
                "win_rate": 0.50,
            }
        }
    }
    shrunk = historical_confidence_shrink(
        0.84,
        family="generic",
        calibration_buckets=buckets,
        min_samples=15,
    )
    not_raised = historical_confidence_shrink(
        0.84,
        family="generic",
        calibration_buckets={
            "all": {
                0.8: {
                    "sample_size": 30,
                    "win_rate": 0.95,
                }
            }
        },
        min_samples=15,
    )
    assert shrunk.applied is True
    assert shrunk.calibrated_confidence < 0.84
    assert not_raised.applied is False
    assert not_raised.calibrated_confidence == 0.84


def test_historical_confidence_shrink_family_scoped() -> None:
    buckets = {
        "all": {
            0.8: {
                "sample_size": 40,
                "win_rate": 0.65,
            }
        },
        "weather": {
            0.8: {
                "sample_size": 40,
                "win_rate": 0.35,
            }
        },
    }
    weather_result = historical_confidence_shrink(
        0.84,
        family="weather",
        calibration_buckets=buckets,
        min_samples=15,
    )
    generic_result = historical_confidence_shrink(
        0.84,
        family="generic",
        calibration_buckets=buckets,
        min_samples=15,
    )
    assert weather_result.family_used == "weather"
    assert weather_result.calibrated_confidence < generic_result.calibrated_confidence


def test_historical_shrink_with_backfilled_buckets() -> None:
    """When buckets are populated (as after backfill), low-confidence trades shrink."""
    buckets = {
        "all": {
            0.6: {"sample_size": 30, "win_rate": 0.45},
            0.7: {"sample_size": 50, "win_rate": 0.55},
            0.8: {"sample_size": 40, "win_rate": 0.60},
        },
    }
    low_conf = historical_confidence_shrink(
        0.65, family="generic", calibration_buckets=buckets, min_samples=15,
    )
    assert low_conf.applied is True
    assert low_conf.calibrated_confidence < 0.65
    assert low_conf.observed_win_rate == 0.45

    high_conf = historical_confidence_shrink(
        0.82, family="generic", calibration_buckets=buckets, min_samples=15,
    )
    assert high_conf.applied is True
    assert high_conf.calibrated_confidence < 0.82
