from __future__ import annotations

import math

from lmsr import (
    infer_quantities_from_prices,
    inefficiency_signal,
    lmsr_cost,
    lmsr_prices,
    price_impact,
    trade_cost,
)


def test_lmsr_prices_sum_to_one() -> None:
    prices = lmsr_prices([10.0, -5.0, 0.0], b=1000.0)
    assert len(prices) == 3
    assert math.isclose(sum(prices), 1.0, rel_tol=1e-12, abs_tol=1e-12)
    assert all(0.0 < price < 1.0 for price in prices)


def test_lmsr_cost_increases_with_buy_quantity() -> None:
    base_quantities = [0.0, 0.0]
    before = lmsr_cost(base_quantities, b=1000.0)
    after = lmsr_cost([50.0, 0.0], b=1000.0)
    assert after > before


def test_trade_cost_positive_for_positive_delta() -> None:
    cost = trade_cost([0.0, 0.0], outcome_idx=0, delta=25.0, b=1000.0)
    assert cost > 0.0


def test_price_impact_sign_matches_trade_direction() -> None:
    upward = price_impact([0.0, 0.0], outcome_idx=0, delta=20.0, b=1000.0)
    downward = price_impact([0.0, 0.0], outcome_idx=0, delta=-20.0, b=1000.0)
    assert upward > 0.0
    assert downward < 0.0


def test_quantity_inference_round_trip() -> None:
    observed_prices = [0.62, 0.38]
    inferred_quantities = infer_quantities_from_prices(observed_prices, b=1000.0)
    reconstructed = lmsr_prices(inferred_quantities, b=1000.0)
    assert math.isclose(reconstructed[0], observed_prices[0], rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(reconstructed[1], observed_prices[1], rel_tol=1e-6, abs_tol=1e-6)


def test_inefficiency_signal_uses_probability_delta() -> None:
    signal = inefficiency_signal(posterior=0.67, market_price=0.59)
    assert math.isclose(signal, 0.08, rel_tol=1e-9, abs_tol=1e-9)
