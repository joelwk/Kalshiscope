from __future__ import annotations

import math

_EPSILON = 1e-12


def _validate_liquidity_param(b: float) -> float:
    if not math.isfinite(b) or b <= 0:
        raise ValueError("Liquidity parameter b must be a positive finite number.")
    return float(b)


def _validate_quantities(quantities: list[float]) -> list[float]:
    if not quantities:
        raise ValueError("Quantities must be non-empty.")
    values = [float(q) for q in quantities]
    if any(not math.isfinite(q) for q in values):
        raise ValueError("Quantities must be finite values.")
    return values


def _validate_probabilities(prices: list[float]) -> list[float]:
    if not prices:
        raise ValueError("Prices must be non-empty.")
    probs = [float(price) for price in prices]
    if any(not math.isfinite(price) for price in probs):
        raise ValueError("Prices must be finite values.")
    if any(price <= 0 for price in probs):
        raise ValueError("Prices must all be positive.")
    total = sum(probs)
    if total <= 0:
        raise ValueError("Price sum must be positive.")
    return [price / total for price in probs]


def _validate_outcome_index(quantities: list[float], outcome_idx: int) -> int:
    if outcome_idx < 0 or outcome_idx >= len(quantities):
        raise IndexError("Outcome index out of bounds.")
    return outcome_idx


def lmsr_cost(quantities: list[float], b: float) -> float:
    """Compute LMSR cost C(q) = b * log(sum_i exp(q_i / b))."""
    values = _validate_quantities(quantities)
    liquidity_param = _validate_liquidity_param(b)
    scaled = [value / liquidity_param for value in values]
    max_scaled = max(scaled)
    exp_shifted = [math.exp(value - max_scaled) for value in scaled]
    return liquidity_param * (max_scaled + math.log(sum(exp_shifted)))


def lmsr_prices(quantities: list[float], b: float) -> list[float]:
    """Compute LMSR outcome prices as softmax(q / b)."""
    values = _validate_quantities(quantities)
    liquidity_param = _validate_liquidity_param(b)
    scaled = [value / liquidity_param for value in values]
    max_scaled = max(scaled)
    exp_shifted = [math.exp(value - max_scaled) for value in scaled]
    total = sum(exp_shifted)
    if total <= 0:
        raise ValueError("Invalid LMSR price computation.")
    return [value / total for value in exp_shifted]


def infer_quantities_from_prices(prices: list[float], b: float) -> list[float]:
    """Infer a canonical quantity vector from prices, up to additive constant."""
    normalized_prices = _validate_probabilities(prices)
    liquidity_param = _validate_liquidity_param(b)
    # Canonical representative of the equivalence class with translation fixed at 0.
    return [liquidity_param * math.log(max(price, _EPSILON)) for price in normalized_prices]


def trade_cost(quantities: list[float], outcome_idx: int, delta: float, b: float) -> float:
    """Compute LMSR cost change for buying delta of outcome_idx."""
    values = _validate_quantities(quantities)
    index = _validate_outcome_index(values, outcome_idx)
    if not math.isfinite(delta):
        raise ValueError("Delta must be a finite value.")
    before = lmsr_cost(values, b)
    updated = list(values)
    updated[index] += float(delta)
    after = lmsr_cost(updated, b)
    return after - before


def inefficiency_signal(posterior: float, market_price: float) -> float:
    """Return posterior - market_price with finite-value validation."""
    posterior_value = float(posterior)
    market_price_value = float(market_price)
    if not math.isfinite(posterior_value) or not math.isfinite(market_price_value):
        raise ValueError("Posterior and market price must be finite values.")
    posterior_clamped = max(0.0, min(1.0, posterior_value))
    market_clamped = max(0.0, min(1.0, market_price_value))
    return posterior_clamped - market_clamped


def price_impact(quantities: list[float], outcome_idx: int, delta: float, b: float) -> float:
    """Return own-outcome price change after a trade quantity delta."""
    values = _validate_quantities(quantities)
    index = _validate_outcome_index(values, outcome_idx)
    before_prices = lmsr_prices(values, b)
    updated = list(values)
    updated[index] += float(delta)
    after_prices = lmsr_prices(updated, b)
    return after_prices[index] - before_prices[index]
