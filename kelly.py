from __future__ import annotations

import math

_MIN_DENOMINATOR = 1e-9


def _validate_probability(value: float) -> float:
    probability = float(value)
    if not math.isfinite(probability):
        raise ValueError("Probability must be finite.")
    return max(0.0, min(1.0, probability))


def _validate_fraction(value: float) -> float:
    fraction = float(value)
    if not math.isfinite(fraction) or fraction < 0:
        raise ValueError("Kelly fraction must be a non-negative finite value.")
    return fraction


def kelly_fraction(posterior: float, market_price: float) -> float:
    """Binary-market Kelly fraction: f* = (p_hat - p) / (1 - p)."""
    posterior_prob = _validate_probability(posterior)
    market_prob = _validate_probability(market_price)
    denominator = max(_MIN_DENOMINATOR, 1.0 - market_prob)
    raw_fraction = (posterior_prob - market_prob) / denominator
    return max(0.0, raw_fraction)


def fractional_kelly(posterior: float, market_price: float, fraction: float) -> float:
    """Apply fractional Kelly multiplier to Kelly optimal fraction."""
    fraction_multiplier = _validate_fraction(fraction)
    return max(0.0, kelly_fraction(posterior, market_price) * fraction_multiplier)


def kelly_bet_pct(
    posterior: float,
    market_price: float,
    fraction: float,
    min_edge: float,
) -> float:
    """Compute final bet size percentage from edge and fractional Kelly."""
    posterior_prob = _validate_probability(posterior)
    market_prob = _validate_probability(market_price)
    if posterior_prob - market_prob < float(min_edge):
        return 0.0
    return max(0.0, min(1.0, fractional_kelly(posterior_prob, market_prob, fraction)))
