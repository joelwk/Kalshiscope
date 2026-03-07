from __future__ import annotations

import math

from kelly import fractional_kelly, kelly_bet_pct, kelly_fraction


def test_kelly_fraction_zero_when_no_edge() -> None:
    assert math.isclose(kelly_fraction(0.55, 0.55), 0.0, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(kelly_fraction(0.50, 0.60), 0.0, rel_tol=1e-12, abs_tol=1e-12)


def test_kelly_fraction_positive_with_edge() -> None:
    value = kelly_fraction(0.65, 0.55)
    assert value > 0.0
    assert value <= 1.0


def test_fractional_kelly_scales_raw_kelly() -> None:
    raw = kelly_fraction(0.67, 0.52)
    quarter = fractional_kelly(0.67, 0.52, 0.25)
    assert math.isclose(quarter, raw * 0.25, rel_tol=1e-12, abs_tol=1e-12)


def test_kelly_bet_pct_respects_min_edge_gate() -> None:
    bet_pct = kelly_bet_pct(
        posterior=0.58,
        market_price=0.55,
        fraction=0.25,
        min_edge=0.05,
    )
    assert math.isclose(bet_pct, 0.0, rel_tol=1e-12, abs_tol=1e-12)


def test_kelly_bet_pct_in_range() -> None:
    bet_pct = kelly_bet_pct(
        posterior=0.75,
        market_price=0.55,
        fraction=0.25,
        min_edge=0.05,
    )
    assert 0.0 < bet_pct <= 1.0
