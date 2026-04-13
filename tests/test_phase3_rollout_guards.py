from __future__ import annotations

from pydantic import ValidationError

from config import Settings
from main import (
    _applied_bayesian_posterior,
    _cap_bayesian_confidence_boost,
    _compute_lmsr_execution_price_for_outcome,
    _resolve_min_bet_floor,
    _should_skip_flip_refinement,
)
from market_state import MarketStateManager
from models import Market, MarketOutcome, TradeDecision


def test_applied_bayesian_posterior_respects_min_updates() -> None:
    assert _applied_bayesian_posterior(0.62, bayesian_update_count=0, min_updates_for_trade=2) is None
    assert _applied_bayesian_posterior(0.62, bayesian_update_count=1, min_updates_for_trade=2) is None
    assert _applied_bayesian_posterior(0.62, bayesian_update_count=2, min_updates_for_trade=2) == 0.62


def test_cap_bayesian_confidence_boost_limits_uplift() -> None:
    capped = _cap_bayesian_confidence_boost(
        base_confidence=0.53,
        candidate_confidence=0.97,
        max_boost=0.15,
    )
    assert capped == 0.68


def test_lmsr_execution_price_increases_with_trade_size() -> None:
    market = Market(
        id="m1",
        question="Test market",
        outcomes=[
            MarketOutcome(name="YES", price=0.60),
            MarketOutcome(name="NO", price=0.40),
        ],
    )
    settings = Settings(LMSR_LIQUIDITY_PARAM_B=1_000.0)
    small_trade_price = _compute_lmsr_execution_price_for_outcome(
        market=market,
        decision_outcome="YES",
        amount_usdc=1.0,
        settings=settings,
    )
    larger_trade_price = _compute_lmsr_execution_price_for_outcome(
        market=market,
        decision_outcome="YES",
        amount_usdc=3.0,
        settings=settings,
    )
    assert small_trade_price is not None
    assert larger_trade_price is not None
    assert small_trade_price > 0.60
    assert larger_trade_price > small_trade_price


def test_seed_bayesian_state_does_not_increment_update_count(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        manager.update_bayesian_state(
            market_id="m1",
            outcome="YES",
            log_prior=-0.6931471805599453,  # log(0.5)
            log_likelihood=0.0,
            count_as_update=False,
        )
        state = manager.get_bayesian_state("m1")["YES"]
        assert state.update_count == 0
        assert state.log_likelihoods == []

        manager.update_bayesian_state(
            market_id="m1",
            outcome="YES",
            log_prior=-0.6931471805599453,
            log_likelihood=0.2,
            count_as_update=True,
        )
        updated = manager.get_bayesian_state("m1")["YES"]
        assert updated.update_count == 1
    finally:
        manager.close()


def test_trade_decision_rejects_zero_likelihood_ratio() -> None:
    try:
        TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.6,
            bet_size_pct=0.1,
            reasoning="test",
            likelihood_ratio=0.0,
        )
    except ValidationError:
        return
    raise AssertionError("Expected likelihood_ratio=0.0 to fail validation")


def test_resolve_min_bet_floor_skips_sub_floor_kelly_by_default() -> None:
    adjusted, adjusted_pct, floor_applied, sub_floor_skipped, policy = _resolve_min_bet_floor(
        bet_amount=1.25,
        min_bet_usdc=2.0,
        max_bet_usdc=4.0,
        kelly_path_active=True,
        min_bet_policy="skip",
        edge_scaling_bet_pct=0.60,
    )
    assert adjusted == 1.25
    assert adjusted_pct == 0.3125
    assert floor_applied is False
    assert sub_floor_skipped is True
    assert policy == "skip"


def test_resolve_min_bet_floor_applies_floor_policy() -> None:
    adjusted, adjusted_pct, floor_applied, sub_floor_skipped, policy = _resolve_min_bet_floor(
        bet_amount=1.25,
        min_bet_usdc=2.0,
        max_bet_usdc=4.0,
        kelly_path_active=True,
        min_bet_policy="floor",
        edge_scaling_bet_pct=0.20,
    )
    assert adjusted == 2.0
    assert adjusted_pct == 0.5
    assert floor_applied is True
    assert sub_floor_skipped is False
    assert policy == "floor"


def test_resolve_min_bet_floor_fallback_edge_scaling_uses_edge_size() -> None:
    adjusted, adjusted_pct, floor_applied, sub_floor_skipped, policy = _resolve_min_bet_floor(
        bet_amount=1.25,
        min_bet_usdc=2.0,
        max_bet_usdc=8.0,
        kelly_path_active=True,
        min_bet_policy="fallback_edge_scaling",
        edge_scaling_bet_pct=0.40,
    )
    assert adjusted == 3.2
    assert adjusted_pct == 0.4
    assert floor_applied is False
    assert sub_floor_skipped is False
    assert policy == "fallback_edge_scaling"


def test_flip_refinement_precheck_blocks_unreachable_conf_gain() -> None:
    settings = Settings()
    market = Market(
        id="m2",
        question="Will team A win?",
        outcomes=[
            MarketOutcome(name="YES", price=0.60),
            MarketOutcome(name="NO", price=0.40),
        ],
    )
    decision = TradeDecision(
        should_trade=True,
        outcome="NO",
        confidence=0.62,
        bet_size_pct=0.2,
        reasoning="flip candidate",
        evidence_quality=0.8,
    )
    should_skip, reason, payload = _should_skip_flip_refinement(
        market=market,
        decision=decision,
        anchor_analysis={"outcome": "YES", "confidence": 0.97},
        settings=settings,
    )
    assert should_skip is True
    assert reason is not None
    assert "conf_gain_unreachable" in reason
    assert payload is not None
