from __future__ import annotations

from pydantic import ValidationError

from config import Settings
from main import _applied_bayesian_posterior, _compute_lmsr_execution_price_for_outcome
from market_state import MarketStateManager
from models import Market, MarketOutcome, TradeDecision


def test_applied_bayesian_posterior_respects_min_updates() -> None:
    assert _applied_bayesian_posterior(0.62, bayesian_update_count=0, min_updates_for_trade=2) is None
    assert _applied_bayesian_posterior(0.62, bayesian_update_count=1, min_updates_for_trade=2) is None
    assert _applied_bayesian_posterior(0.62, bayesian_update_count=2, min_updates_for_trade=2) == 0.62


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
