from __future__ import annotations

import math

from bayesian_engine import (
    BayesianState,
    compute_posterior,
    initial_state,
    log_likelihood_from_ratio,
    posterior_from_state,
    update,
)


def test_initial_state_binary_scalar_prior() -> None:
    states = initial_state(2, prior=0.65)
    posterior = posterior_from_state(states)
    assert math.isclose(posterior[0], 0.65, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(posterior[1], 0.35, rel_tol=1e-9, abs_tol=1e-9)


def test_log_likelihood_ratio_neutral_is_zero() -> None:
    assert math.isclose(log_likelihood_from_ratio(1.0), 0.0, rel_tol=1e-12, abs_tol=1e-12)


def test_update_increases_update_count() -> None:
    state = BayesianState(log_prior=math.log(0.5))
    updated = update(state, math.log(2.0))
    assert updated.update_count == state.update_count + 1
    assert len(updated.log_likelihoods) == 1


def test_neutral_likelihood_preserves_posterior() -> None:
    states = initial_state(2, prior=0.5)
    updated_states = [update(states[0], 0.0), update(states[1], 0.0)]
    posterior = posterior_from_state(updated_states)
    assert math.isclose(posterior[0], 0.5, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(posterior[1], 0.5, rel_tol=1e-9, abs_tol=1e-9)


def test_positive_likelihood_moves_posterior_toward_selected_outcome() -> None:
    states = initial_state(2, prior=0.5)
    log_lr = log_likelihood_from_ratio(2.0)
    updated_states = [update(states[0], log_lr), update(states[1], -log_lr)]
    posterior = posterior_from_state(updated_states)
    assert posterior[0] > 0.5
    assert posterior[1] < 0.5
    assert math.isclose(sum(posterior), 1.0, rel_tol=1e-12, abs_tol=1e-12)


def test_compute_posterior_stable_for_large_log_values() -> None:
    posterior = compute_posterior([10_000.0, 9_999.0, 9_998.0])
    assert len(posterior) == 3
    assert math.isclose(sum(posterior), 1.0, rel_tol=1e-12, abs_tol=1e-12)
