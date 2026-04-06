from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

_EPSILON = 1e-12


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_log_value(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Log value must be finite.")
    return float(value)


def _validate_probability(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Probability must be finite.")
    clamped = max(_EPSILON, min(1.0, float(value)))
    return clamped


@dataclass(frozen=True)
class BayesianState:
    log_prior: float
    log_likelihoods: list[float] = field(default_factory=list)
    update_count: int = 0
    last_updated: str = field(default_factory=_utc_now_iso)

    @property
    def log_posterior_unnormalized(self) -> float:
        return self.log_prior + sum(self.log_likelihoods)


def initial_state(
    n_outcomes: int,
    prior: float | list[float] | tuple[float, ...] | None = None,
) -> list[BayesianState]:
    """Create initial per-outcome Bayesian states in log-space."""
    if n_outcomes <= 0:
        raise ValueError("n_outcomes must be > 0.")
    if prior is None:
        probability = 1.0 / n_outcomes
        priors = [probability] * n_outcomes
    elif isinstance(prior, (list, tuple)):
        if len(prior) != n_outcomes:
            raise ValueError("Prior length must match n_outcomes.")
        priors = [float(value) for value in prior]
        total = sum(priors)
        if total <= 0:
            raise ValueError("Prior sum must be > 0.")
        priors = [value / total for value in priors]
    else:
        single_prior = _validate_probability(float(prior))
        if n_outcomes != 2:
            raise ValueError("Scalar prior is supported only for binary outcomes.")
        priors = [single_prior, 1.0 - single_prior]

    states: list[BayesianState] = []
    timestamp = _utc_now_iso()
    for probability in priors:
        states.append(
            BayesianState(
                log_prior=math.log(_validate_probability(probability)),
                log_likelihoods=[],
                update_count=0,
                last_updated=timestamp,
            )
        )
    return states


def update(state: BayesianState, log_likelihood: float) -> BayesianState:
    """Return a new state with an appended log-likelihood update."""
    log_likelihood_value = _validate_log_value(log_likelihood)
    updated_likelihoods = list(state.log_likelihoods)
    updated_likelihoods.append(log_likelihood_value)
    return BayesianState(
        log_prior=_validate_log_value(state.log_prior),
        log_likelihoods=updated_likelihoods,
        update_count=state.update_count + 1,
        last_updated=_utc_now_iso(),
    )


def compute_posterior(log_posteriors: list[float]) -> list[float]:
    """Convert unnormalized log-posteriors into normalized probabilities."""
    if not log_posteriors:
        raise ValueError("log_posteriors must be non-empty.")
    values = [_validate_log_value(value) for value in log_posteriors]
    max_value = max(values)
    exp_shifted = [math.exp(value - max_value) for value in values]
    total = sum(exp_shifted)
    if total <= 0:
        raise ValueError("Posterior normalization failed.")
    return [value / total for value in exp_shifted]


def posterior_from_state(states: list[BayesianState]) -> list[float]:
    """Compute normalized posterior from per-outcome Bayesian states."""
    if not states:
        raise ValueError("states must be non-empty.")
    log_posteriors = [state.log_posterior_unnormalized for state in states]
    return compute_posterior(log_posteriors)


def log_likelihood_from_ratio(likelihood_ratio: float) -> float:
    """Convert likelihood ratio to log-likelihood for binary updates."""
    ratio = float(likelihood_ratio)
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("likelihood_ratio must be a positive finite value.")
    return math.log(ratio)
