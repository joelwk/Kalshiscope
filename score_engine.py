from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from models import Market, TradeDecision


@dataclass(frozen=True)
class ScoreResult:
    final_score: float
    edge_market: float
    edge_external: float
    evidence_quality: float
    liquidity_penalty: float
    staleness_penalty: float
    evidence_component: float
    bayesian_component: float
    inefficiency_component: float
    kelly_component: float
    confidence_alignment_bonus: float
    weather_uncertainty_penalty: float = 0.0
    bayesian_posterior: float | None = None
    lmsr_price: float | None = None
    inefficiency_signal: float | None = None
    kelly_raw: float | None = None


def compute_final_score(
    market: Market,
    decision: TradeDecision,
    implied_prob_market: float | None,
    bayesian_posterior: float | None = None,
    lmsr_price: float | None = None,
    inefficiency_signal: float | None = None,
    kelly_raw: float | None = None,
    is_weather_market: bool = False,
    weather_score_penalty: float = 0.0,
    now: datetime | None = None,
) -> ScoreResult:
    now = now or datetime.now(timezone.utc)
    edge_market = 0.0
    if implied_prob_market is not None:
        edge_market = decision.confidence - implied_prob_market

    edge_external = decision.edge_external or 0.0
    evidence_quality = max(0.0, min(1.0, decision.evidence_quality))

    liquidity = market.liquidity_usdc or 0.0
    # Penalize thin markets; no penalty above $500.
    liquidity_penalty = max(0.0, min(0.20, (500.0 - liquidity) / 5000.0))

    staleness_penalty = 0.0
    if market.close_time:
        close_time = market.close_time
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=timezone.utc)
        if close_time - now > timedelta(days=7):
            staleness_penalty = 0.05
    else:
        staleness_penalty = 0.03

    evidence_multiplier = 0.5 + (0.5 * evidence_quality)
    weighted_edge = (0.50 * edge_market) + (0.35 * edge_external)
    evidence_component = 0.15 * evidence_quality
    bayesian_component = 0.0
    if bayesian_posterior is not None:
        bayesian_component = 0.05 * (bayesian_posterior - 0.5)
    inefficiency_component = 0.0
    if inefficiency_signal is not None:
        inefficiency_component = 0.05 * abs(inefficiency_signal)
    kelly_component = 0.0
    if kelly_raw is not None:
        kelly_component = 0.10 * max(0.0, min(1.0, kelly_raw))
    confidence_alignment_bonus = 0.0
    if (
        bayesian_posterior is not None
        and kelly_raw is not None
        and edge_market > 0
        and bayesian_posterior > 0.5
        and kelly_raw > 0
    ):
        confidence_alignment_bonus = 0.03

    weather_uncertainty_penalty = 0.0
    if is_weather_market:
        penalty_scale = 1.0
        if market.close_time is None:
            penalty_scale = 1.25
        else:
            close_time = market.close_time
            if close_time.tzinfo is None:
                close_time = close_time.replace(tzinfo=timezone.utc)
            hours_to_close = (close_time - now).total_seconds() / 3600.0
            if hours_to_close > 168:
                penalty_scale = 2.0
            elif hours_to_close > 72:
                penalty_scale = 1.5
        weather_uncertainty_penalty = max(0.0, weather_score_penalty) * penalty_scale

    final_score = (
        (evidence_multiplier * weighted_edge)
        + evidence_component
        + bayesian_component
        + inefficiency_component
        + kelly_component
        + confidence_alignment_bonus
        - liquidity_penalty
        - staleness_penalty
        - weather_uncertainty_penalty
    )

    return ScoreResult(
        final_score=final_score,
        edge_market=edge_market,
        edge_external=edge_external,
        evidence_quality=evidence_quality,
        liquidity_penalty=liquidity_penalty,
        staleness_penalty=staleness_penalty,
        evidence_component=evidence_component,
        bayesian_component=bayesian_component,
        inefficiency_component=inefficiency_component,
        kelly_component=kelly_component,
        confidence_alignment_bonus=confidence_alignment_bonus,
        weather_uncertainty_penalty=weather_uncertainty_penalty,
        bayesian_posterior=bayesian_posterior,
        lmsr_price=lmsr_price,
        inefficiency_signal=inefficiency_signal,
        kelly_raw=kelly_raw,
    )

