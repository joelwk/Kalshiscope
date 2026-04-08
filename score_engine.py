from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from models import Market, TradeDecision

_WEATHER_BIN_TICKER_PATTERN = re.compile(r"-B\d", re.IGNORECASE)
_NARROW_WEATHER_BIN_PENALTY = 0.03
_MENTION_MARKET_TICKER_PATTERN = re.compile(r"MENTION", re.IGNORECASE)


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
    observed_data_bonus: float = 0.0
    low_information_penalty: float = 0.0
    no_external_odds_penalty: float = 0.0
    repeated_analysis_penalty: float = 0.0
    mention_market_penalty: float = 0.0
    confidence_calibration_penalty: float = 0.0
    weather_uncertainty_penalty: float = 0.0
    weather_bin_penalty: float = 0.0
    rejection_reasons: tuple[str, ...] = ()
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
    low_info_penalty_threshold: float = 0.55,
    low_info_penalty_base: float = 0.05,
    repeated_analysis_count: int = 0,
    repeated_analysis_penalty_base: float = 0.05,
    repeated_analysis_penalty_start_count: int = 1,
    mention_market_penalty_base: float = 0.0,
    confidence_calibration_floor: float = 0.50,
    confidence_calibration_penalty_scale: float = 0.0,
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
    observed_data_bonus = 0.0
    if evidence_quality >= 0.80 and edge_market > 0.10:
        observed_data_bonus = 0.05
    low_information_penalty = 0.0
    low_info_threshold = max(0.0, min(1.0, low_info_penalty_threshold))
    low_info_base = max(0.0, low_info_penalty_base)
    edge_source = (decision.edge_source or "").strip().lower()
    if evidence_quality < low_info_threshold and edge_source in {"fallback", "none"}:
        low_info_shortfall = low_info_threshold - evidence_quality
        low_information_penalty = low_info_base + (low_info_base * low_info_shortfall)
    no_external_odds_penalty = 0.0
    if edge_source in {"fallback", "none"} and decision.implied_prob_external is None:
        no_external_odds_penalty = 0.02
    repeated_analysis_penalty = 0.0
    repeated_penalty_start = max(0, int(repeated_analysis_penalty_start_count))
    if repeated_analysis_count > repeated_penalty_start:
        repeated_analysis_penalty = max(
            0.0,
            (repeated_analysis_count - repeated_penalty_start)
            * max(0.0, repeated_analysis_penalty_base),
        )
    mention_market_penalty = 0.0
    if _is_mention_market(market):
        mention_market_penalty = max(0.0, mention_market_penalty_base)
    confidence_calibration_penalty = 0.0
    normalized_conf_floor = max(0.0, min(1.0, confidence_calibration_floor))
    if decision.confidence < normalized_conf_floor:
        confidence_shortfall = normalized_conf_floor - max(0.0, decision.confidence)
        confidence_calibration_penalty = (
            confidence_shortfall * max(0.0, confidence_calibration_penalty_scale)
        )

    weather_uncertainty_penalty = 0.0
    weather_bin_penalty = 0.0
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
        weather_bin_penalty = _weather_bin_penalty(market)

    final_score = (
        (evidence_multiplier * weighted_edge)
        + evidence_component
        + bayesian_component
        + inefficiency_component
        + kelly_component
        + confidence_alignment_bonus
        + observed_data_bonus
        - low_information_penalty
        - no_external_odds_penalty
        - repeated_analysis_penalty
        - mention_market_penalty
        - confidence_calibration_penalty
        - liquidity_penalty
        - staleness_penalty
        - weather_uncertainty_penalty
        - weather_bin_penalty
    )

    rejection_reasons: list[str] = []
    if edge_market <= 0:
        rejection_reasons.append("non_positive_market_edge")
    if evidence_quality < low_info_threshold:
        rejection_reasons.append("low_evidence_quality")
    if low_information_penalty > 0:
        rejection_reasons.append("low_information_penalty")
    if no_external_odds_penalty > 0:
        rejection_reasons.append("no_external_odds_penalty")
    if repeated_analysis_penalty > 0:
        rejection_reasons.append("repeated_analysis_penalty")
    if mention_market_penalty > 0:
        rejection_reasons.append("mention_market_penalty")
    if confidence_calibration_penalty > 0:
        rejection_reasons.append("confidence_calibration_penalty")
    if liquidity_penalty > 0:
        rejection_reasons.append("thin_liquidity_penalty")
    if staleness_penalty > 0:
        rejection_reasons.append("staleness_penalty")
    if weather_uncertainty_penalty > 0:
        rejection_reasons.append("weather_uncertainty_penalty")
    if weather_bin_penalty > 0:
        rejection_reasons.append("weather_bin_penalty")

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
        observed_data_bonus=observed_data_bonus,
        low_information_penalty=low_information_penalty,
        no_external_odds_penalty=no_external_odds_penalty,
        repeated_analysis_penalty=repeated_analysis_penalty,
        mention_market_penalty=mention_market_penalty,
        confidence_calibration_penalty=confidence_calibration_penalty,
        weather_uncertainty_penalty=weather_uncertainty_penalty,
        weather_bin_penalty=weather_bin_penalty,
        rejection_reasons=tuple(rejection_reasons),
        bayesian_posterior=bayesian_posterior,
        lmsr_price=lmsr_price,
        inefficiency_signal=inefficiency_signal,
        kelly_raw=kelly_raw,
    )


def _weather_bin_penalty(market: Market) -> float:
    market_id = (market.id or "").strip()
    if _WEATHER_BIN_TICKER_PATTERN.search(market_id):
        return _NARROW_WEATHER_BIN_PENALTY
    return 0.0


def _is_mention_market(market: Market) -> bool:
    market_id = (market.id or "").strip()
    return bool(_MENTION_MARKET_TICKER_PATTERN.search(market_id))

