from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from models import Market, TradeDecision

_WEATHER_BIN_TICKER_PATTERN = re.compile(r"-B\d", re.IGNORECASE)
_NARROW_WEATHER_BIN_PENALTY = 0.03
_MENTION_MARKET_TICKER_PATTERN = re.compile(r"MENTION", re.IGNORECASE)
_GENERIC_BIN_TICKER_PATTERN = re.compile(r"-B\d", re.IGNORECASE)
_CONFIDENCE_SHRINKAGE_FLOOR = 0.50
_CONFIDENCE_SHRINKAGE_FACTOR = 0.50
_OVERCONFIDENCE_GAP_FREE_BAND = 0.12
_OVERCONFIDENCE_GAP_MAX = 0.20


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
    computed_edge_bonus: float = 0.0
    definitive_outcome_bonus: float = 0.0
    evidence_basis_bonus: float = 0.0
    observed_data_bonus: float = 0.0
    low_information_penalty: float = 0.0
    no_external_odds_penalty: float = 0.0
    repeated_analysis_penalty: float = 0.0
    mention_market_penalty: float = 0.0
    confidence_calibration_penalty: float = 0.0
    weather_uncertainty_penalty: float = 0.0
    weather_bin_penalty: float = 0.0
    generic_bin_penalty: float = 0.0
    ambiguous_resolution_penalty: float = 0.0
    fallback_edge_penalty: float = 0.0
    proxy_evidence_penalty: float = 0.0
    overconfidence_penalty: float = 0.0
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
    non_actionable_streak: int = 0,
    repeated_analysis_penalty_base: float = 0.05,
    repeated_analysis_penalty_start_count: int = 1,
    mention_market_penalty_base: float = 0.0,
    confidence_calibration_floor: float = 0.50,
    confidence_calibration_penalty_scale: float = 0.0,
    fallback_edge_penalty_base: float = 0.04,
    computed_edge_bonus_base: float = 0.03,
    proxy_evidence_penalty_base: float = 0.05,
    overconfidence_penalty_base: float = 0.08,
    generic_bin_penalty_base: float = 0.03,
    ambiguous_resolution_penalty_base: float = 0.06,
    now: datetime | None = None,
    evidence_basis_class: str = "",
    edge_source: str = "",
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
    normalized_evidence_basis = str(evidence_basis_class or "").strip().lower()
    evidence_basis_bonus = 0.0
    if normalized_evidence_basis == "direct":
        if evidence_quality >= 0.80:
            evidence_basis_bonus = 0.08
        elif evidence_quality >= 0.65:
            evidence_basis_bonus = 0.04
    low_information_penalty = 0.0
    low_info_threshold = max(0.0, min(1.0, low_info_penalty_threshold))
    low_info_base = max(0.0, low_info_penalty_base)
    normalized_edge_source = (edge_source or decision.edge_source or "").strip().lower()
    definitive_outcome_bonus = 0.0
    likelihood_ratio = decision.likelihood_ratio
    if (
        normalized_evidence_basis == "direct"
        and normalized_edge_source in {"fallback", "none"}
        and evidence_quality >= 0.55
        and likelihood_ratio is not None
        and likelihood_ratio >= 10.0
    ):
        definitive_outcome_bonus = 0.06
    computed_edge_bonus = 0.0
    if normalized_edge_source == "computed":
        computed_edge_bonus = max(0.0, computed_edge_bonus_base) * (
            0.50 + (0.50 * evidence_quality)
        )
    if evidence_quality < low_info_threshold and normalized_edge_source in {"fallback", "none"}:
        low_info_shortfall = low_info_threshold - evidence_quality
        low_information_penalty = low_info_base + (low_info_base * low_info_shortfall)
    no_external_odds_penalty = 0.0
    if normalized_edge_source in {"fallback", "none"} and decision.implied_prob_external is None:
        no_external_odds_penalty = 0.04
    repeated_analysis_penalty = 0.0
    repeated_penalty_start = max(0, int(repeated_analysis_penalty_start_count))
    if repeated_analysis_count > repeated_penalty_start:
        repeated_analysis_penalty = max(
            0.0,
            (repeated_analysis_count - repeated_penalty_start)
            * max(0.0, repeated_analysis_penalty_base),
        )
    if non_actionable_streak > 3:
        repeated_analysis_penalty += (
            (non_actionable_streak - 3)
            * max(0.0, repeated_analysis_penalty_base)
            * 0.5
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
    overconfidence_penalty = 0.0
    overconfidence_gap = max(
        0.0,
        decision.confidence - evidence_quality - _OVERCONFIDENCE_GAP_FREE_BAND,
    )
    if overconfidence_gap > 0:
        normalized_gap = min(1.0, overconfidence_gap / _OVERCONFIDENCE_GAP_MAX)
        overconfidence_penalty = max(0.0, overconfidence_penalty_base) * normalized_gap

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
        if normalized_evidence_basis == "direct":
            weather_uncertainty_penalty *= 0.25
        weather_bin_penalty = _weather_bin_penalty(market)
    fallback_edge_penalty = 0.0
    proxy_evidence_penalty = 0.0
    generic_bin_penalty = 0.0
    ambiguous_resolution_penalty = 0.0
    if normalized_edge_source in {"fallback", "none"}:
        fallback_edge_penalty = max(0.0, fallback_edge_penalty_base)
        evidence_shortfall = max(0.0, 0.70 - evidence_quality)
        proxy_evidence_penalty = max(0.0, proxy_evidence_penalty_base) * (
            1.0 + evidence_shortfall
        )
        if decision.confidence >= 0.75 and evidence_quality < 0.65:
            proxy_evidence_penalty += 0.06
        if decision.confidence >= 0.70 and evidence_quality < 0.50:
            proxy_evidence_penalty += 0.04
        if normalized_evidence_basis == "direct":
            if evidence_quality >= 0.75:
                fallback_edge_penalty *= 0.10
                proxy_evidence_penalty *= 0.10
            elif evidence_quality >= 0.55:
                fallback_edge_penalty *= 0.25
                proxy_evidence_penalty *= 0.25
            else:
                fallback_edge_penalty *= 0.40
                proxy_evidence_penalty *= 0.40
    if _GENERIC_BIN_TICKER_PATTERN.search((market.id or "").strip()) and not _is_weather_market(market):
        generic_bin_penalty = max(0.0, generic_bin_penalty_base) * (1.0 + max(0.0, 0.65 - evidence_quality))
    if not (getattr(market, "resolution_criteria", "") or "").strip():
        ambiguous_resolution_penalty = max(0.0, ambiguous_resolution_penalty_base)

    final_score = (
        (evidence_multiplier * weighted_edge)
        + evidence_component
        + bayesian_component
        + inefficiency_component
        + kelly_component
        + confidence_alignment_bonus
        + computed_edge_bonus
        + definitive_outcome_bonus
        + evidence_basis_bonus
        + observed_data_bonus
        - low_information_penalty
        - no_external_odds_penalty
        - repeated_analysis_penalty
        - mention_market_penalty
        - confidence_calibration_penalty
        - overconfidence_penalty
        - fallback_edge_penalty
        - proxy_evidence_penalty
        - liquidity_penalty
        - staleness_penalty
        - weather_uncertainty_penalty
        - weather_bin_penalty
        - generic_bin_penalty
        - ambiguous_resolution_penalty
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
    if overconfidence_penalty > 0:
        rejection_reasons.append("overconfidence_penalty")
    if fallback_edge_penalty > 0:
        rejection_reasons.append("fallback_edge_penalty")
    if proxy_evidence_penalty > 0:
        rejection_reasons.append("proxy_evidence_penalty")
    if liquidity_penalty > 0:
        rejection_reasons.append("thin_liquidity_penalty")
    if staleness_penalty > 0:
        rejection_reasons.append("staleness_penalty")
    if weather_uncertainty_penalty > 0:
        rejection_reasons.append("weather_uncertainty_penalty")
    if weather_bin_penalty > 0:
        rejection_reasons.append("weather_bin_penalty")
    if generic_bin_penalty > 0:
        rejection_reasons.append("generic_bin_penalty")
    if ambiguous_resolution_penalty > 0:
        rejection_reasons.append("ambiguous_resolution_penalty")

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
        computed_edge_bonus=computed_edge_bonus,
        definitive_outcome_bonus=definitive_outcome_bonus,
        evidence_basis_bonus=evidence_basis_bonus,
        observed_data_bonus=observed_data_bonus,
        low_information_penalty=low_information_penalty,
        no_external_odds_penalty=no_external_odds_penalty,
        repeated_analysis_penalty=repeated_analysis_penalty,
        mention_market_penalty=mention_market_penalty,
        confidence_calibration_penalty=confidence_calibration_penalty,
        overconfidence_penalty=overconfidence_penalty,
        weather_uncertainty_penalty=weather_uncertainty_penalty,
        weather_bin_penalty=weather_bin_penalty,
        generic_bin_penalty=generic_bin_penalty,
        ambiguous_resolution_penalty=ambiguous_resolution_penalty,
        fallback_edge_penalty=fallback_edge_penalty,
        proxy_evidence_penalty=proxy_evidence_penalty,
        rejection_reasons=tuple(rejection_reasons),
        bayesian_posterior=bayesian_posterior,
        lmsr_price=lmsr_price,
        inefficiency_signal=inefficiency_signal,
        kelly_raw=kelly_raw,
    )


def calibrate_confidence(
    raw_confidence: float,
    *,
    shrinkage_floor: float = _CONFIDENCE_SHRINKAGE_FLOOR,
    shrinkage_factor: float = _CONFIDENCE_SHRINKAGE_FACTOR,
    evidence_basis_class: str = "",
    definitive_outcome: bool = False,
) -> float:
    """Shrink high confidence values toward a neutral baseline."""
    bounded_confidence = max(0.0, min(1.0, raw_confidence))
    bounded_floor = max(0.0, min(1.0, shrinkage_floor))
    bounded_factor = max(0.0, min(1.0, shrinkage_factor))
    if str(evidence_basis_class or "").strip().lower() == "direct":
        bounded_floor = max(bounded_floor, 0.55)
        bounded_factor = min(1.0, bounded_factor * 1.5)
    if definitive_outcome:
        bounded_floor = max(bounded_floor, 0.60)
        bounded_factor = min(1.0, bounded_factor * 2.5)
    if bounded_confidence <= bounded_floor:
        return bounded_confidence
    calibrated = bounded_floor + ((bounded_confidence - bounded_floor) * bounded_factor)
    return max(0.0, min(1.0, calibrated))


def _weather_bin_penalty(market: Market) -> float:
    market_id = (market.id or "").strip()
    if _WEATHER_BIN_TICKER_PATTERN.search(market_id):
        return _NARROW_WEATHER_BIN_PENALTY
    return 0.0


def _is_mention_market(market: Market) -> bool:
    market_id = (market.id or "").strip()
    return bool(_MENTION_MARKET_TICKER_PATTERN.search(market_id))


def _is_weather_market(market: Market) -> bool:
    category = (market.category or "").lower()
    question = (market.question or "").lower()
    text = f"{category} {question}"
    return any(token in text for token in ("weather", "temperature", "rain", "snow", "wind", "nws"))

