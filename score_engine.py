from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from models import Market, TradeDecision

_WEATHER_BIN_TICKER_PATTERN = re.compile(r"-B\d", re.IGNORECASE)
_NARROW_WEATHER_BIN_PENALTY = 0.015
_MENTION_MARKET_TICKER_PATTERN = re.compile(r"MENTION", re.IGNORECASE)
_GENERIC_BIN_TICKER_PATTERN = re.compile(r"-B\d", re.IGNORECASE)
_NUMERIC_STRIKE_TICKER_PATTERN = re.compile(r"-T[-\d.]+$", re.IGNORECASE)
_CONFIDENCE_SHRINKAGE_FLOOR = 0.50
_CONFIDENCE_SHRINKAGE_FACTOR = 0.50
_OVERCONFIDENCE_PENALTY_SCALE_BASE = 0.08
_OVERCONFIDENCE_STEP_LOW_GAP = 0.15
_OVERCONFIDENCE_STEP_HIGH_GAP = 0.25
_OVERCONFIDENCE_STEP_LOW_PENALTY = 0.06
_OVERCONFIDENCE_STEP_HIGH_PENALTY = 0.12
_LATE_STAGE_OVERCONFIDENCE_THRESHOLD = 0.85
# Default score weights for the model-edge signals (Bayesian posterior, LMSR
# inefficiency, Kelly fraction). These are the strategy signals the bot is meant
# to follow, so they must carry enough weight to clear the score gate when a
# genuine edge exists. Raised again (0.08/0.10/0.15 -> 0.10/0.18/0.30) after a
# 10-cycle review found these components contributed 0.0 to the gating/ranking
# score (they were never passed at ranking time) while the proxy/fallback penalty
# stack routinely exceeded 0.4. These are now defaults; callers override them
# from Settings (SCORE_{BAYESIAN,INEFFICIENCY,KELLY}_COMPONENT_WEIGHT).
_BAYESIAN_COMPONENT_WEIGHT = 0.10
_INEFFICIENCY_COMPONENT_WEIGHT = 0.18
_KELLY_COMPONENT_WEIGHT = 0.30


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
    source_confirmed_edge_bonus: float = 0.0
    source_confirmed_edge: bool = False
    source_confirmed_edge_value: float = 0.0
    definitive_outcome_bonus: float = 0.0
    evidence_basis_bonus: float = 0.0
    source_alignment_bonus: float = 0.0
    proxy_penalty_reduced: bool = False
    proxy_penalty_reduction_reason: str = ""
    family_conditional_bonus_applied: bool = False
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
    extreme_confidence_penalty: float = 0.0
    numeric_strike_bin_penalty: float = 0.0
    fallback_high_confidence_penalty: float = 0.0
    extreme_market_edge_penalty: float = 0.0
    hallucinated_edge_penalty: float = 0.0
    hallucinated_edge_penalty_suppressed: bool = False
    high_edge_calibration_penalty: float = 0.0
    extreme_edge_learning_queue: bool = False
    coinflip_sports_penalty: float = 0.0
    late_stage_overconfidence_penalty: float = 0.0
    short_prefix_penalty: float = 0.0
    historical_family_bonus: float = 0.0
    historical_family_signal: float = 0.0
    historical_family_score_adjustment: float = 0.0
    historical_family_size_multiplier: float = 1.0
    historical_prefix_bonus: float = 0.0
    historical_prefix_penalty: float = 0.0
    extreme_confidence_band_penalty: float = 0.0
    numeric_strike_computed_overconfidence_penalty: float = 0.0
    volume_amplifier_discount: float = 0.0
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
    edge_market_confidence_override: float | None = None,
    is_weather_market: bool = False,
    weather_score_penalty: float = 0.0,
    low_info_penalty_threshold: float = 0.55,
    low_info_penalty_base: float = 0.05,
    repeated_analysis_count: int = 0,
    non_actionable_streak: int = 0,
    repeated_analysis_penalty_base: float = 0.025,
    repeated_analysis_penalty_start_count: int = 1,
    mention_market_penalty_base: float = 0.0,
    confidence_calibration_floor: float = 0.50,
    confidence_calibration_penalty_scale: float = 0.0,
    fallback_edge_penalty_base: float = 0.04,
    computed_edge_bonus_base: float = 0.03,
    source_confirmed_edge_min: float = 0.20,
    source_confirmed_edge_min_evidence_quality: float = 0.90,
    source_confirmed_edge_bonus_base: float = 0.06,
    proxy_evidence_penalty_base: float = 0.05,
    overconfidence_penalty_base: float = 0.05,
    generic_bin_penalty_base: float = 0.015,
    ambiguous_resolution_penalty_base: float = 0.06,
    max_reasonable_edge: float = 0.45,
    hallucinated_edge_penalty_base: float = 0.08,
    extreme_market_edge_penalty_base: float = 0.08,
    late_stage_overconfidence_penalty_base: float = 0.08,
    extreme_confidence_threshold: float = 0.90,
    extreme_confidence_penalty_base: float = 0.0,
    short_prefix_penalty: float = 0.0,
    suppress_hallucinated_edge_penalty: bool = False,
    definitive_outcome_eligible: bool = False,
    historical_family_pnl_total: float | None = None,
    historical_family_sample_size: int = 0,
    historical_family_win_rate: float | None = None,
    historical_family_deployed_usdc: float | None = None,
    historical_family_high_conf_losses: int = 0,
    historical_family_bonus_base: float = 0.04,
    historical_family_min_samples: int = 8,
    historical_family_positive_pnl_threshold: float = 10.0,
    historical_family_signal_enabled: bool = True,
    historical_family_signal_score_scale: float = 0.06,
    historical_family_size_scale_max: float = 0.25,
    historical_family_size_scale_max_negative: float = 0.25,
    now: datetime | None = None,
    evidence_basis_class: str = "",
    edge_source: str = "",
    source_match_class: str = "",
    primary_source_url_present: bool = False,
    market_family: str = "",
    coinflip_price_lower: float = 0.45,
    coinflip_price_upper: float = 0.55,
    historical_prefix_pnl_per_trade: float | None = None,
    historical_prefix_sample_size: int = 0,
    volume_amplifier_enabled: bool = True,
    proxy_penalty_convergent_reduction_enabled: bool = True,
    self_consistency_passed: bool = False,
    historical_family_high_conf_loss_relax_threshold: float = 0.05,
    historical_family_boost_evidence_min: float = 0.44,
    historical_family_loss_drag_scale: float = 1.8,
    historical_family_loss_drag_sample_min: int = 30,
    bayesian_component_weight: float = _BAYESIAN_COMPONENT_WEIGHT,
    inefficiency_component_weight: float = _INEFFICIENCY_COMPONENT_WEIGHT,
    kelly_component_weight: float = _KELLY_COMPONENT_WEIGHT,
) -> ScoreResult:
    now = now or datetime.now(timezone.utc)
    edge_market = 0.0
    if implied_prob_market is not None:
        # When a direct-evidence posterior floor is supplied, use it (never below
        # the model's own confidence) so calibration shrink cannot invert a real
        # positive edge into a negative market edge at the score gate, matching
        # the edge gate and Kelly sizing.
        edge_market_confidence = decision.confidence
        if edge_market_confidence_override is not None:
            edge_market_confidence = max(
                decision.confidence, float(edge_market_confidence_override)
            )
        edge_market = edge_market_confidence - implied_prob_market

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
        bayesian_component = max(0.0, bayesian_component_weight) * (bayesian_posterior - 0.5)
    inefficiency_component = 0.0
    if inefficiency_signal is not None:
        chosen_side_inefficiency = max(0.0, float(inefficiency_signal))
        inefficiency_component = (
            max(0.0, inefficiency_component_weight) * chosen_side_inefficiency
        )
    kelly_component = 0.0
    if kelly_raw is not None:
        kelly_component = max(0.0, kelly_component_weight) * max(0.0, min(1.0, kelly_raw))
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
    normalized_source_match = str(source_match_class or "").strip().lower()
    source_alignment_bonus = 0.0
    settlement_aligned_high_quality = (
        normalized_source_match == "settlement_aligned"
        and evidence_quality >= 0.90
        and bool(primary_source_url_present)
        and normalized_evidence_basis in {"direct", "proxy"}
    )
    if settlement_aligned_high_quality:
        source_alignment_bonus = 0.02 if normalized_evidence_basis == "direct" else 0.03
    low_information_penalty = 0.0
    low_info_threshold = max(0.0, min(1.0, low_info_penalty_threshold))
    low_info_base = max(0.0, low_info_penalty_base)
    normalized_edge_source = (edge_source or decision.edge_source or "").strip().lower()
    source_confirmed_edge_value = max(edge_market, edge_external)
    source_confirmed_edge = (
        normalized_edge_source == "computed"
        and bool(primary_source_url_present)
        and evidence_quality >= max(0.0, min(1.0, float(source_confirmed_edge_min_evidence_quality)))
        and source_confirmed_edge_value >= max(0.0, min(1.0, float(source_confirmed_edge_min)))
        and (
            normalized_evidence_basis == "direct"
            or (
                normalized_source_match == "settlement_aligned"
                and normalized_evidence_basis in {"direct", "proxy"}
            )
        )
    )
    source_confirmed_edge_bonus = 0.0
    if source_confirmed_edge:
        edge_headroom = max(
            0.0,
            source_confirmed_edge_value - max(0.0, float(source_confirmed_edge_min)),
        )
        source_confirmed_edge_bonus = max(
            0.0,
            min(0.10, float(source_confirmed_edge_bonus_base) + min(0.04, edge_headroom * 0.20)),
        )
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
    elif definitive_outcome_eligible:
        # Auto-detected via whitelisted primary source + my_prob near 0/1 +
        # direct basis. This is a stronger signal than likelihood_ratio
        # inference since it requires a settlement-aligned source and a
        # near-binary read of the outcome.
        definitive_outcome_bonus = 0.10
    computed_edge_bonus = 0.0
    if normalized_edge_source == "computed":
        computed_edge_bonus = max(0.0, computed_edge_bonus_base) * (
            0.50 + (0.50 * evidence_quality)
        )
    if evidence_quality < low_info_threshold and normalized_edge_source in {"fallback", "none"}:
        low_info_shortfall = low_info_threshold - evidence_quality
        low_information_penalty = low_info_base + (low_info_base * low_info_shortfall)
    no_external_odds_penalty = 0.0
    if (
        normalized_edge_source in {"fallback", "none"}
        and decision.implied_prob_external is None
        and not definitive_outcome_eligible
    ):
        no_external_odds_penalty = 0.04
    if normalized_edge_source == "none" and not definitive_outcome_eligible:
        no_external_odds_penalty += 0.06
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
    # Penalize against pre-calibration (raw) confidence so this does not
    # double-count the static + historical confidence shrink that already lowered
    # decision.confidence. Penalizing the shrunk value was self-defeating:
    # calibration lowered confidence, then the score punished it again, pushing
    # every market below the gate. Falls back to decision.confidence when raw is
    # unavailable (e.g. synthetic decisions, tests).
    confidence_for_calibration_penalty = (
        decision.raw_confidence
        if decision.raw_confidence is not None
        else decision.confidence
    )
    if confidence_for_calibration_penalty < normalized_conf_floor:
        confidence_shortfall = normalized_conf_floor - max(
            0.0, confidence_for_calibration_penalty
        )
        confidence_calibration_penalty = (
            confidence_shortfall * max(0.0, confidence_calibration_penalty_scale)
        )
    overconfidence_penalty = 0.0
    overconfidence_gap = max(0.0, decision.confidence - evidence_quality)
    penalty_scale = 0.0
    if overconfidence_penalty_base > 0:
        penalty_scale = overconfidence_penalty_base / _OVERCONFIDENCE_PENALTY_SCALE_BASE
    if overconfidence_gap > _OVERCONFIDENCE_STEP_HIGH_GAP:
        overconfidence_penalty = _OVERCONFIDENCE_STEP_HIGH_PENALTY * penalty_scale
    elif overconfidence_gap > _OVERCONFIDENCE_STEP_LOW_GAP:
        overconfidence_penalty = _OVERCONFIDENCE_STEP_LOW_PENALTY * penalty_scale

    late_stage_overconfidence_penalty = 0.0
    if (
        decision.confidence >= _LATE_STAGE_OVERCONFIDENCE_THRESHOLD
        and normalized_evidence_basis != "direct"
    ):
        late_stage_overconfidence_penalty = max(
            0.0, late_stage_overconfidence_penalty_base
        )
    extreme_confidence_penalty = 0.0
    normalized_extreme_confidence_threshold = max(
        0.0, min(1.0, extreme_confidence_threshold)
    )
    if decision.confidence >= normalized_extreme_confidence_threshold:
        extreme_confidence_penalty = max(0.0, extreme_confidence_penalty_base)
        if normalized_evidence_basis != "direct":
            extreme_confidence_penalty += 0.04

    normalized_max_reasonable_edge = max(0.0, min(1.0, max_reasonable_edge))
    fallback_high_confidence_penalty = 0.0
    if normalized_edge_source in {"fallback", "none"} and decision.confidence >= 0.85:
        fallback_high_confidence_penalty = 0.10 + (
            max(0.0, decision.confidence - 0.85) * 2.0
        )
        confidence_evidence_gap = max(0.0, decision.confidence - evidence_quality - 0.15)
        fallback_high_confidence_penalty += confidence_evidence_gap * 0.5

    extreme_market_edge_penalty = 0.0
    if (
        edge_market >= normalized_max_reasonable_edge
        and normalized_evidence_basis != "direct"
        and normalized_edge_source != "computed"
    ):
        extreme_market_edge_penalty = max(0.0, extreme_market_edge_penalty_base)

    hallucinated_edge_penalty = 0.0
    if not suppress_hallucinated_edge_penalty and (
        abs(edge_market) >= normalized_max_reasonable_edge
        or abs(edge_external) >= normalized_max_reasonable_edge
    ):
        hallucinated_edge_penalty = max(0.0, hallucinated_edge_penalty_base)

    high_edge_calibration_penalty = 0.0
    extreme_edge_learning_queue = False
    edge_abs_max = max(abs(edge_market), abs(edge_external))
    high_edge_threshold = min(normalized_max_reasonable_edge, 0.32)
    # The penalty is suppressed when:
    # - evidence_basis=direct AND definitive_outcome_eligible (legacy exemption
    #   for definitive-outcome trades), OR
    # - evidence_basis=direct AND suppress_hallucinated_edge_penalty=True
    #   (the high-quality settlement-aligned path added in the cycle 1
    #   review: same conditions that suppress hallucinated_edge should
    #   suppress this matching penalty so the score gate sees the unstacked
    #   score).
    # Suppression requires direct evidence in both cases — proxy/fallback
    # high edges still receive the penalty as a safety net for the largest
    # realized PnL leak class.
    high_edge_exempt_via_direct = normalized_evidence_basis == "direct" and (
        definitive_outcome_eligible or suppress_hallucinated_edge_penalty
    )
    if edge_abs_max > high_edge_threshold and not high_edge_exempt_via_direct:
        high_edge_calibration_penalty = min(
            0.18,
            max(0.0, edge_abs_max - high_edge_threshold) * 0.50,
        )
        if edge_abs_max > max(0.45, high_edge_threshold):
            high_edge_calibration_penalty = max(high_edge_calibration_penalty, 0.12)
            extreme_edge_learning_queue = True
        if edge_abs_max >= 0.55:
            high_edge_calibration_penalty = max(high_edge_calibration_penalty, 0.18)
            extreme_edge_learning_queue = True

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
    proxy_penalty_reduced = False
    proxy_penalty_reduction_reason = ""
    generic_bin_penalty = 0.0
    numeric_strike_bin_penalty = 0.0
    ambiguous_resolution_penalty = 0.0
    if normalized_edge_source in {"fallback", "none"} and not definitive_outcome_eligible:
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
                fallback_edge_penalty *= 0.20
                proxy_evidence_penalty *= 0.20
                proxy_penalty_reduced = True
                proxy_penalty_reduction_reason = "direct_high_quality"
            elif evidence_quality >= 0.55:
                fallback_edge_penalty *= 0.25
                proxy_evidence_penalty *= 0.25
                proxy_penalty_reduced = True
                proxy_penalty_reduction_reason = "direct_high_quality"
            else:
                fallback_edge_penalty *= 0.40
                proxy_evidence_penalty *= 0.40
                proxy_penalty_reduced = True
                proxy_penalty_reduction_reason = "direct_high_quality"
        elif settlement_aligned_high_quality and proxy_evidence_penalty > 0.0:
            proxy_evidence_penalty *= 0.60
            proxy_penalty_reduced = True
            proxy_penalty_reduction_reason = "settlement_aligned_high_quality"
        convergent_family_is_profitable = False
        if proxy_penalty_convergent_reduction_enabled:
            convergent_family_is_profitable = (
                historical_family_pnl_total is not None
                and float(historical_family_pnl_total) > 0.0
                and historical_family_sample_size >= 20
            )
        if proxy_penalty_convergent_reduction_enabled and proxy_evidence_penalty > 0.0:
            original_proxy_penalty = proxy_evidence_penalty
            if self_consistency_passed and convergent_family_is_profitable:
                proxy_evidence_penalty *= 0.50
                proxy_penalty_reduced = True
                proxy_penalty_reduction_reason = "self_consistency_plus_family"
            elif convergent_family_is_profitable:
                proxy_evidence_penalty *= 0.70
                proxy_penalty_reduced = True
                if not proxy_penalty_reduction_reason:
                    proxy_penalty_reduction_reason = "family_profitable_alone"
            proxy_evidence_penalty = max(
                proxy_evidence_penalty,
                original_proxy_penalty * 0.15,
            )
        if (
            proxy_penalty_convergent_reduction_enabled
            and no_external_odds_penalty > 0.0
            and self_consistency_passed
            and convergent_family_is_profitable
        ):
            no_external_odds_penalty *= 0.50
    if _GENERIC_BIN_TICKER_PATTERN.search((market.id or "").strip()) and not _is_weather_market(market):
        generic_bin_penalty = max(0.0, generic_bin_penalty_base) * (1.0 + max(0.0, 0.65 - evidence_quality))
    if (
        _NUMERIC_STRIKE_TICKER_PATTERN.search((market.id or "").strip())
        and not _is_weather_market(market)
        and (
            normalized_evidence_basis != "direct"
            or normalized_edge_source != "computed"
        )
    ):
        numeric_strike_bin_penalty = max(0.0, generic_bin_penalty_base) * (
            1.0 + max(0.0, 0.65 - evidence_quality)
        )
    short_prefix_penalty = max(0.0, float(short_prefix_penalty))
    historical_family_bonus = 0.0
    historical_family_signal = 0.0
    historical_family_score_adjustment = 0.0
    historical_family_size_multiplier = 1.0
    family_conditional_bonus_applied = False
    normalized_market_family = str(market_family or "").strip().lower()
    has_continuous_family_inputs = (
        historical_family_win_rate is not None
        or historical_family_deployed_usdc is not None
        or int(historical_family_high_conf_losses or 0) > 0
    )
    if (
        historical_family_signal_enabled
        and has_continuous_family_inputs
        and historical_family_sample_size >= max(1, int(historical_family_min_samples))
    ):
        sample_size = max(0, int(historical_family_sample_size))
        wins_rate = (
            max(0.0, min(1.0, float(historical_family_win_rate)))
            if historical_family_win_rate is not None
            else 0.50
        )
        pnl_total = float(historical_family_pnl_total or 0.0)
        deployed = float(historical_family_deployed_usdc or 0.0)
        if deployed > 0.0:
            pnl_efficiency = pnl_total / deployed
        else:
            pnl_efficiency = pnl_total / max(1.0, float(sample_size) * 5.0)
        high_conf_loss_rate = max(0.0, float(historical_family_high_conf_losses or 0)) / max(
            1.0, float(sample_size)
        )
        raw_signal = ((wins_rate - 0.52) * 2.0) + pnl_efficiency
        raw_signal -= min(0.35, high_conf_loss_rate * 2.0)
        if (
            normalized_market_family == "sports"
            and pnl_total > 0.0
            and wins_rate >= 0.55
            and high_conf_loss_rate
            <= max(0.0, float(historical_family_high_conf_loss_relax_threshold))
            and source_confirmed_edge_value > 0.0
            and evidence_quality >= max(0.0, float(historical_family_boost_evidence_min))
        ):
            raw_signal += 0.08
            family_conditional_bonus_applied = True
        elif normalized_market_family in {"generic", "crypto"} and pnl_efficiency < 0.0:
            # Historical generic/crypto losses should scale conviction and size,
            # not categorically exclude otherwise eligible source-backed markets.
            drag_scale = (
                float(historical_family_loss_drag_scale)
                if sample_size >= max(1, int(historical_family_loss_drag_sample_min))
                else 1.5
            )
            raw_signal -= min(
                0.55,
                (abs(pnl_efficiency) * drag_scale) + (high_conf_loss_rate * drag_scale),
            )
        if normalized_evidence_basis != "direct" and raw_signal < 0.0:
            raw_signal *= 1.25
        shrink = sample_size / (sample_size + 20.0)
        historical_family_signal = max(-1.0, min(1.0, raw_signal * shrink))
        score_scale = max(0.0, float(historical_family_signal_score_scale))
        historical_family_score_adjustment = historical_family_signal * score_scale
        if historical_family_score_adjustment > 0.0 and not (
            max(edge_market, edge_external) > 0.0 and evidence_quality >= 0.65
        ):
            historical_family_score_adjustment = 0.0
        size_scale = max(0.0, min(1.0, float(historical_family_size_scale_max)))
        # Losing families get steeper downsizing authority than winners get
        # upsizing: oversizing a persistent loser is the dominant drawdown risk,
        # while inflating a thin winner adds risk for little proven edge.
        negative_size_scale = max(
            size_scale,
            min(1.0, float(historical_family_size_scale_max_negative)),
        )
        if historical_family_signal < 0.0:
            historical_family_size_multiplier = max(
                1.0 - negative_size_scale,
                1.0 + (historical_family_signal * negative_size_scale),
            )
        else:
            historical_family_size_multiplier = min(
                1.0 + size_scale,
                1.0 + (historical_family_signal * size_scale),
            )
        historical_family_bonus = max(0.0, historical_family_score_adjustment)
    elif (
        historical_family_pnl_total is not None
        and historical_family_sample_size >= max(1, int(historical_family_min_samples))
        and float(historical_family_pnl_total)
        > float(historical_family_positive_pnl_threshold)
    ):
        historical_family_bonus = max(0.0, float(historical_family_bonus_base))
    _HISTORICAL_PREFIX_MIN_SAMPLES = 8
    _HISTORICAL_PREFIX_BONUS_THRESHOLD = 1.0
    _HISTORICAL_PREFIX_PENALTY_THRESHOLD = -0.5
    _HISTORICAL_PREFIX_SCALE = 0.08
    _HISTORICAL_PREFIX_NORM = 3.0
    historical_prefix_bonus = 0.0
    historical_prefix_penalty = 0.0
    if (
        historical_prefix_pnl_per_trade is not None
        and historical_prefix_sample_size >= _HISTORICAL_PREFIX_MIN_SAMPLES
    ):
        if historical_prefix_pnl_per_trade > _HISTORICAL_PREFIX_BONUS_THRESHOLD:
            historical_prefix_bonus = _HISTORICAL_PREFIX_SCALE * min(
                1.0, historical_prefix_pnl_per_trade / _HISTORICAL_PREFIX_NORM
            )
        elif historical_prefix_pnl_per_trade < _HISTORICAL_PREFIX_PENALTY_THRESHOLD:
            historical_prefix_penalty = _HISTORICAL_PREFIX_SCALE * min(
                1.0, abs(historical_prefix_pnl_per_trade) / _HISTORICAL_PREFIX_NORM
            )

    extreme_confidence_band_penalty = 0.0
    if (
        decision.confidence >= 0.95
        and not (
            normalized_evidence_basis == "direct" and definitive_outcome_eligible
        )
    ):
        extreme_confidence_band_penalty = 0.10 * (decision.confidence - 0.94) / 0.05

    numeric_strike_computed_overconfidence_penalty = 0.0
    if (
        _NUMERIC_STRIKE_TICKER_PATTERN.search((market.id or "").strip())
        and normalized_edge_source == "computed"
        and decision.confidence >= 0.85
    ):
        numeric_strike_computed_overconfidence_penalty = min(
            0.06, 0.05 * (decision.confidence - 0.85) * 4
        )

    coinflip_sports_penalty = 0.0
    if (
        normalized_market_family == "sports"
        and implied_prob_market is not None
        and coinflip_price_lower <= implied_prob_market <= coinflip_price_upper
        and decision.confidence >= 0.80
        and normalized_evidence_basis != "direct"
    ):
        coinflip_sports_penalty = 0.10

    if not (getattr(market, "resolution_criteria", "") or "").strip():
        ambiguous_resolution_penalty = max(0.0, ambiguous_resolution_penalty_base)

    positive_score = (
        (evidence_multiplier * weighted_edge)
        + evidence_component
        + bayesian_component
        + inefficiency_component
        + kelly_component
        + confidence_alignment_bonus
        + computed_edge_bonus
        + source_confirmed_edge_bonus
        + definitive_outcome_bonus
        + evidence_basis_bonus
        + source_alignment_bonus
        + observed_data_bonus
        + historical_family_bonus
        + historical_prefix_bonus
    )
    total_penalty = (
        historical_prefix_penalty
        + max(0.0, -historical_family_score_adjustment)
        + extreme_confidence_band_penalty
        + numeric_strike_computed_overconfidence_penalty
        + low_information_penalty
        + no_external_odds_penalty
        + repeated_analysis_penalty
        + mention_market_penalty
        + confidence_calibration_penalty
        + overconfidence_penalty
        + extreme_confidence_penalty
        + late_stage_overconfidence_penalty
        + fallback_high_confidence_penalty
        + extreme_market_edge_penalty
        + hallucinated_edge_penalty
        + high_edge_calibration_penalty
        + fallback_edge_penalty
        + proxy_evidence_penalty
        + liquidity_penalty
        + staleness_penalty
        + weather_uncertainty_penalty
        + weather_bin_penalty
        + generic_bin_penalty
        + numeric_strike_bin_penalty
        + short_prefix_penalty
        + ambiguous_resolution_penalty
        + coinflip_sports_penalty
    )
    volume_amplifier_discount = 0.0
    if volume_amplifier_enabled and total_penalty > 0 and (liquidity > 500.0 or evidence_quality > 0.80):
        volume_amplifier_discount = total_penalty * 0.20
    final_score = positive_score - total_penalty + volume_amplifier_discount

    rejection_reasons: list[str] = []
    if edge_market <= 0 and not source_confirmed_edge:
        rejection_reasons.append("non_positive_market_edge")
    if evidence_quality < low_info_threshold:
        rejection_reasons.append("low_evidence_quality")
    if low_information_penalty > 0:
        rejection_reasons.append("low_information_penalty")
    if no_external_odds_penalty > 0:
        rejection_reasons.append("no_external_odds_penalty")
        if normalized_edge_source in {"fallback", "none"}:
            rejection_reasons.append("fallback_without_external_odds")
    if repeated_analysis_penalty > 0:
        rejection_reasons.append("repeated_analysis_penalty")
    if mention_market_penalty > 0:
        rejection_reasons.append("mention_market_penalty")
    if confidence_calibration_penalty > 0:
        rejection_reasons.append("confidence_calibration_penalty")
    if overconfidence_penalty > 0:
        rejection_reasons.append("overconfidence_penalty")
    if extreme_confidence_penalty > 0:
        rejection_reasons.append("extreme_confidence_penalty")
    if late_stage_overconfidence_penalty > 0:
        rejection_reasons.append("late_stage_overconfidence")
    if fallback_high_confidence_penalty > 0:
        rejection_reasons.append("fallback_high_confidence_trade")
    if extreme_market_edge_penalty > 0:
        rejection_reasons.append("extreme_market_edge_penalty")
    if hallucinated_edge_penalty > 0:
        rejection_reasons.append("hallucinated_edge")
    if high_edge_calibration_penalty > 0:
        rejection_reasons.append("high_edge_calibration_penalty")
    if extreme_edge_learning_queue:
        rejection_reasons.append("extreme_edge_learning_queue")
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
    if numeric_strike_bin_penalty > 0:
        rejection_reasons.append("numeric_strike_bin")
    if short_prefix_penalty > 0:
        rejection_reasons.append("historical_short_prefix_pnl")
    if ambiguous_resolution_penalty > 0:
        rejection_reasons.append("ambiguous_resolution_penalty")
    if historical_prefix_penalty > 0:
        rejection_reasons.append("historical_prefix_unprofitable")
    if historical_family_score_adjustment < 0:
        rejection_reasons.append("historical_family_negative_signal")
    if extreme_confidence_band_penalty > 0:
        rejection_reasons.append("extreme_confidence_band")
    if numeric_strike_computed_overconfidence_penalty > 0:
        rejection_reasons.append("numeric_strike_computed_overconfidence")
    if coinflip_sports_penalty > 0:
        rejection_reasons.append("coinflip_sports_penalty")

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
        source_confirmed_edge_bonus=source_confirmed_edge_bonus,
        source_confirmed_edge=source_confirmed_edge,
        source_confirmed_edge_value=source_confirmed_edge_value,
        definitive_outcome_bonus=definitive_outcome_bonus,
        evidence_basis_bonus=evidence_basis_bonus,
        source_alignment_bonus=source_alignment_bonus,
        proxy_penalty_reduced=proxy_penalty_reduced,
        proxy_penalty_reduction_reason=proxy_penalty_reduction_reason,
        family_conditional_bonus_applied=family_conditional_bonus_applied,
        observed_data_bonus=observed_data_bonus,
        low_information_penalty=low_information_penalty,
        no_external_odds_penalty=no_external_odds_penalty,
        repeated_analysis_penalty=repeated_analysis_penalty,
        mention_market_penalty=mention_market_penalty,
        confidence_calibration_penalty=confidence_calibration_penalty,
        overconfidence_penalty=overconfidence_penalty,
        extreme_confidence_penalty=extreme_confidence_penalty,
        numeric_strike_bin_penalty=numeric_strike_bin_penalty,
        fallback_high_confidence_penalty=fallback_high_confidence_penalty,
        extreme_market_edge_penalty=extreme_market_edge_penalty,
        hallucinated_edge_penalty=hallucinated_edge_penalty,
        hallucinated_edge_penalty_suppressed=bool(suppress_hallucinated_edge_penalty),
        high_edge_calibration_penalty=high_edge_calibration_penalty,
        extreme_edge_learning_queue=extreme_edge_learning_queue,
        coinflip_sports_penalty=coinflip_sports_penalty,
        late_stage_overconfidence_penalty=late_stage_overconfidence_penalty,
        short_prefix_penalty=short_prefix_penalty,
        historical_family_bonus=historical_family_bonus,
        historical_family_signal=historical_family_signal,
        historical_family_score_adjustment=historical_family_score_adjustment,
        historical_family_size_multiplier=historical_family_size_multiplier,
        historical_prefix_bonus=historical_prefix_bonus,
        historical_prefix_penalty=historical_prefix_penalty,
        extreme_confidence_band_penalty=extreme_confidence_band_penalty,
        numeric_strike_computed_overconfidence_penalty=numeric_strike_computed_overconfidence_penalty,
        volume_amplifier_discount=volume_amplifier_discount,
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
    family_shrinkage_override: float | None = None,
    evidence_basis_class: str = "",
    definitive_outcome: bool = False,
    has_primary_source_url: bool = False,
    direct_shrinkage_boost_factor: float = 1.5,
) -> float:
    """Shrink high confidence values toward a neutral baseline."""
    bounded_confidence = max(0.0, min(1.0, raw_confidence))
    bounded_floor = max(0.0, min(1.0, shrinkage_floor))
    bounded_factor = max(0.0, min(1.0, shrinkage_factor))
    if str(evidence_basis_class or "").strip().lower() == "direct":
        bounded_floor = max(bounded_floor, 0.55)
        bounded_factor = min(1.0, bounded_factor * 1.5)
        if has_primary_source_url:
            bounded_factor = min(
                0.80,
                bounded_factor * max(1.0, float(direct_shrinkage_boost_factor)),
            )
    if definitive_outcome:
        bounded_floor = max(bounded_floor, 0.60)
        bounded_factor = min(1.0, bounded_factor * 2.5)
    if family_shrinkage_override is not None:
        bounded_factor = min(
            bounded_factor,
            max(0.0, min(1.0, float(family_shrinkage_override))),
        )
    if bounded_confidence <= bounded_floor:
        return bounded_confidence
    calibrated = bounded_floor + ((bounded_confidence - bounded_floor) * bounded_factor)
    return max(0.0, min(1.0, calibrated))


def score_breakdown_explanation(result: ScoreResult) -> str:
    """One-line human-readable summary of the top 3 bonuses and top 3 penalties."""
    bonus_fields = [
        ("evidence", result.evidence_component),
        ("bayesian", result.bayesian_component),
        ("inefficiency", result.inefficiency_component),
        ("kelly", result.kelly_component),
        ("alignment", result.confidence_alignment_bonus),
        ("computed_edge", result.computed_edge_bonus),
        ("src_conf_edge", result.source_confirmed_edge_bonus),
        ("definitive", result.definitive_outcome_bonus),
        ("evidence_basis", result.evidence_basis_bonus),
        ("source_align", result.source_alignment_bonus),
        ("observed_data", result.observed_data_bonus),
        ("hist_family", result.historical_family_bonus),
        ("hist_prefix", result.historical_prefix_bonus),
        ("volume_amp", result.volume_amplifier_discount),
    ]
    penalty_fields = [
        ("low_info", result.low_information_penalty),
        ("hist_family", max(0.0, -result.historical_family_score_adjustment)),
        ("no_ext_odds", result.no_external_odds_penalty),
        ("repeated", result.repeated_analysis_penalty),
        ("mention", result.mention_market_penalty),
        ("conf_cal", result.confidence_calibration_penalty),
        ("overconf", result.overconfidence_penalty),
        ("extreme_conf", result.extreme_confidence_penalty),
        ("late_overconf", result.late_stage_overconfidence_penalty),
        ("fb_high_conf", result.fallback_high_confidence_penalty),
        ("extreme_edge", result.extreme_market_edge_penalty),
        ("halluc_edge", result.hallucinated_edge_penalty),
        ("high_edge_cal", result.high_edge_calibration_penalty),
        ("fb_edge", result.fallback_edge_penalty),
        ("proxy_ev", result.proxy_evidence_penalty),
        ("liquidity", result.liquidity_penalty),
        ("staleness", result.staleness_penalty),
        ("weather_unc", result.weather_uncertainty_penalty),
        ("weather_bin", result.weather_bin_penalty),
        ("generic_bin", result.generic_bin_penalty),
        ("num_strike", result.numeric_strike_bin_penalty),
        ("short_pfx", result.short_prefix_penalty),
        ("ambig_res", result.ambiguous_resolution_penalty),
        ("hist_pfx_pen", result.historical_prefix_penalty),
        ("ext_conf_band", result.extreme_confidence_band_penalty),
        ("num_strike_oc", result.numeric_strike_computed_overconfidence_penalty),
        ("coinflip_sp", result.coinflip_sports_penalty),
    ]
    top_bonus = sorted(
        ((n, v) for n, v in bonus_fields if v > 0),
        key=lambda x: -x[1],
    )[:3]
    top_penalty = sorted(
        ((n, v) for n, v in penalty_fields if v > 0),
        key=lambda x: -x[1],
    )[:3]
    parts: list[str] = [f"score={result.final_score:.4f}"]
    if top_bonus:
        parts.append("+" + "+".join(f"{n}:{v:.3f}" for n, v in top_bonus))
    if top_penalty:
        parts.append("-" + "-".join(f"{n}:{v:.3f}" for n, v in top_penalty))
    return " | ".join(parts)


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

