"""Participation tier model for market selection decisions.

Single source of truth for the participation tier enum, decision dataclass,
and the pure classification function used by pre-analysis gating and
downstream validation gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        def __str__(self) -> str:
            return str(self.value)
from typing import Any


class ParticipationTier(StrEnum):
    EXECUTION_ELIGIBLE = "execution_eligible"
    DEEP_RESEARCH_REQUIRED = "deep_research_required"
    RESEARCH_ONLY_LEARNING_QUEUE = "research_only_learning_queue"
    MONITOR_ONLY = "monitor_only"
    SKIP_FOR_NOW_WITH_REASON = "skip_for_now_with_reason"
    OPERATIONAL_ERROR_RETRY = "operational_error_retry"
    TERMINAL_REJECT = "terminal_reject"


@dataclass(frozen=True)
class ParticipationDecision:
    tier: ParticipationTier
    primary_reason: str
    contributing_reasons: tuple[str, ...] = ()
    why_not_execution_eligible: str | None = None
    what_to_learn_next: str | None = None
    confidence_in_classification: float = 1.0
    sample_size_signal: int | None = None
    tier_metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata_tuple(self) -> tuple[bool, str | None, dict[str, Any]]:
        """Convert decision to the (demoted, reason, metadata) shape used by callers.

        ``demoted`` is True for any tier other than EXECUTION_ELIGIBLE and
        DEEP_RESEARCH_REQUIRED (the two tiers that allow continuation through
        analysis or execution). For demoted tiers, callers consume
        ``participation_demotion_reason`` plus the structured tier on
        ``participation_tier`` to decide downstream routing.
        """
        demoted = self.tier not in {
            ParticipationTier.EXECUTION_ELIGIBLE,
            ParticipationTier.DEEP_RESEARCH_REQUIRED,
        }
        reason = self.primary_reason if demoted else None
        metadata = dict(self.tier_metadata)
        metadata["participation_tier"] = str(self.tier)
        metadata["participation_decision"] = str(self.primary_reason)
        metadata["participation_terminal_reject"] = (
            self.tier == ParticipationTier.TERMINAL_REJECT
        )
        if self.why_not_execution_eligible:
            metadata["why_not_execution_eligible"] = self.why_not_execution_eligible
        if self.what_to_learn_next:
            metadata["what_to_learn_next"] = self.what_to_learn_next
        if self.sample_size_signal is not None:
            metadata["sample_size_signal"] = self.sample_size_signal
        if demoted:
            metadata["participation_demotion_reason"] = self.primary_reason
        return demoted, reason, metadata


@dataclass(frozen=True)
class HistoricalGateResult:
    allowed: bool
    reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    tier: str = "neutral"
    sample_size: int | None = None
    wilson_win_rate_lower_bound: float | None = None
    shrunk_pnl_per_trade: float | None = None
    what_to_learn_next: str | None = None


@dataclass(frozen=True)
class TimeoutState:
    timed_out: bool = False
    retriable: bool = False
    timeout_streak: int = 0
    search_profile: str = "generic"


_TIER_FOR_REJECTION_REASON: dict[str, ParticipationTier] = {
    "historical_prefix_pnl_block": ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE,
    "historical_prefix_small_sample_negative": ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE,
    "historical_family_pnl_block": ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE,
    "score_soft_research": ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE,
    # Score gates below the research band: routed to SKIP_FOR_NOW_WITH_REASON
    # (not RESEARCH_ONLY) because the score is too far from the threshold to
    # be a near-miss; surface them so operators can review structural penalty
    # mix without populating the learning queue with low-priority entries.
    "score_below_min": ParticipationTier.SKIP_FOR_NOW_WITH_REASON,
    "score_far_below_min": ParticipationTier.SKIP_FOR_NOW_WITH_REASON,
    "fallback_edge_high_churn": ParticipationTier.SKIP_FOR_NOW_WITH_REASON,
    "zero_action_family": ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE,
    "crypto_historically_unprofitable": ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE,
    "repeated_non_actionable_market": ParticipationTier.SKIP_FOR_NOW_WITH_REASON,
    "repeated_non_actionable_bin_market": ParticipationTier.SKIP_FOR_NOW_WITH_REASON,
    "repeated_churn_market": ParticipationTier.SKIP_FOR_NOW_WITH_REASON,
    "daily_drawdown_blocked": ParticipationTier.MONITOR_ONLY,
}

_LEARN_NEXT_FOR_REASON: dict[str, str] = {
    "historical_prefix_pnl_block": (
        "Need >=10 prefix samples with Wilson lower-bound above cutoff to upgrade; "
        "track outcomes to recalibrate."
    ),
    "historical_prefix_small_sample_negative": (
        "Small-sample negative PnL on this prefix; need more outcomes before "
        "the gate can make a confident call."
    ),
    "historical_family_pnl_block": (
        "Family-level PnL is negative; monitor for recovery across the family "
        "before re-enabling execution."
    ),
    "score_soft_research": (
        "Pre-analysis score is near the execution threshold; learn whether better "
        "sources, fresher prices, or settlement context would make this actionable."
    ),
    "score_below_min": (
        "Pre-analysis score is below the research band; revisit only when the "
        "score components (price center, liquidity, horizon, source quality) "
        "improve materially or after a settlement outcome lands."
    ),
    "score_far_below_min": (
        "Pre-analysis score is materially below the research band; this market "
        "is unlikely to become actionable without a structural change in "
        "pricing, liquidity, or evidence availability."
    ),
    "zero_action_family": (
        "This family has never been deep-analyzed to execution; needs a probe "
        "trade to break the self-fulfilling loop."
    ),
    "crypto_historically_unprofitable": (
        "Crypto family historically unprofitable; monitor for improved "
        "conditions or better edge signal."
    ),
    "daily_drawdown_blocked": (
        "Daily drawdown cap already exceeded; monitor for drawdown reset "
        "(new trading day or position close) before re-evaluating."
    ),
}


def _augment_with_threshold_gap(
    base_message: str | None,
    metadata: dict[str, Any],
) -> str | None:
    """Append a quantitative threshold-gap hint when one is available in metadata.

    Helps research-only entries stay actionable: an operator scanning the
    learning queue can immediately see how close the market was to passing
    the gate without rerunning analytics scripts.
    """
    if not base_message:
        return base_message
    gap = metadata.get("pre_analysis_threshold_gap")
    score = metadata.get("pre_analysis_score")
    threshold = metadata.get("pre_analysis_threshold")
    fragments: list[str] = []
    try:
        if gap is not None:
            gap_value = float(gap)
            if gap_value > 0.0:
                fragments.append(f"threshold_gap={gap_value:.3f}")
    except (TypeError, ValueError):
        pass
    try:
        if score is not None and threshold is not None:
            fragments.append(
                f"score={float(score):.3f} vs min={float(threshold):.3f}"
            )
    except (TypeError, ValueError):
        pass
    if not fragments:
        return base_message
    return f"{base_message} ({', '.join(fragments)})"


def classify_participation(
    *,
    historical_gate: HistoricalGateResult | None = None,
    pre_analysis_rejection_reason: str | None = None,
    pre_analysis_metadata: dict[str, Any] | None = None,
    timeout_state: TimeoutState | None = None,
    analysis_failed: bool = False,
    analysis_error_retriable: bool = False,
    decision_should_trade: bool | None = None,
    decision_abstain: bool | None = None,
    decision_definitive_outcome: bool = False,
    decision_evidence_basis: str | None = None,
    decision_edge_source: str | None = None,
    decision_primary_source_whitelisted: bool = False,
    decision_evidence_quality: float | None = None,
    evidence_quality_threshold: float | None = None,
    confidence_value: float | None = None,
    confidence_threshold: float | None = None,
    edge_value: float | None = None,
    edge_reasonable_max: float = 0.35,
    definitive_edge_reasonable_max: float = 0.65,
    score_gate_blocked: bool = False,
    score_gate_reason: str | None = None,
    downstream_gate_reason: str | None = None,
    probe_trade_enabled: bool = False,
    proven_winning_families: frozenset[str] | None = None,
) -> ParticipationDecision:
    """Pure classification of a market into a participation tier.

    Handles pre-analysis, analysis, and post-analysis gates in a single
    function. Callers pass whichever signals are available at their stage.
    """
    metadata = dict(pre_analysis_metadata or {})

    if timeout_state and timeout_state.timed_out:
        timeout_tier = (
            ParticipationTier.OPERATIONAL_ERROR_RETRY
            if timeout_state.retriable and timeout_state.timeout_streak <= 1
            else ParticipationTier.MONITOR_ONLY
        )
        timeout_metadata = {
            **metadata,
            "operational_search_profile": str(timeout_state.search_profile or "generic"),
            "operational_timeout_streak": int(timeout_state.timeout_streak),
            "operational_retriable": bool(timeout_state.retriable),
        }
        return ParticipationDecision(
            tier=timeout_tier,
            primary_reason="grok_stream_timeout",
            why_not_execution_eligible="Analysis timed out before completion",
            what_to_learn_next=(
                f"Retry with extended timeout or cheaper preflight; "
                f"profile={timeout_state.search_profile}, "
                f"streak={timeout_state.timeout_streak}"
            ),
            sample_size_signal=None,
            tier_metadata=timeout_metadata,
        )

    if analysis_failed:
        tier = (
            ParticipationTier.OPERATIONAL_ERROR_RETRY
            if analysis_error_retriable
            else ParticipationTier.MONITOR_ONLY
        )
        failure_metadata = {
            **metadata,
            "operational_retriable": bool(analysis_error_retriable),
        }
        return ParticipationDecision(
            tier=tier,
            primary_reason="analysis_failure",
            why_not_execution_eligible="Analysis failed; no decision produced",
            what_to_learn_next="Retry on next cycle if retriable; otherwise monitor",
            tier_metadata=failure_metadata,
        )

    if pre_analysis_rejection_reason:
        normalized_reason = pre_analysis_rejection_reason
        if normalized_reason.startswith("pre_analysis_"):
            normalized_reason = normalized_reason[len("pre_analysis_"):]
        tier = _TIER_FOR_REJECTION_REASON.get(
            normalized_reason,
            ParticipationTier.SKIP_FOR_NOW_WITH_REASON,
        )
        what_to_learn = _LEARN_NEXT_FOR_REASON.get(normalized_reason)
        sample_size = None
        if historical_gate and historical_gate.sample_size is not None:
            sample_size = historical_gate.sample_size
            if historical_gate.wilson_win_rate_lower_bound is not None:
                what_to_learn = (
                    f"Prefix sample_size={sample_size}, Wilson LB win rate="
                    f"{historical_gate.wilson_win_rate_lower_bound:.2f}; "
                    f"need >=10 samples for confident gate. "
                    f"{what_to_learn or ''}"
                )
        what_to_learn = _augment_with_threshold_gap(what_to_learn, metadata)

        why_not = _why_not_for_reason(pre_analysis_rejection_reason, metadata)

        if (
            probe_trade_enabled
            and normalized_reason == "historical_prefix_small_sample_negative"
            and proven_winning_families
        ):
            gate_family = (
                (historical_gate.metrics.get("family", "") if historical_gate else "")
                or metadata.get("participation_demotion_family", "")
            ).strip().lower()
            if gate_family in proven_winning_families:
                probe_metadata = dict(metadata)
                probe_metadata["probe_trade"] = True
                return ParticipationDecision(
                    tier=ParticipationTier.EXECUTION_ELIGIBLE,
                    primary_reason="probe_trade_small_sample_winning_family",
                    why_not_execution_eligible=None,
                    what_to_learn_next=(
                        f"Probe trade: small sample on prefix but family '{gate_family}' "
                        "is a proven winner; capped at PROBE_TRADE_MAX_USDC."
                    ),
                    sample_size_signal=sample_size,
                    tier_metadata=probe_metadata,
                )

        return ParticipationDecision(
            tier=tier,
            primary_reason=pre_analysis_rejection_reason,
            why_not_execution_eligible=why_not,
            what_to_learn_next=what_to_learn,
            sample_size_signal=sample_size,
            tier_metadata=metadata,
        )

    is_definitive = (
        decision_definitive_outcome
        and decision_evidence_basis == "direct"
        and decision_primary_source_whitelisted
    )

    if decision_abstain:
        return ParticipationDecision(
            tier=ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE,
            primary_reason="abstain_low_evidence",
            why_not_execution_eligible="Model abstained due to low evidence",
            what_to_learn_next=(
                "Find a settlement-aligned primary source (whitelisted "
                "domain) or a current external odds reference; abstain "
                "should clear once evidence_basis becomes direct."
            ),
            tier_metadata=metadata,
        )

    if decision_should_trade is False:
        normalized_evidence_basis_no = str(decision_evidence_basis or "").strip().lower()
        normalized_edge_source_no = str(decision_edge_source or "").strip().lower()
        # Distinguish between "model judgment says no" and "model couldn't
        # research enough to form a market judgment". The placeholder triplet
        # (eq=0/edge_source=none/basis=absence_only) is consistently a
        # research gap, not a no-trade verdict on the market itself.
        is_research_gap = (
            normalized_evidence_basis_no == "absence_only"
            or normalized_edge_source_no == "none"
            or (
                decision_evidence_quality is not None
                and decision_evidence_quality <= 0.0
            )
        )
        if is_research_gap:
            return ParticipationDecision(
                tier=ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE,
                primary_reason="no_trade_research_gap",
                why_not_execution_eligible=(
                    "No-trade decision was driven by a research gap "
                    "(absence_only / edge_source=none / evidence_quality=0), "
                    "not a market-quality judgment"
                ),
                what_to_learn_next=(
                    "Treat as a research failure: locate a settlement-aligned "
                    "primary source and a usable orderbook before relying on "
                    "this no-trade decision."
                ),
                tier_metadata={
                    **metadata,
                    "evidence_gap_classified_as_research": True,
                    "skip_due_to": "lack_of_evidence",
                },
            )
        return ParticipationDecision(
            tier=ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE,
            primary_reason="no_trade_recommended",
            why_not_execution_eligible="Model did not recommend trading",
            what_to_learn_next="Monitor for condition changes",
            tier_metadata=metadata,
        )

    if decision_should_trade is True:
        normalized_evidence_basis = str(decision_evidence_basis or "").strip().lower()
        normalized_edge_source = str(decision_edge_source or "").strip().lower()
        if (
            not is_definitive
            and (
                normalized_evidence_basis == "absence_only"
                or normalized_edge_source == "none"
                or (
                    decision_evidence_quality is not None
                    and decision_evidence_quality <= 0.0
                )
            )
        ):
            return ParticipationDecision(
                tier=ParticipationTier.DEEP_RESEARCH_REQUIRED,
                primary_reason="research_gap_not_market_judgment",
                why_not_execution_eligible=(
                    "Trade conviction exists but evidence/edge source is missing"
                ),
                what_to_learn_next=(
                    "Find direct settlement-aligned evidence, a current orderbook, "
                    "and a computed edge before treating this as executable."
                ),
                tier_metadata={
                    **metadata,
                    "blocked_conviction": True,
                    "skip_due_to": "lack_of_evidence",
                },
            )

        if (
            decision_evidence_quality is not None
            and evidence_quality_threshold is not None
            and decision_evidence_quality < evidence_quality_threshold
            and not is_definitive
        ):
            return ParticipationDecision(
                tier=ParticipationTier.SKIP_FOR_NOW_WITH_REASON,
                primary_reason="evidence_quality_below_min",
                why_not_execution_eligible=(
                    f"evidence_quality={decision_evidence_quality:.2f} "
                    f"< threshold={evidence_quality_threshold:.2f}"
                ),
                what_to_learn_next="Need higher evidence quality from primary sources",
                tier_metadata={
                    **metadata,
                    "blocked_conviction": True,
                    "skip_due_to": "lack_of_evidence",
                },
            )

        if (
            confidence_value is not None
            and confidence_threshold is not None
            and confidence_value < confidence_threshold
            and not is_definitive
        ):
            return ParticipationDecision(
                tier=ParticipationTier.SKIP_FOR_NOW_WITH_REASON,
                primary_reason="confidence_below_min",
                why_not_execution_eligible=(
                    f"confidence={confidence_value:.2f} "
                    f"< threshold={confidence_threshold:.2f}"
                ),
                what_to_learn_next=(
                    "Need stronger direct evidence or a valid configured "
                    "confidence override before execution."
                ),
                tier_metadata={
                    **metadata,
                    "blocked_conviction": True,
                    "skip_due_to": "weak_edge",
                },
            )

        effective_edge_max = (
            definitive_edge_reasonable_max if is_definitive else edge_reasonable_max
        )
        if edge_value is not None and abs(edge_value) > effective_edge_max + 1e-9:
            if not is_definitive:
                return ParticipationDecision(
                    tier=ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE,
                    primary_reason="edge_above_reasonable_max",
                    why_not_execution_eligible=(
                        f"Edge {edge_value:.4f} exceeds max {effective_edge_max:.2f}"
                    ),
                    what_to_learn_next="Verify edge is not hallucinated",
                    tier_metadata={
                        **metadata,
                        "blocked_conviction": True,
                        "skip_due_to": "weak_edge",
                    },
                )

        if score_gate_blocked:
            return ParticipationDecision(
                tier=ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE,
                primary_reason=score_gate_reason or "score_gate_blocked",
                why_not_execution_eligible="Score gate rejected this candidate",
                what_to_learn_next="Improve score components",
                tier_metadata={
                    **metadata,
                    "blocked_conviction": True,
                    "skip_due_to": score_gate_reason or "score_gate",
                },
            )

        if downstream_gate_reason:
            normalized_gate_reason = str(downstream_gate_reason).strip().lower()
            risk_cap_reasons = {
                "balance_exhausted_skip",
                "daily_drawdown_limit",
                "daily_limit_reached",
                "event_side_conflict_blocked",
                "insufficient_balance",
                "position_adjustment_blocked",
                "position_saturated",
            }
            transient_price_reasons = {
                "entry_price_too_low",
                "order_price_outside_submission_band",
                "refreshed_edge_gate_blocked",
                "uniform_implied_probability",
            }
            research_gap_reasons = {
                "evidence_quality_below_min",
                "weather_evidence_quality_below_min",
                "non_sports_missing_primary_source",
                "non_sports_needs_direct",
                "non_sports_needs_direct_evidence",
            }
            weak_edge_reasons = {
                "edge_above_reasonable_max",
                "edge_gate_blocked",
                "hallucinated_edge",
                "lmsr_gate_blocked",
                "refreshed_edge_gate_blocked",
                "score_gate_blocked",
                "score_gate_critical_rejection",
            }
            weak_confidence_reasons = {"confidence_below_min"}
            terminal_reasons = {
                "market_closed",
                "market_closed_during_cycle",
                "market_unavailable",
                "outcome_mismatch_blocked",
                "ambiguous_resolution",
            }
            if normalized_gate_reason in risk_cap_reasons:
                return ParticipationDecision(
                    tier=ParticipationTier.MONITOR_ONLY,
                    primary_reason=normalized_gate_reason,
                    why_not_execution_eligible=(
                        "Transient risk, balance, position, or daily-limit gate blocked execution"
                    ),
                    what_to_learn_next="Re-evaluate after the risk cap or position constraint clears.",
                    tier_metadata={
                        **metadata,
                        "blocked_conviction": True,
                        "skip_due_to": "risk_cap",
                    },
                )
            if normalized_gate_reason in research_gap_reasons:
                return ParticipationDecision(
                    tier=ParticipationTier.DEEP_RESEARCH_REQUIRED,
                    primary_reason=normalized_gate_reason,
                    why_not_execution_eligible="Execution blocked by missing or weak direct evidence",
                    what_to_learn_next=(
                        "Find a direct settlement-aligned primary source and recompute edge before execution."
                    ),
                    tier_metadata={
                        **metadata,
                        "blocked_conviction": True,
                        "skip_due_to": "lack_of_evidence",
                    },
                )
            if normalized_gate_reason in weak_confidence_reasons:
                return ParticipationDecision(
                    tier=ParticipationTier.SKIP_FOR_NOW_WITH_REASON,
                    primary_reason=normalized_gate_reason,
                    why_not_execution_eligible="Confidence gate blocked execution",
                    what_to_learn_next="Need stronger direct evidence or a valid confidence override.",
                    tier_metadata={
                        **metadata,
                        "blocked_conviction": True,
                        "skip_due_to": "weak_edge",
                    },
                )
            if normalized_gate_reason in weak_edge_reasons:
                return ParticipationDecision(
                    tier=ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE,
                    primary_reason=normalized_gate_reason,
                    why_not_execution_eligible="Edge, score, or market-pricing gate blocked execution",
                    what_to_learn_next=(
                        "Verify the edge against refreshed orderbook data and source quality before execution."
                    ),
                    tier_metadata={
                        **metadata,
                        "blocked_conviction": True,
                        "skip_due_to": "weak_edge",
                    },
                )
            if normalized_gate_reason in terminal_reasons:
                return ParticipationDecision(
                    tier=ParticipationTier.TERMINAL_REJECT,
                    primary_reason=normalized_gate_reason,
                    why_not_execution_eligible="Market is not currently tradable for this outcome",
                    what_to_learn_next="Do not re-analyze unless market status or resolution metadata changes.",
                    tier_metadata={
                        **metadata,
                        "blocked_conviction": True,
                        "skip_due_to": "ambiguous_resolution",
                    },
                )
            if normalized_gate_reason in transient_price_reasons:
                return ParticipationDecision(
                    tier=ParticipationTier.MONITOR_ONLY,
                    primary_reason=normalized_gate_reason,
                    why_not_execution_eligible=(
                        "Execution-quality market judgment was blocked by current price/orderbook state"
                    ),
                    what_to_learn_next="Refresh orderbook and entry price before reconsidering execution.",
                    tier_metadata={
                        **metadata,
                        "blocked_conviction": True,
                        "skip_due_to": "stale_evidence",
                    },
                )

        return ParticipationDecision(
            tier=ParticipationTier.EXECUTION_ELIGIBLE,
            primary_reason="all_gates_passed",
            tier_metadata=metadata,
        )

    return ParticipationDecision(
        tier=ParticipationTier.DEEP_RESEARCH_REQUIRED,
        primary_reason="awaiting_analysis",
        why_not_execution_eligible="No analysis performed yet",
        what_to_learn_next="Run deep analysis to determine tradability",
        tier_metadata=metadata,
    )


def _why_not_for_reason(reason: str, metadata: dict[str, Any]) -> str:
    if "historical" in reason:
        sample = metadata.get("historical_gate_prefix_sample_size", "?")
        wr = metadata.get("historical_gate_prefix_win_rate", "?")
        pnl = metadata.get("historical_gate_prefix_pnl_total", "?")
        return (
            f"Historical prefix gate blocked: sample_size={sample}, "
            f"win_rate={wr}, pnl_total={pnl}"
        )
    if "zero_action" in reason:
        fam = metadata.get("participation_demotion_family", "?")
        sz = metadata.get("participation_demotion_family_sample_size", "?")
        rate = metadata.get("participation_demotion_family_action_rate", "?")
        return (
            f"Family '{fam}' has zero action rate: "
            f"sample_size={sz}, action_rate={rate}"
        )
    if "crypto" in reason:
        return "Crypto family historically unprofitable"
    if "churn" in reason:
        streak = metadata.get("participation_demotion_non_actionable_streak", "?")
        count = metadata.get("participation_demotion_analysis_count", "?")
        return f"Repeated churn: streak={streak}, analyses={count}"
    if "fallback" in reason:
        return "High fallback-edge churn with non-actionable streak"
    return f"Pre-analysis rejection: {reason}"


def wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    """Wilson score interval lower bound for binomial proportion."""
    if n <= 0:
        return 0.0
    p_hat = wins / n
    denominator = 1.0 + z * z / n
    center = p_hat + z * z / (2.0 * n)
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * n)) / n)
    return max(0.0, (center - spread) / denominator)


def bayesian_shrunk_pnl(
    pnl: float,
    n: int,
    prior_pnl_per_trade: float = 0.0,
    prior_strength: float = 10.0,
) -> float:
    """Shrink observed PnL/trade toward a prior using sample-size weighting."""
    if n <= 0:
        return prior_pnl_per_trade
    observed_pnl_per_trade = pnl / n
    weight = n / (n + prior_strength)
    return weight * observed_pnl_per_trade + (1.0 - weight) * prior_pnl_per_trade
