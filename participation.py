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
    edge_value: float | None = None,
    edge_reasonable_max: float = 0.35,
    definitive_edge_reasonable_max: float = 0.65,
    score_gate_blocked: bool = False,
    score_gate_reason: str | None = None,
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
            tier_metadata=metadata,
        )

    if analysis_failed:
        tier = (
            ParticipationTier.OPERATIONAL_ERROR_RETRY
            if analysis_error_retriable
            else ParticipationTier.MONITOR_ONLY
        )
        return ParticipationDecision(
            tier=tier,
            primary_reason="analysis_failure",
            why_not_execution_eligible="Analysis failed; no decision produced",
            what_to_learn_next="Retry on next cycle if retriable; otherwise monitor",
            tier_metadata=metadata,
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
            what_to_learn_next="Gather better evidence sources for this market",
            tier_metadata=metadata,
        )

    if decision_should_trade is False:
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
