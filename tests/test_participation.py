from __future__ import annotations

from participation import (
    HistoricalGateResult,
    ParticipationDecision,
    ParticipationTier,
    TimeoutState,
    bayesian_shrunk_pnl,
    bayesian_shrunk_win_rate,
    classify_participation,
    wilson_lower_bound,
)


def test_wilson_lower_bound_zero_samples() -> None:
    assert wilson_lower_bound(0, 0) == 0.0


def test_wilson_lower_bound_all_wins() -> None:
    result = wilson_lower_bound(10, 10)
    assert 0.70 < result < 1.0


def test_wilson_lower_bound_no_wins() -> None:
    result = wilson_lower_bound(0, 10)
    assert 0.0 <= result < 0.05


def test_wilson_lower_bound_small_sample() -> None:
    result = wilson_lower_bound(1, 3)
    assert 0.0 < result < 0.50


def test_bayesian_shrunk_pnl_no_samples() -> None:
    assert bayesian_shrunk_pnl(0.0, 0) == 0.0


def test_bayesian_shrunk_pnl_large_sample_dominates() -> None:
    result = bayesian_shrunk_pnl(-100.0, 100, prior_pnl_per_trade=0.0, prior_strength=10.0)
    assert result < -0.80


def test_bayesian_shrunk_pnl_small_sample_shrinks_toward_prior() -> None:
    result = bayesian_shrunk_pnl(-9.0, 3, prior_pnl_per_trade=0.0, prior_strength=10.0)
    assert -3.0 < result < 0.0


def test_bayesian_shrunk_win_rate_no_samples() -> None:
    assert bayesian_shrunk_win_rate(0, 0, prior_win_rate=0.50) == 0.50


def test_bayesian_shrunk_win_rate_large_sample_dominates() -> None:
    result = bayesian_shrunk_win_rate(20, 100, prior_win_rate=0.50, prior_strength=10.0)
    assert 0.20 < result < 0.30


def test_bayesian_shrunk_win_rate_small_sample_shrinks_toward_prior() -> None:
    result = bayesian_shrunk_win_rate(0, 3, prior_win_rate=0.50, prior_strength=10.0)
    assert 0.30 < result < 0.50


def test_classify_small_sample_neg_pnl_returns_research_only() -> None:
    decision = classify_participation(
        historical_gate=HistoricalGateResult(
            allowed=False,
            reason="historical_prefix_small_sample_negative",
            sample_size=4,
            wilson_win_rate_lower_bound=0.05,
        ),
        pre_analysis_rejection_reason="historical_prefix_small_sample_negative",
        pre_analysis_metadata={
            "historical_gate_prefix_sample_size": 4,
            "historical_gate_prefix_win_rate": 0.25,
            "historical_gate_prefix_pnl_total": -8.0,
        },
    )
    assert decision.tier == ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE
    assert decision.what_to_learn_next is not None
    assert "4" in decision.what_to_learn_next


def test_classify_large_sample_neg_pnl_returns_research_only() -> None:
    decision = classify_participation(
        historical_gate=HistoricalGateResult(
            allowed=False,
            reason="historical_prefix_pnl_block",
            sample_size=15,
            wilson_win_rate_lower_bound=0.10,
        ),
        pre_analysis_rejection_reason="historical_prefix_pnl_block",
        pre_analysis_metadata={},
    )
    assert decision.tier == ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE


def test_classify_first_timeout_returns_operational_retry() -> None:
    decision = classify_participation(
        timeout_state=TimeoutState(
            timed_out=True,
            retriable=True,
            timeout_streak=1,
            search_profile="crypto",
        ),
    )
    assert decision.tier == ParticipationTier.OPERATIONAL_ERROR_RETRY
    assert "crypto" in (decision.what_to_learn_next or "")


def test_classify_repeated_timeout_returns_monitor_only() -> None:
    decision = classify_participation(
        timeout_state=TimeoutState(
            timed_out=True,
            retriable=True,
            timeout_streak=2,
            search_profile="generic",
        ),
    )
    assert decision.tier == ParticipationTier.MONITOR_ONLY


def test_classify_analysis_failure_retriable_returns_operational_error() -> None:
    decision = classify_participation(
        analysis_failed=True,
        analysis_error_retriable=True,
    )
    assert decision.tier == ParticipationTier.OPERATIONAL_ERROR_RETRY


def test_classify_analysis_failure_non_retriable_returns_monitor_only() -> None:
    decision = classify_participation(
        analysis_failed=True,
        analysis_error_retriable=False,
    )
    assert decision.tier == ParticipationTier.MONITOR_ONLY


def test_classify_score_soft_research_returns_research_only() -> None:
    decision = classify_participation(
        pre_analysis_rejection_reason="pre_analysis_score_soft_research",
        pre_analysis_metadata={"pre_analysis_score": 0.55},
    )
    assert decision.tier == ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE
    assert "source" in (decision.what_to_learn_next or "").lower()


def test_classify_should_trade_true_all_gates_pass() -> None:
    decision = classify_participation(
        decision_should_trade=True,
        decision_evidence_quality=0.80,
        evidence_quality_threshold=0.75,
        edge_value=0.10,
        edge_reasonable_max=0.35,
    )
    assert decision.tier == ParticipationTier.EXECUTION_ELIGIBLE


def test_classify_should_trade_true_evidence_quality_below_min() -> None:
    decision = classify_participation(
        decision_should_trade=True,
        decision_evidence_quality=0.60,
        evidence_quality_threshold=0.75,
    )
    assert decision.tier == ParticipationTier.SKIP_FOR_NOW_WITH_REASON
    assert "evidence_quality" in decision.primary_reason


def test_classify_should_trade_true_absence_only_is_research_gap() -> None:
    decision = classify_participation(
        decision_should_trade=True,
        decision_evidence_basis="absence_only",
        decision_edge_source="none",
        decision_evidence_quality=0.0,
        evidence_quality_threshold=0.75,
    )
    assert decision.tier == ParticipationTier.DEEP_RESEARCH_REQUIRED
    assert decision.primary_reason == "research_gap_not_market_judgment"
    assert decision.tier_metadata["blocked_conviction"] is True


def test_classify_definitive_outcome_bypasses_evidence_quality() -> None:
    decision = classify_participation(
        decision_should_trade=True,
        decision_definitive_outcome=True,
        decision_evidence_basis="direct",
        decision_primary_source_whitelisted=True,
        decision_evidence_quality=0.60,
        evidence_quality_threshold=0.75,
    )
    assert decision.tier == ParticipationTier.EXECUTION_ELIGIBLE


def test_classify_definitive_outcome_edge_049_passes() -> None:
    decision = classify_participation(
        decision_should_trade=True,
        decision_definitive_outcome=True,
        decision_evidence_basis="direct",
        decision_primary_source_whitelisted=True,
        decision_evidence_quality=0.85,
        evidence_quality_threshold=0.75,
        edge_value=0.49,
        edge_reasonable_max=0.35,
        definitive_edge_reasonable_max=0.65,
    )
    assert decision.tier == ParticipationTier.EXECUTION_ELIGIBLE


def test_classify_non_definitive_edge_049_blocked() -> None:
    decision = classify_participation(
        decision_should_trade=True,
        decision_evidence_quality=0.85,
        evidence_quality_threshold=0.75,
        edge_value=0.49,
        edge_reasonable_max=0.35,
    )
    assert decision.tier == ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE
    assert "edge_above_reasonable_max" in decision.primary_reason


def test_classify_should_trade_true_confidence_below_min_is_blocked_conviction() -> None:
    decision = classify_participation(
        decision_should_trade=True,
        decision_evidence_basis="direct",
        decision_edge_source="computed",
        decision_evidence_quality=0.90,
        evidence_quality_threshold=0.75,
        confidence_value=0.58,
        confidence_threshold=0.62,
        edge_value=0.10,
    )
    assert decision.tier == ParticipationTier.SKIP_FOR_NOW_WITH_REASON
    assert decision.primary_reason == "confidence_below_min"
    assert decision.tier_metadata["blocked_conviction"] is True
    assert decision.tier_metadata["skip_due_to"] == "weak_edge"


def test_classify_should_trade_true_daily_limit_is_monitor_only() -> None:
    decision = classify_participation(
        decision_should_trade=True,
        decision_evidence_basis="direct",
        decision_edge_source="computed",
        decision_evidence_quality=0.90,
        evidence_quality_threshold=0.75,
        confidence_value=0.70,
        confidence_threshold=0.62,
        edge_value=0.10,
        downstream_gate_reason="daily_limit_reached",
    )
    assert decision.tier == ParticipationTier.MONITOR_ONLY
    assert decision.primary_reason == "daily_limit_reached"
    assert decision.tier_metadata["skip_due_to"] == "risk_cap"


def test_classify_score_gate_blocked() -> None:
    decision = classify_participation(
        decision_should_trade=True,
        decision_evidence_quality=0.85,
        evidence_quality_threshold=0.75,
        edge_value=0.10,
        score_gate_blocked=True,
        score_gate_reason="score_gate_critical_rejection",
    )
    assert decision.tier == ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE


def test_classify_abstain_returns_research_only() -> None:
    decision = classify_participation(
        decision_abstain=True,
    )
    assert decision.tier == ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE


def test_classify_no_trade_recommended() -> None:
    decision = classify_participation(
        decision_should_trade=False,
    )
    assert decision.tier == ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE


def test_classify_awaiting_analysis() -> None:
    decision = classify_participation()
    assert decision.tier == ParticipationTier.DEEP_RESEARCH_REQUIRED


def test_to_metadata_tuple_for_demotion() -> None:
    decision = classify_participation(
        pre_analysis_rejection_reason="pre_analysis_repeated_churn_market",
        pre_analysis_metadata={"some_key": "some_val"},
    )
    demoted, reason, metadata = decision.to_metadata_tuple()
    assert demoted is True
    assert reason == "pre_analysis_repeated_churn_market"
    assert metadata["participation_demotion_reason"] == "pre_analysis_repeated_churn_market"
    assert metadata["participation_tier"] == str(ParticipationTier.SKIP_FOR_NOW_WITH_REASON)
    assert metadata["participation_terminal_reject"] is False


def test_to_metadata_tuple_for_execution() -> None:
    decision = classify_participation(
        decision_should_trade=True,
        decision_evidence_quality=0.90,
        evidence_quality_threshold=0.75,
        edge_value=0.10,
    )
    demoted, reason, metadata = decision.to_metadata_tuple()
    assert demoted is False
    assert reason is None
    assert metadata["participation_tier"] == str(ParticipationTier.EXECUTION_ELIGIBLE)
    assert metadata["participation_terminal_reject"] is False
    assert "participation_demotion_reason" not in metadata


def test_probe_trade_enabled_routes_winning_family_small_sample_to_execution_eligible():
    gate = HistoricalGateResult(
        allowed=False,
        reason="historical_prefix_small_sample_negative",
        metrics={"family": "sports"},
        sample_size=3,
    )
    decision = classify_participation(
        historical_gate=gate,
        pre_analysis_rejection_reason="pre_analysis_historical_prefix_small_sample_negative",
        pre_analysis_metadata={"participation_demotion_family": "sports"},
        probe_trade_enabled=True,
        proven_winning_families=frozenset({"sports"}),
    )
    assert decision.tier == ParticipationTier.EXECUTION_ELIGIBLE
    assert decision.tier_metadata.get("probe_trade") is True
    assert "probe_trade_small_sample_winning_family" in decision.primary_reason


def test_probe_trade_disabled_preserves_research_only_default():
    gate = HistoricalGateResult(
        allowed=False,
        reason="historical_prefix_small_sample_negative",
        metrics={"family": "sports"},
        sample_size=3,
    )
    decision = classify_participation(
        historical_gate=gate,
        pre_analysis_rejection_reason="pre_analysis_historical_prefix_small_sample_negative",
        pre_analysis_metadata={"participation_demotion_family": "sports"},
        probe_trade_enabled=False,
        proven_winning_families=frozenset({"sports"}),
    )
    assert decision.tier == ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE


def test_probe_trade_non_winning_family_stays_research_only():
    gate = HistoricalGateResult(
        allowed=False,
        reason="historical_prefix_small_sample_negative",
        metrics={"family": "crypto"},
        sample_size=3,
    )
    decision = classify_participation(
        historical_gate=gate,
        pre_analysis_rejection_reason="pre_analysis_historical_prefix_small_sample_negative",
        pre_analysis_metadata={"participation_demotion_family": "crypto"},
        probe_trade_enabled=True,
        proven_winning_families=frozenset({"sports"}),
    )
    assert decision.tier == ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE


def test_research_queued_does_not_set_terminal_reject_flag():
    """Research-queued tier is NOT a terminal rejection. The
    participation_terminal_reject flag must stay False so analytics can
    distinguish true terminal rejects from soft demotions."""
    decision = classify_participation(
        pre_analysis_rejection_reason="pre_analysis_historical_prefix_pnl_block",
        pre_analysis_metadata={"some_key": 42},
    )
    demoted, reason, metadata = decision.to_metadata_tuple()
    assert demoted is True
    assert metadata["participation_terminal_reject"] is False
    assert metadata["participation_demotion_reason"] == "pre_analysis_historical_prefix_pnl_block"
    assert metadata["participation_tier"] == str(ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE)


def test_terminal_reject_sets_terminal_reject_flag_true():
    """Only ParticipationTier.TERMINAL_REJECT should set
    participation_terminal_reject=True."""
    decision = ParticipationDecision(
        tier=ParticipationTier.TERMINAL_REJECT,
        primary_reason="market_untradable",
        why_not_execution_eligible="Market is permanently invalid",
    )
    _, _, metadata = decision.to_metadata_tuple()
    assert metadata["participation_terminal_reject"] is True
    assert metadata["participation_demotion_reason"] == "market_untradable"
    assert metadata["participation_tier"] == str(ParticipationTier.TERMINAL_REJECT)


def test_classify_daily_drawdown_blocked_returns_monitor_only():
    """daily_drawdown_blocked is a transient risk-cap signal, not a market
    judgment, so the participation tier should be MONITOR_ONLY (not
    RESEARCH_ONLY or TERMINAL_REJECT). The decision must surface a non-empty
    what_to_learn_next so receipts capture how to recover."""
    decision = classify_participation(
        pre_analysis_rejection_reason="pre_analysis_daily_drawdown_blocked",
        pre_analysis_metadata={
            "daily_drawdown_usdc": 25.5,
            "max_daily_drawdown_usdc": 20.0,
        },
    )
    assert decision.tier == ParticipationTier.MONITOR_ONLY
    assert decision.what_to_learn_next is not None
    assert "drawdown" in decision.what_to_learn_next.lower()
    demoted, reason, metadata = decision.to_metadata_tuple()
    assert demoted is True
    assert metadata["participation_terminal_reject"] is False
    assert metadata["participation_tier"] == str(ParticipationTier.MONITOR_ONLY)


def test_non_terminal_tiers_have_terminal_reject_false():
    """Spec: only TERMINAL_REJECT sets participation_terminal_reject=True.
    All other tiers must surface False."""
    non_terminal = [
        ParticipationTier.EXECUTION_ELIGIBLE,
        ParticipationTier.DEEP_RESEARCH_REQUIRED,
        ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE,
        ParticipationTier.MONITOR_ONLY,
        ParticipationTier.SKIP_FOR_NOW_WITH_REASON,
        ParticipationTier.OPERATIONAL_ERROR_RETRY,
    ]
    for tier in non_terminal:
        decision = ParticipationDecision(
            tier=tier,
            primary_reason="some_reason",
        )
        _, _, metadata = decision.to_metadata_tuple()
        assert metadata["participation_terminal_reject"] is False, (
            f"Tier {tier} should NOT set participation_terminal_reject=True"
        )


def test_classify_no_trade_with_absence_only_is_research_gap_not_market_judgment() -> None:
    """should_trade=False with the (eq=0/edge_source=none/basis=absence_only)
    placeholder triplet must be classified as a research gap, not as a no-trade
    market verdict. This protects the system from treating Grok's evidence
    failures as market-quality decisions."""
    decision = classify_participation(
        decision_should_trade=False,
        decision_evidence_basis="absence_only",
        decision_edge_source="none",
        decision_evidence_quality=0.0,
    )
    assert decision.tier == ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE
    assert decision.primary_reason == "no_trade_research_gap"
    assert decision.tier_metadata["evidence_gap_classified_as_research"] is True
    assert decision.tier_metadata["skip_due_to"] == "lack_of_evidence"
    assert "research gap" in (decision.why_not_execution_eligible or "").lower()


def test_classify_no_trade_with_real_evidence_stays_no_trade_recommended() -> None:
    """A genuine market judgment (direct evidence, computed edge) where the
    model still abstains must not be reclassified as a research gap."""
    decision = classify_participation(
        decision_should_trade=False,
        decision_evidence_basis="direct",
        decision_edge_source="computed",
        decision_evidence_quality=0.80,
    )
    assert decision.tier == ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE
    assert decision.primary_reason == "no_trade_recommended"


def test_classify_score_far_below_min_routes_to_skip_with_reason() -> None:
    """pre_analysis_score_far_below_min should not pollute the learning queue
    (the soft-research band is the right place for near-miss markets); it
    routes to SKIP_FOR_NOW_WITH_REASON with an actionable learning hint."""
    decision = classify_participation(
        pre_analysis_rejection_reason="pre_analysis_score_far_below_min",
        pre_analysis_metadata={
            "pre_analysis_score": 0.10,
            "pre_analysis_threshold": 0.42,
            "pre_analysis_threshold_gap": 0.32,
        },
    )
    assert decision.tier == ParticipationTier.SKIP_FOR_NOW_WITH_REASON
    assert decision.what_to_learn_next is not None
    assert "structural" in decision.what_to_learn_next.lower()


def test_classify_score_soft_research_includes_threshold_gap_hint() -> None:
    """Receipts in the learning queue should expose how close a market was
    to passing the gate so operators can scan for near-misses without
    rerunning analytics scripts."""
    decision = classify_participation(
        pre_analysis_rejection_reason="pre_analysis_score_soft_research",
        pre_analysis_metadata={
            "pre_analysis_score": 0.376,
            "pre_analysis_threshold": 0.42,
            "pre_analysis_threshold_gap": 0.044,
        },
    )
    assert decision.tier == ParticipationTier.RESEARCH_ONLY_LEARNING_QUEUE
    assert decision.what_to_learn_next is not None
    assert "threshold_gap" in decision.what_to_learn_next
    assert "0.044" in decision.what_to_learn_next


def test_classify_timeout_metadata_includes_operational_context() -> None:
    """Timeout-routed markets must surface the search profile, retriable
    flag, and timeout streak in tier_metadata so persistence captures the
    operational context for retry/monitor decisions."""
    decision = classify_participation(
        timeout_state=TimeoutState(
            timed_out=True,
            retriable=True,
            timeout_streak=1,
            search_profile="crypto",
        ),
    )
    _, _, metadata = decision.to_metadata_tuple()
    assert metadata["operational_search_profile"] == "crypto"
    assert metadata["operational_timeout_streak"] == 1
    assert metadata["operational_retriable"] is True


def test_classify_repeated_timeout_metadata_marks_non_retriable_streak() -> None:
    """Second timeout in a row should escalate to MONITOR_ONLY and the
    streak count must persist into receipts."""
    decision = classify_participation(
        timeout_state=TimeoutState(
            timed_out=True,
            retriable=True,
            timeout_streak=2,
            search_profile="weather",
        ),
    )
    assert decision.tier == ParticipationTier.MONITOR_ONLY
    _, _, metadata = decision.to_metadata_tuple()
    assert metadata["operational_search_profile"] == "weather"
    assert metadata["operational_timeout_streak"] == 2


def test_classify_analysis_failure_metadata_marks_retriable() -> None:
    """Non-timeout analysis failures must capture retriable status in
    tier_metadata for downstream operational alerting."""
    decision = classify_participation(
        analysis_failed=True,
        analysis_error_retriable=False,
    )
    assert decision.tier == ParticipationTier.MONITOR_ONLY
    _, _, metadata = decision.to_metadata_tuple()
    assert metadata["operational_retriable"] is False
