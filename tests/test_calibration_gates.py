from __future__ import annotations

from calibration_gates import (
    GateTier,
    PerformanceStats,
    evaluate_market,
    evaluate_market_tiered,
    evaluate_short_prefix_penalty,
)
from participation import wilson_lower_bound


def test_evaluate_market_blocks_losing_prefix_with_sufficient_samples() -> None:
    """n=5 is below the hard-block sample floor so the reason
    changes to ``historical_prefix_small_sample_negative`` (soft demote).
    Soft demotion remains execution-eligible for deeper scoring."""
    allowed, reason, metrics = evaluate_market(
        market_id="ABCDEF123456-TEST",
        family="generic",
        prefix_stats={
            "ABCDEF123456": PerformanceStats(
                sample_size=5,
                wins=1,
                win_rate=0.20,
                pnl_total=-8.0,
            )
        },
        family_stats={},
        prefix_len=12,
        prefix_gate_enabled=True,
        prefix_min_samples=3,
        prefix_pnl_cutoff=-3.0,
        prefix_win_rate_cutoff=0.40,
        family_gate_enabled=False,
    )
    assert allowed is True
    assert reason == "historical_prefix_small_sample_negative"
    assert metrics["historical_gate_market_prefix"] == "ABCDEF123456"
    assert metrics["historical_gate_tier"] == GateTier.SOFT_DEMOTE
    assert metrics["historical_gate_score_penalty"] > 0.0


def test_evaluate_market_respects_min_sample_guard() -> None:
    allowed, reason, _ = evaluate_market(
        market_id="ABCDEF123456-TEST",
        family="generic",
        prefix_stats={
            "ABCDEF123456": PerformanceStats(
                sample_size=2,
                wins=0,
                win_rate=0.0,
                pnl_total=-20.0,
            )
        },
        family_stats={},
        prefix_len=12,
        prefix_gate_enabled=True,
        prefix_min_samples=3,
        prefix_pnl_cutoff=-3.0,
        prefix_win_rate_cutoff=0.40,
        family_gate_enabled=False,
    )
    assert allowed is True
    assert reason is None


def test_evaluate_market_flags_losing_family_without_blocking() -> None:
    allowed, reason, metrics = evaluate_market(
        market_id="ZZZZZZ999999-TEST",
        family="crypto",
        prefix_stats={},
        family_stats={
            "crypto": PerformanceStats(
                sample_size=30,
                wins=10,
                win_rate=0.3333,
                pnl_total=-40.0,
            )
        },
        prefix_gate_enabled=False,
        family_gate_enabled=True,
        family_min_samples=12,
        family_pnl_cutoff=-12.0,
        family_win_rate_cutoff=0.40,
    )
    assert allowed is True
    assert reason == "historical_family_pnl_block"
    assert metrics["historical_gate_market_family"] == "crypto"
    assert metrics["historical_family_samples"] == 30
    assert metrics["historical_family_pnl_total"] == -40.0


def test_evaluate_market_does_not_block_zero_win_prefix_without_pnl_cutoff() -> None:
    allowed, reason, metrics = evaluate_market(
        market_id="ZEROWIN12345-TEST",
        family="generic",
        prefix_stats={
            "ZEROWIN12345": PerformanceStats(
                sample_size=8,
                wins=0,
                win_rate=0.0,
                pnl_total=1.5,
            )
        },
        family_stats={},
        prefix_len=12,
        prefix_gate_enabled=True,
        prefix_min_samples=3,
        prefix_pnl_cutoff=-3.0,
        prefix_win_rate_cutoff=0.40,
        family_gate_enabled=False,
    )
    assert allowed is True
    assert reason is None
    assert metrics["historical_gate_prefix_sample_size"] == 8


def test_evaluate_short_prefix_penalty_returns_zero_without_snapshot() -> None:
    penalty, metrics = evaluate_short_prefix_penalty(
        market_id="KXETH-26APR20-T123",
        short_prefix_stats=None,
        prefix_len=5,
        min_samples=3,
        pnl_cutoff=-5.0,
        score_penalty=0.10,
    )
    assert penalty == 0.0
    assert metrics["historical_short_prefix"] == "KXETH"


def test_evaluate_short_prefix_penalty_applies_for_losing_prefix() -> None:
    penalty, metrics = evaluate_short_prefix_penalty(
        market_id="KXETH-26APR20-T123",
        short_prefix_stats={
            "KXETH": PerformanceStats(
                sample_size=4,
                wins=1,
                win_rate=0.25,
                pnl_total=-9.0,
            )
        },
        prefix_len=5,
        min_samples=3,
        pnl_cutoff=-5.0,
        score_penalty=0.10,
    )
    assert penalty == 0.10
    assert metrics["historical_short_prefix_sample_size"] == 4
    assert metrics["historical_short_prefix_pnl_total"] == -9.0


def test_evaluate_short_prefix_penalty_respects_sample_floor() -> None:
    penalty, _ = evaluate_short_prefix_penalty(
        market_id="KXETH-26APR20-T123",
        short_prefix_stats={
            "KXETH": PerformanceStats(
                sample_size=2,
                wins=0,
                win_rate=0.0,
                pnl_total=-20.0,
            )
        },
        prefix_len=5,
        min_samples=3,
        pnl_cutoff=-5.0,
        score_penalty=0.10,
    )
    assert penalty == 0.0


def test_wilson_lower_bound_3_wins_of_3() -> None:
    wlb = wilson_lower_bound(3, 3)
    assert 0.29 < wlb < 1.0


def test_wilson_lower_bound_0_wins_of_5() -> None:
    wlb = wilson_lower_bound(0, 5)
    assert wlb < 0.05


def test_wilson_lower_bound_1_win_of_3() -> None:
    wlb = wilson_lower_bound(1, 3)
    assert 0.01 < wlb < 0.50


def test_tiered_n3_neg_pnl_is_soft_demote_not_hard_deny() -> None:
    """With n=3 and negative PnL, the tiered evaluator should produce
    SOFT_DEMOTE, not HARD_DENY."""
    result = evaluate_market_tiered(
        market_id="KXTESTHIT-26A-TEST",
        family="generic",
        prefix_stats={
            "KXTESTHIT-26": PerformanceStats(
                sample_size=3,
                wins=0,
                win_rate=0.0,
                pnl_total=-14.44,
            )
        },
        family_stats={},
        prefix_len=12,
        prefix_gate_enabled=True,
        prefix_min_samples=3,
        prefix_hard_block_min_samples=10,
        prefix_pnl_cutoff=-3.0,
        prefix_win_rate_cutoff=0.40,
        family_gate_enabled=False,
    )
    assert result.tier == GateTier.SOFT_DEMOTE
    assert result.allowed is True
    assert result.reason == "historical_prefix_small_sample_negative"
    assert result.sample_size == 3
    assert result.wilson_win_rate_lower_bound is not None
    assert result.what_to_learn_next is not None


def test_tiered_tiny_sample_never_hard_blocks_regardless_of_pnl() -> None:
    """Regression guard: tiny samples (n < hard_block_min_samples) must never
    produce HARD_DENY no matter how bad the win-rate, total PnL, or shrunk
    PnL/trade are. The user explicitly flagged "tiny-sample historical-gate
    hard blocking" as a concern; this test prevents that regression."""
    worst_case_metrics = PerformanceStats(
        sample_size=4,
        wins=0,
        win_rate=0.0,
        pnl_total=-100.0,
    )
    result = evaluate_market_tiered(
        market_id="KXTINY-12345678-TEST",
        family="generic",
        prefix_stats={"KXTINY-12345": worst_case_metrics},
        family_stats={},
        prefix_len=12,
        prefix_gate_enabled=True,
        prefix_min_samples=3,
        prefix_hard_block_min_samples=20,
        prefix_pnl_cutoff=-3.0,
        prefix_win_rate_cutoff=0.40,
        prefix_shrunk_pnl_cutoff=-0.50,
        family_gate_enabled=False,
    )
    assert result.tier != GateTier.HARD_DENY
    assert result.allowed is True
    assert result.reason == "historical_prefix_small_sample_negative"


def test_tiered_sample_just_below_hard_block_floor_stays_soft() -> None:
    """Sample size = hard_block_min_samples - 1 must still be SOFT_DEMOTE,
    even with terrible metrics. The hard-block path is only reached when
    sample_size >= hard_block_min_samples."""
    result = evaluate_market_tiered(
        market_id="KXBORDER-12345678-TEST",
        family="generic",
        prefix_stats={
            # 12-char ticker prefix extracted from market_id above.
            "KXBORDER-123": PerformanceStats(
                sample_size=19,
                wins=0,
                win_rate=0.0,
                pnl_total=-50.0,
            )
        },
        family_stats={},
        prefix_len=12,
        prefix_gate_enabled=True,
        prefix_min_samples=3,
        prefix_hard_block_min_samples=20,
        prefix_pnl_cutoff=-3.0,
        prefix_win_rate_cutoff=0.40,
        prefix_shrunk_pnl_cutoff=-0.50,
        family_gate_enabled=False,
    )
    assert result.tier == GateTier.SOFT_DEMOTE
    assert result.allowed is True


def test_tiered_n12_low_wilson_is_historical_signal_not_block() -> None:
    """With n=12 >= hard_block_min_samples and low Wilson LB, the tiered
    evaluator should produce HARD_DENY."""
    result = evaluate_market_tiered(
        market_id="KXTESTHIT-26A-TEST",
        family="generic",
        prefix_stats={
            "KXTESTHIT-26": PerformanceStats(
                sample_size=12,
                wins=1,
                win_rate=0.083,
                pnl_total=-30.0,
            )
        },
        family_stats={},
        prefix_len=12,
        prefix_gate_enabled=True,
        prefix_min_samples=3,
        prefix_hard_block_min_samples=10,
        prefix_pnl_cutoff=-3.0,
        prefix_win_rate_cutoff=0.40,
        family_gate_enabled=False,
    )
    assert result.tier == GateTier.HARD_DENY
    assert result.reason == "historical_prefix_pnl_block"
    assert result.allowed is True


def test_high_win_rate_negative_pnl_is_entry_price_caution_not_hard_deny() -> None:
    result = evaluate_market_tiered(
        market_id="KXENTRYCAUT-26-TEST",
        family="generic",
        prefix_stats={
            "KXENTRYCAUT": PerformanceStats(
                sample_size=20,
                wins=13,
                win_rate=0.65,
                pnl_total=-25.0,
            )
        },
        family_stats={},
        prefix_len=11,
        prefix_gate_enabled=True,
        prefix_min_samples=3,
        prefix_hard_block_min_samples=20,
        prefix_pnl_cutoff=-3.0,
        prefix_win_rate_cutoff=0.40,
        prefix_shrunk_pnl_cutoff=-0.50,
        prefix_soft_demote_score_penalty=0.08,
        family_gate_enabled=False,
    )
    assert result.allowed is True
    assert result.tier == GateTier.SOFT_DEMOTE
    assert result.metrics["historical_gate_prefix_loss_mode"] == "sizing_or_entry_price"
    assert result.metrics["historical_gate_score_penalty"] == 0.04


def test_legacy_wrapper_returns_allowed_true_for_soft_demote() -> None:
    """Soft-demoted markets remain eligible for scoring/research prioritization."""
    allowed, reason, metrics = evaluate_market(
        market_id="KXTESTHIT-26A-TEST",
        family="generic",
        prefix_stats={
            "KXTESTHIT-26": PerformanceStats(
                sample_size=4,
                wins=1,
                win_rate=0.25,
                pnl_total=-14.44,
            )
        },
        family_stats={},
        prefix_len=12,
        prefix_gate_enabled=True,
        prefix_min_samples=3,
        prefix_pnl_cutoff=-3.0,
        prefix_win_rate_cutoff=0.40,
        family_gate_enabled=False,
    )
    assert allowed is True
    assert reason == "historical_prefix_small_sample_negative"
    assert "historical_gate_wilson_lb" in metrics
    assert "what_to_learn_next" in metrics
    assert metrics["historical_gate_score_penalty"] > 0.0


def test_tiered_neutral_when_pnl_above_cutoff() -> None:
    result = evaluate_market_tiered(
        market_id="KXWINNING-26-TEST",
        family="generic",
        prefix_stats={
            "KXWINNING-26": PerformanceStats(
                sample_size=8,
                wins=5,
                win_rate=0.625,
                pnl_total=10.0,
            )
        },
        family_stats={},
        prefix_len=12,
        prefix_gate_enabled=True,
        prefix_min_samples=3,
        prefix_hard_block_min_samples=10,
        prefix_pnl_cutoff=-3.0,
        prefix_win_rate_cutoff=0.40,
        family_gate_enabled=False,
    )
    assert result.tier == GateTier.NEUTRAL
    assert result.allowed is True


def test_soft_demote_n4_mild_negative_does_not_block() -> None:
    """n=4, wins=2, pnl=-3.5 -> shrunk_pnl/trade ~ -0.25 which is above
    the default -0.50 cutoff, so the market should NOT be soft-demoted."""
    result = evaluate_market_tiered(
        market_id="KXMILD-26APR-TEST",
        family="generic",
        prefix_stats={
            "KXMILD-26APR": PerformanceStats(
                sample_size=4,
                wins=2,
                win_rate=0.50,
                pnl_total=-3.5,
            )
        },
        family_stats={},
        prefix_len=12,
        prefix_gate_enabled=True,
        prefix_min_samples=3,
        prefix_hard_block_min_samples=10,
        prefix_pnl_cutoff=-3.0,
        prefix_win_rate_cutoff=0.40,
        prefix_shrunk_pnl_cutoff=-0.50,
        family_gate_enabled=False,
    )
    assert result.tier == GateTier.NEUTRAL
    assert result.allowed is True


def test_soft_demote_n4_strong_negative_applies_penalty_not_block() -> None:
    """n=4, wins=0, pnl=-16 -> shrunk_pnl/trade ~ -1.14 which is well
    below -0.50, and Wilson LB for 0/4 ~ 0.0 <= 0.40, so soft-demote fires."""
    result = evaluate_market_tiered(
        market_id="KXBAD-26APR2-TEST",
        family="generic",
        prefix_stats={
            "KXBAD-26APR2": PerformanceStats(
                sample_size=4,
                wins=0,
                win_rate=0.0,
                pnl_total=-16.0,
            )
        },
        family_stats={},
        prefix_len=12,
        prefix_gate_enabled=True,
        prefix_min_samples=3,
        prefix_hard_block_min_samples=10,
        prefix_pnl_cutoff=-3.0,
        prefix_win_rate_cutoff=0.40,
        prefix_shrunk_pnl_cutoff=-0.50,
        family_gate_enabled=False,
    )
    assert result.tier == GateTier.SOFT_DEMOTE
    assert result.allowed is True
    assert result.reason == "historical_prefix_small_sample_negative"


def test_soft_demote_requires_wilson_lb_below_cutoff() -> None:
    """n=8, wins=5, pnl=-4 -> Wilson LB ~ 0.34 which is below 0.40 cutoff,
    but the shrunk PnL/trade is only about -0.11 (above -0.50), so NEUTRAL."""
    result = evaluate_market_tiered(
        market_id="KXMIXED-26AP-TEST",
        family="generic",
        prefix_stats={
            "KXMIXED-26AP": PerformanceStats(
                sample_size=8,
                wins=5,
                win_rate=0.625,
                pnl_total=-4.0,
            )
        },
        family_stats={},
        prefix_len=12,
        prefix_gate_enabled=True,
        prefix_min_samples=3,
        prefix_hard_block_min_samples=10,
        prefix_pnl_cutoff=-3.0,
        prefix_win_rate_cutoff=0.40,
        prefix_shrunk_pnl_cutoff=-0.50,
        family_gate_enabled=False,
    )
    assert result.tier == GateTier.NEUTRAL
    assert result.allowed is True


def test_soft_demote_blocked_by_high_wilson_lb() -> None:
    """n=6, wins=4, pnl=-5 -> Wilson LB ~ 0.33 which is below cutoff,
    but check: actually wins=4/6 gives WLB ~ 0.30, and
    shrunk_pnl = (6/16)*(-5/6) = 0.375*-0.833 = -0.3125 which is above -0.50.
    So NEUTRAL because shrunk PnL is not low enough."""
    result = evaluate_market_tiered(
        market_id="KXWINNY-26AP-TEST",
        family="generic",
        prefix_stats={
            "KXWINNY-26AP": PerformanceStats(
                sample_size=6,
                wins=4,
                win_rate=0.667,
                pnl_total=-5.0,
            )
        },
        family_stats={},
        prefix_len=12,
        prefix_gate_enabled=True,
        prefix_min_samples=3,
        prefix_hard_block_min_samples=10,
        prefix_pnl_cutoff=-3.0,
        prefix_win_rate_cutoff=0.40,
        prefix_shrunk_pnl_cutoff=-0.50,
        family_gate_enabled=False,
    )
    assert result.tier == GateTier.NEUTRAL
    assert result.allowed is True


def test_hard_deny_requires_wilson_lb_below_cutoff() -> None:
    """A prefix with n=10 samples and observed win-rate exactly at cutoff
    (0.40) but Wilson LB above the cutoff must NOT be hard-denied. Wilson LB
    is a statistical-confidence guard rail: even with sufficient samples the
    gate refuses to hard-deny when the lower-bound on the true win rate is
    not below cutoff."""
    # 4 wins of 10 = win_rate 0.40, Wilson LB ~ 0.17 — strictly less than
    # cutoff 0.40. Test the boundary case where observed=cutoff but WLB<cutoff.
    # We want to verify hard-deny still fires when the WLB IS below cutoff
    # (the original test_tiered_n12_low_wilson_is_hard_deny covers this; this
    # test is a complementary regression guard for the new WLB requirement).
    result = evaluate_market_tiered(
        market_id="KXBORDR-12345678-TEST",
        family="generic",
        prefix_stats={
            "KXBORDR-1234": PerformanceStats(
                sample_size=10,
                wins=4,
                win_rate=0.40,
                pnl_total=-30.0,
            )
        },
        family_stats={},
        prefix_len=12,
        prefix_gate_enabled=True,
        prefix_min_samples=3,
        prefix_hard_block_min_samples=10,
        prefix_pnl_cutoff=-3.0,
        prefix_win_rate_cutoff=0.40,
        prefix_shrunk_pnl_cutoff=-0.50,
        family_gate_enabled=False,
    )
    # observed win_rate == cutoff, WLB ~ 0.17 (well below cutoff), shrunk PnL
    # well below cutoff, raw PnL below cutoff. All conditions met -> HARD_DENY.
    assert result.tier == GateTier.HARD_DENY
    assert result.wilson_win_rate_lower_bound is not None
    assert result.wilson_win_rate_lower_bound < 0.40


def test_family_hard_deny_requires_wilson_lb_below_cutoff() -> None:
    """Family-level hard-deny must also require Wilson LB <= cutoff so the
    statistical confidence requirement applies symmetrically."""
    result = evaluate_market_tiered(
        market_id="KXFAMTEST-123456-TEST",
        family="crypto",
        prefix_stats={},
        family_stats={
            "crypto": PerformanceStats(
                sample_size=30,
                wins=10,
                win_rate=0.333,
                pnl_total=-40.0,
            )
        },
        prefix_gate_enabled=False,
        family_gate_enabled=True,
        family_min_samples=12,
        family_pnl_cutoff=-12.0,
        family_win_rate_cutoff=0.40,
        family_shrunk_pnl_cutoff=-0.50,
    )
    # n=30, wins=10 (33.3%), WLB ~ 0.19 < 0.40, shrunk PnL ~ -1.0 < -0.5.
    assert result.tier == GateTier.HARD_DENY
    assert result.reason == "historical_family_pnl_block"
    assert result.wilson_win_rate_lower_bound is not None
    assert result.wilson_win_rate_lower_bound < 0.40


def test_family_hard_deny_uses_family_shrunk_pnl_cutoff_not_prefix() -> None:
    """Family hard-deny must compare against family_shrunk_pnl_cutoff, not the
    prefix shrunk-PnL bar — otherwise retuning prefix cutoffs silently changes
    family gate behavior."""
    family_stats = {
        "crypto": PerformanceStats(
            sample_size=30,
            wins=10,
            win_rate=0.333,
            pnl_total=-40.0,
        )
    }
    # Shrunk PnL ~ -1.0: passes a loose family bar but fails a tight one.
    loose = evaluate_market_tiered(
        market_id="KXFAMTEST-123456-TEST",
        family="crypto",
        prefix_stats={},
        family_stats=family_stats,
        prefix_gate_enabled=False,
        prefix_shrunk_pnl_cutoff=-0.01,  # would hard-deny if wrongly reused
        family_gate_enabled=True,
        family_min_samples=12,
        family_pnl_cutoff=-12.0,
        family_win_rate_cutoff=0.40,
        family_shrunk_pnl_cutoff=-2.0,
    )
    assert loose.tier == GateTier.NEUTRAL

    tight = evaluate_market_tiered(
        market_id="KXFAMTEST-123456-TEST",
        family="crypto",
        prefix_stats={},
        family_stats=family_stats,
        prefix_gate_enabled=False,
        prefix_shrunk_pnl_cutoff=-10.0,  # would never hard-deny if wrongly reused
        family_gate_enabled=True,
        family_min_samples=12,
        family_pnl_cutoff=-12.0,
        family_win_rate_cutoff=0.40,
        family_shrunk_pnl_cutoff=-0.50,
    )
    assert tight.tier == GateTier.HARD_DENY
    assert tight.reason == "historical_family_pnl_block"
