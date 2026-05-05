from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from market_state import MarketStateManager
from participation import wilson_lower_bound, bayesian_shrunk_pnl


@dataclass(frozen=True)
class PerformanceStats:
    sample_size: int
    wins: int
    win_rate: float
    pnl_total: float


class GateTier:
    HARD_DENY = "hard_deny"
    SOFT_DEMOTE = "soft_demote"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class EvaluateMarketResult:
    tier: str
    allowed: bool
    reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    wilson_win_rate_lower_bound: float | None = None
    shrunk_pnl_per_trade: float | None = None
    sample_size: int | None = None
    what_to_learn_next: str | None = None


def _cutoff_iso(lookback_days: int) -> str:
    days = max(1, int(lookback_days))
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return cutoff.isoformat()


def load_ticker_prefix_stats(
    manager: MarketStateManager,
    *,
    prefix_len: int = 12,
    lookback_days: int = 30,
) -> dict[str, PerformanceStats]:
    normalized_prefix_len = max(1, int(prefix_len))
    cutoff = _cutoff_iso(lookback_days)
    rows = manager._conn.execute(  # noqa: SLF001 - local read-only analytics helper
        """
        SELECT
            SUBSTR(UPPER(COALESCE(market_id, '')), 1, ?) AS ticker_prefix,
            COUNT(*) AS sample_size,
            SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END) AS wins,
            SUM(COALESCE(pnl_estimate, 0.0)) AS pnl_total
        FROM trade_outcomes
        WHERE won IS NOT NULL
          AND COALESCE(resolved_at, last_updated, '') >= ?
          AND COALESCE(market_id, '') <> ''
        GROUP BY ticker_prefix
        """,
        (normalized_prefix_len, cutoff),
    ).fetchall()
    snapshot: dict[str, PerformanceStats] = {}
    for row in rows:
        prefix = str(row["ticker_prefix"] or "").strip().upper()
        if not prefix:
            continue
        sample_size = int(row["sample_size"] or 0)
        wins = int(row["wins"] or 0)
        pnl_total = float(row["pnl_total"] or 0.0)
        snapshot[prefix] = PerformanceStats(
            sample_size=sample_size,
            wins=wins,
            win_rate=(wins / sample_size) if sample_size > 0 else 0.0,
            pnl_total=pnl_total,
        )
    return snapshot


def load_short_prefix_stats(
    manager: MarketStateManager,
    *,
    prefix_len: int = 5,
    lookback_days: int = 30,
) -> dict[str, PerformanceStats]:
    """Load compact ticker-prefix performance for soft score penalties."""
    return load_ticker_prefix_stats(
        manager,
        prefix_len=prefix_len,
        lookback_days=lookback_days,
    )


def load_family_stats(
    manager: MarketStateManager,
    *,
    lookback_days: int = 30,
) -> dict[str, PerformanceStats]:
    cutoff = _cutoff_iso(lookback_days)
    rows = manager._conn.execute(  # noqa: SLF001 - local read-only analytics helper
        """
        SELECT
            t.market_id AS market_id,
            COALESCE(m.question, '') AS question,
            COALESCE(m.category, '') AS category,
            t.won AS won,
            COALESCE(t.pnl_estimate, 0.0) AS pnl_estimate
        FROM trade_outcomes t
        LEFT JOIN markets m ON m.id = t.market_id
        WHERE t.won IS NOT NULL
          AND COALESCE(t.resolved_at, t.last_updated, '') >= ?
        """,
        (cutoff,),
    ).fetchall()
    grouped: dict[str, dict[str, float | int]] = {}
    for row in rows:
        family = MarketStateManager._infer_family_from_state_row(  # noqa: SLF001
            market_id=str(row["market_id"] or ""),
            question=str(row["question"] or ""),
            category=str(row["category"] or ""),
        )
        bucket = grouped.setdefault(
            family,
            {"sample_size": 0, "wins": 0, "pnl_total": 0.0},
        )
        bucket["sample_size"] = int(bucket["sample_size"]) + 1
        if int(row["won"] or 0) == 1:
            bucket["wins"] = int(bucket["wins"]) + 1
        bucket["pnl_total"] = float(bucket["pnl_total"]) + float(row["pnl_estimate"] or 0.0)
    snapshot: dict[str, PerformanceStats] = {}
    for family, raw in grouped.items():
        sample_size = int(raw["sample_size"])
        wins = int(raw["wins"])
        pnl_total = float(raw["pnl_total"])
        snapshot[family] = PerformanceStats(
            sample_size=sample_size,
            wins=wins,
            win_rate=(wins / sample_size) if sample_size > 0 else 0.0,
            pnl_total=pnl_total,
        )
    return snapshot


def evaluate_short_prefix_penalty(
    *,
    market_id: str,
    short_prefix_stats: dict[str, PerformanceStats] | None,
    prefix_len: int = 5,
    min_samples: int = 3,
    pnl_cutoff: float = -5.0,
    score_penalty: float = 0.10,
) -> tuple[float, dict[str, Any]]:
    normalized_market_id = str(market_id or "").strip().upper()
    normalized_prefix_len = max(1, int(prefix_len))
    market_prefix = normalized_market_id[:normalized_prefix_len]
    metrics: dict[str, Any] = {
        "historical_short_prefix": market_prefix,
    }
    if not short_prefix_stats:
        return 0.0, metrics
    prefix_snapshot = short_prefix_stats.get(market_prefix)
    if prefix_snapshot is None:
        return 0.0, metrics
    metrics.update(
        {
            "historical_short_prefix_sample_size": prefix_snapshot.sample_size,
            "historical_short_prefix_win_rate": prefix_snapshot.win_rate,
            "historical_short_prefix_pnl_total": prefix_snapshot.pnl_total,
        }
    )
    if (
        prefix_snapshot.sample_size >= max(1, int(min_samples))
        and prefix_snapshot.pnl_total <= float(pnl_cutoff)
    ):
        return max(0.0, float(score_penalty)), metrics
    return 0.0, metrics


def evaluate_market_tiered(
    *,
    market_id: str,
    family: str,
    prefix_stats: dict[str, PerformanceStats] | None,
    family_stats: dict[str, PerformanceStats] | None,
    prefix_len: int = 12,
    prefix_gate_enabled: bool = True,
    prefix_min_samples: int = 3,
    prefix_hard_block_min_samples: int = 20,
    prefix_pnl_cutoff: float = -3.0,
    prefix_win_rate_cutoff: float = 0.40,
    prefix_shrinkage_enabled: bool = True,
    prefix_prior_win_rate: float = 0.50,
    prefix_prior_strength: float = 10.0,
    prefix_shrunk_pnl_cutoff: float = -0.50,
    prefix_soft_demote_score_penalty: float = 0.08,
    family_gate_enabled: bool = True,
    family_min_samples: int = 12,
    family_pnl_cutoff: float = -12.0,
    family_win_rate_cutoff: float = 0.40,
) -> EvaluateMarketResult:
    """Tiered market evaluation using Wilson lower-bound and Bayesian PnL shrinkage."""
    normalized_market_id = str(market_id or "").strip().upper()
    normalized_family = str(family or "").strip().lower()
    normalized_prefix_len = max(1, int(prefix_len))
    market_prefix = normalized_market_id[:normalized_prefix_len]
    metrics: dict[str, Any] = {
        "historical_gate_market_prefix": market_prefix,
        "historical_gate_market_family": normalized_family,
        "historical_gate_prefix_len": normalized_prefix_len,
        "historical_gate_prefix_specificity": (
            "event_prefix" if normalized_prefix_len >= 12 else "short_family_prefix"
        ),
        "historical_gate_loss_source_uncertain": True,
    }

    if prefix_gate_enabled and prefix_stats:
        prefix_snapshot = prefix_stats.get(market_prefix)
        if prefix_snapshot is not None:
            n = prefix_snapshot.sample_size
            wlb = wilson_lower_bound(prefix_snapshot.wins, n)
            shrunk_pnl = bayesian_shrunk_pnl(
                prefix_snapshot.pnl_total,
                n,
                prior_pnl_per_trade=0.0,
                prior_strength=prefix_prior_strength,
            ) if prefix_shrinkage_enabled else (
                prefix_snapshot.pnl_total / n if n > 0 else 0.0
            )
            metrics.update(
                {
                    "historical_gate_prefix_sample_size": n,
                    "historical_gate_prefix_win_rate": prefix_snapshot.win_rate,
                    "historical_gate_prefix_pnl_total": prefix_snapshot.pnl_total,
                    "historical_gate_prefix_wilson_lb": round(wlb, 4),
                    "historical_gate_prefix_shrunk_pnl_per_trade": round(shrunk_pnl, 4),
                }
            )

            soft_min = max(1, int(prefix_min_samples))
            hard_block_min = max(soft_min, int(prefix_hard_block_min_samples))
            sample_weight = min(1.0, n / hard_block_min) if hard_block_min > 0 else 1.0
            score_penalty = max(0.0, float(prefix_soft_demote_score_penalty)) * sample_weight
            metrics.update(
                {
                    "historical_gate_sample_weight": round(sample_weight, 4),
                    "historical_gate_score_penalty": round(score_penalty, 4),
                }
            )

            # Hard-deny requires both observed and Wilson-LB win rates to be
            # below the cutoff. Adding the Wilson-LB requirement enforces
            # statistical confidence on top of observed win-rate / PnL signals
            # so even a sufficient-sample prefix is not hard-blocked unless
            # the lower bound on its true win rate is below cutoff.
            if (
                n >= hard_block_min
                and prefix_snapshot.win_rate <= float(prefix_win_rate_cutoff)
                and wlb <= float(prefix_win_rate_cutoff)
                and shrunk_pnl <= float(prefix_shrunk_pnl_cutoff)
                and prefix_snapshot.pnl_total <= float(prefix_pnl_cutoff)
            ):
                return EvaluateMarketResult(
                    tier=GateTier.HARD_DENY,
                    allowed=False,
                    reason="historical_prefix_pnl_block",
                    metrics=metrics,
                    wilson_win_rate_lower_bound=wlb,
                    shrunk_pnl_per_trade=shrunk_pnl,
                    sample_size=n,
                    what_to_learn_next=(
                        f"Prefix '{market_prefix}' has {n} samples, "
                        f"observed win_rate={prefix_snapshot.win_rate:.2f}, "
                        f"Wilson LB={wlb:.2f}, shrunk PnL/trade={shrunk_pnl:.2f}; "
                        "execution needs current direct evidence and recovery in outcomes."
                    ),
                )

            if (
                n >= soft_min
                and (
                    shrunk_pnl <= float(prefix_shrunk_pnl_cutoff)
                    or (
                        prefix_snapshot.win_rate <= float(prefix_win_rate_cutoff)
                        and prefix_snapshot.pnl_total <= float(prefix_pnl_cutoff)
                    )
                )
            ):
                return EvaluateMarketResult(
                    tier=GateTier.SOFT_DEMOTE,
                    allowed=True,
                    reason="historical_prefix_small_sample_negative",
                    metrics=metrics,
                    wilson_win_rate_lower_bound=wlb,
                    shrunk_pnl_per_trade=shrunk_pnl,
                    sample_size=n,
                    what_to_learn_next=(
                        f"Historical soft-demotion on prefix '{market_prefix}': "
                        f"sample_size={n}, observed win_rate={prefix_snapshot.win_rate:.2f}, "
                        f"Wilson LB={wlb:.2f}, shrunk PnL/trade={shrunk_pnl:.2f}. "
                        "Treat as a score penalty and learning signal, not a terminal judgment."
                    ),
                )

    if family_gate_enabled and family_stats:
        family_snapshot = family_stats.get(normalized_family)
        if family_snapshot is not None:
            n_fam = family_snapshot.sample_size
            wlb_fam = wilson_lower_bound(family_snapshot.wins, n_fam)
            shrunk_pnl_fam = bayesian_shrunk_pnl(
                family_snapshot.pnl_total,
                n_fam,
                prior_pnl_per_trade=0.0,
                prior_strength=prefix_prior_strength,
            ) if prefix_shrinkage_enabled else (
                family_snapshot.pnl_total / n_fam if n_fam > 0 else 0.0
            )
            metrics.update(
                {
                    "historical_gate_family_sample_size": n_fam,
                    "historical_gate_family_win_rate": family_snapshot.win_rate,
                    "historical_gate_family_pnl_total": family_snapshot.pnl_total,
                    "historical_family_samples": n_fam,
                    "historical_family_pnl_total": family_snapshot.pnl_total,
                    "historical_gate_family_wilson_lb": round(wlb_fam, 4),
                    "historical_gate_family_shrunk_pnl_per_trade": round(shrunk_pnl_fam, 4),
                }
            )
            if (
                n_fam >= max(1, int(family_min_samples))
                and family_snapshot.pnl_total <= float(family_pnl_cutoff)
                and family_snapshot.win_rate <= float(family_win_rate_cutoff)
                and wlb_fam <= float(family_win_rate_cutoff)
                and shrunk_pnl_fam <= float(prefix_shrunk_pnl_cutoff)
            ):
                return EvaluateMarketResult(
                    tier=GateTier.HARD_DENY,
                    allowed=False,
                    reason="historical_family_pnl_block",
                    metrics=metrics,
                    wilson_win_rate_lower_bound=wlb_fam,
                    shrunk_pnl_per_trade=shrunk_pnl_fam,
                    sample_size=n_fam,
                    what_to_learn_next=(
                        f"Family '{normalized_family}' has {n_fam} samples, "
                        f"Wilson LB={wlb_fam:.2f}, shrunk PnL/trade={shrunk_pnl_fam:.2f}; "
                        "execution requires direct settlement-aligned evidence and "
                        "recovery in the family before this gate clears."
                    ),
                )

    return EvaluateMarketResult(
        tier=GateTier.NEUTRAL,
        allowed=True,
        reason=None,
        metrics=metrics,
    )


def evaluate_market(
    *,
    market_id: str,
    family: str,
    prefix_stats: dict[str, PerformanceStats] | None,
    family_stats: dict[str, PerformanceStats] | None,
    prefix_len: int = 12,
    prefix_gate_enabled: bool = True,
    prefix_min_samples: int = 3,
    prefix_hard_block_min_samples: int = 20,
    prefix_pnl_cutoff: float = -3.0,
    prefix_win_rate_cutoff: float = 0.40,
    prefix_shrinkage_enabled: bool = True,
    prefix_prior_win_rate: float = 0.50,
    prefix_prior_strength: float = 10.0,
    prefix_shrunk_pnl_cutoff: float = -0.50,
    prefix_soft_demote_score_penalty: float = 0.08,
    family_gate_enabled: bool = True,
    family_min_samples: int = 12,
    family_pnl_cutoff: float = -12.0,
    family_win_rate_cutoff: float = 0.40,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Backward-compatible wrapper returning (allowed, reason, metrics) tuple.

    Delegates to ``evaluate_market_tiered`` and maps the result back to the
    legacy shape so that existing callers and tests keep working without changes.
    """
    result = evaluate_market_tiered(
        market_id=market_id,
        family=family,
        prefix_stats=prefix_stats,
        family_stats=family_stats,
        prefix_len=prefix_len,
        prefix_gate_enabled=prefix_gate_enabled,
        prefix_min_samples=prefix_min_samples,
        prefix_hard_block_min_samples=prefix_hard_block_min_samples,
        prefix_pnl_cutoff=prefix_pnl_cutoff,
        prefix_win_rate_cutoff=prefix_win_rate_cutoff,
        prefix_shrinkage_enabled=prefix_shrinkage_enabled,
        prefix_prior_win_rate=prefix_prior_win_rate,
        prefix_prior_strength=prefix_prior_strength,
        prefix_shrunk_pnl_cutoff=prefix_shrunk_pnl_cutoff,
        prefix_soft_demote_score_penalty=prefix_soft_demote_score_penalty,
        family_gate_enabled=family_gate_enabled,
        family_min_samples=family_min_samples,
        family_pnl_cutoff=family_pnl_cutoff,
        family_win_rate_cutoff=family_win_rate_cutoff,
    )
    enriched_metrics = dict(result.metrics)
    enriched_metrics["historical_gate_tier"] = result.tier
    if result.wilson_win_rate_lower_bound is not None:
        enriched_metrics["historical_gate_wilson_lb"] = result.wilson_win_rate_lower_bound
    if result.shrunk_pnl_per_trade is not None:
        enriched_metrics["historical_gate_shrunk_pnl_per_trade"] = result.shrunk_pnl_per_trade
    if result.what_to_learn_next:
        enriched_metrics["what_to_learn_next"] = result.what_to_learn_next
    if result.sample_size is not None:
        enriched_metrics["historical_gate_sample_size"] = result.sample_size
    return result.allowed, result.reason, enriched_metrics
