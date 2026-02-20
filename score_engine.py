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


def compute_final_score(
    market: Market,
    decision: TradeDecision,
    implied_prob_market: float | None,
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

    final_score = (
        (0.45 * edge_market)
        + (0.30 * edge_external)
        + (0.25 * evidence_quality)
        - liquidity_penalty
        - staleness_penalty
    )

    return ScoreResult(
        final_score=final_score,
        edge_market=edge_market,
        edge_external=edge_external,
        evidence_quality=evidence_quality,
        liquidity_penalty=liquidity_penalty,
        staleness_penalty=staleness_penalty,
    )

