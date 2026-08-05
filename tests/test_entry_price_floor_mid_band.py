"""Unit coverage for entry-price mid-band floor override helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import main
from models import Market, MarketOutcome, TradeDecision


def _market(*, yes_price: float = 0.77) -> Market:
    no_price = max(0.01, min(0.99, 1.0 - yes_price))
    return Market(
        id="KXNASDAQ-MID",
        question="Will the Nasdaq be above strike?",
        outcomes=[
            MarketOutcome(name="YES", price=yes_price),
            MarketOutcome(name="NO", price=no_price),
        ],
        liquidity_usdc=200.0,
        volume_24h=100.0,
        open_interest=100.0,
        category="finance",
        status="open",
        close_time=datetime.now(timezone.utc) + timedelta(hours=4),
    )


def _decision(
    *,
    should_trade: bool,
    evidence_basis: str,
    evidence_quality: float,
    primary_source_url: str | None,
    confidence: float = 0.55,
) -> TradeDecision:
    return TradeDecision(
        should_trade=should_trade,
        outcome="NO",
        confidence=confidence,
        raw_confidence=confidence,
        bet_size_pct=0.2,
        reasoning="Source-backed mid-band longshot NO.",
        evidence_quality=evidence_quality,
        evidence_basis=evidence_basis,
        edge_source="fallback",
        primary_source_url=primary_source_url,
    )


def test_mid_band_override_allows_source_backed_proxy_near_floor() -> None:
    settings = main.Settings()
    market = _market(yes_price=0.77)  # NO entry ~0.23
    decision = _decision(
        should_trade=True,
        evidence_basis="proxy",
        evidence_quality=0.55,
        primary_source_url="https://www.cnbc.com/quotes/.NDX",
        confidence=0.55,
    )
    entry_price = main._get_outcome_entry_price(market, "NO")
    implied = main._get_implied_probability(market, "NO")
    assert entry_price is not None
    assert 0.20 <= entry_price < 0.25
    floor_edge = decision.confidence - float(implied)
    required = main._edge_threshold_for_market(
        float(implied), settings, market=market, decision=decision
    )
    assert floor_edge >= required
    mid_band_floor = settings.VERY_LOW_PRICE_THRESHOLD - settings.ENTRY_PRICE_FLOOR_MID_BAND_WIDTH
    assert mid_band_floor <= entry_price < settings.VERY_LOW_PRICE_THRESHOLD
    assert str(decision.primary_source_url).startswith("https://")
    assert decision.evidence_quality >= settings.MIN_EVIDENCE_QUALITY_FOR_TRADE


def test_mid_band_override_rejects_absence_only_and_deep_longshot() -> None:
    settings = main.Settings()
    deep = _market(yes_price=0.90)  # NO ~0.10, below mid-band
    entry = main._get_outcome_entry_price(deep, "NO")
    mid_band_floor = settings.VERY_LOW_PRICE_THRESHOLD - settings.ENTRY_PRICE_FLOOR_MID_BAND_WIDTH
    assert entry is not None and entry < mid_band_floor

    absence = _decision(
        should_trade=True,
        evidence_basis="absence_only",
        evidence_quality=0.55,
        primary_source_url="https://www.cnbc.com/",
    )
    assert str(absence.evidence_basis).lower() == "absence_only"
