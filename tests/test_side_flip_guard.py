from __future__ import annotations

from config import Settings
from main import _apply_flip_guard
from models import Market, MarketOutcome, TradeDecision


def _market() -> Market:
    return Market(
        id="m1",
        question="Who wins?",
        outcomes=[
            MarketOutcome(name="YES", price=0.60),
            MarketOutcome(name="NO", price=0.40),
        ],
    )


def _decision(outcome: str, confidence: float, evidence_quality: float = 0.8) -> TradeDecision:
    return TradeDecision(
        should_trade=True,
        outcome=outcome,
        confidence=confidence,
        bet_size_pct=0.5,
        reasoning="test",
        evidence_quality=evidence_quality,
    )


def test_flip_guard_blocks_low_quality_flip() -> None:
    settings = Settings()
    market = _market()
    decision = _decision("NO", 0.62, evidence_quality=0.9)
    anchor = {"outcome": "YES", "confidence": 0.70}

    guarded, triggered, blocked = _apply_flip_guard(
        market,
        decision,
        anchor,
        settings,
    )

    assert triggered is True
    assert blocked is True
    assert guarded.should_trade is False
    assert guarded.bet_size_pct == 0.0
    assert guarded.outcome == "YES"
    assert "[FlipGuard blocked:" in guarded.reasoning


def test_flip_guard_allows_materially_stronger_flip() -> None:
    settings = Settings()
    market = _market()
    decision = _decision("NO", 0.80, evidence_quality=0.85)
    anchor = {"outcome": "YES", "confidence": 0.66}

    guarded, triggered, blocked = _apply_flip_guard(
        market,
        decision,
        anchor,
        settings,
    )

    assert triggered is True
    assert blocked is False
    assert guarded.should_trade is True
    assert guarded.outcome == "NO"


def test_flip_guard_bypasses_low_confidence_anchor() -> None:
    settings = Settings()
    market = _market()
    decision = _decision("NO", 0.62, evidence_quality=0.9)
    anchor = {"outcome": "YES", "confidence": 0.45}

    guarded, triggered, blocked = _apply_flip_guard(
        market,
        decision,
        anchor,
        settings,
    )

    assert triggered is False
    assert blocked is False
    assert guarded.should_trade is True
    assert guarded.outcome == "NO"


def test_flip_guard_allows_high_evidence_override() -> None:
    settings = Settings()
    market = _market()
    decision = _decision("NO", 0.92, evidence_quality=0.95)
    anchor = {"outcome": "YES", "confidence": 0.90}

    guarded, triggered, blocked = _apply_flip_guard(
        market,
        decision,
        anchor,
        settings,
    )

    assert triggered is True
    assert blocked is False
    assert guarded.should_trade is True
    assert guarded.outcome == "NO"


def test_flip_guard_uses_raw_confidence_for_direct_high_likelihood_flips() -> None:
    settings = Settings()
    market = _market()
    decision = _decision("NO", 0.60, evidence_quality=0.92).model_copy(
        update={
            "raw_confidence": 0.88,
            "evidence_basis": "direct",
            "likelihood_ratio": 10.0,
        }
    )
    anchor = {"outcome": "YES", "confidence": 0.70}

    guarded, triggered, blocked = _apply_flip_guard(
        market,
        decision,
        anchor,
        settings,
    )

    assert triggered is True
    assert blocked is False
    assert guarded.should_trade is True
    assert guarded.outcome == "NO"
