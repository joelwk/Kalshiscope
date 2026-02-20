from __future__ import annotations

from grok_client import _format_previous_analysis
from models import TradeDecision


def test_format_previous_analysis_none() -> None:
    assert _format_previous_analysis(None) == "None"


def test_format_previous_analysis_truncates_reasoning() -> None:
    long_reasoning = "a" * 500
    decision = TradeDecision(
        should_trade=True,
        outcome="YES",
        confidence=0.9,
        bet_size_pct=0.5,
        reasoning=long_reasoning,
    )
    summary = _format_previous_analysis(decision)
    assert "should_trade=True" in summary
    assert "outcome=YES" in summary
    assert "confidence=0.90" in summary
    assert "..." in summary
    assert summary.endswith("...'") is True
