from __future__ import annotations

from unittest.mock import Mock, patch

from logging_config import log_trade_decision


def test_log_trade_decision_merges_execution_audit() -> None:
    trade_logger = Mock()
    decision = {
        "should_trade": True,
        "confidence": 0.62,
        "outcome": "YES",
        "bet_size_pct": 0.12,
        "reasoning": "[Validated eq=1.00 gate=allow reason=ok] test reasoning",
        "implied_prob_external": 0.58,
        "my_prob": 0.62,
        "edge_external": 0.04,
        "evidence_quality": 1.0,
    }
    execution_audit = {
        "decision_phase": "post_sizing",
        "sizing_mode": "kelly",
        "adjusted_bet_pct": 0.08,
        "kelly_raw": 0.18,
    }
    with patch("logging_config.get_trade_logger", return_value=trade_logger):
        log_trade_decision(
            market_id="m1",
            question="Question",
            decision=decision,
            execution_audit=execution_audit,
        )

    assert trade_logger.info.called
    payload = trade_logger.info.call_args.kwargs.get("data") or {}
    audit = payload.get("audit") or {}
    assert audit.get("decision_phase") == "post_sizing"
    assert audit.get("sizing_mode") == "kelly"
    assert audit.get("adjusted_bet_pct") == 0.08
    assert audit.get("kelly_raw") == 0.18


def test_log_trade_decision_includes_terminal_audit_fields() -> None:
    trade_logger = Mock()
    decision = {
        "should_trade": False,
        "confidence": 0.62,
        "outcome": "YES",
        "bet_size_pct": 0.0,
        "reasoning": "[Validated eq=1.00 gate=block reason=ok] test reasoning",
    }
    execution_audit = {
        "decision_terminal": True,
        "final_action": "skip",
        "final_reason": "kelly_sub_floor_skip",
    }
    with patch("logging_config.get_trade_logger", return_value=trade_logger):
        log_trade_decision(
            market_id="m2",
            question="Question",
            decision=decision,
            execution_audit=execution_audit,
        )

    payload = trade_logger.info.call_args.kwargs.get("data") or {}
    audit = payload.get("audit") or {}
    assert audit.get("decision_terminal") is True
    assert audit.get("final_action") == "skip"
    assert audit.get("final_reason") == "kelly_sub_floor_skip"
