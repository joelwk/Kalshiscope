from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from logging_config import (
    get_logger,
    log_trade_decision,
    set_correlation_id,
    setup_logging,
)


@pytest.fixture
def fresh_logging(tmp_path: Path):
    """Reset root logger after the test so subsequent tests run clean."""
    yield tmp_path
    root = logging.getLogger()
    for handler in root.handlers[:]:
        try:
            handler.close()
        finally:
            root.removeHandler(handler)
    root.setLevel(logging.WARNING)


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


def test_predictbot_errors_log_captures_error_level_events(fresh_logging: Path) -> None:
    """Cycle 2 emitted a ``logger.error("Cycle yield alert (sustained...)")``
    that needs to land in ``predictbot_errors.log`` so operators can see
    sustained selection-failure escalations without scanning the full debug
    log. This regression test exercises the root-attached error handler so
    we know the routing works end-to-end."""
    setup_logging(
        level="DEBUG",
        file_level="DEBUG",
        log_dir=fresh_logging,
        enable_file_logging=True,
        enable_json_logging=True,
        enable_colors=False,
    )
    set_correlation_id("test_cid")
    logger = get_logger("predictbot")
    logger.error(
        "Cycle yield alert (sustained, %d cycles)",
        2,
        data={"cycle_yield_alert": True, "research_queue_size": 168},
    )
    for handler in logging.getLogger().handlers:
        handler.flush()

    error_log = fresh_logging / "predictbot_errors.log"
    assert error_log.exists(), "predictbot_errors.log was not created"
    content = error_log.read_text(encoding="utf-8").strip()
    assert content, "predictbot_errors.log is empty after ERROR event"
    record = json.loads(content.splitlines()[-1])
    assert record["level"] == "ERROR"
    assert "Cycle yield alert" in record["message"]
    assert record["correlation_id"] == "test_cid"
    assert record.get("data", {}).get("cycle_yield_alert") is True


def test_predictbot_errors_log_ignores_warning_level(fresh_logging: Path) -> None:
    """Sanity check: WARNING-level events stay out of the dedicated error
    log so it remains a curated stream of true escalations."""
    setup_logging(
        level="DEBUG",
        file_level="DEBUG",
        log_dir=fresh_logging,
        enable_file_logging=True,
        enable_json_logging=True,
        enable_colors=False,
    )
    logger = get_logger("predictbot")
    logger.warning("Cycle yield alert (single cycle)")
    for handler in logging.getLogger().handlers:
        handler.flush()

    error_log = fresh_logging / "predictbot_errors.log"
    if not error_log.exists():
        return
    assert error_log.read_text(encoding="utf-8").strip() == ""
