"""Centralized logging configuration for PredictBot.

Provides structured logging with:
- Console output (colored, human-readable)
- File output with rotation (detailed JSON format)
- Separate error log for critical failures
- Correlation IDs for request tracing
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

LOG_DIR = Path("logs")
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_FILE_LOG_LEVEL = "DEBUG"
MAX_LOG_FILE_SIZE_MB = 10
LOG_BACKUP_COUNT = 5

correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
_RE_GATE_REASON = re.compile(r"reason=([^\]\s]+)")


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for request tracing."""
    return str(uuid.uuid4())[:8]


def set_correlation_id(cid: str | None = None) -> str:
    """Set the correlation ID for the current context."""
    cid = cid or generate_correlation_id()
    correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return correlation_id.get()


class CorrelationFilter(logging.Filter):
    """Add correlation ID to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured file logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "-"),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data

        return json.dumps(log_data, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for human-readable output."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def __init__(self, use_colors: bool = True) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s [%(correlation_id)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "correlation_id"):
            record.correlation_id = "-"

        if self.use_colors:
            original_level = record.levelname
            color = self.COLORS.get(record.levelname, "")
            record.levelname = f"{color}{record.levelname}{self.RESET}"
            try:
                return super().format(record)
            finally:
                record.levelname = original_level
        return super().format(record)


class StructuredLogger(logging.LoggerAdapter):
    """Logger adapter that supports structured extra data."""

    def process(
        self, msg: str, kwargs: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        extra = kwargs.get("extra", {})
        extra["correlation_id"] = get_correlation_id() or "-"

        if "data" in kwargs:
            extra["extra_data"] = kwargs.pop("data")

        kwargs["extra"] = extra
        return msg, kwargs

    def with_context(self, **context: Any) -> StructuredLogger:
        """Create a child logger with additional context."""
        new_extra = {**self.extra, **context}
        return StructuredLogger(self.logger, new_extra)


def setup_logging(
    level: str = DEFAULT_LOG_LEVEL,
    file_level: str = DEFAULT_FILE_LOG_LEVEL,
    log_dir: Path | str = LOG_DIR,
    enable_file_logging: bool = True,
    enable_json_logging: bool = True,
    enable_colors: bool = True,
) -> None:
    """Configure the logging system.

    Args:
        level: Console log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        file_level: File log level
        log_dir: Directory for log files
        enable_file_logging: Enable rotating file logs
        enable_json_logging: Use JSON format for file logs
        enable_colors: Enable colored console output
    """
    log_dir = Path(log_dir)
    if not log_dir.is_absolute():
        log_dir = Path.cwd() / log_dir
    
    root_logger = logging.getLogger()

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.addFilter(CorrelationFilter())
    console_handler.setFormatter(ColoredFormatter(use_colors=enable_colors))
    root_logger.addHandler(console_handler)

    if enable_file_logging:
        os.makedirs(log_dir, exist_ok=True)

        main_log_file = log_dir / "predictbot.log"
        file_handler = RotatingFileHandler(
            main_log_file,
            maxBytes=MAX_LOG_FILE_SIZE_MB * 1024 * 1024,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
        file_handler.addFilter(CorrelationFilter())

        if enable_json_logging:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s [%(correlation_id)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s"
                )
            )
        root_logger.addHandler(file_handler)

        error_log_file = log_dir / "predictbot_errors.log"
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=MAX_LOG_FILE_SIZE_MB * 1024 * 1024,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.addFilter(CorrelationFilter())
        if enable_json_logging:
            error_handler.setFormatter(JsonFormatter())
        else:
            error_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s [%(correlation_id)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s"
                )
            )
        root_logger.addHandler(error_handler)

        trade_log_file = log_dir / "trades.log"
        trade_handler = RotatingFileHandler(
            trade_log_file,
            maxBytes=MAX_LOG_FILE_SIZE_MB * 1024 * 1024,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        trade_handler.setLevel(logging.INFO)
        trade_handler.addFilter(CorrelationFilter())
        trade_handler.addFilter(logging.Filter("predictbot.trades"))
        if enable_json_logging:
            trade_handler.setFormatter(JsonFormatter())
        else:
            trade_handler.setFormatter(
                logging.Formatter("%(asctime)s %(message)s")
            )
        root_logger.addHandler(trade_handler)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        StructuredLogger instance with correlation ID support
    """
    return StructuredLogger(logging.getLogger(name), {})


def get_trade_logger() -> StructuredLogger:
    """Get the dedicated trade logger for tracking all trade decisions."""
    return StructuredLogger(logging.getLogger("predictbot.trades"), {})


def log_api_call(
    logger: StructuredLogger,
    method: str,
    url: str,
    status_code: int | None = None,
    duration_ms: float | None = None,
    error: str | None = None,
) -> None:
    """Log an API call with standard format."""
    data = {
        "method": method,
        "url": url,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2) if duration_ms else None,
    }
    if error:
        data["error"] = error
        logger.error("API call failed: %s %s", method, url, data=data)
    else:
        logger.debug("API call: %s %s -> %s (%.2fms)", method, url, status_code, duration_ms or 0, data=data)


def log_trade_decision(
    market_id: str,
    question: str,
    decision: dict[str, Any],
    order: dict[str, Any] | None = None,
    execution_audit: dict[str, Any] | None = None,
) -> None:
    """Log a trade decision to the dedicated trade log."""
    trade_logger = get_trade_logger()
    reasoning = str(decision.get("reasoning") or "")
    gate_reason = None
    gate_match = _RE_GATE_REASON.search(reasoning)
    if gate_match:
        gate_reason = gate_match.group(1)
    audit = {
        "gate_reason": gate_reason,
        "should_trade": decision.get("should_trade"),
        "bet_size_pct": decision.get("bet_size_pct"),
        "implied_prob_external": decision.get("implied_prob_external"),
        "my_prob": decision.get("my_prob"),
        "edge_external": decision.get("edge_external"),
        "evidence_quality": decision.get("evidence_quality"),
    }
    if execution_audit:
        audit.update(execution_audit)
    data = {
        "market_id": market_id,
        "question": question[:100],
        "decision": decision,
        "order": order,
        "audit": audit,
    }
    trade_logger.info(
        "Trade decision: market=%s should_trade=%s confidence=%.2f outcome=%s",
        market_id,
        decision.get("should_trade"),
        decision.get("confidence", 0),
        decision.get("outcome"),
        data=data,
    )


def log_transaction(
    logger: StructuredLogger,
    tx_type: str,
    tx_hash: str,
    details: dict[str, Any],
) -> None:
    """Log a blockchain transaction."""
    data = {"tx_type": tx_type, "tx_hash": tx_hash, **details}
    logger.info("Transaction submitted: type=%s hash=%s", tx_type, tx_hash, data=data)

