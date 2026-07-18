from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from bayesian_engine import BayesianState
from logging_config import get_logger
from models import MarketState, OrderResponse, Position, TradeDecision
from research_profiles import family_from_text

logger = get_logger(__name__)

_CONFIDENCE_TREND_WINDOW = 5
_RE_VALIDATED_PREFIX = re.compile(r"^\[Validated\b[^\]]*\]\s*")
_NON_ACTIONABLE_TERMINAL_OUTCOMES = {
    "analysis_failure",
    "analysis_only_insufficient_balance",
    "bet_amount_zero",
    "coinflip_market",
    "confidence_below_min",
    "evidence_quality_below_min",
    "edge_gate_blocked",
    "kelly_sub_floor_skip",
    "lmsr_gate_blocked",
    "max_trades_per_cycle_reached",
    "no_trade_recommended",
    "orderbook_spread_too_wide",
    "position_adjustment_blocked",
    "score_gate_blocked",
    "stale_market_data_refresh_failed",
    "uniform_implied_probability",
    "zero_bet_after_sizing",
}


class MarketStateManager:
    """SQLite-backed state manager for market analyses and positions."""

    def __init__(self, db_path: str = "data/market_state.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS markets (
                    id TEXT PRIMARY KEY,
                    question TEXT,
                    close_time TEXT,
                    category TEXT,
                    last_terminal_outcome TEXT,
                    non_actionable_streak INTEGER DEFAULT 0,
                    fill_failure_count INTEGER DEFAULT 0,
                    next_eligible_cycle INTEGER DEFAULT 0
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id TEXT NOT NULL,
                    confidence REAL,
                    outcome TEXT,
                    reasoning TEXT,
                    reasoning_hash TEXT,
                    timestamp TEXT,
                    is_refined INTEGER,
                    refinement_reason TEXT
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    market_id TEXT PRIMARY KEY,
                    outcome TEXT,
                    total_amount REAL,
                    avg_confidence REAL,
                    order_ids TEXT
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id TEXT NOT NULL,
                    amount REAL,
                    outcome TEXT,
                    order_id TEXT,
                    timestamp TEXT
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_outcomes (
                    market_id TEXT PRIMARY KEY,
                    predicted_outcome TEXT,
                    entry_price REAL,
                    implied_prob REAL,
                    confidence REAL,
                    amount_usdc REAL,
                    shares REAL,
                    resolved_winning_outcome TEXT,
                    won INTEGER,
                    pnl_estimate REAL,
                    resolved_at TEXT,
                    last_updated TEXT,
                    resolution_state TEXT DEFAULT 'unresolved'
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_outcome_events (
                    market_id TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    predicted_outcome TEXT,
                    entry_price REAL,
                    implied_prob REAL,
                    confidence REAL,
                    amount_usdc REAL,
                    shares REAL,
                    timestamp TEXT,
                    resolved_winning_outcome TEXT,
                    won INTEGER,
                    pnl_estimate REAL,
                    resolved_at TEXT,
                    resolution_state TEXT DEFAULT 'unresolved',
                    PRIMARY KEY (market_id, order_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bayesian_state (
                    market_id TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    log_prior REAL NOT NULL,
                    log_likelihood_sum REAL NOT NULL DEFAULT 0.0,
                    update_count INTEGER NOT NULL DEFAULT 0,
                    last_updated TEXT,
                    PRIMARY KEY (market_id, outcome)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cycle_receipts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id TEXT,
                    cycle_number INTEGER,
                    timestamp TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS exchange_settlements (
                    settlement_id TEXT PRIMARY KEY,
                    market_id TEXT NOT NULL,
                    predicted_outcome TEXT,
                    winning_outcome TEXT,
                    won INTEGER,
                    pnl_realized REAL,
                    contracts INTEGER,
                    avg_price REAL,
                    settled_at TEXT,
                    raw_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS confidence_calibration_online (
                    family TEXT NOT NULL,
                    bucket REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    sample_size INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (family, bucket)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_receipts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id TEXT,
                    market_id TEXT NOT NULL,
                    final_action TEXT,
                    final_reason TEXT,
                    timestamp TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    order_json TEXT,
                    audit_json TEXT,
                    score_json TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_analyses_market_id ON analyses (market_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trade_log_market_id ON trade_log (market_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trade_outcomes_market_id ON trade_outcomes (market_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trade_outcome_events_market_id ON trade_outcome_events (market_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bayesian_state_market_id ON bayesian_state (market_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cycle_receipts_cycle_id ON cycle_receipts (cycle_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_decision_receipts_market_id ON decision_receipts (market_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_exchange_settlements_market_id ON exchange_settlements (market_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_confidence_calibration_online_family ON confidence_calibration_online (family)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_queue_entries (
                    market_id TEXT PRIMARY KEY,
                    cycle_id TEXT,
                    queued_at TEXT NOT NULL,
                    gate_name TEXT,
                    reason TEXT,
                    threshold_gap REAL,
                    what_to_learn_next TEXT,
                    last_seen TEXT NOT NULL,
                    expires_at TEXT,
                    last_decision_json TEXT,
                    times_seen INTEGER DEFAULT 1
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rq_last_seen ON research_queue_entries (last_seen)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_flags (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._run_migrations()
            self._backfill_resolution_state()

    def get_market_state(self, market_id: str) -> MarketState | None:
        market_row = self._conn.execute(
            """
            SELECT last_terminal_outcome, non_actionable_streak
                , fill_failure_count, next_eligible_cycle
            FROM markets
            WHERE id = ?
            """,
            (market_id,),
        ).fetchone()
        last_terminal_outcome = (
            str(market_row["last_terminal_outcome"])
            if market_row and market_row["last_terminal_outcome"] is not None
            else None
        )
        non_actionable_streak = (
            int(market_row["non_actionable_streak"] or 0)
            if market_row and market_row["non_actionable_streak"] is not None
            else 0
        )
        fill_failure_count = (
            int(market_row["fill_failure_count"] or 0)
            if market_row and market_row["fill_failure_count"] is not None
            else 0
        )
        next_eligible_cycle = (
            int(market_row["next_eligible_cycle"] or 0)
            if market_row and market_row["next_eligible_cycle"] is not None
            else 0
        )
        latest_row = self._conn.execute(
            """
            SELECT confidence, timestamp
            FROM analyses
            WHERE market_id = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT 1
            """,
            (market_id,),
        ).fetchone()

        count_row = self._conn.execute(
            "SELECT COUNT(*) AS analysis_count FROM analyses WHERE market_id = ?",
            (market_id,),
        ).fetchone()
        analysis_count = count_row["analysis_count"] if count_row else 0

        if not latest_row:
            if not self._market_exists(market_id):
                return None
            return MarketState(
                market_id=market_id,
                last_terminal_outcome=last_terminal_outcome,
                non_actionable_streak=non_actionable_streak,
                fill_failure_count=fill_failure_count,
                next_eligible_cycle=next_eligible_cycle,
            )

        trend_rows = self._conn.execute(
            """
            SELECT confidence
            FROM analyses
            WHERE market_id = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (market_id, _CONFIDENCE_TREND_WINDOW),
        ).fetchall()
        confidence_trend = [row["confidence"] for row in reversed(trend_rows or [])]

        return MarketState(
            market_id=market_id,
            last_analysis=_parse_timestamp(latest_row["timestamp"]),
            analysis_count=analysis_count,
            last_confidence=latest_row["confidence"],
            confidence_trend=confidence_trend,
            last_terminal_outcome=last_terminal_outcome,
            non_actionable_streak=non_actionable_streak,
            fill_failure_count=fill_failure_count,
            next_eligible_cycle=next_eligible_cycle,
        )

    def get_position(self, market_id: str) -> Position | None:
        row = self._conn.execute(
            """
            SELECT market_id, outcome, total_amount, avg_confidence, order_ids
            FROM positions
            WHERE market_id = ?
            """,
            (market_id,),
        ).fetchone()
        if not row:
            return None

        meta = self._conn.execute(
            """
            SELECT COUNT(*) AS trade_count,
                   MIN(timestamp) AS first_trade,
                   MAX(timestamp) AS last_trade
            FROM trade_log
            WHERE market_id = ?
            """,
            (market_id,),
        ).fetchone()

        trade_count = meta["trade_count"] if meta else 0
        first_trade = _parse_timestamp(meta["first_trade"] if meta else None)
        last_trade = _parse_timestamp(meta["last_trade"] if meta else None)
        if not first_trade or not last_trade:
            now = datetime.now(timezone.utc)
            first_trade = first_trade or now
            last_trade = last_trade or now
            if trade_count == 0:
                logger.warning(
                    "Position found without trade log entries: market=%s",
                    market_id,
                )

        return Position(
            market_id=row["market_id"],
            outcome=row["outcome"] or "UNKNOWN",
            total_amount_usdc=float(row["total_amount"] or 0.0),
            avg_confidence=float(row["avg_confidence"] or 0.0),
            trade_count=trade_count,
            first_trade=first_trade,
            last_trade=last_trade,
        )

    def get_open_position_market_ids_for_event(self, event_ticker_prefix: str) -> list[str]:
        normalized_prefix = str(event_ticker_prefix or "").strip().upper()
        if not normalized_prefix:
            return []
        rows = self._conn.execute(
            """
            SELECT market_id
            FROM positions
            WHERE total_amount > 0
              AND UPPER(COALESCE(market_id, '')) LIKE ?
            """,
            (f"{normalized_prefix}%",),
        ).fetchall()
        return [str(row["market_id"]) for row in rows if row["market_id"]]

    def get_last_trade_entry_price(self, market_id: str) -> float | None:
        row = self._conn.execute(
            """
            SELECT entry_price
            FROM trade_outcome_events
            WHERE market_id = ?
            ORDER BY timestamp DESC, order_id DESC
            LIMIT 1
            """,
            (market_id,),
        ).fetchone()
        if row is None or row["entry_price"] is None:
            return None
        try:
            return float(row["entry_price"])
        except (TypeError, ValueError):
            return None

    def get_anchor_analysis(
        self,
        market_id: str,
        min_confidence: float,
    ) -> dict[str, Any] | None:
        """Return anchor analysis row for side-stability checks.

        Preference order:
        1) Latest analysis at/above min_confidence.
        2) Latest analysis regardless of confidence.
        """
        row = self._conn.execute(
            """
            SELECT market_id, outcome, confidence, reasoning, timestamp, is_refined, refinement_reason
            FROM analyses
            WHERE market_id = ?
              AND confidence IS NOT NULL
              AND confidence >= ?
            ORDER BY timestamp DESC, id DESC
            LIMIT 1
            """,
            (market_id, min_confidence),
        ).fetchone()
        if row is not None:
            return dict(row)

        fallback = self._conn.execute(
            """
            SELECT market_id, outcome, confidence, reasoning, timestamp, is_refined, refinement_reason
            FROM analyses
            WHERE market_id = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT 1
            """,
            (market_id,),
        ).fetchone()
        if fallback is None:
            return None
        return dict(fallback)

    def get_bayesian_state(self, market_id: str) -> dict[str, BayesianState]:
        rows = self._conn.execute(
            """
            SELECT outcome, log_prior, log_likelihood_sum, update_count, last_updated
            FROM bayesian_state
            WHERE market_id = ?
            """,
            (market_id,),
        ).fetchall()
        if not rows:
            return {}

        states: dict[str, BayesianState] = {}
        for row in rows:
            log_likelihood_sum = float(row["log_likelihood_sum"] or 0.0)
            # Stored as running sum for compact persistence; materialize as one aggregate update.
            log_likelihoods = [log_likelihood_sum] if log_likelihood_sum != 0.0 else []
            states[str(row["outcome"])] = BayesianState(
                log_prior=float(row["log_prior"]),
                log_likelihoods=log_likelihoods,
                update_count=int(row["update_count"] or 0),
                last_updated=row["last_updated"] or datetime.now(timezone.utc).isoformat(),
            )
        return states

    def update_bayesian_state(
        self,
        market_id: str,
        outcome: str,
        log_prior: float,
        log_likelihood: float,
        count_as_update: bool = True,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        row = self._conn.execute(
            """
            SELECT log_likelihood_sum, update_count
            FROM bayesian_state
            WHERE market_id = ? AND outcome = ?
            """,
            (market_id, outcome),
        ).fetchone()
        if row:
            existing_sum = float(row["log_likelihood_sum"] or 0.0)
            existing_count = int(row["update_count"] or 0)
            updated_sum = (
                existing_sum + float(log_likelihood)
                if count_as_update
                else existing_sum
            )
            updated_count = (
                existing_count + 1
                if count_as_update
                else existing_count
            )
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE bayesian_state
                    SET log_prior = ?, log_likelihood_sum = ?, update_count = ?, last_updated = ?
                    WHERE market_id = ? AND outcome = ?
                    """,
                    (
                        float(log_prior),
                        updated_sum,
                        updated_count,
                        timestamp,
                        market_id,
                        outcome,
                    ),
                )
            return

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO bayesian_state (
                    market_id, outcome, log_prior, log_likelihood_sum, update_count, last_updated
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    market_id,
                    outcome,
                    float(log_prior),
                    float(log_likelihood) if count_as_update else 0.0,
                    1 if count_as_update else 0,
                    timestamp,
                ),
            )

    def reset_bayesian_state(self, market_id: str) -> None:
        with self._conn:
            self._conn.execute(
                "DELETE FROM bayesian_state WHERE market_id = ?",
                (market_id,),
            )

    def record_analysis(
        self,
        market_id: str,
        decision: TradeDecision,
        is_refined: bool,
        refinement_reason: str | None = None,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        reasoning_hash = _build_reasoning_hash(
            decision.reasoning,
            decision.outcome,
            decision.confidence,
        )
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO analyses (
                    market_id, confidence, outcome, reasoning, reasoning_hash, timestamp,
                    is_refined, refinement_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    market_id,
                    decision.confidence,
                    decision.outcome,
                    decision.reasoning,
                    reasoning_hash,
                    timestamp,
                    1 if is_refined else 0,
                    refinement_reason,
                ),
            )
        logger.debug(
            "Recorded analysis: market=%s confidence=%.4f refined=%s reason=%s",
            market_id,
            decision.confidence,
            is_refined,
            refinement_reason or "-",
        )

    def record_terminal_outcome(self, market_id: str, terminal_outcome: str) -> None:
        normalized = (terminal_outcome or "").strip().lower()
        with self._conn:
            row = self._conn.execute(
                """
                SELECT non_actionable_streak
                FROM markets
                WHERE id = ?
                """,
                (market_id,),
            ).fetchone()
            previous_streak = int(row["non_actionable_streak"] or 0) if row else 0
            next_streak = (
                previous_streak + 1
                if normalized in _NON_ACTIONABLE_TERMINAL_OUTCOMES
                else 0
            )
            self._conn.execute(
                """
                INSERT INTO markets (id, last_terminal_outcome, non_actionable_streak)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    last_terminal_outcome = excluded.last_terminal_outcome,
                    non_actionable_streak = excluded.non_actionable_streak
                """,
                (market_id, terminal_outcome, next_streak),
            )

    def set_market_cooldown_cycle(self, market_id: str, next_eligible_cycle: int) -> None:
        normalized_market_id = str(market_id or "").strip()
        if not normalized_market_id:
            return
        normalized_cycle = max(0, int(next_eligible_cycle or 0))
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO markets (id, next_eligible_cycle)
                VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    next_eligible_cycle = excluded.next_eligible_cycle
                """,
                (normalized_market_id, normalized_cycle),
            )

    def record_cycle_receipt(self, cycle_id: str, cycle_number: int, payload: dict[str, Any]) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        payload_json = json.dumps(payload, sort_keys=True, default=str)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO cycle_receipts (
                    cycle_id, cycle_number, timestamp, payload_json
                )
                VALUES (?, ?, ?, ?)
                """,
                (cycle_id, cycle_number, timestamp, payload_json),
            )

    def record_decision_receipt(
        self,
        *,
        cycle_id: str,
        market_id: str,
        decision: dict[str, Any],
        order: dict[str, Any] | None = None,
        execution_audit: dict[str, Any] | None = None,
        score_breakdown: dict[str, Any] | None = None,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        normalized_score_breakdown = score_breakdown
        if normalized_score_breakdown is None and isinstance(execution_audit, dict):
            candidate_score = execution_audit.get("score_breakdown")
            if isinstance(candidate_score, dict):
                normalized_score_breakdown = candidate_score
        final_action = (
            str((execution_audit or {}).get("final_action", "")).strip() or None
        )
        final_reason = (
            str((execution_audit or {}).get("final_reason", "")).strip() or None
        )
        normalized_order = order
        if normalized_order is None and isinstance(execution_audit, dict):
            order_summary = {
                key: execution_audit.get(key)
                for key in (
                    "order_id",
                    "order_status",
                    "order_cancel_reason",
                    "order_fill_count",
                    "fallback_order_id",
                    "fallback_order_status",
                )
                if execution_audit.get(key) is not None
            }
            normalized_order = order_summary or None
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO decision_receipts (
                    cycle_id, market_id, final_action, final_reason, timestamp,
                    decision_json, order_json, audit_json, score_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cycle_id,
                    market_id,
                    final_action,
                    final_reason,
                    timestamp,
                    json.dumps(decision or {}, sort_keys=True, default=str),
                    json.dumps(normalized_order, sort_keys=True, default=str)
                    if normalized_order is not None
                    else None,
                    json.dumps(execution_audit, sort_keys=True, default=str)
                    if execution_audit is not None
                    else None,
                    json.dumps(normalized_score_breakdown, sort_keys=True, default=str)
                    if normalized_score_breakdown is not None
                    else None,
                ),
            )

    def get_daily_order_attempt_summary(
        self,
        *,
        since: datetime,
        include_dry_run: bool = True,
    ) -> tuple[int, int, float]:
        """Return attempted orders, credited exposures, and their EV since `since`."""
        since_utc = since
        if since_utc.tzinfo is None:
            since_utc = since_utc.replace(tzinfo=timezone.utc)
        else:
            since_utc = since_utc.astimezone(timezone.utc)
        row = self._conn.execute(
            """
            SELECT
                COUNT(*) AS attempt_count,
                COALESCE(
                    SUM(
                        CASE
                            WHEN COALESCE(
                                CAST(json_extract(audit_json, '$.daily_expectancy_ev_credited') AS INTEGER),
                                CASE
                                    WHEN final_reason IN ('order_submitted', 'dry_run') THEN 1
                                    ELSE 0
                                END
                            ) = 1
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS credited_exposure_count,
                COALESCE(
                    SUM(
                        CASE
                            WHEN COALESCE(
                                CAST(json_extract(audit_json, '$.daily_expectancy_ev_credited') AS INTEGER),
                                CASE
                                    WHEN final_reason IN ('order_submitted', 'dry_run') THEN 1
                                    ELSE 0
                                END
                            ) = 1
                            THEN COALESCE(
                                CAST(json_extract(audit_json, '$.expected_value_usdc') AS REAL),
                                0.0
                            )
                            ELSE 0.0
                        END
                    ),
                    0.0
                ) AS projected_expected_value_usdc
            FROM decision_receipts
            WHERE timestamp >= ?
              AND final_action = 'order_attempt'
              AND (? = 1 OR final_reason != 'dry_run')
            """,
            (since_utc.isoformat(), 1 if include_dry_run else 0),
        ).fetchone()
        if row is None:
            return 0, 0, 0.0
        return (
            int(row["attempt_count"] or 0),
            int(row["credited_exposure_count"] or 0),
            float(row["projected_expected_value_usdc"] or 0.0),
        )

    def upsert_position_snapshot(
        self,
        *,
        market_id: str,
        outcome: str,
        total_amount_usdc: float,
    ) -> None:
        normalized_market_id = str(market_id or "").strip()
        if not normalized_market_id:
            return
        normalized_outcome = str(outcome or "").strip().upper()
        if normalized_outcome not in {"YES", "NO"}:
            return
        normalized_total = max(0.0, float(total_amount_usdc or 0.0))
        with self._conn:
            row = self._conn.execute(
                """
                SELECT avg_confidence, order_ids
                FROM positions
                WHERE market_id = ?
                """,
                (normalized_market_id,),
            ).fetchone()
            avg_confidence = float(row["avg_confidence"] or 0.0) if row else 0.0
            order_ids_raw = row["order_ids"] if row else "[]"
            self._conn.execute(
                """
                INSERT INTO positions (market_id, outcome, total_amount, avg_confidence, order_ids)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(market_id) DO UPDATE SET
                    outcome = excluded.outcome,
                    total_amount = excluded.total_amount,
                    avg_confidence = excluded.avg_confidence,
                    order_ids = excluded.order_ids
                """,
                (
                    normalized_market_id,
                    normalized_outcome,
                    normalized_total,
                    avg_confidence,
                    order_ids_raw if order_ids_raw else "[]",
                ),
            )

    def increment_fill_failure_count(self, market_id: str) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO markets (id, fill_failure_count)
                VALUES (?, 1)
                ON CONFLICT(id) DO UPDATE SET
                    fill_failure_count = COALESCE(markets.fill_failure_count, 0) + 1
                """,
                (market_id,),
            )

    def reset_fill_failure_count(self, market_id: str) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO markets (id, fill_failure_count)
                VALUES (?, 0)
                ON CONFLICT(id) DO UPDATE SET
                    fill_failure_count = 0
                """,
                (market_id,),
            )

    def record_trade(
        self,
        market_id: str,
        order: OrderResponse,
        amount: float,
        outcome: str | None = None,
        entry_price: float | None = None,
        implied_prob: float | None = None,
        confidence: float | None = None,
        shares: float | None = None,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        order_id = _extract_order_id(order)
        event_order_id = order_id or f"LOCAL#{market_id}#{int(datetime.now(timezone.utc).timestamp() * 1000)}"
        # Use explicit outcome if provided, otherwise try to extract from response
        if not outcome:
            outcome = _extract_order_outcome(order)

        with self._conn:
            position_row = self._conn.execute(
                """
                SELECT outcome, total_amount, avg_confidence, order_ids
                FROM positions
                WHERE market_id = ?
                """,
                (market_id,),
            ).fetchone()

            existing_total = float(position_row["total_amount"] or 0.0) if position_row else 0.0
            existing_avg = float(position_row["avg_confidence"] or 0.0) if position_row else 0.0
            existing_order_ids = _parse_order_ids(
                position_row["order_ids"] if position_row else None
            )
            existing_outcome = position_row["outcome"] if position_row else None

            if not outcome:
                outcome = existing_outcome or "UNKNOWN"
            elif existing_outcome and outcome != existing_outcome:
                logger.warning(
                    "Position outcome mismatch: market=%s existing=%s new=%s",
                    market_id,
                    existing_outcome,
                    outcome,
                )

            self._conn.execute(
                """
                INSERT INTO trade_log (
                    market_id, amount, outcome, order_id, timestamp
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (market_id, amount, outcome, order_id, timestamp),
            )

            if order_id and order_id not in existing_order_ids:
                existing_order_ids.append(order_id)

            trade_count_row = self._conn.execute(
                "SELECT COUNT(*) AS trade_count FROM trade_log WHERE market_id = ?",
                (market_id,),
            ).fetchone()
            trade_count = trade_count_row["trade_count"] if trade_count_row else 0

            latest_confidence = self._get_latest_confidence(market_id)
            new_avg_confidence = _update_avg_confidence(
                existing_avg,
                trade_count,
                latest_confidence,
            )
            new_total = existing_total + amount

            if position_row:
                self._conn.execute(
                    """
                    UPDATE positions
                    SET outcome = ?, total_amount = ?, avg_confidence = ?, order_ids = ?
                    WHERE market_id = ?
                    """,
                    (
                        outcome,
                        new_total,
                        new_avg_confidence,
                        json.dumps(existing_order_ids),
                        market_id,
                    ),
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO positions (
                        market_id, outcome, total_amount, avg_confidence, order_ids
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        market_id,
                        outcome,
                        new_total,
                        new_avg_confidence,
                        json.dumps(existing_order_ids),
                    ),
                )

            self._upsert_trade_outcome_entry(
                market_id=market_id,
                predicted_outcome=outcome,
                entry_price=entry_price,
                implied_prob=implied_prob,
                confidence=confidence,
                amount_usdc=amount,
                shares=shares,
                timestamp=timestamp,
            )
            self._upsert_trade_outcome_event(
                market_id=market_id,
                order_id=event_order_id,
                predicted_outcome=outcome,
                entry_price=entry_price,
                implied_prob=implied_prob,
                confidence=confidence,
                amount_usdc=amount,
                shares=shares,
                timestamp=timestamp,
            )

        logger.info(
            "Recorded trade: market=%s amount=%.2f outcome=%s order_id=%s",
            market_id,
            amount,
            outcome,
            order_id or "-",
        )

    def get_traded_market_ids(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT market_id FROM trade_log"
        ).fetchall()
        return [row["market_id"] for row in rows]

    def get_known_order_ids(self) -> set[str]:
        rows = self._conn.execute(
            """
            SELECT DISTINCT order_id
            FROM trade_log
            WHERE order_id IS NOT NULL AND TRIM(order_id) <> ''
            """
        ).fetchall()
        return {str(row["order_id"]) for row in rows if row["order_id"]}

    def get_unresolved_traded_market_ids(self) -> list[str]:
        rows = self._conn.execute(
            """
            SELECT market_id
            FROM trade_outcomes
            WHERE COALESCE(resolution_state, 'unresolved') = 'unresolved'
            """
        ).fetchall()
        return [str(row["market_id"]) for row in rows if row["market_id"]]

    def market_has_recent_fallback_edge(self, market_id: str, lookback: int = 3) -> bool:
        window = max(1, int(lookback))
        rows = self._conn.execute(
            """
            SELECT reasoning
            FROM analyses
            WHERE market_id = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (market_id, window),
        ).fetchall()
        for row in rows:
            reasoning = str(row["reasoning"] or "").lower()
            if "edge_source=fallback" in reasoning or "edge_source=none" in reasoning:
                return True
        return False

    def get_family_fallback_edge_rate(
        self,
        family: str,
        *,
        lookback: int = 200,
    ) -> tuple[float, int]:
        normalized_family = str(family or "").strip().lower()
        window = max(1, int(lookback))
        row = self._conn.execute(
            """
            SELECT
                COUNT(*) AS total_count,
                SUM(
                    CASE
                        WHEN LOWER(COALESCE(json_extract(decision_json, '$.edge_source'), '')) IN ('fallback', 'none')
                        THEN 1
                        ELSE 0
                    END
                ) AS fallback_count
            FROM (
                SELECT decision_json
                FROM decision_receipts
                WHERE LOWER(COALESCE(json_extract(audit_json, '$.market_family'), 'unknown')) = ?
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (normalized_family, window),
        ).fetchone()
        total_count = int(row["total_count"] or 0) if row else 0
        fallback_count = int(row["fallback_count"] or 0) if row and row["fallback_count"] is not None else 0
        if total_count <= 0:
            return 0.0, 0
        return fallback_count / total_count, total_count

    def get_family_outcome_snapshot(
        self,
        *,
        lookback: int = 400,
    ) -> dict[str, dict[str, float | int]]:
        """Return resolved trade performance by inferred market family."""
        window = max(1, int(lookback))
        rows = self._conn.execute(
            """
            SELECT
                t.market_id AS market_id,
                COALESCE(m.question, '') AS question,
                COALESCE(m.category, '') AS category,
                t.won AS won,
                t.pnl_estimate AS pnl_estimate
            FROM trade_outcomes t
            LEFT JOIN markets m ON m.id = t.market_id
            WHERE COALESCE(t.resolution_state, '') LIKE 'resolved%'
              AND t.won IS NOT NULL
            ORDER BY COALESCE(t.resolved_at, t.last_updated, '') DESC
            LIMIT ?
            """,
            (window,),
        ).fetchall()
        grouped: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {"sample_size": 0, "wins": 0, "pnl_total": 0.0}
        )
        for row in rows:
            family = self._infer_family_from_state_row(
                market_id=str(row["market_id"] or ""),
                question=str(row["question"] or ""),
                category=str(row["category"] or ""),
            )
            bucket = grouped[family]
            bucket["sample_size"] = int(bucket["sample_size"]) + 1
            if int(row["won"] or 0) == 1:
                bucket["wins"] = int(bucket["wins"]) + 1
            bucket["pnl_total"] = float(bucket["pnl_total"]) + float(row["pnl_estimate"] or 0.0)
        snapshot: dict[str, dict[str, float | int]] = {}
        for family, values in grouped.items():
            sample_size = int(values["sample_size"])
            wins = int(values["wins"])
            pnl_total = float(values["pnl_total"])
            snapshot[family] = {
                "sample_size": sample_size,
                "wins": wins,
                "win_rate": (wins / sample_size) if sample_size > 0 else 0.0,
                "pnl_total": pnl_total,
            }
        return snapshot

    def get_family_signal_snapshot(
        self,
        *,
        lookback: int = 400,
    ) -> dict[str, dict[str, float | int]]:
        """Return family PnL efficiency and high-confidence loss signals.

        This is intentionally read-only and uses existing trade_outcomes rows;
        no schema migration is needed for the execution score/sizing signal.
        """
        window = max(1, int(lookback))
        rows = self._conn.execute(
            """
            SELECT
                t.market_id AS market_id,
                COALESCE(m.question, '') AS question,
                COALESCE(m.category, '') AS category,
                t.won AS won,
                t.pnl_estimate AS pnl_estimate,
                t.amount_usdc AS amount_usdc,
                t.confidence AS confidence
            FROM trade_outcomes t
            LEFT JOIN markets m ON m.id = t.market_id
            WHERE COALESCE(t.resolution_state, '') LIKE 'resolved%'
              AND t.won IS NOT NULL
            ORDER BY COALESCE(t.resolved_at, t.last_updated, '') DESC
            LIMIT ?
            """,
            (window,),
        ).fetchall()
        grouped: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {
                "sample_size": 0,
                "wins": 0,
                "pnl_total": 0.0,
                "deployed_usdc": 0.0,
                "high_conf_losses": 0,
            }
        )
        for row in rows:
            family = self._infer_family_from_state_row(
                market_id=str(row["market_id"] or ""),
                question=str(row["question"] or ""),
                category=str(row["category"] or ""),
            )
            bucket = grouped[family]
            won = int(row["won"] or 0)
            confidence = float(row["confidence"] or 0.0)
            bucket["sample_size"] = int(bucket["sample_size"]) + 1
            if won == 1:
                bucket["wins"] = int(bucket["wins"]) + 1
            elif confidence >= 0.90:
                bucket["high_conf_losses"] = int(bucket["high_conf_losses"]) + 1
            bucket["pnl_total"] = float(bucket["pnl_total"]) + float(
                row["pnl_estimate"] or 0.0
            )
            bucket["deployed_usdc"] = float(bucket["deployed_usdc"]) + abs(
                float(row["amount_usdc"] or 0.0)
            )
        snapshot: dict[str, dict[str, float | int]] = {}
        for family, values in grouped.items():
            sample_size = int(values["sample_size"])
            wins = int(values["wins"])
            pnl_total = float(values["pnl_total"])
            deployed_usdc = float(values["deployed_usdc"])
            snapshot[family] = {
                "sample_size": sample_size,
                "wins": wins,
                "win_rate": (wins / sample_size) if sample_size > 0 else 0.0,
                "pnl_total": pnl_total,
                "deployed_usdc": deployed_usdc,
                "pnl_per_deployed_usdc": (
                    pnl_total / deployed_usdc if deployed_usdc > 0.0 else 0.0
                ),
                "high_conf_losses": int(values["high_conf_losses"]),
            }
        return snapshot

    def get_confidence_tier_outcomes(self) -> list[dict[str, float | int | str]]:
        rows = self._conn.execute(
            """
            SELECT
                CASE
                    WHEN confidence >= 0.90 THEN '0.90+'
                    WHEN confidence >= 0.80 THEN '0.80-0.89'
                    WHEN confidence >= 0.70 THEN '0.70-0.79'
                    WHEN confidence >= 0.60 THEN '0.60-0.69'
                    ELSE '<0.60'
                END AS tier,
                COUNT(*) AS sample_size,
                SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN won = 0 THEN 1 ELSE 0 END) AS losses,
                ROUND(SUM(COALESCE(pnl_estimate, 0.0)), 4) AS pnl_total
            FROM trade_outcomes
            WHERE confidence IS NOT NULL
              AND won IS NOT NULL
            GROUP BY tier
            ORDER BY
                CASE tier
                    WHEN '0.90+' THEN 1
                    WHEN '0.80-0.89' THEN 2
                    WHEN '0.70-0.79' THEN 3
                    WHEN '0.60-0.69' THEN 4
                    ELSE 5
                END
            """
        ).fetchall()
        snapshot: list[dict[str, float | int | str]] = []
        for row in rows:
            sample_size = int(row["sample_size"] or 0)
            wins = int(row["wins"] or 0)
            losses = int(row["losses"] or 0)
            snapshot.append(
                {
                    "tier": str(row["tier"]),
                    "sample_size": sample_size,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": (wins / sample_size) if sample_size > 0 else 0.0,
                    "pnl_total": float(row["pnl_total"] or 0.0),
                }
            )
        return snapshot

    def get_confidence_bucket_calibration(
        self,
        *,
        lookback_days: int = 14,
    ) -> list[dict[str, float | int | str]]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))).isoformat()
        rows = self._conn.execute(
            """
            SELECT
                CASE
                    WHEN confidence >= 0.90 THEN '0.90+'
                    WHEN confidence >= 0.80 THEN '0.80-0.89'
                    WHEN confidence >= 0.70 THEN '0.70-0.79'
                    WHEN confidence >= 0.60 THEN '0.60-0.69'
                    WHEN confidence >= 0.50 THEN '0.50-0.59'
                    ELSE '<0.50'
                END AS bucket,
                COUNT(*) AS sample_size,
                SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END) AS wins,
                AVG(COALESCE(confidence, 0.0)) AS mean_confidence
            FROM trade_outcomes
            WHERE confidence IS NOT NULL
              AND won IS NOT NULL
              AND COALESCE(resolved_at, last_updated, '') >= ?
            GROUP BY bucket
            ORDER BY
                CASE bucket
                    WHEN '0.90+' THEN 1
                    WHEN '0.80-0.89' THEN 2
                    WHEN '0.70-0.79' THEN 3
                    WHEN '0.60-0.69' THEN 4
                    WHEN '0.50-0.59' THEN 5
                    ELSE 6
                END
            """,
            (cutoff,),
        ).fetchall()
        calibration_rows: list[dict[str, float | int | str]] = []
        for row in rows:
            sample_size = int(row["sample_size"] or 0)
            wins = int(row["wins"] or 0)
            calibration_rows.append(
                {
                    "bucket": str(row["bucket"] or ""),
                    "sample_size": sample_size,
                    "wins": wins,
                    "win_rate": (wins / sample_size) if sample_size > 0 else 0.0,
                    "mean_confidence": float(row["mean_confidence"] or 0.0),
                }
            )
        return calibration_rows

    def load_confidence_calibration_buckets(
        self,
        *,
        days: int = 30,
    ) -> dict[str, dict[float, dict[str, float | int]]]:
        """Return confidence-bucket outcomes for global and family-specific calibration."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))
        ).isoformat()
        rows = self._conn.execute(
            """
            SELECT
                t.market_id AS market_id,
                COALESCE(m.question, '') AS question,
                COALESCE(m.category, '') AS category,
                t.confidence AS confidence,
                t.won AS won
            FROM trade_outcomes t
            LEFT JOIN markets m ON m.id = t.market_id
            WHERE t.confidence IS NOT NULL
              AND t.won IS NOT NULL
              AND COALESCE(t.resolved_at, t.last_updated, '') >= ?
            """,
            (cutoff,),
        ).fetchall()
        grouped: dict[str, dict[float, dict[str, float | int]]] = defaultdict(dict)
        for row in rows:
            confidence_value = float(row["confidence"] or 0.0)
            bounded_confidence = max(0.0, min(1.0, confidence_value))
            bucket = int(bounded_confidence * 10.0) / 10.0
            won = int(row["won"] or 0) == 1
            family = self._infer_family_from_state_row(
                market_id=str(row["market_id"] or ""),
                question=str(row["question"] or ""),
                category=str(row["category"] or ""),
            )
            for family_key in ("all", family):
                bucket_state = grouped[family_key].get(bucket)
                if bucket_state is None:
                    bucket_state = {
                        "sample_size": 0,
                        "wins": 0,
                        "total_confidence": 0.0,
                    }
                    grouped[family_key][bucket] = bucket_state
                bucket_state["sample_size"] = int(bucket_state["sample_size"]) + 1
                if won:
                    bucket_state["wins"] = int(bucket_state["wins"]) + 1
                bucket_state["total_confidence"] = float(bucket_state["total_confidence"]) + bounded_confidence

        snapshot: dict[str, dict[float, dict[str, float | int]]] = {}
        for family_key, bucket_map in grouped.items():
            snapshot[family_key] = {}
            for bucket, bucket_state in bucket_map.items():
                sample_size = int(bucket_state["sample_size"])
                wins = int(bucket_state["wins"])
                total_confidence = float(bucket_state["total_confidence"])
                snapshot[family_key][float(bucket)] = {
                    "sample_size": sample_size,
                    "wins": wins,
                    "win_rate": (wins / sample_size) if sample_size > 0 else 0.0,
                    "mean_confidence": (
                        total_confidence / sample_size
                        if sample_size > 0
                        else 0.0
                    ),
                }
        online_rows = self._conn.execute(
            """
            SELECT family, bucket, win_rate, sample_size
            FROM confidence_calibration_online
            WHERE sample_size > 0
            """
        ).fetchall()
        for row in online_rows:
            family_key = str(row["family"] or "all")
            bucket = float(row["bucket"] or 0.0)
            online_sample_size = int(row["sample_size"] or 0)
            if online_sample_size <= 0:
                continue
            online_win_rate = max(0.0, min(1.0, float(row["win_rate"] or 0.0)))
            family_snapshot = snapshot.setdefault(family_key, {})
            existing = family_snapshot.get(bucket)
            if existing is None:
                family_snapshot[bucket] = {
                    "sample_size": online_sample_size,
                    "wins": online_win_rate * online_sample_size,
                    "win_rate": online_win_rate,
                    "mean_confidence": bucket,
                }
                continue
            existing_sample_size = int(existing.get("sample_size", 0) or 0)
            combined_sample_size = existing_sample_size + online_sample_size
            if combined_sample_size <= 0:
                continue
            existing_win_rate = max(
                0.0,
                min(1.0, float(existing.get("win_rate", 0.0) or 0.0)),
            )
            combined_win_rate = (
                (existing_win_rate * existing_sample_size)
                + (online_win_rate * online_sample_size)
            ) / combined_sample_size
            existing_mean = float(existing.get("mean_confidence", bucket) or bucket)
            combined_mean = (
                (existing_mean * existing_sample_size) + (bucket * online_sample_size)
            ) / combined_sample_size
            family_snapshot[bucket] = {
                "sample_size": combined_sample_size,
                "wins": combined_win_rate * combined_sample_size,
                "win_rate": combined_win_rate,
                "mean_confidence": combined_mean,
            }
        return snapshot

    def record_online_confidence_calibration(
        self,
        *,
        market_id: str,
        confidence: float | None,
        won: bool | int | None,
        question: str = "",
        category: str = "",
        alpha: float = 0.15,
        max_samples_per_bucket: int = 500,
        updated_at: datetime | None = None,
    ) -> bool:
        if confidence is None or won is None:
            return False
        try:
            bounded_confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            return False
        sample_value = 1.0 if int(won) == 1 else 0.0
        bucket = int(bounded_confidence * 10.0) / 10.0
        family = self._infer_family_from_state_row(
            market_id=str(market_id or ""),
            question=str(question or ""),
            category=str(category or ""),
        )
        normalized_alpha = max(0.0, min(1.0, float(alpha or 0.15)))
        sample_cap = max(1, int(max_samples_per_bucket or 500))
        timestamp = (updated_at or datetime.now(timezone.utc)).isoformat()
        changed = False
        with self._conn:
            for family_key in ("all", family):
                row = self._conn.execute(
                    """
                    SELECT win_rate, sample_size
                    FROM confidence_calibration_online
                    WHERE family = ? AND bucket = ?
                    """,
                    (family_key, bucket),
                ).fetchone()
                if row is None:
                    next_win_rate = sample_value
                    next_sample_size = 1
                else:
                    old_win_rate = max(0.0, min(1.0, float(row["win_rate"] or 0.0)))
                    old_sample_size = max(0, int(row["sample_size"] or 0))
                    next_win_rate = (
                        normalized_alpha * sample_value
                        + (1.0 - normalized_alpha) * old_win_rate
                    )
                    next_sample_size = min(sample_cap, old_sample_size + 1)
                self._conn.execute(
                    """
                    INSERT INTO confidence_calibration_online (
                        family, bucket, win_rate, sample_size, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(family, bucket) DO UPDATE SET
                        win_rate = excluded.win_rate,
                        sample_size = excluded.sample_size,
                        updated_at = excluded.updated_at
                    """,
                    (family_key, bucket, next_win_rate, next_sample_size, timestamp),
                )
                changed = True
        return changed

    def get_runtime_flag(self, key: str) -> str | None:
        """Return a persisted runtime flag value, or None if unset."""
        normalized_key = str(key or "").strip()
        if not normalized_key:
            return None
        row = self._conn.execute(
            "SELECT value FROM runtime_flags WHERE key = ?",
            (normalized_key,),
        ).fetchone()
        if row is None:
            return None
        return str(row["value"])

    def set_runtime_flag(self, key: str, value: str) -> None:
        """Persist a runtime flag across bot restarts."""
        normalized_key = str(key or "").strip()
        if not normalized_key:
            raise ValueError("runtime flag key must be non-empty")
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO runtime_flags (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (normalized_key, str(value), timestamp),
            )

    def clear_runtime_flag(self, key: str) -> bool:
        """Delete a runtime flag. Returns True when a row was removed."""
        normalized_key = str(key or "").strip()
        if not normalized_key:
            return False
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM runtime_flags WHERE key = ?",
                (normalized_key,),
            )
        return int(cursor.rowcount or 0) > 0

    def neutralize_pathological_online_calibration(
        self,
        *,
        family: str,
        win_rate_floor: float = 0.01,
        win_rate_ceiling: float = 0.99,
        min_samples: int = 30,
        neutral_win_rate: float = 0.50,
    ) -> int:
        """Reset extreme online calibration buckets toward a neutral prior.

        Pathological entries (e.g. sports@0.7 with ~0% WR at high sample count)
        otherwise permanently crush confidence via historical shrink.
        """
        family_key = str(family or "").strip().lower()
        if not family_key:
            return 0
        floor = max(0.0, min(1.0, float(win_rate_floor)))
        ceiling = max(floor, min(1.0, float(win_rate_ceiling)))
        sample_floor = max(1, int(min_samples))
        neutral = max(0.0, min(1.0, float(neutral_win_rate)))
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._conn:
            cursor = self._conn.execute(
                """
                UPDATE confidence_calibration_online
                SET win_rate = ?, updated_at = ?
                WHERE family = ?
                  AND sample_size >= ?
                  AND (win_rate < ? OR win_rate > ?)
                """,
                (neutral, timestamp, family_key, sample_floor, floor, ceiling),
            )
        return int(cursor.rowcount or 0)

    def record_online_confidence_calibration_from_trade(
        self,
        market_id: str,
        *,
        alpha: float = 0.15,
        max_samples_per_bucket: int = 500,
    ) -> bool:
        row = self._conn.execute(
            """
            SELECT
                t.confidence AS confidence,
                t.won AS won,
                COALESCE(m.question, '') AS question,
                COALESCE(m.category, '') AS category,
                COALESCE(t.resolved_at, t.last_updated, '') AS updated_at
            FROM trade_outcomes t
            LEFT JOIN markets m ON m.id = t.market_id
            WHERE t.market_id = ?
              AND t.won IS NOT NULL
            """,
            (market_id,),
        ).fetchone()
        if row is None:
            return False
        confidence = row["confidence"]
        if confidence is None:
            confidence = self._get_latest_confidence(market_id)
        updated_at = _parse_timestamp(row["updated_at"]) or datetime.now(timezone.utc)
        return self.record_online_confidence_calibration(
            market_id=market_id,
            confidence=confidence,
            won=row["won"],
            question=str(row["question"] or ""),
            category=str(row["category"] or ""),
            alpha=alpha,
            max_samples_per_bucket=max_samples_per_bucket,
            updated_at=updated_at,
        )

    def get_exchange_realized_pnl_total(self) -> float:
        row = self._conn.execute(
            """
            SELECT COALESCE(SUM(pnl_realized), 0.0) AS pnl_total
            FROM exchange_settlements
            """
        ).fetchone()
        if not row:
            return 0.0
        return float(row["pnl_total"] or 0.0)

    def get_exchange_realized_pnl_since_hours(self, hours: float) -> float:
        """Sum realized PnL from exchange settlements within the last *hours*."""
        if hours <= 0:
            return self.get_exchange_realized_pnl_total()
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=float(hours))
        ).isoformat()
        row = self._conn.execute(
            """
            SELECT COALESCE(SUM(pnl_realized), 0.0) AS pnl_total
            FROM exchange_settlements
            WHERE settled_at IS NOT NULL
              AND settled_at >= ?
            """,
            (cutoff,),
        ).fetchone()
        if not row:
            return 0.0
        return float(row["pnl_total"] or 0.0)

    def get_attributed_daily_realized_pnl(self, since: datetime) -> float:
        """Realized PnL since *since*, restricted to markets entered since *since*.

        Daily risk gates need today's decision quality, not settlement timing:
        positions opened on earlier days can settle in a batch today and would
        otherwise dominate a balance-delta drawdown measure. Only settlements
        whose market also has a trade_log entry inside the window count.
        """
        since_utc = since if since.tzinfo is not None else since.replace(tzinfo=timezone.utc)
        since_iso = since_utc.astimezone(timezone.utc).isoformat()
        row = self._conn.execute(
            """
            SELECT COALESCE(SUM(s.pnl_realized), 0.0) AS pnl_total
            FROM exchange_settlements s
            WHERE s.settled_at IS NOT NULL
              AND s.settled_at >= ?
              AND EXISTS (
                  SELECT 1
                  FROM trade_log t
                  WHERE t.market_id = s.market_id
                    AND t.timestamp >= ?
              )
            """,
            (since_iso, since_iso),
        ).fetchone()
        if not row:
            return 0.0
        return float(row["pnl_total"] or 0.0)

    def get_prefix_pnl_stats(self, prefix: str) -> dict[str, float | int]:
        """Aggregate settlement PnL for markets whose id starts with *prefix*.

        Returns ``{n, wins, total_pnl}`` from ``exchange_settlements``.
        """
        row = self._conn.execute(
            """
            SELECT
                COUNT(*) AS n,
                COALESCE(SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END), 0) AS wins,
                COALESCE(SUM(pnl_realized), 0.0) AS total_pnl
            FROM exchange_settlements
            WHERE market_id LIKE ? || '%'
            """,
            (prefix,),
        ).fetchone()
        if not row:
            return {"n": 0, "wins": 0, "total_pnl": 0.0}
        return {
            "n": int(row["n"] or 0),
            "wins": int(row["wins"] or 0),
            "total_pnl": float(row["total_pnl"] or 0.0),
        }

    def get_market_analysis_count_today(self, market_id: str) -> int:
        """Count analyses for *market_id* since midnight UTC today."""
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM analyses
            WHERE market_id = ?
              AND timestamp >= date('now')
            """,
            (market_id,),
        ).fetchone()
        return int(row["cnt"] or 0) if row else 0

    def record_research_queue_entry(
        self,
        *,
        market_id: str,
        cycle_id: str,
        gate_name: str,
        reason: str,
        threshold_gap: float = 0.0,
        what_to_learn_next: str | None = None,
        expires_at: str | None = None,
        last_decision_json: str | None = None,
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO research_queue_entries
                (market_id, cycle_id, queued_at, gate_name, reason,
                 threshold_gap, what_to_learn_next, last_seen, expires_at,
                 last_decision_json, times_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(market_id) DO UPDATE SET
                cycle_id = excluded.cycle_id,
                gate_name = excluded.gate_name,
                reason = excluded.reason,
                threshold_gap = excluded.threshold_gap,
                what_to_learn_next = excluded.what_to_learn_next,
                last_seen = excluded.last_seen,
                expires_at = excluded.expires_at,
                last_decision_json = COALESCE(excluded.last_decision_json, last_decision_json),
                times_seen = COALESCE(research_queue_entries.times_seen, 0) + 1
            """,
            (
                market_id,
                cycle_id,
                now_iso,
                gate_name,
                reason,
                round(max(0.0, threshold_gap), 4),
                what_to_learn_next,
                now_iso,
                expires_at,
                last_decision_json,
            ),
        )
        self._conn.commit()

    def mark_research_queue_drain_attempt(
        self,
        market_id: str,
        *,
        cycle_id: str | None = None,
        attempted_at: datetime | None = None,
    ) -> None:
        """Record a queue-drain probe in the existing JSON audit payload."""
        row = self._conn.execute(
            """
            SELECT last_decision_json
            FROM research_queue_entries
            WHERE market_id = ?
            """,
            (market_id,),
        ).fetchone()
        if not row:
            return
        payload: dict[str, Any]
        raw_payload = row["last_decision_json"]
        if raw_payload:
            try:
                loaded = json.loads(raw_payload)
            except (TypeError, ValueError):
                loaded = {}
            payload = loaded if isinstance(loaded, dict) else {}
        else:
            payload = {}
        raw_audit = payload.get("audit")
        audit: dict[str, Any] = raw_audit if isinstance(raw_audit, dict) else {}
        try:
            attempts = int(audit.get("research_queue_drain_attempts") or 0)
        except (TypeError, ValueError):
            attempts = 0
        timestamp = attempted_at or datetime.now(timezone.utc)
        audit["research_queue_drain_attempts"] = attempts + 1
        audit["research_queue_last_drain_attempt_at"] = timestamp.isoformat()
        if cycle_id:
            audit["research_queue_last_drain_cycle_id"] = cycle_id
        payload["audit"] = audit
        with self._conn:
            self._conn.execute(
                """
                UPDATE research_queue_entries
                SET last_decision_json = ?
                WHERE market_id = ?
                """,
                (json.dumps(payload, sort_keys=True), market_id),
            )

    @staticmethod
    def research_queue_drain_attempt_metadata(
        entry: dict[str, Any],
    ) -> tuple[int, datetime | None]:
        """Return persisted drain attempts and most recent attempt timestamp."""
        payload: dict[str, Any] | None = None
        audit: dict[str, Any] = {}
        decision_json = entry.get("last_decision_json")
        if decision_json:
            try:
                loaded = json.loads(decision_json)
            except (TypeError, ValueError):
                loaded = None
            if isinstance(loaded, dict):
                payload = loaded
                raw_audit = loaded.get("audit")
                if isinstance(raw_audit, dict):
                    audit = raw_audit
        attempts_raw = audit.get("research_queue_drain_attempts")
        if attempts_raw is None and isinstance(payload, dict):
            attempts_raw = payload.get("research_queue_drain_attempts")
        try:
            attempts = max(0, int(attempts_raw or 0))
        except (TypeError, ValueError):
            attempts = 0
        last_raw = audit.get("research_queue_last_drain_attempt_at")
        if last_raw is None and isinstance(payload, dict):
            last_raw = payload.get("research_queue_last_drain_attempt_at")
        last_attempt: datetime | None = None
        if isinstance(last_raw, str) and last_raw.strip():
            try:
                parsed = datetime.fromisoformat(last_raw.strip().replace("Z", "+00:00"))
                last_attempt = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                last_attempt = None
        return attempts, last_attempt

    @staticmethod
    def is_repeated_low_yield_research_entry(
        entry: dict[str, Any],
        *,
        min_attempts: int = 4,
        min_times_seen: int = 8,
        min_gap: float = 0.08,
    ) -> bool:
        """Detect stale synthetic queue placeholders without hard-blocking families."""
        payload: dict[str, Any] | None = None
        audit: dict[str, Any] = {}
        decision_json = entry.get("last_decision_json")
        if decision_json:
            try:
                loaded = json.loads(decision_json)
            except (TypeError, ValueError):
                loaded = None
            if isinstance(loaded, dict):
                payload = loaded
                raw_audit = loaded.get("audit")
                if isinstance(raw_audit, dict):
                    audit = raw_audit
        attempts, _last_attempt = MarketStateManager.research_queue_drain_attempt_metadata(entry)
        try:
            times_seen = max(0, int(entry.get("times_seen") or 0))
        except (TypeError, ValueError):
            times_seen = 0
        if attempts < max(1, int(min_attempts)) and times_seen < max(1, int(min_times_seen)):
            return False

        threshold_gap = entry.get("threshold_gap")
        if threshold_gap is None:
            threshold_gap = audit.get("threshold_gap")
        try:
            gap = float(threshold_gap) if threshold_gap is not None else None
        except (TypeError, ValueError):
            gap = None
        if gap is not None and gap <= max(0.0, float(min_gap)):
            return False

        source_match = str(
            audit.get("source_match_class")
            or (payload or {}).get("source_match_class")
            or ""
        ).strip().lower()
        evidence_basis = str(
            audit.get("evidence_basis")
            or audit.get("evidence_basis_class")
            or (payload or {}).get("evidence_basis")
            or ""
        ).strip().lower()
        primary_source_url = str(
            audit.get("primary_source_url")
            or (payload or {}).get("primary_source_url")
            or ""
        ).strip()
        evidence_quality = 0.0
        for source in (audit, payload or {}, entry):
            raw_eq = source.get("evidence_quality") if isinstance(source, dict) else None
            if isinstance(raw_eq, (int, float)):
                evidence_quality = max(0.0, min(1.0, float(raw_eq)))
                break
        if (
            evidence_quality >= 0.65
            or evidence_basis == "direct"
            or source_match == "settlement_aligned"
            or primary_source_url
        ):
            return False

        decision_origin = str(
            audit.get("decision_origin") or (payload or {}).get("decision_origin") or ""
        ).strip().lower()
        synthetic = bool(audit.get("synthetic_decision")) or decision_origin.startswith(
            "synthetic_"
        )
        confidence = None
        for source in (payload or {}, audit):
            raw_conf = source.get("confidence") if isinstance(source, dict) else None
            if isinstance(raw_conf, (int, float)):
                confidence = float(raw_conf)
                break
        placeholder_confidence = confidence is None or abs(confidence - 0.50) <= 0.05
        edge_source = str(
            audit.get("edge_source") or (payload or {}).get("edge_source") or ""
        ).strip().lower()
        reason = str(entry.get("reason") or audit.get("final_reason") or "").lower()
        placeholder_reason = (
            "pre_analysis" in reason
            or "soft_research" in reason
            or "analysis_cap" in reason
            or "lifetime" in reason
        )
        return bool(
            (synthetic or placeholder_reason)
            and placeholder_confidence
            and evidence_quality <= 0.05
            and edge_source in {"", "none", "fallback"}
        )

    def get_active_research_entries(
        self,
        lookback_hours: int = 6,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        now_iso = datetime.now(timezone.utc).isoformat()
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=max(1, int(lookback_hours)))
        ).isoformat()
        rows = self._conn.execute(
            """
            SELECT market_id, cycle_id, queued_at, gate_name, reason,
                   threshold_gap, what_to_learn_next, last_seen, expires_at,
                   last_decision_json, COALESCE(times_seen, 1) AS times_seen
            FROM research_queue_entries
            WHERE last_seen >= ?
              AND (expires_at IS NULL OR expires_at >= ?)
            ORDER BY last_seen DESC
            LIMIT ?
            """,
            (cutoff, now_iso, max(1, int(limit))),
        ).fetchall()
        return [dict(row) for row in rows]

    def prune_expired_research_entries(self) -> int:
        now_iso = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            "DELETE FROM research_queue_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now_iso,),
        )
        self._conn.commit()
        return cursor.rowcount

    _JURISDICTION_SPORTS_HOLD_MARKERS = (
        "jurisdiction_sports_hold",
        "jurisdiction_sports_analysis_held",
        "jurisdiction_sports_blocked",
    )

    @staticmethod
    def is_jurisdiction_sports_hold_entry(entry: dict[str, Any]) -> bool:
        """True when a queue row is a legacy sports jurisdiction parking entry.

        New runs keep jurisdiction errors order-scoped, but historical rows must
        remain excluded so they cannot waste probe slots intended for
        edge/conviction near-misses.
        """
        gate_name = str(entry.get("gate_name") or "").strip().lower()
        reason = str(entry.get("reason") or "").strip().lower()
        markers = MarketStateManager._JURISDICTION_SPORTS_HOLD_MARKERS
        if any(marker in gate_name for marker in markers):
            return True
        if any(marker in reason for marker in markers):
            return True
        decision_json = entry.get("last_decision_json")
        if decision_json:
            try:
                payload = json.loads(decision_json)
            except (TypeError, ValueError):
                payload = None
            if isinstance(payload, dict):
                audit = payload.get("audit")
                if isinstance(audit, dict):
                    final_reason = str(audit.get("final_reason") or "").strip().lower()
                    if any(marker in final_reason for marker in markers):
                        return True
        return False

    _SOFT_RESEARCH_DRAIN_PLACEHOLDER_MARKERS = (
        "soft_research",
        "pre_analysis_score_soft_research",
        "pre_analysis_score_far_below_min",
        "pre_analysis_score_below_min",
    )

    @staticmethod
    def is_soft_research_drain_placeholder(entry: dict[str, Any]) -> bool:
        """True for pre-analysis soft-research placeholders that starve drain.

        Soft-research rows dominate the queue by age. When priority filtering is
        active they should not consume the over-fetch window ahead of edge /
        conviction near-misses (score-promotion still resurfaces soft-research).
        """
        if MarketStateManager.is_repeated_low_yield_research_entry(entry):
            return True
        gate_name = str(entry.get("gate_name") or "").strip().lower()
        reason = str(entry.get("reason") or "").strip().lower()
        markers = MarketStateManager._SOFT_RESEARCH_DRAIN_PLACEHOLDER_MARKERS
        if any(marker in gate_name for marker in markers):
            return True
        if any(marker in reason for marker in markers):
            return True
        # Movement-score soft band uses gate_name=pre_analysis_movement_score.
        if "pre_analysis" in gate_name and "soft_research" in reason:
            return True
        return False

    def get_drainable_research_entries(
        self,
        *,
        min_age_hours: float = 1.0,
        max_age_hours: float = 12.0,
        limit: int = 5,
        excluded_market_ids: tuple[str, ...] | None = None,
        included_market_ids: tuple[str, ...] | None = None,
        min_priority: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return research-queue entries eligible for forced re-analysis.

        Eligibility window is ``min_age_hours <= age_since_queued <= max_age_hours``.
        Caller passes ``excluded_market_ids`` (e.g. tickers already on this cycle's
        candidate list, already-traded markets, or recently-resolved markets) so
        we don't double-promote. ``included_market_ids`` optionally restricts
        drain candidates to the current filtered market set, preventing stale
        queue rows from consuming the over-fetch pool.

        When ``min_priority`` is set, soft-research / pre-analysis placeholders
        are excluded from the candidate set and remaining rows are ranked by
        estimated priority (desc) then age (oldest first) so aged soft-research
        cannot starve edge/conviction near-misses in the over-fetch window.

        ``min_priority`` (optional) filters out entries whose proxied priority is
        below the cutoff. Priority is read from
        ``estimate_research_entry_priority``: ``last_decision_json.audit
        .pre_analysis_score`` when present, otherwise ``1.0 - threshold_gap``.
        Entries without enough metadata to estimate a priority are kept (treated
        as "unknown" rather than penalized). When the filter is active the
        function over-fetches so priority ranking still yields enough
        qualifying rows after pruning. Callers that need per-cycle telemetry
        on how many were skipped should call ``estimate_research_entry_priority``
        themselves; this entry point only returns the qualifying rows.

        Legacy sports jurisdiction holds are excluded from drain promotion.
        """
        now = datetime.now(timezone.utc)
        max_cutoff_iso = (
            now - timedelta(hours=max(0.0, float(max_age_hours)))
        ).isoformat()
        min_cutoff_iso = (
            now - timedelta(hours=max(0.0, float(min_age_hours)))
        ).isoformat()
        excluded = tuple(
            str(mid).strip()
            for mid in (excluded_market_ids or ())
            if str(mid or "").strip()
        )
        included: tuple[str, ...] | None = None
        if included_market_ids is not None:
            included = tuple(
                str(mid).strip()
                for mid in included_market_ids
                if str(mid or "").strip()
            )
            if not included:
                return []
        effective_limit = max(0, int(limit))
        fetch_limit = effective_limit
        # Over-fetch when post-filters (priority / jurisdiction / soft-research)
        # may drop rows. Fetch a wider window so priority ranking can surface
        # high-value near-misses buried behind aged soft-research.
        if effective_limit > 0:
            fetch_limit = max(effective_limit, effective_limit * 8)
        where_clauses = [
            "queued_at >= ?",
            "queued_at <= ?",
            "(expires_at IS NULL OR expires_at >= ?)",
        ]
        params_list: list[Any] = [
            max_cutoff_iso,
            min_cutoff_iso,
            now.isoformat(),
        ]
        if included:
            placeholders = ",".join("?" * len(included))
            where_clauses.append(f"market_id IN ({placeholders})")
            params_list.extend(included)
        if excluded:
            placeholders = ",".join("?" * len(excluded))
            where_clauses.append(f"market_id NOT IN ({placeholders})")
            params_list.extend(excluded)
        # When priority filtering is active, exclude soft-research placeholders
        # in SQL so they cannot fill the over-fetch window.
        if min_priority is not None:
            soft_markers = MarketStateManager._SOFT_RESEARCH_DRAIN_PLACEHOLDER_MARKERS
            soft_clauses = []
            for marker in soft_markers:
                soft_clauses.append("lower(coalesce(gate_name, '')) NOT LIKE ?")
                params_list.append(f"%{marker}%")
                soft_clauses.append("lower(coalesce(reason, '')) NOT LIKE ?")
                params_list.append(f"%{marker}%")
            where_clauses.append("(" + " AND ".join(soft_clauses) + ")")
        sql = f"""
            SELECT market_id, cycle_id, queued_at, gate_name, reason,
                   threshold_gap, what_to_learn_next, last_seen, expires_at,
                   last_decision_json, COALESCE(times_seen, 1) AS times_seen
            FROM research_queue_entries
            WHERE {' AND '.join(where_clauses)}
            ORDER BY queued_at ASC
            LIMIT ?
        """
        params_list.append(fetch_limit)
        params: tuple[Any, ...] = tuple(params_list)
        rows = self._conn.execute(sql, params).fetchall()
        results = [
            dict(row)
            for row in rows
            if not MarketStateManager.is_jurisdiction_sports_hold_entry(dict(row))
            and not (
                min_priority is not None
                and MarketStateManager.is_soft_research_drain_placeholder(dict(row))
            )
        ]
        if not results:
            return []
        if min_priority is None:
            return results[:effective_limit] if effective_limit else results

        cutoff = float(min_priority)

        def _drain_rank_key(entry: dict[str, Any]) -> tuple[float, str]:
            priority = self.estimate_research_entry_priority(entry)
            # Unknown priority ranks as admissible (0.0 sort key inverted via
            # treating None as meeting cutoff and sorting just below cutoff).
            rank_priority = float(cutoff) if priority is None else float(priority)
            queued_at = str(entry.get("queued_at") or "")
            return (-rank_priority, queued_at)

        ranked = sorted(results, key=_drain_rank_key)
        filtered: list[dict[str, Any]] = []
        for entry in ranked:
            priority = self.estimate_research_entry_priority(entry)
            if priority is None or priority >= cutoff:
                filtered.append(entry)
            if len(filtered) >= effective_limit:
                break
        return filtered

    @staticmethod
    def estimate_research_entry_priority(entry: dict[str, Any]) -> float | None:
        """Best-effort priority for a queued research entry.

        Prefer explicit persisted ``research_priority`` first, then enrich it
        with near-miss, repeated-sighting, source-alignment, evidence, and
        family-performance signals. Returns ``None`` when no signal is
        available so callers can treat unknown-priority entries as admissible
        rather than low-priority.
        """
        payload: dict[str, Any] | None = None
        audit: dict[str, Any] = {}
        decision_json = entry.get("last_decision_json")
        if decision_json:
            try:
                payload = json.loads(decision_json)
            except (TypeError, ValueError):
                payload = None
            if isinstance(payload, dict):
                raw_audit = payload.get("audit")
                if isinstance(raw_audit, dict):
                    audit = raw_audit
        signals_present = False
        priority: float | None = None
        for source in (entry, audit, payload or {}):
            score = source.get("research_priority") if isinstance(source, dict) else None
            if isinstance(score, (int, float)):
                priority = float(score)
                signals_present = True
                break
        if priority is None:
            for source in (audit, payload or {}):
                score = source.get("pre_analysis_score") if isinstance(source, dict) else None
                if isinstance(score, (int, float)):
                    priority = float(score)
                    signals_present = True
                    break
        threshold_gap = entry.get("threshold_gap")
        if threshold_gap is None and isinstance(audit, dict):
            threshold_gap = audit.get("threshold_gap")
        if isinstance(threshold_gap, (int, float)):
            gap = max(0.0, float(threshold_gap))
            if priority is None:
                priority = max(0.0, 1.0 - gap)
            if gap <= 0.03:
                priority += 0.10
            elif gap <= 0.08:
                priority += 0.05
            signals_present = True
        try:
            times_seen = max(0, int(entry.get("times_seen") or 0))
        except (TypeError, ValueError):
            times_seen = 0
        if times_seen > 1:
            priority = (priority if priority is not None else 0.0) + min(
                0.10,
                float(times_seen - 1) * 0.01,
            )
            signals_present = True
        source_match = str(
            (audit or {}).get("source_match_class")
            or (payload or {}).get("source_match_class")
            or ""
        ).strip().lower()
        if source_match == "settlement_aligned":
            priority = (priority if priority is not None else 0.0) + 0.12
            signals_present = True
        evidence_quality = None
        for source in (audit, payload or {}, entry):
            raw_eq = source.get("evidence_quality") if isinstance(source, dict) else None
            if isinstance(raw_eq, (int, float)):
                evidence_quality = float(raw_eq)
                break
        if evidence_quality is not None and evidence_quality >= 0.90:
            priority = (priority if priority is not None else 0.0) + 0.05
            signals_present = True
        family_pnl = (audit or {}).get("historical_family_pnl_total")
        family_samples = (
            (audit or {}).get("historical_family_samples")
            or (audit or {}).get("historical_family_sample_size")
        )
        if isinstance(family_pnl, (int, float)) and isinstance(family_samples, (int, float)):
            if float(family_pnl) > 0.0 and int(family_samples) >= 20:
                priority = (priority if priority is not None else 0.0) + 0.10
                signals_present = True
            elif float(family_pnl) > 0.0 and int(family_samples) >= 10:
                priority = (priority if priority is not None else 0.0) + 0.05
                signals_present = True
        reason_text = str(entry.get("reason") or (audit or {}).get("final_reason") or "").lower()
        if "extended_research_cooldown" in reason_text and (
            (isinstance(threshold_gap, (int, float)) and float(threshold_gap) <= 0.08)
            or times_seen >= 3
            or source_match == "settlement_aligned"
        ):
            priority = (priority if priority is not None else 0.0) + 0.08
            signals_present = True
        gate_name = str(entry.get("gate_name") or "").strip().lower()
        if MarketStateManager.is_jurisdiction_sports_hold_entry(entry):
            # Legacy jurisdiction parking rows are not drain candidates.
            return 0.0
        if gate_name == "conviction_repair":
            # Repair passes already found strong edge/evidence but produced no
            # executable decision; these are the highest-value retry candidates
            # in the queue (June 2026: 197 parked with zero prioritized drains).
            priority = (priority if priority is not None else 0.0) + 0.15
            signals_present = True
        # Edge near-misses within 3pp of the gate are high-value drain targets
        # (Jul 2026: weather EQ=1.0 setups parked at 0.12 vs 0.14 WEATHER_MIN_EDGE).
        _EDGE_NEAR_MISS_MARKERS = (
            "edge_gate_blocked",
            "edge_below_min",
            "edge below min",
            "weather_evidence_quality_below_min",
        )
        if any(marker in reason_text for marker in _EDGE_NEAR_MISS_MARKERS) and (
            isinstance(threshold_gap, (int, float)) and float(threshold_gap) <= 0.03
        ):
            priority = (priority if priority is not None else 0.0) + 0.15
            signals_present = True
        if MarketStateManager.is_repeated_low_yield_research_entry(entry):
            if priority is None:
                return 0.20
            priority = min(float(priority), 0.25)
            signals_present = True
        if not signals_present or priority is None:
            return None
        return max(0.0, min(1.0, float(priority)))

    @staticmethod
    def _infer_family_from_state_row(*, market_id: str, question: str, category: str) -> str:
        return family_from_text(f"{market_id} {question} {category}")

    def record_resolution(
        self,
        market_id: str,
        winning_outcome: str,
        resolved_at: datetime | None,
        *,
        online_calibration_enabled: bool = False,
        online_calibration_alpha: float = 0.15,
        online_calibration_max_samples_per_bucket: int = 500,
    ) -> bool:
        resolved_ts = resolved_at or datetime.now(timezone.utc)
        row = self._conn.execute(
            """
            SELECT predicted_outcome, entry_price, amount_usdc, shares, resolved_winning_outcome
            FROM trade_outcomes
            WHERE market_id = ?
            """,
            (market_id,),
        ).fetchone()
        if not row:
            return False
        existing_winner = row["resolved_winning_outcome"]
        if existing_winner and existing_winner == winning_outcome:
            return False
        predicted_outcome = row["predicted_outcome"]
        won = int(predicted_outcome == winning_outcome) if predicted_outcome else None
        pnl_estimate = _estimate_pnl(
            entry_price=row["entry_price"],
            amount_usdc=row["amount_usdc"],
            shares=row["shares"],
            won=won,
        )
        with self._conn:
            self._conn.execute(
                """
                UPDATE trade_outcomes
                SET resolved_winning_outcome = ?, won = ?, pnl_estimate = ?, resolved_at = ?, last_updated = ?,
                    resolution_state = 'resolved_valid'
                WHERE market_id = ?
                """,
                (
                    winning_outcome,
                    won,
                    pnl_estimate,
                    resolved_ts.isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                    market_id,
                ),
            )
            self._conn.execute(
                """
                UPDATE trade_outcome_events
                SET resolved_winning_outcome = ?, won = ?, pnl_estimate = ?, resolved_at = ?, resolution_state = 'resolved_valid'
                WHERE market_id = ?
                """,
                (
                    winning_outcome,
                    won,
                    pnl_estimate,
                    resolved_ts.isoformat(),
                    market_id,
                ),
            )
        logger.info(
            "Recorded resolution: market=%s winning=%s won=%s pnl=%.2f",
            market_id,
            winning_outcome,
            won,
            pnl_estimate if pnl_estimate is not None else 0.0,
        )
        if online_calibration_enabled:
            self.record_online_confidence_calibration_from_trade(
                market_id,
                alpha=online_calibration_alpha,
                max_samples_per_bucket=online_calibration_max_samples_per_bucket,
            )
        return True

    def record_exchange_settlement(
        self,
        *,
        settlement_id: str,
        market_id: str,
        winning_outcome: str | None,
        predicted_outcome: str | None,
        pnl_realized: float | None,
        contracts: int | None,
        avg_price: float | None,
        settled_at: datetime | None,
        raw: dict[str, Any],
        online_calibration_enabled: bool = False,
        online_calibration_alpha: float = 0.15,
        online_calibration_max_samples_per_bucket: int = 500,
    ) -> None:
        normalized_settlement_id = str(settlement_id or "").strip()
        normalized_market_id = str(market_id or "").strip()
        if not normalized_settlement_id or not normalized_market_id:
            return
        normalized_winning_outcome = str(winning_outcome or "").strip().upper() or None
        if normalized_winning_outcome not in {None, "YES", "NO"}:
            normalized_winning_outcome = None
        normalized_predicted_outcome = str(predicted_outcome or "").strip().upper() or None
        if normalized_predicted_outcome not in {None, "YES", "NO"}:
            normalized_predicted_outcome = None
        won: int | None = None
        if normalized_winning_outcome and normalized_predicted_outcome:
            won = int(normalized_winning_outcome == normalized_predicted_outcome)
        timestamp = (settled_at or datetime.now(timezone.utc)).isoformat()
        realized_pnl = float(pnl_realized or 0.0)
        normalized_contracts = int(contracts or 0)
        normalized_avg_price = (
            max(0.0, min(1.0, float(avg_price)))
            if avg_price is not None
            else None
        )
        amount_usdc: float | None = None
        if normalized_avg_price is not None and normalized_contracts > 0:
            amount_usdc = float(normalized_contracts) * normalized_avg_price
        resolution_state = (
            "resolved_exchange"
            if normalized_winning_outcome is not None
            else "unresolved_exchange"
        )
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO exchange_settlements (
                    settlement_id, market_id, predicted_outcome, winning_outcome, won,
                    pnl_realized, contracts, avg_price, settled_at, raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(settlement_id) DO UPDATE SET
                    market_id = excluded.market_id,
                    predicted_outcome = excluded.predicted_outcome,
                    winning_outcome = excluded.winning_outcome,
                    won = excluded.won,
                    pnl_realized = excluded.pnl_realized,
                    contracts = excluded.contracts,
                    avg_price = excluded.avg_price,
                    settled_at = excluded.settled_at,
                    raw_json = excluded.raw_json
                """,
                (
                    normalized_settlement_id,
                    normalized_market_id,
                    normalized_predicted_outcome,
                    normalized_winning_outcome,
                    won,
                    realized_pnl,
                    normalized_contracts,
                    normalized_avg_price,
                    timestamp,
                    json.dumps(raw or {}, sort_keys=True, default=str),
                ),
            )
            self._conn.execute(
                """
                INSERT INTO trade_outcomes (
                    market_id, predicted_outcome, entry_price, implied_prob, confidence, amount_usdc, shares,
                    resolved_winning_outcome, won, pnl_estimate, resolved_at, last_updated, resolution_state
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(market_id) DO UPDATE SET
                    predicted_outcome = COALESCE(trade_outcomes.predicted_outcome, excluded.predicted_outcome),
                    entry_price = COALESCE(trade_outcomes.entry_price, excluded.entry_price),
                    amount_usdc = COALESCE(trade_outcomes.amount_usdc, excluded.amount_usdc),
                    shares = COALESCE(trade_outcomes.shares, excluded.shares),
                    resolved_winning_outcome = COALESCE(excluded.resolved_winning_outcome, trade_outcomes.resolved_winning_outcome),
                    won = COALESCE(excluded.won, trade_outcomes.won),
                    pnl_estimate = COALESCE(excluded.pnl_estimate, trade_outcomes.pnl_estimate),
                    resolved_at = COALESCE(excluded.resolved_at, trade_outcomes.resolved_at),
                    last_updated = excluded.last_updated,
                    resolution_state = CASE
                        WHEN excluded.resolution_state LIKE 'resolved%' THEN excluded.resolution_state
                        ELSE trade_outcomes.resolution_state
                    END
                """,
                (
                    normalized_market_id,
                    normalized_predicted_outcome,
                    normalized_avg_price,
                    normalized_avg_price,
                    None,
                    amount_usdc,
                    float(normalized_contracts) if normalized_contracts > 0 else None,
                    normalized_winning_outcome,
                    won,
                    realized_pnl,
                    timestamp if normalized_winning_outcome else None,
                    datetime.now(timezone.utc).isoformat(),
                    resolution_state,
                ),
            )
        if online_calibration_enabled:
            self.record_online_confidence_calibration_from_trade(
                normalized_market_id,
                alpha=online_calibration_alpha,
                max_samples_per_bucket=online_calibration_max_samples_per_bucket,
            )

    def _upsert_trade_outcome_entry(
        self,
        market_id: str,
        predicted_outcome: str,
        entry_price: float | None,
        implied_prob: float | None,
        confidence: float | None,
        amount_usdc: float | None,
        shares: float | None,
        timestamp: str,
    ) -> None:
        row = self._conn.execute(
            """
            SELECT entry_price, implied_prob, confidence, amount_usdc, shares
            FROM trade_outcomes
            WHERE market_id = ?
            """,
            (market_id,),
        ).fetchone()
        if row:
            total_amount = (row["amount_usdc"] or 0.0) + (amount_usdc or 0.0)
            total_shares = (row["shares"] or 0.0) + (shares or 0.0)
            weighted_price = _weighted_average(
                current=row["entry_price"],
                current_weight=row["shares"],
                new=entry_price,
                new_weight=shares,
            )
            weighted_implied = _weighted_average(
                current=row["implied_prob"],
                current_weight=row["shares"],
                new=implied_prob,
                new_weight=shares,
            )
            self._conn.execute(
                """
                UPDATE trade_outcomes
                SET predicted_outcome = ?, entry_price = ?, implied_prob = ?, confidence = ?, amount_usdc = ?,
                    shares = ?, last_updated = ?, resolution_state = COALESCE(resolution_state, 'unresolved')
                WHERE market_id = ?
                """,
                (
                    predicted_outcome,
                    weighted_price,
                    weighted_implied,
                    confidence,
                    total_amount,
                    total_shares,
                    timestamp,
                    market_id,
                ),
            )
            return
        self._conn.execute(
            """
            INSERT INTO trade_outcomes (
                market_id, predicted_outcome, entry_price, implied_prob, confidence, amount_usdc, shares,
                resolved_winning_outcome, won, pnl_estimate, resolved_at, last_updated, resolution_state
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                market_id,
                predicted_outcome,
                entry_price,
                implied_prob,
                confidence,
                amount_usdc,
                shares,
                None,
                None,
                None,
                None,
                timestamp,
                "unresolved",
            ),
        )

    def _upsert_trade_outcome_event(
        self,
        market_id: str,
        order_id: str,
        predicted_outcome: str,
        entry_price: float | None,
        implied_prob: float | None,
        confidence: float | None,
        amount_usdc: float | None,
        shares: float | None,
        timestamp: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO trade_outcome_events (
                market_id, order_id, predicted_outcome, entry_price, implied_prob, confidence,
                amount_usdc, shares, timestamp, resolved_winning_outcome, won, pnl_estimate, resolved_at, resolution_state
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                market_id,
                order_id,
                predicted_outcome,
                entry_price,
                implied_prob,
                confidence,
                amount_usdc,
                shares,
                timestamp,
                None,
                None,
                None,
                None,
                "unresolved",
            ),
        )

    def get_markets_needing_reanalysis(self, hours_since: int) -> list[str]:
        hours_since = max(hours_since, 0)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_since)
        cutoff_iso = cutoff.isoformat()

        rows = self._conn.execute(
            """
            SELECT market_id, MAX(timestamp) AS last_analysis
            FROM analyses
            GROUP BY market_id
            HAVING last_analysis <= ?
            ORDER BY last_analysis ASC
            """,
            (cutoff_iso,),
        ).fetchall()

        return [row["market_id"] for row in rows]

    def export_to_json(self, path: str) -> None:
        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        markets = _rows_to_dicts(
            self._conn.execute("SELECT * FROM markets").fetchall()
        )
        analyses = _rows_to_dicts(
            self._conn.execute("SELECT * FROM analyses").fetchall()
        )
        positions = _rows_to_dicts(
            self._conn.execute("SELECT * FROM positions").fetchall()
        )
        trade_log = _rows_to_dicts(
            self._conn.execute("SELECT * FROM trade_log").fetchall()
        )
        trade_outcomes = _rows_to_dicts(
            self._conn.execute("SELECT * FROM trade_outcomes").fetchall()
        )
        trade_outcome_events = _rows_to_dicts(
            self._conn.execute("SELECT * FROM trade_outcome_events").fetchall()
        )
        bayesian_state = _rows_to_dicts(
            self._conn.execute("SELECT * FROM bayesian_state").fetchall()
        )
        cycle_receipts = _rows_to_dicts(
            self._conn.execute("SELECT * FROM cycle_receipts ORDER BY id ASC").fetchall()
        )
        decision_receipts = _rows_to_dicts(
            self._conn.execute("SELECT * FROM decision_receipts ORDER BY id ASC").fetchall()
        )

        for row in positions:
            row["order_ids"] = _parse_order_ids(row.get("order_ids"))

        for row in analyses:
            if "is_refined" in row:
                row["is_refined"] = bool(row["is_refined"])

        payload = {
            "markets": markets,
            "analyses": analyses,
            "positions": positions,
            "trade_log": trade_log,
            "trade_outcomes": trade_outcomes,
            "trade_outcome_events": trade_outcome_events,
            "bayesian_state": bayesian_state,
            "cycle_receipts": cycle_receipts,
            "decision_receipts": decision_receipts,
        }

        export_path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        logger.debug("Exported market state to %s", export_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()

    def _get_latest_confidence(self, market_id: str) -> float | None:
        row = self._conn.execute(
            """
            SELECT confidence
            FROM analyses
            WHERE market_id = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT 1
            """,
            (market_id,),
        ).fetchone()
        if not row:
            return None
        return row["confidence"]

    def get_last_reasoning_hash(self, market_id: str) -> str | None:
        row = self._conn.execute(
            """
            SELECT reasoning_hash
            FROM analyses
            WHERE market_id = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT 1
            """,
            (market_id,),
        ).fetchone()
        if not row:
            return None
        value = row["reasoning_hash"]
        return str(value) if value else None

    def get_outcome_flip_count(self, market_id: str) -> int:
        rows = self._conn.execute(
            """
            SELECT outcome
            FROM analyses
            WHERE market_id = ?
              AND outcome IS NOT NULL
            ORDER BY timestamp ASC, id ASC
            """,
            (market_id,),
        ).fetchall()
        flip_count = 0
        previous_outcome: str | None = None
        for row in rows:
            current_outcome = str(row["outcome"] or "").strip().upper()
            if not current_outcome:
                continue
            if previous_outcome is not None and current_outcome != previous_outcome:
                flip_count += 1
            previous_outcome = current_outcome
        return flip_count

    def _market_exists(self, market_id: str) -> bool:
        return any(
            (
                self._has_row(
                    "SELECT 1 FROM markets WHERE id = ? LIMIT 1", (market_id,)
                ),
                self._has_row(
                    "SELECT 1 FROM positions WHERE market_id = ? LIMIT 1",
                    (market_id,),
                ),
                self._has_row(
                    "SELECT 1 FROM trade_log WHERE market_id = ? LIMIT 1",
                    (market_id,),
                ),
            )
        )

    def _has_row(self, query: str, params: tuple[Any, ...]) -> bool:
        return self._conn.execute(query, params).fetchone() is not None

    def _run_migrations(self) -> None:
        self._ensure_column("analyses", "refinement_reason", "TEXT")
        self._ensure_column("analyses", "reasoning_hash", "TEXT")
        self._ensure_column("markets", "last_terminal_outcome", "TEXT")
        self._ensure_column("markets", "non_actionable_streak", "INTEGER DEFAULT 0")
        self._ensure_column("markets", "fill_failure_count", "INTEGER DEFAULT 0")
        self._ensure_column("markets", "next_eligible_cycle", "INTEGER DEFAULT 0")
        self._ensure_column("decision_receipts", "score_json", "TEXT")
        self._ensure_column(
            "research_queue_entries",
            "times_seen",
            "INTEGER DEFAULT 1",
        )
        self._ensure_column(
            "trade_outcomes",
            "resolution_state",
            "TEXT DEFAULT 'unresolved'",
        )

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        columns = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {row["name"] for row in columns}
        if column in existing:
            return
        self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def _backfill_resolution_state(self) -> None:
        unresolved_tokens = {"", "-1", "18446744073709551615"}
        with self._conn:
            self._conn.execute(
                """
                UPDATE trade_outcomes
                SET resolution_state = 'unresolved'
                WHERE resolution_state IS NULL
                """
            )
            self._conn.execute(
                """
                UPDATE trade_outcomes
                SET resolution_state = 'unresolved', won = NULL, pnl_estimate = NULL,
                    resolved_winning_outcome = NULL, resolved_at = NULL
                WHERE COALESCE(resolved_winning_outcome, '') IN (?, ?, ?)
                """,
                tuple(unresolved_tokens),
            )
            self._conn.execute(
                """
                UPDATE trade_outcomes
                SET resolution_state = 'resolved_valid'
                WHERE resolved_winning_outcome IS NOT NULL
                  AND resolved_winning_outcome NOT IN (?, ?, ?)
                  AND won IS NOT NULL
                """,
                tuple(unresolved_tokens),
            )


    def backfill_outcomes_from_settlements(self) -> int:
        """Forward-fill won/pnl_estimate into trade_outcomes from exchange_settlements.

        Returns the number of rows updated.
        """
        with self._conn:
            cursor = self._conn.execute(
                """
                UPDATE trade_outcomes
                SET
                    won = (
                        SELECT es.won
                        FROM exchange_settlements es
                        WHERE es.market_id = trade_outcomes.market_id
                          AND es.won IS NOT NULL
                        LIMIT 1
                    ),
                    pnl_estimate = COALESCE(trade_outcomes.pnl_estimate, (
                        SELECT es.pnl_realized
                        FROM exchange_settlements es
                        WHERE es.market_id = trade_outcomes.market_id
                          AND es.pnl_realized IS NOT NULL
                        LIMIT 1
                    )),
                    resolution_state = 'resolved_valid',
                    resolved_at = COALESCE(trade_outcomes.resolved_at, (
                        SELECT es.settled_at
                        FROM exchange_settlements es
                        WHERE es.market_id = trade_outcomes.market_id
                        LIMIT 1
                    ))
                WHERE trade_outcomes.won IS NULL
                  AND EXISTS (
                      SELECT 1
                      FROM exchange_settlements es
                      WHERE es.market_id = trade_outcomes.market_id
                        AND es.won IS NOT NULL
                  )
                """
            )
        return cursor.rowcount


def _parse_order_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [str(item) for item in data if item]
    return []


def _build_reasoning_hash(reasoning: str | None, outcome: str | None, confidence: float | None) -> str:
    reasoning_text = _RE_VALIDATED_PREFIX.sub("", (reasoning or "").strip())[:200]
    outcome_text = (outcome or "").strip().lower()
    rounded_confidence = round(float(confidence or 0.0), 2)
    payload = f"{outcome_text}|{rounded_confidence:.2f}|{reasoning_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _extract_order_id(order: OrderResponse) -> str | None:
    if order.id:
        return str(order.id)
    raw = order.raw or {}
    # Check top-level fields
    for key in ("id", "order_id", "orderId", "orderRef", "clientOrderId"):
        value = raw.get(key)
        if value:
            return str(value)
    # Check nested order field
    nested = raw.get("order")
    if isinstance(nested, dict):
        for key in ("id", "order_id", "orderId", "orderRef", "clientOrderId"):
            value = nested.get(key)
            if value:
                return str(value)
    # Check meta field for clientOrderId
    meta = raw.get("meta")
    if isinstance(meta, dict):
        client_order_id = meta.get("clientOrderId")
        if client_order_id:
            return str(client_order_id)
    return None


def _extract_order_outcome(order: OrderResponse) -> str | None:
    raw = order.raw or {}
    for key in ("outcome", "market_outcome", "option"):
        value = raw.get(key)
        if value:
            return str(value)
    nested = raw.get("order")
    if isinstance(nested, dict):
        for key in ("outcome", "market_outcome", "option"):
            value = nested.get(key)
            if value:
                return str(value)
    return None


def _update_avg_confidence(
    existing_avg: float,
    trade_count: int,
    latest_confidence: float | None,
) -> float:
    if trade_count <= 0:
        return 0.0
    if latest_confidence is None:
        if trade_count == 1:
            return 0.0
        return existing_avg
    if trade_count == 1:
        return float(latest_confidence)
    return ((existing_avg * (trade_count - 1)) + latest_confidence) / trade_count


def _weighted_average(
    current: float | None,
    current_weight: float | None,
    new: float | None,
    new_weight: float | None,
) -> float | None:
    if new is None and current is None:
        return None
    if current is None or (current_weight or 0) <= 0:
        return new
    if new is None or (new_weight or 0) <= 0:
        return current
    total_weight = (current_weight or 0) + (new_weight or 0)
    if total_weight <= 0:
        return current
    return ((current * current_weight) + (new * new_weight)) / total_weight


def _estimate_pnl(
    entry_price: float | None,
    amount_usdc: float | None,
    shares: float | None,
    won: int | None,
) -> float | None:
    if won is None or entry_price is None:
        return None
    if shares is None or shares <= 0:
        if amount_usdc is None or amount_usdc <= 0:
            return None
        shares = amount_usdc / entry_price if entry_price > 0 else None
    if shares is None:
        return None
    if won:
        return shares * (1 - entry_price)
    return -shares * entry_price
