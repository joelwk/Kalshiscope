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
                    fill_failure_count INTEGER DEFAULT 0
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
            self._run_migrations()
            self._backfill_resolution_state()

    def get_market_state(self, market_id: str) -> MarketState | None:
        market_row = self._conn.execute(
            """
            SELECT last_terminal_outcome, non_actionable_streak
                , fill_failure_count
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
                    json.dumps(order, sort_keys=True, default=str) if order is not None else None,
                    json.dumps(execution_audit, sort_keys=True, default=str)
                    if execution_audit is not None
                    else None,
                    json.dumps(normalized_score_breakdown, sort_keys=True, default=str)
                    if normalized_score_breakdown is not None
                    else None,
                ),
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

    def get_family_action_snapshot(
        self,
        *,
        lookback: int = 400,
    ) -> dict[str, dict[str, float | int]]:
        """Return decision action rates grouped by inferred family."""
        window = max(1, int(lookback))
        rows = self._conn.execute(
            """
            SELECT
                LOWER(COALESCE(json_extract(audit_json, '$.market_family'), 'generic')) AS market_family,
                LOWER(COALESCE(final_action, '')) AS final_action,
                LOWER(COALESCE(final_reason, '')) AS final_reason
            FROM decision_receipts
            ORDER BY id DESC
            LIMIT ?
            """,
            (window,),
        ).fetchall()
        grouped: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {"sample_size": 0, "order_attempts": 0}
        )
        for row in rows:
            family = str(row["market_family"] or "generic").strip().lower() or "generic"
            final_action = str(row["final_action"] or "").strip().lower()
            final_reason = str(row["final_reason"] or "").strip().lower()
            bucket = grouped[family]
            bucket["sample_size"] = int(bucket["sample_size"]) + 1
            if final_action == "order_attempt" and final_reason != "dry_run":
                bucket["order_attempts"] = int(bucket["order_attempts"]) + 1
        snapshot: dict[str, dict[str, float | int]] = {}
        for family, values in grouped.items():
            sample_size = int(values["sample_size"])
            order_attempts = int(values["order_attempts"])
            snapshot[family] = {
                "sample_size": sample_size,
                "order_attempts": order_attempts,
                "action_rate": (order_attempts / sample_size) if sample_size > 0 else 0.0,
            }
        return snapshot

    @staticmethod
    def _infer_family_from_state_row(*, market_id: str, question: str, category: str) -> str:
        text = f"{market_id} {question} {category}".upper()
        if any(token in text for token in ("KXKBOGAME", "KXNPBGAME", "SPORT", "MATCH", "WINNER")):
            return "sports"
        if any(token in text for token in ("KXBTC", "KXETH", "KXSOL", "KXXRP", "KXDOGE", "KXBNB", "CRYPTO")):
            return "crypto"
        if any(token in text for token in ("MENTION", "SPEECH", "TRANSCRIPT", "LASTWORD", "TRUTHSOCIAL")):
            return "speech"
        if any(token in text for token in ("STREAM", "SPOTIFY", "ARTIST", "SONG", "MUSIC", "LUMINATE")):
            return "music"
        if any(
            token in text
            for token in ("KXHIGH", "KXLOW", "KXTEMP", "KXRAIN", "WEATHER", "TEMPERATURE")
        ):
            return "weather"
        if any(
            token in text
            for token in ("KXGOLD", "KXSILVER", "KXNATGAS", "KXBRENT", "KXHOIL", "KXSUGAR", "KXCOFFEE", "KXCORN", "KXSOYBEAN")
        ):
            return "commodities"
        if any(token in text for token in ("KXINX", "KXNASDAQ", "S&P", "NASDAQ", "DOW")):
            return "index"
        return "generic"

    def record_resolution(
        self,
        market_id: str,
        winning_outcome: str,
        resolved_at: datetime | None,
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
        self._ensure_column("decision_receipts", "score_json", "TEXT")
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
