from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from logging_config import get_logger
from models import MarketState, OrderResponse, Position, TradeDecision

logger = get_logger(__name__)

_CONFIDENCE_TREND_WINDOW = 5


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
                    category TEXT
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
            self._run_migrations()
            self._backfill_resolution_state()

    def get_market_state(self, market_id: str) -> MarketState | None:
        latest_row = self._conn.execute(
            """
            SELECT confidence, timestamp
            FROM analyses
            WHERE market_id = ?
            ORDER BY timestamp DESC
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
            return MarketState(market_id=market_id)

        trend_rows = self._conn.execute(
            """
            SELECT confidence
            FROM analyses
            WHERE market_id = ?
            ORDER BY timestamp DESC
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

    def record_analysis(
        self,
        market_id: str,
        decision: TradeDecision,
        is_refined: bool,
        refinement_reason: str | None = None,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO analyses (
                    market_id, confidence, outcome, reasoning, timestamp, is_refined, refinement_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    market_id,
                    decision.confidence,
                    decision.outcome,
                    decision.reasoning,
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
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (market_id,),
        ).fetchone()
        if not row:
            return None
        return row["confidence"]

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
