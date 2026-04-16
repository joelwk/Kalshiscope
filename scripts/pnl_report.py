from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import load_settings
from kalshi_client import KalshiClient
from market_state import MarketStateManager

DEFAULT_DB_PATH = "data/market_state.db"
DEFAULT_SETTLEMENT_PAGE_SIZE = 200
DEFAULT_MAX_SYNC_PAGES = 0
SUMMARY_SEPARATOR_WIDTH = 72
CATEGORY_HEADER = (
    f"{'Category':<14} {'Trades':>7} {'Wins':>6} {'Losses':>7} "
    f"{'WinRate':>8} {'TotalPnL':>12} {'AvgPnL':>12}"
)
MONTHLY_HEADER = f"{'Month':<10} {'Trades':>7} {'PnL':>12} {'Cumulative':>12}"
POSITIONS_HEADER = f"{'Ticker':<28} {'Side':<4} {'Contracts':>10} {'Exposure':>12}"


@dataclass(frozen=True)
class SyncStats:
    pages_fetched: int
    settlements_seen: int
    settlements_parsed: int
    cursor_stopped: bool


@dataclass(frozen=True)
class PositionRow:
    market_id: str
    side: str
    contracts: int
    exposure_usd: float


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _coerce_dollars(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    if isinstance(value, (list, tuple)):
        if not value:
            return 0.0
        if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
            whole = float(value[0])
            fractional = float(value[1]) / 100.0
            return whole + fractional if whole >= 0 else whole - fractional
        first = value[0]
        return _coerce_dollars(first)
    return 0.0


def _coerce_cents_to_dollars(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value) / 100.0
    if isinstance(value, str):
        try:
            return float(value) / 100.0
        except ValueError:
            return 0.0
    if isinstance(value, (list, tuple)):
        if not value:
            return 0.0
        first = value[0]
        if isinstance(first, (int, float)):
            return float(first) / 100.0
    return 0.0


def _iter_exchange_settlement_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in ("settlements", "market_settlements", "data"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _normalize_outcome(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"yes", "true", "1"}:
        return "YES"
    if text in {"no", "false", "0"}:
        return "NO"
    return None


def _parse_exchange_settlement_row(row: dict[str, Any]) -> dict[str, Any] | None:
    settlement_id = str(
        row.get("settlement_id")
        or row.get("id")
        or row.get("trade_id")
        or row.get("market_ticker")
        or row.get("ticker")
        or ""
    ).strip()
    market_id = str(
        row.get("market_ticker")
        or row.get("ticker")
        or row.get("market_id")
        or ""
    ).strip()
    if not settlement_id or not market_id:
        return None

    winning_outcome = _normalize_outcome(
        row.get("market_result")
        or row.get("winning_outcome")
        or row.get("settlement_result")
    )

    yes_contracts = int(
        _coerce_float(
            row.get("yes_count")
            or row.get("yes_count_fp")
            or row.get("yes_contracts_owned")
            or 0
        ) or 0.0
    )
    no_contracts = int(
        _coerce_float(
            row.get("no_count")
            or row.get("no_count_fp")
            or row.get("no_contracts_owned")
            or 0
        ) or 0.0
    )
    predicted_outcome: str | None = None
    contracts = 0
    avg_price: float | None = None
    cost_dollars: float = 0.0
    if yes_contracts > 0:
        predicted_outcome = "YES"
        contracts = yes_contracts
        cost_dollars_raw = _coerce_float(row.get("yes_total_cost_dollars"))
        if cost_dollars_raw is not None:
            cost_dollars = cost_dollars_raw
            avg_price = cost_dollars / yes_contracts if yes_contracts > 0 else None
        else:
            avg_price = _coerce_float(
                row.get("yes_total_cost") or row.get("yes_contracts_average_price")
            )
            if avg_price is None:
                avg_price = _coerce_float(row.get("yes_contracts_average_price_in_cents"))
                if avg_price is not None and avg_price > 1.0:
                    avg_price /= 100.0
            if avg_price is not None:
                cost_dollars = avg_price * yes_contracts
    elif no_contracts > 0:
        predicted_outcome = "NO"
        contracts = no_contracts
        cost_dollars_raw = _coerce_float(row.get("no_total_cost_dollars"))
        if cost_dollars_raw is not None:
            cost_dollars = cost_dollars_raw
            avg_price = cost_dollars / no_contracts if no_contracts > 0 else None
        else:
            avg_price = _coerce_float(
                row.get("no_total_cost") or row.get("no_contracts_average_price")
            )
            if avg_price is None:
                avg_price = _coerce_float(row.get("no_contracts_average_price_in_cents"))
                if avg_price is not None and avg_price > 1.0:
                    avg_price /= 100.0
            if avg_price is not None:
                cost_dollars = avg_price * no_contracts

    pnl_realized = _coerce_float(
        row.get("profit")
        or row.get("profit_in_dollars")
        or row.get("pnl")
        or row.get("realized_pnl")
    )
    if pnl_realized is None:
        revenue_raw = _coerce_float(row.get("revenue"))
        fee_raw = _coerce_float(row.get("fee_cost"))
        revenue_dollars = (revenue_raw / 100.0) if revenue_raw is not None else 0.0
        fee_dollars = fee_raw if fee_raw is not None else 0.0
        pnl_realized = revenue_dollars - cost_dollars - fee_dollars

    settled_at = (
        _coerce_datetime(row.get("settled_time"))
        or _coerce_datetime(row.get("created_time"))
        or _coerce_datetime(row.get("created_at"))
    )

    return {
        "settlement_id": settlement_id,
        "market_id": market_id,
        "winning_outcome": winning_outcome,
        "predicted_outcome": predicted_outcome,
        "pnl_realized": float(pnl_realized),
        "contracts": contracts if contracts > 0 else None,
        "avg_price": avg_price,
        "settled_at": settled_at,
        "raw": row,
    }


def _extract_cursor(payload: dict[str, Any]) -> str | None:
    for key in ("cursor", "next_cursor", "nextCursor"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _sync_settlements_from_exchange(
    *,
    state_manager: MarketStateManager,
    kalshi_client: KalshiClient,
    page_size: int,
    max_pages: int,
) -> SyncStats:
    pages_fetched = 0
    settlements_seen = 0
    settlements_parsed = 0
    cursor: str | None = None
    seen_cursors: set[str] = set()
    cursor_stopped = False

    while True:
        payload = kalshi_client.get_settlements(limit=page_size, cursor=cursor)
        rows = _iter_exchange_settlement_rows(payload)
        settlements_seen += len(rows)
        for row in rows:
            parsed = _parse_exchange_settlement_row(row)
            if parsed is None:
                continue
            state_manager.record_exchange_settlement(**parsed)
            settlements_parsed += 1
        pages_fetched += 1

        if max_pages > 0 and pages_fetched >= max_pages:
            break

        next_cursor = _extract_cursor(payload)
        if not next_cursor:
            break
        if next_cursor in seen_cursors:
            cursor_stopped = True
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    return SyncStats(
        pages_fetched=pages_fetched,
        settlements_seen=settlements_seen,
        settlements_parsed=settlements_parsed,
        cursor_stopped=cursor_stopped,
    )


def _iter_exchange_position_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in ("market_positions", "positions", "portfolio_positions", "data"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _position_exposure_dollars(row: dict[str, Any], side: str, contracts: int) -> float:
    side_prefix = "yes" if side == "YES" else "no"
    candidate_keys = (
        ("market_exposure_dollars", True),
        (f"{side_prefix}_total_cost_dollars", True),
        ("total_cost_dollars", True),
        (f"{side_prefix}_total_cost", False),
        ("market_exposure", False),
        ("total_cost", False),
    )
    for key, is_dollars in candidate_keys:
        value = row.get(key)
        amount = _coerce_dollars(value) if is_dollars else _coerce_cents_to_dollars(value)
        if amount > 0:
            return abs(amount)
    return float(contracts)


def _parse_open_positions(payload: dict[str, Any] | None) -> list[PositionRow]:
    positions: list[PositionRow] = []
    for row in _iter_exchange_position_rows(payload):
        market_id = str(
            row.get("ticker")
            or row.get("market_ticker")
            or row.get("market_id")
            or ""
        ).strip()
        if not market_id:
            continue
        contracts_raw = row.get("position") or row.get("position_fp")
        if contracts_raw is None:
            yes_count = float(row.get("yes_count") or row.get("yes_count_fp") or 0.0)
            no_count = float(row.get("no_count") or row.get("no_count_fp") or 0.0)
            contracts_raw = yes_count - no_count
        try:
            signed_contracts = int(float(contracts_raw or 0.0))
        except (TypeError, ValueError):
            continue
        if signed_contracts == 0:
            continue
        side = "YES" if signed_contracts > 0 else "NO"
        contracts = abs(signed_contracts)
        exposure_usd = _position_exposure_dollars(row, side, contracts)
        positions.append(
            PositionRow(
                market_id=market_id,
                side=side,
                contracts=contracts,
                exposure_usd=exposure_usd,
            )
        )
    positions.sort(key=lambda item: item.exposure_usd, reverse=True)
    return positions


def _read_settlement_report_rows(db_path: str) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            """
            SELECT
                s.settlement_id,
                s.market_id,
                s.won,
                s.pnl_realized,
                s.contracts,
                s.settled_at,
                COALESCE(m.question, '') AS question,
                COALESCE(m.category, '') AS category
            FROM exchange_settlements s
            LEFT JOIN markets m ON m.id = s.market_id
            ORDER BY s.settled_at ASC, s.settlement_id ASC
            """
        ).fetchall()
    finally:
        conn.close()


def _format_currency(value: float, *, signed: bool = False) -> str:
    if signed:
        return f"{value:+,.2f}"
    return f"{value:,.2f}"


def _print_section(title: str) -> None:
    print()
    print(title)
    print("-" * max(len(title), SUMMARY_SEPARATOR_WIDTH // 2))


def _print_sync_summary(stats: SyncStats) -> None:
    _print_section("Sync Summary")
    print(f"Pages fetched:        {stats.pages_fetched}")
    print(f"Settlements fetched:  {stats.settlements_seen}")
    print(f"Settlements imported: {stats.settlements_parsed}")
    print(f"Cursor loop stopped:  {stats.cursor_stopped}")


def _print_account_summary(
    balance_available: float | None,
    balance_position: float | None,
    balance_total: float | None,
    open_positions_count: int | None,
) -> None:
    _print_section("Account Summary")
    if balance_available is None or balance_position is None or balance_total is None:
        print("Live API data unavailable (`--no-api` mode or API call failed).")
        return
    print(f"Available balance:    ${_format_currency(balance_available)}")
    print(f"Position value:       ${_format_currency(balance_position)}")
    print(f"Total portfolio:      ${_format_currency(balance_total)}")
    if open_positions_count is not None:
        print(f"Open positions:       {open_positions_count}")


def _print_total_realized_pnl(rows: list[sqlite3.Row], state_manager: MarketStateManager) -> None:
    _print_section("Total Realized PnL")
    if not rows:
        print("No exchange settlements found in the local database.")
        return

    total_pnl = state_manager.get_exchange_realized_pnl_total()
    total_trades = len(rows)
    wins = sum(1 for row in rows if row["won"] == 1)
    losses = sum(1 for row in rows if row["won"] == 0)
    win_rate_denominator = wins + losses
    win_rate = (wins / win_rate_denominator) if win_rate_denominator > 0 else 0.0
    contracts_total = sum(int(row["contracts"] or 0) for row in rows)

    print(f"Settled trades:       {total_trades}")
    print(f"Wins / losses:        {wins} / {losses}")
    print(f"Win rate:             {win_rate:.2%}")
    print(f"Total contracts:      {contracts_total}")
    print(f"Realized PnL:         ${_format_currency(total_pnl, signed=True)}")


def _print_category_breakdown(rows: list[sqlite3.Row]) -> None:
    _print_section("PnL by Category")
    if not rows:
        print("No category breakdown available (no settlements).")
        return

    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        family = MarketStateManager._infer_family_from_state_row(
            market_id=str(row["market_id"] or ""),
            question=str(row["question"] or ""),
            category=str(row["category"] or ""),
        )
        bucket = grouped.setdefault(
            family,
            {
                "trades": 0.0,
                "wins": 0.0,
                "losses": 0.0,
                "pnl_total": 0.0,
            },
        )
        bucket["trades"] += 1.0
        if row["won"] == 1:
            bucket["wins"] += 1.0
        elif row["won"] == 0:
            bucket["losses"] += 1.0
        bucket["pnl_total"] += float(row["pnl_realized"] or 0.0)

    print(CATEGORY_HEADER)
    print("-" * len(CATEGORY_HEADER))
    sorted_items = sorted(grouped.items(), key=lambda item: item[1]["pnl_total"], reverse=True)
    for family, stats in sorted_items:
        trades = int(stats["trades"])
        wins = int(stats["wins"])
        losses = int(stats["losses"])
        win_rate_denominator = wins + losses
        win_rate = (wins / win_rate_denominator) if win_rate_denominator > 0 else 0.0
        pnl_total = float(stats["pnl_total"])
        avg_pnl = pnl_total / trades if trades > 0 else 0.0
        print(
            f"{family:<14} {trades:>7} {wins:>6} {losses:>7} "
            f"{win_rate:>7.1%} {pnl_total:>12,.2f} {avg_pnl:>12,.2f}"
        )


def _print_monthly_breakdown(rows: list[sqlite3.Row]) -> None:
    _print_section("PnL by Month")
    if not rows:
        print("No monthly breakdown available (no settlements).")
        return

    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        settled_at = _coerce_datetime(row["settled_at"])
        month = settled_at.strftime("%Y-%m") if settled_at else "unknown"
        bucket = grouped.setdefault(month, {"trades": 0.0, "pnl": 0.0})
        bucket["trades"] += 1.0
        bucket["pnl"] += float(row["pnl_realized"] or 0.0)

    print(MONTHLY_HEADER)
    print("-" * len(MONTHLY_HEADER))
    cumulative = 0.0
    for month in sorted(grouped):
        trades = int(grouped[month]["trades"])
        pnl = float(grouped[month]["pnl"])
        cumulative += pnl
        print(f"{month:<10} {trades:>7} {pnl:>12,.2f} {cumulative:>12,.2f}")


def _print_open_positions(positions: list[PositionRow], *, live_api_available: bool) -> None:
    _print_section("Open Positions (Unrealized Exposure)")
    if not live_api_available:
        print("Live API data unavailable (`--no-api` mode or API call failed).")
        return
    if not positions:
        print("No open positions reported by Kalshi.")
        return

    print(POSITIONS_HEADER)
    print("-" * len(POSITIONS_HEADER))
    total_exposure = 0.0
    for position in positions:
        total_exposure += position.exposure_usd
        print(
            f"{position.market_id:<28} {position.side:<4} "
            f"{position.contracts:>10} {position.exposure_usd:>12,.2f}"
        )
    print("-" * len(POSITIONS_HEADER))
    print(f"{'Total exposure':<44} {total_exposure:>12,.2f}")


def _build_client() -> KalshiClient:
    settings = load_settings()
    return KalshiClient(
        base_url=settings.KALSHI_API_BASE_URL,
        api_key_id=settings.KALSHI_API_KEY_ID,
        private_key_path=settings.KALSHI_PRIVATE_KEY_PATH,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a total PnL report with category and monthly breakdowns."
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help="Path to the market-state sqlite database.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Sync settlements from Kalshi API before generating the report.",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Skip all live API calls and generate a DB-only report.",
    )
    parser.add_argument(
        "--settlement-page-size",
        type=int,
        default=DEFAULT_SETTLEMENT_PAGE_SIZE,
        help="Settlement page size when syncing (default: 200).",
    )
    parser.add_argument(
        "--max-sync-pages",
        type=int,
        default=DEFAULT_MAX_SYNC_PAGES,
        help="Cap synced settlement pages; 0 means no cap.",
    )
    args = parser.parse_args()

    state_manager = MarketStateManager(args.db)
    sync_stats: SyncStats | None = None
    live_api_available = False
    balance_available: float | None = None
    balance_position: float | None = None
    balance_total: float | None = None
    open_positions: list[PositionRow] = []

    try:
        client: KalshiClient | None = None
        if not args.no_api:
            try:
                client = _build_client()
                live_api_available = True
            except Exception as exc:
                print(f"Warning: failed to initialize Kalshi API client: {exc}")
                live_api_available = False

        if args.sync:
            if client is None:
                print("Warning: `--sync` requested, but API client is unavailable.")
            else:
                sync_stats = _sync_settlements_from_exchange(
                    state_manager=state_manager,
                    kalshi_client=client,
                    page_size=max(1, args.settlement_page_size),
                    max_pages=max(0, args.max_sync_pages),
                )

        if client is not None:
            try:
                portfolio_balance = client.get_portfolio_balance()
                balance_available = portfolio_balance.available_balance
                balance_position = portfolio_balance.position_value
                balance_total = portfolio_balance.total_portfolio_value
            except Exception as exc:
                live_api_available = False
                print(f"Warning: failed to fetch portfolio balance: {exc}")

            try:
                positions_payload = client.get_positions()
                open_positions = _parse_open_positions(positions_payload)
            except Exception as exc:
                print(f"Warning: failed to fetch open positions: {exc}")

        report_rows = _read_settlement_report_rows(args.db)

        print("PnL Report")
        print("=" * SUMMARY_SEPARATOR_WIDTH)
        print(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        print(f"Database:  {Path(args.db).resolve()}")
        print(f"Mode:      {'DB-only' if args.no_api else 'DB + Live API'}")
        print(f"Rows:      {len(report_rows)} settlements")

        if sync_stats is not None:
            _print_sync_summary(sync_stats)

        _print_account_summary(
            balance_available=balance_available,
            balance_position=balance_position,
            balance_total=balance_total,
            open_positions_count=len(open_positions) if live_api_available else None,
        )
        _print_total_realized_pnl(report_rows, state_manager)
        _print_category_breakdown(report_rows)
        _print_monthly_breakdown(report_rows)
        _print_open_positions(open_positions, live_api_available=live_api_available)
    finally:
        state_manager.close()


if __name__ == "__main__":
    main()
