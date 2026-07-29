"""Stream selected SQLite audit tables to JSON or NDJSON.

This is the full-history counterpart to the bounded schema-v2 runtime snapshot.
It opens SQLite in read-only mode and never changes or vacuums the database.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
from typing import IO, Iterable


TIME_COLUMNS = (
    "timestamp",
    "created_at",
    "updated_at",
    "last_updated",
    "last_seen_at",
    "last_synced_at",
    "last_attempt_at",
    "resolved_at",
    "settled_at",
    "filled_at",
    "queued_at",
    "recorded_at",
    "exported_at",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream complete Prediscope audit history from SQLite.",
    )
    parser.add_argument("--db", default="data/market_state.db")
    parser.add_argument(
        "--table",
        action="append",
        dest="tables",
        help="Table to export; repeat for multiple tables (default: all).",
    )
    parser.add_argument("--since", help="Inclusive ISO timestamp boundary.")
    parser.add_argument("--until", help="Exclusive ISO timestamp boundary.")
    parser.add_argument(
        "--format",
        choices=("json", "ndjson"),
        default="ndjson",
    )
    parser.add_argument("--output", help="Output path (default: stdout).")
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args()


def _open_read_only(db_path: str) -> sqlite3.Connection:
    resolved = Path(db_path).expanduser().resolve()
    connection = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _available_tables(connection: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
    ]


def _time_column(connection: sqlite3.Connection, table: str) -> str | None:
    columns = {
        str(row[1])
        for row in connection.execute(f'SELECT * FROM pragma_table_info("{table}")')
    }
    return next((column for column in TIME_COLUMNS if column in columns), None)


def _iter_rows(
    connection: sqlite3.Connection,
    table: str,
    *,
    since: str | None,
    until: str | None,
    batch_size: int,
) -> Iterable[dict[str, object]]:
    time_column = _time_column(connection, table)
    if (since or until) and time_column is None:
        raise ValueError(
            f"table {table!r} has no supported time column for boundaries"
        )
    clauses: list[str] = []
    params: list[str] = []
    if since:
        clauses.append(f'"{time_column}" >= ?')
        params.append(since)
    if until:
        clauses.append(f'"{time_column}" < ?')
        params.append(until)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    cursor = connection.execute(f'SELECT * FROM "{table}"{where}', params)
    while batch := cursor.fetchmany(max(1, batch_size)):
        yield from (dict(row) for row in batch)


def _write_ndjson(
    stream: IO[str],
    connection: sqlite3.Connection,
    tables: list[str],
    args: argparse.Namespace,
) -> None:
    for table in tables:
        for row in _iter_rows(
            connection,
            table,
            since=args.since,
            until=args.until,
            batch_size=args.batch_size,
        ):
            stream.write(
                json.dumps({"table": table, "row": row}, default=str) + "\n"
            )


def _write_json(
    stream: IO[str],
    connection: sqlite3.Connection,
    tables: list[str],
    args: argparse.Namespace,
) -> None:
    stream.write(
        '{"schema_version":1,"exported_at":'
        + json.dumps(datetime.now(timezone.utc).isoformat())
        + ',"tables":{'
    )
    for table_index, table in enumerate(tables):
        if table_index:
            stream.write(",")
        stream.write(json.dumps(table) + ":[")
        for row_index, row in enumerate(
            _iter_rows(
                connection,
                table,
                since=args.since,
                until=args.until,
                batch_size=args.batch_size,
            )
        ):
            if row_index:
                stream.write(",")
            stream.write(json.dumps(row, default=str))
        stream.write("]")
    stream.write("}}\n")


def main() -> int:
    args = _parse_args()
    if args.output and Path(args.output).expanduser().resolve() == Path(
        args.db
    ).expanduser().resolve():
        raise ValueError("audit output path must not overwrite the SQLite database")
    with _open_read_only(args.db) as connection:
        available = _available_tables(connection)
        tables = args.tables or available
        unknown = sorted(set(tables).difference(available))
        if unknown:
            raise ValueError(f"unknown table(s): {', '.join(unknown)}")
        output_context = (
            open(args.output, "w", encoding="utf-8")
            if args.output
            else nullcontext(sys.stdout)
        )
        with output_context as stream:
            if args.format == "json":
                _write_json(stream, connection, tables, args)
            else:
                _write_ndjson(stream, connection, tables, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
