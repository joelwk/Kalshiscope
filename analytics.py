from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict


def _bucket(confidence: float) -> str:
    left = int(confidence * 10) / 10
    right = left + 0.1
    return f"{left:.1f}-{right:.1f}"


def run(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT confidence, implied_prob, won
            FROM trade_outcomes
            WHERE confidence IS NOT NULL
              AND won IS NOT NULL
            """
        ).fetchall()
        if not rows:
            print("No resolved outcomes with confidence found.")
            return

        total = len(rows)
        wins = sum(1 for row in rows if row["won"] == 1)
        print(f"Resolved trades: {total}")
        print(f"Win rate: {wins / total:.2%}")

        brier_sum = 0.0
        by_bucket: dict[str, list[int]] = defaultdict(list)
        edge_rows: list[tuple[float, int]] = []
        for row in rows:
            confidence = float(row["confidence"])
            outcome = int(row["won"])
            brier_sum += (confidence - outcome) ** 2
            by_bucket[_bucket(confidence)].append(outcome)
            implied = row["implied_prob"]
            if implied is not None:
                edge_rows.append((confidence - float(implied), outcome))

        print(f"Brier score: {brier_sum / total:.4f}")
        print("\nCalibration by confidence bucket:")
        for bucket in sorted(by_bucket):
            samples = by_bucket[bucket]
            win_rate = sum(samples) / len(samples)
            print(f"  {bucket}: n={len(samples)} win_rate={win_rate:.2%}")

        if edge_rows:
            edge_rows.sort(key=lambda item: item[0])
            decile_size = max(1, len(edge_rows) // 10)
            print("\nEdge deciles (lowest to highest):")
            for idx in range(0, len(edge_rows), decile_size):
                decile = edge_rows[idx : idx + decile_size]
                avg_edge = sum(item[0] for item in decile) / len(decile)
                win_rate = sum(item[1] for item in decile) / len(decile)
                print(
                    f"  decile={idx // decile_size + 1} n={len(decile)} "
                    f"avg_edge={avg_edge:.4f} win_rate={win_rate:.2%}"
                )
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily calibration metrics report.")
    parser.add_argument(
        "--db",
        default="data/market_state.db",
        help="Path to market state sqlite database",
    )
    args = parser.parse_args()
    run(args.db)


if __name__ == "__main__":
    main()

