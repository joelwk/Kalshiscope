from __future__ import annotations

import argparse

from main import main as run_main


def _positive_cycle_count(value: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("--cycles must be a positive integer")
    return parsed_value


def main() -> None:
    parser = argparse.ArgumentParser(description="PredictBot - Kalshi trading bot")
    parser.add_argument(
        "--cycles",
        type=_positive_cycle_count,
        default=None,
        help="Stop after N cycles (default: run indefinitely)",
    )
    args = parser.parse_args()
    run_main(max_cycles=args.cycles)


if __name__ == "__main__":
    main()

