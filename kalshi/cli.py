from __future__ import annotations

import argparse

from main import (
    GuaranteedOrdersIncompleteError,
    GuaranteedPlanLifecycleError,
    main as run_main,
)


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
    plan_actions = parser.add_mutually_exclusive_group()
    plan_actions.add_argument(
        "--abandon-guaranteed-plan",
        action="store_true",
        help=(
            "Clear the active guaranteed-order plan and exit without "
            "initializing API clients"
        ),
    )
    plan_actions.add_argument(
        "--new-guaranteed-run",
        action="store_true",
        help=(
            "Replace the active guaranteed-order plan with a fresh plan using "
            "GUARANTEED_ORDERS_N"
        ),
    )
    args = parser.parse_args()
    try:
        run_main(
            max_cycles=args.cycles,
            abandon_guaranteed_plan=args.abandon_guaranteed_plan,
            new_guaranteed_run=args.new_guaranteed_run,
        )
    except (GuaranteedOrdersIncompleteError, GuaranteedPlanLifecycleError) as exc:
        parser.exit(2, f"PredictBot stopped: {exc}\n")


if __name__ == "__main__":
    main()

