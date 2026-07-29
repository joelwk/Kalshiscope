"""Quick diagnostic to check Kalshi balance, positions, and order cancellation."""
from typing import Any

from config import load_settings
from kalshi_client import KalshiClient


POSITION_RESPONSE_KEYS = ("market_positions", "positions", "portfolio_positions", "data")


def get_client():
    """Initialize and return Kalshi client."""
    settings = load_settings()
    return KalshiClient(
        base_url=settings.KALSHI_API_BASE_URL,
        api_key_id=settings.KALSHI_API_KEY_ID,
        private_key_path=settings.KALSHI_PRIVATE_KEY_PATH,
    )


def cancel_order(order_id: str):
    """Cancel a specific order by ID.
    
    Get order IDs from Kalshi account activity.
    """
    client = get_client()
    print(f"Cancelling order {order_id}...")
    try:
        result = client.cancel_order(order_id)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Error: {e}")


def _extract_market_positions(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in POSITION_RESPONSE_KEYS:
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        dict_rows = [row for row in rows if isinstance(row, dict)]
        if dict_rows:
            return dict_rows
    return []


def show_balance_and_positions() -> None:
    """Print current account balance and a positions summary."""
    client = get_client()
    balance = client.get_balance()
    positions = client.get_positions()
    market_positions = _extract_market_positions(positions)
    print(f"Available balance: ${balance:.2f}")
    print(f"Open market positions: {len(market_positions)}")


def main():
    print("Kalshi Account Diagnostic")
    print("=" * 40)
    show_balance_and_positions()
    print("To cancel a specific order, run:")
    print('  poetry run python -c "from check_balance import cancel_order; cancel_order(\'ORDER_ID\')"')


if __name__ == "__main__":
    main()

