"""Quick diagnostic to check PredictBase balance and cancel orders.

NOTE: PredictBase API does not expose endpoints for:
  - Listing open orders
  - Checking account balance

You must check these via the PredictBase web UI:
  https://predictbase.app

To cancel a specific order, use: cancel_order("<order_id>")
"""
from config import load_settings
from predictbase_client import PredictBaseClient
from web3_client import Web3Client


def get_client():
    """Initialize and return PredictBase client."""
    settings = load_settings()

    wallet_address = None
    if settings.WALLET_PRIVATE_KEY:
        web3_client = Web3Client(
            rpc_url=settings.ALCHEMY_RPC_URL,
            private_key=settings.WALLET_PRIVATE_KEY,
            usdc_token_address=settings.USDC_TOKEN_ADDRESS,
            chain_id=settings.CHAIN_ID,
        )
        wallet_address = web3_client.address
        print(f"Wallet: {wallet_address}")

    return PredictBaseClient(
        base_url=settings.PREDICTBASE_API_BASE_URL,
        api_key=settings.PREDICTBASE_API_KEY,
        api_key_header=settings.PREDICTBASE_API_KEY_HEADER,
        api_key_prefix=settings.PREDICTBASE_API_KEY_PREFIX,
        wallet_address=wallet_address,
    )


def cancel_order(order_id: str):
    """Cancel a specific order by ID.
    
    Get order IDs from the PredictBase web UI.
    """
    client = get_client()
    print(f"Cancelling order {order_id}...")
    try:
        result = client.cancel_order(order_id)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Error: {e}")


def main():
    print("PredictBase Order Management")
    print("=" * 40)
    print()
    print("The PredictBase API does not have endpoints to:")
    print("  - List open orders")
    print("  - Check account balance")
    print()
    print("Please check your orders and balance at:")
    print("  https://predictbase.app")
    print()
    print("To cancel a specific order, run:")
    print('  poetry run python -c "from check_balance import cancel_order; cancel_order(\'ORDER_ID\')"')
    print()
    print("The negative 'available' balance (-108.99) means your open")
    print("limit orders are reserving more funds than you have deposited.")
    print("Cancel stale orders via the web UI to free up funds.")


if __name__ == "__main__":
    main()

