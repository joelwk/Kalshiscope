import unittest
from types import SimpleNamespace
from unittest.mock import patch

from models import OnChainPayload
from web3_client import Web3Client


class FakeContractFunctions:
    def __init__(self):
        self.last_approve = None

    def approve(self, spender, amount):
        self.last_approve = (spender, amount)

        def build_transaction(tx):
            tx["data"] = "approve"
            return tx

        return SimpleNamespace(build_transaction=build_transaction)

    def allowance(self, owner, spender):
        return SimpleNamespace(call=lambda: 123)


class FakeContract:
    def __init__(self):
        self.functions = FakeContractFunctions()


class FakeAccount:
    def __init__(self):
        self.address = "0xabc"
        self.last_tx = None

    def sign_transaction(self, tx):
        self.last_tx = tx
        return SimpleNamespace(rawTransaction=b"raw")


class FakeAccountModule:
    def from_key(self, key):
        return FakeAccount()


class FakeEth:
    def __init__(self):
        self.account = FakeAccountModule()
        self.chain_id = 8453
        self.gas_price = 123
        self.last_raw = None
        self.contract_instance = FakeContract()

    def get_transaction_count(self, address):
        return 7

    def estimate_gas(self, tx):
        return 21000

    def send_raw_transaction(self, raw):
        self.last_raw = raw
        return b"\x12\x34"

    def contract(self, address, abi):
        return self.contract_instance


class FakeWeb3:
    class HTTPProvider:
        def __init__(self, url):
            self.url = url

    def __init__(self, provider):
        self.provider = provider
        self.eth = FakeEth()

    def is_connected(self):
        return True

    @staticmethod
    def to_checksum_address(address):
        return address


class TestWeb3Client(unittest.TestCase):
    def test_approve_and_send_payload(self) -> None:
        with patch("web3_client.Web3", FakeWeb3):
            client = Web3Client(
                rpc_url="https://rpc.example",
                private_key="0xkey",
                usdc_token_address="0xusdc",
                chain_id=1,
            )

            tx_hash = client.approve_usdc("0xspender", 1.5, decimals=6)
            self.assertEqual(tx_hash, "1234")
            spender, amount = client.usdc_contract.functions.last_approve
            self.assertEqual(spender, "0xspender")
            self.assertEqual(amount, 1_500_000)
            self.assertEqual(client.account.last_tx["chainId"], 1)

            payload = OnChainPayload(to="0xtrade", data="0xdead", value_wei=5)
            trade_hash = client.send_onchain_payload(payload)
            self.assertEqual(trade_hash, "1234")
            self.assertEqual(client.account.last_tx["to"], "0xtrade")
            self.assertEqual(client.account.last_tx["data"], "0xdead")
            self.assertEqual(client.account.last_tx["value"], 5)


if __name__ == "__main__":
    unittest.main()
