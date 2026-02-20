from __future__ import annotations

import time
from typing import Any

from web3 import Web3

from logging_config import get_logger, log_transaction
from models import OnChainPayload

logger = get_logger(__name__)

ERC20_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}, {"name": "_spender", "type": "address"}],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
]


class Web3Client:
    """Client for Web3 blockchain interactions."""

    def __init__(
        self,
        rpc_url: str,
        private_key: str,
        usdc_token_address: str,
        chain_id: int | None = None,
    ) -> None:
        logger.debug("Initializing Web3Client with RPC URL: %s", rpc_url[:50] + "...")

        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.web3.is_connected():
            logger.error("Failed to connect to RPC provider: %s", rpc_url[:50])
            raise ConnectionError("Unable to connect to RPC provider")

        self.account = self.web3.eth.account.from_key(private_key)
        self.address = self.account.address
        self.usdc_token_address = Web3.to_checksum_address(usdc_token_address)
        self.chain_id = chain_id or self.web3.eth.chain_id
        self.usdc_contract = self.web3.eth.contract(
            address=self.usdc_token_address, abi=ERC20_ABI
        )

        logger.info(
            "Web3Client initialized: address=%s, chain_id=%d, usdc_token=%s",
            self.address,
            self.chain_id,
            self.usdc_token_address,
            data={
                "wallet_address": self.address,
                "chain_id": self.chain_id,
                "usdc_token_address": self.usdc_token_address,
            },
        )

    def approve_usdc(self, spender: str, amount_usdc: float, decimals: int = 6) -> str:
        """Approve USDC spending for a contract.

        Args:
            spender: Contract address to approve
            amount_usdc: Amount to approve in USDC
            decimals: USDC decimals (default 6)

        Returns:
            Transaction hash
        """
        spender = Web3.to_checksum_address(spender)
        amount_wei = int(amount_usdc * (10**decimals))
        start_time = time.monotonic()

        logger.info(
            "Approving USDC: spender=%s, amount=%.2f USDC (%d wei)",
            spender,
            amount_usdc,
            amount_wei,
            data={
                "spender": spender,
                "amount_usdc": amount_usdc,
                "amount_wei": amount_wei,
            },
        )

        try:
            tx = self.usdc_contract.functions.approve(spender, amount_wei).build_transaction(
                self._build_base_tx()
            )
            tx_hash = self._sign_and_send(tx)
            duration_ms = (time.monotonic() - start_time) * 1000

            log_transaction(
                logger,
                tx_type="USDC_APPROVE",
                tx_hash=tx_hash,
                details={
                    "spender": spender,
                    "amount_usdc": amount_usdc,
                    "amount_wei": amount_wei,
                    "duration_ms": round(duration_ms, 2),
                },
            )

            logger.info(
                "USDC approval submitted: tx_hash=%s, duration=%.2fms",
                tx_hash,
                duration_ms,
            )
            return tx_hash

        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "USDC approval failed: spender=%s, error=%s, duration=%.2fms",
                spender,
                exc,
                duration_ms,
                data={
                    "spender": spender,
                    "amount_usdc": amount_usdc,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            raise

    def get_allowance(self, spender: str) -> int:
        """Get current USDC allowance for a spender.

        Args:
            spender: Contract address to check

        Returns:
            Current allowance in wei
        """
        spender = Web3.to_checksum_address(spender)
        start_time = time.monotonic()

        try:
            allowance = self.usdc_contract.functions.allowance(self.address, spender).call()
            duration_ms = (time.monotonic() - start_time) * 1000

            logger.debug(
                "Allowance check: spender=%s, allowance=%d, duration=%.2fms",
                spender,
                allowance,
                duration_ms,
                data={
                    "spender": spender,
                    "allowance_wei": allowance,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            return allowance

        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Allowance check failed: spender=%s, error=%s",
                spender,
                exc,
                data={
                    "spender": spender,
                    "error": str(exc),
                    "duration_ms": round(duration_ms, 2),
                },
            )
            raise

    def get_usdc_balance(self) -> int:
        """Get USDC balance in smallest units (wei/smallest denomination).

        Returns:
            Balance in wei
        """
        start_time = time.monotonic()

        try:
            balance = self.usdc_contract.functions.balanceOf(self.address).call()
            duration_ms = (time.monotonic() - start_time) * 1000

            logger.debug(
                "Balance check: address=%s, balance=%d wei, duration=%.2fms",
                self.address,
                balance,
                duration_ms,
            )
            return balance

        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Balance check failed: error=%s",
                exc,
                data={
                    "address": self.address,
                    "error": str(exc),
                    "duration_ms": round(duration_ms, 2),
                },
            )
            raise

    def has_sufficient_balance(self, amount_usdc: float, decimals: int = 6) -> bool:
        """Check if the *wallet* has enough on-chain USDC for the specified amount.

        Args:
            amount_usdc: Required amount in USDC
            decimals: USDC decimals (default 6)

        Returns:
            True if balance is sufficient
        """
        balance = self.get_usdc_balance()
        required = int(amount_usdc * (10**decimals))
        has_enough = balance >= required

        balance_usdc = balance / (10**decimals)

        if not has_enough:
            logger.warning(
                "Insufficient on-chain wallet USDC balance: have=%.2f, need=%.2f, shortfall=%.2f",
                balance_usdc,
                amount_usdc,
                amount_usdc - balance_usdc,
                data={
                    "wallet_address": self.address,
                    "balance_wei": balance,
                    "balance_usdc": balance_usdc,
                    "required_usdc": amount_usdc,
                    "shortfall_usdc": amount_usdc - balance_usdc,
                },
            )
        else:
            logger.debug(
                "On-chain wallet USDC balance check passed: have=%.2f, need=%.2f",
                balance_usdc,
                amount_usdc,
            )

        return has_enough

    def send_onchain_payload(self, payload: OnChainPayload) -> str:
        """Send an on-chain transaction payload.

        Args:
            payload: Transaction payload with to, data, and optional value

        Returns:
            Transaction hash
        """
        start_time = time.monotonic()
        to_address = Web3.to_checksum_address(payload.to)

        logger.info(
            "Sending on-chain payload: to=%s, data_length=%d, value=%s",
            to_address,
            len(payload.data) if payload.data else 0,
            payload.value_wei or 0,
            data={
                "to": to_address,
                "data_length": len(payload.data) if payload.data else 0,
                "value_wei": payload.value_wei or 0,
            },
        )

        try:
            tx = self._build_base_tx()
            tx.update(
                {
                    "to": to_address,
                    "data": payload.data,
                    "value": payload.value_wei or 0,
                }
            )
            tx_hash = self._sign_and_send(tx)
            duration_ms = (time.monotonic() - start_time) * 1000

            log_transaction(
                logger,
                tx_type="TRADE",
                tx_hash=tx_hash,
                details={
                    "to": to_address,
                    "value_wei": payload.value_wei or 0,
                    "duration_ms": round(duration_ms, 2),
                },
            )

            logger.info(
                "On-chain trade submitted: tx_hash=%s, duration=%.2fms",
                tx_hash,
                duration_ms,
            )
            return tx_hash

        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "On-chain trade failed: to=%s, error=%s, duration=%.2fms",
                to_address,
                exc,
                duration_ms,
                data={
                    "to": to_address,
                    "value_wei": payload.value_wei or 0,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            raise

    def _build_base_tx(self) -> dict[str, Any]:
        """Build base transaction parameters.

        Returns:
            Transaction dictionary with common fields
        """
        nonce = self.web3.eth.get_transaction_count(self.address)
        gas_price = self.web3.eth.gas_price

        logger.debug(
            "Building transaction: nonce=%d, gas_price=%d, chain_id=%d",
            nonce,
            gas_price,
            self.chain_id,
        )

        return {
            "from": self.address,
            "nonce": nonce,
            "gasPrice": gas_price,
            "chainId": self.chain_id,
        }

    def _sign_and_send(self, tx: dict[str, Any]) -> str:
        """Sign and send a transaction.

        Args:
            tx: Transaction dictionary

        Returns:
            Transaction hash
        """
        if "gas" not in tx:
            estimated_gas = self.web3.eth.estimate_gas(tx)
            tx["gas"] = estimated_gas
            logger.debug("Estimated gas: %d", estimated_gas)

        logger.debug(
            "Signing transaction: nonce=%d, gas=%d, gas_price=%d",
            tx.get("nonce"),
            tx.get("gas"),
            tx.get("gasPrice"),
        )

        signed = self.account.sign_transaction(tx)
        tx_hash = self.web3.eth.send_raw_transaction(signed.rawTransaction)
        tx_hash_hex = tx_hash.hex()

        logger.debug("Transaction sent: hash=%s", tx_hash_hex)
        return tx_hash_hex
