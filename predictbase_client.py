from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import requests

from logging_config import get_logger, log_api_call
from models import (
    InsufficientBalanceError,
    Market,
    MarketOutcome,
    OnChainPayload,
    OrderRequest,
    OrderResponse,
)

logger = get_logger(__name__)

USDC_MICRO_PRECISION = 1_000_000


class PredictBaseClient:
    """Client for interacting with the PredictBase API."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        api_key_header: str = "Authorization",
        api_key_prefix: str = "Bearer",
        timeout_sec: int = 20,
        wallet_address: str | None = None,
        slippage_confidence_threshold: float = 0.70,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.wallet_address = wallet_address
        self.slippage_confidence_threshold = slippage_confidence_threshold
        self.session = requests.Session()
        if api_key:
            if api_key_prefix:
                value = f"{api_key_prefix} {api_key}"
            else:
                value = api_key
            self.session.headers[api_key_header] = value
        logger.debug(
            "PredictBaseClient initialized: base_url=%s, timeout=%ds, wallet=%s",
            self.base_url,
            timeout_sec,
            wallet_address[:10] + "..." if wallet_address else None,
        )

    def get_markets(self) -> list[Market]:
        """Fetch all active markets from PredictBase.

        Returns:
            List of parsed Market objects
        """
        url = f"{self.base_url}/get_active_markets"
        start_time = time.monotonic()

        logger.debug("Fetching markets from %s", url)

        try:
            resp = self.session.get(url, timeout=self.timeout_sec)
            duration_ms = (time.monotonic() - start_time) * 1000

            log_api_call(
                logger,
                method="GET",
                url=url,
                status_code=resp.status_code,
                duration_ms=duration_ms,
            )

            resp.raise_for_status()
            payload = resp.json()

            if isinstance(payload, list):
                raw_markets = payload
            else:
                raw_markets = payload.get("markets") or payload.get("data") or payload

            logger.debug(
                "Received %d raw market entries from API",
                len(raw_markets) if isinstance(raw_markets, list) else 0,
            )

            # Log sample market keys to help debug field names
            if raw_markets and isinstance(raw_markets, list) and len(raw_markets) > 0:
                sample = raw_markets[0]
                logger.debug(
                    "Sample market keys: %s",
                    list(sample.keys()) if isinstance(sample, dict) else "N/A",
                )

            markets: list[Market] = []
            parse_errors = 0
            for raw in raw_markets:
                try:
                    markets.append(_parse_market(raw))
                except Exception as exc:
                    parse_errors += 1
                    logger.warning(
                        "Skipping market due to parse error: %s",
                        exc,
                        data={"raw_market": str(raw)[:200], "error": str(exc)},
                    )

            logger.info(
                "Markets fetched: count=%d, parse_errors=%d, duration=%.2fms",
                len(markets),
                parse_errors,
                duration_ms,
                data={
                    "count": len(markets),
                    "parse_errors": parse_errors,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            return markets

        except requests.exceptions.RequestException as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            log_api_call(
                logger,
                method="GET",
                url=url,
                duration_ms=duration_ms,
                error=str(exc),
            )
            logger.error(
                "Failed to fetch markets: %s",
                exc,
                data={"url": url, "error": str(exc), "duration_ms": round(duration_ms, 2)},
            )
            raise

    def submit_order(
        self,
        order: OrderRequest,
        market: Market | None = None,
        slippage_pct: float = 0.0,
    ) -> OrderResponse:
        """Submit an order to PredictBase.

        Args:
            order: The order request to submit
            market: Optional market for outcome resolution

        Returns:
            OrderResponse with order details
        """
        url = f"{self.base_url}/create-order"
        start_time = time.monotonic()

        option_index = _find_option_index(order.outcome, market) if market else 0

        side = order.side.lower() if order.side else "buy"

        # Get price from market outcomes, with validation
        # PredictBase prices must be between 0.01 and 1.00
        MIN_VALID_PRICE = 0.01
        MAX_VALID_PRICE = 1.00
        DEFAULT_PRICE = 0.50
        price = DEFAULT_PRICE

        if market and market.outcomes:
            for outcome in market.outcomes:
                if outcome.name.upper() == order.outcome.upper():
                    # Only use price if it's in valid range (0.01-1.00)
                    if (
                        outcome.price is not None
                        and MIN_VALID_PRICE <= outcome.price <= MAX_VALID_PRICE
                    ):
                        price = outcome.price
                    elif outcome.odds is not None and outcome.odds > 0:
                        calculated_price = 1.0 / outcome.odds
                        if MIN_VALID_PRICE <= calculated_price <= MAX_VALID_PRICE:
                            price = calculated_price
                    break

        # Ensure price is valid for API (must be 0.01-1.00)
        if not (MIN_VALID_PRICE <= price <= MAX_VALID_PRICE):
            logger.warning(
                "Price out of range (%.4f), using default %.2f for market=%s",
                price,
                DEFAULT_PRICE,
                order.market_id,
            )
            price = DEFAULT_PRICE

        confidence = order.confidence
        if (
            confidence is not None
            and confidence >= self.slippage_confidence_threshold
            and slippage_pct > 0
        ):
            base_price = price
            if side == "buy":
                price = price * (1 + slippage_pct)
            else:
                price = price * (1 - slippage_pct)
            price = max(MIN_VALID_PRICE, min(MAX_VALID_PRICE, price))
            if price != base_price:
                logger.debug(
                    "Applied slippage buffer: market=%s, side=%s, conf=%.2f, "
                    "base_price=%.4f, adjusted_price=%.4f, pct=%.4f",
                    order.market_id,
                    side.upper(),
                    confidence,
                    base_price,
                    price,
                    slippage_pct,
                )

        # Calculate quantity as number of shares based on USDC amount and price
        # qty = amount_usdc / price (e.g., $5 at $0.42/share = 11 shares)
        qty_shares = int(order.amount_usdc / price)

        if qty_shares < 1:
            logger.warning(
                "Order quantity too small: amount=%.2f, price=%.4f, shares=%d",
                order.amount_usdc,
                price,
                qty_shares,
            )
            qty_shares = 1

        order_data = {
            "type": "LIMIT",
            "side": side.upper(),
            "marketId": order.market_id,
            "optionIndex": option_index,
            "qty": qty_shares,
            "price": price,
            "timeInForce": "GTC",
            # Docs include receivedAt (ms). Server may accept omission, but providing it
            # helps with debugging and deterministic ordering on their side.
            "receivedAt": int(time.time() * 1000),
        }

        # Add userId if wallet address is configured (required for API key binding)
        if self.wallet_address:
            order_data["userId"] = self.wallet_address

        payload = {
            "kind": "NEW_ORDER",
            "order": order_data,
            "meta": {
                # Keep stable client order ids for debugging / dedupe if server supports it.
                "clientOrderId": f"BOT#{order.market_id}#{int(time.time() * 1000)}",
                "originalQty": qty_shares,
            },
        }

        logger.debug(
            "Submitting order: market=%s, outcome=%s, amount=%.2f, price=%.4f, qty_shares=%d",
            order.market_id,
            order.outcome,
            order.amount_usdc,
            price,
            qty_shares,
            data={
                "market_id": order.market_id,
                "outcome": order.outcome,
                "amount_usdc": order.amount_usdc,
                "side": side.upper(),
                "price": price,
                "qty_shares": qty_shares,
                "option_index": option_index,
            },
        )

        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout_sec)
            duration_ms = (time.monotonic() - start_time) * 1000

            log_api_call(
                logger,
                method="POST",
                url=url,
                status_code=resp.status_code,
                duration_ms=duration_ms,
            )

            resp.raise_for_status()
            response_data = resp.json()

            logger.debug(
                "Order API response: market=%s, response=%s",
                order.market_id,
                response_data,
            )

            response_data["client_price"] = price
            response_data["client_qty_shares"] = qty_shares
            response_data["client_amount_usdc"] = order.amount_usdc

            onchain_payload = _parse_onchain_payload(response_data)

            # Try common field names for order ID
            order_id = (
                response_data.get("id")
                or response_data.get("orderId")
                or response_data.get("order_id")
                or response_data.get("orderRef")
                or response_data.get("clientOrderId")
            )

            # Check nested order field
            if not order_id:
                nested_order = response_data.get("order")
                if isinstance(nested_order, dict):
                    order_id = (
                        nested_order.get("id")
                        or nested_order.get("orderId")
                        or nested_order.get("orderRef")
                    )

            # Check meta field for clientOrderId
            if not order_id:
                meta = response_data.get("meta")
                if isinstance(meta, dict):
                    order_id = meta.get("clientOrderId")

            response = OrderResponse(
                id=str(order_id) if order_id else None,
                status=response_data.get("status") or response_data.get("orderStatus"),
                onchain_payload=onchain_payload,
                raw=response_data,
            )

            logger.info(
                "Order submitted successfully: id=%s, status=%s, has_onchain=%s, duration=%.2fms",
                response.id,
                response.status,
                onchain_payload is not None,
                duration_ms,
                data={
                    "order_id": response.id,
                    "status": response.status,
                    "market_id": order.market_id,
                    "outcome": order.outcome,
                    "amount_usdc": order.amount_usdc,
                    "has_onchain_payload": onchain_payload is not None,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            return response

        except requests.exceptions.RequestException as exc:
            duration_ms = (time.monotonic() - start_time) * 1000

            # Try to extract response body for better error messages
            response_body = None
            available_balance = None
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    response_body = exc.response.text
                    # Parse JSON to extract available balance
                    import json
                    try:
                        error_data = json.loads(response_body)
                        if "available" in error_data:
                            available_balance = float(error_data["available"])
                    except (json.JSONDecodeError, ValueError, TypeError):
                        pass
                except Exception:
                    pass

            log_api_call(
                logger,
                method="POST",
                url=url,
                duration_ms=duration_ms,
                error=str(exc),
            )
            logger.error(
                "Failed to submit order: market=%s, error=%s, response=%s",
                order.market_id,
                exc,
                response_body,
                data={
                    "market_id": order.market_id,
                    "outcome": order.outcome,
                    "amount_usdc": order.amount_usdc,
                    "error": str(exc),
                    "response_body": response_body,
                    "payload": payload,
                    "duration_ms": round(duration_ms, 2),
                    "available_balance": available_balance,
                },
            )

            # Raise specific exception for insufficient balance.
            # Avoid false positives like "InsufficientAllowance" by requiring "balance" context,
            # and prefer structured fields when present.
            lower_body = (response_body or "").lower()
            is_insufficient_balance = (
                ("insufficient" in lower_body and "balance" in lower_body)
                or available_balance is not None
            )
            if is_insufficient_balance:
                raise InsufficientBalanceError(
                    f"Insufficient balance: available={available_balance}",
                    available=available_balance,
                ) from exc

            raise

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an open order.

        Uses POST /cancel-order per PredictBase API docs.

        Args:
            order_id: The ID of the order to cancel

        Returns:
            API response dict
        """
        url = f"{self.base_url}/cancel-order"
        start_time = time.monotonic()

        payload = {
            "orderId": order_id,
            "user": self.wallet_address,
        }

        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout_sec)
            duration_ms = (time.monotonic() - start_time) * 1000

            log_api_call(
                logger,
                method="POST",
                url=url,
                status_code=resp.status_code,
                duration_ms=duration_ms,
            )

            resp.raise_for_status()
            data = resp.json()

            logger.info("Order cancelled: order_id=%s", order_id)
            return data

        except requests.exceptions.RequestException as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            log_api_call(
                logger,
                method="POST",
                url=url,
                duration_ms=duration_ms,
                error=str(exc),
            )
            raise


def _parse_market(raw: dict[str, Any]) -> Market:
    """Parse raw API response into a Market object."""
    market_id = str(
        raw.get("id")
        or raw.get("market_id")
        or raw.get("uuid")
        or raw.get("slug")
    )
    question = (
        raw.get("question")
        or raw.get("title")
        or raw.get("name")
        or ""
    )
    # Parse outcomes from various API formats
    outcomes = []

    # PredictBase format: optionTitles + optionPrices as parallel arrays
    # Docs: optionPrices are strings in 1e6 precision (e.g., "550000" => 0.55 USDC).
    option_titles = raw.get("optionTitles") or []
    option_prices = raw.get("optionPrices") or []
    if option_titles:
        for idx, title in enumerate(option_titles):
            price = option_prices[idx] if idx < len(option_prices) else None
            outcomes.append(
                MarketOutcome(
                    name=str(title),
                    price=_coerce_predictbase_price(price),
                )
            )
    else:
        # Fallback to other common formats
        outcomes_raw = raw.get("outcomes") or raw.get("answers") or raw.get("options") or []
        for entry in outcomes_raw:
            if isinstance(entry, str):
                outcomes.append(MarketOutcome(name=entry))
            elif isinstance(entry, dict):
                name = str(entry.get("name") or entry.get("label") or entry.get("outcome"))
                outcomes.append(
                    MarketOutcome(
                        name=name,
                        odds=_coerce_float(entry.get("odds")),
                        price=_coerce_float(entry.get("price")),
                    )
                )
    liquidity = _coerce_float(raw.get("liquidity_usdc") or raw.get("liquidity") or raw.get("totalLiquidity"))
    if liquidity is None:
        # PredictBase /get_active_markets returns `volume` in 1e6 precision (USDC micro-units).
        liquidity = _coerce_predictbase_amount_usdc(raw.get("volume"))
    category = raw.get("category") or _first_tag(raw.get("tags"))
    status = raw.get("status")
    winning_option_raw = (
        raw.get("winningOption")
        or raw.get("winning_option")
        or raw.get("winningOptionIndex")
        or raw.get("winning_option_index")
        or raw.get("winningOutcome")
        or raw.get("winning_outcome")
    )
    # Try multiple field names for close/end time
    close_time_raw = (
        raw.get("endsAt")  # PredictBase API field name
        or raw.get("close_time")
        or raw.get("closeTime")
        or raw.get("end_time")
        or raw.get("endTime")
        or raw.get("expires_at")
        or raw.get("expiresAt")
        or raw.get("resolution_time")
        or raw.get("resolutionTime")
        or raw.get("deadline")
        or raw.get("endDate")
        or raw.get("end_date")
    )
    close_time = _parse_datetime(close_time_raw)
    url = raw.get("url") or raw.get("market_url")

    if not market_id or not question:
        raise ValueError("Missing market id or question")

    extra = dict(raw)
    for key in (
        "id",
        "market_id",
        "uuid",
        "slug",
        "question",
        "title",
        "name",
        "outcomes",
        "answers",
        "options",
        "optionTitles",
        "optionPrices",
        "optionVolumes",
        "liquidity_usdc",
        "liquidity",
        "totalLiquidity",
        "category",
        "tags",
        "endsAt",
        "close_time",
        "closeTime",
        "end_time",
        "endTime",
        "expires_at",
        "expiresAt",
        "resolution_time",
        "resolutionTime",
        "deadline",
        "endDate",
        "end_date",
        "url",
        "market_url",
        "status",
        "winningOption",
        "winning_option",
        "winningOptionIndex",
        "winning_option_index",
        "winningOutcome",
        "winning_outcome",
    ):
        extra.pop(key, None)

    return Market(
        id=market_id,
        question=question,
        outcomes=outcomes,
        liquidity_usdc=liquidity,
        category=category,
        close_time=close_time,
        url=url,
        status=status,
        winning_option_raw=winning_option_raw,
        **extra,
    )


def _parse_onchain_payload(payload: dict[str, Any]) -> OnChainPayload | None:
    """Parse on-chain payload from order response."""
    data = payload.get("onchain_payload") or payload.get("tx") or payload.get("transaction")
    if not isinstance(data, dict):
        return None
    to = data.get("to") or data.get("contract")
    call_data = data.get("data") or data.get("calldata")
    if not to or not call_data:
        return None
    value = data.get("value") or data.get("value_wei")
    value_int = int(value) if value is not None else None
    return OnChainPayload(to=to, data=call_data, value_wei=value_int)


def _coerce_float(value: Any) -> float | None:
    """Safely convert value to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _coerce_predictbase_amount_usdc(value: Any) -> float | None:
    """Convert PredictBase USDC amounts that may be reported in 1e6 precision."""
    numeric = _coerce_float(value)
    if numeric is None:
        return None
    # Heuristic: values >= 1e3 are almost certainly micro-units, not dollars.
    # (A market with $1000+ volume would still be 1000.00, while micro-units would be 1_000_000_000.)
    if numeric >= 1_000:
        return numeric / USDC_MICRO_PRECISION
    return numeric


def _coerce_predictbase_price(value: Any) -> float | None:
    """Convert PredictBase optionPrices to dollars (0.00-1.00) when provided in 1e6 precision."""
    numeric = _coerce_float(value)
    if numeric is None:
        return None
    # Docs guarantee 1e6 precision for optionPrices; keep backward-compat for already-normalized values.
    if numeric > 1.0:
        return numeric / USDC_MICRO_PRECISION
    return numeric


def _parse_datetime(value: Any) -> datetime | None:
    """Parse datetime from various formats including Unix timestamps and ISO strings."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value

    # Handle Unix timestamps (seconds or milliseconds)
    if isinstance(value, (int, float)):
        try:
            # If value > 10 billion, assume milliseconds
            ts = value / 1000 if value > 10_000_000_000 else value
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (ValueError, OSError):
            return None

    str_value = str(value).strip()

    # Handle numeric strings (Unix timestamps passed as strings)
    if str_value.isdigit() or (str_value.startswith("-") and str_value[1:].isdigit()):
        try:
            ts = int(str_value)
            # If value > 10 billion, assume milliseconds
            ts = ts / 1000 if ts > 10_000_000_000 else ts
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (ValueError, OSError):
            pass

    # Handle 'Z' suffix (common UTC indicator) by replacing with +00:00
    if str_value.endswith("Z"):
        str_value = str_value[:-1] + "+00:00"

    # Try various ISO formats
    formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            parsed = datetime.strptime(str_value, fmt)
            # Ensure timezone-aware (assume UTC if naive)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            continue

    logger.debug("Failed to parse datetime: %s", value)
    return None


def _first_tag(tags: Any) -> str | None:
    """Extract first tag from tags list."""
    if isinstance(tags, (list, tuple)) and tags:
        return str(tags[0])
    return None


def _find_option_index(outcome: str, market: Market) -> int:
    """Find the index of the outcome in the market's outcomes list."""
    outcome_upper = outcome.upper()
    for idx, market_outcome in enumerate(market.outcomes):
        if market_outcome.name.upper() == outcome_upper:
            logger.debug(
                "Found outcome '%s' at index %d for market %s",
                outcome,
                idx,
                market.id,
            )
            return idx

    if outcome_upper in ("YES", "TRUE", "1"):
        logger.debug(
            "Mapping outcome '%s' to index 0 (YES/TRUE/1) for market %s",
            outcome,
            market.id,
        )
        return 0
    elif outcome_upper in ("NO", "FALSE", "0"):
        logger.debug(
            "Mapping outcome '%s' to index 1 (NO/FALSE/0) for market %s",
            outcome,
            market.id,
        )
        return 1

    logger.warning(
        "Could not find outcome '%s' in market %s, defaulting to index 0",
        outcome,
        market.id,
        data={
            "outcome": outcome,
            "market_id": market.id,
            "available_outcomes": [o.name for o in market.outcomes],
        },
    )
    return 0
