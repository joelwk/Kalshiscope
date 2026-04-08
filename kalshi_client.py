from __future__ import annotations

import base64
import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from urllib3.util.retry import Retry

from logging_config import get_logger, log_api_call
from models import (
    InsufficientBalanceError,
    Market,
    MarketClosedError,
    MarketOutcome,
    OrderRequest,
    OrderResponse,
)

logger = get_logger(__name__)

_ONE_HUNDRED = 100
_MIN_VALID_PRICE = 0.01
_MAX_VALID_PRICE = 0.99
_DEFAULT_PRICE = 0.50
_DEFAULT_LIMIT = 1000
_DEFAULT_TIME_IN_FORCE = "immediate_or_cancel"
_MARKET_TIME_IN_FORCE = "fill_or_kill"
_MARKET_FALLBACK_YES_PRICE_CENTS = 97
_MARKET_FALLBACK_NO_PRICE_CENTS = 3
_ORDER_SUBMISSION_MIN_PRICE = 0.03
_ORDER_SUBMISSION_MAX_PRICE = 0.97
_KALSHI_RETRY_TOTAL = 3
_KALSHI_RETRY_BACKOFF_FACTOR = 0.5
_KALSHI_RETRYABLE_STATUS_CODES = (502, 503, 504)
_KALSHI_RETRY_ALLOWED_METHODS = frozenset({"GET", "POST", "PUT", "DELETE"})
_MARKETS_PAGE_MAX_ATTEMPTS = 3
_MARKETS_PAGE_RETRY_DELAY_SECONDS = 0.5
_TICKER_CONTEXT_MAX_LEN = 200
_TICKER_CITY_MAP = {
    "AUS": "Austin",
    "BOS": "Boston",
    "CHI": "Chicago",
    "DEN": "Denver",
    "HOU": "Houston",
    "LAX": "Los Angeles",
    "LV": "Las Vegas",
    "MIA": "Miami",
    "MIN": "Minneapolis",
    "NOLA": "New Orleans",
    "NY": "New York",
    "NYC": "New York City",
    "NYCH": "New York City",
    "OKC": "Oklahoma City",
    "PHX": "Phoenix",
    "SATX": "San Antonio",
    "SEA": "Seattle",
    "SFO": "San Francisco",
}


class KalshiClient:
    """Client for interacting with the Kalshi Trade API v2."""

    def __init__(
        self,
        base_url: str,
        api_key_id: str,
        private_key_path: str,
        timeout_sec: int = 20,
        order_price_improvement_cents: int = 0,
        max_fetch_pages: int | None = 10,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_id = api_key_id
        self.timeout_sec = timeout_sec
        self.order_price_improvement_cents = max(0, int(order_price_improvement_cents))
        self.max_fetch_pages = (
            None if max_fetch_pages is None or int(max_fetch_pages) <= 0
            else max(1, int(max_fetch_pages))
        )
        self.private_key = self._load_private_key(private_key_path)
        self.session = self._create_session()
        logger.debug(
            "KalshiClient initialized: base_url=%s timeout=%ds order_price_improvement_cents=%d",
            self.base_url,
            self.timeout_sec,
            self.order_price_improvement_cents,
            data={"max_fetch_pages": self.max_fetch_pages},
        )

    def _create_session(self) -> requests.Session:
        retry_strategy = Retry(
            total=_KALSHI_RETRY_TOTAL,
            connect=_KALSHI_RETRY_TOTAL,
            read=_KALSHI_RETRY_TOTAL,
            status=_KALSHI_RETRY_TOTAL,
            backoff_factor=_KALSHI_RETRY_BACKOFF_FACTOR,
            status_forcelist=_KALSHI_RETRYABLE_STATUS_CODES,
            allowed_methods=_KALSHI_RETRY_ALLOWED_METHODS,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def reset_session(self) -> None:
        self.session.close()
        self.session = self._create_session()
        logger.debug("Kalshi API session reset with retry adapter")

    @staticmethod
    def _load_private_key(private_key_path: str):
        pem_bytes = Path(private_key_path).read_bytes()
        return serialization.load_pem_private_key(pem_bytes, password=None)

    def _build_signed_headers(self, method: str, path: str) -> dict[str, str]:
        timestamp_ms = str(int(time.time() * 1000))
        message = f"{timestamp_ms}{method.upper()}{path}".encode("utf-8")
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        signature_b64 = base64.b64encode(signature).decode("utf-8")
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": signature_b64,
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> requests.Response:
        url = urljoin(f"{self.base_url}/", path.lstrip("/"))
        parsed = urlparse(url)
        signed_path = parsed.path
        headers = self._build_signed_headers(method, signed_path)
        start_time = time.monotonic()
        response = None
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=json,
                headers=headers,
                timeout=self.timeout_sec,
            )
            duration_ms = (time.monotonic() - start_time) * 1000
            log_api_call(
                logger,
                method=method.upper(),
                url=url,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            log_api_call(
                logger,
                method=method.upper(),
                url=url,
                duration_ms=duration_ms,
                error=str(exc),
            )
            raise

    def get_markets(
        self,
        status: str = "open",
        limit: int = _DEFAULT_LIMIT,
        *,
        close_time_start: datetime | None = None,
        close_time_end: datetime | None = None,
    ) -> list[Market]:
        """Fetch all paginated markets from Kalshi."""
        markets: list[Market] = []
        seen_market_ids: set[str] = set()
        cursor: str | None = None
        pages_fetched = 0
        while True:
            params: dict[str, Any] = {"limit": limit}
            if status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor
            if close_time_start is not None:
                params["close_time_start"] = close_time_start.astimezone(timezone.utc).isoformat()
            if close_time_end is not None:
                params["close_time_end"] = close_time_end.astimezone(timezone.utc).isoformat()
            page_attempt = 0
            while True:
                page_attempt += 1
                try:
                    response = self._request("GET", "/markets", params=params)
                    break
                except requests.exceptions.RequestException as exc:
                    if not _should_retry_market_page(exc) or page_attempt >= _MARKETS_PAGE_MAX_ATTEMPTS:
                        raise
                    next_attempt = page_attempt + 1
                    logger.warning(
                        "Kalshi market page fetch failed; retrying page attempt %d/%d: %s",
                        next_attempt,
                        _MARKETS_PAGE_MAX_ATTEMPTS,
                        exc,
                        data={
                            "error": str(exc),
                            "cursor": cursor,
                            "status": status,
                            "limit": limit,
                            "next_attempt": next_attempt,
                        },
                    )
                    self.reset_session()
                    time.sleep(_MARKETS_PAGE_RETRY_DELAY_SECONDS)
            payload = response.json()
            raw_markets = payload.get("markets", []) if isinstance(payload, dict) else []
            for raw_market in raw_markets:
                try:
                    parsed_market = _parse_market(raw_market)
                except (ValueError, KeyError, TypeError) as exc:
                    logger.warning(
                        "Skipping malformed market payload during pagination: %s",
                        exc,
                        data={
                            "error": str(exc),
                            "cursor": cursor,
                            "raw_ticker": (
                                raw_market.get("ticker")
                                if isinstance(raw_market, dict)
                                else None
                            ),
                        },
                    )
                    continue
                if parsed_market.id in seen_market_ids:
                    continue
                seen_market_ids.add(parsed_market.id)
                markets.append(parsed_market)
            pages_fetched += 1
            cursor = payload.get("cursor") if isinstance(payload, dict) else None
            if self.max_fetch_pages is not None and pages_fetched >= self.max_fetch_pages:
                if cursor:
                    logger.warning(
                        "Kalshi market fetch stopped at configured page cap: pages=%d",
                        pages_fetched,
                        data={
                            "pages_fetched": pages_fetched,
                            "max_fetch_pages": self.max_fetch_pages,
                        },
                    )
                break
            if not cursor:
                break
        return markets

    def get_market(self, market_ticker: str) -> Market:
        response = self._request("GET", f"/markets/{market_ticker}")
        payload = response.json()
        raw_market = payload.get("market", payload) if isinstance(payload, dict) else payload
        return _parse_market(raw_market)

    def get_market_orderbook(self, market_ticker: str) -> dict[str, Any]:
        response = self._request("GET", f"/markets/{market_ticker}/orderbook")
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Orderbook response is not a JSON object")
        return payload

    def get_balance(self) -> float:
        """Return available balance in dollars."""
        response = self._request("GET", "/portfolio/balance")
        payload = response.json()
        if not isinstance(payload, dict):
            return 0.0
        value_cents = (
            payload.get("available_balance")
            or payload.get("available")
            or payload.get("balance")
            or payload.get("portfolio_balance")
            or 0
        )
        return _coerce_money_to_dollars(value_cents)

    def get_positions(self) -> dict[str, Any]:
        response = self._request("GET", "/portfolio/positions")
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Positions response is not a JSON object")
        return payload

    def create_order(self, order: OrderRequest, market: Market | None = None) -> OrderResponse:
        return self.submit_order(order, market=market)

    def submit_order(
        self,
        order: OrderRequest,
        market: Market | None = None,
        *,
        retry_suffix: str | None = None,
    ) -> OrderResponse:
        """Submit a buy/sell order in Kalshi format."""
        side = _to_kalshi_side(order.outcome)
        action = order.side.lower() if order.side else "buy"
        if action not in {"buy", "sell"}:
            action = "buy"
        normalized_order_type = (order.order_type or "limit").strip().lower()
        is_market_order = normalized_order_type == "market"

        price = _resolve_order_price(
            order=order,
            market=market,
            action=action,
            order_price_improvement_cents=self.order_price_improvement_cents,
        )
        if price < _ORDER_SUBMISSION_MIN_PRICE or price > _ORDER_SUBMISSION_MAX_PRICE:
            raise ValueError(
                f"Order price {price:.3f} outside submission band "
                f"[{_ORDER_SUBMISSION_MIN_PRICE:.2f}, {_ORDER_SUBMISSION_MAX_PRICE:.2f}]"
            )
        count = max(1, int(order.amount_usdc / price))
        price_cents = int(round(price * _ONE_HUNDRED))
        price_cents = max(1, min(99, price_cents))
        if is_market_order:
            price_cents = (
                _MARKET_FALLBACK_YES_PRICE_CENTS
                if side == "yes"
                else _MARKET_FALLBACK_NO_PRICE_CENTS
            )

        payload = {
            "ticker": order.market_id,
            "client_order_id": _build_client_order_id(
                order.market_id or "",
                suffix=retry_suffix,
            ),
            "type": "market" if is_market_order else "limit",
            "time_in_force": _MARKET_TIME_IN_FORCE if is_market_order else _DEFAULT_TIME_IN_FORCE,
            "action": action,
            "side": side,
            "count": count,
        }
        if side == "yes":
            payload["yes_price"] = price_cents
        else:
            payload["no_price"] = price_cents

        try:
            response = self._request("POST", "/portfolio/orders", json=payload)
        except requests.exceptions.HTTPError as exc:
            response_text = ""
            if exc.response is not None:
                response_text = exc.response.text.lower()
                logger.error(
                    "Order rejected by Kalshi: market=%s status=%s payload=%s body=%s",
                    order.market_id,
                    exc.response.status_code,
                    payload,
                    exc.response.text,
                    data={
                        "market_id": order.market_id,
                        "status_code": exc.response.status_code,
                        "payload": payload,
                        "response_body": exc.response.text,
                    },
                )
            if "insufficient" in response_text and "balance" in response_text:
                raise InsufficientBalanceError("Insufficient balance on Kalshi account") from exc
            if "market_closed" in response_text or "market closed" in response_text:
                raise MarketClosedError("Market closed before order submission") from exc
            raise

        response_data = response.json()
        response_order = response_data.get("order", response_data) if isinstance(response_data, dict) else {}
        order_id = (
            response_order.get("order_id")
            or response_order.get("id")
            or response_data.get("order_id")
            or response_data.get("id")
        )
        status = response_order.get("status") or response_data.get("status")
        response_data["client_price"] = price
        response_data["client_qty_shares"] = count
        response_data["client_amount_usdc"] = order.amount_usdc
        return OrderResponse(
            id=str(order_id) if order_id else None,
            status=status,
            raw=response_data,
        )

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        response = self._request("DELETE", f"/portfolio/orders/{order_id}")
        payload = response.json()
        if not isinstance(payload, dict):
            return {"ok": True}
        return payload


def _to_kalshi_side(outcome: str) -> str:
    normalized = (outcome or "").strip().lower()
    if normalized in {"yes", "true", "1"}:
        return "yes"
    if normalized in {"no", "false", "0"}:
        return "no"
    return "yes"


def _resolve_order_price(
    order: OrderRequest,
    market: Market | None,
    *,
    action: str,
    order_price_improvement_cents: int,
) -> float:
    resolved_price = _DEFAULT_PRICE
    if market:
        for outcome in market.outcomes:
            if outcome.name.strip().lower() == order.outcome.strip().lower() and outcome.price is not None:
                resolved_price = outcome.price
                break
    if action.strip().lower() == "buy" and order_price_improvement_cents > 0:
        resolved_price += order_price_improvement_cents / _ONE_HUNDRED
    return max(_MIN_VALID_PRICE, min(_MAX_VALID_PRICE, resolved_price))


def _build_client_order_id(market_id: str, suffix: str | None = None) -> str:
    """Build a compact, deterministic client order id within API length limits."""
    timestamp_ms = int(time.time() * 1000)
    digest = hashlib.sha1(market_id.encode("utf-8")).hexdigest()[:12]
    client_order_id = f"BOT-{digest}-{timestamp_ms}"
    if suffix:
        safe_suffix = re.sub(r"[^A-Za-z0-9_-]", "", suffix)
        if safe_suffix:
            client_order_id = f"{client_order_id}-{safe_suffix}"
    return client_order_id


def _parse_market(raw: dict[str, Any]) -> Market:
    ticker = str(raw.get("ticker") or raw.get("id") or "")
    title = _clean_text(raw.get("title"))
    subtitle = _clean_text(raw.get("subtitle"))
    fallback_question = _clean_text(raw.get("question"))
    rules_primary = _clean_text(raw.get("rules_primary"))
    question = _build_market_question(
        ticker=ticker,
        title=title,
        subtitle=subtitle,
        fallback_question=fallback_question,
    )
    if not ticker or not question:
        raise ValueError("Missing ticker/title in market payload")

    yes_price = _extract_yes_price(raw)
    no_price = None if yes_price is None else max(0.0, min(1.0, 1.0 - yes_price))
    outcomes = [
        MarketOutcome(name="YES", price=yes_price),
        MarketOutcome(name="NO", price=no_price),
    ]
    close_time = _parse_datetime(raw.get("close_time") or raw.get("expiration_time"))

    volume_value = _coerce_float(raw.get("volume"))
    if volume_value is None:
        volume_value = _coerce_float(raw.get("volume_fp"))
    open_interest_value = _coerce_float(raw.get("open_interest"))
    if open_interest_value is None:
        open_interest_value = _coerce_float(raw.get("open_interest_fp"))
    liquidity_value = _coerce_float(raw.get("liquidity"))
    if liquidity_value is None:
        liquidity_value = _coerce_float(raw.get("liquidity_dollars"))

    liquidity_usdc = volume_value
    if liquidity_usdc is None:
        liquidity_usdc = open_interest_value
    if liquidity_usdc is None:
        liquidity_usdc = liquidity_value

    volume_24h = _coerce_float(raw.get("volume_24h"))
    if volume_24h is None:
        volume_24h = _coerce_float(raw.get("volume_24h_fp"))
    event_ticker = str(raw.get("event_ticker") or "").strip() or None
    series_ticker = str(raw.get("series_ticker") or "").strip() or None
    market_type = str(raw.get("market_type") or "").strip() or None

    extra = dict(raw)
    for key in (
        "ticker",
        "id",
        "title",
        "subtitle",
        "question",
        "close_time",
        "expiration_time",
        "status",
        "volume",
        "volume_fp",
        "volume_24h",
        "volume_24h_fp",
        "open_interest",
        "open_interest_fp",
        "liquidity",
        "liquidity_dollars",
        "event_ticker",
        "series_ticker",
        "market_type",
    ):
        extra.pop(key, None)

    return Market(
        id=ticker,
        question=question,
        subtitle=subtitle,
        resolution_criteria=rules_primary,
        outcomes=outcomes,
        liquidity_usdc=liquidity_usdc,
        category=raw.get("category") or series_ticker,
        event_ticker=event_ticker,
        series_ticker=series_ticker,
        market_type=market_type,
        volume=volume_value,
        open_interest=open_interest_value,
        volume_24h=volume_24h,
        close_time=close_time,
        url=None,
        status=raw.get("status"),
        **extra,
    )


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _build_market_question(
    *,
    ticker: str,
    title: str,
    subtitle: str,
    fallback_question: str,
) -> str:
    base_question = title or subtitle or fallback_question
    ticker_context = _extract_ticker_context(ticker=ticker, base_question=base_question)
    if ticker_context:
        if base_question:
            return f"{base_question} [Ticker context: {ticker_context}]"
        return f"{ticker} [Ticker context: {ticker_context}]"
    return base_question


def _extract_ticker_context(*, ticker: str, base_question: str) -> str | None:
    if not ticker:
        return None
    parts: list[str] = []
    city_context = _extract_weather_city_from_ticker(ticker)
    if city_context and city_context.lower() not in base_question.lower():
        parts.append(f"location={city_context}")

    threshold_token = _extract_ticker_value_token(ticker, "T")
    if threshold_token is not None and str(threshold_token).lower() not in base_question.lower():
        parts.append(f"threshold={threshold_token}")

    bin_token = _extract_ticker_value_token(ticker, "B")
    if bin_token is not None and str(bin_token).lower() not in base_question.lower():
        parts.append(f"bin_center={bin_token}")

    if not parts:
        return None
    return ", ".join(parts)[:_TICKER_CONTEXT_MAX_LEN]


def _extract_weather_city_from_ticker(ticker: str) -> str | None:
    weather_match = re.match(r"^KX(?:LOWT|HIGHT|TEMP)([A-Z]+)-", ticker.upper())
    if not weather_match:
        return None
    city_code = weather_match.group(1)
    return _TICKER_CITY_MAP.get(city_code, city_code)


def _extract_ticker_value_token(ticker: str, token_prefix: str) -> str | None:
    for token in ticker.split("-"):
        if not token.startswith(token_prefix):
            continue
        if len(token) <= 1:
            continue
        candidate = token[1:]
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", candidate):
            return candidate
    return None


def _extract_yes_price(raw: dict[str, Any]) -> float | None:
    candidates = [
        raw.get("yes_ask_dollars"),
        raw.get("yes_bid_dollars"),
        raw.get("last_price_dollars"),
        raw.get("yes_ask"),
        raw.get("yes_bid"),
        raw.get("last_price"),
    ]
    for candidate in candidates:
        parsed = _coerce_price(candidate)
        if parsed is not None:
            return parsed
    return None


def _coerce_price(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric > 1.0:
        numeric = numeric / _ONE_HUNDRED
    return max(0.0, min(1.0, numeric))


def _coerce_money_to_dollars(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric >= _ONE_HUNDRED:
        return numeric / _ONE_HUNDRED
    return numeric


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def _should_retry_market_page(exc: requests.exceptions.RequestException) -> bool:
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    response = getattr(exc, "response", None)
    if response is None:
        return False
    return response.status_code in _KALSHI_RETRYABLE_STATUS_CODES
