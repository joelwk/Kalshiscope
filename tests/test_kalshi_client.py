import base64
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import requests

from kalshi_client import KalshiClient, _parse_market
from models import Market, MarketClosedError, OrderRequest


class _DummyPrivateKey:
    def sign(self, message, padding, algorithm):  # noqa: ARG002
        return b"test-signature"


class _DummyResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _DummyHttpResponse:
    def __init__(self, text: str, status_code: int = 409) -> None:
        self.text = text
        self.status_code = status_code


class TestKalshiClient(unittest.TestCase):
    def _client(self) -> KalshiClient:
        with patch.object(KalshiClient, "_load_private_key", return_value=_DummyPrivateKey()):
            return KalshiClient(
                base_url="https://api.example/trade-api/v2",
                api_key_id="test-key",
                private_key_path="unused.pem",
            )

    def test_signed_headers_include_access_fields(self) -> None:
        client = self._client()
        headers = client._build_signed_headers("GET", "/trade-api/v2/markets")
        self.assertEqual(headers["KALSHI-ACCESS-KEY"], "test-key")
        self.assertIn("KALSHI-ACCESS-TIMESTAMP", headers)
        self.assertEqual(
            headers["KALSHI-ACCESS-SIGNATURE"],
            base64.b64encode(b"test-signature").decode("utf-8"),
        )

    def test_parse_market_builds_binary_outcomes(self) -> None:
        market = _parse_market(
            {
                "ticker": "KXBTC-26APR06-T100000",
                "title": "Will BTC close above 100k?",
                "status": "open",
                "yes_ask": 62,
                "close_time": "2026-04-06T20:00:00Z",
                "series_ticker": "KXBTC",
                "event_ticker": "KXBTC-26APR06",
                "volume_24h": 12345,
                "market_type": "binary",
            }
        )
        self.assertIsInstance(market, Market)
        self.assertEqual(market.id, "KXBTC-26APR06-T100000")
        self.assertEqual(len(market.outcomes), 2)
        self.assertEqual(market.outcomes[0].name, "YES")
        self.assertAlmostEqual(market.outcomes[0].price or 0.0, 0.62)
        self.assertEqual(market.outcomes[1].name, "NO")
        self.assertAlmostEqual(market.outcomes[1].price or 0.0, 0.38)
        self.assertEqual(market.event_ticker, "KXBTC-26APR06")
        self.assertEqual(market.series_ticker, "KXBTC")
        self.assertEqual(market.market_type, "binary")
        self.assertEqual(market.volume_24h, 12345)

    def test_parse_market_uses_liquidity_fallback_when_volume_missing(self) -> None:
        market = _parse_market(
            {
                "ticker": "KXBTC-26APR06-T95000",
                "title": "Will BTC close above 95k?",
                "status": "open",
                "yes_ask": 44,
                "liquidity": 7500,
            }
        )
        self.assertIsInstance(market, Market)
        self.assertEqual(market.liquidity_usdc, 7500)

    def test_parse_market_supports_fixed_point_volume_fields(self) -> None:
        market = _parse_market(
            {
                "ticker": "KXBTC-26APR06-T90000",
                "title": "Will BTC close above 90k?",
                "status": "open",
                "yes_ask_dollars": 0.42,
                "volume_fp": 33,
                "volume_24h_fp": 12,
                "open_interest_fp": 21,
                "liquidity_dollars": 0,
            }
        )
        self.assertEqual(market.volume, 33)
        self.assertEqual(market.open_interest, 21)
        self.assertEqual(market.volume_24h, 12)
        self.assertEqual(market.liquidity_usdc, 33)

    def test_parse_market_enriches_weather_question_and_resolution(self) -> None:
        market = _parse_market(
            {
                "ticker": "KXLOWTCHI-26APR06-B33.5",
                "title": "Will the minimum temperature be 33-34 on Apr 6, 2026?",
                "subtitle": "Chicago overnight low range",
                "rules_primary": "Resolves to the official city weather station minimum.",
                "yes_ask": 42,
                "status": "open",
            }
        )
        self.assertIn("Ticker context", market.question)
        self.assertIn("location=Chicago", market.question)
        self.assertIn("bin_center=33.5", market.question)
        self.assertEqual(market.subtitle, "Chicago overnight low range")
        self.assertEqual(
            market.resolution_criteria,
            "Resolves to the official city weather station minimum.",
        )

    def test_get_markets_handles_cursor_pagination(self) -> None:
        client = self._client()
        pages = [
            _DummyResponse(
                {
                    "markets": [
                        {
                            "ticker": "MKT-1",
                            "title": "Q1",
                            "yes_ask": 55,
                        }
                    ],
                    "cursor": "next-1",
                }
            ),
            _DummyResponse(
                {
                    "markets": [
                        {
                            "ticker": "MKT-2",
                            "title": "Q2",
                            "yes_ask": 45,
                        }
                    ],
                    "cursor": "",
                }
            ),
        ]

        with patch.object(client, "_request", side_effect=pages):
            markets = client.get_markets()
        self.assertEqual([m.id for m in markets], ["MKT-1", "MKT-2"])

    def test_get_markets_passes_close_time_window_filters(self) -> None:
        client = self._client()
        pages = [
            _DummyResponse(
                {
                    "markets": [
                        {
                            "ticker": "MKT-1",
                            "title": "Q1",
                            "yes_ask": 55,
                        }
                    ],
                    "cursor": "",
                }
            ),
        ]
        start = datetime(2026, 4, 6, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 4, 7, 0, 0, tzinfo=timezone.utc)

        with patch.object(client, "_request", side_effect=pages) as request_mock:
            markets = client.get_markets(close_time_start=start, close_time_end=end)
        self.assertEqual([m.id for m in markets], ["MKT-1"])
        params = request_mock.call_args.kwargs["params"]
        self.assertEqual(params["close_time_start"], "2026-04-06T00:00:00+00:00")
        self.assertEqual(params["close_time_end"], "2026-04-07T00:00:00+00:00")

    def test_get_markets_respects_max_fetch_pages_cap(self) -> None:
        client = self._client()
        client.max_fetch_pages = 1
        pages = [
            _DummyResponse(
                {
                    "markets": [
                        {"ticker": "MKT-1", "title": "Q1", "yes_ask": 55},
                    ],
                    "cursor": "next-1",
                }
            ),
            _DummyResponse(
                {
                    "markets": [
                        {"ticker": "MKT-2", "title": "Q2", "yes_ask": 45},
                    ],
                    "cursor": "",
                }
            ),
        ]

        with patch.object(client, "_request", side_effect=pages) as request_mock:
            markets = client.get_markets()
        self.assertEqual([m.id for m in markets], ["MKT-1"])
        self.assertEqual(request_mock.call_count, 1)

    def test_get_markets_skips_malformed_market_payload_without_aborting(self) -> None:
        client = self._client()
        pages = [
            _DummyResponse(
                {
                    "markets": [
                        {"ticker": "MKT-OK-1", "title": "Q1", "yes_ask": 55},
                        {"ticker": "MKT-BAD", "title": ""},
                    ],
                    "cursor": "next-1",
                }
            ),
            _DummyResponse(
                {
                    "markets": [
                        {"ticker": "MKT-OK-2", "title": "Q2", "yes_ask": 45},
                    ],
                    "cursor": "",
                }
            ),
        ]
        with patch.object(client, "_request", side_effect=pages):
            markets = client.get_markets()
        self.assertEqual([m.id for m in markets], ["MKT-OK-1", "MKT-OK-2"])

    def test_get_markets_deduplicates_market_ids_across_pages(self) -> None:
        client = self._client()
        pages = [
            _DummyResponse(
                {
                    "markets": [
                        {"ticker": "MKT-DUP", "title": "Q1", "yes_ask": 55},
                    ],
                    "cursor": "next-1",
                }
            ),
            _DummyResponse(
                {
                    "markets": [
                        {"ticker": "MKT-DUP", "title": "Q1 duplicate", "yes_ask": 55},
                        {"ticker": "MKT-UNIQ", "title": "Q2", "yes_ask": 45},
                    ],
                    "cursor": "",
                }
            ),
        ]
        with patch.object(client, "_request", side_effect=pages):
            markets = client.get_markets()
        self.assertEqual([m.id for m in markets], ["MKT-DUP", "MKT-UNIQ"])

    def test_submit_order_maps_amount_to_count(self) -> None:
        client = self._client()
        market = Market(
            id="MKT-3",
            question="Question",
            outcomes=[{"name": "YES", "price": 0.50}, {"name": "NO", "price": 0.50}],
        )
        order = OrderRequest(
            market_id="MKT-3",
            outcome="YES",
            amount_usdc=5.0,
            side="BUY",
        )
        response_payload = {
            "order": {"order_id": "ord-1", "status": "resting"},
        }

        with patch.object(client, "_request", return_value=_DummyResponse(response_payload)) as req_mock:
            response = client.submit_order(order, market=market)

        sent_payload = req_mock.call_args.kwargs["json"]
        self.assertEqual(sent_payload["ticker"], "MKT-3")
        self.assertEqual(sent_payload["side"], "yes")
        self.assertEqual(sent_payload["time_in_force"], "immediate_or_cancel")
        self.assertEqual(sent_payload["count"], 10)
        self.assertEqual(sent_payload["yes_price"], 50)
        self.assertNotIn("no_price", sent_payload)
        self.assertEqual(response.id, "ord-1")
        self.assertEqual(response.status, "resting")

    def test_submit_order_uses_no_price_for_no_side(self) -> None:
        client = self._client()
        market = Market(
            id="MKT-4",
            question="Question",
            outcomes=[{"name": "YES", "price": 0.40}, {"name": "NO", "price": 0.60}],
        )
        order = OrderRequest(
            market_id="MKT-4",
            outcome="NO",
            amount_usdc=6.0,
            side="BUY",
        )
        response_payload = {"order": {"order_id": "ord-2", "status": "resting"}}

        with patch.object(client, "_request", return_value=_DummyResponse(response_payload)) as req_mock:
            response = client.submit_order(order, market=market)

        sent_payload = req_mock.call_args.kwargs["json"]
        self.assertEqual(sent_payload["side"], "no")
        self.assertEqual(sent_payload["count"], 10)
        self.assertEqual(sent_payload["no_price"], 60)
        self.assertNotIn("yes_price", sent_payload)
        self.assertEqual(response.id, "ord-2")

    def test_submit_order_market_uses_fill_or_kill_and_fallback_suffix(self) -> None:
        client = self._client()
        market = Market(
            id="MKT-6",
            question="Question",
            outcomes=[{"name": "YES", "price": 0.40}, {"name": "NO", "price": 0.60}],
        )
        order = OrderRequest(
            market_id="MKT-6",
            outcome="YES",
            amount_usdc=6.0,
            side="BUY",
            order_type="market",
        )
        response_payload = {"order": {"order_id": "ord-6", "status": "filled"}}

        with patch.object(client, "_request", return_value=_DummyResponse(response_payload)) as req_mock:
            response = client.submit_order(order, market=market, retry_suffix="fb")

        sent_payload = req_mock.call_args.kwargs["json"]
        self.assertEqual(sent_payload["type"], "market")
        self.assertEqual(sent_payload["time_in_force"], "fill_or_kill")
        self.assertEqual(sent_payload["yes_price"], 97)
        self.assertTrue(sent_payload["client_order_id"].endswith("-fb"))
        self.assertEqual(response.id, "ord-6")

    def test_submit_order_rejects_untradeable_price_band(self) -> None:
        client = self._client()
        market = Market(
            id="MKT-7",
            question="Question",
            outcomes=[{"name": "YES", "price": 0.99}, {"name": "NO", "price": 0.01}],
        )
        order = OrderRequest(
            market_id="MKT-7",
            outcome="YES",
            amount_usdc=5.0,
            side="BUY",
        )

        with self.assertRaises(ValueError):
            client.submit_order(order, market=market)

    def test_submit_order_raises_market_closed_error(self) -> None:
        client = self._client()
        market = Market(
            id="MKT-5",
            question="Question",
            outcomes=[{"name": "YES", "price": 0.50}, {"name": "NO", "price": 0.50}],
        )
        order = OrderRequest(
            market_id="MKT-5",
            outcome="YES",
            amount_usdc=5.0,
            side="BUY",
        )
        response = _DummyHttpResponse(
            '{"error":{"code":"market_closed","message":"market closed"}}'
        )
        http_error = requests.exceptions.HTTPError("409 market closed", response=response)

        with patch.object(client, "_request", side_effect=http_error):
            with self.assertRaises(MarketClosedError):
                client.submit_order(order, market=market)


if __name__ == "__main__":
    unittest.main()
