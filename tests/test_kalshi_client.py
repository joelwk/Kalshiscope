import base64
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import requests

from kalshi_client import KalshiClient, _normalize_time_in_force, _parse_market
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
        self.assertEqual(market.liquidity_usdc, 21)

    def test_parse_market_uses_open_interest_fallback_when_liquidity_missing(self) -> None:
        market = _parse_market(
            {
                "ticker": "KXBTC-26APR06-T87500",
                "title": "Will BTC close above 87.5k?",
                "status": "open",
                "yes_ask_dollars": 0.42,
                "volume_fp": 33,
                "open_interest_fp": 21,
            }
        )
        self.assertEqual(market.volume, 33)
        self.assertEqual(market.open_interest, 21)
        self.assertEqual(market.liquidity_usdc, 21)

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

    def test_get_portfolio_balance_reads_cash_and_total_value(self) -> None:
        client = self._client()
        payload = {
            "available_balance": 8305,
            "position_value": 1364,
            "total_portfolio_value": 9669,
        }
        with patch.object(client, "_request", return_value=_DummyResponse(payload)):
            balance = client.get_portfolio_balance()
        self.assertAlmostEqual(balance.available_balance, 83.05)
        self.assertAlmostEqual(balance.position_value, 13.64)
        self.assertAlmostEqual(balance.total_portfolio_value, 96.69)

    def test_get_portfolio_balance_falls_back_to_available_plus_pnl(self) -> None:
        client = self._client()
        payload = {
            "available_balance": 4505,
            "portfolio_pnl": 5164,
        }
        with patch.object(client, "_request", return_value=_DummyResponse(payload)):
            balance = client.get_portfolio_balance()
        self.assertAlmostEqual(balance.available_balance, 45.05)
        self.assertAlmostEqual(balance.position_value, 51.64)
        self.assertAlmostEqual(balance.total_portfolio_value, 96.69)

    def test_get_settlements_and_fills_return_payload(self) -> None:
        client = self._client()
        settlements_payload = {"settlements": [{"id": "s1"}]}
        fills_payload = {"fills": [{"id": "f1"}]}
        with patch.object(client, "_request", side_effect=[_DummyResponse(settlements_payload), _DummyResponse(fills_payload)]):
            settlements = client.get_settlements(limit=50, cursor="abc")
            fills = client.get_fills(limit=25)
        self.assertEqual(settlements, settlements_payload)
        self.assertEqual(fills, fills_payload)

    def test_portfolio_stream_methods_forward_cursor_and_fixed_point_filters(self) -> None:
        client = self._client()
        with patch.object(
            client,
            "_request",
            side_effect=[
                _DummyResponse({"market_positions": [], "cursor": ""}),
                _DummyResponse({"orders": [], "cursor": ""}),
                _DummyResponse({"order": {"order_id": "o-1"}}),
                _DummyResponse({"fills": [], "cursor": ""}),
                _DummyResponse({"settlements": [], "cursor": ""}),
            ],
        ) as request:
            client.get_positions(
                limit=1000,
                cursor="p-1",
                count_filter="position",
                subaccount=2,
            )
            client.get_orders(
                status="resting",
                limit=1000,
                cursor="o-1",
                subaccount=2,
            )
            client.get_order("o-1", subaccount=2)
            client.get_fills(
                limit=1000,
                cursor="f-1",
                order_id="o-1",
                min_ts=100,
                max_ts=200,
                subaccount=2,
            )
            client.get_settlements(
                limit=1000,
                cursor="s-1",
                min_ts=100,
                max_ts=200,
                subaccount=2,
            )

        self.assertEqual(
            request.call_args_list[0].kwargs["params"],
            {
                "limit": 1000,
                "cursor": "p-1",
                "count_filter": "position",
                "subaccount": 2,
            },
        )
        self.assertEqual(request.call_args_list[1].kwargs["params"]["status"], "resting")
        self.assertNotIn("params", request.call_args_list[2].kwargs)
        self.assertEqual(request.call_args_list[3].kwargs["params"]["min_ts"], 100)
        self.assertEqual(request.call_args_list[4].kwargs["params"]["max_ts"], 200)

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
        self.assertNotIn("mve_filter", params)

    def test_get_markets_forwards_mve_filter_query_param(self) -> None:
        """Cycle 2 review fix: the mve_filter=exclude server-side parameter
        drops KXMVE combo markets so the page cap is filled with individual
        sports/weather/crypto/music markets instead of combinatorial noise."""
        client = self._client()
        pages = [
            _DummyResponse(
                {
                    "markets": [
                        {"ticker": "MKT-INDIVIDUAL-1", "title": "Q1", "yes_ask": 55},
                    ],
                    "cursor": "",
                }
            ),
        ]

        with patch.object(client, "_request", side_effect=pages) as request_mock:
            markets = client.get_markets(mve_filter="exclude")
        self.assertEqual([m.id for m in markets], ["MKT-INDIVIDUAL-1"])
        params = request_mock.call_args.kwargs["params"]
        self.assertEqual(params["mve_filter"], "exclude")
        self.assertEqual(client.last_fetch_mve_filter, "exclude")
        self.assertEqual(client.last_fetch_pages, 1)
        self.assertFalse(client.last_fetch_cap_hit)

    def test_get_markets_normalizes_mve_filter_case_and_whitespace(self) -> None:
        client = self._client()
        pages = [
            _DummyResponse(
                {
                    "markets": [
                        {"ticker": "MKT-A", "title": "Q1", "yes_ask": 55},
                    ],
                    "cursor": "",
                }
            ),
        ]

        with patch.object(client, "_request", side_effect=pages) as request_mock:
            client.get_markets(mve_filter="  EXCLUDE  ")
        self.assertEqual(request_mock.call_args.kwargs["params"]["mve_filter"], "exclude")

    def test_get_markets_drops_unsupported_mve_filter_value(self) -> None:
        """An invalid mve_filter value should not be forwarded to Kalshi (which
        would 400 the request); a warning is logged and the param is omitted."""
        client = self._client()
        pages = [
            _DummyResponse(
                {
                    "markets": [
                        {"ticker": "MKT-A", "title": "Q1", "yes_ask": 55},
                    ],
                    "cursor": "",
                }
            ),
        ]

        with patch.object(client, "_request", side_effect=pages) as request_mock:
            client.get_markets(mve_filter="weird-value")
        params = request_mock.call_args.kwargs["params"]
        self.assertNotIn("mve_filter", params)
        self.assertIsNone(client.last_fetch_mve_filter)

    def test_get_markets_omits_mve_filter_when_unset(self) -> None:
        client = self._client()
        pages = [
            _DummyResponse(
                {
                    "markets": [
                        {"ticker": "MKT-A", "title": "Q1", "yes_ask": 55},
                    ],
                    "cursor": "",
                }
            ),
        ]

        with patch.object(client, "_request", side_effect=pages) as request_mock:
            client.get_markets()
        params = request_mock.call_args.kwargs["params"]
        self.assertNotIn("mve_filter", params)
        self.assertIsNone(client.last_fetch_mve_filter)

    def test_get_markets_tracks_pagination_metadata_when_cap_hit(self) -> None:
        """When the page cap stops pagination with a remaining cursor, the
        client should expose last_fetch_pages and last_fetch_cap_hit so the
        cycle receipt can record catalog topology."""
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
        ]
        with patch.object(client, "_request", side_effect=pages):
            client.get_markets()
        self.assertEqual(client.last_fetch_pages, 1)
        self.assertTrue(client.last_fetch_cap_hit)

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
            "order_id": "ord-1",
            "fill_count": "0.00",
            "remaining_count": "10.00",
            "ts_ms": 1,
        }

        with patch.object(client, "_request", return_value=_DummyResponse(response_payload)) as req_mock:
            response = client.submit_order(order, market=market)

        sent_payload = req_mock.call_args.kwargs["json"]
        self.assertEqual(sent_payload["ticker"], "MKT-3")
        # YES book: buying YES is a ``bid`` quoted in fixed-point dollars.
        self.assertEqual(sent_payload["side"], "bid")
        self.assertEqual(sent_payload["time_in_force"], "good_till_canceled")
        self.assertEqual(sent_payload["self_trade_prevention_type"], "taker_at_cross")
        self.assertEqual(sent_payload["count"], "10")
        self.assertEqual(sent_payload["price"], "0.5000")
        self.assertNotIn("yes_price", sent_payload)
        self.assertNotIn("no_price", sent_payload)
        self.assertNotIn("type", sent_payload)
        self.assertEqual(response.id, "ord-1")
        # Unfilled good-till-canceled limit rests on the book (not canceled).
        self.assertEqual(response.status, "resting")

    def test_submit_order_uses_explicit_idempotency_key(self) -> None:
        client = self._client()
        market = Market(
            id="MKT-IDEMPOTENT",
            question="Question",
            outcomes=[{"name": "YES", "price": 0.50}, {"name": "NO", "price": 0.50}],
        )
        order = OrderRequest(
            market_id=market.id,
            outcome="YES",
            amount_usdc=5.0,
        )
        response_payload = {
            "order_id": "ord-idempotent",
            "fill_count": "0.00",
            "remaining_count": "10.00",
        }

        with patch.object(
            client,
            "_request",
            return_value=_DummyResponse(response_payload),
        ) as req_mock:
            client.submit_order(
                order,
                market=market,
                client_order_id="BOT-GUAR-run-001",
            )

        self.assertEqual(
            req_mock.call_args.kwargs["json"]["client_order_id"],
            "BOT-GUAR-run-001",
        )

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
        response_payload = {
            "order_id": "ord-2",
            "fill_count": "10.00",
            "remaining_count": "0.00",
        }

        with patch.object(client, "_request", return_value=_DummyResponse(response_payload)) as req_mock:
            response = client.submit_order(order, market=market)

        sent_payload = req_mock.call_args.kwargs["json"]
        # Buying NO at 0.60 is selling YES at 0.40, i.e. an ``ask`` on the YES book.
        self.assertEqual(sent_payload["side"], "ask")
        self.assertEqual(sent_payload["count"], "10")
        self.assertEqual(sent_payload["price"], "0.4000")
        self.assertNotIn("yes_price", sent_payload)
        self.assertNotIn("no_price", sent_payload)
        self.assertEqual(response.id, "ord-2")
        # Fully filled order reports executed.
        self.assertEqual(response.status, "executed")

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
        response_payload = {
            "order_id": "ord-6",
            "fill_count": "15.00",
            "remaining_count": "0.00",
        }

        with patch.object(client, "_request", return_value=_DummyResponse(response_payload)) as req_mock:
            response = client.submit_order(order, market=market, retry_suffix="fb")

        sent_payload = req_mock.call_args.kwargs["json"]
        self.assertEqual(sent_payload["time_in_force"], "fill_or_kill")
        # Marketable buy YES crosses the spread with an aggressive YES bid.
        self.assertEqual(sent_payload["side"], "bid")
        self.assertEqual(sent_payload["price"], "0.9700")
        self.assertEqual(sent_payload["count"], "6")
        self.assertEqual(sent_payload["self_trade_prevention_type"], "taker_at_cross")
        self.assertNotIn("type", sent_payload)
        self.assertTrue(sent_payload["client_order_id"].endswith("-fb"))
        self.assertEqual(response.id, "ord-6")
        self.assertEqual(response.raw["client_price"], 0.97)
        self.assertEqual(response.raw["client_requested_notional_usdc"], 5.82)

    def _client_with_bet_floor(self, min_bet: float, max_bet: float) -> KalshiClient:
        with patch.object(KalshiClient, "_load_private_key", return_value=_DummyPrivateKey()):
            return KalshiClient(
                base_url="https://api.example/trade-api/v2",
                api_key_id="test-key",
                private_key_path="unused.pem",
                min_bet_usdc=min_bet,
                max_bet_usdc=max_bet,
            )

    def test_submit_order_floors_count_within_approved_amount(self) -> None:
        client = self._client_with_bet_floor(2.0, 12.0)
        market = Market(
            id="MKT-MIN",
            question="Question",
            outcomes=[{"name": "YES", "price": 0.37}, {"name": "NO", "price": 0.63}],
        )
        order = OrderRequest(
            market_id="MKT-MIN",
            outcome="YES",
            amount_usdc=2.0,
            side="BUY",
        )
        response_payload = {"order_id": "ord-min", "fill_count": "5.00", "remaining_count": "0.00"}

        with patch.object(client, "_request", return_value=_DummyResponse(response_payload)) as req_mock:
            response = client.submit_order(order, market=market)

        sent_payload = req_mock.call_args.kwargs["json"]
        # The execution pipeline approved $2.00, so 6 contracts ($2.22) are
        # forbidden even though 5 contracts ($1.85) land below MIN_BET_USDC.
        self.assertEqual(sent_payload["count"], "5")
        self.assertEqual(sent_payload["price"], "0.3700")
        self.assertEqual(response.raw["client_amount_usdc"], 2.0)
        self.assertEqual(response.raw["client_requested_notional_usdc"], 1.85)

    def test_submit_order_caps_count_at_configured_max_bet(self) -> None:
        client = self._client_with_bet_floor(2.0, 2.0)
        market = Market(
            id="MKT-MINCAP",
            question="Question",
            outcomes=[{"name": "YES", "price": 0.59}, {"name": "NO", "price": 0.41}],
        )
        order = OrderRequest(
            market_id="MKT-MINCAP",
            outcome="YES",
            amount_usdc=5.0,
            side="BUY",
        )
        response_payload = {"order_id": "ord-cap", "fill_count": "3.00", "remaining_count": "0.00"}

        with patch.object(client, "_request", return_value=_DummyResponse(response_payload)) as req_mock:
            client.submit_order(order, market=market)

        sent_payload = req_mock.call_args.kwargs["json"]
        # The request exceeds MAX_BET_USDC, so the final integer count is based
        # on the tighter $2.00 cap: floor($2.00 / $0.59) = 3 contracts.
        self.assertEqual(sent_payload["count"], "3")

    def test_submit_order_rejects_budget_below_one_contract(self) -> None:
        client = self._client_with_bet_floor(0.0, 12.0)
        market = Market(
            id="MKT-SMALL",
            question="Question",
            outcomes=[{"name": "YES", "price": 0.59}, {"name": "NO", "price": 0.41}],
        )
        order = OrderRequest(
            market_id="MKT-SMALL",
            outcome="YES",
            amount_usdc=0.30,
            side="BUY",
        )

        with patch.object(client, "_request") as req_mock:
            with self.assertRaisesRegex(ValueError, "cannot fund one contract"):
                client.submit_order(order, market=market)

        req_mock.assert_not_called()

    def test_submit_order_fill_or_kill_zero_fill_maps_to_canceled(self) -> None:
        client = self._client()
        market = Market(
            id="MKT-6B",
            question="Question",
            outcomes=[{"name": "YES", "price": 0.40}, {"name": "NO", "price": 0.60}],
        )
        order = OrderRequest(
            market_id="MKT-6B",
            outcome="YES",
            amount_usdc=6.0,
            side="BUY",
            order_type="market",
        )
        # V2 fill-or-kill that finds no liquidity returns zero fills and no status.
        response_payload = {
            "order_id": "ord-6b",
            "fill_count": "0.00",
            "remaining_count": "15.00",
        }

        with patch.object(client, "_request", return_value=_DummyResponse(response_payload)):
            response = client.submit_order(order, market=market)

        self.assertEqual(response.id, "ord-6b")
        self.assertEqual(response.status, "canceled")

    def test_normalize_time_in_force_maps_legacy_values_to_valid_rest_values(self) -> None:
        self.assertEqual(_normalize_time_in_force(None), "good_till_canceled")
        self.assertEqual(_normalize_time_in_force("day"), "good_till_canceled")
        self.assertEqual(_normalize_time_in_force("gtc"), "good_till_canceled")
        self.assertEqual(_normalize_time_in_force("good_till_canceled"), "good_till_canceled")
        self.assertEqual(_normalize_time_in_force("ioc"), "immediate_or_cancel")
        self.assertEqual(_normalize_time_in_force("fok"), "fill_or_kill")

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

    def test_submit_order_attaches_response_body_on_http_error(self) -> None:
        client = self._client()
        market = Market(
            id="MKT-MI",
            question="Sports question",
            outcomes=[{"name": "YES", "price": 0.50}, {"name": "NO", "price": 0.50}],
        )
        order = OrderRequest(
            market_id="MKT-MI",
            outcome="YES",
            amount_usdc=5.0,
            side="BUY",
        )
        body = (
            '{"error":{"code":"michigan_residents_are_not_currently_'
            'allowed_to_open_positions_in_Sports"}}'
        )
        response = _DummyHttpResponse(body, status_code=403)
        http_error = requests.exceptions.HTTPError(
            "403 Client Error: Forbidden for url: https://api.example/orders",
            response=response,
        )

        with patch.object(client, "_request", side_effect=http_error):
            with self.assertRaises(requests.exceptions.HTTPError) as raised:
                client.submit_order(order, market=market)
        self.assertEqual(getattr(raised.exception, "_kalshi_response_body", None), body)

    def _rate_limit_error(self) -> requests.exceptions.HTTPError:
        response = _DummyHttpResponse(
            '{"error":{"code":"too_many_requests","message":"too many requests"}}',
            status_code=429,
        )
        return requests.exceptions.HTTPError("429 Too Many Requests", response=response)

    def test_submit_order_retries_rate_limit_with_same_payload(self) -> None:
        client = self._client()
        market = Market(
            id="MKT-8",
            question="Question",
            outcomes=[{"name": "YES", "price": 0.50}, {"name": "NO", "price": 0.50}],
        )
        order = OrderRequest(
            market_id="MKT-8",
            outcome="YES",
            amount_usdc=5.0,
            side="BUY",
        )
        success = _DummyResponse({"order": {"order_id": "ord-8", "status": "resting"}})

        with patch("kalshi_client.time.sleep") as sleep_mock:
            with patch.object(
                client,
                "_request",
                side_effect=[self._rate_limit_error(), success],
            ) as req_mock:
                response = client.submit_order(order, market=market)

        self.assertEqual(response.id, "ord-8")
        self.assertEqual(req_mock.call_count, 2)
        first_payload = req_mock.call_args_list[0].kwargs["json"]
        second_payload = req_mock.call_args_list[1].kwargs["json"]
        # Idempotent retry: a 429 was never accepted, so the identical
        # client_order_id must be reused.
        self.assertEqual(first_payload, second_payload)
        sleep_mock.assert_called_once()

    def test_submit_order_rate_limit_exhausts_retries_and_raises(self) -> None:
        client = self._client()
        market = Market(
            id="MKT-9",
            question="Question",
            outcomes=[{"name": "YES", "price": 0.50}, {"name": "NO", "price": 0.50}],
        )
        order = OrderRequest(
            market_id="MKT-9",
            outcome="YES",
            amount_usdc=5.0,
            side="BUY",
        )

        with patch("kalshi_client.time.sleep"):
            with patch.object(
                client,
                "_request",
                side_effect=[
                    self._rate_limit_error(),
                    self._rate_limit_error(),
                    self._rate_limit_error(),
                ],
            ) as req_mock:
                with self.assertRaises(requests.exceptions.HTTPError):
                    client.submit_order(order, market=market)

        self.assertEqual(req_mock.call_count, 3)


if __name__ == "__main__":
    unittest.main()
