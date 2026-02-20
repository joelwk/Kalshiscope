import unittest
from datetime import datetime
from types import SimpleNamespace

from models import Market, MarketOutcome, OrderRequest
from predictbase_client import PredictBaseClient, _parse_market, _parse_onchain_payload


class FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class TestPredictBaseClient(unittest.TestCase):
    def test_parse_market(self) -> None:
        raw = {
            "market_id": "123",
            "title": "Will it rain?",
            "options": [
                {"label": "YES", "odds": "0.6", "price": "0.6"},
                "NO",
            ],
            "liquidity": "150",
            "tags": ["weather"],
            "closeTime": "2026-01-31T12:00:00",
            "url": "https://example.com/market/123",
        }
        market = _parse_market(raw)
        self.assertEqual(market.id, "123")
        self.assertEqual(market.question, "Will it rain?")
        self.assertEqual(len(market.outcomes), 2)
        self.assertEqual(market.outcomes[0].name, "YES")
        self.assertAlmostEqual(market.outcomes[0].odds, 0.6)
        self.assertAlmostEqual(market.liquidity_usdc, 150.0)
        self.assertEqual(market.category, "weather")
        self.assertIsInstance(market.close_time, datetime)

    def test_parse_market_predictbase_option_prices_micro_precision(self) -> None:
        raw = {
            "id": "123",
            "question": "Will BTC close above $100k in 2026?",
            "status": 0,
            "winningOption": "18446744073709551615",
            "optionTitles": ["Yes", "No"],
            "optionPrices": ["550000", "480000"],
            "volume": "123000000",
            "endsAt": "1735689600",
        }
        market = _parse_market(raw)
        self.assertEqual(market.id, "123")
        self.assertEqual(market.outcomes[0].name, "Yes")
        self.assertAlmostEqual(market.outcomes[0].price, 0.55)
        self.assertAlmostEqual(market.outcomes[1].price, 0.48)
        self.assertEqual(market.status, 0)
        self.assertEqual(market.winning_option_raw, "18446744073709551615")
        # volume is in 1e6 precision per docs
        self.assertAlmostEqual(market.liquidity_usdc, 123.0)

    def test_parse_onchain_payload(self) -> None:
        payload = {
            "onchain_payload": {"to": "0xabc", "data": "0x123", "value": 0}
        }
        onchain = _parse_onchain_payload(payload)
        self.assertIsNotNone(onchain)
        self.assertEqual(onchain.to, "0xabc")
        self.assertEqual(onchain.data, "0x123")

        self.assertIsNone(_parse_onchain_payload({"foo": "bar"}))

    def test_get_markets_skips_invalid(self) -> None:
        client = PredictBaseClient(base_url="https://api.example")
        good = {"id": "m1", "question": "Q1"}
        bad = {"id": "", "question": ""}
        client.session.get = lambda *args, **kwargs: FakeResponse({"markets": [good, bad]})

        markets = client.get_markets()
        self.assertEqual(len(markets), 1)
        self.assertEqual(markets[0].id, "m1")

    def test_submit_order_parses_payload(self) -> None:
        client = PredictBaseClient(base_url="https://api.example")
        payload = {
            "id": "order-1",
            "status": "open",
            "onchain_payload": {"to": "0xabc", "data": "0x123", "value": 0},
        }
        client.session.post = lambda *args, **kwargs: FakeResponse(payload)

        order = OrderRequest(market_id="m1", outcome="YES", amount_usdc=10)
        response = client.submit_order(order)
        self.assertEqual(response.id, "order-1")
        self.assertIsNotNone(response.onchain_payload)
        self.assertEqual(response.onchain_payload.to, "0xabc")

    def test_submit_order_applies_slippage_for_high_confidence(self) -> None:
        client = PredictBaseClient(
            base_url="https://api.example",
            slippage_confidence_threshold=0.7,
        )
        captured = {}

        def _capture_post(url, json, timeout):
            captured["payload"] = json
            return FakeResponse({"ok": True})

        client.session.post = _capture_post

        market = Market(
            id="m1",
            question="Q1",
            outcomes=[MarketOutcome(name="YES", price=0.5)],
        )
        order = OrderRequest(
            market_id="m1",
            outcome="YES",
            amount_usdc=10,
            confidence=0.8,
        )
        client.submit_order(order, market=market, slippage_pct=0.02)

        price = captured["payload"]["order"]["price"]
        self.assertAlmostEqual(price, 0.51)


if __name__ == "__main__":
    unittest.main()
