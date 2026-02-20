import unittest
from datetime import datetime, timedelta, timezone

from main import _calculate_bet, _filter_markets
from models import Market


class TestMainUtils(unittest.TestCase):
    def test_filter_markets(self) -> None:
        markets = [
            Market(id="1", question="Q1", liquidity_usdc=50, category="sports"),
            Market(id="2", question="Q2", liquidity_usdc=150, category="sports"),
            Market(id="3", question="Q3", liquidity_usdc=200, category="politics"),
        ]
        filtered = _filter_markets(
            markets,
            min_liquidity=100,
            allowlist=("sports",),
            blocklist=("politics",),
        )
        self.assertEqual([m.id for m in filtered], ["2"])

    def test_filter_markets_by_close_date(self) -> None:
        now = datetime.now(timezone.utc)
        markets = [
            Market(id="1", question="Q1", close_time=now + timedelta(hours=6)),
            Market(id="2", question="Q2", close_time=now + timedelta(days=3)),
            Market(id="3", question="Q3", close_time=now + timedelta(days=10)),
            Market(id="4", question="Q4", close_time=None),
        ]
        # Filter: only markets closing between 1 and 7 days from now
        filtered = _filter_markets(
            markets,
            min_liquidity=0,
            allowlist=(),
            blocklist=(),
            min_close_days=1,
            max_close_days=7,
        )
        # Market 1 closes too soon (<1 day), Market 3 closes too far (>7 days)
        # Market 4 has no close_time, so it passes (no filter applied)
        self.assertEqual([m.id for m in filtered], ["2", "4"])

    def test_filter_markets_max_close_days_only(self) -> None:
        now = datetime.now(timezone.utc)
        markets = [
            Market(id="1", question="Q1", close_time=now + timedelta(hours=12)),
            Market(id="2", question="Q2", close_time=now + timedelta(days=5)),
        ]
        # Only set max_close_days (markets closing within 3 days)
        filtered = _filter_markets(
            markets,
            min_liquidity=0,
            allowlist=(),
            blocklist=(),
            max_close_days=3,
        )
        self.assertEqual([m.id for m in filtered], ["1"])

    def test_calculate_bet(self) -> None:
        self.assertEqual(_calculate_bet(100, 0.5), 50)
        self.assertEqual(_calculate_bet(100, -1), 0)
        self.assertEqual(_calculate_bet(100, 2), 100)


if __name__ == "__main__":
    unittest.main()
