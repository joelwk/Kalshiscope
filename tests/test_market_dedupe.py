from __future__ import annotations

from models import Market
from main import _dedupe_markets_by_matchup


def test_dedupe_matchups_keeps_lowest_id() -> None:
    markets = [
        Market(id="11795", question="NHL: Avalanche vs. Lightning"),
        Market(id="11595", question="NHL: Lightning vs. Avalanche"),
        Market(id="20000", question="Crypto: Will BTC be above 100k?"),
    ]

    deduped = _dedupe_markets_by_matchup(markets)

    ids = {market.id for market in deduped}
    assert len(deduped) == 2
    assert "11595" in ids
    assert "11795" not in ids
