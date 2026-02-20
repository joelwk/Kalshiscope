from predictbase_client import _parse_market


def test_parse_predictbase_snapshot(predictbase_markets_snapshot) -> None:
    markets = [_parse_market(entry) for entry in predictbase_markets_snapshot["markets"]]

    assert len(markets) == 2
    assert markets[0].id == "mkt-1"
    assert markets[0].question
    assert markets[0].liquidity_usdc == 250.5
    assert markets[0].outcomes[0].name == "YES"
    assert markets[1].id == "mkt-2"
    assert markets[1].category == "sports"
