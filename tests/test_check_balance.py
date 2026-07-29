import pytest

import check_balance


class StubKalshiClient:
    def __init__(self, positions_payload: dict[str, object]) -> None:
        self.positions_payload = positions_payload

    def get_balance(self) -> float:
        return 42.50

    def get_positions(self) -> dict[str, object]:
        return self.positions_payload


@pytest.mark.parametrize(
    "positions_key",
    ("market_positions", "positions", "portfolio_positions", "data"),
)
def test_show_balance_counts_supported_position_response_keys(
    positions_key: str,
    monkeypatch,
    capsys,
) -> None:
    client = StubKalshiClient({positions_key: [{"ticker": "TEST-MARKET"}]})
    monkeypatch.setattr(check_balance, "get_client", lambda: client)

    check_balance.show_balance_and_positions()

    assert "Open market positions: 1" in capsys.readouterr().out


def test_extract_market_positions_skips_empty_leading_keys() -> None:
    payload = {
        "market_positions": [],
        "positions": [{"ticker": "FALLBACK-MARKET"}],
    }

    rows = check_balance._extract_market_positions(payload)

    assert rows == [{"ticker": "FALLBACK-MARKET"}]


def test_extract_market_positions_returns_empty_when_all_keys_empty() -> None:
    payload = {
        "market_positions": [],
        "positions": [],
        "portfolio_positions": [],
        "data": [],
    }

    assert check_balance._extract_market_positions(payload) == []
