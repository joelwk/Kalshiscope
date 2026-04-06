from dataclasses import replace

import pytest

import main
from models import OrderResponse


class DummyGrok:
    def __init__(self, decision):
        self._decision = decision

    def analyze_market(self, market, search_config=None, previous_analysis=None):
        return self._decision

    def analyze_market_deep(
        self,
        market,
        previous_analysis=None,
        search_config=None,
    ):
        return self._decision


class DummyKalshi:
    def __init__(self, markets):
        self._markets = markets
        self.submitted = False

    def get_markets(self):
        return self._markets

    def submit_order(self, order, **kwargs):
        self.submitted = True
        return OrderResponse(id="order-1", status="open")


def test_bot_smoke_dry_run(
    monkeypatch, sample_market, sample_decision, dummy_settings
) -> None:
    dummy_kalshi = DummyKalshi([sample_market])

    monkeypatch.setattr(main, "load_settings", lambda: dummy_settings)
    monkeypatch.setattr(main, "GrokClient", lambda *args, **kwargs: DummyGrok(sample_decision))
    monkeypatch.setattr(
        main,
        "KalshiClient",
        lambda *args, **kwargs: dummy_kalshi,
    )

    def _stop_sleep(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(main.time, "sleep", _stop_sleep)

    with pytest.raises(KeyboardInterrupt):
        main.main()

    assert dummy_kalshi.submitted is False


def test_bot_smoke_parallel_analysis_dry_run(
    monkeypatch, sample_market, sample_decision, dummy_settings
) -> None:
    second_market = sample_market.model_copy(update={"id": "m2", "question": "Will it snow?"})
    dummy_kalshi = DummyKalshi([sample_market, second_market])
    tuned_settings = replace(
        dummy_settings,
        PARALLEL_ANALYSIS_ENABLED=True,
        ANALYSIS_MAX_WORKERS=2,
        PRE_ORDER_MARKET_REFRESH=True,
        ORDERBOOK_PRECHECK_ENABLED=True,
        ORDERBOOK_PRECHECK_MIN_CONFIDENCE=0.5,
    )

    monkeypatch.setattr(main, "load_settings", lambda: tuned_settings)
    monkeypatch.setattr(main, "GrokClient", lambda *args, **kwargs: DummyGrok(sample_decision))
    monkeypatch.setattr(
        main,
        "KalshiClient",
        lambda *args, **kwargs: dummy_kalshi,
    )

    def _stop_sleep(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(main.time, "sleep", _stop_sleep)

    with pytest.raises(KeyboardInterrupt):
        main.main()

    assert dummy_kalshi.submitted is False
