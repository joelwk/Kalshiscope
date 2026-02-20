import pytest

import main
from models import OrderResponse


class DummyGrok:
    def __init__(self, decision):
        self._decision = decision

    def analyze_market(self, market):
        return self._decision


class DummyPredictBase:
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
    dummy_predictbase = DummyPredictBase([sample_market])

    monkeypatch.setattr(main, "load_settings", lambda: dummy_settings)
    monkeypatch.setattr(main, "GrokClient", lambda *args, **kwargs: DummyGrok(sample_decision))
    monkeypatch.setattr(
        main,
        "PredictBaseClient",
        lambda *args, **kwargs: dummy_predictbase,
    )

    def _stop_sleep(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(main.time, "sleep", _stop_sleep)

    with pytest.raises(KeyboardInterrupt):
        main.main()

    assert dummy_predictbase.submitted is False
