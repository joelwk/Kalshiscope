from dataclasses import replace

import pytest

import main
from bootstrap_checks import BootstrapError, run_bootstrap_checks
from models import OrderResponse


class DummyGrok:
    def __init__(self, decision):
        self._decision = decision

    def analyze_market(self, market, search_config=None, previous_analysis=None, **kwargs):
        return self._decision

    def analyze_market_deep(
        self,
        market,
        previous_analysis=None,
        search_config=None,
        **kwargs,
    ):
        return self._decision


class DummyKalshi:
    def __init__(self, markets):
        self._markets = markets
        self.submitted = False
        self.last_fetch_pages = 1
        self.last_fetch_cap_hit = False
        self.last_fetch_mve_filter = None

    def get_markets(
        self,
        *,
        close_time_start=None,
        close_time_end=None,
        mve_filter=None,
    ):
        _ = close_time_start, close_time_end
        self.last_fetch_mve_filter = mve_filter
        return self._markets

    def reset_session(self):
        return None

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


def test_bootstrap_check_fails_fast_on_missing_certifi_bundle(monkeypatch) -> None:
    import certifi
    monkeypatch.setattr(certifi, "where", lambda: "/nonexistent/path/cacert.pem")
    with pytest.raises(BootstrapError, match="TLS CA certificate bundle not found"):
        run_bootstrap_checks()


def test_bootstrap_check_passes_when_cert_exists(monkeypatch, tmp_path) -> None:
    import certifi
    cert_file = tmp_path / "cacert.pem"
    cert_file.write_text("dummy")
    monkeypatch.setattr(certifi, "where", lambda: str(cert_file))
    run_bootstrap_checks(skip_api_checks=True)


def test_cycle_receipt_contains_forensic_keys(
    monkeypatch, sample_market, sample_decision, dummy_settings
) -> None:
    """Verify the new forensic cycle-receipt keys are emitted during a smoke cycle."""
    import json
    from dataclasses import replace

    captured_receipts: list[dict] = []
    original_info = main.logger.info

    def _capture_info(msg, *args, **kwargs):
        data = kwargs.get("data") or {}
        if isinstance(data, dict) and "cycle_receipt" in data:
            captured_receipts.append(data["cycle_receipt"])
        return original_info(msg, *args, **kwargs)

    monkeypatch.setattr(main.logger, "info", _capture_info)

    dummy_kalshi = DummyKalshi([sample_market])
    tuned = replace(
        dummy_settings,
        PRE_ANALYSIS_OPPORTUNITY_ENABLED=False,
    )
    monkeypatch.setattr(main, "load_settings", lambda: tuned)
    monkeypatch.setattr(main, "GrokClient", lambda *a, **kw: DummyGrok(sample_decision))
    monkeypatch.setattr(main, "KalshiClient", lambda *a, **kw: dummy_kalshi)

    def _stop_sleep(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(main.time, "sleep", _stop_sleep)

    with pytest.raises(KeyboardInterrupt):
        main.main()

    assert len(captured_receipts) >= 1, "Should capture at least one cycle receipt"
    receipt = captured_receipts[0]
    assert "top_candidates_summary" in receipt
    assert "confidence_bucket_decision_counts" in receipt
    assert "pre_analysis_research_routed_count" in receipt
    assert isinstance(receipt["top_candidates_summary"], list)
    assert isinstance(receipt["confidence_bucket_decision_counts"], dict)
    assert isinstance(receipt["pre_analysis_research_routed_count"], int)
