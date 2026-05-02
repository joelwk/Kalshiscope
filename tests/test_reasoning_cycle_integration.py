from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import pytest

from config import Settings
from logging_config import get_correlation_id, set_correlation_id
from main import _analyze_market_candidate, _analyze_market_candidate_for_worker
from models import Market, MarketOutcome, MarketState, TradeDecision


class DummyGrokClient:
    def __init__(self, analyze_decision: TradeDecision, deep_decision: TradeDecision | None = None) -> None:
        self.analyze_decision = analyze_decision
        self.deep_decision = deep_decision or analyze_decision
        self.seen_correlation_id: str | None = None
        self.last_search_config = None

    def analyze_market(self, market, search_config=None, previous_analysis=None):
        self.seen_correlation_id = get_correlation_id()
        self.last_search_config = search_config
        return self.analyze_decision

    def analyze_market_deep(self, market, previous_analysis=None, search_config=None):
        return self.deep_decision


def _market() -> Market:
    return Market(
        id="m1",
        question="Will Team A win?",
        outcomes=[MarketOutcome(name="Team A", price=0.55), MarketOutcome(name="Team B", price=0.45)],
        liquidity_usdc=300.0,
    )


def _settings() -> Settings:
    return Settings(
        XAI_API_KEY="xai-key",
        KALSHI_API_KEY_ID="kalshi-key-id",
        KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
    )


def test_analyze_market_candidate_end_to_end_no_refinement() -> None:
    decision = TradeDecision(
        should_trade=True,
        outcome="Team A",
        confidence=0.9,
        bet_size_pct=0.4,
        reasoning="Implied prob: 55%, My prob: 90%, Edge: 35%",
        implied_prob_external=0.55,
        my_prob=0.9,
        edge_external=0.35,
        evidence_quality=1.0,
    )
    result = _analyze_market_candidate(
        market=_market(),
        state=None,
        anchor_analysis=None,
        settings=_settings(),
        grok_client=DummyGrokClient(decision),
    )
    assert result["decision"].outcome == "Team A"
    assert result["decision"].confidence == pytest.approx(0.6364)
    assert result["confidence_calibration_applied"] is True
    assert result["decision"].bet_size_pct > 0
    assert result["was_refined"] is False


def test_analyze_market_candidate_refines_and_blocks_flip_when_weak() -> None:
    initial = TradeDecision(
        should_trade=True,
        outcome="Team A",
        confidence=0.66,
        bet_size_pct=0.4,
        reasoning="Implied prob: 55%, My prob: 66%, Edge: 11%",
        implied_prob_external=0.55,
        my_prob=0.66,
        edge_external=0.11,
    )
    deep = TradeDecision(
        should_trade=True,
        outcome="Team B",
        confidence=0.67,
        bet_size_pct=0.4,
        reasoning="Implied prob: 45%, My prob: 67%, Edge: 22%",
        implied_prob_external=0.45,
        my_prob=0.67,
        edge_external=0.22,
    )
    state = MarketState(
        market_id="m1",
        analysis_count=2,
        last_confidence=0.70,
    )
    result = _analyze_market_candidate(
        market=_market(),
        state=state,
        anchor_analysis={"outcome": "Team A", "confidence": 0.70},
        settings=_settings(),
        grok_client=DummyGrokClient(initial, deep),
    )
    assert result["was_refined"] is True
    # Flip guard should preserve the anchor side in weak improvement cases.
    assert result["decision"].outcome == "Team A"


def test_worker_thread_propagates_cycle_correlation_id() -> None:
    parent_correlation_id = set_correlation_id("cycleabcd")
    decision = TradeDecision(
        should_trade=True,
        outcome="Team A",
        confidence=0.9,
        bet_size_pct=0.4,
        reasoning="Implied prob: 55%, My prob: 90%, Edge: 35%",
        implied_prob_external=0.55,
        my_prob=0.9,
        edge_external=0.35,
    )
    worker_client = DummyGrokClient(decision)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _analyze_market_candidate_for_worker,
            _market(),
            None,
            None,
            _settings(),
            worker_client,
            None,
            parent_correlation_id,
        )
        _ = future.result()
    assert worker_client.seen_correlation_id == parent_correlation_id


def test_worker_thread_supports_extended_research_flag() -> None:
    decision = TradeDecision(
        should_trade=True,
        outcome="Team A",
        confidence=0.9,
        bet_size_pct=0.4,
        reasoning="Implied prob: 55%, My prob: 90%, Edge: 35%",
        implied_prob_external=0.55,
        my_prob=0.9,
        edge_external=0.35,
    )
    worker_client = DummyGrokClient(decision)
    result = _analyze_market_candidate_for_worker(
        _market(),
        None,
        None,
        _settings(),
        worker_client,
        None,
        "cycle-xyz",
        True,
    )
    assert result["used_extended_research"] is True
    assert worker_client.last_search_config is not None
    assert (worker_client.last_search_config.lookback_hours or 0) > 0
