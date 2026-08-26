from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
import requests

import main
from kalshi_client import PortfolioBalance
from market_state import MarketStateManager
from models import Market, MarketOutcome, OrderResponse, TradeDecision


class _GuaranteedGrok:
    def __init__(self, decision: TradeDecision) -> None:
        self.decision = decision
        self.initial_calls: list[str] = []
        self.deep_calls: list[str] = []

    def analyze_market(self, market, **kwargs):
        self.initial_calls.append(market.id)
        return self.decision

    def analyze_market_deep(self, market, **kwargs):
        self.deep_calls.append(market.id)
        return self.decision


class _GuaranteedKalshi:
    def __init__(self, markets: list[Market]) -> None:
        self.markets = markets
        self.submitted_market_ids: list[str] = []
        self.last_fetch_pages = 1
        self.last_fetch_cap_hit = False
        self.last_fetch_mve_filter = None
        # Bankroll-derived bet bounds are refreshed on the client each cycle.
        self.min_bet_usdc = 0.0
        self.max_bet_usdc = 0.0

    def get_markets(self, **kwargs):
        self.last_fetch_mve_filter = kwargs.get("mve_filter")
        return self.markets

    def get_portfolio_balance(self):
        return PortfolioBalance(
            available_balance=50.0,
            position_value=25.0,
            total_portfolio_value=75.0,
            raw_payload={},
        )

    def reset_session(self):
        return None

    def submit_order(self, order, **kwargs):
        self.submitted_market_ids.append(order.market_id)
        raise AssertionError("dry-run guarantee must not call submit_order")


class _LiveGuaranteedKalshi(_GuaranteedKalshi):
    def __init__(self, markets: list[Market]) -> None:
        super().__init__(markets)
        self.client_order_ids: list[str | None] = []

    def get_market(self, market_id: str) -> Market:
        return next(market for market in self.markets if market.id == market_id)

    def get_market_orderbook(self, market_id: str) -> dict:
        return {}

    def submit_order(self, order, **kwargs):
        self.submitted_market_ids.append(order.market_id)
        self.client_order_ids.append(kwargs.get("client_order_id"))
        return OrderResponse(
            id=f"order-{order.market_id}",
            status="open",
            raw={
                "client_qty_shares": 9,
                "client_price": 0.56,
                "fill_count": 0,
            },
        )


class _JurisdictionThenSuccessKalshi(_LiveGuaranteedKalshi):
    def submit_order(self, order, **kwargs):
        self.submitted_market_ids.append(order.market_id)
        self.client_order_ids.append(kwargs.get("client_order_id"))
        market = next(item for item in self.markets if item.id == order.market_id)
        if main.market_family(market) == "sports":
            raise RuntimeError(
                "Michigan_residents_are_not_currently_allowed_to_open_positions_in_Sports"
            )
        return OrderResponse(
            id=f"order-{order.market_id}",
            status="open",
            raw={
                "client_qty_shares": 9,
                "client_price": 0.56,
                "fill_count": 0,
            },
        )


class _NevadaRestrictedThenSuccessKalshi(_LiveGuaranteedKalshi):
    def submit_order(self, order, **kwargs):
        self.submitted_market_ids.append(order.market_id)
        self.client_order_ids.append(kwargs.get("client_order_id"))
        market = next(item for item in self.markets if item.id == order.market_id)
        family = main.market_family(market)
        category = str(market.category or "").strip().lower()
        if family in {"sports", "entertainment", "music", "politics"} or category in {
            "entertainment",
            "sports",
            "elections",
            "politics",
        }:
            raise RuntimeError(
                "Nevada_residents_are_not_currently_allowed_to_open_positions_in_"
                "Sports,_Elections_and_Entertainment. Check your email for more details."
            )
        return OrderResponse(
            id=f"order-{order.market_id}",
            status="open",
            raw={
                "client_qty_shares": 9,
                "client_price": 0.56,
                "fill_count": 0,
            },
        )


class _MarketNotFoundThenSuccessKalshi(_LiveGuaranteedKalshi):
    def submit_order(self, order, **kwargs):
        if order.market_id.startswith("dead-"):
            self.submitted_market_ids.append(order.market_id)
            self.client_order_ids.append(kwargs.get("client_order_id"))
            raise _http_market_not_found()
        return super().submit_order(order, **kwargs)


def _market(
    market_id: str,
    *,
    liquidity: float = 200.0,
    category: str = "politics",
    event_ticker: str | None = None,
    yes_price: float = 0.55,
    status: str = "open",
    close_in: timedelta | None = None,
) -> Market:
    no_price = round(max(0.01, min(0.99, 1.0 - yes_price)), 2)
    return Market(
        id=market_id,
        event_ticker=event_ticker or f"EVENT-{market_id}",
        question=f"Will {market_id} happen?",
        outcomes=[
            MarketOutcome(name="YES", price=yes_price),
            MarketOutcome(name="NO", price=no_price),
        ],
        liquidity_usdc=liquidity,
        volume_24h=100.0,
        open_interest=100.0,
        category=category,
        status=status,
        close_time=datetime.now(timezone.utc) + (close_in or timedelta(days=2)),
    )


def _http_market_not_found() -> requests.exceptions.HTTPError:
    response = requests.models.Response()
    response.status_code = 404
    response._content = (
        b'{"error":{"code":"market_not_found","message":"market not found"}}'
    )
    exc = requests.exceptions.HTTPError(
        "404 Client Error: Not Found for url: "
        "https://api.elections.kalshi.com/trade-api/v2/portfolio/events/orders",
        response=response,
    )
    setattr(exc, "_kalshi_response_body", response.text)
    return exc


def _decision(
    *,
    evidence_basis: str = "proxy",
    edge_source: str = "computed",
    evidence_quality: float = 0.8,
    confidence: float = 0.72,
    my_prob: float | None = None,
    outcome: str = "YES",
    should_trade: bool = False,
    primary_source_url: str | None = "https://example.com/source",
    edge_mechanism: str | None = None,
) -> TradeDecision:
    yes_prob = (
        float(my_prob)
        if my_prob is not None
        else (confidence if outcome == "YES" else max(0.0, 1.0 - confidence))
    )
    return TradeDecision(
        should_trade=should_trade,
        outcome=outcome,
        probability_yes=yes_prob,
        confidence=confidence,
        my_prob=yes_prob,
        bet_size_pct=0.0,
        reasoning="Deep evidence favors YES.",
        evidence_quality=evidence_quality,
        evidence_basis=evidence_basis,
        edge_source=edge_source,
        edge_mechanism=edge_mechanism,
        primary_source_url=primary_source_url,
        abstain=not should_trade,
        prompt_tokens=10,
        completion_tokens=5,
        reasoning_tokens=2,
        cached_tokens=1,
    )


def test_lock_guaranteed_markets_locks_exact_target_and_honors_exclusions() -> None:
    settings = main.Settings(GUARANTEED_ORDERS_N=2)
    plan = main.GuaranteedOrderPlan(target=2)
    markets = [_market("low", liquidity=100.0), _market("high", liquidity=500.0), _market("excluded", liquidity=900.0)]

    locked = main._lock_guaranteed_order_markets(
        plan,
        markets,
        excluded_market_ids={"excluded"},
        settings=settings,
        cycle_number=1,
    )

    assert len(locked) == 2
    assert plan.is_fully_locked
    assert [slot.market_id for slot in plan.slots] == ["high", "low"]
    assert main._lock_guaranteed_order_markets(
        plan,
        markets,
        excluded_market_ids=set(),
        settings=settings,
        cycle_number=2,
    ) == []
    assert len(plan.slots) == 2


def test_lock_guaranteed_markets_excludes_known_unexecutable_family() -> None:
    settings = main.Settings(GUARANTEED_ORDERS_N=1)
    plan = main.GuaranteedOrderPlan(target=1)
    markets = [
        _market("sports-high", liquidity=900.0, category="mlb"),
        _market("politics-low", liquidity=100.0),
    ]

    locked = main._lock_guaranteed_order_markets(
        plan,
        markets,
        excluded_market_ids=set(),
        excluded_market_families={"sports"},
        settings=settings,
        cycle_number=1,
    )

    assert [slot.market_id for slot in locked] == ["politics-low"]
    assert plan.is_fully_locked


def test_lock_guaranteed_markets_skips_soon_to_close() -> None:
    settings = main.Settings(GUARANTEED_ORDERS_N=1)
    plan = main.GuaranteedOrderPlan(target=1)
    soon = _market("soon-high", liquidity=900.0, close_in=timedelta(minutes=15))
    later = _market("later-low", liquidity=100.0, close_in=timedelta(days=2))

    locked = main._lock_guaranteed_order_markets(
        plan,
        [soon, later],
        excluded_market_ids=set(),
        settings=settings,
        cycle_number=1,
    )

    assert [slot.market_id for slot in locked] == ["later-low"]


def test_lock_retires_closed_locked_market_and_fills_from_catalog() -> None:
    settings = main.Settings(GUARANTEED_ORDERS_N=1)
    plan = main.GuaranteedOrderPlan(target=1)
    live = _market("was-live", liquidity=900.0)
    replacement = _market("still-open", liquidity=100.0)
    main._lock_guaranteed_order_markets(
        plan,
        [live],
        excluded_market_ids=set(),
        settings=settings,
        cycle_number=1,
    )
    assert plan.slots[0].market_id == "was-live"

    closed = _market("was-live", liquidity=900.0, status="closed")
    newly = main._lock_guaranteed_order_markets(
        plan,
        [closed, replacement],
        excluded_market_ids=set(),
        settings=settings,
        cycle_number=2,
    )

    assert [slot.market_id for slot in newly] == ["still-open"]
    assert plan.slots[0].market_id == "still-open"
    assert "was-live" in plan.retired_market_ids


def test_unexecutable_market_error_detects_kalshi_gone_tickers() -> None:
    assert main._is_unexecutable_market_error(
        'HTTPError: 404 Client Error: Not Found\n{"error":{"code":"market_not_found"}}'
    )
    assert main._is_unexecutable_market_error(
        "MarketClosedError: Market closed before order submission"
    )
    assert not main._is_unexecutable_market_error(
        "Insufficient balance on Kalshi account"
    )
    assert not main._is_unexecutable_market_error("429 Too Many Requests")


def test_guaranteed_dry_run_forces_deep_side_and_counts_one_attempt(tmp_path) -> None:
    market = _market("forced")
    slot = main.GuaranteedOrderSlot(
        slot_number=1,
        market_id=market.id,
        market=market,
        locked_cycle=1,
        client_order_id="BOT-GUAR-test-001",
    )
    grok = _GuaranteedGrok(_decision())
    kalshi = _GuaranteedKalshi([market])
    state = MarketStateManager(str(tmp_path / "state.db"))
    try:
        result = main._attempt_guaranteed_order_slot(
            slot,
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
            settings=main.Settings(
                DRY_RUN=True,
                GUARANTEED_ORDERS_N=1,
            ),
            min_bet_usdc=5.0,
            max_bet_usdc=12.0,
        )
    finally:
        state.close()

    assert result.status == "dry_run"
    assert result.decision is not None
    assert result.decision.should_trade is True
    assert result.decision.abstain is False
    assert result.decision.outcome == "YES"
    assert result.amount_usdc == 5.0
    assert grok.initial_calls == [market.id]
    assert grok.deep_calls == [market.id]
    assert slot.research_completed is True
    assert slot.submission_attempts == 1
    assert kalshi.submitted_market_ids == []
    assert result.token_usage == {
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "reasoning_tokens": 4,
        "cached_tokens": 2,
    }


def test_guaranteed_absence_only_is_replaceable_not_forced(tmp_path) -> None:
    market = _market("gap-forced")
    slot = main.GuaranteedOrderSlot(
        slot_number=1,
        market_id=market.id,
        market=market,
        locked_cycle=1,
        client_order_id="BOT-GUAR-gap-001",
    )
    grok = _GuaranteedGrok(
        _decision(evidence_basis="absence_only", edge_source="none", evidence_quality=0.2)
    )
    kalshi = _GuaranteedKalshi([market])
    state = MarketStateManager(str(tmp_path / "state.db"))
    try:
        result = main._attempt_guaranteed_order_slot(
            slot,
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
            settings=main.Settings(
                DRY_RUN=True,
                GUARANTEED_ORDERS_N=1,
            ),
            min_bet_usdc=5.0,
            max_bet_usdc=12.0,
        )
    finally:
        state.close()

    assert result.status == "research_gap_replaceable"
    assert result.error == "guaranteed_order_research_gap_absence_only"
    assert result.amount_usdc == 0.0
    assert slot.submission_attempts == 0
    assert kalshi.submitted_market_ids == []


def test_guaranteed_phase_defers_weak_evidence_without_replacement(tmp_path) -> None:
    gap_market = _market("gap-only", liquidity=900.0)
    gap_decision = _decision(
        evidence_basis="absence_only", edge_source="none", evidence_quality=0.1
    )

    grok = _GuaranteedGrok(gap_decision)
    kalshi = _LiveGuaranteedKalshi([gap_market])
    state = MarketStateManager(str(tmp_path / "state.db"))
    plan = main.GuaranteedOrderPlan(target=1, run_id="gap-force")
    decisions: list[dict] = []
    try:
        result = main._run_guaranteed_order_phase(
            plan=plan,
            markets=[gap_market],
            excluded_market_ids=set(),
            cycle_number=1,
            settings=main.Settings(
                DRY_RUN=False,
                GUARANTEED_ORDERS_N=1,
            ),
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
            min_bet_usdc=5.0,
            max_bet_usdc=12.0,
            log_decision=lambda **kwargs: decisions.append(kwargs),
            extended_research_market_ids=set(),
        )
    finally:
        state.close()

    assert result.completed == 0
    assert plan.is_complete is False
    assert plan.abandoned_count == 0
    assert plan.is_resolved is False
    assert plan.slots[0].needs_replacement is True
    assert plan.research_gap_replacements == 0
    assert kalshi.submitted_market_ids == []
    assert decisions[-1]["execution_audit"]["guaranteed_order_retry_pending"] is True
    assert plan.suppresses_normal_execution is True


def test_guaranteed_phase_fills_deferred_slot_from_next_cycle_plus_ev(
    tmp_path,
) -> None:
    gap_market = _market("gap-high", liquidity=900.0)
    good_market = _market("good-low", liquidity=100.0)
    gap_decision = _decision(
        evidence_basis="absence_only", edge_source="none", evidence_quality=0.1
    )
    good_decision = _decision(evidence_basis="proxy", evidence_quality=0.85)

    class _SelectiveGrok:
        def __init__(self) -> None:
            self.deep_calls: list[str] = []

        def analyze_market(self, market, **kwargs):
            return gap_decision if market.id == "gap-high" else good_decision

        def analyze_market_deep(self, market, **kwargs):
            self.deep_calls.append(market.id)
            return gap_decision if market.id == "gap-high" else good_decision

    grok = _SelectiveGrok()
    kalshi = _LiveGuaranteedKalshi([gap_market, good_market])
    state = MarketStateManager(str(tmp_path / "state.db"))
    plan = main.GuaranteedOrderPlan(target=1, run_id="gap-defer-next")
    settings = main.Settings(DRY_RUN=False, GUARANTEED_ORDERS_N=1)
    try:
        cycle_one = main._run_guaranteed_order_phase(
            plan=plan,
            markets=[gap_market],
            excluded_market_ids=set(),
            cycle_number=1,
            settings=settings,
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
            min_bet_usdc=5.0,
            max_bet_usdc=12.0,
            log_decision=lambda **kwargs: None,
            extended_research_market_ids=set(),
            priority_by_market_id={"gap-high": 0.40},
            seed_decisions_by_market_id={"gap-high": gap_decision},
        )
        assert cycle_one.completed == 0
        assert plan.slots[0].needs_replacement is True
        assert plan.is_resolved is False

        cycle_two = main._run_guaranteed_order_phase(
            plan=plan,
            markets=[gap_market, good_market],
            excluded_market_ids=set(),
            cycle_number=2,
            settings=settings,
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
            min_bet_usdc=5.0,
            max_bet_usdc=12.0,
            log_decision=lambda **kwargs: None,
            extended_research_market_ids=set(),
            priority_by_market_id={"good-low": 0.80, "gap-high": 0.10},
            seed_decisions_by_market_id={
                "good-low": good_decision,
                "gap-high": gap_decision,
            },
        )
    finally:
        state.close()

    assert cycle_two.completed == 1
    assert plan.is_complete
    assert plan.slots[0].market_id == "good-low"
    assert "gap-high" in plan.retired_market_ids
    assert kalshi.submitted_market_ids == ["good-low"]
    assert grok.deep_calls == ["gap-high"]


def test_guaranteed_phase_replaces_weak_evidence_when_alternate_exists(tmp_path) -> None:
    gap_market = _market("gap-high", liquidity=900.0)
    good_market = _market("good-low", liquidity=100.0)
    gap_decision = _decision(
        evidence_basis="absence_only", edge_source="none", evidence_quality=0.1
    )
    good_decision = _decision(evidence_basis="proxy", evidence_quality=0.85)

    class _SelectiveGrok:
        def __init__(self) -> None:
            self.initial_calls: list[str] = []
            self.deep_calls: list[str] = []

        def analyze_market(self, market, **kwargs):
            self.initial_calls.append(market.id)
            return gap_decision if market.id == "gap-high" else good_decision

        def analyze_market_deep(self, market, **kwargs):
            self.deep_calls.append(market.id)
            return gap_decision if market.id == "gap-high" else good_decision

    grok = _SelectiveGrok()
    kalshi = _LiveGuaranteedKalshi([gap_market, good_market])
    state = MarketStateManager(str(tmp_path / "state.db"))
    plan = main.GuaranteedOrderPlan(target=1, run_id="gap-replace")
    decisions: list[dict] = []
    try:
        result = main._run_guaranteed_order_phase(
            plan=plan,
            markets=[gap_market, good_market],
            excluded_market_ids=set(),
            cycle_number=1,
            settings=main.Settings(
                DRY_RUN=False,
                GUARANTEED_ORDERS_N=1,
            ),
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
            min_bet_usdc=5.0,
            max_bet_usdc=12.0,
            log_decision=lambda **kwargs: decisions.append(kwargs),
            extended_research_market_ids=set(),
        )
    finally:
        state.close()

    assert result.completed == 1
    assert plan.is_complete
    assert plan.slots[0].market_id == "good-low"
    assert "gap-high" in plan.retired_market_ids
    assert plan.research_gap_replacements == 1
    assert kalshi.submitted_market_ids == ["good-low"]
    assert grok.deep_calls == ["gap-high", "good-low"]


def test_guaranteed_phase_defers_analyzed_gap_instead_of_unanalyzed_liquid(
    tmp_path,
) -> None:
    gap_market = _market("analyzed-gap", liquidity=50.0)
    liquid = _market("unanalyzed-liquid", liquidity=900.0)
    gap_decision = _decision(
        evidence_basis="absence_only",
        edge_source="none",
        evidence_quality=0.1,
        confidence=0.80,
    )
    liquid_decision = _decision(evidence_basis="proxy", evidence_quality=0.85)

    class _SelectiveGrok:
        def __init__(self) -> None:
            self.initial_calls: list[str] = []
            self.deep_calls: list[str] = []

        def analyze_market(self, market, **kwargs):
            self.initial_calls.append(market.id)
            return gap_decision if market.id == "analyzed-gap" else liquid_decision

        def analyze_market_deep(self, market, **kwargs):
            self.deep_calls.append(market.id)
            return gap_decision if market.id == "analyzed-gap" else liquid_decision

    grok = _SelectiveGrok()
    kalshi = _LiveGuaranteedKalshi([gap_market, liquid])
    state = MarketStateManager(str(tmp_path / "state.db"))
    plan = main.GuaranteedOrderPlan(target=1, run_id="no-yeet")
    decisions: list[dict] = []
    try:
        result = main._run_guaranteed_order_phase(
            plan=plan,
            markets=[gap_market, liquid],
            excluded_market_ids=set(),
            cycle_number=1,
            settings=main.Settings(
                DRY_RUN=False,
                GUARANTEED_ORDERS_N=1,
            ),
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
            min_bet_usdc=5.0,
            max_bet_usdc=12.0,
            log_decision=lambda **kwargs: decisions.append(kwargs),
            extended_research_market_ids=set(),
            priority_by_market_id={"analyzed-gap": 0.50},
        )
    finally:
        state.close()

    assert result.completed == 0
    assert plan.abandoned_count == 0
    assert plan.slots[0].needs_replacement is True
    assert plan.slots[0].market_id == "analyzed-gap"
    assert plan.is_resolved is False
    assert kalshi.submitted_market_ids == []
    assert grok.deep_calls == ["analyzed-gap"]
    assert "analyzed-gap" in plan.retired_market_ids
    assert "unanalyzed-liquid" not in plan.retired_market_ids


def test_lock_guaranteed_markets_prefers_analyzed_confidence() -> None:
    settings = main.Settings(GUARANTEED_ORDERS_N=1)
    plan = main.GuaranteedOrderPlan(target=1)
    low_liq_confident = _market("confident", liquidity=50.0)
    high_liq_weak = _market("liquid", liquidity=900.0)

    locked = main._lock_guaranteed_order_markets(
        plan,
        [high_liq_weak, low_liq_confident],
        excluded_market_ids=set(),
        priority_by_market_id={"confident": 0.91, "liquid": 0.40},
        settings=settings,
        cycle_number=1,
    )

    assert [slot.market_id for slot in locked] == ["confident"]


def test_lock_analyzed_beats_unanalyzed_even_when_priority_negative() -> None:
    settings = main.Settings(GUARANTEED_ORDERS_N=1)
    plan = main.GuaranteedOrderPlan(target=1)
    analyzed = _market("analyzed", liquidity=50.0)
    liquid = _market("liquid", liquidity=900.0)

    locked = main._lock_guaranteed_order_markets(
        plan,
        [liquid, analyzed],
        excluded_market_ids=set(),
        priority_by_market_id={"analyzed": -0.20},
        settings=settings,
        cycle_number=1,
    )

    assert [slot.market_id for slot in locked] == ["analyzed"]


def test_lock_fills_remaining_slots_from_unanalyzed_catalog() -> None:
    settings = main.Settings(GUARANTEED_ORDERS_N=2)
    plan = main.GuaranteedOrderPlan(target=2)
    analyzed = _market("analyzed", liquidity=50.0)
    liquid = _market("liquid", liquidity=900.0)
    thin = _market("thin", liquidity=80.0)

    locked = main._lock_guaranteed_order_markets(
        plan,
        [thin, liquid, analyzed],
        excluded_market_ids=set(),
        priority_by_market_id={"analyzed": 0.40},
        settings=settings,
        cycle_number=1,
    )

    assert [slot.market_id for slot in locked] == ["analyzed", "liquid"]


def test_research_gap_replacement_stays_inside_analyzed_set() -> None:
    settings = main.Settings(GUARANTEED_ORDERS_N=1)
    plan = main.GuaranteedOrderPlan(target=1)
    plan.slots = [
        main.GuaranteedOrderSlot(
            slot_number=1,
            market_id="gap",
            market=_market("gap", liquidity=50.0),
            locked_cycle=1,
            client_order_id="BOT-GUAR-gap-001",
            needs_replacement=True,
            replacement_reason="guaranteed_order_research_gap_absence_only",
        )
    ]
    plan.retired_market_ids.add("gap")
    next_analyzed = _market("next-analyzed", liquidity=40.0)
    unanalyzed_liquid = _market("unanalyzed-liquid", liquidity=900.0)

    locked = main._lock_guaranteed_order_markets(
        plan,
        [unanalyzed_liquid, next_analyzed, _market("gap", liquidity=50.0)],
        excluded_market_ids=set(),
        priority_by_market_id={"gap": 0.10, "next-analyzed": 0.05},
        require_cycle_analysis=True,
        settings=settings,
        cycle_number=1,
    )

    assert [slot.market_id for slot in locked] == ["next-analyzed"]
    assert plan.slots[0].market_id == "next-analyzed"


def test_research_gap_replacement_skips_unanalyzed_when_none_analyzed_remain() -> None:
    settings = main.Settings(GUARANTEED_ORDERS_N=1)
    plan = main.GuaranteedOrderPlan(target=1)
    gap_market = _market("gap", liquidity=50.0)
    plan.slots = [
        main.GuaranteedOrderSlot(
            slot_number=1,
            market_id="gap",
            market=gap_market,
            locked_cycle=1,
            client_order_id="BOT-GUAR-gap-001",
            needs_replacement=True,
            replacement_reason="guaranteed_order_research_gap_absence_only",
        )
    ]
    plan.retired_market_ids.add("gap")

    locked = main._lock_guaranteed_order_markets(
        plan,
        [gap_market, _market("unanalyzed-liquid", liquidity=900.0)],
        excluded_market_ids=set(),
        priority_by_market_id={"gap": 0.10},
        require_cycle_analysis=True,
        settings=settings,
        cycle_number=1,
    )

    assert locked == []
    assert plan.slots[0].needs_replacement is True
    assert plan.slots[0].market_id == "gap"


def test_guaranteed_priority_prefers_high_eq_over_absence_only() -> None:
    scores = main._guaranteed_order_priority_scores(
        {
            "absence": {
                "decision": _decision(
                    evidence_basis="absence_only",
                    evidence_quality=0.1,
                    confidence=0.55,
                ),
                "pre_execution_final_score": 0.9,
            },
            "strong": {
                "decision": _decision(
                    evidence_basis="proxy",
                    evidence_quality=0.8,
                    confidence=0.50,
                    should_trade=True,
                ),
                "pre_execution_final_score": 0.2,
            },
        }
    )
    assert scores["strong"] > scores["absence"]


def test_guaranteed_priority_prefers_named_edge_mechanism() -> None:
    scores = main._guaranteed_order_priority_scores(
        {
            "hunch": {
                "decision": _decision(
                    evidence_basis="proxy",
                    evidence_quality=0.7,
                    confidence=0.60,
                    edge_mechanism="none",
                ),
            },
            "mechanism": {
                "decision": _decision(
                    evidence_basis="proxy",
                    evidence_quality=0.7,
                    confidence=0.60,
                    edge_mechanism="observed_vs_strike",
                ),
            },
        }
    )
    assert scores["mechanism"] > scores["hunch"]


def test_research_gap_reason_ignores_unlabeled_mechanism() -> None:
    settings = main.Settings()
    decision = _decision(
        evidence_basis="proxy",
        edge_source="computed",
        evidence_quality=0.8,
        edge_mechanism="none",
    )
    assert main._guaranteed_order_research_gap_reason(decision, settings) is None


def test_guaranteed_forces_computed_plus_ev_with_unlabeled_mechanism(tmp_path) -> None:
    """Live miss: TEMPMIAH-style computed +EV with edge_mechanism=none."""
    market = _market("temp-mia", yes_price=0.67)
    slot = main.GuaranteedOrderSlot(
        slot_number=1,
        market_id=market.id,
        market=market,
        locked_cycle=1,
        client_order_id="BOT-GUAR-temp-001",
    )
    grok = _GuaranteedGrok(
        _decision(
            outcome="NO",
            confidence=0.52,
            evidence_basis="proxy",
            edge_source="computed",
            evidence_quality=0.60,
            edge_mechanism="none",
        )
    )
    kalshi = _GuaranteedKalshi([market])
    state = MarketStateManager(str(tmp_path / "state.db"))
    try:
        result = main._attempt_guaranteed_order_slot(
            slot,
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
            settings=main.Settings(DRY_RUN=True, GUARANTEED_ORDERS_N=1),
            min_bet_usdc=5.0,
            max_bet_usdc=12.0,
        )
    finally:
        state.close()

    assert result.status == "dry_run"
    assert result.amount_usdc > 0
    assert result.error is None


def test_guaranteed_forces_low_eq_proxy_when_chosen_side_edge_clears(
    tmp_path,
) -> None:
    """Live miss: diesel-style eq=0.45 unlabeled proxy with large NO edge."""
    market = _market("diesel", yes_price=0.65)
    slot = main.GuaranteedOrderSlot(
        slot_number=1,
        market_id=market.id,
        market=market,
        locked_cycle=1,
        client_order_id="BOT-GUAR-diesel-001",
    )
    grok = _GuaranteedGrok(
        _decision(
            outcome="NO",
            confidence=0.62,
            evidence_basis="proxy",
            edge_source="none",
            evidence_quality=0.45,
            edge_mechanism="none",
            primary_source_url=None,
        )
    )
    kalshi = _GuaranteedKalshi([market])
    state = MarketStateManager(str(tmp_path / "state.db"))
    try:
        result = main._attempt_guaranteed_order_slot(
            slot,
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
            settings=main.Settings(DRY_RUN=True, GUARANTEED_ORDERS_N=1),
            min_bet_usdc=5.0,
            max_bet_usdc=12.0,
        )
    finally:
        state.close()

    assert result.status == "dry_run"
    assert result.amount_usdc > 0


def test_named_mechanism_uses_guaranteed_min_edge_not_proxy_floor(tmp_path) -> None:
    """Live miss: Brent 12pp observed_vs_strike failed the 15pp proxy floor."""
    market = _market("brent", yes_price=0.60)
    slot = main.GuaranteedOrderSlot(
        slot_number=1,
        market_id=market.id,
        market=market,
        locked_cycle=1,
        client_order_id="BOT-GUAR-brent-001",
    )
    grok = _GuaranteedGrok(
        _decision(
            outcome="YES",
            confidence=0.72,
            evidence_basis="proxy",
            edge_source="none",
            evidence_quality=0.60,
            edge_mechanism="observed_vs_strike",
        )
    )
    kalshi = _GuaranteedKalshi([market])
    state = MarketStateManager(str(tmp_path / "state.db"))
    try:
        result = main._attempt_guaranteed_order_slot(
            slot,
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
            settings=main.Settings(DRY_RUN=True, GUARANTEED_ORDERS_N=1),
            min_bet_usdc=5.0,
            max_bet_usdc=12.0,
        )
    finally:
        state.close()

    assert result.status == "dry_run"
    assert result.amount_usdc > 0


def test_unlabeled_proxy_still_requires_proxy_min_edge() -> None:
    settings = main.Settings()
    market = _market("thin-proxy", yes_price=0.60)
    decision = _decision(
        outcome="YES",
        confidence=0.72,
        evidence_basis="proxy",
        edge_source="none",
        evidence_quality=0.60,
        edge_mechanism="none",
    )
    assert (
        main._guaranteed_order_reject_reason(decision, market, settings)
        == "guaranteed_order_edge_below_min"
    )


def test_unlabeled_weather_proxy_uses_min_edge_not_proxy_floor() -> None:
    """Live miss: TEMPLAXH/TEMPMIAH +14pp unlabeled proxy deferred at 0.15."""
    settings = main.Settings()
    market = _market(
        "KXTEMPLAXH-26AUG2513-T81.99",
        yes_price=0.59,
        category="weather",
        event_ticker="KXTEMPLAXH-26AUG2513-T81.99",
    )
    decision = _decision(
        outcome="NO",
        confidence=0.55,
        evidence_basis="proxy",
        edge_source="none",
        evidence_quality=0.45,
        edge_mechanism="none",
    )
    assert main._guaranteed_order_min_edge(decision, market, settings) == 0.12
    assert main._guaranteed_order_reject_reason(decision, market, settings) is None


def test_lock_guaranteed_markets_rejects_same_event_prefix() -> None:
    settings = main.Settings(GUARANTEED_ORDERS_N=2)
    plan = main.GuaranteedOrderPlan(target=2)
    markets = [
        _market("HIGHMIA-B88.5", liquidity=500.0, event_ticker="HIGHMIA"),
        _market("HIGHMIA-B90.5", liquidity=400.0, event_ticker="HIGHMIA"),
        _market("OTHER-A", liquidity=100.0, event_ticker="OTHER"),
    ]

    locked = main._lock_guaranteed_order_markets(
        plan,
        markets,
        excluded_market_ids=set(),
        settings=settings,
        cycle_number=1,
    )

    assert [slot.market_id for slot in locked] == ["HIGHMIA-B88.5", "OTHER-A"]
    assert plan.is_fully_locked


def test_lock_collapses_weather_bins_with_full_event_tickers() -> None:
    settings = main.Settings(GUARANTEED_ORDERS_N=2)
    plan = main.GuaranteedOrderPlan(target=2)
    markets = [
        _market(
            "KXHIGHTOKC-26AUG25-B103.5",
            liquidity=500.0,
            event_ticker="KXHIGHTOKC-26AUG25-B103.5",
            category="weather",
        ),
        _market(
            "KXHIGHTOKC-26AUG25-B101.5",
            liquidity=400.0,
            event_ticker="KXHIGHTOKC-26AUG25-B101.5",
            category="weather",
        ),
        _market(
            "KXHIGHPHIL-26AUG25-B82.5",
            liquidity=100.0,
            event_ticker="KXHIGHPHIL-26AUG25-B82.5",
            category="weather",
        ),
    ]

    locked = main._lock_guaranteed_order_markets(
        plan,
        markets,
        excluded_market_ids=set(),
        settings=settings,
        cycle_number=1,
    )

    assert [slot.market_id for slot in locked] == [
        "KXHIGHTOKC-26AUG25-B103.5",
        "KXHIGHPHIL-26AUG25-B82.5",
    ]


def test_lock_skips_negative_edge_analyzed_seeds_for_catalog_fill() -> None:
    settings = main.Settings(GUARANTEED_ORDERS_N=1)
    plan = main.GuaranteedOrderPlan(target=1)
    analyzed_dog = _market("analyzed-neg", liquidity=900.0, yes_price=0.55)
    catalog = _market("catalog-fresh", liquidity=80.0)
    seeds = {
        "analyzed-neg": _decision(confidence=0.50, evidence_quality=0.8),
    }
    priority = main._guaranteed_order_priority_scores(
        {market_id: {"decision": decision} for market_id, decision in seeds.items()},
        markets_by_id={"analyzed-neg": analyzed_dog},
    )

    locked = main._lock_guaranteed_order_markets(
        plan,
        [analyzed_dog, catalog],
        excluded_market_ids=set(),
        priority_by_market_id=priority,
        seed_decisions_by_market_id=seeds,
        settings=settings,
        cycle_number=1,
    )

    assert [slot.market_id for slot in locked] == ["catalog-fresh"]


def test_seeded_guaranteed_slot_skips_initial_analysis(tmp_path) -> None:
    market = _market("seeded")
    seeded = _decision(evidence_quality=0.7)
    slot = main.GuaranteedOrderSlot(
        slot_number=1,
        market_id=market.id,
        market=market,
        locked_cycle=1,
        client_order_id="BOT-GUAR-seed-001",
        decision=seeded,
        research_completed=False,
    )
    grok = _GuaranteedGrok(_decision(evidence_quality=0.85))
    kalshi = _GuaranteedKalshi([market])
    state = MarketStateManager(str(tmp_path / "state.db"))
    try:
        result = main._attempt_guaranteed_order_slot(
            slot,
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
            settings=main.Settings(
                DRY_RUN=True,
                GUARANTEED_ORDERS_N=1,
            ),
            min_bet_usdc=5.0,
            max_bet_usdc=12.0,
        )
    finally:
        state.close()

    assert result.status == "dry_run"
    assert grok.initial_calls == []
    assert grok.deep_calls == []


def test_guaranteed_live_slot_submits_and_persists_pending_order(tmp_path) -> None:
    market = _market("live-forced")
    slot = main.GuaranteedOrderSlot(
        slot_number=1,
        market_id=market.id,
        market=market,
        locked_cycle=1,
        client_order_id="BOT-GUAR-test-001",
    )
    grok = _GuaranteedGrok(_decision())
    kalshi = _LiveGuaranteedKalshi([market])
    state = MarketStateManager(str(tmp_path / "state.db"))
    try:
        result = main._attempt_guaranteed_order_slot(
            slot,
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
            settings=main.Settings(
                DRY_RUN=False,
                GUARANTEED_ORDERS_N=1,
            ),
            min_bet_usdc=5.0,
            max_bet_usdc=12.0,
        )
        pending = state.get_pending_orders()
    finally:
        state.close()

    assert result.status == "submitted"
    assert result.submission_attempted is True
    assert result.order_response is not None
    assert result.order_response.id == "order-live-forced"
    assert kalshi.submitted_market_ids == [market.id]
    assert kalshi.client_order_ids == ["BOT-GUAR-test-001"]
    assert slot.submission_attempts == 1
    assert [row["order_id"] for row in pending] == ["order-live-forced"]


def test_guaranteed_phase_replaces_market_not_found_same_cycle(tmp_path) -> None:
    dead = _market("dead-high", liquidity=900.0)
    good = _market("good-low", liquidity=100.0)
    grok = _GuaranteedGrok(_decision())
    kalshi = _MarketNotFoundThenSuccessKalshi([dead, good])
    state = MarketStateManager(str(tmp_path / "state.db"))
    plan = main.GuaranteedOrderPlan(target=1, run_id="not-found")
    try:
        result = main._run_guaranteed_order_phase(
            plan=plan,
            markets=[dead, good],
            excluded_market_ids=set(),
            cycle_number=1,
            settings=main.Settings(
                DRY_RUN=False,
                GUARANTEED_ORDERS_N=1,
            ),
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
            min_bet_usdc=5.0,
            max_bet_usdc=12.0,
            log_decision=lambda **kwargs: None,
            extended_research_market_ids=set(),
        )
    finally:
        state.close()

    assert result.completed == 1
    assert plan.is_complete
    assert plan.slots[0].market_id == "good-low"
    assert "dead-high" in plan.retired_market_ids
    assert kalshi.submitted_market_ids == ["dead-high", "good-low"]


def test_guaranteed_phase_replaces_jurisdiction_blocked_sports_slot_same_cycle(
    tmp_path,
) -> None:
    sports = _market("sports-high", liquidity=900.0, category="mlb")
    politics = _market("politics-low", liquidity=100.0)
    grok = _GuaranteedGrok(_decision())
    kalshi = _JurisdictionThenSuccessKalshi([sports, politics])
    state = MarketStateManager(str(tmp_path / "state.db"))
    plan = main.GuaranteedOrderPlan(target=1, run_id="replacement")
    decisions: list[dict] = []
    try:
        result = main._run_guaranteed_order_phase(
            plan=plan,
            markets=[sports, politics],
            excluded_market_ids=set(),
            cycle_number=1,
            settings=main.Settings(
                DRY_RUN=False,
                GUARANTEED_ORDERS_N=1,
            ),
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
            min_bet_usdc=5.0,
            max_bet_usdc=12.0,
            log_decision=lambda **kwargs: decisions.append(kwargs),
            extended_research_market_ids=set(),
        )
        hold = state.get_runtime_flag("sports_jurisdiction_blocked")
    finally:
        state.close()

    assert kalshi.submitted_market_ids == ["sports-high", "politics-low"]
    assert kalshi.client_order_ids == [
        "BOT-GUAR-replacement-001",
        "BOT-GUAR-replacement-001-R001",
    ]
    assert result.attempted == 2
    assert result.completed == 1
    assert plan.is_complete
    assert plan.slots[0].market_id == "politics-low"
    assert plan.slots[0].replacement_count == 1
    assert "sports-high" in plan.retired_market_ids
    assert hold == "1"
    assert decisions[0]["execution_audit"]["final_reason"] == (
        "jurisdiction_sports_blocked"
    )
    assert decisions[-1]["execution_audit"]["final_reason"] == "order_submitted"


def test_guaranteed_phase_replaces_nevada_restricted_entertainment_slot(
    tmp_path,
) -> None:
    entertainment = _market(
        "KXYTVIEWSW-TAY26AUG16-14.5M",
        liquidity=900.0,
        category="entertainment",
    )
    weather = _market(
        "KXHIGHNY-26AUG17-T88",
        liquidity=100.0,
        category="weather",
    )
    grok = _GuaranteedGrok(_decision())
    kalshi = _NevadaRestrictedThenSuccessKalshi([entertainment, weather])
    state = MarketStateManager(str(tmp_path / "state.db"))
    plan = main.GuaranteedOrderPlan(target=1, run_id="nevada-replacement")
    decisions: list[dict] = []
    try:
        result = main._run_guaranteed_order_phase(
            plan=plan,
            markets=[entertainment, weather],
            excluded_market_ids=set(),
            cycle_number=1,
            settings=main.Settings(
                DRY_RUN=False,
                GUARANTEED_ORDERS_N=1,
            ),
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
            min_bet_usdc=5.0,
            max_bet_usdc=12.0,
            log_decision=lambda **kwargs: decisions.append(kwargs),
            extended_research_market_ids=set(),
        )
        hold = main._jurisdiction_blocked_families(state)
    finally:
        state.close()

    assert kalshi.submitted_market_ids == [
        "KXYTVIEWSW-TAY26AUG16-14.5M",
        "KXHIGHNY-26AUG17-T88",
    ]
    assert result.attempted == 2
    assert result.completed == 1
    assert plan.is_complete
    assert plan.slots[0].market_id == "KXHIGHNY-26AUG17-T88"
    assert "KXYTVIEWSW-TAY26AUG16-14.5M" in plan.retired_market_ids
    assert hold == {"sports", "politics", "entertainment", "music"}
    assert decisions[0]["execution_audit"]["final_reason"] == "jurisdiction_restricted"
    assert decisions[-1]["execution_audit"]["final_reason"] == "order_submitted"


def test_guaranteed_phase_honors_existing_sports_jurisdiction_hold(tmp_path) -> None:
    sports = _market("sports-high", liquidity=900.0, category="mlb")
    politics = _market("politics-low", liquidity=100.0)
    grok = _GuaranteedGrok(_decision())
    kalshi = _LiveGuaranteedKalshi([sports, politics])
    state = MarketStateManager(str(tmp_path / "state.db"))
    plan = main.GuaranteedOrderPlan(target=1, run_id="existing-hold")
    try:
        state.set_runtime_flag("sports_jurisdiction_blocked", "1")
        result = main._run_guaranteed_order_phase(
            plan=plan,
            markets=[sports, politics],
            excluded_market_ids=set(),
            cycle_number=1,
            settings=main.Settings(
                DRY_RUN=False,
                GUARANTEED_ORDERS_N=1,
            ),
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
            min_bet_usdc=5.0,
            max_bet_usdc=12.0,
            log_decision=lambda **kwargs: None,
            extended_research_market_ids=set(),
        )
    finally:
        state.close()

    assert result.completed == 1
    assert plan.is_complete
    assert [slot.market_id for slot in plan.slots] == ["politics-low"]
    assert kalshi.submitted_market_ids == ["politics-low"]
    assert grok.initial_calls == ["politics-low"]
    assert grok.deep_calls == ["politics-low"]


def test_bounded_main_dry_run_records_exact_guaranteed_target(
    monkeypatch,
    dummy_settings,
) -> None:
    markets = [_market("g1", liquidity=500.0), _market("g2", liquidity=400.0)]
    grok = _GuaranteedGrok(_decision())
    kalshi = _GuaranteedKalshi(markets)
    settings = replace(
        dummy_settings,
        GUARANTEED_ORDERS_N=2,
        DRY_RUN=True,
        MIN_VOLUME_24H=0.0,
        MIN_OPEN_INTEREST=0.0,
        MARKET_MIN_CLOSE_DAYS=None,
        MARKET_MAX_CLOSE_DAYS=None,
        POLL_INTERVAL_SEC=0,
    )
    monkeypatch.setattr(main, "load_settings", lambda: settings)
    monkeypatch.setattr(main, "GrokClient", lambda *args, **kwargs: grok)
    monkeypatch.setattr(main, "KalshiClient", lambda *args, **kwargs: kalshi)
    monkeypatch.setattr(main.time, "sleep", lambda _: None)

    main.main(max_cycles=1)

    verifier = MarketStateManager(settings.STATE_DB_PATH)
    try:
        attempts = verifier._conn.execute(
            """
            SELECT market_id, audit_json
            FROM decision_receipts
            WHERE final_action = 'order_attempt'
              AND final_reason = 'dry_run'
            ORDER BY market_id
            """
        ).fetchall()
        cycle_receipt = verifier._conn.execute(
            "SELECT payload_json FROM cycle_receipts ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        verifier.close()

    assert [row["market_id"] for row in attempts] == ["g1", "g2"]
    assert all('"guaranteed_order_forced_execution": true' in row["audit_json"] for row in attempts)
    assert cycle_receipt is not None
    assert '"guaranteed_orders_complete": true' in cycle_receipt["payload_json"]
    assert kalshi.submitted_market_ids == []


def test_guaranteed_allows_edge_source_none_with_proxy_url_and_eq(tmp_path) -> None:
    market = _market("proxy-forced")
    slot = main.GuaranteedOrderSlot(
        slot_number=1,
        market_id=market.id,
        market=market,
        locked_cycle=1,
        client_order_id="BOT-GUAR-proxy-001",
    )
    grok = _GuaranteedGrok(
        _decision(evidence_basis="proxy", edge_source="none", evidence_quality=0.8)
    )
    kalshi = _GuaranteedKalshi([market])
    state = MarketStateManager(str(tmp_path / "state.db"))
    try:
        result = main._attempt_guaranteed_order_slot(
            slot,
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
            settings=main.Settings(
                DRY_RUN=True,
                GUARANTEED_ORDERS_N=1,
            ),
            min_bet_usdc=5.0,
            max_bet_usdc=12.0,
        )
    finally:
        state.close()

    assert result.status == "dry_run"
    assert result.decision is not None
    assert result.decision.edge_source == "computed"
    assert result.decision.should_trade is True
    assert slot.submission_attempts == 1


def test_guaranteed_phase_abandons_after_weak_evidence_replace_cap(
    tmp_path,
) -> None:
    markets = [
        _market(f"gap-{idx}", liquidity=1000.0 - idx) for idx in range(4)
    ]
    gap_decision = _decision(
        evidence_basis="absence_only", edge_source="none", evidence_quality=0.1
    )

    class _AllGapGrok:
        def __init__(self) -> None:
            self.deep_calls: list[str] = []

        def analyze_market(self, market, **kwargs):
            return gap_decision

        def analyze_market_deep(self, market, **kwargs):
            self.deep_calls.append(market.id)
            return gap_decision

    grok = _AllGapGrok()
    kalshi = _LiveGuaranteedKalshi(markets)
    state = MarketStateManager(str(tmp_path / "state.db"))
    plan = main.GuaranteedOrderPlan(target=1, run_id="gap-force-cap")
    decisions: list[dict] = []
    try:
        result = main._run_guaranteed_order_phase(
            plan=plan,
            markets=markets,
            excluded_market_ids=set(),
            cycle_number=1,
            settings=main.Settings(
                DRY_RUN=False,
                GUARANTEED_ORDERS_N=1,
                GUARANTEED_ORDER_MAX_RESEARCH_GAP_REPLACEMENTS=2,
            ),
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
            min_bet_usdc=5.0,
            max_bet_usdc=12.0,
            log_decision=lambda **kwargs: decisions.append(kwargs),
            extended_research_market_ids=set(),
        )
    finally:
        state.close()

    assert result.completed == 0
    assert plan.is_complete is False
    assert plan.abandoned_count == 1
    assert plan.research_gap_replacements == 2
    assert len(grok.deep_calls) == 3
    assert kalshi.submitted_market_ids == []
    assert plan.suppresses_normal_execution is False


def test_bounded_main_exits_early_when_guaranteed_target_complete(
    monkeypatch,
    dummy_settings,
) -> None:
    markets = [_market("early-exit", liquidity=500.0)]
    grok = _GuaranteedGrok(_decision())
    fetch_cycles = {"count": 0}

    class _CountingKalshi(_GuaranteedKalshi):
        def get_markets(self, **kwargs):
            fetch_cycles["count"] += 1
            return super().get_markets(**kwargs)

    kalshi = _CountingKalshi(markets)
    settings = replace(
        dummy_settings,
        GUARANTEED_ORDERS_N=1,
        DRY_RUN=True,
        MIN_VOLUME_24H=0.0,
        MIN_OPEN_INTEREST=0.0,
        MARKET_MIN_CLOSE_DAYS=None,
        MARKET_MAX_CLOSE_DAYS=None,
        POLL_INTERVAL_SEC=0,
    )
    monkeypatch.setattr(main, "load_settings", lambda: settings)
    monkeypatch.setattr(main, "GrokClient", lambda *args, **kwargs: grok)
    monkeypatch.setattr(main, "KalshiClient", lambda *args, **kwargs: kalshi)
    monkeypatch.setattr(main.time, "sleep", lambda _: None)

    main.main(max_cycles=3)

    assert fetch_cycles["count"] == 1


def test_bounded_main_does_not_complete_guaranteed_target_with_weak_evidence(
    monkeypatch,
    dummy_settings,
) -> None:
    markets = [_market(f"gap-{idx}", liquidity=500.0 - idx) for idx in range(3)]
    gap_decision = _decision(
        evidence_basis="absence_only", edge_source="none", evidence_quality=0.1
    )
    grok = _GuaranteedGrok(gap_decision)
    kalshi = _GuaranteedKalshi(markets)
    settings = replace(
        dummy_settings,
        GUARANTEED_ORDERS_N=1,
        GUARANTEED_ORDER_MAX_RESEARCH_GAP_REPLACEMENTS=1,
        DRY_RUN=True,
        MIN_VOLUME_24H=0.0,
        MIN_OPEN_INTEREST=0.0,
        MARKET_MIN_CLOSE_DAYS=None,
        MARKET_MAX_CLOSE_DAYS=None,
        POLL_INTERVAL_SEC=0,
    )
    monkeypatch.setattr(main, "load_settings", lambda: settings)
    monkeypatch.setattr(main, "GrokClient", lambda *args, **kwargs: grok)
    monkeypatch.setattr(main, "KalshiClient", lambda *args, **kwargs: kalshi)
    monkeypatch.setattr(main.time, "sleep", lambda _: None)

    with pytest.raises(main.GuaranteedOrdersIncompleteError, match="completed=0/1"):
        main.main(max_cycles=1)

    verifier = MarketStateManager(settings.STATE_DB_PATH)
    try:
        cycle_receipt = verifier._conn.execute(
            "SELECT payload_json FROM cycle_receipts ORDER BY id DESC LIMIT 1"
        ).fetchone()
        attempts = verifier._conn.execute(
            """
            SELECT market_id
            FROM decision_receipts
            WHERE final_action = 'order_attempt'
              AND final_reason = 'dry_run'
            """
        ).fetchall()
    finally:
        verifier.close()

    assert cycle_receipt is not None
    assert '"guaranteed_orders_complete": true' not in cycle_receipt["payload_json"]
    assert len(attempts) == 0


def test_bounded_main_fails_when_target_cannot_be_fully_locked(
    monkeypatch,
    dummy_settings,
) -> None:
    market = _market("only-one")
    grok = _GuaranteedGrok(_decision())
    kalshi = _GuaranteedKalshi([market])
    settings = replace(
        dummy_settings,
        GUARANTEED_ORDERS_N=2,
        DRY_RUN=True,
        MIN_VOLUME_24H=0.0,
        MIN_OPEN_INTEREST=0.0,
        MARKET_MIN_CLOSE_DAYS=None,
        MARKET_MAX_CLOSE_DAYS=None,
        POLL_INTERVAL_SEC=0,
    )
    monkeypatch.setattr(main, "load_settings", lambda: settings)
    monkeypatch.setattr(main, "GrokClient", lambda *args, **kwargs: grok)
    monkeypatch.setattr(main, "KalshiClient", lambda *args, **kwargs: kalshi)
    monkeypatch.setattr(main.time, "sleep", lambda _: None)

    with pytest.raises(main.GuaranteedOrdersIncompleteError, match="completed=1/2"):
        main.main(max_cycles=1)


def test_guaranteed_priority_prefers_calibrated_edge_over_high_conf_zero_edge() -> None:
    zero_edge_market = _market("zero", yes_price=0.80)
    plus_edge_market = _market("edged", yes_price=0.55)
    scores = main._guaranteed_order_priority_scores(
        {
            "zero": {
                "decision": _decision(
                    confidence=0.80,
                    evidence_quality=0.95,
                    evidence_basis="direct",
                ),
            },
            "edged": {
                "decision": _decision(
                    confidence=0.70,
                    evidence_quality=0.80,
                    should_trade=True,
                    edge_mechanism="observed_vs_strike",
                ),
            },
        },
        markets_by_id={"zero": zero_edge_market, "edged": plus_edge_market},
    )
    assert scores["edged"] > scores["zero"]


def test_lock_prefers_positive_edge_analyzed_seeds() -> None:
    settings = main.Settings(GUARANTEED_ORDERS_N=1)
    plan = main.GuaranteedOrderPlan(target=1)
    weak = _market("weak-edge", liquidity=900.0)
    strong = _market("strong-edge", liquidity=50.0)
    seeds = {
        "weak-edge": _decision(confidence=0.50),
        "strong-edge": _decision(
            confidence=0.72,
            evidence_quality=0.8,
            edge_mechanism="observed_vs_strike",
        ),
    }
    priority = main._guaranteed_order_priority_scores(
        {market_id: {"decision": decision} for market_id, decision in seeds.items()},
        markets_by_id={"weak-edge": weak, "strong-edge": strong},
    )

    locked = main._lock_guaranteed_order_markets(
        plan,
        [weak, strong],
        excluded_market_ids=set(),
        priority_by_market_id=priority,
        seed_decisions_by_market_id=seeds,
        settings=settings,
        cycle_number=1,
    )

    assert [slot.market_id for slot in locked] == ["strong-edge"]


def test_guaranteed_phase_replaces_negative_edge_when_alternate_exists(tmp_path) -> None:
    weak_market = _market("weak-high", liquidity=900.0)
    good_market = _market("good-low", liquidity=100.0)
    weak_decision = _decision(confidence=0.50, evidence_quality=0.8)
    good_decision = _decision(
        evidence_quality=0.85,
        edge_mechanism="observed_vs_strike",
    )

    class _SelectiveGrok:
        def __init__(self) -> None:
            self.deep_calls: list[str] = []

        def analyze_market(self, market, **kwargs):
            return weak_decision if market.id == "weak-high" else good_decision

        def analyze_market_deep(self, market, **kwargs):
            self.deep_calls.append(market.id)
            return weak_decision if market.id == "weak-high" else good_decision

    grok = _SelectiveGrok()
    kalshi = _LiveGuaranteedKalshi([weak_market, good_market])
    state = MarketStateManager(str(tmp_path / "state.db"))
    plan = main.GuaranteedOrderPlan(target=1, run_id="neg-edge-replace")
    try:
        result = main._run_guaranteed_order_phase(
            plan=plan,
            markets=[weak_market, good_market],
            excluded_market_ids=set(),
            cycle_number=1,
            settings=main.Settings(
                DRY_RUN=False,
                GUARANTEED_ORDERS_N=1,
            ),
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
            min_bet_usdc=5.0,
            max_bet_usdc=12.0,
            log_decision=lambda **kwargs: None,
            extended_research_market_ids=set(),
        )
    finally:
        state.close()

    assert result.completed == 1
    assert plan.slots[0].market_id == "good-low"
    assert "weak-high" in plan.retired_market_ids
    assert kalshi.submitted_market_ids == ["good-low"]
    assert grok.deep_calls == ["weak-high", "good-low"]


def test_guaranteed_reject_reason_flags_non_positive_edge() -> None:
    settings = main.Settings()
    market = _market("flat")
    decision = _decision(confidence=0.50)
    assert (
        main._guaranteed_order_reject_reason(decision, market, settings)
        == "guaranteed_order_non_positive_edge"
    )
    below = _decision(confidence=0.62)
    assert (
        main._guaranteed_order_reject_reason(below, market, settings)
        == "guaranteed_order_edge_below_min"
    )


def test_guaranteed_sizing_scales_with_bankroll() -> None:
    market = _market("sized")
    decision = _decision(
        confidence=0.85,
        evidence_quality=0.9,
        evidence_basis="direct",
        edge_mechanism="observed_vs_strike",
    )
    settings = main.Settings()
    min_75, max_75 = main._effective_bet_bounds_usdc(settings, 75.0)
    min_150, max_150 = main._effective_bet_bounds_usdc(settings, 150.0)
    amount_75, audit_75 = main._guaranteed_order_sized_amount_usdc(
        decision=decision,
        market=market,
        settings=settings,
        min_bet_usdc=min_75,
        max_bet_usdc=max_75,
    )
    amount_150, audit_150 = main._guaranteed_order_sized_amount_usdc(
        decision=decision,
        market=market,
        settings=settings,
        min_bet_usdc=min_150,
        max_bet_usdc=max_150,
    )
    assert amount_75 > 0
    assert amount_150 > amount_75
    assert audit_75["guaranteed_order_sizing_mode"] == "kelly"
    assert audit_150["guaranteed_order_sizing_mode"] == "kelly"


def test_guaranteed_sizing_strong_edge_exceeds_min_bet() -> None:
    market = _market("sized")
    decision = _decision(
        confidence=0.85,
        evidence_quality=0.9,
        evidence_basis="direct",
        edge_mechanism="observed_vs_strike",
    )
    amount, audit = main._guaranteed_order_sized_amount_usdc(
        decision=decision,
        market=market,
        settings=main.Settings(),
        min_bet_usdc=2.0,
        max_bet_usdc=20.0,
    )
    assert amount > 2.0
    assert audit["guaranteed_order_sizing_mode"] == "kelly"


def test_bounded_main_records_five_positive_ev_guaranteed_orders(
    monkeypatch,
    dummy_settings,
) -> None:
    markets = [
        _market(f"g{idx}", liquidity=500.0 - idx, event_ticker=f"EVT{idx}")
        for idx in range(5)
    ]
    grok = _GuaranteedGrok(
        _decision(
            evidence_basis="direct",
            evidence_quality=0.85,
            edge_mechanism="observed_vs_strike",
        )
    )
    kalshi = _GuaranteedKalshi(markets)
    settings = replace(
        dummy_settings,
        GUARANTEED_ORDERS_N=5,
        DRY_RUN=True,
        MIN_VOLUME_24H=0.0,
        MIN_OPEN_INTEREST=0.0,
        MARKET_MIN_CLOSE_DAYS=None,
        MARKET_MAX_CLOSE_DAYS=None,
        POLL_INTERVAL_SEC=0,
    )
    monkeypatch.setattr(main, "load_settings", lambda: settings)
    monkeypatch.setattr(main, "GrokClient", lambda *args, **kwargs: grok)
    monkeypatch.setattr(main, "KalshiClient", lambda *args, **kwargs: kalshi)
    monkeypatch.setattr(main.time, "sleep", lambda _: None)

    main.main(max_cycles=5)

    verifier = MarketStateManager(settings.STATE_DB_PATH)
    try:
        attempts = verifier._conn.execute(
            """
            SELECT market_id
            FROM decision_receipts
            WHERE final_action = 'order_attempt'
              AND final_reason = 'dry_run'
            ORDER BY market_id
            """
        ).fetchall()
        cycle_receipt = verifier._conn.execute(
            "SELECT payload_json FROM cycle_receipts ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        verifier.close()

    assert [row["market_id"] for row in attempts] == [f"g{idx}" for idx in range(5)]
    assert cycle_receipt is not None
    assert '"guaranteed_orders_complete": true' in cycle_receipt["payload_json"]
