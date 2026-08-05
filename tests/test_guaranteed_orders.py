from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

import main
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

    def get_markets(self, **kwargs):
        self.last_fetch_mve_filter = kwargs.get("mve_filter")
        return self.markets

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


def _market(
    market_id: str,
    *,
    liquidity: float = 200.0,
    category: str = "politics",
    event_ticker: str | None = None,
) -> Market:
    return Market(
        id=market_id,
        event_ticker=event_ticker or f"EVENT-{market_id}",
        question=f"Will {market_id} happen?",
        outcomes=[
            MarketOutcome(name="YES", price=0.55),
            MarketOutcome(name="NO", price=0.45),
        ],
        liquidity_usdc=liquidity,
        volume_24h=100.0,
        open_interest=100.0,
        category=category,
        status="open",
        close_time=datetime.now(timezone.utc) + timedelta(days=2),
    )


def _decision(
    *,
    evidence_basis: str = "proxy",
    edge_source: str = "computed",
    evidence_quality: float = 0.8,
    confidence: float = 0.72,
    should_trade: bool = False,
    primary_source_url: str | None = "https://example.com/source",
) -> TradeDecision:
    return TradeDecision(
        should_trade=should_trade,
        outcome="YES",
        probability_yes=0.72,
        confidence=confidence,
        my_prob=0.72,
        bet_size_pct=0.0,
        reasoning="Deep evidence favors YES.",
        evidence_quality=evidence_quality,
        evidence_basis=evidence_basis,
        edge_source=edge_source,
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
                MIN_BET_USDC=5.0,
                MAX_BET_USDC=12.0,
            ),
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


def test_guaranteed_forces_absence_only_after_deep_research(tmp_path) -> None:
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
                MIN_BET_USDC=5.0,
                MAX_BET_USDC=12.0,
            ),
        )
    finally:
        state.close()

    assert result.status == "dry_run"
    assert result.decision is not None
    assert result.decision.should_trade is True
    assert result.amount_usdc == 5.0
    assert result.sizing_audit["guaranteed_order_research_gap_bypassed"] == (
        "guaranteed_order_research_gap_absence_only"
    )
    assert slot.submission_attempts == 1
    assert kalshi.submitted_market_ids == []


def test_guaranteed_phase_forces_weak_evidence_without_replacement(tmp_path) -> None:
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
                MIN_BET_USDC=5.0,
                MAX_BET_USDC=12.0,
            ),
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
            log_decision=lambda **kwargs: decisions.append(kwargs),
            extended_research_market_ids=set(),
        )
    finally:
        state.close()

    assert result.completed == 1
    assert plan.is_complete
    assert plan.slots[0].market_id == "gap-only"
    assert kalshi.submitted_market_ids == ["gap-only"]
    assert decisions[-1]["execution_audit"]["final_reason"] == "order_submitted"
    assert decisions[-1]["execution_audit"][
        "guaranteed_order_research_gap_bypassed"
    ] == "guaranteed_order_research_gap_absence_only"


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
                MIN_BET_USDC=5.0,
                MAX_BET_USDC=12.0,
            ),
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
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
                MIN_BET_USDC=5.0,
                MAX_BET_USDC=12.0,
            ),
        )
    finally:
        state.close()

    assert result.status == "dry_run"
    assert grok.initial_calls == []
    assert grok.deep_calls == ["seeded"]


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
                MIN_BET_USDC=5.0,
                MAX_BET_USDC=12.0,
            ),
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
                MIN_BET_USDC=5.0,
                MAX_BET_USDC=12.0,
            ),
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
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
                MIN_BET_USDC=5.0,
                MAX_BET_USDC=12.0,
            ),
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
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
                MIN_BET_USDC=5.0,
                MAX_BET_USDC=12.0,
            ),
        )
    finally:
        state.close()

    assert result.status == "dry_run"
    assert result.decision is not None
    assert result.decision.edge_source == "computed"
    assert result.decision.should_trade is True
    assert slot.submission_attempts == 1


def test_guaranteed_phase_forces_fill_despite_weak_evidence_across_candidates(
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
                MIN_BET_USDC=5.0,
                MAX_BET_USDC=12.0,
            ),
            grok_client=grok,
            kalshi_client=kalshi,
            state_manager=state,
            log_decision=lambda **kwargs: decisions.append(kwargs),
            extended_research_market_ids=set(),
        )
    finally:
        state.close()

    assert result.completed == 1
    assert plan.is_complete is True
    assert plan.abandoned_count == 0
    # Soft-replace twice, then force on the third researched market.
    assert plan.research_gap_replacements == 2
    assert len(grok.deep_calls) == 3
    assert kalshi.submitted_market_ids == ["gap-2"]
    assert decisions[-1]["execution_audit"]["final_reason"] == "order_submitted"
    assert plan.suppresses_normal_execution is False
    family_stats = result.family_execution.get("politics") or result.family_execution.get(
        main.market_family(markets[2])
    )
    assert family_stats is not None
    assert family_stats["usd_submitted"] > 0.0
    assert family_stats["order_attempts"] >= 1.0


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


def test_bounded_main_completes_guaranteed_target_with_weak_evidence(
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
    assert '"guaranteed_orders_resolved": true' in cycle_receipt["payload_json"]
    assert '"guaranteed_orders_complete": true' in cycle_receipt["payload_json"]
    assert len(attempts) == 1


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

    with pytest.raises(main.GuaranteedOrdersIncompleteError, match="completed=0/2"):
        main.main(max_cycles=1)
