from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from market_state import MarketStateManager
from models import OrderResponse, TradeDecision


def _decision(confidence: float, outcome: str = "YES") -> TradeDecision:
    return TradeDecision(
        should_trade=True,
        outcome=outcome,
        confidence=confidence,
        bet_size_pct=0.5,
        reasoning="test",
    )


def test_market_state_trend_and_counts(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m1"
        confidences = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        for confidence in confidences:
            manager.record_analysis(market_id, _decision(confidence), is_refined=False)

        state = manager.get_market_state(market_id)
        assert state is not None
        assert state.analysis_count == len(confidences)
        assert state.last_confidence == confidences[-1]
        assert state.last_analysis is not None
        assert state.confidence_trend == confidences[-5:]
    finally:
        manager.close()


def test_record_trade_updates_position_and_avg_confidence(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m2"
        manager.record_analysis(market_id, _decision(0.8), is_refined=False)
        order = OrderResponse(id="o1", raw={"outcome": "YES"})
        manager.record_trade(market_id, order, 25.0)

        position = manager.get_position(market_id)
        assert position is not None
        assert position.total_amount_usdc == 25.0
        assert position.avg_confidence == 0.8
        assert position.trade_count == 1

        manager.record_analysis(market_id, _decision(0.6), is_refined=True)
        order2 = OrderResponse(id="o2", raw={"outcome": "YES"})
        manager.record_trade(market_id, order2, 25.0)

        position = manager.get_position(market_id)
        assert position is not None
        assert position.total_amount_usdc == 50.0
        assert round(position.avg_confidence, 4) == 0.7
        assert position.trade_count == 2
    finally:
        manager.close()


def test_get_markets_needing_reanalysis(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m3"
        manager.record_analysis(market_id, _decision(0.7), is_refined=False)
        old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()
        manager._conn.execute(
            "UPDATE analyses SET timestamp = ? WHERE market_id = ?",
            (old_timestamp, market_id),
        )
        manager._conn.commit()

        needs_reanalysis = manager.get_markets_needing_reanalysis(6)
        assert market_id in needs_reanalysis

        fresh_only = manager.get_markets_needing_reanalysis(12)
        assert market_id not in fresh_only
    finally:
        manager.close()


def test_export_to_json(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m4"
        manager.record_analysis(market_id, _decision(0.9), is_refined=False)
        order = OrderResponse(id="o9", raw={"outcome": "YES"})
        manager.record_trade(market_id, order, 15.0)

        export_path = tmp_path / "state.json"
        manager.export_to_json(str(export_path))

        payload = json.loads(export_path.read_text(encoding="utf-8"))
        assert set(payload.keys()) == {
            "markets",
            "analyses",
            "positions",
            "trade_log",
            "trade_outcomes",
            "trade_outcome_events",
        }
        assert payload["analyses"]
        assert payload["positions"]
        assert payload["trade_log"]
        assert payload["trade_outcomes"]
        assert payload["trade_outcome_events"]
        assert payload["positions"][0]["order_ids"] == ["o9"]
    finally:
        manager.close()


def test_record_resolution_idempotent(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m5"
        manager.record_analysis(market_id, _decision(0.7), is_refined=False)
        manager.record_trade(market_id, OrderResponse(id="o1", raw={"outcome": "YES"}), 10.0, outcome="YES")
        updated = manager.record_resolution(market_id, "YES", datetime.now(timezone.utc))
        assert updated is True
        updated_again = manager.record_resolution(market_id, "YES", datetime.now(timezone.utc))
        assert updated_again is False
    finally:
        manager.close()


def test_backfill_sentinel_resolution_to_unresolved(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m6"
        manager.record_analysis(market_id, _decision(0.7), is_refined=False)
        manager.record_trade(market_id, OrderResponse(id="o2", raw={"outcome": "YES"}), 5.0, outcome="YES")
        manager._conn.execute(
            """
            UPDATE trade_outcomes
            SET resolved_winning_outcome = '18446744073709551615', won = 0, pnl_estimate = -1.0, resolved_at = ?
            WHERE market_id = ?
            """,
            (datetime.now(timezone.utc).isoformat(), market_id),
        )
        manager._conn.commit()
        manager._backfill_resolution_state()
        row = manager._conn.execute(
            "SELECT resolved_winning_outcome, won, pnl_estimate, resolved_at, resolution_state FROM trade_outcomes WHERE market_id = ?",
            (market_id,),
        ).fetchone()
        assert row["resolved_winning_outcome"] is None
        assert row["won"] is None
        assert row["pnl_estimate"] is None
        assert row["resolved_at"] is None
        assert row["resolution_state"] == "unresolved"
    finally:
        manager.close()


def test_get_anchor_analysis_prefers_high_confidence(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m7"
        manager.record_analysis(market_id, _decision(0.55, outcome="YES"), is_refined=False)
        manager.record_analysis(market_id, _decision(0.72, outcome="NO"), is_refined=False)
        manager.record_analysis(market_id, _decision(0.61, outcome="YES"), is_refined=False)

        anchor = manager.get_anchor_analysis(market_id, min_confidence=0.65)
        assert anchor is not None
        assert anchor["outcome"] == "NO"
        assert round(float(anchor["confidence"]), 2) == 0.72
    finally:
        manager.close()


def test_get_anchor_analysis_falls_back_to_latest(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m8"
        manager.record_analysis(market_id, _decision(0.50, outcome="YES"), is_refined=False)
        manager.record_analysis(market_id, _decision(0.58, outcome="NO"), is_refined=False)

        anchor = manager.get_anchor_analysis(market_id, min_confidence=0.65)
        assert anchor is not None
        assert anchor["outcome"] == "NO"
        assert round(float(anchor["confidence"]), 2) == 0.58
    finally:
        manager.close()


def test_record_terminal_outcome_persists_on_market_state(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m9"
        manager.record_analysis(market_id, _decision(0.61), is_refined=False)
        manager.record_terminal_outcome(market_id, "no_trade_recommended")
        state = manager.get_market_state(market_id)
        assert state is not None
        assert state.last_terminal_outcome == "no_trade_recommended"
    finally:
        manager.close()


def test_reasoning_hash_and_stale_bayesian_update(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m-stale"
        decision = _decision(0.66, outcome="YES")
        manager.record_analysis(market_id, decision, is_refined=False)
        first_hash = manager.get_last_reasoning_hash(market_id)
        assert first_hash is not None

        manager.record_analysis(market_id, decision, is_refined=False)
        second_hash = manager.get_last_reasoning_hash(market_id)
        assert second_hash == first_hash

        manager.update_bayesian_state(
            market_id=market_id,
            outcome="YES",
            log_prior=0.0,
            log_likelihood=0.2,
            count_as_update=True,
        )
        manager.update_bayesian_state(
            market_id=market_id,
            outcome="YES",
            log_prior=0.0,
            log_likelihood=0.2,
            count_as_update=False,
        )
        state = manager.get_bayesian_state(market_id)["YES"]
        assert state.update_count == 1
    finally:
        manager.close()


def test_reasoning_hash_ignores_validated_prefix_variation(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m-hash"
        manager.record_analysis(
            market_id,
            TradeDecision(
                should_trade=False,
                outcome="YES",
                confidence=0.66,
                bet_size_pct=0.0,
                reasoning="[Validated eq=1.00 edge_market=0.031] thesis text",
            ),
            is_refined=False,
        )
        first_hash = manager.get_last_reasoning_hash(market_id)
        manager.record_analysis(
            market_id,
            TradeDecision(
                should_trade=False,
                outcome="YES",
                confidence=0.66,
                bet_size_pct=0.0,
                reasoning="[Validated eq=0.95 edge_market=0.028] thesis text",
            ),
            is_refined=False,
        )
        second_hash = manager.get_last_reasoning_hash(market_id)
        assert first_hash == second_hash
    finally:
        manager.close()


def test_get_outcome_flip_count_counts_transitions(tmp_path) -> None:
    manager = MarketStateManager(str(tmp_path / "state.db"))
    try:
        market_id = "m-flips"
        manager.record_analysis(market_id, _decision(0.60, outcome="YES"), is_refined=False)
        manager.record_analysis(market_id, _decision(0.62, outcome="NO"), is_refined=False)
        manager.record_analysis(market_id, _decision(0.64, outcome="YES"), is_refined=False)
        manager.record_analysis(market_id, _decision(0.66, outcome="NO"), is_refined=False)
        assert manager.get_outcome_flip_count(market_id) == 3
    finally:
        manager.close()
