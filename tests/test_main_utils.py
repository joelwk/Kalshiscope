import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from config import Settings
from main import (
    _best_orderbook_sell_price,
    _build_reasoning_hash,
    _cap_analysis_candidates,
    _cap_effective_confidence_for_market,
    _calculate_bet,
    _collapse_event_ladders,
    _confidence_gate_override_metrics,
    _compute_next_wakeup_seconds,
    _edge_threshold_for_market,
    _effective_position_override_threshold,
    _filter_markets,
    _log_settings_summary,
    _should_adjust_position,
)
from models import Market, MarketOutcome, MarketState, Position, TradeDecision


class DummyStateManager:
    def __init__(self, mapping: dict[str, MarketState | None]) -> None:
        self.mapping = mapping

    def get_market_state(self, market_id: str) -> MarketState | None:
        return self.mapping.get(market_id)


class TestMainUtils(unittest.TestCase):
    def test_filter_markets(self) -> None:
        markets = [
            Market(id="1", question="Q1", liquidity_usdc=50, category="sports"),
            Market(id="2", question="Q2", liquidity_usdc=150, category="sports"),
            Market(id="3", question="Q3", liquidity_usdc=200, category="politics"),
        ]
        filtered = _filter_markets(
            markets,
            min_liquidity=100,
            allowlist=("sports",),
            blocklist=("politics",),
        )
        self.assertEqual([m.id for m in filtered], ["2"])

    def test_filter_markets_by_close_date(self) -> None:
        now = datetime.now(timezone.utc)
        markets = [
            Market(id="1", question="Q1", close_time=now + timedelta(hours=6)),
            Market(id="2", question="Q2", close_time=now + timedelta(days=3)),
            Market(id="3", question="Q3", close_time=now + timedelta(days=10)),
            Market(id="4", question="Q4", close_time=None),
        ]
        # Filter: only markets closing between 1 and 7 days from now
        filtered = _filter_markets(
            markets,
            min_liquidity=0,
            allowlist=(),
            blocklist=(),
            min_close_days=1,
            max_close_days=7,
        )
        # Market 1 closes too soon (<1 day), Market 3 closes too far (>7 days)
        # Market 4 has no close_time, so it passes (no filter applied)
        self.assertEqual([m.id for m in filtered], ["2", "4"])

    def test_filter_markets_max_close_days_only(self) -> None:
        now = datetime.now(timezone.utc)
        markets = [
            Market(id="1", question="Q1", close_time=now + timedelta(hours=12)),
            Market(id="2", question="Q2", close_time=now + timedelta(days=5)),
        ]
        # Only set max_close_days (markets closing within 3 days)
        filtered = _filter_markets(
            markets,
            min_liquidity=0,
            allowlist=(),
            blocklist=(),
            max_close_days=3,
        )
        self.assertEqual([m.id for m in filtered], ["1"])

    def test_calculate_bet(self) -> None:
        self.assertEqual(_calculate_bet(100, 0.5), 50)
        self.assertEqual(_calculate_bet(100, -1), 0)
        self.assertEqual(_calculate_bet(100, 2), 100)

    def test_filter_markets_populates_skip_counters(self) -> None:
        now = datetime.now(timezone.utc)
        markets = [
            Market(id="open", question="Open market", category="sports", liquidity_usdc=200, close_time=now + timedelta(days=2)),
            Market(id="low", question="Low liquidity", category="sports", liquidity_usdc=10, close_time=now + timedelta(days=2)),
            Market(id="blocked", question="Blocked category", category="politics", liquidity_usdc=200, close_time=now + timedelta(days=2)),
            Market(id="soon", question="Closing soon", category="sports", liquidity_usdc=200, close_time=now + timedelta(hours=4)),
        ]
        stats: dict[str, int] = {}
        filtered = _filter_markets(
            markets,
            min_liquidity=100,
            allowlist=(),
            blocklist=("politics",),
            min_close_days=1,
            stats=stats,
        )
        self.assertEqual([m.id for m in filtered], ["open"])
        self.assertEqual(stats["kept"], 1)
        self.assertEqual(stats["skipped_liquidity"], 1)
        self.assertEqual(stats["skipped_blocklist"], 1)
        self.assertEqual(stats["skipped_close_too_soon"], 1)

    def test_filter_markets_applies_ticker_prefix_blocklist(self) -> None:
        markets = [
            Market(id="KXBTC15M-26APR061800-00", question="15m market", liquidity_usdc=200),
            Market(id="KXBTCD-26APR0717-T70000", question="Daily market", liquidity_usdc=200),
        ]
        stats: dict[str, int] = {}
        filtered = _filter_markets(
            markets,
            min_liquidity=100,
            allowlist=(),
            blocklist=(),
            ticker_prefix_blocklist=("KXBTC15M-",),
            stats=stats,
        )
        self.assertEqual([m.id for m in filtered], ["KXBTCD-26APR0717-T70000"])
        self.assertEqual(stats["skipped_ticker_prefix_blocklist"], 1)

    def test_filter_markets_treats_null_liquidity_as_zero(self) -> None:
        markets = [
            Market(id="null-liq", question="Null liquidity", liquidity_usdc=None),
            Market(id="ok-liq", question="Sufficient liquidity", liquidity_usdc=150),
        ]
        filtered = _filter_markets(
            markets,
            min_liquidity=100,
            allowlist=(),
            blocklist=(),
        )
        self.assertEqual([m.id for m in filtered], ["ok-liq"])

    def test_filter_markets_applies_volume_and_extreme_price_filters(self) -> None:
        markets = [
            Market(
                id="low-volume",
                question="Low volume",
                outcomes=[MarketOutcome(name="YES", price=0.50), MarketOutcome(name="NO", price=0.50)],
                liquidity_usdc=200,
                volume_24h=5,
            ),
            Market(
                id="extreme-price",
                question="Extreme price",
                outcomes=[MarketOutcome(name="YES", price=0.99), MarketOutcome(name="NO", price=0.01)],
                liquidity_usdc=200,
                volume_24h=100,
            ),
            Market(
                id="kept",
                question="Kept market",
                outcomes=[MarketOutcome(name="YES", price=0.52), MarketOutcome(name="NO", price=0.48)],
                liquidity_usdc=200,
                volume_24h=100,
            ),
        ]
        stats: dict[str, int] = {}
        filtered = _filter_markets(
            markets,
            min_liquidity=100,
            allowlist=(),
            blocklist=(),
            stats=stats,
            min_volume_24h=10,
            extreme_yes_price_lower=0.05,
            extreme_yes_price_upper=0.95,
        )
        self.assertEqual([m.id for m in filtered], ["kept"])
        self.assertEqual(stats["skipped_volume_24h"], 1)
        self.assertEqual(stats["skipped_extreme_price"], 1)

    def test_collapse_event_ladders_keeps_most_informative_brackets(self) -> None:
        event_markets = [
            Market(id="m1", event_ticker="E1", question="Q", outcomes=[MarketOutcome(name="YES", price=0.10)]),
            Market(id="m2", event_ticker="E1", question="Q", outcomes=[MarketOutcome(name="YES", price=0.49)]),
            Market(id="m3", event_ticker="E1", question="Q", outcomes=[MarketOutcome(name="YES", price=0.51)]),
            Market(id="m4", event_ticker="E1", question="Q", outcomes=[MarketOutcome(name="YES", price=0.90)]),
            Market(id="m5", event_ticker="E1", question="Q", outcomes=[MarketOutcome(name="YES", price=0.70)]),
            Market(id="m6", event_ticker="E2", question="Q", outcomes=[MarketOutcome(name="YES", price=0.30)]),
            Market(id="no-event", question="Q", outcomes=[MarketOutcome(name="YES", price=0.60)]),
        ]
        collapsed = _collapse_event_ladders(
            event_markets,
            ladder_collapse_threshold=4,
            max_brackets_per_event=3,
        )
        collapsed_ids = {market.id for market in collapsed}
        self.assertEqual(len([m for m in collapsed if m.event_ticker == "E1"]), 3)
        self.assertIn("m2", collapsed_ids)
        self.assertIn("m3", collapsed_ids)
        self.assertIn("m5", collapsed_ids)
        self.assertIn("m6", collapsed_ids)
        self.assertIn("no-event", collapsed_ids)

    def test_cap_analysis_candidates_limits_list_size(self) -> None:
        candidates = [{"market": f"m{i}"} for i in range(6)]
        capped = _cap_analysis_candidates(candidates, max_markets_per_cycle=3)
        self.assertEqual(len(capped), 3)
        self.assertEqual(capped[0]["market"], "m0")
        self.assertEqual(capped[-1]["market"], "m2")

    def test_best_orderbook_sell_price(self) -> None:
        orderbook = {
            "sells": [
                {"optionIndex": 0, "price": 0.62},
                {"optionIndex": 1, "price": 0.44},
                {"optionIndex": 0, "price": 0.60},
            ]
        }
        self.assertAlmostEqual(_best_orderbook_sell_price(orderbook, 0) or 0.0, 0.60)
        self.assertAlmostEqual(_best_orderbook_sell_price(orderbook, 1) or 0.0, 0.44)
        self.assertIsNone(_best_orderbook_sell_price(orderbook, 2))

    def test_log_settings_summary_includes_phase1_flags(self) -> None:
        settings = Settings(
            BAYESIAN_ENABLED=False,
            LMSR_ENABLED=False,
            KELLY_SIZING_ENABLED=True,
            KELLY_FRACTION_DEFAULT=0.2,
            KELLY_FRACTION_SHORT_HORIZON_HOURS=1,
            KELLY_FRACTION_SHORT_HORIZON=0.1,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        with patch("main.logger.info") as info_mock:
            _log_settings_summary(settings)

        self.assertTrue(info_mock.called)
        summary_data = {}
        strict_hint_data = {}
        for call in info_mock.call_args_list:
            data = call.kwargs.get("data") or {}
            if "dry_run" in data:
                summary_data = data
            if "effective_min_bet_pct" in data:
                strict_hint_data = data
        data = summary_data
        self.assertEqual(data.get("bayesian_enabled"), False)
        self.assertEqual(data.get("lmsr_enabled"), False)
        self.assertEqual(data.get("kelly_sizing_enabled"), True)
        self.assertEqual(data.get("kelly_fraction_default"), 0.2)
        self.assertEqual(data.get("kelly_fraction_short_horizon_hours"), 1)
        self.assertEqual(data.get("kelly_fraction_short_horizon"), 0.1)
        self.assertEqual(data.get("kelly_min_bet_policy"), "fallback_edge_scaling")
        self.assertGreater(strict_hint_data.get("effective_min_bet_pct", 0.0), 0.0)

    def test_compute_next_wakeup_seconds_uses_action_aware_cooldown(self) -> None:
        now = datetime.now(timezone.utc)
        market = Market(
            id="m-cooldown",
            question="Cooldown test",
            outcomes=[MarketOutcome(name="YES"), MarketOutcome(name="NO")],
            close_time=now + timedelta(days=2),
        )
        state = MarketState(
            market_id="m-cooldown",
            last_analysis=now - timedelta(minutes=20),
            analysis_count=1,
            last_confidence=0.55,
            confidence_trend=[0.55],
            last_terminal_outcome="no_trade_recommended",
        )
        state_manager = DummyStateManager({"m-cooldown": state})
        settings = Settings(
            REANALYSIS_COOLDOWN_HOURS=6,
            URGENT_REANALYSIS_DAYS_BEFORE_CLOSE=1,
            URGENT_REANALYSIS_COOLDOWN_HOURS=1,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        wakeup_seconds = _compute_next_wakeup_seconds(
            [market],
            state_manager,
            settings,
            now=now,
        )
        self.assertEqual(wakeup_seconds, 1)

    def test_cap_effective_confidence_for_market_respects_category_caps(self) -> None:
        settings = Settings(
            MAX_SPORTS_CONFIDENCE=0.80,
            MAX_ESPORTS_CONFIDENCE=0.75,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        sports_market = Market(id="s1", question="NBA: A vs B", category="sports")
        esports_market = Market(id="e1", question="Esports: A vs B", category="esports")
        politics_market = Market(id="p1", question="Election", category="politics")

        self.assertEqual(
            _cap_effective_confidence_for_market(0.99, sports_market, settings),
            0.80,
        )
        self.assertEqual(
            _cap_effective_confidence_for_market(0.99, esports_market, settings),
            0.75,
        )
        self.assertEqual(
            _cap_effective_confidence_for_market(0.99, politics_market, settings),
            0.99,
        )

    def test_edge_threshold_applies_fallback_and_coinflip_guards(self) -> None:
        settings = Settings(
            MIN_EDGE=0.05,
            LOW_PRICE_MIN_EDGE=0.08,
            LOW_PRICE_THRESHOLD=0.50,
            COINFLIP_PRICE_LOWER=0.45,
            COINFLIP_PRICE_UPPER=0.55,
            FALLBACK_EDGE_MIN_EDGE=0.08,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        self.assertEqual(_edge_threshold_for_market(0.60, settings, "computed"), 0.05)
        self.assertEqual(_edge_threshold_for_market(0.52, settings, "computed"), 0.08)
        self.assertEqual(_edge_threshold_for_market(0.60, settings, "fallback"), 0.08)

    def test_should_adjust_position_uses_bankroll_relative_cap(self) -> None:
        settings = Settings(
            MAX_POSITION_PER_MARKET_USDC=200.0,
            MAX_POSITION_PCT_OF_BANKROLL=0.15,
            MAX_BET_USDC=50.0,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.70,
            bet_size_pct=1.0,
            reasoning="test",
        )
        existing_position = Position(
            market_id="m-bankroll",
            outcome="YES",
            total_amount_usdc=2.5,
            avg_confidence=0.60,
            trade_count=1,
            first_trade=datetime.now(timezone.utc),
            last_trade=datetime.now(timezone.utc),
        )
        allowed, bet_pct, reason = _should_adjust_position(
            decision=decision,
            market=Market(id="m-bankroll", question="Q", category="sports"),
            existing_position=existing_position,
            state=None,
            settings=settings,
            cycle_bankroll=20.0,
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "confidence_increase_threshold_met")
        self.assertAlmostEqual(bet_pct, 0.01, places=4)

    def test_build_reasoning_hash_ignores_validated_prefix_variation(self) -> None:
        decision_a = TradeDecision(
            should_trade=False,
            outcome="Yes",
            confidence=0.70,
            bet_size_pct=0.0,
            reasoning=(
                "[Validated eq=1.00 gate=allow reason=ok edge_market=0.041 "
                "edge_source=computed] Core thesis unchanged"
            ),
        )
        decision_b = TradeDecision(
            should_trade=False,
            outcome="Yes",
            confidence=0.70,
            bet_size_pct=0.0,
            reasoning=(
                "[Validated eq=0.95 gate=allow reason=ok edge_market=0.038 "
                "edge_source=computed] Core thesis unchanged"
            ),
        )
        self.assertEqual(_build_reasoning_hash(decision_a), _build_reasoning_hash(decision_b))

    def test_effective_position_override_threshold_not_capped_by_category(self) -> None:
        settings = Settings(
            HIGH_CONFIDENCE_POSITION_OVERRIDE=0.85,
            MAX_SPORTS_CONFIDENCE=0.80,
            XAI_API_KEY="xai-key",
            KALSHI_API_KEY_ID="kalshi-key-id",
            KALSHI_PRIVATE_KEY_PATH="kalshi-scope.txt",
        )
        sports_market = Market(id="s2", question="NBA: A vs B", category="sports")
        threshold = _effective_position_override_threshold(sports_market, settings)
        self.assertEqual(threshold, 0.85)
        self.assertFalse(0.80 >= threshold)

    def test_confidence_gate_override_metrics_prefers_stronger_edge(self) -> None:
        market = Market(
            id="m-override",
            question="Will Team A win?",
            outcomes=[MarketOutcome(name="YES", price=0.10), MarketOutcome(name="NO", price=0.90)],
        )
        decision = TradeDecision(
            should_trade=True,
            outcome="YES",
            confidence=0.40,
            bet_size_pct=0.5,
            reasoning="test",
            edge_external=0.12,
        )
        override_edge, market_edge = _confidence_gate_override_metrics(market, decision)
        self.assertAlmostEqual(market_edge or 0.0, 0.30)
        self.assertAlmostEqual(override_edge or 0.0, 0.30)

if __name__ == "__main__":
    unittest.main()
