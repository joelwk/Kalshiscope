import os
import unittest
from pathlib import Path
from unittest.mock import patch

import config


class TestConfig(unittest.TestCase):
    def _required_env(self) -> dict[str, str]:
        return {
            "XAI_API_KEY": "xai-key",
            "KALSHI_API_KEY_ID": "kalshi-key-id",
            "KALSHI_PRIVATE_KEY_PATH": "kalshi-scope.txt",
        }

    def test_load_settings_success(self) -> None:
        env = {
            **self._required_env(),
            "MARKET_CATEGORIES_ALLOWLIST": "sports, politics",
            "MARKET_CATEGORIES_BLOCKLIST": "crypto",
            "MIN_BET_USDC": "10",
            "MAX_BET_USDC": "75",
            "DRY_RUN": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()

        self.assertEqual(settings.XAI_API_KEY, self._required_env()["XAI_API_KEY"])
        self.assertEqual(settings.KALSHI_API_KEY_ID, self._required_env()["KALSHI_API_KEY_ID"])
        self.assertEqual(
            settings.KALSHI_PRIVATE_KEY_PATH,
            self._required_env()["KALSHI_PRIVATE_KEY_PATH"],
        )
        self.assertEqual(settings.MARKET_CATEGORIES_ALLOWLIST, ("sports", "politics"))
        self.assertEqual(settings.MARKET_CATEGORIES_BLOCKLIST, ("crypto",))
        self.assertEqual(settings.MIN_BET_USDC, 10.0)
        self.assertEqual(settings.MAX_BET_USDC, 75.0)
        self.assertFalse(settings.DRY_RUN)

    def test_close_days_filter_settings(self) -> None:
        env = {
            **self._required_env(),
            "MARKET_MIN_CLOSE_DAYS": "1",
            "MARKET_MAX_CLOSE_DAYS": "7",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()

        self.assertEqual(settings.MARKET_MIN_CLOSE_DAYS, 1)
        self.assertEqual(settings.MARKET_MAX_CLOSE_DAYS, 7)

    def test_market_filtering_tuning_settings_overrides(self) -> None:
        env = {
            **self._required_env(),
            "MIN_VOLUME_24H": "1250",
            "EXTREME_YES_PRICE_LOWER": "0.04",
            "EXTREME_YES_PRICE_UPPER": "0.96",
            "MIN_TRADEABLE_IMPLIED_PRICE": "0.06",
            "MAX_TRADEABLE_IMPLIED_PRICE": "0.94",
            "MARKET_FAMILY_BLOCKLIST": "weather, crypto",
            "LADDER_COLLAPSE_THRESHOLD": "7",
            "MAX_BRACKETS_PER_EVENT": "4",
            "MAX_MARKETS_PER_CYCLE": "80",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()

        self.assertEqual(settings.MIN_VOLUME_24H, 1250.0)
        self.assertEqual(settings.EXTREME_YES_PRICE_LOWER, 0.04)
        self.assertEqual(settings.EXTREME_YES_PRICE_UPPER, 0.96)
        self.assertEqual(settings.MIN_TRADEABLE_IMPLIED_PRICE, 0.06)
        self.assertEqual(settings.MAX_TRADEABLE_IMPLIED_PRICE, 0.94)
        self.assertEqual(settings.MARKET_FAMILY_BLOCKLIST, ("weather", "crypto"))
        self.assertEqual(settings.LADDER_COLLAPSE_THRESHOLD, 7)
        self.assertEqual(settings.MAX_BRACKETS_PER_EVENT, 4)
        self.assertEqual(settings.MAX_MARKETS_PER_CYCLE, 80)

    def test_close_days_filter_defaults_to_none(self) -> None:
        env = self._required_env()
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()

        self.assertIsNone(settings.MARKET_MIN_CLOSE_DAYS)
        self.assertIsNone(settings.MARKET_MAX_CLOSE_DAYS)

    def test_dry_run_and_no_blockchain_flags_exist(self) -> None:
        env = {**self._required_env(), "DRY_RUN": "true"}
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()
        self.assertTrue(settings.DRY_RUN)
        self.assertTrue(settings.PRE_ORDER_MARKET_REFRESH)
        self.assertEqual(settings.MAX_MARKET_DATA_AGE_SECONDS, 120)

    def test_missing_required_env_raises(self) -> None:
        env = {"XAI_API_KEY": "xai-key"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                config.load_settings()

    def test_search_settings_overrides(self) -> None:
        env = {
            **self._required_env(),
            "SEARCH_LOOKBACK_HOURS": "12",
            "SEARCH_ALLOWED_DOMAINS": "example.com, news.example",
            "SEARCH_ALLOWED_X_HANDLES": "Foo, Bar",
            "MULTIMEDIA_CONFIDENCE_THRESHOLD": "0.60, 0.70",
            "SEARCH_PROFILE_MAX_DOMAINS": "4",
            "SEARCH_PROFILE_MAX_X_HANDLES": "8",
            "EXTENDED_RESEARCH_SOURCE_OFFSET": "4",
            "EXTENDED_RESEARCH_X_HANDLE_OFFSET": "8",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()

        self.assertEqual(settings.SEARCH_LOOKBACK_HOURS, 12)
        self.assertEqual(settings.SEARCH_ALLOWED_DOMAINS, ("example.com", "news.example"))
        self.assertEqual(settings.SEARCH_ALLOWED_X_HANDLES, ("Foo", "Bar"))
        self.assertEqual(settings.MULTIMEDIA_CONFIDENCE_THRESHOLD, (0.6, 0.7))
        self.assertEqual(settings.SEARCH_PROFILE_MAX_DOMAINS, 4)
        self.assertEqual(settings.SEARCH_PROFILE_MAX_X_HANDLES, 8)
        self.assertEqual(settings.EXTENDED_RESEARCH_SOURCE_OFFSET, 4)
        self.assertEqual(settings.EXTENDED_RESEARCH_X_HANDLE_OFFSET, 8)

    def test_search_profile_limits_clamp_to_xai_provider_caps(self) -> None:
        env = {
            **self._required_env(),
            "SEARCH_PROFILE_MAX_DOMAINS": "8",
            "SEARCH_PROFILE_MAX_X_HANDLES": "14",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()

        self.assertEqual(settings.SEARCH_PROFILE_MAX_DOMAINS, 5)
        self.assertEqual(settings.SEARCH_PROFILE_MAX_X_HANDLES, 10)

    def test_weather_profile_settings_overrides(self) -> None:
        env = {
            **self._required_env(),
            "WEATHER_ALLOWED_DOMAINS": "weather.gov,weather.com",
            "WEATHER_ALLOWED_X_HANDLES": "NWS,weatherchannel",
            "SKIP_WEATHER_BIN_MARKETS": "false",
            "MAX_WEATHER_CONFIDENCE": "0.78",
            "WEATHER_MIN_EDGE": "0.09",
            "WEATHER_SCORE_PENALTY": "0.04",
            "WEATHER_MIN_EVIDENCE_QUALITY": "0.72",
            "SPORTS_MIN_EVIDENCE_QUALITY": "0.61",
            "WEATHER_FALLBACK_EDGE_MIN_EDGE": "0.18",
            "DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_SPORTS": "0.59",
            "MAX_WEATHER_CANDIDATES_PER_CYCLE": "2",
            "KELLY_FRACTION_WEATHER": "0.45",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()
        self.assertEqual(settings.WEATHER_ALLOWED_DOMAINS, ("weather.gov", "weather.com"))
        self.assertEqual(settings.WEATHER_ALLOWED_X_HANDLES, ("NWS", "weatherchannel"))
        self.assertFalse(settings.SKIP_WEATHER_BIN_MARKETS)
        self.assertEqual(settings.MAX_WEATHER_CONFIDENCE, 0.78)
        self.assertEqual(settings.WEATHER_MIN_EDGE, 0.09)
        self.assertEqual(settings.WEATHER_SCORE_PENALTY, 0.04)
        self.assertEqual(settings.WEATHER_MIN_EVIDENCE_QUALITY, 0.72)
        self.assertEqual(settings.SPORTS_MIN_EVIDENCE_QUALITY, 0.61)
        self.assertEqual(settings.WEATHER_FALLBACK_EDGE_MIN_EDGE, 0.18)
        self.assertEqual(settings.DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_SPORTS, 0.59)
        self.assertEqual(settings.MAX_WEATHER_CANDIDATES_PER_CYCLE, 2)
        self.assertEqual(settings.KELLY_FRACTION_WEATHER, 0.45)

    def test_weather_profitability_defaults_are_conservative(self) -> None:
        self.assertEqual(config.Settings.MAX_WEATHER_CONFIDENCE, 0.65)
        self.assertEqual(config.Settings.WEATHER_SCORE_PENALTY, 0.12)
        self.assertEqual(config.Settings.WEATHER_MIN_EVIDENCE_QUALITY, 0.80)
        self.assertEqual(config.Settings.SPORTS_MIN_EVIDENCE_QUALITY, 0.55)
        self.assertEqual(config.Settings.WEATHER_MIN_EDGE, 0.14)
        self.assertEqual(config.Settings.WEATHER_FALLBACK_EDGE_MIN_EDGE, 0.34)
        self.assertEqual(config.Settings.SCORE_GATE_THRESHOLD_WEATHER_DIRECT, 0.12)
        self.assertEqual(config.Settings.MAX_WEATHER_CANDIDATES_PER_CYCLE, 1)

    def test_profit_tuning_defaults_are_loaded(self) -> None:
        self.assertEqual(config.Settings.LOW_PRICE_MIN_EDGE, 0.18)
        self.assertEqual(config.Settings.VERY_LOW_PRICE_THRESHOLD, 0.25)
        self.assertEqual(config.Settings.VERY_LOW_PRICE_MIN_EDGE, 0.28)
        self.assertEqual(config.Settings.LOW_PRICE_MIN_EDGE_MULTIPLIER, 0.85)
        self.assertEqual(config.Settings.MIN_EDGE_HIGH_LIQUIDITY_MULTIPLIER, 0.70)
        self.assertEqual(config.Settings.MIN_EDGE_MEDIUM_LIQUIDITY_MULTIPLIER, 0.85)
        self.assertEqual(config.Settings.MIN_TRADEABLE_IMPLIED_PRICE, 0.12)
        self.assertEqual(config.Settings.SCORE_GATE_THRESHOLD, 0.52)

    def test_env_example_profit_thresholds_match_conservative_defaults(self) -> None:
        env_example = Path(".env.example").read_text(encoding="utf-8")
        expected_lines = {
            "MIN_EDGE=0.12",
            "SPORTS_MIN_EVIDENCE_QUALITY=0.55",
            "LOW_PRICE_MIN_EDGE=0.18",
            "VERY_LOW_PRICE_MIN_EDGE=0.28",
            "LOW_PRICE_MIN_EDGE_MULTIPLIER=0.85",
            "FALLBACK_EDGE_MIN_EDGE=0.25",
            "FALLBACK_EDGE_MIN_EDGE_MULTIPLIER=0.90",
            "MAX_REASONABLE_EDGE=0.40",
            "DEFINITIVE_OUTCOME_EDGE_REASONABLE_MAX=0.50",
            "HIGH_QUALITY_SETTLED_EVIDENCE_MIN_EQ=0.95",
            "CONFIDENCE_GATE_MIN_EDGE=0.08",
            "MIN_EVIDENCE_QUALITY_FOR_TRADE=0.55",
            "SCORE_GATE_THRESHOLD=0.48",
            "SCORE_GATE_THRESHOLD_DIRECT_HIGH_QUALITY=0.40",
            "MAX_MARKETS_PER_CYCLE=20",
            "MAX_TRADES_PER_CYCLE=4",
            "MAX_TRADES_PER_DAY=8",
            "KALSHI_MVE_FILTER=exclude",
            "KALSHI_ELIGIBLE_FLOOR=100",
            "KALSHI_FETCH_TOPUP_ENABLED=false",
            "KELLY_DYNAMIC_ENABLED=true",
            "KELLY_FRACTION_DEFAULT=0.45",
            "GROK_SELF_CONSISTENCY_ENABLED=true",
            "GROK_ANALYSIS_MAX_BUDGET_SECONDS=420",
            "GROK_SELF_CONSISTENCY_LIQUIDITY_THRESHOLD=400",
            "GROK_SELF_CONSISTENCY_EDGE_THRESHOLD=0.15",
            "CALIBRATION_ONLINE_UPDATE_ENABLED=true",
            "CALIBRATION_ONLINE_ALPHA=0.15",
            "CALIBRATION_ONLINE_MAX_SAMPLES_PER_BUCKET=500",
            "SEARCH_PROFILE_MAX_DOMAINS=5",
            "EXTENDED_RESEARCH_SOURCE_OFFSET=5",
            "DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_SPORTS=0.60",
            "PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE=0.28",
            "PRE_ANALYSIS_ADAPTIVE_BOOST=0.03",
            "PRE_ANALYSIS_REDUCED_MAX_CANDIDATES=8",
            "RESEARCH_QUEUE_DRAIN_PER_CYCLE=8",
            "RESEARCH_QUEUE_DRAIN_MIN_PRIORITY=0.40",
            "RESEARCH_QUEUE_DRAIN_FORCE_EXTENDED_RESEARCH=true",
        }
        for expected_line in expected_lines:
            self.assertIn(expected_line, env_example)

    def test_historical_family_penalty_settings_overrides(self) -> None:
        env = {
            **self._required_env(),
            "PRE_ANALYSIS_HISTORICAL_FAMILY_MIN_SAMPLES": "12",
            "PRE_ANALYSIS_HISTORICAL_FAMILY_WIN_RATE_THRESHOLD": "0.40",
            "PRE_ANALYSIS_HISTORICAL_FAMILY_PENALTY": "0.15",
            "PRE_ANALYSIS_ADAPTIVE_BOOST": "0.05",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()
        self.assertEqual(settings.PRE_ANALYSIS_HISTORICAL_FAMILY_MIN_SAMPLES, 12)
        self.assertEqual(settings.PRE_ANALYSIS_HISTORICAL_FAMILY_WIN_RATE_THRESHOLD, 0.40)
        self.assertEqual(settings.PRE_ANALYSIS_HISTORICAL_FAMILY_PENALTY, 0.15)
        self.assertEqual(settings.PRE_ANALYSIS_ADAPTIVE_BOOST, 0.05)

    def test_weather_profile_defaults_include_official_sources(self) -> None:
        self.assertIn("weather.gov", config.Settings.WEATHER_ALLOWED_DOMAINS)
        self.assertIn("forecast.weather.gov", config.Settings.WEATHER_ALLOWED_DOMAINS)
        self.assertIn("noaa.gov", config.Settings.WEATHER_ALLOWED_DOMAINS)
        self.assertIn("NWS", config.Settings.WEATHER_ALLOWED_X_HANDLES)
        self.assertIn("NWSSPC", config.Settings.WEATHER_ALLOWED_X_HANDLES)
        self.assertIn("NHC_Atlantic", config.Settings.WEATHER_ALLOWED_X_HANDLES)

    def test_entertainment_profile_settings_overrides(self) -> None:
        env = {
            **self._required_env(),
            "ENTERTAINMENT_ALLOWED_DOMAINS": "netflix.com,flixpatrol.com",
            "ENTERTAINMENT_ALLOWED_X_HANDLES": "Netflix,flixpatrol",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()
        self.assertEqual(
            settings.ENTERTAINMENT_ALLOWED_DOMAINS,
            ("netflix.com", "flixpatrol.com"),
        )
        self.assertEqual(
            settings.ENTERTAINMENT_ALLOWED_X_HANDLES,
            ("Netflix", "flixpatrol"),
        )

    def test_build_search_config(self) -> None:
        env = {
            **self._required_env(),
            "SEARCH_LOOKBACK_HOURS": "6",
            "SEARCH_ALLOWED_DOMAINS": "example.com",
            "SEARCH_ALLOWED_X_HANDLES": "Foo",
            "MULTIMEDIA_CONFIDENCE_THRESHOLD": "0.55,0.75",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()

        search_config = config.build_search_config(settings)
        self.assertIsInstance(search_config, config.SearchConfig)
        self.assertEqual(search_config.allowed_domains, ["example.com"])
        self.assertEqual(search_config.allowed_x_handles, ["Foo"])
        self.assertEqual(search_config.multimedia_confidence_range, (0.55, 0.75))
        self.assertIsNotNone(search_config.from_date)
        self.assertIsNotNone(search_config.to_date)
        delta_hours = (search_config.to_date - search_config.from_date).total_seconds() / 3600
        self.assertTrue(5.9 <= delta_hours <= 6.1)

    def test_flip_guard_settings_overrides(self) -> None:
        env = {
            **self._required_env(),
            "FLIP_GUARD_ENABLED": "false",
            "FLIP_GUARD_MIN_ABS_CONFIDENCE": "0.70",
            "FLIP_GUARD_MIN_CONF_GAIN": "0.10",
            "FLIP_GUARD_MIN_EDGE_GAIN": "0.05",
            "FLIP_GUARD_MIN_EVIDENCE_QUALITY": "0.75",
            "FLIP_CIRCUIT_BREAKER_ENABLED": "false",
            "FLIP_CIRCUIT_BREAKER_MAX_FLIPS": "5",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()

        self.assertFalse(settings.FLIP_GUARD_ENABLED)
        self.assertEqual(settings.FLIP_GUARD_MIN_ABS_CONFIDENCE, 0.70)
        self.assertEqual(settings.FLIP_GUARD_MIN_CONF_GAIN, 0.10)
        self.assertEqual(settings.FLIP_GUARD_MIN_EDGE_GAIN, 0.05)
        self.assertEqual(settings.FLIP_GUARD_MIN_EVIDENCE_QUALITY, 0.75)
        self.assertFalse(settings.FLIP_CIRCUIT_BREAKER_ENABLED)
        self.assertEqual(settings.FLIP_CIRCUIT_BREAKER_MAX_FLIPS, 5)

    def test_parallel_and_execution_guard_settings_overrides(self) -> None:
        env = {
            **self._required_env(),
            "PARALLEL_ANALYSIS_ENABLED": "true",
            "ANALYSIS_MAX_WORKERS": "4",
            "MAX_MARKETS_PER_CYCLE": "25",
            "MAX_TRADES_PER_CYCLE": "6",
            "MAX_BETS_PER_EVENT": "3",
            "MAX_TRADES_PER_DAY": "18",
            "MAX_DAILY_DRAWDOWN_USDC": "22",
            "XAI_CIRCUIT_BREAKER_MAX_FAILURES": "4",
            "KALSHI_MAX_FETCH_PAGES": "12",
            "XAI_CLIENT_TIMEOUT_SECONDS": "75",
            "GROK_STREAM_TIMEOUT_SECONDS": "80",
            "GROK_ANALYSIS_MAX_BUDGET_SECONDS": "55",
            "GROK_SELF_CONSISTENCY_ENABLED": "false",
            "GROK_SELF_CONSISTENCY_LIQUIDITY_THRESHOLD": "425",
            "GROK_SELF_CONSISTENCY_EDGE_THRESHOLD": "0.18",
            "GROK_SELF_CONSISTENCY_PRIMARY_TEMPERATURE": "0.25",
            "GROK_SELF_CONSISTENCY_SECONDARY_TEMPERATURE": "0.85",
            "PRE_ORDER_MARKET_REFRESH": "true",
            "ORDERBOOK_PRECHECK_ENABLED": "true",
            "ORDERBOOK_PRECHECK_MIN_CONFIDENCE": "0.8",
            "ORDER_SUBMISSION_MIN_PRICE": "0.04",
            "ORDER_SUBMISSION_MAX_PRICE": "0.96",
            "ORDER_FALLBACK_TO_MARKET": "false",
            "ORDER_FALLBACK_MIN_CONFIDENCE": "0.9",
            "EVIDENCE_QUALITY_HIGH_CONFIDENCE_OVERRIDE": "false",
            "CALIBRATION_MODE_ENABLED": "true",
            "CALIBRATION_MIN_SAMPLES": "25",
            "CALIBRATION_ONLINE_UPDATE_ENABLED": "false",
            "CALIBRATION_ONLINE_ALPHA": "0.2",
            "CALIBRATION_ONLINE_MAX_SAMPLES_PER_BUCKET": "250",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()

        self.assertTrue(settings.PARALLEL_ANALYSIS_ENABLED)
        self.assertEqual(settings.ANALYSIS_MAX_WORKERS, 4)
        self.assertEqual(settings.MAX_MARKETS_PER_CYCLE, 25)
        self.assertEqual(settings.MAX_TRADES_PER_CYCLE, 6)
        self.assertEqual(settings.MAX_BETS_PER_EVENT, 3)
        self.assertEqual(settings.MAX_TRADES_PER_DAY, 18)
        self.assertEqual(settings.MAX_DAILY_DRAWDOWN_USDC, 22.0)
        self.assertEqual(settings.XAI_CIRCUIT_BREAKER_MAX_FAILURES, 4)
        self.assertEqual(settings.KALSHI_MAX_FETCH_PAGES, 12)
        self.assertEqual(settings.XAI_CLIENT_TIMEOUT_SECONDS, 75)
        self.assertEqual(settings.GROK_STREAM_TIMEOUT_SECONDS, 80)
        self.assertEqual(settings.GROK_ANALYSIS_MAX_BUDGET_SECONDS, 55)
        self.assertFalse(settings.GROK_SELF_CONSISTENCY_ENABLED)
        self.assertEqual(settings.GROK_SELF_CONSISTENCY_LIQUIDITY_THRESHOLD, 425.0)
        self.assertEqual(settings.GROK_SELF_CONSISTENCY_EDGE_THRESHOLD, 0.18)
        self.assertEqual(settings.GROK_SELF_CONSISTENCY_PRIMARY_TEMPERATURE, 0.25)
        self.assertEqual(settings.GROK_SELF_CONSISTENCY_SECONDARY_TEMPERATURE, 0.85)
        self.assertTrue(settings.PRE_ORDER_MARKET_REFRESH)
        self.assertTrue(settings.ORDERBOOK_PRECHECK_ENABLED)
        self.assertEqual(settings.ORDERBOOK_PRECHECK_MIN_CONFIDENCE, 0.8)
        self.assertEqual(settings.ORDER_SUBMISSION_MIN_PRICE, 0.04)
        self.assertEqual(settings.ORDER_SUBMISSION_MAX_PRICE, 0.96)
        self.assertFalse(settings.ORDER_FALLBACK_TO_MARKET)
        self.assertEqual(settings.ORDER_FALLBACK_MIN_CONFIDENCE, 0.9)
        self.assertFalse(settings.EVIDENCE_QUALITY_HIGH_CONFIDENCE_OVERRIDE)
        self.assertTrue(settings.CALIBRATION_MODE_ENABLED)
        self.assertEqual(settings.CALIBRATION_MIN_SAMPLES, 25)
        self.assertFalse(settings.CALIBRATION_ONLINE_UPDATE_ENABLED)
        self.assertEqual(settings.CALIBRATION_ONLINE_ALPHA, 0.2)
        self.assertEqual(settings.CALIBRATION_ONLINE_MAX_SAMPLES_PER_BUCKET, 250)

    def test_bayesian_lmsr_kelly_settings_overrides(self) -> None:
        env = {
            **self._required_env(),
            "BAYESIAN_ENABLED": "true",
            "BAYESIAN_SKIP_STALE_UPDATES": "false",
            "BAYESIAN_PRIOR_DEFAULT": "0.58",
            "BAYESIAN_MIN_UPDATES_FOR_TRADE": "3",
            "LMSR_ENABLED": "true",
            "LMSR_LIQUIDITY_PARAM_B": "120000",
            "LMSR_MIN_INEFFICIENCY": "0.04",
            "KELLY_SIZING_ENABLED": "true",
            "KELLY_DYNAMIC_ENABLED": "false",
            "KELLY_FRACTION_DEFAULT": "0.2",
            "KELLY_FRACTION_SHORT_HORIZON_HOURS": "2",
            "KELLY_FRACTION_SHORT_HORIZON": "0.1",
            "KELLY_MIN_BET_POLICY": "floor",
            "MAX_POSITION_PCT_OF_BANKROLL": "0.12",
            "COINFLIP_PRICE_LOWER": "0.46",
            "COINFLIP_PRICE_UPPER": "0.54",
            "FALLBACK_EDGE_MIN_EDGE": "0.09",
            "FALLBACK_EDGE_MIN_EDGE_MULTIPLIER": "0.8",
            "MIN_EDGE_HIGH_LIQUIDITY_MULTIPLIER": "0.6",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()

        self.assertTrue(settings.BAYESIAN_ENABLED)
        self.assertFalse(settings.BAYESIAN_SKIP_STALE_UPDATES)
        self.assertEqual(settings.BAYESIAN_PRIOR_DEFAULT, 0.58)
        self.assertEqual(settings.BAYESIAN_MIN_UPDATES_FOR_TRADE, 3)
        self.assertEqual(settings.BAYESIAN_MAX_POSTERIOR, 0.90)
        self.assertTrue(settings.LMSR_ENABLED)
        self.assertEqual(settings.LMSR_LIQUIDITY_PARAM_B, 120000.0)
        self.assertEqual(settings.LMSR_MIN_INEFFICIENCY, 0.04)
        self.assertTrue(settings.KELLY_SIZING_ENABLED)
        self.assertFalse(settings.KELLY_DYNAMIC_ENABLED)
        self.assertEqual(settings.KELLY_FRACTION_DEFAULT, 0.2)
        self.assertEqual(settings.KELLY_FRACTION_SHORT_HORIZON_HOURS, 2)
        self.assertEqual(settings.KELLY_FRACTION_SHORT_HORIZON, 0.1)
        self.assertEqual(settings.KELLY_MIN_BET_POLICY, "floor")
        self.assertEqual(settings.MAX_POSITION_PCT_OF_BANKROLL, 0.12)
        self.assertEqual(settings.COINFLIP_PRICE_LOWER, 0.46)
        self.assertEqual(settings.COINFLIP_PRICE_UPPER, 0.54)
        self.assertEqual(settings.FALLBACK_EDGE_MIN_EDGE, 0.09)
        self.assertEqual(settings.FALLBACK_EDGE_MIN_EDGE_MULTIPLIER, 0.8)
        self.assertEqual(settings.MIN_EDGE_HIGH_LIQUIDITY_MULTIPLIER, 0.6)

    def test_profit_guardrail_defaults(self) -> None:
        settings = config.Settings()
        self.assertEqual(settings.CONFIDENCE_GATE_MIN_EDGE, 0.08)
        self.assertEqual(settings.MIN_EVIDENCE_QUALITY_FOR_TRADE, 0.55)
        self.assertEqual(settings.SPORTS_MIN_EVIDENCE_QUALITY, 0.55)
        self.assertEqual(settings.DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_WEATHER, 0.72)
        self.assertEqual(settings.DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_SPORTS, 0.65)
        self.assertEqual(settings.DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_DEFAULT, 0.75)
        self.assertIn("weather.gov", settings.DIRECT_SOURCE_WHITELIST)
        self.assertEqual(settings.SCORE_GATE_MODE, "active")
        self.assertEqual(settings.SCORE_GATE_THRESHOLD, 0.52)
        self.assertEqual(settings.SCORE_LOW_INFO_PENALTY_THRESHOLD, 0.60)
        self.assertEqual(settings.SCORE_LOW_INFO_PENALTY_BASE, 0.08)
        self.assertEqual(settings.PRE_ANALYSIS_OPPORTUNITY_MIN_SCORE, 0.28)
        self.assertEqual(settings.PRE_ANALYSIS_REDUCED_MAX_CANDIDATES, 8)
        self.assertEqual(settings.PRE_ANALYSIS_ADAPTIVE_BOOST, 0.03)
        self.assertEqual(settings.MAX_MARKETS_PER_CYCLE, 20)
        self.assertEqual(settings.MAX_CRYPTO_CANDIDATES_PER_CYCLE, 1)
        self.assertEqual(settings.MAX_SPEECH_CANDIDATES_PER_CYCLE, 2)
        self.assertEqual(settings.MAX_MUSIC_CANDIDATES_PER_CYCLE, 2)
        self.assertEqual(settings.MAX_TRADES_PER_CYCLE, 4)
        self.assertEqual(settings.MAX_BETS_PER_EVENT, 2)
        self.assertEqual(settings.MAX_TRADES_PER_DAY, 6)
        self.assertEqual(settings.MAX_DAILY_DRAWDOWN_USDC, 30.0)
        self.assertTrue(settings.POSITION_SYNC_ENABLED)
        self.assertEqual(settings.POSITION_SYNC_INTERVAL_CYCLES, 3)
        self.assertEqual(settings.ORDER_PRICE_IMPROVEMENT_CENTS, 1)
        self.assertEqual(settings.ORDER_DEFAULT_TIF, "gtc")
        self.assertEqual(settings.ORDER_SUBMISSION_MIN_PRICE, 0.03)
        self.assertEqual(settings.ORDER_SUBMISSION_MAX_PRICE, 0.97)
        self.assertEqual(settings.MIN_TRADEABLE_IMPLIED_PRICE, 0.12)
        self.assertEqual(settings.MAX_TRADEABLE_IMPLIED_PRICE, 0.95)
        self.assertEqual(settings.KALSHI_MAX_FETCH_PAGES, 50)
        self.assertEqual(settings.XAI_CIRCUIT_BREAKER_MAX_FAILURES, 3)
        self.assertEqual(settings.XAI_CLIENT_TIMEOUT_SECONDS, 120)
        self.assertEqual(settings.GROK_STREAM_TIMEOUT_SECONDS, 75)
        self.assertEqual(settings.GROK_ANALYSIS_MAX_BUDGET_SECONDS, 420)
        self.assertTrue(settings.GROK_SELF_CONSISTENCY_ENABLED)
        self.assertEqual(settings.GROK_SELF_CONSISTENCY_LIQUIDITY_THRESHOLD, 400.0)
        self.assertEqual(settings.GROK_SELF_CONSISTENCY_EDGE_THRESHOLD, 0.15)
        self.assertEqual(settings.GROK_SELF_CONSISTENCY_PRIMARY_TEMPERATURE, 0.3)
        self.assertEqual(settings.GROK_SELF_CONSISTENCY_SECONDARY_TEMPERATURE, 0.7)
        self.assertTrue(settings.CALIBRATION_ONLINE_UPDATE_ENABLED)
        self.assertEqual(settings.CALIBRATION_ONLINE_ALPHA, 0.15)
        self.assertEqual(settings.CALIBRATION_ONLINE_MAX_SAMPLES_PER_BUCKET, 500)
        self.assertFalse(settings.EVIDENCE_QUALITY_HIGH_CONFIDENCE_OVERRIDE)
        self.assertEqual(settings.CONFIDENCE_GATE_OVERRIDE_MIN_CONFIDENCE, 0.58)
        self.assertEqual(settings.KELLY_MIN_BANKROLL_USDC, 40.0)
        self.assertEqual(settings.SCORE_REPEATED_ANALYSIS_PENALTY_BASE, 0.025)
        self.assertEqual(settings.SCORE_REPEATED_ANALYSIS_PENALTY_START_COUNT, 1)
        self.assertEqual(settings.SCORE_GENERIC_BIN_PENALTY_BASE, 0.015)
        self.assertEqual(settings.SCORE_AMBIGUOUS_RESOLUTION_PENALTY_BASE, 0.08)
        self.assertEqual(settings.SCORE_OVERCONFIDENCE_PENALTY_BASE, 0.05)
        self.assertTrue(settings.SCORE_VOLUME_AMPLIFIER_ENABLED)
        self.assertEqual(settings.MENTION_MARKET_SCORE_PENALTY, 0.10)
        self.assertTrue(settings.PRE_ANALYSIS_OPPORTUNITY_ENABLED)
        self.assertEqual(settings.PRE_ANALYSIS_NON_ACTIONABLE_STREAK_PENALTY, 0.25)
        self.assertEqual(settings.PRE_ANALYSIS_ANALYSIS_COUNT_PENALTY, 0.15)
        self.assertEqual(settings.GROK_PROXY_CONFIDENCE_CAP, 0.78)
        self.assertEqual(settings.GROK_LOW_INFO_CONFIDENCE_CAP, 0.70)
        self.assertEqual(settings.MAX_GLOBAL_CONFIDENCE, 0.82)
        self.assertEqual(settings.MAX_INDEX_CONFIDENCE, 0.70)
        self.assertEqual(settings.MAX_COMMODITY_CONFIDENCE, 0.78)
        self.assertEqual(settings.MAX_CRYPTO_CONFIDENCE, 0.72)
        self.assertEqual(settings.MAX_WEATHER_CONFIDENCE, 0.65)
        self.assertEqual(settings.MAX_REASONABLE_EDGE, 0.40)
        self.assertTrue(settings.NON_SPORTS_REQUIRES_DIRECT_EVIDENCE)
        self.assertTrue(settings.NON_SPORTS_REQUIRES_PRIMARY_SOURCE_URL)
        self.assertTrue(settings.DRY_STREAK_SLEEP_ENABLED)
        self.assertTrue(settings.HISTORICAL_TICKER_PREFIX_GATE_ENABLED)
        self.assertEqual(settings.HISTORICAL_TICKER_PREFIX_MIN_SAMPLES, 3)
        self.assertEqual(settings.HISTORICAL_TICKER_PREFIX_HARD_BLOCK_MIN_SAMPLES, 20)
        self.assertEqual(settings.HISTORICAL_TICKER_PREFIX_PNL_CUTOFF, -2.0)
        self.assertEqual(settings.HISTORICAL_TICKER_PREFIX_WIN_RATE_CUTOFF, 0.40)
        self.assertEqual(settings.HISTORICAL_TICKER_PREFIX_SOFT_DEMOTE_SCORE_PENALTY, 0.08)
        self.assertTrue(settings.HISTORICAL_FAMILY_GATE_ENABLED)
        self.assertEqual(settings.HISTORICAL_FAMILY_MIN_SAMPLES, 12)
        self.assertEqual(settings.HISTORICAL_FAMILY_PNL_CUTOFF, -12.0)
        self.assertEqual(settings.HISTORICAL_FAMILY_WIN_RATE_CUTOFF, 0.40)
        self.assertEqual(settings.SCORE_HALLUCINATED_EDGE_PENALTY_BASE, 0.08)
        self.assertEqual(settings.SCORE_EXTREME_MARKET_EDGE_PENALTY_BASE, 0.08)
        self.assertEqual(settings.SCORE_LATE_STAGE_OVERCONFIDENCE_PENALTY_BASE, 0.12)
        self.assertEqual(settings.MAX_SPEECH_CONFIDENCE, 0.65)
        self.assertEqual(settings.CONFIDENCE_SHRINKAGE_FLOOR, 0.55)
        self.assertEqual(settings.CONFIDENCE_SHRINKAGE_FACTOR, 0.32)
        self.assertEqual(settings.CONFIDENCE_SHRINKAGE_FACTOR_HIGH, 0.28)
        self.assertTrue(settings.HISTORICAL_CONFIDENCE_SHRINK_ENABLED)
        self.assertEqual(settings.HISTORICAL_CONFIDENCE_SHRINK_MIN_SAMPLES, 15)
        self.assertEqual(settings.HISTORICAL_CONFIDENCE_SHRINK_LOOKBACK_DAYS, 30)
        self.assertEqual(settings.SCORE_EXTREME_CONFIDENCE_THRESHOLD, 0.90)
        self.assertEqual(settings.SCORE_EXTREME_CONFIDENCE_PENALTY_BASE, 0.08)
        self.assertEqual(settings.HISTORICAL_SHORT_PREFIX_LEN, 5)
        self.assertEqual(settings.HISTORICAL_SHORT_PREFIX_MIN_SAMPLES, 3)
        self.assertEqual(settings.HISTORICAL_SHORT_PREFIX_PNL_CUTOFF, -5.0)
        self.assertEqual(settings.HISTORICAL_SHORT_PREFIX_SCORE_PENALTY, 0.10)
        self.assertEqual(settings.EXTENDED_RESEARCH_AFTER_STREAK, 2)
        self.assertEqual(settings.EXTENDED_RESEARCH_COOLDOWN_CYCLES, 5)
        self.assertEqual(settings.CALIBRATION_DIRECT_SHRINKAGE_FACTOR_BOOST, 2.0)
        self.assertEqual(settings.PRE_ANALYSIS_ZERO_TRADE_RATE_PENALTY, 0.04)
        self.assertTrue(settings.RESEARCH_QUEUE_ENABLED)
        self.assertTrue(settings.RESEARCH_QUEUE_PRIORITY_ENABLED)
        self.assertEqual(settings.MARKET_TICKER_BLOCKLIST_PREFIXES, ())
        self.assertFalse(settings.SKIP_WEATHER_BIN_MARKETS)
        self.assertFalse(settings.CRYPTO_BIN_MARKET_BLOCKLIST_ENABLED)
        self.assertIn("billboard.com", settings.MUSIC_ALLOWED_DOMAINS)
        self.assertIn("SpotifyCharts", settings.MUSIC_ALLOWED_X_HANDLES)

    def test_tennis_sources_present_in_sports_profile_defaults(self) -> None:
        self.assertIn("atptour.com", config.Settings.SPORTS_ALLOWED_DOMAINS)
        self.assertIn("espncricinfo.com", config.Settings.SPORTS_ALLOWED_DOMAINS)
        self.assertIn("cricbuzz.com", config.Settings.SPORTS_ALLOWED_DOMAINS)
        self.assertIn("wtatennis.com", config.Settings.SPORTS_ALLOWED_DOMAINS)
        self.assertIn("tennisexplorer.com", config.Settings.SPORTS_ALLOWED_DOMAINS)
        self.assertIn("flashscore.com", config.Settings.SPORTS_ALLOWED_DOMAINS)
        self.assertIn("atptour", config.Settings.SPORTS_ALLOWED_X_HANDLES)
        self.assertIn("ESPNcricinfo", config.Settings.SPORTS_ALLOWED_X_HANDLES)
        self.assertIn("WTA", config.Settings.SPORTS_ALLOWED_X_HANDLES)

    def test_profit_leak_fix_defaults(self) -> None:
        """Verify the profit-leak-fix defaults match .env.example values."""
        defaults = config.Settings
        self.assertEqual(defaults.DEFINITIVE_OUTCOME_EDGE_REASONABLE_MAX, 0.50)
        self.assertEqual(defaults.DIRECT_SOURCE_MIN_EVIDENCE_QUALITY_DEFAULT, 0.75)
        self.assertEqual(defaults.MAX_REANALYSES_PER_MARKET_PER_DAY, 2)
        self.assertEqual(defaults.MAX_LIFETIME_ANALYSES_PER_MARKET, 8)
        self.assertEqual(defaults.GROK_STREAM_TIMEOUT_SECONDS, 75)
        self.assertEqual(defaults.HISTORICAL_TICKER_PREFIX_PNL_CUTOFF, -2.0)
        self.assertEqual(defaults.HISTORICAL_TICKER_PREFIX_HARD_BLOCK_MIN_SAMPLES, 20)
        self.assertTrue(defaults.XAI_QUOTA_BREAKER_ENABLED)
        self.assertEqual(defaults.XAI_QUOTA_PAUSE_MINUTES, 30)

    def test_cycle1_review_defaults(self) -> None:
        """Defaults introduced or tuned by the cycle 1 log review."""
        defaults = config.Settings
        self.assertEqual(defaults.MAX_REASONABLE_EDGE, 0.40)
        self.assertEqual(defaults.DEFINITIVE_OUTCOME_EDGE_REASONABLE_MAX, 0.50)
        self.assertEqual(defaults.HIGH_QUALITY_SETTLED_EVIDENCE_MIN_EQ, 0.95)
        self.assertEqual(defaults.GROK_STREAM_TIMEOUT_SECONDS_WEATHER, 120)

    def test_high_quality_settled_evidence_min_eq_env_override(self) -> None:
        env = {
            **self._required_env(),
            "HIGH_QUALITY_SETTLED_EVIDENCE_MIN_EQ": "0.90",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()
        self.assertEqual(settings.HIGH_QUALITY_SETTLED_EVIDENCE_MIN_EQ, 0.90)

    def test_grok_stream_timeout_seconds_weather_env_override(self) -> None:
        env = {
            **self._required_env(),
            "GROK_STREAM_TIMEOUT_SECONDS_WEATHER": "150",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()
        self.assertEqual(settings.GROK_STREAM_TIMEOUT_SECONDS_WEATHER, 150)

    def test_cycle2_fetch_strategy_defaults(self) -> None:
        """Defaults introduced by the cycle 2 fetch-strategy review."""
        defaults = config.Settings
        self.assertEqual(defaults.KALSHI_MVE_FILTER, "exclude")
        self.assertEqual(defaults.KALSHI_ELIGIBLE_FLOOR, 100)
        self.assertFalse(defaults.KALSHI_FETCH_TOPUP_ENABLED)

    def test_kalshi_mve_filter_env_override(self) -> None:
        env = {
            **self._required_env(),
            "KALSHI_MVE_FILTER": "only",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()
        self.assertEqual(settings.KALSHI_MVE_FILTER, "only")

    def test_kalshi_eligible_floor_env_override(self) -> None:
        env = {
            **self._required_env(),
            "KALSHI_ELIGIBLE_FLOOR": "250",
            "KALSHI_FETCH_TOPUP_ENABLED": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()
        self.assertEqual(settings.KALSHI_ELIGIBLE_FLOOR, 250)
        self.assertTrue(settings.KALSHI_FETCH_TOPUP_ENABLED)

    def test_quota_breaker_env_override(self) -> None:
        env = {
            **self._required_env(),
            "XAI_QUOTA_BREAKER_ENABLED": "false",
            "XAI_QUOTA_PAUSE_MINUTES": "60",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()
        self.assertFalse(settings.XAI_QUOTA_BREAKER_ENABLED)
        self.assertEqual(settings.XAI_QUOTA_PAUSE_MINUTES, 60)

    def test_pre_analysis_participation_alias_overrides_legacy(self) -> None:
        """The new PRE_ANALYSIS_PARTICIPATION_* names take precedence over the
        legacy PRE_ANALYSIS_HARD_REJECTION_* names so the rename is real but
        non-breaking."""
        env = {
            **self._required_env(),
            "PRE_ANALYSIS_HARD_REJECTION_ENABLED": "false",
            "PRE_ANALYSIS_PARTICIPATION_GATING_ENABLED": "true",
            "PRE_ANALYSIS_HARD_REJECTION_MIN_STREAK": "3",
            "PRE_ANALYSIS_PARTICIPATION_MIN_STREAK": "7",
            "PRE_ANALYSIS_HARD_REJECTION_MIN_ANALYSES": "5",
            "PRE_ANALYSIS_PARTICIPATION_MIN_ANALYSES": "9",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()
        self.assertTrue(settings.PRE_ANALYSIS_HARD_REJECTION_ENABLED)
        self.assertEqual(settings.PRE_ANALYSIS_HARD_REJECTION_MIN_STREAK, 7)
        self.assertEqual(settings.PRE_ANALYSIS_HARD_REJECTION_MIN_ANALYSES, 9)

    def test_pre_analysis_legacy_name_still_works_when_alias_unset(self) -> None:
        """When the new alias is unset, the legacy PRE_ANALYSIS_HARD_REJECTION_*
        env vars are still honored. Operators don't need to migrate today."""
        env = {
            **self._required_env(),
            "PRE_ANALYSIS_HARD_REJECTION_ENABLED": "false",
            "PRE_ANALYSIS_HARD_REJECTION_MIN_STREAK": "11",
            "PRE_ANALYSIS_HARD_REJECTION_MIN_ANALYSES": "13",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()
        self.assertFalse(settings.PRE_ANALYSIS_HARD_REJECTION_ENABLED)
        self.assertEqual(settings.PRE_ANALYSIS_HARD_REJECTION_MIN_STREAK, 11)
        self.assertEqual(settings.PRE_ANALYSIS_HARD_REJECTION_MIN_ANALYSES, 13)

    def test_research_queue_drain_settings_load_with_defaults(self) -> None:
        env = self._required_env()
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()
        self.assertTrue(settings.RESEARCH_QUEUE_DRAIN_ENABLED)
        self.assertEqual(settings.RESEARCH_QUEUE_DRAIN_PER_CYCLE, 8)
        self.assertEqual(settings.RESEARCH_QUEUE_DRAIN_MIN_AGE_HOURS, 1.0)
        self.assertEqual(settings.RESEARCH_QUEUE_DRAIN_MAX_AGE_HOURS, 12.0)
        self.assertEqual(settings.RESEARCH_QUEUE_DRAIN_MIN_PRIORITY, 0.40)
        self.assertTrue(settings.RESEARCH_QUEUE_DRAIN_FORCE_EXTENDED_RESEARCH)

    def test_research_queue_drain_settings_env_override(self) -> None:
        env = {
            **self._required_env(),
            "RESEARCH_QUEUE_DRAIN_ENABLED": "false",
            "RESEARCH_QUEUE_DRAIN_PER_CYCLE": "3",
            "RESEARCH_QUEUE_DRAIN_MIN_AGE_HOURS": "0.25",
            "RESEARCH_QUEUE_DRAIN_MAX_AGE_HOURS": "24.0",
            "RESEARCH_QUEUE_DRAIN_MIN_PRIORITY": "0.65",
            "RESEARCH_QUEUE_DRAIN_FORCE_EXTENDED_RESEARCH": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()
        self.assertFalse(settings.RESEARCH_QUEUE_DRAIN_ENABLED)
        self.assertEqual(settings.RESEARCH_QUEUE_DRAIN_PER_CYCLE, 3)
        self.assertEqual(settings.RESEARCH_QUEUE_DRAIN_MIN_AGE_HOURS, 0.25)
        self.assertEqual(settings.RESEARCH_QUEUE_DRAIN_MAX_AGE_HOURS, 24.0)
        self.assertEqual(settings.RESEARCH_QUEUE_DRAIN_MIN_PRIORITY, 0.65)
        self.assertFalse(settings.RESEARCH_QUEUE_DRAIN_FORCE_EXTENDED_RESEARCH)

    def test_research_queue_drain_higher_per_cycle_value_loads(self) -> None:
        """Cycle 4 recovery: a growing research queue (127+ entries) needs
        more drain throughput per cycle. Verify the per-cycle setting can
        be raised without other drain knobs silently changing."""
        env = {
            **self._required_env(),
            "RESEARCH_QUEUE_DRAIN_PER_CYCLE": "2",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()
        self.assertTrue(settings.RESEARCH_QUEUE_DRAIN_ENABLED)
        self.assertEqual(settings.RESEARCH_QUEUE_DRAIN_PER_CYCLE, 2)
        self.assertEqual(settings.RESEARCH_QUEUE_DRAIN_MIN_AGE_HOURS, 1.0)
        self.assertEqual(settings.RESEARCH_QUEUE_DRAIN_MAX_AGE_HOURS, 12.0)
        self.assertEqual(settings.RESEARCH_QUEUE_DRAIN_MIN_PRIORITY, 0.40)

    def test_max_sports_candidates_per_cycle_default_unset(self) -> None:
        """Cycle 4 recovery: the new sports cap defaults to 0 (no cap) so
        existing deployments retain prior behavior."""
        env = self._required_env()
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()
        self.assertEqual(settings.MAX_SPORTS_CANDIDATES_PER_CYCLE, 0)

    def test_max_sports_candidates_per_cycle_env_override(self) -> None:
        """Operators can opt into family diversification by setting the cap
        to a positive integer."""
        env = {
            **self._required_env(),
            "MAX_SPORTS_CANDIDATES_PER_CYCLE": "3",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()
        self.assertEqual(settings.MAX_SPORTS_CANDIDATES_PER_CYCLE, 3)

    def test_daily_drawdown_preflight_setting_loads(self) -> None:
        env = {**self._required_env(), "DAILY_DRAWDOWN_PREFLIGHT_ENABLED": "false"}
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()
        self.assertFalse(settings.DAILY_DRAWDOWN_PREFLIGHT_ENABLED)
        with patch.dict(os.environ, self._required_env(), clear=True):
            defaults = config.load_settings()
        self.assertTrue(defaults.DAILY_DRAWDOWN_PREFLIGHT_ENABLED)

    def test_cycle_yield_alert_escalation_setting_loads(self) -> None:
        env = {**self._required_env(), "CYCLE_YIELD_ALERT_ESCALATE_AFTER": "5"}
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()
        self.assertEqual(settings.CYCLE_YIELD_ALERT_ESCALATE_AFTER, 5)
        with patch.dict(os.environ, self._required_env(), clear=True):
            defaults = config.load_settings()
        self.assertEqual(defaults.CYCLE_YIELD_ALERT_ESCALATE_AFTER, 2)


if __name__ == "__main__":
    unittest.main()
