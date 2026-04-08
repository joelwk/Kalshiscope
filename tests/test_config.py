import os
import unittest
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
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()

        self.assertEqual(settings.SEARCH_LOOKBACK_HOURS, 12)
        self.assertEqual(settings.SEARCH_ALLOWED_DOMAINS, ("example.com", "news.example"))
        self.assertEqual(settings.SEARCH_ALLOWED_X_HANDLES, ("Foo", "Bar"))
        self.assertEqual(settings.MULTIMEDIA_CONFIDENCE_THRESHOLD, (0.6, 0.7))

    def test_weather_profile_settings_overrides(self) -> None:
        env = {
            **self._required_env(),
            "WEATHER_ALLOWED_DOMAINS": "weather.gov,weather.com",
            "WEATHER_ALLOWED_X_HANDLES": "NWS,weatherchannel",
            "SKIP_WEATHER_BIN_MARKETS": "false",
            "MAX_WEATHER_CONFIDENCE": "0.78",
            "WEATHER_MIN_EDGE": "0.09",
            "WEATHER_SCORE_PENALTY": "0.04",
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
        self.assertEqual(settings.KELLY_FRACTION_WEATHER, 0.45)

    def test_weather_profile_defaults_include_official_sources(self) -> None:
        self.assertIn("weather.gov", config.Settings.WEATHER_ALLOWED_DOMAINS)
        self.assertIn("forecast.weather.gov", config.Settings.WEATHER_ALLOWED_DOMAINS)
        self.assertIn("noaa.gov", config.Settings.WEATHER_ALLOWED_DOMAINS)
        self.assertIn("NWS", config.Settings.WEATHER_ALLOWED_X_HANDLES)
        self.assertIn("NWSSPC", config.Settings.WEATHER_ALLOWED_X_HANDLES)
        self.assertIn("NHC_Atlantic", config.Settings.WEATHER_ALLOWED_X_HANDLES)

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
            "XAI_CIRCUIT_BREAKER_MAX_FAILURES": "4",
            "KALSHI_MAX_FETCH_PAGES": "12",
            "XAI_CLIENT_TIMEOUT_SECONDS": "75",
            "GROK_STREAM_TIMEOUT_SECONDS": "80",
            "GROK_ANALYSIS_MAX_BUDGET_SECONDS": "55",
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
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()

        self.assertTrue(settings.PARALLEL_ANALYSIS_ENABLED)
        self.assertEqual(settings.ANALYSIS_MAX_WORKERS, 4)
        self.assertEqual(settings.MAX_MARKETS_PER_CYCLE, 25)
        self.assertEqual(settings.MAX_TRADES_PER_CYCLE, 6)
        self.assertEqual(settings.XAI_CIRCUIT_BREAKER_MAX_FAILURES, 4)
        self.assertEqual(settings.KALSHI_MAX_FETCH_PAGES, 12)
        self.assertEqual(settings.XAI_CLIENT_TIMEOUT_SECONDS, 75)
        self.assertEqual(settings.GROK_STREAM_TIMEOUT_SECONDS, 80)
        self.assertEqual(settings.GROK_ANALYSIS_MAX_BUDGET_SECONDS, 55)
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
            "KELLY_FRACTION_DEFAULT": "0.2",
            "KELLY_FRACTION_SHORT_HORIZON_HOURS": "2",
            "KELLY_FRACTION_SHORT_HORIZON": "0.1",
            "KELLY_MIN_BET_POLICY": "floor",
            "MAX_POSITION_PCT_OF_BANKROLL": "0.12",
            "COINFLIP_PRICE_LOWER": "0.46",
            "COINFLIP_PRICE_UPPER": "0.54",
            "FALLBACK_EDGE_MIN_EDGE": "0.09",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()

        self.assertTrue(settings.BAYESIAN_ENABLED)
        self.assertFalse(settings.BAYESIAN_SKIP_STALE_UPDATES)
        self.assertEqual(settings.BAYESIAN_PRIOR_DEFAULT, 0.58)
        self.assertEqual(settings.BAYESIAN_MIN_UPDATES_FOR_TRADE, 3)
        self.assertEqual(settings.BAYESIAN_MAX_POSTERIOR, 0.97)
        self.assertTrue(settings.LMSR_ENABLED)
        self.assertEqual(settings.LMSR_LIQUIDITY_PARAM_B, 120000.0)
        self.assertEqual(settings.LMSR_MIN_INEFFICIENCY, 0.04)
        self.assertTrue(settings.KELLY_SIZING_ENABLED)
        self.assertEqual(settings.KELLY_FRACTION_DEFAULT, 0.2)
        self.assertEqual(settings.KELLY_FRACTION_SHORT_HORIZON_HOURS, 2)
        self.assertEqual(settings.KELLY_FRACTION_SHORT_HORIZON, 0.1)
        self.assertEqual(settings.KELLY_MIN_BET_POLICY, "floor")
        self.assertEqual(settings.MAX_POSITION_PCT_OF_BANKROLL, 0.12)
        self.assertEqual(settings.COINFLIP_PRICE_LOWER, 0.46)
        self.assertEqual(settings.COINFLIP_PRICE_UPPER, 0.54)
        self.assertEqual(settings.FALLBACK_EDGE_MIN_EDGE, 0.09)

    def test_profit_guardrail_defaults(self) -> None:
        settings = config.Settings()
        self.assertEqual(settings.MIN_EVIDENCE_QUALITY_FOR_TRADE, 0.50)
        self.assertEqual(settings.SCORE_GATE_MODE, "active")
        self.assertEqual(settings.SCORE_GATE_THRESHOLD, 0.12)
        self.assertEqual(settings.MAX_MARKETS_PER_CYCLE, 20)
        self.assertEqual(settings.MAX_TRADES_PER_CYCLE, 5)
        self.assertEqual(settings.ORDER_PRICE_IMPROVEMENT_CENTS, 1)
        self.assertEqual(settings.ORDER_SUBMISSION_MIN_PRICE, 0.03)
        self.assertEqual(settings.ORDER_SUBMISSION_MAX_PRICE, 0.97)
        self.assertEqual(settings.MIN_TRADEABLE_IMPLIED_PRICE, 0.05)
        self.assertEqual(settings.MAX_TRADEABLE_IMPLIED_PRICE, 0.95)
        self.assertEqual(settings.KALSHI_MAX_FETCH_PAGES, 0)
        self.assertEqual(settings.XAI_CIRCUIT_BREAKER_MAX_FAILURES, 3)
        self.assertEqual(settings.XAI_CLIENT_TIMEOUT_SECONDS, 120)
        self.assertEqual(settings.GROK_STREAM_TIMEOUT_SECONDS, 120)
        self.assertEqual(settings.GROK_ANALYSIS_MAX_BUDGET_SECONDS, 180)
        self.assertTrue(settings.EVIDENCE_QUALITY_HIGH_CONFIDENCE_OVERRIDE)

    def test_tennis_sources_present_in_sports_profile_defaults(self) -> None:
        self.assertIn("atptour.com", config.Settings.SPORTS_ALLOWED_DOMAINS)
        self.assertIn("wtatennis.com", config.Settings.SPORTS_ALLOWED_DOMAINS)
        self.assertIn("tennisexplorer.com", config.Settings.SPORTS_ALLOWED_DOMAINS)
        self.assertIn("flashscore.com", config.Settings.SPORTS_ALLOWED_DOMAINS)
        self.assertIn("atptour", config.Settings.SPORTS_ALLOWED_X_HANDLES)
        self.assertIn("WTA", config.Settings.SPORTS_ALLOWED_X_HANDLES)


if __name__ == "__main__":
    unittest.main()
