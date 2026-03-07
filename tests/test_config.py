import os
import unittest
from unittest.mock import patch

import config


class TestConfig(unittest.TestCase):
    def test_load_settings_success(self) -> None:
        env = {
            "XAI_API_KEY": "xai-key",
            "ALCHEMY_RPC_URL": "https://rpc.example",
            "WALLET_PRIVATE_KEY": "0xabc",
            "MARKET_CATEGORIES_ALLOWLIST": "sports, politics",
            "MARKET_CATEGORIES_BLOCKLIST": "crypto",
            "MIN_BET_USDC": "10",
            "MAX_BET_USDC": "75",
            "DRY_RUN": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()

        self.assertEqual(settings.XAI_API_KEY, "xai-key")
        self.assertEqual(settings.ALCHEMY_RPC_URL, "https://rpc.example")
        self.assertEqual(settings.WALLET_PRIVATE_KEY, "0xabc")
        self.assertEqual(settings.MARKET_CATEGORIES_ALLOWLIST, ("sports", "politics"))
        self.assertEqual(settings.MARKET_CATEGORIES_BLOCKLIST, ("crypto",))
        self.assertEqual(settings.MIN_BET_USDC, 10.0)
        self.assertEqual(settings.MAX_BET_USDC, 75.0)
        self.assertFalse(settings.DRY_RUN)

    def test_close_days_filter_settings(self) -> None:
        env = {
            "XAI_API_KEY": "xai-key",
            "WALLET_PRIVATE_KEY": "0xabc",
            "MARKET_MIN_CLOSE_DAYS": "1",
            "MARKET_MAX_CLOSE_DAYS": "7",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()

        self.assertEqual(settings.MARKET_MIN_CLOSE_DAYS, 1)
        self.assertEqual(settings.MARKET_MAX_CLOSE_DAYS, 7)

    def test_close_days_filter_defaults_to_none(self) -> None:
        env = {
            "XAI_API_KEY": "xai-key",
            "WALLET_PRIVATE_KEY": "0xabc",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()

        self.assertIsNone(settings.MARKET_MIN_CLOSE_DAYS)
        self.assertIsNone(settings.MARKET_MAX_CLOSE_DAYS)

    def test_safe_mode_allows_missing_alchemy_rpc_url(self) -> None:
        env = {
            "XAI_API_KEY": "xai-key",
            "WALLET_PRIVATE_KEY": "0xabc",
            "DRY_RUN": "true",
            "EXECUTE_ONCHAIN": "false",
            "AUTO_APPROVE_USDC": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()

        self.assertEqual(settings.ALCHEMY_RPC_URL, "")

    def test_execute_onchain_requires_alchemy_rpc_url(self) -> None:
        env = {
            "XAI_API_KEY": "xai-key",
            "WALLET_PRIVATE_KEY": "0xabc",
            "EXECUTE_ONCHAIN": "true",
            "AUTO_APPROVE_USDC": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "ALCHEMY_RPC_URL"):
                config.load_settings()

    def test_auto_approve_requires_predictbase_contract_address(self) -> None:
        env = {
            "XAI_API_KEY": "xai-key",
            "WALLET_PRIVATE_KEY": "0xabc",
            "AUTO_APPROVE_USDC": "true",
            "EXECUTE_ONCHAIN": "false",
            "ALCHEMY_RPC_URL": "https://rpc.example",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "PREDICTBASE_CONTRACT_ADDRESS"):
                config.load_settings()

    def test_missing_required_env_raises(self) -> None:
        env = {
            "XAI_API_KEY": "xai-key",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                config.load_settings()

    def test_search_settings_overrides(self) -> None:
        env = {
            "XAI_API_KEY": "xai-key",
            "WALLET_PRIVATE_KEY": "0xabc",
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

    def test_build_search_config(self) -> None:
        env = {
            "XAI_API_KEY": "xai-key",
            "WALLET_PRIVATE_KEY": "0xabc",
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
            "XAI_API_KEY": "xai-key",
            "WALLET_PRIVATE_KEY": "0xabc",
            "FLIP_GUARD_ENABLED": "false",
            "FLIP_GUARD_MIN_ABS_CONFIDENCE": "0.70",
            "FLIP_GUARD_MIN_CONF_GAIN": "0.10",
            "FLIP_GUARD_MIN_EDGE_GAIN": "0.05",
            "FLIP_GUARD_MIN_EVIDENCE_QUALITY": "0.75",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()

        self.assertFalse(settings.FLIP_GUARD_ENABLED)
        self.assertEqual(settings.FLIP_GUARD_MIN_ABS_CONFIDENCE, 0.70)
        self.assertEqual(settings.FLIP_GUARD_MIN_CONF_GAIN, 0.10)
        self.assertEqual(settings.FLIP_GUARD_MIN_EDGE_GAIN, 0.05)
        self.assertEqual(settings.FLIP_GUARD_MIN_EVIDENCE_QUALITY, 0.75)

    def test_parallel_and_execution_guard_settings_overrides(self) -> None:
        env = {
            "XAI_API_KEY": "xai-key",
            "WALLET_PRIVATE_KEY": "0xabc",
            "PARALLEL_ANALYSIS_ENABLED": "true",
            "ANALYSIS_MAX_WORKERS": "4",
            "PRE_ORDER_MARKET_REFRESH": "true",
            "ORDERBOOK_PRECHECK_ENABLED": "true",
            "ORDERBOOK_PRECHECK_MIN_CONFIDENCE": "0.8",
            "CALIBRATION_MODE_ENABLED": "true",
            "CALIBRATION_MIN_SAMPLES": "25",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()

        self.assertTrue(settings.PARALLEL_ANALYSIS_ENABLED)
        self.assertEqual(settings.ANALYSIS_MAX_WORKERS, 4)
        self.assertTrue(settings.PRE_ORDER_MARKET_REFRESH)
        self.assertTrue(settings.ORDERBOOK_PRECHECK_ENABLED)
        self.assertEqual(settings.ORDERBOOK_PRECHECK_MIN_CONFIDENCE, 0.8)
        self.assertTrue(settings.CALIBRATION_MODE_ENABLED)
        self.assertEqual(settings.CALIBRATION_MIN_SAMPLES, 25)

    def test_bayesian_lmsr_kelly_settings_overrides(self) -> None:
        env = {
            "XAI_API_KEY": "xai-key",
            "WALLET_PRIVATE_KEY": "0xabc",
            "BAYESIAN_ENABLED": "true",
            "BAYESIAN_PRIOR_DEFAULT": "0.58",
            "BAYESIAN_MIN_UPDATES_FOR_TRADE": "3",
            "LMSR_ENABLED": "true",
            "LMSR_LIQUIDITY_PARAM_B": "120000",
            "LMSR_MIN_INEFFICIENCY": "0.04",
            "KELLY_SIZING_ENABLED": "true",
            "KELLY_FRACTION_DEFAULT": "0.2",
            "KELLY_FRACTION_SHORT_HORIZON_HOURS": "2",
            "KELLY_FRACTION_SHORT_HORIZON": "0.1",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.load_settings()

        self.assertTrue(settings.BAYESIAN_ENABLED)
        self.assertEqual(settings.BAYESIAN_PRIOR_DEFAULT, 0.58)
        self.assertEqual(settings.BAYESIAN_MIN_UPDATES_FOR_TRADE, 3)
        self.assertTrue(settings.LMSR_ENABLED)
        self.assertEqual(settings.LMSR_LIQUIDITY_PARAM_B, 120000.0)
        self.assertEqual(settings.LMSR_MIN_INEFFICIENCY, 0.04)
        self.assertTrue(settings.KELLY_SIZING_ENABLED)
        self.assertEqual(settings.KELLY_FRACTION_DEFAULT, 0.2)
        self.assertEqual(settings.KELLY_FRACTION_SHORT_HORIZON_HOURS, 2)
        self.assertEqual(settings.KELLY_FRACTION_SHORT_HORIZON, 0.1)


if __name__ == "__main__":
    unittest.main()
