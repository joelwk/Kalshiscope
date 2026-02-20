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


if __name__ == "__main__":
    unittest.main()
