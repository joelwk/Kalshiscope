from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from config import SearchConfig
from xai_provider import XAIProvider


class _FlakyChat:
    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient xai failure")
        return {"ok": True, "kwargs": kwargs}


class _FakeClient:
    def __init__(self, chat: _FlakyChat) -> None:
        self.chat = chat


def _search_config() -> SearchConfig:
    return SearchConfig(
        from_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        to_date=datetime(2026, 1, 2, tzinfo=timezone.utc),
        allowed_domains=["example.com"],
        allowed_x_handles=["handle_a"],
    )


def test_create_chat_retries_and_recovers() -> None:
    flaky_chat = _FlakyChat(fail_times=2)
    with patch("xai_provider.Client", return_value=_FakeClient(flaky_chat)):
        provider = XAIProvider(
            api_key="xai-key",
            timeout_seconds=5,
            create_chat_max_attempts=3,
            create_chat_backoff_seconds=0.0,
        )
    with patch("xai_provider.web_search", return_value={"tool": "web"}), patch(
        "xai_provider.x_search", return_value={"tool": "x"}
    ):
        response = provider.create_chat(
            model="grok-test",
            response_format=dict,
            config=_search_config(),
            enable_multimedia=False,
        )
    assert response["ok"] is True
    assert flaky_chat.calls == 3


def test_create_chat_raises_after_retry_exhaustion() -> None:
    flaky_chat = _FlakyChat(fail_times=5)
    with patch("xai_provider.Client", return_value=_FakeClient(flaky_chat)):
        provider = XAIProvider(
            api_key="xai-key",
            timeout_seconds=5,
            create_chat_max_attempts=3,
            create_chat_backoff_seconds=0.0,
        )
    with patch("xai_provider.web_search", return_value={"tool": "web"}), patch(
        "xai_provider.x_search", return_value={"tool": "x"}
    ):
        with pytest.raises(RuntimeError):
            provider.create_chat(
                model="grok-test",
                response_format=dict,
                config=_search_config(),
                enable_multimedia=False,
            )
    assert flaky_chat.calls == 3


def test_create_chat_passes_image_understanding_to_web_search() -> None:
    flaky_chat = _FlakyChat(fail_times=0)
    with patch("xai_provider.Client", return_value=_FakeClient(flaky_chat)):
        provider = XAIProvider(
            api_key="xai-key",
            timeout_seconds=5,
            create_chat_max_attempts=1,
            create_chat_backoff_seconds=0.0,
        )

    captured: dict[str, dict] = {}

    def fake_web_search(*args, **kwargs):
        captured["web"] = kwargs
        return {"tool": "web"}

    def fake_x_search(*args, **kwargs):
        captured["x"] = kwargs
        return {"tool": "x"}

    with patch("xai_provider.web_search", side_effect=fake_web_search), patch(
        "xai_provider.x_search", side_effect=fake_x_search
    ):
        provider.create_chat(
            model="grok-test",
            response_format=dict,
            config=_search_config(),
            enable_multimedia=True,
        )

    assert captured["web"]["enable_image_understanding"] is True
    assert captured["x"]["enable_image_understanding"] is True
