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


def test_create_chat_uses_request_specific_timeout_client() -> None:
    default_chat = _FlakyChat(fail_times=0)
    override_chat = _FlakyChat(fail_times=0)
    with patch(
        "xai_provider.Client",
        side_effect=[_FakeClient(default_chat), _FakeClient(override_chat)],
    ) as client_ctor:
        provider = XAIProvider(
            api_key="xai-key",
            timeout_seconds=5,
            create_chat_max_attempts=1,
            create_chat_backoff_seconds=0.0,
        )
        provider._is_real_xai_client = lambda: True
        with patch("xai_provider.web_search", return_value={"tool": "web"}), patch(
            "xai_provider.x_search", return_value={"tool": "x"}
        ):
            response = provider.create_chat(
                model="grok-test",
                response_format=dict,
                config=_search_config(),
                enable_multimedia=False,
                timeout_seconds=7,
            )

    assert response["ok"] is True
    assert default_chat.calls == 0
    assert override_chat.calls == 1
    assert client_ctor.call_args_list[0].kwargs["timeout"] == 5.0
    assert client_ctor.call_args_list[1].kwargs["timeout"] == 7.0


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


def test_create_chat_passes_temperature_to_sdk() -> None:
    flaky_chat = _FlakyChat(fail_times=0)
    with patch("xai_provider.Client", return_value=_FakeClient(flaky_chat)):
        provider = XAIProvider(
            api_key="xai-key",
            timeout_seconds=5,
            create_chat_max_attempts=1,
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
            temperature=0.7,
        )

    assert response["kwargs"]["temperature"] == 0.7


def test_create_chat_passes_include_and_reasoning_effort() -> None:
    flaky_chat = _FlakyChat(fail_times=0)
    with patch("xai_provider.Client", return_value=_FakeClient(flaky_chat)):
        provider = XAIProvider(
            api_key="xai-key",
            timeout_seconds=5,
            create_chat_max_attempts=1,
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
            reasoning_effort="high",
        )

    assert response["kwargs"]["include"] == ["inline_citations"]
    assert response["kwargs"]["reasoning_effort"] == "high"


def test_create_chat_retries_without_reasoning_effort_on_unimplemented() -> None:
    class _UnimplementedThenOk:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def create(self, **kwargs):
            self.calls.append(dict(kwargs))
            if "reasoning_effort" in kwargs:
                raise RuntimeError("StatusCode.UNIMPLEMENTED: reasoning_effort")
            return {"ok": True, "kwargs": kwargs}

    chat = _UnimplementedThenOk()
    with patch("xai_provider.Client", return_value=_FakeClient(chat)):
        provider = XAIProvider(
            api_key="xai-key",
            timeout_seconds=5,
            create_chat_max_attempts=1,
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
            reasoning_effort="high",
        )

    assert response["ok"] is True
    assert len(chat.calls) == 2
    assert chat.calls[0]["reasoning_effort"] == "high"
    assert "reasoning_effort" not in chat.calls[1]
