from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

from config import SearchConfig
from xai_provider import XAIProvider


class _CaptureChat:
    def __init__(self) -> None:
        self.last_kwargs: dict = {}

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return {"ok": True}


class _FakeClient:
    def __init__(self, chat: _CaptureChat) -> None:
        self.chat = chat


def test_create_chat_appends_code_execution_when_enabled() -> None:
    capture_chat = _CaptureChat()
    with patch("xai_provider.Client", return_value=_FakeClient(capture_chat)):
        provider = XAIProvider(
            api_key="xai-key",
            timeout_seconds=5,
            create_chat_max_attempts=1,
            create_chat_backoff_seconds=0.0,
        )
    with patch("xai_provider.web_search", return_value={"tool": "web"}), patch(
        "xai_provider.x_search", return_value={"tool": "x"}
    ), patch("xai_provider.code_execution", return_value={"tool": "code"}):
        provider.create_chat(
            model="grok-test",
            response_format=dict,
            config=SearchConfig(
                from_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
                to_date=datetime(2026, 1, 2, tzinfo=timezone.utc),
            ),
            enable_multimedia=False,
            enable_code_execution=True,
        )
    tools = capture_chat.last_kwargs.get("tools") or []
    assert len(tools) == 3
    assert tools[-1] == {"tool": "code"}
