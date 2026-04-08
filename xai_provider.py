from __future__ import annotations

import time
from typing import Any

from xai_sdk import Client
from xai_sdk.chat import system, user
from xai_sdk.tools import web_search, x_search

from config import SearchConfig
from logging_config import get_logger

logger = get_logger(__name__)

_DEFAULT_CREATE_CHAT_MAX_ATTEMPTS = 3
_DEFAULT_CREATE_CHAT_BACKOFF_SECONDS = 0.5
_MAX_CREATE_CHAT_BACKOFF_SECONDS = 4.0


class XAIProvider:
    """Isolate xAI SDK access behind one wrapper."""

    def __init__(
        self,
        api_key: str,
        timeout_seconds: int,
        create_chat_max_attempts: int = _DEFAULT_CREATE_CHAT_MAX_ATTEMPTS,
        create_chat_backoff_seconds: float = _DEFAULT_CREATE_CHAT_BACKOFF_SECONDS,
    ) -> None:
        self.client = Client(api_key=api_key, timeout=timeout_seconds)
        self.create_chat_max_attempts = max(1, int(create_chat_max_attempts))
        self.create_chat_backoff_seconds = max(0.0, float(create_chat_backoff_seconds))

    def create_chat(
        self,
        *,
        model: str,
        response_format: Any,
        config: SearchConfig,
        enable_multimedia: bool,
    ):
        for attempt in range(1, self.create_chat_max_attempts + 1):
            try:
                return self.client.chat.create(
                    model=model,
                    response_format=response_format,
                    tools=[
                        web_search(allowed_domains=config.allowed_domains),
                        x_search(
                            from_date=config.from_date,
                            to_date=config.to_date,
                            allowed_x_handles=config.allowed_x_handles,
                            enable_image_understanding=enable_multimedia,
                            enable_video_understanding=enable_multimedia,
                        ),
                    ],
                )
            except Exception:
                if attempt >= self.create_chat_max_attempts:
                    raise
                backoff_seconds = min(
                    self.create_chat_backoff_seconds * (2 ** (attempt - 1)),
                    _MAX_CREATE_CHAT_BACKOFF_SECONDS,
                )
                logger.warning(
                    "xAI create_chat failed, retrying: attempt=%d/%d backoff=%.2fs",
                    attempt,
                    self.create_chat_max_attempts,
                    backoff_seconds,
                )
                if backoff_seconds > 0:
                    time.sleep(backoff_seconds)

    @staticmethod
    def system_message(content: str):
        return system(content)

    @staticmethod
    def user_message(content: str):
        return user(content)
