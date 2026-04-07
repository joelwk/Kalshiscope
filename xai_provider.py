from __future__ import annotations

from typing import Any

from xai_sdk import Client
from xai_sdk.chat import system, user
from xai_sdk.tools import web_search, x_search

from config import SearchConfig


class XAIProvider:
    """Isolate xAI SDK access behind one wrapper."""

    def __init__(self, api_key: str, timeout_seconds: int) -> None:
        self.client = Client(api_key=api_key, timeout=timeout_seconds)

    def create_chat(
        self,
        *,
        model: str,
        response_format: Any,
        config: SearchConfig,
        enable_multimedia: bool,
    ):
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

    @staticmethod
    def system_message(content: str):
        return system(content)

    @staticmethod
    def user_message(content: str):
        return user(content)
