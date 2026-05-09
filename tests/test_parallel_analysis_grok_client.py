"""Tests for the per-worker GrokClient cache used by parallel analysis.

Cycle 1 review observed 8+ ``GrokClient initialized`` debug messages in a
single cycle because the parallel-analysis loop built a fresh client for
every candidate. The thread-local cache below ensures each worker thread
reuses the same client across all candidates it processes.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from config import Settings
import main


def _stub_settings() -> Settings:
    return Settings(
        XAI_API_KEY="test-key",
        KALSHI_API_KEY_ID="test-kalshi-id",
        KALSHI_PRIVATE_KEY_PATH="test-key.pem",
    )


class _DummyGrokClient:
    instances = 0
    lock = threading.Lock()

    def __init__(self) -> None:
        with _DummyGrokClient.lock:
            _DummyGrokClient.instances += 1
            self.id = _DummyGrokClient.instances


def _build_dummy(_settings, provider=None):  # noqa: ARG001 - signature parity
    return _DummyGrokClient()


def _capture_client_id(_settings) -> int:
    client = main._get_or_create_worker_grok_client(_settings)
    return client.id


def test_thread_local_cache_returns_same_client_for_repeat_calls() -> None:
    main.reset_worker_grok_client_cache()
    _DummyGrokClient.instances = 0
    settings = _stub_settings()

    with patch.object(main, "_build_grok_client_for_worker", _build_dummy):
        first = main._get_or_create_worker_grok_client(settings)
        second = main._get_or_create_worker_grok_client(settings)

    assert first is second
    assert _DummyGrokClient.instances == 1


def test_thread_local_cache_builds_one_client_per_worker_thread() -> None:
    """The whole point of the cycle 1 fix: 4 candidates submitted across
    2 worker threads should produce exactly 2 GrokClient instances, not 4."""
    main.reset_worker_grok_client_cache()
    _DummyGrokClient.instances = 0
    settings = _stub_settings()

    seen_ids: list[int] = []
    seen_threads: set[int] = set()

    def worker(_payload: int) -> int:
        seen_threads.add(threading.get_ident())
        client = main._get_or_create_worker_grok_client(settings)
        return client.id

    with patch.object(main, "_build_grok_client_for_worker", _build_dummy):
        with ThreadPoolExecutor(max_workers=2) as executor:
            for fut in [executor.submit(worker, i) for i in range(8)]:
                seen_ids.append(fut.result())

    assert len(seen_threads) <= 2, "expected at most 2 worker threads"
    distinct_clients = set(seen_ids)
    assert len(distinct_clients) == len(seen_threads), (
        f"each worker thread should produce a single client; "
        f"saw {distinct_clients} from threads {seen_threads}"
    )
    assert _DummyGrokClient.instances == len(seen_threads)


def test_reset_worker_grok_client_cache_forces_rebuild() -> None:
    main.reset_worker_grok_client_cache()
    _DummyGrokClient.instances = 0
    settings = _stub_settings()

    with patch.object(main, "_build_grok_client_for_worker", _build_dummy):
        first = main._get_or_create_worker_grok_client(settings)
        main.reset_worker_grok_client_cache()
        second = main._get_or_create_worker_grok_client(settings)

    assert first is not second
    assert _DummyGrokClient.instances == 2
