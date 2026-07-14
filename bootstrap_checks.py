"""Fail-fast startup health checks for PredictBot.

Run before the main cycle loop to avoid burning API tokens against a broken
environment (e.g. missing TLS cert bundle, expired credentials).
"""

from __future__ import annotations

import os
from logging_config import get_logger

logger = get_logger(__name__)

# kalshi_auth log values: passed | skipped | not_applicable
KALSHI_AUTH_PASSED = "passed"
KALSHI_AUTH_SKIPPED = "skipped"
KALSHI_AUTH_NOT_APPLICABLE = "not_applicable"


class BootstrapError(RuntimeError):
    """Raised when a startup health check fails."""


def run_bootstrap_checks(
    *,
    kalshi_client: object | None = None,
    skip_api_checks: bool = False,
) -> None:
    """Execute pre-flight checks; raise BootstrapError on any failure."""
    _check_tls_cert_bundle()

    if skip_api_checks:
        kalshi_auth = KALSHI_AUTH_SKIPPED
    elif kalshi_client is None:
        kalshi_auth = KALSHI_AUTH_NOT_APPLICABLE
    else:
        kalshi_auth = _check_kalshi_auth(kalshi_client)

    logger.info(
        "Bootstrap checks passed",
        data={"tls_ok": True, "kalshi_auth": kalshi_auth},
    )


def _check_tls_cert_bundle() -> None:
    try:
        import certifi
        cert_path = certifi.where()
    except ImportError:
        logger.warning("certifi not installed; skipping TLS cert check")
        return

    if not os.path.isfile(cert_path):
        raise BootstrapError(
            f"TLS CA certificate bundle not found at {cert_path!r}. "
            "The bot will fail on every HTTPS request. "
            "Run `pip install --upgrade certifi` or fix the path."
        )
    logger.debug("TLS cert bundle OK: %s", cert_path)


def _check_kalshi_auth(kalshi_client: object) -> str:
    get_balance = getattr(kalshi_client, "get_portfolio_balance", None)
    if get_balance is None:
        logger.debug("Kalshi client has no get_portfolio_balance; skipping auth check")
        return KALSHI_AUTH_NOT_APPLICABLE
    try:
        balance = get_balance()
        logger.debug(
            "Kalshi auth check OK, balance=%s",
            balance,
            data={"balance": balance},
        )
        return KALSHI_AUTH_PASSED
    except Exception as exc:
        raise BootstrapError(
            f"Kalshi API auth check failed: {exc}. "
            "Verify KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH, and network."
        ) from exc
