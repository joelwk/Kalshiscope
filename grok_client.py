from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from config import (
    SearchConfig,
    Settings,
    XAI_WEB_SEARCH_ALLOWED_DOMAINS_LIMIT,
    XAI_X_SEARCH_ALLOWED_HANDLES_LIMIT,
)
from logging_config import get_logger
from models import Market, TradeDecision
from prompts.loader import load_lines, load_prompt, render
from research_profiles import is_commodity_market
from xai_provider import XAIProvider

logger = get_logger(__name__)

_RE_IMPLIED = re.compile(r"implied prob(?:ability)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)%?", re.IGNORECASE)
_RE_MY_PROB = re.compile(r"my prob(?:ability)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)%?", re.IGNORECASE)
_RE_EDGE = re.compile(r"edge\s*[:=]\s*([+-]?[0-9]+(?:\.[0-9]+)?)%?", re.IGNORECASE)
_RE_SPORTS_MISMATCH = re.compile(r"unrelated sports content", re.IGNORECASE)
_RE_NO_EXTERNAL_ODDS = re.compile(
    r"(no (?:external )?(?:betting )?odds found|implied[_ ]prob(?:ability)?\s*[:=]\s*(?:unknown|n/?a|null))",
    re.IGNORECASE,
)
_RE_LOW_INFORMATION = re.compile(
    r"(no (?:search )?results|zero mentions|no mentions of|no evidence(?: found)?|"
    r"no information(?: found)?|no data(?: available)?|could not find (?:any )?"
    r"(?:evidence|information|data))",
    re.IGNORECASE,
)
_RE_PREVIEW_OR_PROXY_SOURCE = re.compile(
    r"\b(preview|probable|probables|projected|projection|expected|matchup|form|"
    r"pre-game|pregame|lineup preview|odds preview|scheduled)\b",
    re.IGNORECASE,
)
_RE_WEATHER_OBS_LOCKED = re.compile(
    r"(observ(?:ed|ation)[^\.]{0,80}(?:already|exceed|surpass|hit|locked)|already (?:above|below|over|under|exceeded)|"
    r"physically impossible|threshold (?:already )?(?:met|exceeded)|high already reached)",
    re.IGNORECASE,
)
_RE_DEFINITIVE_OUTCOME_SIGNAL = re.compile(
    r"(final score|game (?:completed|concluded|final)|confirmed|official recap|box score)",
    re.IGNORECASE,
)
_RE_SETTLEMENT_ALIGNED_SOURCE_SIGNAL = re.compile(
    r"(settlement|resolution criter(?:ia|ion)|official recap|box score|final score|"
    r"game (?:completed|concluded|final)|weather\.gov|nws|noaa|asos|metar|"
    r"observation|observed|official station|exchange bulletin|confirmed outcome|"
    r"threshold (?:already )?(?:met|exceeded)|live quote|quote page|spot price|"
    r"current spot|current price|observed value|timestamp)",
    re.IGNORECASE,
)
_REQUIRED_DECISION_FIELDS = {"should_trade", "outcome", "confidence", "bet_size_pct", "reasoning"}
_DEFAULT_XAI_CLIENT_TIMEOUT_SECONDS = 120
_DEFAULT_STREAM_TIMEOUT_SECONDS = 120
_EDGE_CONSISTENCY_TOLERANCE = 0.03
_PROB_CONSISTENCY_TOLERANCE = 0.08
_MIN_MARKET_EDGE_FOR_TRADE = 0.03
_LOW_QUALITY_EDGE_BUFFER = 0.08
_LOW_QUALITY_EVIDENCE_THRESHOLD = 0.45
_DOUBLE_BLIND_ABSTAIN_EVIDENCE_THRESHOLD = 0.50
_EVIDENCE_OVERRIDE_MIN_CONFIDENCE = 0.90
_EVIDENCE_OVERRIDE_MIN_MARKET_EDGE = 0.15
_EVIDENCE_OVERRIDE_MIN_QUALITY = 0.60
_WEATHER_OBS_CONFIDENCE_FLOOR = 0.85
_WEATHER_OBS_EVIDENCE_FLOOR = 0.75
_DIRECT_FALLBACK_GATE_OVERRIDE_MIN_EVIDENCE = 0.65
_DEFINITIVE_OUTCOME_EVIDENCE_FLOOR = 0.72
_VERIFIABLE_EVIDENCE_KEYWORDS = (
    "official",
    "official recap",
    "reuters",
    "associated press",
    "ap ",
    "box score",
    "final score",
    "nws",
    "weather.gov",
    "cli",
    "asos",
    "metar",
    "observation",
    "confirmed",
    "resolved",
    "settlement",
    "exchange",
)
_MAX_MODEL_RESPONSE_LOG_CHARS = 500
_ANALYSIS_MAX_ATTEMPTS = 3
# Deep refinement always has `previous_analysis` as a safe fallback in the
# caller, so a single failed attempt should short-circuit rather than burn
# budget that other candidates could use. Initial analyses (no fallback)
# still use the full attempt count.
_DEEP_ANALYSIS_MAX_ATTEMPTS = 1
_ANALYSIS_RETRY_WAIT_SECONDS = 2
_DEFAULT_MAX_ANALYSIS_BUDGET_SECONDS = 240
_SLOW_FAILURE_THRESHOLD_MS = 15_000
# Reserve a small cushion inside the per-attempt deadline so post-stream work
# (JSON parse, validation, logging) still fits within the overall budget.
_STREAM_DEADLINE_SAFETY_MARGIN_SECONDS = 1.0
# Do not start a new stream attempt unless this much budget remains; avoids
# spinning up an xAI request that is virtually guaranteed to time out.
_MIN_STREAM_ATTEMPT_SECONDS = 8.0
_SYSTEM_PROMPT_SHARED = load_prompt("system/shared_output_rules")
_SYSTEM_PROMPT_ANALYZE = load_prompt("system/analyze_market")
_SYSTEM_PROMPT_DEEP = load_prompt("system/analyze_market_deep")


def _default_search_config(settings: Settings | None = None) -> SearchConfig:
    resolved = settings or Settings()
    now = datetime.now(timezone.utc)
    from_date = now - timedelta(hours=resolved.SEARCH_LOOKBACK_HOURS)
    return SearchConfig(
        from_date=from_date,
        to_date=now,
        allowed_domains=list(resolved.SEARCH_ALLOWED_DOMAINS),
        allowed_x_handles=list(resolved.SEARCH_ALLOWED_X_HANDLES),
        source_domains_pool=list(resolved.SEARCH_ALLOWED_DOMAINS),
        source_x_handles_pool=list(resolved.SEARCH_ALLOWED_X_HANDLES),
        max_allowed_domains=resolved.SEARCH_PROFILE_MAX_DOMAINS,
        max_allowed_x_handles=resolved.SEARCH_PROFILE_MAX_X_HANDLES,
        multimedia_confidence_range=resolved.MULTIMEDIA_CONFIDENCE_THRESHOLD,
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON object from potentially mixed text response."""
    if not text:
        raise ValueError("Empty response from Grok")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in Grok response")
    snippet = text[start : end + 1]
    return json.loads(snippet)


def _normalize_model_response_text(text: str) -> str:
    """Normalize model responses before JSON parsing."""
    normalized = text.strip()
    fenced_block = re.search(r"```(?:json)?\s*(.*?)\s*```", normalized, re.IGNORECASE | re.DOTALL)
    if fenced_block:
        return fenced_block.group(1).strip()
    return normalized


def _repair_common_json_key_issues(text: str) -> str:
    """Repair common JSON-like key formatting issues without touching values."""
    return re.sub(r"([{,]\s*)'([A-Za-z_][A-Za-z0-9_]*)'\s*:", r'\1"\2":', text)


def _response_preview(text: str, max_chars: int = _MAX_MODEL_RESPONSE_LOG_CHARS) -> str:
    preview = " ".join(text.split())
    if len(preview) <= max_chars:
        return preview
    return preview[:max_chars] + "..."


_TRANSPORT_RESET_MARKERS: tuple[str, ...] = (
    "rst_stream",
    "statuscode.unavailable",
    "unavailable",
    "connection reset",
    "connection aborted",
    "connection refused",
    "broken pipe",
    "eof occurred",
)


def _is_transport_reset_error(exc: Exception) -> bool:
    """Transport-layer gRPC/HTTP resets that are safely retriable."""
    error_text = str(exc).lower()
    return any(marker in error_text for marker in _TRANSPORT_RESET_MARKERS)


def _is_timeout_class_error(exc: Exception) -> bool:
    """Server-side gRPC deadline or our own stream timeout — both transient."""
    error_text = str(exc).lower()
    if "deadline_exceeded" in error_text or "deadline exceeded" in error_text:
        return True
    if isinstance(exc, TimeoutError) and "grok stream exceeded" in error_text:
        return True
    return False


def _is_retriable_grok_error(exc: Exception, duration_ms: float) -> bool:
    """Classify transient failures that should be retried.

    Timeout-class errors (gRPC DEADLINE_EXCEEDED and our own stream timeout)
    and transport-layer resets (RST_STREAM / UNAVAILABLE) are always retriable
    regardless of duration, since they indicate an interrupted stream rather
    than a content failure.

    A fast "Empty response from Grok" (sub-_SLOW_FAILURE_THRESHOLD_MS) is
    treated as transient: a stream that finishes in single-digit seconds with
    zero content is far more consistent with an upstream blip than a real
    content failure. Slow empty responses still fall through to the slow-
    failure short-circuit so we don't burn budget retrying genuine outages.
    """
    if _is_timeout_class_error(exc) or _is_transport_reset_error(exc):
        return True
    error_text = str(exc).lower()
    if (
        "empty response from grok" in error_text
        and duration_ms < _SLOW_FAILURE_THRESHOLD_MS
    ):
        return True
    if duration_ms >= _SLOW_FAILURE_THRESHOLD_MS:
        return False
    retriable_markers = (
        "statuscode.internal",
        "internal server error",
        "503",
        "temporarily unavailable",
    )
    return any(marker in error_text for marker in retriable_markers)


_QUOTA_EXHAUSTED_MARKERS: tuple[str, ...] = (
    "resource_exhausted",
    "available credits",
    "monthly spending limit",
    "reached its monthly spending limit",
)


def _is_quota_exhausted_grok_error(exc: Exception) -> bool:
    """Detect xAI account quota/credit exhaustion (non-retriable, non-transient)."""
    error_text = str(exc).lower()
    return any(marker in error_text for marker in _QUOTA_EXHAUSTED_MARKERS)


def _extract_usage_metrics(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
            "cached_tokens": None,
        }

    def _read(mapping_or_obj: Any, key: str) -> Any:
        if mapping_or_obj is None:
            return None
        if isinstance(mapping_or_obj, dict):
            return mapping_or_obj.get(key)
        return getattr(mapping_or_obj, key, None)

    prompt_details = _read(usage, "prompt_tokens_details")
    completion_details = _read(usage, "completion_tokens_details")
    return {
        "prompt_tokens": _read(usage, "prompt_tokens"),
        "completion_tokens": _read(usage, "completion_tokens"),
        "reasoning_tokens": _read(completion_details, "reasoning_tokens"),
        "cached_tokens": _read(prompt_details, "cached_tokens"),
    }


def _format_previous_analysis(previous: TradeDecision | None) -> str:
    if not previous:
        return "None"
    reasoning = previous.reasoning or ""
    reasoning = reasoning.replace("\n", " ").strip()
    if len(reasoning) > 400:
        reasoning = reasoning[:400] + "..."
    return render(
        "user/fragments/previous_analysis_summary",
        should_trade=previous.should_trade,
        outcome=previous.outcome,
        confidence=previous.confidence,
        bet_size_pct=previous.bet_size_pct,
        edge_external=previous.edge_external,
        evidence_quality=previous.evidence_quality,
        reasoning=reasoning,
    )


def _format_market_outcome_prices(market: Market) -> str:
    parts: list[str] = []
    for outcome in market.outcomes or []:
        if outcome.price is not None and 0.0 <= outcome.price <= 1.0:
            parts.append(f"{outcome.name}: {outcome.price:.3f}")
            continue
        if outcome.odds is not None and outcome.odds > 0:
            implied = 1.0 / outcome.odds
            parts.append(f"{outcome.name}: {implied:.3f} (from odds)")
            continue
        parts.append(f"{outcome.name}: N/A")
    return ", ".join(parts) if parts else "N/A"


def _category_research_hint(profile_name: str, market: Market | None = None) -> str:
    if profile_name == "sports":
        return load_prompt("user/category_hints/sports")
    if profile_name == "politics":
        return load_prompt("user/category_hints/politics")
    if profile_name == "crypto":
        return load_prompt("user/category_hints/crypto")
    if profile_name == "weather":
        return load_prompt("user/category_hints/weather")
    if profile_name == "speech":
        is_mention_market = bool(
            market is not None and re.search(r"MENTION", market.id or "", re.IGNORECASE)
        )
        if is_mention_market:
            return load_prompt("user/category_hints/speech_mention")
        return load_prompt("user/category_hints/speech_general")
    if profile_name == "music":
        return load_prompt("user/category_hints/music")
    if profile_name == "entertainment":
        return load_prompt("user/category_hints/entertainment")
    if market is not None and is_commodity_market(market):
        return load_prompt("user/category_hints/commodities")
    return load_prompt("user/category_hints/generic")


class GrokClient:
    """Client for interacting with xAI Grok for market analysis."""
    _init_log_emitted = False

    def __init__(
        self,
        api_key: str,
        model: str = "grok-3",
        model_deep: str | None = None,
        min_bet_usdc: float = 2.0,
        max_bet_usdc: float = 10.0,
        search_config: SearchConfig | None = None,
        settings: Settings | None = None,
        provider: XAIProvider | None = None,
    ) -> None:
        resolved_settings = settings or Settings()
        self.settings = resolved_settings
        self.xai_client_timeout_seconds = max(
            1,
            int(
                getattr(
                    resolved_settings,
                    "XAI_CLIENT_TIMEOUT_SECONDS",
                    _DEFAULT_XAI_CLIENT_TIMEOUT_SECONDS,
                )
            ),
        )
        self.stream_timeout_seconds = max(
            1,
            int(
                getattr(
                    resolved_settings,
                    "GROK_STREAM_TIMEOUT_SECONDS",
                    _DEFAULT_STREAM_TIMEOUT_SECONDS,
                )
            ),
        )
        self.analysis_budget_seconds = max(
            1,
            int(
                getattr(
                    resolved_settings,
                    "GROK_ANALYSIS_MAX_BUDGET_SECONDS",
                    _DEFAULT_MAX_ANALYSIS_BUDGET_SECONDS,
                )
            ),
        )
        self.provider = provider or XAIProvider(
            api_key=api_key,
            timeout_seconds=self.xai_client_timeout_seconds,
        )
        self.model = model
        self.model_deep = model_deep or model
        self.min_bet_usdc = min_bet_usdc
        self.max_bet_usdc = max_bet_usdc
        self.default_search_config = search_config or _default_search_config(settings)
        log_fn = logger.info if not GrokClient._init_log_emitted else logger.debug
        GrokClient._init_log_emitted = True
        log_fn(
            "GrokClient initialized: model=%s model_deep=%s stream_timeout=%ds analysis_budget=%ds xai_client_timeout=%ds",
            model,
            self.model_deep,
            self.stream_timeout_seconds,
            self.analysis_budget_seconds,
            self.xai_client_timeout_seconds,
            data={
                "model": model,
                "model_deep": self.model_deep,
                "stream_timeout_seconds": self.stream_timeout_seconds,
                "analysis_budget_seconds": self.analysis_budget_seconds,
                "xai_client_timeout_seconds": self.xai_client_timeout_seconds,
            },
        )

    @property
    def client(self):
        return self.provider.client

    @client.setter
    def client(self, value) -> None:
        self.provider.client = value

    def _active_search_config(self, search_config: SearchConfig | None) -> SearchConfig:
        config = search_config or self.default_search_config
        if not config.from_date or not config.to_date:
            defaults = _default_search_config(self.settings)
            config = SearchConfig(
                from_date=config.from_date or defaults.from_date,
                to_date=config.to_date or defaults.to_date,
                allowed_domains=config.allowed_domains or defaults.allowed_domains,
                allowed_x_handles=config.allowed_x_handles or defaults.allowed_x_handles,
                source_domains_pool=config.source_domains_pool or defaults.source_domains_pool,
                source_x_handles_pool=(
                    config.source_x_handles_pool or defaults.source_x_handles_pool
                ),
                max_allowed_domains=(
                    config.max_allowed_domains or defaults.max_allowed_domains
                ),
                max_allowed_x_handles=(
                    config.max_allowed_x_handles or defaults.max_allowed_x_handles
                ),
                enable_multimedia=config.enable_multimedia,
                multimedia_confidence_range=config.multimedia_confidence_range,
                profile_name=config.profile_name,
                lookback_hours=config.lookback_hours,
            )
        # Keep within xAI server-side limits while preserving prioritized profile order.
        max_domains = min(
            XAI_WEB_SEARCH_ALLOWED_DOMAINS_LIMIT,
            max(1, int(config.max_allowed_domains or 5)),
        )
        max_handles = min(
            XAI_X_SEARCH_ALLOWED_HANDLES_LIMIT,
            max(1, int(config.max_allowed_x_handles or 10)),
        )
        if len(config.allowed_domains) > max_domains:
            config.allowed_domains = config.allowed_domains[:max_domains]
        if len(config.allowed_x_handles) > max_handles:
            config.allowed_x_handles = config.allowed_x_handles[:max_handles]
        return config

    def _should_enable_multimedia(
        self,
        market: Market,
        decision: TradeDecision | None,
        config: SearchConfig,
    ) -> bool:
        """Enable multimedia for borderline confidence or urgent markets."""
        if config.profile_name == "speech":
            return True
        if decision:
            lower, upper = config.multimedia_confidence_range
            if lower <= decision.confidence <= upper:
                return True
        if market.close_time:
            close_time = market.close_time
            if close_time.tzinfo is None:
                close_time = close_time.replace(tzinfo=timezone.utc)
            if (close_time - datetime.now(timezone.utc)).total_seconds() <= 86400:
                return True
        return config.enable_multimedia

    @staticmethod
    def _market_implied_probability(market: Market, outcome: str) -> float | None:
        for market_outcome in market.outcomes or []:
            if market_outcome.name.upper() != outcome.upper():
                continue
            if market_outcome.price is not None and 0.0 <= market_outcome.price <= 1.0:
                return market_outcome.price
            if market_outcome.odds is not None and market_outcome.odds > 0:
                return 1.0 / market_outcome.odds
        return None

    @staticmethod
    def _normalize_outcome_label(value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").strip()).lower()

    @classmethod
    def _canonical_outcome_for_market(cls, market: Market, outcome: str) -> str | None:
        if not outcome:
            return None
        normalized = cls._normalize_outcome_label(outcome)
        for market_outcome in market.outcomes or []:
            if cls._normalize_outcome_label(market_outcome.name) == normalized:
                return market_outcome.name

        yes_aliases = {"yes", "true", "1"}
        no_aliases = {"no", "false", "0"}
        if normalized in yes_aliases:
            for market_outcome in market.outcomes or []:
                if cls._normalize_outcome_label(market_outcome.name) in yes_aliases:
                    return market_outcome.name
        if normalized in no_aliases:
            for market_outcome in market.outcomes or []:
                if cls._normalize_outcome_label(market_outcome.name) in no_aliases:
                    return market_outcome.name
        return None

    @staticmethod
    def _extract_metric_from_reasoning(reasoning: str, regex: re.Pattern[str]) -> float | None:
        match = regex.search(reasoning or "")
        if not match:
            return None
        value = float(match.group(1))
        if value > 1.0:
            value = value / 100.0
        return max(0.0, min(1.0, value))

    @staticmethod
    def _extract_edge_from_reasoning(reasoning: str) -> float | None:
        match = _RE_EDGE.search(reasoning or "")
        if not match:
            return None
        value = float(match.group(1))
        if abs(value) > 1.0:
            value = value / 100.0
        return max(-1.0, min(1.0, value))

    def _derive_edge(
        self,
        implied: float | None,
        my_prob: float | None,
        explicit_edge: float | None,
        reasoning: str,
        market_id: str,
    ) -> tuple[float | None, str]:
        # Deterministic primary source: if both probabilities exist, edge is computed.
        if implied is not None and my_prob is not None:
            return max(-1.0, min(1.0, my_prob - implied)), "computed"

        fallback_edge = explicit_edge
        if fallback_edge is None:
            fallback_edge = self._extract_edge_from_reasoning(reasoning)
        if fallback_edge is not None:
            logger.debug(
                "Edge fallback used due to missing implied/my_prob: market=%s",
                market_id,
                data={"market_id": market_id, "edge_fallback": fallback_edge},
            )
            return max(-1.0, min(1.0, fallback_edge)), "fallback"
        return None, "none"

    @staticmethod
    def _has_verifiable_source_signal(reasoning: str) -> bool:
        normalized_reasoning = (reasoning or "").lower()
        return any(keyword in normalized_reasoning for keyword in _VERIFIABLE_EVIDENCE_KEYWORDS)

    @staticmethod
    def _source_match_class(
        reasoning: str,
        *,
        has_verifiable_signal: bool,
        has_definitive_outcome_signal: bool,
        no_external_odds: bool,
        low_information: bool,
    ) -> str:
        normalized_reasoning = (reasoning or "").lower()
        if low_information or (no_external_odds and not has_verifiable_signal):
            return "missing_or_absence_only"
        if _RE_PREVIEW_OR_PROXY_SOURCE.search(normalized_reasoning):
            return "preview_or_proxy"
        if has_definitive_outcome_signal or _RE_WEATHER_OBS_LOCKED.search(normalized_reasoning):
            return "settlement_aligned"
        if _RE_SETTLEMENT_ALIGNED_SOURCE_SIGNAL.search(normalized_reasoning):
            return "settlement_aligned"
        if has_verifiable_signal:
            return "verifiable_unmatched"
        return "unverified"

    @staticmethod
    def _evidence_basis_class(
        reasoning: str,
        edge_source: str,
        has_verifiable_signal: bool,
        low_information: bool,
        source_match_class: str = "",
    ) -> str:
        normalized_reasoning = (reasoning or "").lower()
        has_absence_signal = any(
            token in normalized_reasoning
            for token in (
                "no transcript",
                "no mentions",
                "no evidence",
                "no chart",
                "no data",
                "no external odds",
            )
        )
        if (
            has_absence_signal
            and source_match_class != "preview_or_proxy"
            and edge_source in {"fallback", "none"}
        ):
            return "absence_only"
        if (
            has_verifiable_signal
            and not low_information
            and source_match_class == "settlement_aligned"
        ):
            return "direct"
        return "proxy"

    def _validate_and_enrich_decision(
        self,
        market: Market,
        decision: TradeDecision,
        profile_name: str,
    ) -> TradeDecision:
        canonical_outcome = self._canonical_outcome_for_market(market, decision.outcome)
        if canonical_outcome is None:
            mismatch_reason = (
                f"[Outcome mismatch] Outcome '{decision.outcome}' does not match market outcomes "
                f"{[outcome.name for outcome in market.outcomes]}."
            )
            evidence_quality = max(0.0, min(0.2, decision.evidence_quality or 0.0))
            return decision.model_copy(
                update={
                    "should_trade": False,
                    "bet_size_pct": 0.0,
                    "evidence_quality": evidence_quality,
                    "reasoning": f"{mismatch_reason} {decision.reasoning}",
                }
            )

        implied = decision.implied_prob_external
        my_prob = decision.my_prob
        explicit_edge = decision.edge_external

        if implied is None:
            implied = self._extract_metric_from_reasoning(decision.reasoning, _RE_IMPLIED)
        if my_prob is None:
            my_prob = self._extract_metric_from_reasoning(decision.reasoning, _RE_MY_PROB)

        edge, edge_source = self._derive_edge(
            implied=implied,
            my_prob=my_prob,
            explicit_edge=explicit_edge,
            reasoning=decision.reasoning,
            market_id=market.id,
        )

        consistency_ok = True
        if implied is not None and my_prob is not None and edge is not None:
            expected_edge = my_prob - implied
            if abs(expected_edge - edge) > _EDGE_CONSISTENCY_TOLERANCE:
                consistency_ok = False

        prob_consistency_ok = True
        if my_prob is not None and abs(my_prob - decision.confidence) > _PROB_CONSISTENCY_TOLERANCE:
            prob_consistency_ok = False

        raw_evidence_quality = max(0.0, min(1.0, float(decision.evidence_quality or 0.0)))
        no_external_odds = bool(_RE_NO_EXTERNAL_ODDS.search(decision.reasoning or ""))
        low_information = bool(_RE_LOW_INFORMATION.search(decision.reasoning or ""))
        has_verifiable_signal = self._has_verifiable_source_signal(decision.reasoning)
        has_definitive_outcome_signal = bool(
            _RE_DEFINITIVE_OUTCOME_SIGNAL.search(decision.reasoning or "")
        )
        source_match_class = self._source_match_class(
            decision.reasoning or "",
            has_verifiable_signal=has_verifiable_signal,
            has_definitive_outcome_signal=has_definitive_outcome_signal,
            no_external_odds=no_external_odds,
            low_information=low_information,
        )
        prob_component = 0.0
        if implied is not None and my_prob is not None:
            prob_component = 0.55
        elif my_prob is not None:
            prob_component = 0.25

        source_component = 0.0
        if implied is not None:
            source_component += 0.25
        if decision.reasoning and "as of" in decision.reasoning.lower():
            source_component += 0.05
        if no_external_odds:
            source_component = min(source_component, 0.05)

        consistency_component = 0.2
        if not consistency_ok:
            consistency_component -= 0.15
        if not prob_consistency_ok:
            consistency_component -= 0.10

        evidence_quality = prob_component + source_component + max(0.0, consistency_component)
        if no_external_odds and not has_verifiable_signal:
            evidence_quality = min(evidence_quality, 0.5)
        if low_information:
            evidence_quality = min(evidence_quality, 0.5)
        if profile_name != "sports" and _RE_SPORTS_MISMATCH.search(decision.reasoning or ""):
            evidence_quality = max(0.0, evidence_quality - 0.4)
            logger.warning(
                "Research mismatch detected: market=%s profile=%s",
                market.id,
                profile_name,
                data={"market_id": market.id, "profile_name": profile_name},
            )
        evidence_quality = max(0.0, min(1.0, evidence_quality))
        if no_external_odds and source_match_class != "settlement_aligned":
            evidence_quality = min(evidence_quality, 0.50)
        if source_match_class == "preview_or_proxy":
            evidence_quality = min(evidence_quality, 0.60)
        market_implied = self._market_implied_probability(market, canonical_outcome)
        if edge_source == "fallback":
            if market_implied is not None and my_prob is not None:
                edge = abs(my_prob - market_implied)
            else:
                edge = 0.0
        evidence_basis_class = self._evidence_basis_class(
            reasoning=decision.reasoning,
            edge_source=edge_source,
            has_verifiable_signal=has_verifiable_signal,
            low_information=low_information,
            source_match_class=source_match_class,
        )
        active_settings = self.settings or Settings()
        proxy_confidence_cap = max(0.0, min(1.0, active_settings.GROK_PROXY_CONFIDENCE_CAP))
        low_info_confidence_cap = max(
            0.0,
            min(1.0, active_settings.GROK_LOW_INFO_CONFIDENCE_CAP),
        )
        fallback_min_evidence_quality = max(
            0.0,
            min(1.0, active_settings.GROK_FALLBACK_MIN_EVIDENCE_QUALITY),
        )
        abstain_evidence_threshold = max(
            0.0,
            min(1.0, active_settings.GROK_ABSTAIN_EVIDENCE_THRESHOLD),
        )
        validated_confidence = float(max(0.0, min(1.0, decision.confidence)))
        if (
            decision.should_trade
            and edge_source in {"fallback", "none"}
            and evidence_basis_class != "direct"
            and validated_confidence > proxy_confidence_cap
        ):
            validated_confidence = proxy_confidence_cap
        if (
            decision.should_trade
            and edge_source in {"fallback", "none"}
            and low_information
            and validated_confidence > low_info_confidence_cap
        ):
            validated_confidence = low_info_confidence_cap
        market_edge = (
            (validated_confidence - market_implied)
            if market_implied is not None
            else None
        )
        primary_source_url = str(decision.primary_source_url or "").strip()
        primary_source_required_for_direct = profile_name != "sports"
        primary_source_satisfies_direct = (
            bool(primary_source_url) or not primary_source_required_for_direct
        )
        if evidence_basis_class == "direct" and not primary_source_satisfies_direct:
            evidence_basis_class = "proxy"
        evidence_quality_floor_applied: str | None = None
        evidence_floor_suppressed_reason: str | None = None
        verifiable_floor_allowed = (
            has_verifiable_signal
            and not low_information
            and source_match_class == "settlement_aligned"
            and primary_source_satisfies_direct
        )
        if has_verifiable_signal and not verifiable_floor_allowed:
            if low_information:
                evidence_floor_suppressed_reason = "low_information"
            elif source_match_class == "preview_or_proxy":
                evidence_floor_suppressed_reason = "preview_or_proxy_source"
            elif no_external_odds and source_match_class == "missing_or_absence_only":
                evidence_floor_suppressed_reason = "no_external_odds"
            elif (
                source_match_class == "settlement_aligned"
                and not primary_source_satisfies_direct
            ):
                evidence_floor_suppressed_reason = "missing_primary_source_url"
            else:
                evidence_floor_suppressed_reason = "source_not_settlement_aligned"
        if verifiable_floor_allowed:
            evidence_floor = 0.60
            if (
                has_definitive_outcome_signal
                and decision.likelihood_ratio is not None
                and decision.likelihood_ratio >= 10.0
            ):
                evidence_floor = _DEFINITIVE_OUTCOME_EVIDENCE_FLOOR
            if market_edge is None or abs(market_edge) < _LOW_QUALITY_EDGE_BUFFER:
                evidence_floor = max(evidence_floor, 0.55)
            if evidence_quality < evidence_floor:
                evidence_quality = evidence_floor
                if evidence_floor >= _DEFINITIVE_OUTCOME_EVIDENCE_FLOOR:
                    evidence_quality_floor_applied = "definitive_outcome_floor"
                else:
                    evidence_quality_floor_applied = "verifiable_signal_floor"
        if (
            active_settings.EVIDENCE_QUALITY_HIGH_CONFIDENCE_OVERRIDE
            and market_edge is not None
            and decision.confidence >= _EVIDENCE_OVERRIDE_MIN_CONFIDENCE
            and market_edge >= _EVIDENCE_OVERRIDE_MIN_MARKET_EDGE
            and verifiable_floor_allowed
        ):
            if evidence_quality < _EVIDENCE_OVERRIDE_MIN_QUALITY:
                evidence_quality = _EVIDENCE_OVERRIDE_MIN_QUALITY
                evidence_quality_floor_applied = "high_confidence_override_floor"
        if (
            profile_name == "weather"
            and (decision.raw_confidence or decision.confidence) >= _WEATHER_OBS_CONFIDENCE_FLOOR
            and _RE_WEATHER_OBS_LOCKED.search(decision.reasoning or "")
            and primary_source_satisfies_direct
        ):
            if evidence_quality < _WEATHER_OBS_EVIDENCE_FLOOR:
                evidence_quality = _WEATHER_OBS_EVIDENCE_FLOOR
                evidence_quality_floor_applied = "weather_observed_floor"
        definitive_outcome_detected = (
            source_match_class == "settlement_aligned"
            and has_definitive_outcome_signal
            and decision.likelihood_ratio is not None
            and decision.likelihood_ratio >= 10.0
            and not low_information
            and primary_source_satisfies_direct
        )
        if (
            definitive_outcome_detected
            and evidence_quality < _DEFINITIVE_OUTCOME_EVIDENCE_FLOOR
        ):
            evidence_quality = _DEFINITIVE_OUTCOME_EVIDENCE_FLOOR
            evidence_quality_floor_applied = "definitive_outcome_floor"
        direct_fallback_gate_override = (
            evidence_basis_class == "direct"
            and source_match_class == "settlement_aligned"
            and evidence_quality >= _DIRECT_FALLBACK_GATE_OVERRIDE_MIN_EVIDENCE
        )

        should_trade = decision.should_trade
        gate_reasons: list[str] = []
        if should_trade:
            if market_edge is None:
                should_trade = False
                gate_reasons.append("missing_market_implied")
            elif market_edge < _MIN_MARKET_EDGE_FOR_TRADE:
                should_trade = False
                gate_reasons.append("market_edge_below_min")
            if (
                evidence_quality < _LOW_QUALITY_EVIDENCE_THRESHOLD
                and (market_edge is None or market_edge < _LOW_QUALITY_EDGE_BUFFER)
            ):
                should_trade = False
                gate_reasons.append("low_evidence_quality")
            if (
                not consistency_ok
                and (market_edge is None or market_edge < _LOW_QUALITY_EDGE_BUFFER)
            ):
                should_trade = False
                gate_reasons.append("edge_inconsistent")
            if (
                not prob_consistency_ok
                and (market_edge is None or market_edge < _LOW_QUALITY_EDGE_BUFFER)
            ):
                should_trade = False
                gate_reasons.append("probability_inconsistent")
            if evidence_basis_class == "absence_only":
                should_trade = False
                gate_reasons.append("absence_only_evidence")
            if (
                source_match_class == "preview_or_proxy"
                and edge_source in {"fallback", "none"}
            ):
                should_trade = False
                gate_reasons.append("preview_proxy_without_direct_source")
            if (
                edge_source in {"fallback", "none"}
                and evidence_quality < fallback_min_evidence_quality
                and not direct_fallback_gate_override
            ):
                should_trade = False
                gate_reasons.append("fallback_edge_without_verifiable_signal")

        gate_status = "allow" if should_trade else "block"
        reason_code = ",".join(gate_reasons) if gate_reasons else "ok"
        gate_edge_required = max(_MIN_MARKET_EDGE_FOR_TRADE, _LOW_QUALITY_EDGE_BUFFER)
        gate_edge_actual = market_edge if market_edge is not None else edge
        double_blind_information_gap = no_external_odds and low_information
        abstain = (
            evidence_quality < 0.20
            or (
                double_blind_information_gap
                and evidence_quality
                < max(_DOUBLE_BLIND_ABSTAIN_EVIDENCE_THRESHOLD, abstain_evidence_threshold)
            )
        )
        if abstain:
            should_trade = False
            gate_status = "abstain"
            if "abstain_low_evidence" not in gate_reasons:
                gate_reasons.append("abstain_low_evidence")
            if (
                double_blind_information_gap
                and evidence_quality
                < max(_DOUBLE_BLIND_ABSTAIN_EVIDENCE_THRESHOLD, abstain_evidence_threshold)
                and "abstain_double_blind_information_gap" not in gate_reasons
            ):
                gate_reasons.append("abstain_double_blind_information_gap")
            reason_code = ",".join(gate_reasons)
        if gate_status != "allow":
            logger.debug(
                "Decision validation blocked trade: market=%s reasons=%s gate_edge_required=%.4f gate_edge_actual=%s",
                market.id,
                reason_code,
                gate_edge_required,
                f"{gate_edge_actual:.4f}" if gate_edge_actual is not None else "n/a",
                data={
                    "market_id": market.id,
                    "gate_status": gate_status,
                    "gate_reasons": gate_reasons,
                    "gate_edge_required": gate_edge_required,
                    "gate_edge_actual": gate_edge_actual,
                    "evidence_quality": evidence_quality,
                    "edge_source": edge_source,
                    "evidence_basis_class": evidence_basis_class,
                    "source_match_class": source_match_class,
                    "evidence_floor_suppressed_reason": evidence_floor_suppressed_reason,
                },
            )
        bet_size_pct = decision.bet_size_pct if should_trade else 0.0

        return decision.model_copy(
            update={
                "should_trade": should_trade,
                "abstain": abstain,
                "bet_size_pct": bet_size_pct,
                "outcome": canonical_outcome,
                "confidence": validated_confidence,
                "implied_prob_external": implied,
                "my_prob": my_prob,
                "edge_external": edge,
                "edge_source": edge_source,
                "evidence_basis": evidence_basis_class,
                "evidence_quality": evidence_quality,
                "raw_evidence_quality": raw_evidence_quality,
                "definitive_outcome_detected": definitive_outcome_detected,
                "evidence_quality_floor_applied": evidence_quality_floor_applied,
                "source_match_class": source_match_class,
                "evidence_floor_suppressed_reason": evidence_floor_suppressed_reason,
                "reasoning": (
                    f"[Validated eq={evidence_quality:.2f} gate={gate_status} reason={reason_code} "
                    f"basis={evidence_basis_class} "
                    f"source_match={source_match_class} "
                    f"floor_suppressed={evidence_floor_suppressed_reason or 'none'} "
                    f"edge_market={market_edge if market_edge is not None else 'n/a'} "
                    f"gate_edge_required={gate_edge_required:.4f} "
                    f"gate_edge_actual={gate_edge_actual if gate_edge_actual is not None else 'n/a'} "
                    f"edge_source={edge_source}] {decision.reasoning}"
                ),
            }
        )

    def _merge_partial_deep_response(
        self,
        data: dict[str, Any],
        previous_analysis: TradeDecision | None,
    ) -> dict[str, Any]:
        """Fill missing required decision fields from prior analysis during refinement."""
        if (
            previous_analysis is not None
            and previous_analysis.likelihood_ratio is not None
            and (
                "likelihood_ratio" not in data
                or data.get("likelihood_ratio") is None
            )
        ):
            data = dict(data)
            data["likelihood_ratio"] = previous_analysis.likelihood_ratio
            logger.debug(
                "Deep response omitted likelihood_ratio; reusing previous value: market payload merged",
                data={"likelihood_ratio_source": "inherited_previous"},
            )
        if _REQUIRED_DECISION_FIELDS.issubset(data):
            return data
        if previous_analysis is None:
            return data

        known_updates = {
            key: value
            for key, value in data.items()
            if key in TradeDecision.model_fields and value is not None
        }
        if not known_updates:
            return data

        missing_fields = sorted(_REQUIRED_DECISION_FIELDS - set(data))
        merged = previous_analysis.model_dump()
        merged.update(known_updates)

        if "confidence" not in data and merged.get("my_prob") is not None:
            merged["confidence"] = merged["my_prob"]
        if not str(merged.get("reasoning") or "").strip():
            merged["reasoning"] = previous_analysis.reasoning

        implied = merged.get("implied_prob_external")
        my_prob = merged.get("my_prob")
        if implied is not None and my_prob is not None:
            merged["edge_external"] = my_prob - implied
        elif merged.get("edge_external") is None:
            merged["edge_external"] = previous_analysis.edge_external

        if "should_trade" not in data:
            edge_external = merged.get("edge_external")
            if edge_external is not None and edge_external <= (
                _MIN_MARKET_EDGE_FOR_TRADE + 1e-9
            ):
                merged["should_trade"] = False
        if not merged.get("should_trade", False):
            merged["bet_size_pct"] = 0.0

        logger.warning(
            "Deep analysis returned partial payload; merged with previous analysis: missing=%s",
            ",".join(missing_fields),
            data={
                "missing_fields": missing_fields,
                "provided_fields": sorted(known_updates.keys()),
                "source_completeness": round(
                    len(set(data).intersection(_REQUIRED_DECISION_FIELDS))
                    / len(_REQUIRED_DECISION_FIELDS),
                    3,
                ),
            },
        )
        return merged

    def _normalize_numeric_fields(
        self,
        payload: dict[str, Any],
        market_id: str,
    ) -> dict[str, Any]:
        """Normalize numeric payload fields from LLM output before schema validation."""
        normalized_payload = dict(payload)
        probability_fields = ("confidence", "my_prob", "implied_prob_external")
        edge_fields = ("edge_external",)
        likelihood_fields = ("likelihood_ratio",)

        def _normalize_field(
            field_name: str,
            lower_bound: float,
            upper_bound: float,
        ) -> None:
            if field_name not in normalized_payload:
                return
            raw_value = normalized_payload[field_name]
            if raw_value is None:
                return
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                return
            if not math.isfinite(numeric_value):
                return

            reasons: list[str] = []
            normalized_value = numeric_value
            used_percent_to_decimal = False
            if field_name in probability_fields:
                if normalized_value > 1.0:
                    normalized_value = normalized_value / 100.0
                    reasons.append("percent_to_decimal")
                    used_percent_to_decimal = True
            elif field_name in edge_fields:
                if abs(normalized_value) > 1.0:
                    normalized_value = normalized_value / 100.0
                    reasons.append("percent_to_decimal")
                    used_percent_to_decimal = True

            bounded_value = max(lower_bound, min(upper_bound, normalized_value))
            if bounded_value != normalized_value:
                reasons.append("clamped")

            if bounded_value == numeric_value:
                return

            normalized_payload[field_name] = bounded_value
            log_data = {
                "market_id": market_id,
                "field": field_name,
                "raw_value": raw_value,
                "normalized_value": bounded_value,
                "reason": reasons,
            }
            reason_text = ",".join(reasons) if reasons else "normalized"
            near_boundary_probability_conversion = (
                field_name in probability_fields
                and used_percent_to_decimal
                and 1.0 < numeric_value <= 1.5
                and "clamped" not in reasons
            )
            near_boundary_edge_conversion = (
                field_name in edge_fields
                and used_percent_to_decimal
                and 1.0 < abs(numeric_value) <= 1.5
                and "clamped" not in reasons
            )
            if (
                near_boundary_probability_conversion
                or near_boundary_edge_conversion
            ):
                logger.debug(
                    "Normalized model numeric field: market=%s field=%s raw=%s normalized=%s reason=%s",
                    market_id,
                    field_name,
                    raw_value,
                    bounded_value,
                    reason_text,
                    data=log_data,
                )
            else:
                logger.warning(
                    "Normalized model numeric field: market=%s field=%s raw=%s normalized=%s reason=%s",
                    market_id,
                    field_name,
                    raw_value,
                    bounded_value,
                    reason_text,
                    data=log_data,
                )

        for field_name in probability_fields:
            _normalize_field(field_name, 0.0, 1.0)
        for field_name in edge_fields:
            _normalize_field(field_name, -1.0, 1.0)
        for field_name in likelihood_fields:
            raw_value = normalized_payload.get(field_name)
            if raw_value is None:
                continue
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric_value) or numeric_value <= 0.0:
                normalized_payload[field_name] = None
                logger.warning(
                    "Normalized invalid likelihood ratio to None: market=%s field=%s raw=%s",
                    market_id,
                    field_name,
                    raw_value,
                    data={
                        "market_id": market_id,
                        "field": field_name,
                        "raw_value": raw_value,
                    },
                )
                continue
            normalized_payload[field_name] = numeric_value

        return normalized_payload

    def _build_chat(
        self,
        config: SearchConfig,
        enable_multimedia: bool,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ):
        return self.provider.create_chat(
            model=model or self.model,
            response_format=TradeDecision,
            config=config,
            enable_multimedia=enable_multimedia,
            timeout_seconds=timeout_seconds,
        )

    def _build_market_prompt(
        self,
        market: Market,
        active_config: SearchConfig,
        previous_summary: str,
        deep: bool,
    ) -> str:
        outcome_prices = _format_market_outcome_prices(market)
        constraints = [_category_research_hint(active_config.profile_name, market)]
        constraints.extend(load_lines("user/fragments/constraints_base"))
        if deep:
            constraints.extend(load_lines("user/fragments/constraints_deep"))
        else:
            constraints.insert(
                0,
                render(
                    "user/fragments/bet_range",
                    min_bet_usdc=self.min_bet_usdc,
                    max_bet_usdc=self.max_bet_usdc,
                ),
            )
        return render(
            "user/market_analysis_request",
            ticker=market.id,
            question=market.question,
            subtitle=market.subtitle or "N/A",
            resolution_criteria=market.resolution_criteria or "N/A",
            outcomes=", ".join([o.name for o in market.outcomes]),
            market_outcome_prices=outcome_prices,
            liquidity_usdc=market.liquidity_usdc,
            research_profile=active_config.profile_name,
            lookback_hours=active_config.lookback_hours,
            previous_analysis=previous_summary,
            constraints="\n".join(constraints),
        )

    def _parse_response_payload(self, market_id: str, content: str, deep: bool) -> dict[str, Any]:
        normalized_content = _normalize_model_response_text(content)
        try:
            return json.loads(normalized_content)
        except json.JSONDecodeError:
            deep_suffix = " (deep)" if deep else ""
            logger.warning(
                "Structured response parse fallback invoked for market=%s%s",
                market_id,
                deep_suffix,
                data={"market_id": market_id},
            )
        try:
            return _extract_json(normalized_content)
        except json.JSONDecodeError:
            repaired_content = _repair_common_json_key_issues(normalized_content)
            if repaired_content == normalized_content:
                raise
            deep_suffix = " (deep)" if deep else ""
            logger.warning(
                "Structured response repair fallback invoked for market=%s%s",
                market_id,
                deep_suffix,
                data={"market_id": market_id},
            )
            return _extract_json(repaired_content)

    def _resolve_stream_deadline_seconds(
        self,
        budget_remaining_ms: float | None,
        search_profile: str | None = None,
        deep: bool = False,
    ) -> float:
        """Per-attempt stream deadline, clamped to the remaining analysis budget.

        Profile-aware overrides (when set to a positive value) raise the
        per-attempt cap above the generic ``GROK_STREAM_TIMEOUT_SECONDS``:

        - ``deep=True`` → ``GROK_STREAM_TIMEOUT_SECONDS_DEEP``
        - ``search_profile == "crypto"`` → ``GROK_STREAM_TIMEOUT_SECONDS_CRYPTO``
        - ``search_profile == "weather"`` → ``GROK_STREAM_TIMEOUT_SECONDS_WEATHER``
          (added after cycle 1 review observed weather analyses timing out at
          exactly the legacy 100s ceiling on data-heavy NWS observation prompts)
        """
        base_timeout = float(self.stream_timeout_seconds)
        if (
            deep
            and hasattr(self, "settings")
            and self.settings is not None
        ):
            deep_timeout = getattr(
                self.settings, "GROK_STREAM_TIMEOUT_SECONDS_DEEP", None
            )
            if deep_timeout is not None and int(deep_timeout) > 0:
                base_timeout = max(base_timeout, float(deep_timeout))
        if (
            search_profile == "crypto"
            and hasattr(self, "settings")
            and self.settings is not None
        ):
            crypto_timeout = getattr(
                self.settings, "GROK_STREAM_TIMEOUT_SECONDS_CRYPTO", None
            )
            if crypto_timeout is not None and int(crypto_timeout) > 0:
                base_timeout = max(base_timeout, float(crypto_timeout))
        if (
            search_profile == "weather"
            and hasattr(self, "settings")
            and self.settings is not None
        ):
            weather_timeout = getattr(
                self.settings, "GROK_STREAM_TIMEOUT_SECONDS_WEATHER", None
            )
            if weather_timeout is not None and int(weather_timeout) > 0:
                base_timeout = max(base_timeout, float(weather_timeout))
        if budget_remaining_ms is None or budget_remaining_ms <= 0:
            return base_timeout
        budget_seconds = max(
            0.0, (budget_remaining_ms / 1000.0) - _STREAM_DEADLINE_SAFETY_MARGIN_SECONDS
        )
        return min(base_timeout, budget_seconds)

    def _resolve_rpc_timeout_seconds(self, stream_deadline_seconds: float) -> float:
        """Align the SDK gRPC deadline with the bot's per-attempt stream window."""
        return max(
            1.0,
            min(float(self.xai_client_timeout_seconds), float(stream_deadline_seconds)),
        )

    def _stream_chat_content(
        self,
        chat,
        market_id: str,
        *,
        budget_remaining_ms: float | None = None,
        search_profile: str | None = None,
        deep: bool = False,
        deadline_seconds: float | None = None,
    ) -> tuple[str, int, dict[str, int | None]]:
        content = ""
        chunk_count = 0
        usage_metrics: dict[str, int | None] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
            "cached_tokens": None,
        }
        if deadline_seconds is None:
            deadline_seconds = self._resolve_stream_deadline_seconds(
                budget_remaining_ms,
                search_profile=search_profile,
                deep=deep,
            )
        deadline = time.monotonic() + deadline_seconds
        for response, chunk in chat.stream():
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Grok stream exceeded {deadline_seconds:.1f}s for market {market_id}"
                )
            usage_metrics = _extract_usage_metrics(response)
            if chunk.content:
                content += chunk.content
                chunk_count += 1
        if not content:
            raise ValueError("Empty response from Grok")
        return content, chunk_count, usage_metrics

    def _run_analysis(
        self,
        market: Market,
        *,
        search_config: SearchConfig | None,
        previous_analysis: TradeDecision | None,
        deep: bool,
    ) -> TradeDecision:
        budget_deadline = time.monotonic() + self.analysis_budget_seconds
        last_error: Exception | None = None
        max_attempts = _DEEP_ANALYSIS_MAX_ATTEMPTS if deep else _ANALYSIS_MAX_ATTEMPTS
        for attempt in range(1, max_attempts + 1):
            budget_remaining_ms = max(0.0, (budget_deadline - time.monotonic()) * 1000)
            if budget_remaining_ms <= 0:
                break
            if (
                attempt > 1
                and budget_remaining_ms < _MIN_STREAM_ATTEMPT_SECONDS * 1000.0
            ):
                logger.debug(
                    "Skipping retry with insufficient budget: market=%s attempt=%d remaining_ms=%.0f",
                    market.id,
                    attempt,
                    budget_remaining_ms,
                    data={
                        "market_id": market.id,
                        "retry_attempt": attempt,
                        "budget_remaining_ms": round(budget_remaining_ms, 2),
                        "min_attempt_seconds": _MIN_STREAM_ATTEMPT_SECONDS,
                    },
                )
                break
            try:
                return self._run_analysis_once(
                    market=market,
                    search_config=search_config,
                    previous_analysis=previous_analysis,
                    deep=deep,
                    retry_attempt=attempt,
                    budget_remaining_ms=budget_remaining_ms,
                    max_attempts=max_attempts,
                )
            except Exception as exc:
                last_error = exc
                duration_ms = float(getattr(exc, "_grok_duration_ms", 0.0))
                retriable = _is_retriable_grok_error(exc, duration_ms)
                budget_remaining_ms = max(0.0, (budget_deadline - time.monotonic()) * 1000)
                if (
                    not retriable
                    or attempt >= max_attempts
                    or budget_remaining_ms <= 0
                ):
                    raise
                sleep_seconds = min(
                    _ANALYSIS_RETRY_WAIT_SECONDS,
                    budget_remaining_ms / 1000.0,
                )
                logger.warning(
                    "Retrying %s for market=%s (attempt=%d/%d)",
                    "deep market analysis" if deep else "market analysis",
                    market.id,
                    attempt + 1,
                    max_attempts,
                    data={
                        "market_id": market.id,
                        "deep": deep,
                        "retry_attempt": attempt + 1,
                        "max_attempts": max_attempts,
                        "budget_remaining_ms": round(budget_remaining_ms, 2),
                        "retriable": retriable,
                    },
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
        raise TimeoutError(
            f"Grok analysis budget exhausted for market {market.id}"
        ) from last_error

    def _run_analysis_once(
        self,
        market: Market,
        *,
        search_config: SearchConfig | None,
        previous_analysis: TradeDecision | None,
        deep: bool,
        retry_attempt: int,
        budget_remaining_ms: float,
        max_attempts: int,
    ) -> TradeDecision:
        start_time = time.monotonic()
        active_config = self._active_search_config(search_config)
        previous_summary = _format_previous_analysis(previous_analysis)
        model = self.model_deep if deep else self.model
        phase_label = "deep market analysis" if deep else "market analysis"
        logger.debug(
            "Starting %s: id=%s",
            phase_label,
            market.id,
            data={
                "market_id": market.id,
                "question": market.question[:100],
                "outcomes": [o.name for o in market.outcomes],
                "liquidity_usdc": market.liquidity_usdc,
                "previous_analysis": previous_summary if deep else None,
                "search_profile": active_config.profile_name,
                "lookback_hours": active_config.lookback_hours,
                "model": model,
            },
        )

        content = ""
        try:
            stream_deadline_seconds = self._resolve_stream_deadline_seconds(
                budget_remaining_ms,
                search_profile=getattr(active_config, "profile_name", None),
                deep=deep,
            )
            enable_multimedia = self._should_enable_multimedia(
                market,
                decision=previous_analysis,
                config=active_config,
            )
            chat = self._build_chat(
                active_config,
                enable_multimedia,
                model=model,
                timeout_seconds=self._resolve_rpc_timeout_seconds(
                    stream_deadline_seconds
                ),
            )
            chat.append(
                self.provider.system_message(
                    _SYSTEM_PROMPT_DEEP if deep else _SYSTEM_PROMPT_ANALYZE
                )
            )
            chat.append(
                self.provider.user_message(
                    self._build_market_prompt(
                        market=market,
                        active_config=active_config,
                        previous_summary=previous_summary,
                        deep=deep,
                    )
                )
            )
            content, chunk_count, usage_metrics = self._stream_chat_content(
                chat,
                market.id,
                budget_remaining_ms=budget_remaining_ms,
                search_profile=getattr(active_config, "profile_name", None),
                deep=deep,
                deadline_seconds=stream_deadline_seconds,
            )
            data = self._parse_response_payload(market.id, content, deep=deep)
            raw_payload = dict(data)

            deep_likelihood_ratio_provided = False
            if deep:
                deep_likelihood_ratio_provided = (
                    "likelihood_ratio" in data and data.get("likelihood_ratio") is not None
                )
                data = self._merge_partial_deep_response(data, previous_analysis)

            data = self._normalize_numeric_fields(data, market.id)
            decision = TradeDecision.model_validate(data)
            decision = self._validate_and_enrich_decision(
                market,
                decision,
                profile_name=active_config.profile_name,
            )
            decision = decision.model_copy(
                update={
                    "raw_should_trade": (
                        bool(raw_payload.get("should_trade"))
                        if isinstance(raw_payload.get("should_trade"), bool)
                        else None
                    ),
                    "raw_outcome": (
                        str(raw_payload.get("outcome"))
                        if raw_payload.get("outcome") is not None
                        else None
                    ),
                    "raw_confidence": (
                        float(raw_payload.get("confidence"))
                        if isinstance(raw_payload.get("confidence"), (int, float))
                        else None
                    ),
                    "raw_bet_size_pct": (
                        float(raw_payload.get("bet_size_pct"))
                        if isinstance(raw_payload.get("bet_size_pct"), (int, float))
                        else None
                    ),
                    "raw_reasoning": (
                        str(raw_payload.get("reasoning"))
                        if raw_payload.get("reasoning") is not None
                        else None
                    ),
                    "raw_evidence_quality": (
                        float(raw_payload.get("evidence_quality"))
                        if isinstance(raw_payload.get("evidence_quality"), (int, float))
                        else None
                    ),
                    "prompt_tokens": usage_metrics["prompt_tokens"],
                    "completion_tokens": usage_metrics["completion_tokens"],
                    "reasoning_tokens": usage_metrics["reasoning_tokens"],
                    "cached_tokens": usage_metrics["cached_tokens"],
                }
            )

            likelihood_ratio_source = "missing"
            if decision.likelihood_ratio is not None:
                if deep and deep_likelihood_ratio_provided:
                    likelihood_ratio_source = "deep"
                elif (
                    previous_analysis is not None
                    and previous_analysis.likelihood_ratio is not None
                ):
                    likelihood_ratio_source = "inherited_previous"
                else:
                    likelihood_ratio_source = "unknown"

            total_duration = (time.monotonic() - start_time) * 1000
            question_short = market.question[:60] + "..." if len(market.question) > 60 else market.question
            logger.info(
                "Grok%s decision [%s] '%s' -> trade=%s, conf=%.2f, outcome=%s",
                " deep" if deep else "",
                market.id,
                question_short,
                decision.should_trade,
                decision.confidence,
                decision.outcome,
                data={
                    "market_id": market.id,
                    "question": market.question,
                    "should_trade": decision.should_trade,
                    "abstain": decision.abstain,
                    "confidence": decision.confidence,
                    "outcome": decision.outcome,
                    "bet_size_pct": decision.bet_size_pct,
                    "implied_prob_external": decision.implied_prob_external,
                    "my_prob": decision.my_prob,
                    "edge_external": decision.edge_external,
                    "likelihood_ratio": decision.likelihood_ratio,
                    "likelihood_ratio_source": likelihood_ratio_source if deep else None,
                    "evidence_quality": decision.evidence_quality,
                    "search_profile": active_config.profile_name,
                    "lookback_hours": active_config.lookback_hours,
                    "model": model,
                    "chunks": chunk_count,
                    "prompt_tokens": usage_metrics["prompt_tokens"],
                    "completion_tokens": usage_metrics["completion_tokens"],
                    "reasoning_tokens": usage_metrics["reasoning_tokens"],
                    "cached_tokens": usage_metrics["cached_tokens"],
                    "duration_ms": round(total_duration, 2),
                    "previous_analysis": previous_summary if deep else None,
                },
            )
            return decision
        except Exception as exc:
            duration = (time.monotonic() - start_time) * 1000
            retriable = _is_retriable_grok_error(exc, duration)
            budget_after_error_ms = max(0.0, budget_remaining_ms - duration)
            will_retry = (
                retriable
                and retry_attempt < max_attempts
                and budget_after_error_ms >= _MIN_STREAM_ATTEMPT_SECONDS * 1000.0
            )
            if content:
                logger.debug(
                    "Model response preview for failed %s: market=%s preview=%s",
                    phase_label,
                    market.id,
                    _response_preview(content),
                    data={
                        "market_id": market.id,
                        "response_preview": _response_preview(content),
                    },
                )
            log_fn = logger.warning if will_retry else logger.error
            log_fn(
                "%s failed: id=%s, error=%s, duration=%.2fms",
                phase_label.capitalize(),
                market.id,
                exc,
                duration,
                data={
                    "market_id": market.id,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "duration_ms": round(duration, 2),
                    "retriable": retriable,
                    "will_retry": will_retry,
                    "quota_exhausted": _is_quota_exhausted_grok_error(exc),
                    "retry_attempt": retry_attempt,
                    "budget_remaining_ms": round(budget_after_error_ms, 2),
                    "previous_analysis": previous_summary if deep else None,
                    "search_profile": active_config.profile_name,
                },
            )
            setattr(exc, "_grok_duration_ms", duration)
            raise

    def analyze_market(
        self,
        market: Market,
        search_config: SearchConfig | None = None,
        previous_analysis: TradeDecision | None = None,
    ) -> TradeDecision:
        return self._run_analysis(
            market=market,
            search_config=search_config,
            previous_analysis=previous_analysis,
            deep=False,
        )

    def analyze_market_deep(
        self,
        market: Market,
        previous_analysis: TradeDecision | None = None,
        search_config: SearchConfig | None = None,
    ) -> TradeDecision:
        return self._run_analysis(
            market=market,
            search_config=search_config,
            previous_analysis=previous_analysis,
            deep=True,
        )
