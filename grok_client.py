from __future__ import annotations

import json
import logging
import math
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed
from xai_sdk import Client
from xai_sdk.chat import system, user
from xai_sdk.tools import web_search, x_search

from config import SearchConfig, Settings
from logging_config import get_logger
from models import Market, TradeDecision

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
_REQUIRED_DECISION_FIELDS = {"should_trade", "outcome", "confidence", "bet_size_pct", "reasoning"}
_XAI_CLIENT_TIMEOUT_SECONDS = 600
_STREAM_TIMEOUT_SECONDS = 300
_EDGE_CONSISTENCY_TOLERANCE = 0.03
_PROB_CONSISTENCY_TOLERANCE = 0.08
_MIN_MARKET_EDGE_FOR_TRADE = 0.05
_LOW_QUALITY_EDGE_BUFFER = 0.08
_LOW_QUALITY_EVIDENCE_THRESHOLD = 0.45
_MAX_MODEL_RESPONSE_LOG_CHARS = 500
_SYSTEM_PROMPT_SHARED = (
    "Output must strictly match the response schema.\n"
    "Respond with a single valid JSON object only; use double-quoted keys and no trailing commas.\n"
    "The `outcome` field must exactly match one of the provided market outcomes.\n"
    "If edge is below 5%, uncertain, or unsupported by verified sources, set should_trade=false and bet_size_pct=0.0.\n"
    "Do not fabricate bookmaker odds or implied probabilities. If no external odds are found, explicitly say so.\n"
    "When search results are empty or irrelevant, reduce confidence and keep evidence quality below 0.50.\n"
    "Absence of evidence is not evidence of absence; treat sparse evidence as lower conviction.\n"
)
_SYSTEM_PROMPT_ANALYZE = (
    "You are an autonomous prediction market analyst focused on finding tradable edge, not picking winners.\n"
    "Use web search and X search to gather recent, source-backed evidence.\n\n"
    + _SYSTEM_PROMPT_SHARED
    + "Set should_trade=true only when edge is meaningful and evidence-backed.\n"
    + "Calibrate probabilities conservatively. For sports/esports, avoid overconfidence and generally keep confidence <=0.80.\n"
    + "Provide concise reasoning that includes implied probability, your probability, and edge.\n"
    + "Include likelihood_ratio as a positive decimal equal to "
    + "P(evidence|predicted_outcome)/P(evidence|alternative_outcome). "
    + "Use likelihood_ratio=1.0 when evidence is neutral."
)
_SYSTEM_PROMPT_DEEP = (
    "You are performing deeper value validation on a prior market analysis.\n"
    "Do not flip outcomes unless evidence and edge are materially stronger than the prior analysis.\n"
    + _SYSTEM_PROMPT_SHARED
    + "If you change outcome, explain the stronger evidence and materially better edge."
)


def _default_search_config() -> SearchConfig:
    now = datetime.now(timezone.utc)
    from_date = now - timedelta(hours=Settings.SEARCH_LOOKBACK_HOURS)
    return SearchConfig(
        from_date=from_date,
        to_date=now,
        allowed_domains=list(Settings.SEARCH_ALLOWED_DOMAINS),
        allowed_x_handles=list(Settings.SEARCH_ALLOWED_X_HANDLES),
        multimedia_confidence_range=Settings.MULTIMEDIA_CONFIDENCE_THRESHOLD,
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
    return (
        "should_trade={should_trade}, outcome={outcome}, confidence={confidence:.2f}, "
        "bet_size_pct={bet_size_pct:.2f}, edge_external={edge_external}, evidence_quality={evidence_quality:.2f}, "
        "reasoning='{reasoning}'"
    ).format(
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


def _category_research_hint(profile_name: str) -> str:
    if profile_name == "sports":
        return (
            "Sports guidance: prioritize current injury reports, lineup/rotation news, schedule fatigue, "
            "home-away splits, and credible odds consensus."
        )
    if profile_name == "politics":
        return (
            "Politics guidance: prioritize reputable polling, official filings/statements, and base-rate priors; "
            "discount unverified viral claims."
        )
    if profile_name == "crypto":
        return (
            "Crypto guidance: prioritize exchange/project primary sources, on-chain confirmations, and "
            "time-sensitive catalysts over rumor accounts."
        )
    return (
        "Generic guidance: prefer primary sources and high-credibility reporting; penalize stale or weakly corroborated claims."
    )


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
    ) -> None:
        self.client = Client(api_key=api_key, timeout=_XAI_CLIENT_TIMEOUT_SECONDS)
        self.model = model
        self.model_deep = model_deep or model
        self.min_bet_usdc = min_bet_usdc
        self.max_bet_usdc = max_bet_usdc
        self.default_search_config = search_config or _default_search_config()
        if not GrokClient._init_log_emitted:
            logger.debug(
                "GrokClient initialized with model=%s model_deep=%s",
                model,
                self.model_deep,
            )
            GrokClient._init_log_emitted = True

    def _active_search_config(self, search_config: SearchConfig | None) -> SearchConfig:
        config = search_config or self.default_search_config
        if not config.from_date or not config.to_date:
            defaults = _default_search_config()
            config = SearchConfig(
                from_date=config.from_date or defaults.from_date,
                to_date=config.to_date or defaults.to_date,
                allowed_domains=config.allowed_domains or defaults.allowed_domains,
                allowed_x_handles=config.allowed_x_handles or defaults.allowed_x_handles,
                enable_multimedia=config.enable_multimedia,
                multimedia_confidence_range=config.multimedia_confidence_range,
                profile_name=config.profile_name,
                lookback_hours=config.lookback_hours,
            )
        # Keep within xAI limits while preserving prioritized order from profile builder.
        if len(config.allowed_domains) > 5:
            config.allowed_domains = config.allowed_domains[:5]
        if len(config.allowed_x_handles) > 10:
            config.allowed_x_handles = config.allowed_x_handles[:10]
        return config

    def _should_enable_multimedia(
        self,
        market: Market,
        decision: TradeDecision | None,
        config: SearchConfig,
    ) -> bool:
        """Enable multimedia for borderline confidence or urgent markets."""
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

        no_external_odds = bool(_RE_NO_EXTERNAL_ODDS.search(decision.reasoning or ""))
        low_information = bool(_RE_LOW_INFORMATION.search(decision.reasoning or ""))
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
        if edge_source == "fallback":
            consistency_component -= 0.05
        if not consistency_ok:
            consistency_component -= 0.15
        if not prob_consistency_ok:
            consistency_component -= 0.10

        evidence_quality = prob_component + source_component + max(0.0, consistency_component)
        if no_external_odds:
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
        market_implied = self._market_implied_probability(market, canonical_outcome)
        market_edge = (
            (decision.confidence - market_implied)
            if market_implied is not None
            else None
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

        gate_status = "allow" if should_trade else "block"
        reason_code = ",".join(gate_reasons) if gate_reasons else "ok"
        bet_size_pct = decision.bet_size_pct if should_trade else 0.0

        return decision.model_copy(
            update={
                "should_trade": should_trade,
                "bet_size_pct": bet_size_pct,
                "outcome": canonical_outcome,
                "implied_prob_external": implied,
                "my_prob": my_prob,
                "edge_external": edge,
                "edge_source": edge_source,
                "evidence_quality": evidence_quality,
                "reasoning": (
                    f"[Validated eq={evidence_quality:.2f} gate={gate_status} reason={reason_code} "
                    f"edge_market={market_edge if market_edge is not None else 'n/a'} "
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
            if edge_external is not None and edge_external < _MIN_MARKET_EDGE_FOR_TRADE:
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
            if abs(normalized_value) > 1.0:
                normalized_value = normalized_value / 100.0
                reasons.append("percent_to_decimal")

            bounded_value = max(lower_bound, min(upper_bound, normalized_value))
            if bounded_value != normalized_value:
                reasons.append("clamped")

            if bounded_value == numeric_value:
                return

            normalized_payload[field_name] = bounded_value
            logger.warning(
                "Normalized model numeric field: market=%s field=%s raw=%s normalized=%s reason=%s",
                market_id,
                field_name,
                raw_value,
                bounded_value,
                ",".join(reasons) if reasons else "normalized",
                data={
                    "market_id": market_id,
                    "field": field_name,
                    "raw_value": raw_value,
                    "normalized_value": bounded_value,
                    "reason": reasons,
                },
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
    ):
        return self.client.chat.create(
            model=model or self.model,
            response_format=TradeDecision,
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

    @retry(
        wait=wait_fixed(2),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger.logger, logging.WARNING),
    )
    def analyze_market(
        self,
        market: Market,
        search_config: SearchConfig | None = None,
        previous_analysis: TradeDecision | None = None,
    ) -> TradeDecision:
        start_time = time.monotonic()
        active_config = self._active_search_config(search_config)
        previous_summary = _format_previous_analysis(previous_analysis)
        content = ""
        usage_metrics: dict[str, int | None] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
            "cached_tokens": None,
        }
        logger.debug(
            "Starting market analysis: id=%s",
            market.id,
            data={
                "market_id": market.id,
                "question": market.question[:100],
                "outcomes": [o.name for o in market.outcomes],
                "liquidity_usdc": market.liquidity_usdc,
                "search_profile": active_config.profile_name,
                "lookback_hours": active_config.lookback_hours,
                "model": self.model,
            },
        )

        try:
            enable_multimedia = self._should_enable_multimedia(
                market,
                decision=previous_analysis,
                config=active_config,
            )
            chat = self._build_chat(active_config, enable_multimedia)
            chat.append(
                system(_SYSTEM_PROMPT_ANALYZE)
            )
            outcome_prices = _format_market_outcome_prices(market)
            chat.append(
                user(
                    "<market_data>\n"
                    f"question={market.question}\n"
                    f"outcomes={', '.join([o.name for o in market.outcomes])}\n"
                    f"market_outcome_prices={outcome_prices}\n"
                    f"liquidity_usdc={market.liquidity_usdc}\n"
                    f"research_profile={active_config.profile_name}\n"
                    f"lookback_hours={active_config.lookback_hours}\n"
                    f"previous_analysis={previous_summary}\n"
                    "</market_data>\n"
                    "<constraints>\n"
                    f"bet_range=${self.min_bet_usdc:.2f}-${self.max_bet_usdc:.2f}\n"
                    f"{_category_research_hint(active_config.profile_name)}\n"
                    "</constraints>\n"
                )
            )

            chunk_count = 0
            deadline = time.monotonic() + _STREAM_TIMEOUT_SECONDS
            for response, chunk in chat.stream():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Grok stream exceeded {_STREAM_TIMEOUT_SECONDS}s for market {market.id}")
                usage_metrics = _extract_usage_metrics(response)
                if chunk.content:
                    content += chunk.content
                    chunk_count += 1

            if not content:
                raise ValueError("Empty response from Grok")

            normalized_content = _normalize_model_response_text(content)
            try:
                data = json.loads(normalized_content)
            except json.JSONDecodeError:
                logger.warning(
                    "Structured response parse fallback invoked for market=%s",
                    market.id,
                    data={"market_id": market.id},
                )
                try:
                    data = _extract_json(normalized_content)
                except json.JSONDecodeError:
                    repaired_content = _repair_common_json_key_issues(normalized_content)
                    if repaired_content == normalized_content:
                        raise
                    logger.warning(
                        "Structured response repair fallback invoked for market=%s",
                        market.id,
                        data={"market_id": market.id},
                    )
                    data = _extract_json(repaired_content)

            data = self._normalize_numeric_fields(data, market.id)
            decision = TradeDecision.model_validate(data)
            decision = self._validate_and_enrich_decision(
                market,
                decision,
                profile_name=active_config.profile_name,
            )

            total_duration = (time.monotonic() - start_time) * 1000
            question_short = market.question[:60] + "..." if len(market.question) > 60 else market.question
            logger.info(
                "Grok decision [%s] '%s' -> trade=%s, conf=%.2f, outcome=%s",
                market.id,
                question_short,
                decision.should_trade,
                decision.confidence,
                decision.outcome,
                data={
                    "market_id": market.id,
                    "question": market.question,
                    "should_trade": decision.should_trade,
                    "confidence": decision.confidence,
                    "outcome": decision.outcome,
                    "bet_size_pct": decision.bet_size_pct,
                    "implied_prob_external": decision.implied_prob_external,
                    "my_prob": decision.my_prob,
                    "edge_external": decision.edge_external,
                    "likelihood_ratio": decision.likelihood_ratio,
                    "evidence_quality": decision.evidence_quality,
                    "search_profile": active_config.profile_name,
                    "lookback_hours": active_config.lookback_hours,
                    "model": self.model,
                    "chunks": chunk_count,
                    "prompt_tokens": usage_metrics["prompt_tokens"],
                    "completion_tokens": usage_metrics["completion_tokens"],
                    "reasoning_tokens": usage_metrics["reasoning_tokens"],
                    "cached_tokens": usage_metrics["cached_tokens"],
                    "duration_ms": round(total_duration, 2),
                },
            )
            return decision

        except Exception as exc:
            duration = (time.monotonic() - start_time) * 1000
            if content:
                logger.debug(
                    "Model response preview for failed analysis: market=%s preview=%s",
                    market.id,
                    _response_preview(content),
                    data={
                        "market_id": market.id,
                        "response_preview": _response_preview(content),
                    },
                )
            logger.error(
                "Market analysis failed: id=%s, error=%s, duration=%.2fms",
                market.id,
                exc,
                duration,
                data={
                    "market_id": market.id,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "duration_ms": round(duration, 2),
                    "search_profile": active_config.profile_name,
                },
            )
            raise

    @retry(
        wait=wait_fixed(2),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger.logger, logging.WARNING),
    )
    def analyze_market_deep(
        self,
        market: Market,
        previous_analysis: TradeDecision | None = None,
        search_config: SearchConfig | None = None,
    ) -> TradeDecision:
        start_time = time.monotonic()
        active_config = self._active_search_config(search_config)
        previous_summary = _format_previous_analysis(previous_analysis)
        content = ""
        usage_metrics: dict[str, int | None] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
            "cached_tokens": None,
        }
        logger.debug(
            "Starting deep market analysis: id=%s",
            market.id,
            data={
                "market_id": market.id,
                "question": market.question[:100],
                "outcomes": [o.name for o in market.outcomes],
                "liquidity_usdc": market.liquidity_usdc,
                "previous_analysis": previous_summary,
                "search_profile": active_config.profile_name,
                "lookback_hours": active_config.lookback_hours,
                "model": self.model_deep,
            },
        )

        try:
            enable_multimedia = self._should_enable_multimedia(
                market,
                decision=previous_analysis,
                config=active_config,
            )
            chat = self._build_chat(
                active_config,
                enable_multimedia,
                model=self.model_deep,
            )
            chat.append(
                system(_SYSTEM_PROMPT_DEEP)
            )
            outcome_prices = _format_market_outcome_prices(market)
            chat.append(
                user(
                    "<market_data>\n"
                    f"question={market.question}\n"
                    f"outcomes={', '.join([o.name for o in market.outcomes])}\n"
                    f"market_outcome_prices={outcome_prices}\n"
                    f"liquidity_usdc={market.liquidity_usdc}\n"
                    f"research_profile={active_config.profile_name}\n"
                    f"lookback_hours={active_config.lookback_hours}\n"
                    f"previous_analysis={previous_summary}\n"
                    "</market_data>\n"
                    "<constraints>\n"
                    f"{_category_research_hint(active_config.profile_name)}\n"
                    "Re-check implied probability and edge from multiple sources.\n"
                    "If edge < 5%, return should_trade=false and bet_size_pct=0.0.\n"
                    "</constraints>\n"
                )
            )

            chunk_count = 0
            deadline = time.monotonic() + _STREAM_TIMEOUT_SECONDS
            for response, chunk in chat.stream():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Grok stream exceeded {_STREAM_TIMEOUT_SECONDS}s for market {market.id}")
                usage_metrics = _extract_usage_metrics(response)
                if chunk.content:
                    content += chunk.content
                    chunk_count += 1

            if not content:
                raise ValueError("Empty response from Grok")
            normalized_content = _normalize_model_response_text(content)
            try:
                data = json.loads(normalized_content)
            except json.JSONDecodeError:
                logger.warning(
                    "Structured response parse fallback invoked for market=%s (deep)",
                    market.id,
                    data={"market_id": market.id},
                )
                try:
                    data = _extract_json(normalized_content)
                except json.JSONDecodeError:
                    repaired_content = _repair_common_json_key_issues(normalized_content)
                    if repaired_content == normalized_content:
                        raise
                    logger.warning(
                        "Structured response repair fallback invoked for market=%s (deep)",
                        market.id,
                        data={"market_id": market.id},
                    )
                    data = _extract_json(repaired_content)

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
            likelihood_ratio_source = "missing"
            if decision.likelihood_ratio is not None:
                if deep_likelihood_ratio_provided:
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
                "Grok deep decision [%s] '%s' -> trade=%s, conf=%.2f, outcome=%s",
                market.id,
                question_short,
                decision.should_trade,
                decision.confidence,
                decision.outcome,
                data={
                    "market_id": market.id,
                    "question": market.question,
                    "should_trade": decision.should_trade,
                    "confidence": decision.confidence,
                    "outcome": decision.outcome,
                    "bet_size_pct": decision.bet_size_pct,
                    "implied_prob_external": decision.implied_prob_external,
                    "my_prob": decision.my_prob,
                    "edge_external": decision.edge_external,
                    "likelihood_ratio": decision.likelihood_ratio,
                    "likelihood_ratio_source": likelihood_ratio_source,
                    "evidence_quality": decision.evidence_quality,
                    "search_profile": active_config.profile_name,
                    "lookback_hours": active_config.lookback_hours,
                    "model": self.model_deep,
                    "chunks": chunk_count,
                    "prompt_tokens": usage_metrics["prompt_tokens"],
                    "completion_tokens": usage_metrics["completion_tokens"],
                    "reasoning_tokens": usage_metrics["reasoning_tokens"],
                    "cached_tokens": usage_metrics["cached_tokens"],
                    "duration_ms": round(total_duration, 2),
                    "previous_analysis": previous_summary,
                },
            )
            return decision

        except Exception as exc:
            duration = (time.monotonic() - start_time) * 1000
            if content:
                logger.debug(
                    "Model response preview for failed deep analysis: market=%s preview=%s",
                    market.id,
                    _response_preview(content),
                    data={
                        "market_id": market.id,
                        "response_preview": _response_preview(content),
                    },
                )
            logger.error(
                "Deep market analysis failed: id=%s, error=%s, duration=%.2fms",
                market.id,
                exc,
                duration,
                data={
                    "market_id": market.id,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "duration_ms": round(duration, 2),
                    "previous_analysis": previous_summary,
                    "search_profile": active_config.profile_name,
                },
            )
            raise
