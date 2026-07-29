from __future__ import annotations

from datetime import datetime, timedelta, timezone

from logging_config import get_logger
from grok_client import GrokClient
from config import SearchConfig, Settings
from models import Market, MarketState, TradeDecision
from research_profiles import market_family

logger = get_logger(__name__)

REFINEMENT_CONFIDENCE_MIN = 0.55
REFINEMENT_CONFIDENCE_MAX = 0.80
HIGH_CONFIDENCE_THRESHOLD = 0.70
FLIP_CONFIDENCE_FLOOR_NON_DIRECT = 0.78
FLIP_EDGE_IMPROVEMENT_THRESHOLD = 0.05
LOW_CONFIDENCE_EARLY_EXIT = 0.50
MAX_REFINEMENT_PASSES = 2
LOW_EVIDENCE_REFINE_THRESHOLD = 0.40
UNCERTAIN_EDGE_BAND = 0.02
CLEAR_NEGATIVE_EDGE_THRESHOLD = -0.03
MATERIAL_CONFIDENCE_DELTA = 0.02
MATERIAL_EDGE_DELTA = 0.02


class RefinementStrategy:
    """Strategy for deciding when and how to refine market analysis."""

    def __init__(
        self,
        market: Market | None = None,
        urgent_days_before_close: int = 2,
        high_confidence_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
        skip_borderline_families: tuple[str, ...] = (),
    ) -> None:
        self.market = market
        self.urgent_days_before_close = urgent_days_before_close
        self.high_confidence_threshold = high_confidence_threshold
        # Normalize to lowercase for case-insensitive comparison; settings
        # accept user-provided values like "Sports" or "SPORTS".
        self.skip_borderline_families = tuple(
            family.strip().lower()
            for family in skip_borderline_families
            if family and family.strip()
        )

    def should_refine(
        self,
        decision: TradeDecision,
        state: MarketState | None,
        implied_prob: float | None = None,
        evidence_quality: float | None = None,
        edge_value: float | None = None,
    ) -> bool:
        """Determine if market needs deeper analysis."""
        return bool(
            self.get_refinement_reasons(
                decision,
                state,
                implied_prob=implied_prob,
                evidence_quality=evidence_quality,
                edge_value=edge_value,
            )
        )

    def get_refinement_reasons(
        self,
        decision: TradeDecision,
        state: MarketState | None,
        implied_prob: float | None = None,
        evidence_quality: float | None = None,
        edge_value: float | None = None,
        *,
        settings: Settings | None = None,
        pre_execution_score: float | None = None,
        score_threshold: float | None = None,
    ) -> list[str]:
        reasons: list[str] = []
        quality = decision.evidence_quality if evidence_quality is None else evidence_quality
        edge = decision.edge_external if edge_value is None else edge_value

        # Avoid expensive re-checks when the market edge is clearly negative and stable.
        if (
            not decision.should_trade
            and edge is not None
            and edge <= CLEAR_NEGATIVE_EDGE_THRESHOLD
            and quality >= LOW_EVIDENCE_REFINE_THRESHOLD
        ):
            return reasons

        if decision.should_trade and 0.60 <= decision.confidence <= 0.78:
            # Skip the borderline-trade-confidence trigger for families that
            # move too fast for refinement to add value (e.g. sports player
            # props, F5, RFI). The 1.5-2 minute deep refinement window
            # erodes edge through market movement faster than the deep pass
            # can add new information; previous live runs showed
            # KXMLBF5-...-TORTB-TB initial trade=True (conf=0.62, edge>=0.08)
            # downgraded to trade=False (conf=0.50, edge=0.04) by deep
            # refinement after 2 minutes of market movement. Other refine
            # triggers below still fire for these families.
            if not self._family_skipped_for_borderline_trade():
                reasons.append("borderline_trade_confidence")
        if implied_prob is None and (decision.should_trade or quality < LOW_EVIDENCE_REFINE_THRESHOLD):
            reasons.append("missing_implied_probability")
        if quality < LOW_EVIDENCE_REFINE_THRESHOLD and (
            decision.should_trade
            or edge is None
            or abs(edge) <= UNCERTAIN_EDGE_BAND
        ):
            reasons.append("low_evidence_quality")
        if decision.should_trade and decision.confidence >= 0.78 and edge is not None and edge < 0.08:
            reasons.append("high_conf_small_edge")
        if (
            settings is not None
            and settings.BORDERLINE_CRITIQUE_REFINEMENT_ENABLED
            and pre_execution_score is not None
            and score_threshold is not None
            and bool(str(getattr(decision, "primary_source_url", "") or "").strip())
            and not self._family_skipped_for_borderline_trade()
        ):
            score_band = max(0.0, float(settings.BORDERLINE_CRITIQUE_REFINEMENT_SCORE_BAND))
            lower_bound = float(score_threshold) - score_band
            if lower_bound <= float(pre_execution_score) < float(score_threshold):
                reasons.append("borderline_pre_execution_score")

        if reasons:
            return reasons

        borderline = (
            REFINEMENT_CONFIDENCE_MIN
            <= decision.confidence
            <= REFINEMENT_CONFIDENCE_MAX
        )
        urgent_close = self._is_urgent_close()
        previous_high_confidence = False
        if state and state.last_confidence is not None:
            previous_high_confidence = (
                state.last_confidence >= self.high_confidence_threshold
            )
        if borderline and (urgent_close or previous_high_confidence) and (
            decision.should_trade
            or edge is None
            or abs(edge) <= UNCERTAIN_EDGE_BAND
        ):
            reasons.append("legacy_borderline_urgent")
        logger.debug(
            "Refinement check: reasons=%s urgent_close=%s previous_high=%s",
            reasons,
            urgent_close,
            previous_high_confidence,
            data={
                "market_id": self.market.id if self.market else None,
                "confidence": decision.confidence,
                "last_confidence": state.last_confidence if state else None,
                "urgent_days_before_close": self.urgent_days_before_close,
                "high_confidence_threshold": self.high_confidence_threshold,
                "implied_prob": implied_prob,
                "evidence_quality": quality,
                "edge_value": edge,
            },
        )
        return reasons

    def perform_refinement(
        self,
        grok: GrokClient,
        market: Market,
        initial: TradeDecision,
        search_config: SearchConfig | None = None,
        refinement_reasons: list[str] | None = None,
        *,
        family_is_profitable: bool = False,
    ) -> TradeDecision:
        """Execute multi-pass refinement with flip-flop protection."""
        decision = initial
        initial_outcome = initial.outcome
        if refinement_reasons and "borderline_pre_execution_score" in refinement_reasons:
            decision = decision.model_copy(
                update={
                    "reasoning": (
                        "[BorderlineScoreCritique] Prior score was just below the "
                        "execution threshold. Re-check whether conservative "
                        "probability or evidence assumptions hide a real edge. "
                        f"{decision.reasoning}"
                    )
                }
            )
        
        for pass_index in range(1, MAX_REFINEMENT_PASSES + 1):
            logger.info(
                "Refinement pass %d/%d: market=%s confidence=%.2f outcome=%s",
                pass_index,
                MAX_REFINEMENT_PASSES,
                market.id,
                decision.confidence,
                decision.outcome,
            )
            try:
                new_decision = grok.analyze_market_deep(
                    market,
                    previous_analysis=decision,
                    search_config=search_config,
                    family_is_profitable=family_is_profitable,
                )
            except Exception as deep_exc:
                logger.warning(
                    "Deep refinement failed; falling back to prior decision: "
                    "market=%s pass=%d error=%s",
                    market.id,
                    pass_index,
                    deep_exc,
                    data={
                        "market_id": market.id,
                        "refinement_pass": pass_index,
                        "error": str(deep_exc),
                        "error_type": type(deep_exc).__name__,
                        "fallback_outcome": decision.outcome,
                        "fallback_confidence": decision.confidence,
                    },
                )
                return decision
            
            # FLIP-FLOP PROTECTION: If outcome changed, require higher confidence
            if new_decision.outcome != initial_outcome:
                current_edge = decision.edge_external
                new_edge = new_decision.edge_external
                logger.warning(
                    "Refinement flipped outcome: market=%s, initial=%s, new=%s, new_conf=%.2f",
                    market.id,
                    initial_outcome,
                    new_decision.outcome,
                    new_decision.confidence,
                    data={
                        "market_id": market.id,
                        "initial_outcome": initial_outcome,
                        "new_outcome": new_decision.outcome,
                        "new_confidence": new_decision.confidence,
                        "current_edge": current_edge,
                        "new_edge": new_edge,
                    },
                )
                new_evidence_basis = str(
                    getattr(new_decision, "evidence_basis", "") or ""
                ).strip().lower()
                new_has_primary_url = bool(
                    str(getattr(new_decision, "primary_source_url", "") or "").strip()
                )
                _flip_has_direct_source = (
                    new_evidence_basis == "direct" and new_has_primary_url
                )
                _flip_conf_floor = (
                    HIGH_CONFIDENCE_THRESHOLD
                    if _flip_has_direct_source
                    else FLIP_CONFIDENCE_FLOOR_NON_DIRECT
                )
                if new_decision.confidence < _flip_conf_floor:
                    logger.info(
                        "Rejecting flip: confidence %.2f < %.2f threshold "
                        "(direct_source=%s), reverting to initial",
                        new_decision.confidence,
                        _flip_conf_floor,
                        _flip_has_direct_source,
                        data={
                            "market_id": market.id,
                            "flip_conf_floor": _flip_conf_floor,
                            "flip_has_direct_source": _flip_has_direct_source,
                        },
                    )
                    # Retain the validated probability/source/audit payload from
                    # the initial pass. Reconstructing a minimal TradeDecision
                    # here silently discarded probability_yes, sources, LR,
                    # critique, raw fields, and token usage precisely when the
                    # disagreement made that context most important.
                    return initial.model_copy(
                        update={
                            "should_trade": (
                                initial.should_trade and initial.confidence >= 0.60
                            ),
                            "confidence": max(initial.confidence - 0.05, 0.50),
                            "bet_size_pct": initial.bet_size_pct * 0.8,
                            "evidence_quality": max(
                                0.0, min(initial.evidence_quality, 0.5)
                            ),
                            "reasoning": (
                                "Refinement showed uncertainty "
                                f"(flip to {new_decision.outcome} rejected). "
                                f"{initial.reasoning}"
                            ),
                        }
                    )
                if (
                    current_edge is not None
                    and new_edge is not None
                    and new_edge < (current_edge + FLIP_EDGE_IMPROVEMENT_THRESHOLD)
                    and new_decision.confidence < (decision.confidence + 0.05)
                ):
                    logger.info(
                        "Rejecting flip: no materially better edge/confidence "
                        "(current_edge=%.3f new_edge=%.3f required_improvement=%.3f)",
                        current_edge,
                        new_edge,
                        FLIP_EDGE_IMPROVEMENT_THRESHOLD,
                        data={"market_id": market.id},
                    )
                    return decision
            
            confidence_delta_abs = abs(new_decision.confidence - decision.confidence)
            current_edge = decision.edge_external
            new_edge = new_decision.edge_external
            edge_delta_abs = (
                abs(new_edge - current_edge)
                if current_edge is not None and new_edge is not None
                else None
            )

            if (
                pass_index == 1
                and new_decision.outcome == decision.outcome
                and confidence_delta_abs < MATERIAL_CONFIDENCE_DELTA
                and (edge_delta_abs is None or edge_delta_abs < MATERIAL_EDGE_DELTA)
            ):
                logger.debug(
                    "Refinement stopping after pass 1: no material confidence/edge change",
                    data={
                        "market_id": market.id,
                        "confidence_delta_abs": confidence_delta_abs,
                        "edge_delta_abs": edge_delta_abs,
                    },
                )
                decision = new_decision
                break

            if (
                pass_index == 1
                and not new_decision.should_trade
                and new_edge is not None
                and new_edge <= 0.0
            ):
                logger.debug(
                    "Refinement stopping after pass 1: persistent negative edge",
                    data={
                        "market_id": market.id,
                        "new_edge": new_edge,
                    },
                )
                decision = new_decision
                break

            decision = new_decision
            
            if pass_index == 1 and decision.confidence < LOW_CONFIDENCE_EARLY_EXIT:
                logger.debug(
                    "Refinement stopping early: confidence=%.2f below early-exit threshold",
                    decision.confidence,
                    data={"market_id": market.id},
                )
                break
            if (
                decision.confidence < REFINEMENT_CONFIDENCE_MIN
                or decision.confidence > REFINEMENT_CONFIDENCE_MAX
            ):
                logger.debug(
                    "Refinement stopping early: confidence=%.2f",
                    decision.confidence,
                    data={"market_id": market.id},
                )
                break
        return decision

    def _is_urgent_close(self) -> bool:
        if not self.market or not self.market.close_time:
            return False
        close_time = self.market.close_time
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        urgent_cutoff = now + timedelta(days=self.urgent_days_before_close)
        return close_time <= urgent_cutoff

    def _family_skipped_for_borderline_trade(self) -> bool:
        """Return True when the market's family is in the skip list.

        Used to suppress the borderline_trade_confidence refinement trigger
        for families where deep refinement consistently erodes edge through
        market movement (e.g. sports). When the market is unavailable we
        fall back to "do not skip" so refinement runs normally.
        """
        if not self.skip_borderline_families or self.market is None:
            return False
        family = market_family(self.market).strip().lower()
        return family in self.skip_borderline_families
