from __future__ import annotations

from datetime import datetime, timedelta, timezone

from logging_config import get_logger
from grok_client import GrokClient
from config import SearchConfig
from models import Market, MarketState, TradeDecision

logger = get_logger(__name__)

REFINEMENT_CONFIDENCE_MIN = 0.55
REFINEMENT_CONFIDENCE_MAX = 0.80
HIGH_CONFIDENCE_THRESHOLD = 0.70
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
    ) -> None:
        self.market = market
        self.urgent_days_before_close = urgent_days_before_close
        self.high_confidence_threshold = high_confidence_threshold

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
    ) -> TradeDecision:
        """Execute multi-pass refinement with flip-flop protection."""
        decision = initial
        initial_outcome = initial.outcome
        
        for pass_index in range(1, MAX_REFINEMENT_PASSES + 1):
            logger.info(
                "Refinement pass %d/%d: market=%s confidence=%.2f outcome=%s",
                pass_index,
                MAX_REFINEMENT_PASSES,
                market.id,
                decision.confidence,
                decision.outcome,
            )
            new_decision = grok.analyze_market_deep(
                market,
                previous_analysis=decision,
                search_config=search_config,
            )
            
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
                # Only accept flip if confidence is HIGH (strong conviction)
                if new_decision.confidence < HIGH_CONFIDENCE_THRESHOLD:
                    logger.info(
                        "Rejecting flip: confidence %.2f < %.2f threshold, reverting to initial",
                        new_decision.confidence,
                        HIGH_CONFIDENCE_THRESHOLD,
                        data={"market_id": market.id},
                    )
                    # Reduce confidence of initial decision due to uncertainty
                    return TradeDecision(
                        should_trade=initial.should_trade and initial.confidence >= 0.60,
                        outcome=initial_outcome,
                        confidence=max(initial.confidence - 0.05, 0.50),
                        bet_size_pct=initial.bet_size_pct * 0.8,
                        implied_prob_external=initial.implied_prob_external,
                        my_prob=initial.my_prob,
                        edge_external=initial.edge_external,
                        evidence_quality=max(0.0, min(initial.evidence_quality, 0.5)),
                        reasoning=f"Refinement showed uncertainty (flip to {new_decision.outcome} rejected). {initial.reasoning}",
                    )
                if (
                    current_edge is not None
                    and new_edge is not None
                    and new_edge < (current_edge + 0.03)
                    and new_decision.confidence < (decision.confidence + 0.05)
                ):
                    logger.info(
                        "Rejecting flip: no materially better edge/confidence (current_edge=%.3f new_edge=%.3f)",
                        current_edge,
                        new_edge,
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
