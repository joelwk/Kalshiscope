from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from prompts.loader import load_schema


_TRADE_DECISION_DESCRIPTIONS = load_schema("schema/trade_decision")


class MarketOutcome(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    odds: Optional[float] = None
    price: Optional[float] = None


class Market(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    ticker: str | None = None
    question: str
    subtitle: str | None = None
    resolution_criteria: str | None = None
    outcomes: list[MarketOutcome] = Field(default_factory=list)
    liquidity_usdc: Optional[float] = None
    category: Optional[str] = None
    event_ticker: str | None = None
    series_ticker: str | None = None
    market_type: str | None = None
    yes_price: float | None = None
    no_price: float | None = None
    volume: float | None = None
    volume_24h: float | None = None
    open_interest: float | None = None
    close_time: Optional[datetime] = None
    url: Optional[str] = None
    status: int | str | None = None
    winning_option_raw: str | int | None = None

    @model_validator(mode="after")
    def _normalize_identifiers(self) -> "Market":
        if not self.id and self.ticker:
            self.id = self.ticker
        if not self.ticker and self.id:
            self.ticker = self.id
        return self


class TradeDecision(BaseModel):
    should_trade: bool = Field(
        description=_TRADE_DECISION_DESCRIPTIONS["should_trade"]
    )
    outcome: str = Field(
        description=_TRADE_DECISION_DESCRIPTIONS["outcome"]
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=_TRADE_DECISION_DESCRIPTIONS["confidence"],
    )
    bet_size_pct: float = Field(
        ge=0.0,
        le=1.0,
        description=_TRADE_DECISION_DESCRIPTIONS["bet_size_pct"],
    )
    reasoning: str = Field(
        description=_TRADE_DECISION_DESCRIPTIONS["reasoning"],
    )
    implied_prob_external: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=_TRADE_DECISION_DESCRIPTIONS["implied_prob_external"],
    )
    my_prob: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=_TRADE_DECISION_DESCRIPTIONS["my_prob"],
    )
    edge_external: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description=_TRADE_DECISION_DESCRIPTIONS["edge_external"],
    )
    edge_source: str | None = Field(
        default=None,
        description=_TRADE_DECISION_DESCRIPTIONS["edge_source"],
    )
    evidence_basis: str | None = Field(
        default=None,
        description=_TRADE_DECISION_DESCRIPTIONS["evidence_basis"],
    )
    likelihood_ratio: float | None = Field(
        default=None,
        gt=0.0,
        description=_TRADE_DECISION_DESCRIPTIONS["likelihood_ratio"],
    )
    evidence_quality: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=_TRADE_DECISION_DESCRIPTIONS["evidence_quality"],
    )
    abstain: bool = Field(
        default=False,
        description=_TRADE_DECISION_DESCRIPTIONS["abstain"],
    )
    raw_should_trade: bool | None = Field(
        default=None,
        description=_TRADE_DECISION_DESCRIPTIONS["raw_should_trade"],
    )
    raw_outcome: str | None = Field(
        default=None,
        description=_TRADE_DECISION_DESCRIPTIONS["raw_outcome"],
    )
    raw_confidence: float | None = Field(
        default=None,
        description=_TRADE_DECISION_DESCRIPTIONS["raw_confidence"],
    )
    raw_bet_size_pct: float | None = Field(
        default=None,
        description=_TRADE_DECISION_DESCRIPTIONS["raw_bet_size_pct"],
    )
    raw_reasoning: str | None = Field(
        default=None,
        description=_TRADE_DECISION_DESCRIPTIONS["raw_reasoning"],
    )
    raw_evidence_quality: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=_TRADE_DECISION_DESCRIPTIONS["raw_evidence_quality"],
    )
    definitive_outcome_detected: bool | None = Field(
        default=None,
        description=_TRADE_DECISION_DESCRIPTIONS["definitive_outcome_detected"],
    )
    evidence_quality_floor_applied: str | None = Field(
        default=None,
        description=_TRADE_DECISION_DESCRIPTIONS["evidence_quality_floor_applied"],
    )
    prompt_tokens: int | None = Field(
        default=None,
        ge=0,
        description=_TRADE_DECISION_DESCRIPTIONS["prompt_tokens"],
    )
    completion_tokens: int | None = Field(
        default=None,
        ge=0,
        description=_TRADE_DECISION_DESCRIPTIONS["completion_tokens"],
    )
    reasoning_tokens: int | None = Field(
        default=None,
        ge=0,
        description=_TRADE_DECISION_DESCRIPTIONS["reasoning_tokens"],
    )
    cached_tokens: int | None = Field(
        default=None,
        ge=0,
        description=_TRADE_DECISION_DESCRIPTIONS["cached_tokens"],
    )


class OrderRequest(BaseModel):
    market_id: str | None = None
    ticker: str | None = None
    outcome: str
    amount_usdc: float
    action: str = "buy"
    order_type: str = "limit"
    time_in_force: str | None = None
    count: int | None = None
    yes_price: int | None = None
    side: str = "BUY"
    confidence: float | None = None

    @model_validator(mode="after")
    def _normalize_market_identifier(self) -> "OrderRequest":
        if not self.market_id and self.ticker:
            self.market_id = self.ticker
        if not self.ticker and self.market_id:
            self.ticker = self.market_id
        if not self.market_id:
            raise ValueError("OrderRequest requires market_id or ticker")
        return self


class OrderResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    status: Optional[str] = None
    side: Optional[str] = None
    action: Optional[str] = None
    count: Optional[int] = None
    yes_price: Optional[int] = None
    raw: dict[str, Any] = Field(default_factory=dict)


class PredictionResult(BaseModel):
    market: Market
    decision: TradeDecision
    order: Optional[OrderResponse] = None


class MarketState(BaseModel):
    market_id: str
    last_analysis: datetime | None = None
    analysis_count: int = 0
    last_confidence: float | None = None
    confidence_trend: list[float] = Field(default_factory=list)
    last_terminal_outcome: str | None = None
    non_actionable_streak: int = 0
    fill_failure_count: int = 0


class Position(BaseModel):
    market_id: str
    outcome: str
    total_amount_usdc: float
    avg_confidence: float
    trade_count: int
    first_trade: datetime
    last_trade: datetime


class InsufficientBalanceError(Exception):
    """Raised when account balance is insufficient for an order."""

    def __init__(self, message: str, available: float | None = None):
        super().__init__(message)
        self.available = available


class MarketClosedError(Exception):
    """Raised when submitting an order for a market that has already closed."""
