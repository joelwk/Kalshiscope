from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, ConfigDict


class MarketOutcome(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    odds: Optional[float] = None
    price: Optional[float] = None


class Market(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    question: str
    outcomes: list[MarketOutcome] = Field(default_factory=list)
    liquidity_usdc: Optional[float] = None
    category: Optional[str] = None
    close_time: Optional[datetime] = None
    url: Optional[str] = None
    status: int | str | None = None
    winning_option_raw: str | int | None = None


class TradeDecision(BaseModel):
    should_trade: bool = Field(
        description="Only true if YOUR probability exceeds implied odds probability by 5%+ (meaningful edge)"
    )
    outcome: str = Field(
        description="The outcome you predict will win"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "YOUR probability estimate (0.0-1.0). CALIBRATION: Sports max 0.75-0.80. "
            "If odds imply 60%, only set higher if you have specific edge. "
            "Never 0.90+ for sports - even heavy favorites lose 15-25% of games."
        ),
    )
    bet_size_pct: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of max bet (0.0-1.0). Scale with edge size, not just confidence.",
    )
    reasoning: str = Field(
        description=(
            "MUST include: 1) Implied prob from odds, 2) Your prob estimate, "
            "3) Calculated edge (your prob - implied prob), 4) Why edge exists"
        ),
    )
    implied_prob_external: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional externally sourced implied probability from books/polls/markets.",
    )
    my_prob: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional explicit analyst probability estimate.",
    )
    edge_external: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Optional explicit edge from external implied probability (my_prob - implied_prob_external).",
    )
    likelihood_ratio: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Optional likelihood ratio P(evidence|predicted_outcome) / "
            "P(evidence|alternative_outcome)."
        ),
    )
    evidence_quality: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Evidence quality score (0-1), set by validation layer.",
    )


class OrderRequest(BaseModel):
    market_id: str
    outcome: str
    amount_usdc: float
    side: str = "BUY"
    confidence: float | None = None


class OnChainPayload(BaseModel):
    to: str
    data: str
    value_wei: Optional[int] = None


class OrderResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    status: Optional[str] = None
    onchain_payload: Optional[OnChainPayload] = None
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
