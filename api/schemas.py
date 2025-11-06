from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class TransactionFeatures(BaseModel):
    """Pydantic schema for the input transaction features.

    The exact features expected should match `models/feature_names.json` used at training time.
    """
    # Accept arbitrary mapping of string->number for flexibility in the portfolio piece.
    features: List[float] = Field(..., description="List of feature values in the correct order")


class TransactionIn(BaseModel):
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    features: List[float] = Field(..., description="List of feature values in the correct order")


class PredictAccepted(BaseModel):
    transaction_id: str = Field(..., description="UUID identifying the queued transaction")
    status: str = Field("PENDING", description="Initial status")


class PredictResponse(BaseModel):
    transaction_id: str
    status: str
    prediction_score: Optional[float] = None
    detail: Optional[str] = None


class PredictionOut(BaseModel):
    transaction_id: str
    prediction: int
    score: float
    correlation_id: str
    explanation_status: str