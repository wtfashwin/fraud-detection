from typing import Dict, List, Any
from pydantic import BaseModel, Field


class TransactionFeatures(BaseModel):
    """Pydantic schema for the input transaction features.

    The exact features expected should match `models/feature_names.json` used at training time.
    """
    # Accept arbitrary mapping of string->number for flexibility in the portfolio piece.
    features: Dict[str, float] = Field(..., description="Feature name -> numeric value mapping")


class PredictAccepted(BaseModel):
    transaction_id: str = Field(..., description="UUID identifying the queued transaction")
    status: str = Field("PENDING", description="Initial status")


class PredictResponse(BaseModel):
    transaction_id: str
    status: str
    prediction_score: float | None = None
    detail: str | None = None
