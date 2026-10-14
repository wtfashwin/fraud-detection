import enum
import uuid
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column 
from sqlalchemy import Column, String, Float, DateTime, Enum, Integer, JSON, func
from sqlalchemy.dialects.postgresql import UUID, JSONB

class Base(DeclarativeBase):
    pass

class StatusEnum(str, enum.Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class TransactionResult(Base):
    __tablename__ = "transaction_results"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    input_data = Column(JSONB, nullable=False)
    shap_values = Column(JSONB, nullable=True)
    prediction_score = Column(Float, nullable=True)
    status = Column(Enum(StatusEnum), nullable=False, default=StatusEnum.PENDING)
    created_at = Column(DateTime, default=datetime)
    updated_at = Column(DateTime, default=datetime, onupdate=datetime)


class Explanation(Base):
    """Shared API/worker state, created by Alembic rather than application startup."""
    __tablename__ = "shap_explanations"
    transaction_id = Column(String(128), primary_key=True)
    correlation_id = Column(String(64), nullable=False)
    status = Column(String(16), nullable=False, default="PENDING")
    shap_values = Column(JSON().with_variant(JSONB(), "postgresql"), nullable=True)
    feature_names = Column(JSON().with_variant(JSONB(), "postgresql"), nullable=True)
    base_value = Column(Float, nullable=True)
    explanation_method = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
