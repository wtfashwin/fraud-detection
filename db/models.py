import enum
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, Enum, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

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
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (UniqueConstraint('id', name='uq_transaction_id'),)

class SHAPExplanation(Base):
    __tablename__ = "shap_explanations"

    transaction_id = Column(String(255), primary_key=True)
    correlation_id = Column(String(255))
    shap_values = Column(JSONB, nullable=False)
    feature_names = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Add schema enforcement for JSONB column
    __table_args__ = (
        UniqueConstraint('transaction_id', name='uq_shap_transaction_id'),
        
        CheckConstraint(
            func.jsonb_typeof(shap_values) == 'array', 
            name='check_shap_is_array'
        ),
    )