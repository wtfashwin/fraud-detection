import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@postgres:5432/fraud",
)

logger = logging.getLogger(__name__)

# Use SQLAlchemy engine and sessionmaker for sync access (used by API & Celery worker)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db_tables(Base):
    """Create DB tables. Call this at worker startup to ensure schema exists for the portfolio demo."""
    logger.info("Creating database tables (if not exist)")
    Base.metadata.create_all(bind=engine)
