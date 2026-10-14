"""Migration regression with a synthetic encoded URL; no database connection is made."""
import os
from pathlib import Path
import subprocess
import sys


def test_offline_migration_accepts_percent_encoded_database_url():
    environment = dict(os.environ, DATABASE_URL="postgresql+psycopg2://synthetic:synthetic%40password@localhost/synthetic")
    result = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head", "--sql"],
                            cwd=Path(__file__).resolve().parents[1], env=environment,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "CREATE TABLE shap_explanations" in result.stdout
