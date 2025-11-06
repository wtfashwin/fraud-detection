import asyncio
import logging
import structlog
from typing import Tuple, Optional
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path

logger = structlog.get_logger(__name__)
mlflow.set_experiment("fraud-validation")

async def load_synthetic_data(size: int = 1000) -> pd.DataFrame:
    """Async data gen with fixed seed for reproducibility."""
    import numpy as np
    np.random.seed(42)  
    features = np.random.randn(size, 10)
    labels = (features[:, 0] > 0).astype(int)  
    return pd.DataFrame(features, columns=[f"feat_{i}" for i in range(10)]), labels

async def validate_auc(model_uri: str, threshold: float = 0.95) -> Tuple[float, bool]:
    """Load model, score on synth data; log to MLflow with tags."""
    with mlflow.start_run():
        X, y = await load_synthetic_data()
        model = mlflow.sklearn.load_model(model_uri)
        preds = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, preds)
        mlflow.log_metric("auc_score", auc)
        mlflow.set_tag("validation_pass", auc >= threshold)
        logger.info("AUC validation complete", auc=auc, threshold=threshold)
        return auc, auc >= threshold

if __name__ == "__main__":
    import sys
    uri = sys.argv[1] if len(sys.argv) > 1 else "models:/fraud/prod"
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95
    auc, passed = asyncio.run(validate_auc(uri, threshold))
    sys.exit(0 if passed else 1)