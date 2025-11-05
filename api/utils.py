import os
import json
import logging
from typing import List
from joblib import load

logger = logging.getLogger(__name__)


def load_model_and_features(model_path: str | None = None, features_path: str | None = None):
    model_path = model_path or os.getenv("MODEL_PATH", "./models/logistic_model.joblib")
    features_path = features_path or os.getenv("FEATURE_NAMES_PATH", "./models/feature_names.json")

    if not os.path.exists(model_path):
        logger.error("Model file not found: %s", model_path)
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = load(model_path)

    feature_names = []
    if os.path.exists(features_path):
        with open(features_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

    return model, feature_names
