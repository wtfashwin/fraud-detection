"""Validate the serving schema and fit preprocessing on training rows only."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from training import PROJECT_ROOT

FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def validate_dataset(frame):
    """Return numeric canonical features and binary labels without silently dropping data."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Dataset must be a nonempty DataFrame")
    required = set(FEATURE_NAMES + ["Class"])
    if frame.columns.duplicated().any() or set(frame.columns) != required:
        missing = sorted(required - set(frame.columns))
        extra = sorted(set(frame.columns) - required)
        raise ValueError(f"Expected canonical fraud features and Class; missing={missing}, extra={extra}")
    try:
        features = frame[FEATURE_NAMES].apply(pd.to_numeric, errors="raise").astype(float)
        labels = pd.to_numeric(frame["Class"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Features and Class must be numeric") from exc
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Features must contain only finite values")
    if not np.isfinite(labels.to_numpy()).all() or set(labels.unique()) != {0, 1}:
        raise ValueError("Class must contain both binary labels 0 and 1")
    return features, labels.astype(int)


def make_smote(labels, random_state=42):
    minority_count = int(pd.Series(labels).value_counts().min())
    if minority_count < 2:
        raise ValueError("SMOTE requires at least two minority training rows in every fold")
    return SMOTE(random_state=random_state, k_neighbors=min(5, minority_count - 1))


def prepare_data(frame, test_size=0.2, random_state=42):
    features, labels = validate_dataset(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels,
    )
    scaler = StandardScaler().fit(X_train)
    X_res, y_res = make_smote(y_train, random_state).fit_resample(scaler.transform(X_train), y_train)
    return {"X_res": X_res, "y_res": np.asarray(y_res),
            "X_test": scaler.transform(X_test), "y_test": y_test.to_numpy(),
            "scaler": scaler, "features": FEATURE_NAMES.copy()}


def main():
    source = Path(os.getenv("DATA_CSV", PROJECT_ROOT / "data/creditcard.csv"))
    output = Path(os.getenv("PREPROCESSED_DATA_PATH", PROJECT_ROOT / "data/preprocessed_data.npz"))
    model_dir = Path(os.getenv("PREPROCESS_OUTPUT_DIR", PROJECT_ROOT / "models/preprocessed"))
    result = prepare_data(pd.read_csv(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{k: result[k] for k in ("X_res", "y_res", "X_test", "y_test")})
    joblib.dump(result["scaler"], model_dir / "scaler.joblib")
    joblib.dump(result["features"], model_dir / "columns.joblib")
    (model_dir / "feature_names.json").write_text(json.dumps(result["features"]) + "\n")
    print(f"Prepared training-only scaling and SMOTE; holdout saved to {output}, preprocessing to {model_dir}")
    print("Use python -m training.train for a complete matched model/scaler bundle.")


if __name__ == "__main__":
    main()
