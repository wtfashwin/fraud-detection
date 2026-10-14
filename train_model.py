"""Train a complete, isolated model bundle; holdout metrics are not production proof."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from preprocess import FEATURE_NAMES, make_smote, validate_dataset


def fit_training_fold(features, labels, random_state=42, n_estimators=100):
    # Scaling and resampling are fitted separately within every training fold.
    # SMOTE already balances the classes, so a second positive-class weight is wrong.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", make_smote(labels, random_state)),
        ("model", XGBClassifier(
            objective="binary:logistic", n_estimators=n_estimators,
            learning_rate=0.1, max_depth=5, eval_metric="logloss",
            random_state=random_state, n_jobs=1, scale_pos_weight=1, verbosity=0,
        )),
    ])
    return pipeline.fit(features, labels)


def train_and_evaluate(frame, n_splits=5, random_state=42, n_estimators=100, threshold=0.5):
    from evaluate_model import evaluate_predictions

    features, labels = validate_dataset(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=random_state, stratify=labels,
    )
    if n_splits < 2 or y_train.value_counts().min() < n_splits:
        raise ValueError("Each training class needs at least cv_folds rows; supply more labelled data or fewer folds")
    fold_metrics = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, valid_idx in cv.split(X_train, y_train):
        fold = fit_training_fold(X_train.iloc[train_idx], y_train.iloc[train_idx], random_state, n_estimators)
        scores = fold.predict_proba(X_train.iloc[valid_idx])[:, 1]
        fold_metrics.append(evaluate_predictions(y_train.iloc[valid_idx], scores, threshold))
    pipeline = fit_training_fold(X_train, y_train, random_state, n_estimators)
    metrics = evaluate_predictions(y_test, pipeline.predict_proba(X_test)[:, 1], threshold)
    metrics.update({
        "train_rows": len(y_train), "test_rows": len(y_test), "random_state": random_state,
        "cv_folds": n_splits, "cv_roc_auc_mean": float(np.mean([m["roc_auc"] for m in fold_metrics])),
        "cv_roc_auc_std": float(np.std([m["roc_auc"] for m in fold_metrics])),
        "cv_average_precision_mean": float(np.mean([m["average_precision"] for m in fold_metrics])),
        "cv_fold_metrics": fold_metrics, "evaluation_scope": "random stratified holdout",
        "probability_calibration": "not validated", "feature_names": FEATURE_NAMES.copy(),
        "model_parameters": {key: pipeline.named_steps["model"].get_params()[key] for key in
                             ("n_estimators", "learning_rate", "max_depth", "n_jobs", "scale_pos_weight")},
        "package_versions": {name: version(name) for name in
                             ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost")},
    })
    return {"pipeline": pipeline, "metrics": metrics, "features": FEATURE_NAMES.copy(),
            "input_example": X_test.iloc[:5].copy()}


def save_bundle(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pipeline = result["pipeline"]
    joblib.dump(pipeline.named_steps["model"], directory / "xgb_model.joblib")
    joblib.dump(pipeline.named_steps["scaler"], directory / "scaler.joblib")
    joblib.dump(result["features"], directory / "columns.joblib")
    (directory / "feature_names.json").write_text(json.dumps(result["features"]) + "\n")
    (directory / "metrics.json").write_text(json.dumps(result["metrics"], indent=2, allow_nan=False) + "\n")


def log_to_mlflow(result):
    """Log raw-input pipeline with a signature that matches its Nx2 probabilities."""
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return None
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "fraud-detection"))
    example = result["input_example"]
    pipeline = result["pipeline"]
    with mlflow.start_run() as run:
        mlflow.log_params({"model_type": "XGBoost", "sampling": "training-fold-only SMOTE",
                           "scale_pos_weight": 1, "random_state": result["metrics"]["random_state"],
                           "cv_folds": result["metrics"]["cv_folds"]})
        mlflow.log_metrics({k: float(v) for k, v in result["metrics"].items()
                            if isinstance(v, (int, float))})
        mlflow.log_dict(result["metrics"], "evaluation/metrics.json")
        mlflow.set_tags({"data_kind": result["metrics"].get("data_kind", "unspecified"),
                         "evaluation_scope": "random stratified holdout; not production proof"})
        info = mlflow.sklearn.log_model(
            pipeline, name="model", input_example=example,
            signature=infer_signature(example, pipeline.predict_proba(example)),
            pyfunc_predict_fn="predict_proba",
        )
        # Use this active run's URI, never last_active_run inside the open run.
        model_uri = info.model_uri
        if os.getenv("MLFLOW_REGISTER_MODEL", "0").lower() in {"1", "true"}:
            if result["metrics"].get("data_kind") != "real":
                raise ValueError("Model registration requires explicitly identified real evaluation data")
            minimum_auc = float(os.getenv("MLFLOW_AUC_THRESHOLD", "0.95"))
            if result["metrics"]["roc_auc"] < minimum_auc:
                raise ValueError("Held-out ROC AUC is below the configured registration threshold")
            mlflow.register_model(model_uri, os.getenv("MLFLOW_MODEL_NAME", "fraud-detection-model"))
        print(f"MLflow run {run.info.run_id}: {model_uri}")
        return model_uri


def main():
    source = Path(os.getenv("DATA_CSV", "data/creditcard.csv"))
    directory = Path(os.getenv("MODEL_OUTPUT_DIR", "models/trained"))
    result = train_and_evaluate(pd.read_csv(source),
        n_splits=int(os.getenv("CV_FOLDS", "5")),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        n_estimators=int(os.getenv("N_ESTIMATORS", "100")),
        threshold=float(os.getenv("DECISION_THRESHOLD", "0.5")),
    )
    provenance = source.with_suffix(".metadata.json")
    source_kind = json.loads(provenance.read_text()).get("data_kind") if provenance.exists() else None
    result["metrics"].update({"data_kind": source_kind or os.getenv("DATA_KIND", "unspecified"),
                              "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                              "dataset_path": str(source)})
    save_bundle(result, directory)
    print(json.dumps(result["metrics"], indent=2, allow_nan=False))
    print(f"Saved matched model/scaler/features to {directory}. No production-quality claim.")
    # Explicitly configured tracking failures fail the command; they are not fabricated scores.
    log_to_mlflow(result)


if __name__ == "__main__":
    main()
