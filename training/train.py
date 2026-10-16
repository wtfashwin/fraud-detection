"""Train isolated fraud bundles with untouched holdouts and honest provenance."""
import hashlib
import io
from importlib.metadata import version
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline as InferencePipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, __version__ as xgboost_version, config_context

from training import PROJECT_ROOT
from training.preprocess import FEATURE_NAMES, make_smote, validate_dataset
from training.provenance import save_training_rows, training_row_manifest


def fit_training_fold(features, labels, random_state=42, n_estimators=100,
                      model_params=None, feature_engineering=False):
    steps = []
    if feature_engineering:
        from training.features import FraudFeatureEngineer
        steps.append(("features", FraudFeatureEngineer(random_state=random_state)))
    parameters = dict(objective="binary:logistic", n_estimators=n_estimators,
                      learning_rate=0.1, max_depth=5, eval_metric="logloss",
                      random_state=random_state, n_jobs=1, scale_pos_weight=1, verbosity=0)
    parameters.update(model_params or {})
    # SMOTE targets 1:2, rather than compounding full resampling with original imbalance weights.
    steps.extend([("scaler", StandardScaler()), ("smote", make_smote(labels, random_state)),
                  ("model", XGBClassifier(**parameters))])
    with config_context(verbosity=0):
        return Pipeline(steps).fit(features, labels)


def cross_validate_training(features, labels, n_splits=5, random_state=42,
                            n_estimators=100, threshold=0.5, model_params=None,
                            feature_engineering=False):
    from training.evaluate import evaluate_predictions
    if n_splits < 2 or labels.value_counts().min() < n_splits:
        raise ValueError("Each training class needs at least cv_folds rows; supply more labelled data or fewer folds")
    fold_metrics = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, valid_idx in cv.split(features, labels):
        fold = fit_training_fold(features.iloc[train_idx], labels.iloc[train_idx],
                                 random_state, n_estimators, model_params, feature_engineering)
        scores = fold.predict_proba(features.iloc[valid_idx])[:, 1]
        fold_metrics.append(evaluate_predictions(labels.iloc[valid_idx], scores, threshold))
    return fold_metrics


def fit_and_evaluate_partition(X_train, X_test, y_train, y_test, n_splits=5,
                               random_state=42, n_estimators=100, threshold=0.5,
                               model_params=None, feature_engineering=False,
                               source_info=None, cv_metrics=None):
    """Fit only the supplied training partition; evaluate holdout after selection."""
    from api.drift import create_baseline
    from training.evaluate import evaluate_predictions
    source_info = {"data_kind": "unspecified", **(source_info or {})}
    baseline = create_baseline(X_train, FEATURE_NAMES, source={**source_info, "split": "training_only"})
    if cv_metrics is None:
        cv_metrics = cross_validate_training(X_train, y_train, n_splits, random_state,
                                              n_estimators, threshold, model_params, feature_engineering)
    pipeline = fit_training_fold(X_train, y_train, random_state, n_estimators,
                                 model_params, feature_engineering)
    metrics = evaluate_predictions(y_test, pipeline.predict_proba(X_test)[:, 1], threshold)
    metrics.update({
        **source_info, "train_rows": len(y_train), "test_rows": len(y_test), "random_state": random_state,
        "cv_folds": n_splits, "cv_roc_auc_mean": float(np.mean([m["roc_auc"] for m in cv_metrics])),
        "cv_roc_auc_std": float(np.std([m["roc_auc"] for m in cv_metrics])),
        "cv_average_precision_mean": float(np.mean([m["average_precision"] for m in cv_metrics])),
        "cv_fold_metrics": cv_metrics, "evaluation_scope": "random stratified holdout",
        "probability_calibration": "not validated", "feature_names": FEATURE_NAMES.copy(),
        "feature_engineering": bool(feature_engineering),
        "model_feature_names": list(pipeline.named_steps["scaler"].feature_names_in_),
        "sampling_strategy": 0.5, "smote_applied": pipeline.named_steps["smote"] != "passthrough",
        "model_parameters": {key: pipeline.named_steps["model"].get_params()[key] for key in
                             ("n_estimators", "learning_rate", "max_depth", "n_jobs", "scale_pos_weight")},
        "package_versions": {name: version(name) for name in
                             ("numpy", "pandas", "scikit-learn", "imbalanced-learn")}
                            | {"xgboost": xgboost_version},
    })
    return {"pipeline": pipeline, "metrics": metrics, "features": FEATURE_NAMES.copy(),
            "input_example": X_test.iloc[:5].copy(), "drift_baseline": baseline,
            "training_indices": X_train.index.tolist(), "training_rows": training_row_manifest(X_train)}


def train_and_evaluate(frame, n_splits=5, random_state=42, n_estimators=100, threshold=0.5,
                       model_params=None, feature_engineering=False, source_info=None):
    features, labels = validate_dataset(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=random_state, stratify=labels)
    return fit_and_evaluate_partition(X_train, X_test, y_train, y_test, n_splits, random_state,
                                      n_estimators, threshold, model_params, feature_engineering, source_info)


def inference_pipeline(pipeline):
    """Serving needs fitted transformations and classifier, never a training sampler."""
    return InferencePipeline([(name, step) for name, step in pipeline.steps if name != "smote"])


def save_bundle(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pipeline = result["pipeline"]
    joblib.dump(inference_pipeline(pipeline), directory / "pipeline.joblib")
    joblib.dump(pipeline.named_steps["model"], directory / "xgb_model.joblib")
    joblib.dump(pipeline.named_steps["scaler"], directory / "scaler.joblib")
    joblib.dump(result["features"], directory / "columns.joblib")
    (directory / "feature_names.json").write_text(json.dumps(result["features"]) + "\n")
    result["metrics"].update(save_training_rows(directory, result["training_rows"]))
    (directory / "metrics.json").write_text(json.dumps(result["metrics"], indent=2, allow_nan=False) + "\n")
    (directory / "drift_baseline.json").write_text(json.dumps(result["drift_baseline"], indent=2, allow_nan=False) + "\n")


def make_mlflow_callback(tracking_uri):
    from optuna.integration.mlflow import MLflowCallback
    return MLflowCallback(tracking_uri=tracking_uri, metric_name="cv_roc_auc")


def log_to_mlflow(result, tracking_uri=None, experiment_name=None):
    """Log raw-input probabilities, five CSV rows, and optional scoped registration."""
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return None
    register = os.getenv("MLFLOW_REGISTER_MODEL", "0").lower() in {"1", "true"}
    mode = os.getenv("MLFLOW_REGISTRATION_MODE", "production")
    model_name = os.getenv("MLFLOW_MODEL_NAME", "fraud-detection-model")
    data_kind = result["metrics"].get("data_kind", "unspecified")
    if register:
        if mode == "smoke":
            if not model_name.startswith("ci-smoke-") or not tracking_uri.startswith("file:"):
                raise ValueError("Smoke registration requires a ci-smoke-* namespace and throwaway local file tracking")
        elif mode != "production" or data_kind != "real":
            raise ValueError("Production model registration requires explicitly identified real evaluation data")
        elif result["metrics"]["roc_auc"] <= float(os.getenv("MLFLOW_AUC_THRESHOLD", "0.95")):
            raise ValueError("Held-out ROC AUC must exceed the configured production registration threshold")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name or os.getenv("MLFLOW_EXPERIMENT", "fraud-detection"))
    example = result["input_example"]
    pipeline = inference_pipeline(result["pipeline"])
    trial = result["metrics"].get("optuna_best_trial")
    with mlflow.start_run(run_name=f"best-trial-{trial}" if trial is not None else "train") as run:
        mlflow.log_params({"model_type": "XGBoost", "sampling": "training-fold-only SMOTE 0.5",
                           **result["metrics"]["model_parameters"], "random_state": result["metrics"]["random_state"],
                           "cv_folds": result["metrics"]["cv_folds"]})
        mlflow.log_metrics({k: float(v) for k, v in result["metrics"].items() if isinstance(v, (int, float))})
        mlflow.log_dict(result["metrics"], "evaluation/metrics.json")
        mlflow.log_dict(result["drift_baseline"], "drift/drift_baseline.json")
        mlflow.log_text(example.to_csv(index=False), "inputs/sample.csv")
        mlflow.set_tags({"data_kind": data_kind, "registration_mode": mode,
                         "evaluation_scope": "random stratified holdout; not production proof"})
        info = mlflow.sklearn.log_model(
            pipeline, name="model", input_example=example,
            signature=infer_signature(example, pipeline.predict_proba(example)),
            pyfunc_predict_fn="predict_proba", code_paths=[str(PROJECT_ROOT / "training")])
        if register:
            mlflow.register_model(info.model_uri, model_name, tags={
                "data_kind": data_kind, "registration_mode": mode,
                "production_approved": "false", "validation_scope": "synthetic-smoke-only" if mode == "smoke" else "real-heldout-benchmark"})
        print(f"MLflow run {run.info.run_id}: {info.model_uri}")
        return info.model_uri


def load_training_source(source):
    source = Path(source)
    source_bytes = source.read_bytes()
    provenance = source.with_suffix(".metadata.json")
    sidecar = json.loads(provenance.read_text()) if provenance.exists() else {}
    return pd.read_csv(io.BytesIO(source_bytes), float_precision="round_trip"), {"data_kind": sidecar.get("data_kind") or sidecar.get("kind") or os.getenv("DATA_KIND", "unspecified"),
                                 "dataset_sha256": hashlib.sha256(source_bytes).hexdigest(),
                                 "dataset_path": str(source), "dataset_source": sidecar.get("source_url", sidecar.get("source", "unspecified"))}


def main():
    source = Path(os.getenv("DATA_CSV", PROJECT_ROOT / "data/creditcard.csv"))
    frame, provenance = load_training_source(source)
    result = train_and_evaluate(frame, n_splits=int(os.getenv("CV_FOLDS", "5")),
        random_state=int(os.getenv("RANDOM_STATE", "42")), n_estimators=int(os.getenv("N_ESTIMATORS", "100")),
        threshold=float(os.getenv("DECISION_THRESHOLD", "0.5")),
        feature_engineering=os.getenv("FEATURE_ENGINEERING", "0").lower() in {"1", "true"}, source_info=provenance)
    directory = Path(os.getenv("MODEL_OUTPUT_DIR", PROJECT_ROOT / "models/trained"))
    save_bundle(result, directory)
    print(json.dumps(result["metrics"], indent=2, allow_nan=False))
    print(f"Saved matched raw-input pipeline/features to {directory}. No production-quality claim.")
    log_to_mlflow(result)


if __name__ == "__main__":
    main()
