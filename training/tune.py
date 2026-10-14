"""Tune on training-fold ROC AUC, then evaluate the untouched real/synthetic holdout."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import optuna
from sklearn.model_selection import train_test_split

from training import PROJECT_ROOT
from training.preprocess import validate_dataset
from training.train import (cross_validate_training, fit_and_evaluate_partition, load_training_source,
                            log_to_mlflow, make_mlflow_callback, save_bundle)


def tune_model(frame, n_trials=10, n_estimators=100, max_majority=20000, random_state=42,
               threshold=0.5, feature_engineering=False, source_info=None,
               tracking_uri=None, study_name="fraud-optuna"):
    if n_trials < 1 or max_majority < 5:
        raise ValueError("At least one trial and five majority training rows are required")
    features, labels = validate_dataset(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=random_state, stratify=labels)
    # Split first. Preserve all fraud training rows and never sample or optimize on the holdout.
    majority = y_train[y_train == 0]
    minority = y_train[y_train == 1]
    if len(majority) > max_majority:
        selected = minority.index.union(majority.sample(n=max_majority, random_state=random_state).index)
        X_train, y_train = X_train.loc[selected], y_train.loc[selected]
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or (PROJECT_ROOT / "mlruns").as_uri()
    callback = make_mlflow_callback(tracking_uri)

    def objective(trial):
        parameters = {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                      "max_depth": trial.suggest_int("max_depth", 3, 10),
                      "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 2.0)}
        folds = cross_validate_training(X_train, y_train, n_splits=5, random_state=random_state,
                                        n_estimators=n_estimators, threshold=threshold,
                                        model_params=parameters, feature_engineering=feature_engineering)
        trial.set_user_attr("fold_metrics", folds)
        return float(np.mean([fold["roc_auc"] for fold in folds]))

    study = optuna.create_study(direction="maximize", study_name=study_name,
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, callbacks=[callback], n_jobs=1)
    result = fit_and_evaluate_partition(X_train, X_test, y_train, y_test, n_splits=5,
        random_state=random_state, n_estimators=n_estimators, threshold=threshold,
        model_params=study.best_params, feature_engineering=feature_engineering,
        source_info=source_info, cv_metrics=study.best_trial.user_attrs["fold_metrics"])
    metrics = result["metrics"]
    metrics.update({"optimization_metric": "training_cv_roc_auc", "optuna_best_trial": study.best_trial.number,
                    "optuna_best_cv_roc_auc": study.best_value, "optuna_study_name": study_name,
                    "optuna_trials": [{"number": t.number, "cv_roc_auc": t.value, "parameters": t.params,
                                       "state": t.state.name} for t in study.trials],
                    "training_majority_cap": max_majority,
                    "training_selection": "all fraud training rows; bounded deterministic legitimate training rows",
                    "target_auc": 0.95,
                    "target_met": metrics.get("data_kind") in {"real", "public_benchmark"} and metrics["roc_auc"] > 0.95,
                    "quality_scope": "held-out benchmark only; production performance and calibration unverified"})
    result["mlflow_model_uri"] = log_to_mlflow(result, tracking_uri=tracking_uri, experiment_name=study_name)
    return result, study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(os.getenv("DATA_CSV", PROJECT_ROOT / "data/creditcard.csv")))
    parser.add_argument("--output", type=Path, default=Path(os.getenv("MODEL_OUTPUT_DIR", PROJECT_ROOT / "models/tuned")))
    parser.add_argument("--trials", type=int, default=int(os.getenv("OPTUNA_TRIALS", "10")))
    parser.add_argument("--max-majority", type=int, default=int(os.getenv("TUNE_MAX_MAJORITY", "20000")))
    parser.add_argument("--estimators", type=int, default=int(os.getenv("N_ESTIMATORS", "100")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("RANDOM_STATE", "42")))
    parser.add_argument("--threshold", type=float, default=float(os.getenv("DECISION_THRESHOLD", "0.5")))
    parser.add_argument("--feature-engineering", action="store_true", default=os.getenv("FEATURE_ENGINEERING", "0").lower() in {"1", "true"})
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--study-name", default=os.getenv("OPTUNA_STUDY_NAME", "fraud-optuna"))
    parser.add_argument("--require-target", action="store_true", help="Exit nonzero unless a real holdout ROC AUC exceeds 0.95")
    args = parser.parse_args()
    frame, provenance = load_training_source(args.data)
    result, _ = tune_model(frame, args.trials, args.estimators, args.max_majority, args.seed,
                           args.threshold, args.feature_engineering, provenance, args.tracking_uri, args.study_name)
    save_bundle(result, args.output)
    print(json.dumps(result["metrics"], indent=2, allow_nan=False))
    print(f"Saved tuned raw-input pipeline to {args.output}; target_met={result['metrics']['target_met']}")
    return 1 if args.require_target and not result["metrics"]["target_met"] else 0


if __name__ == "__main__":
    sys.exit(main())
