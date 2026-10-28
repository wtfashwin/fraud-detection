"""Validate a raw-input MLflow pipeline against an explicitly supplied holdout CSV."""
import argparse
import os
from pathlib import Path
import sys


def validate_auc(model_uri, threshold=0.95, data_path=None):
    data_path = data_path or os.getenv("VALIDATION_DATA_CSV")
    if not data_path:
        raise ValueError("Supply an independently held-out labelled CSV using --data or VALIDATION_DATA_CSV")
    if not 0 <= threshold <= 1:
        raise ValueError("AUC threshold must be between 0 and 1")
    import mlflow
    import mlflow.sklearn
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.csv_snapshot import read_checked_csv
    from training.evaluate import evaluate_predictions
    from training.preprocess import validate_dataset

    X, y = validate_dataset(read_checked_csv(data_path))
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    model = mlflow.sklearn.load_model(model_uri)
    if not hasattr(model, "named_steps") or "scaler" not in model.named_steps:
        raise ValueError("Validation requires a logged raw-input pipeline containing its fitted scaler")
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Model must provide fraud class 1")
    probabilities = model.predict_proba(X)[:, classes.index(1)]
    metrics = evaluate_predictions(y, probabilities)
    auc = metrics["roc_auc"]
    passed = auc >= threshold
    if os.getenv("MLFLOW_TRACKING_URI"):
        with mlflow.start_run():
            mlflow.log_metrics({"auc_score": auc, "average_precision": metrics["average_precision"],
                                "fraud_prevalence": metrics["fraud_prevalence"]})
            mlflow.set_tags({"validation_pass": str(passed), "validation_data": str(data_path),
                             "evaluation_scope": "provided holdout; independence not verified"})
    print(f"Provided holdout ROC AUC={auc:.6f}, average precision={metrics['average_precision']:.6f}, threshold={threshold}")
    return [auc, passed]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_uri")
    parser.add_argument("threshold", type=float, nargs="?", default=0.95)
    parser.add_argument("--data", help="Independent held-out CSV with canonical features and Class")
    args = parser.parse_args()
    try:
        _, passed = validate_auc(args.model_uri, args.threshold, args.data)
    except Exception as exc:
        print(f"Validation failed without a score: {exc}", file=sys.stderr)
        return 2
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
