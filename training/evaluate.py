"""Evaluate supplied fraud labels and scores; do not manufacture a test dataset."""
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)


from training import PROJECT_ROOT
from api.inference import artifact_path, load_bundle


def check_training_source(directory, dataset_sha256):
    """Reject known byte-identical source reuse; do not certify independence."""
    if (any(name in os.environ for name in ("MODEL_PATH", "COLUMNS_PATH", "SCALER_PATH"))
            or os.getenv("MLFLOW_MODEL_URI")):
        return {"training_source_check": "unavailable_artifact_overrides"}
    metadata_path = artifact_path(directory) / "metrics.json"
    if not metadata_path.exists():
        return {"training_source_check": "unavailable_missing_provenance"}
    try:
        metadata = json.loads(metadata_path.read_text())
        if not isinstance(metadata, dict):
            raise ValueError("expected a JSON object")
        if "dataset_sha256" not in metadata:
            return {"training_source_check": "unavailable_missing_provenance"}
        training_hash = metadata["dataset_sha256"]
        if (not isinstance(training_hash, str) or len(training_hash) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in training_hash)):
            raise ValueError("dataset_sha256 must be a 64-character hexadecimal SHA-256")
    except (ValueError, OSError, UnicodeError) as exc:
        raise ValueError(f"Invalid training provenance at {metadata_path}: {exc}") from exc
    training_hash = training_hash.lower()
    if training_hash == dataset_sha256:
        raise ValueError("Evaluation CSV matches the saved bundle's training-source SHA-256; supply independently held-out data")
    return {"training_source_check": "different_hash", "training_dataset_sha256": training_hash}


def evaluate_predictions(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("Labels and fraud probabilities must be equal-length one-dimensional arrays")
    if not len(labels) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Evaluation requires both binary labels 0 and 1")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Fraud probabilities must be finite values between 0 and 1")
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Decision threshold must be between 0 and 1")
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "threshold": float(threshold), "fraud_prevalence": float(np.mean(labels)),
        "fraud_rows": int(np.sum(labels == 1)), "legitimate_rows": int(np.sum(labels == 0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def main():
    from training.preprocess import validate_dataset

    source_path = os.getenv("EVAL_DATA_CSV")
    if not source_path:
        raise ValueError("Set EVAL_DATA_CSV to an independently held-out labelled CSV; independence cannot be inferred")
    source = Path(source_path)
    source_bytes = source.read_bytes()
    dataset_sha256 = hashlib.sha256(source_bytes).hexdigest()
    X_test, y_test = validate_dataset(pd.read_csv(io.BytesIO(source_bytes)))
    directory = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
    provenance = check_training_source(directory, dataset_sha256)
    bundle = load_bundle(directory)
    if list(bundle.columns) != list(X_test.columns):
        raise ValueError("Saved model feature order does not match canonical evaluation schema")
    if bundle.raw_pipeline:
        # A saved pipeline includes feature engineering and scaling exactly once.
        model_input = X_test
    else:
        scaler_input = X_test if hasattr(bundle.scaler, "feature_names_in_") else X_test.to_numpy()
        model_input = bundle.scaler.transform(scaler_input)
        if hasattr(bundle.model, "feature_names_in_"):
            model_input = pd.DataFrame(model_input, columns=bundle.columns, index=X_test.index)
    classes = list(bundle.model.classes_)
    if 1 not in classes:
        raise ValueError("Model must provide fraud class 1")
    all_probabilities = np.asarray(bundle.model.predict_proba(model_input), dtype=float)
    if all_probabilities.shape != (len(X_test), len(classes)) or not np.isfinite(all_probabilities).all():
        raise ValueError("Model returned invalid evaluation probability shape or nonfinite values")
    if np.any((all_probabilities < 0) | (all_probabilities > 1)) or not np.allclose(all_probabilities.sum(axis=1), 1.0):
        raise ValueError("Model evaluation probabilities must be normalized values between 0 and 1")
    probabilities = all_probabilities[:, classes.index(1)]
    metrics = evaluate_predictions(y_test, probabilities, float(os.getenv("DECISION_THRESHOLD", "0.5")))
    sidecar = source.with_suffix(".metadata.json")
    data_kind = json.loads(sidecar.read_text()).get("data_kind") if sidecar.exists() else None
    metrics.update({"data_kind": data_kind or os.getenv("EVAL_DATA_KIND", "unspecified"),
                    "dataset_sha256": dataset_sha256, **provenance,
                    "evaluation_scope": "provided evaluation CSV; independence not verified"})
    output = Path(os.getenv("EVALUATION_OUTPUT_DIR", PROJECT_ROOT / "plots/evaluation"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(fpr, tpr, label=f"ROC AUC={metrics['roc_auc']:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set(xlabel="False positive rate", ylabel="Recall", title="ROC curve")
    axes[1].plot(recall, precision, label=f"Average precision={metrics['average_precision']:.4f}")
    axes[1].axhline(metrics["fraud_prevalence"], color="black", linestyle="--", label="Fraud prevalence")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    for axis in axes:
        axis.legend()
    fig.suptitle(f"Fraud evaluation: {metrics['data_kind']} data; independence unverified")
    fig.tight_layout()
    fig.savefig(output / "evaluation_curves.png")
    plt.close(fig)
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Evaluation written to {output}; dataset independence and production quality remain unverified.")


if __name__ == "__main__":
    main()
