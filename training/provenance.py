"""Exact fitted-input reuse checks; these do not establish data independence."""
import hashlib
import json
import os

import numpy as np

from api.inference import artifact_path
from training.preprocess import FEATURE_NAMES

ROW_ALGORITHM = "sha256-float64-be-v1"


def artifact_overrides():
    return (any(name in os.environ for name in ("MODEL_PATH", "COLUMNS_PATH", "SCALER_PATH"))
            or bool(os.getenv("MLFLOW_MODEL_URI")))


def row_fingerprints(features):
    """Hash canonical numeric features, excluding labels and normalizing signed zero."""
    if list(features.columns) != FEATURE_NAMES or features.empty:
        raise ValueError("Row provenance requires nonempty canonical fraud features")
    values = np.array(features, dtype=">f8", copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Row provenance requires finite feature values")
    values[values == 0] = 0.0
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in values]


def training_row_manifest(features):
    return {"version": 1, "algorithm": ROW_ALGORITHM, "feature_names": FEATURE_NAMES.copy(),
            "row_count": len(features), "fingerprints": sorted(set(row_fingerprints(features)))}


def save_training_rows(directory, manifest):
    """Bind the generated row manifest and saved serving pipeline in compact metadata."""
    contents = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    (directory / "training_rows.json").write_bytes(contents)
    return {"training_rows_sha256": hashlib.sha256(contents).hexdigest(),
            "training_pipeline_sha256": hashlib.sha256((directory / "pipeline.joblib").read_bytes()).hexdigest()}


def _sha256(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in value)):
        raise ValueError("expected a 64-character hexadecimal SHA-256")
    return value.lower()


def check_training_rows(directory, features):
    """Reject exact feature-row overlap before loading or scoring a local bundle."""
    if artifact_overrides():
        return {"training_row_check": "unavailable_artifact_overrides"}
    directory = artifact_path(directory)
    metadata_path = directory / "metrics.json"
    if not metadata_path.exists():
        return {"training_row_check": "unavailable_missing_provenance"}
    try:
        metadata = json.loads(metadata_path.read_text())
        if not isinstance(metadata, dict):
            raise ValueError("expected a JSON object in metrics.json")
        declarations = ("training_rows_sha256", "training_pipeline_sha256")
        if not any(name in metadata for name in declarations):
            return {"training_row_check": "unavailable_missing_provenance"}
        manifest_hash = _sha256(metadata.get("training_rows_sha256"))
        pipeline_hash = _sha256(metadata.get("training_pipeline_sha256"))
        # New manifests describe pipeline.joblib, which load_bundle selects before legacy files.
        pipeline = directory / "pipeline.joblib"
        if not pipeline.is_file() or hashlib.sha256(pipeline.read_bytes()).hexdigest() != pipeline_hash:
            raise ValueError("saved pipeline is missing or its SHA-256 does not match")
        contents = (directory / "training_rows.json").read_bytes()
        if hashlib.sha256(contents).hexdigest() != manifest_hash:
            raise ValueError("training_rows.json SHA-256 does not match metrics.json")
        manifest = json.loads(contents)
        if (not isinstance(manifest, dict) or type(manifest.get("version")) is not int
                or manifest["version"] != 1 or manifest.get("algorithm") != ROW_ALGORITHM
                or manifest.get("feature_names") != FEATURE_NAMES):
            raise ValueError("unsupported row manifest version, algorithm or feature schema")
        row_count, fingerprints = manifest.get("row_count"), manifest.get("fingerprints")
        if (type(row_count) is not int or row_count < 1 or row_count != metadata.get("train_rows")
                or not isinstance(fingerprints, list) or not 0 < len(fingerprints) <= row_count):
            raise ValueError("invalid fitted row count or fingerprints")
        normalized = [_sha256(value) for value in fingerprints]
        if normalized != sorted(set(normalized)):
            raise ValueError("row fingerprints must be sorted and unique")
    except (ValueError, OSError, UnicodeError) as exc:
        raise ValueError(f"Invalid training row provenance at {directory}: {exc}") from exc
    training_hashes = set(normalized)
    overlap = sum(value in training_hashes for value in row_fingerprints(features))
    if overlap:
        raise ValueError(f"Evaluation CSV has {overlap} feature rows that overlap the saved pipeline's training rows; supply independently held-out data")
    return {"training_row_check": "no_exact_overlap", "training_rows": row_count,
            "training_unique_rows": len(normalized)}
