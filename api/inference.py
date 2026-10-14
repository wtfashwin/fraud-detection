"""One preprocessing and scoring contract for HTTP requests and explanations."""
from dataclasses import dataclass
from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def artifact_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


@dataclass
class InferenceBundle:
    model: object
    scaler: object
    columns: list[str]
    source: str = "local"
    raw_pipeline: bool = False

    def prepare(self, features):
        values = np.asarray(features, dtype=float)
        if values.shape != (len(self.columns),) or not np.isfinite(values).all():
            raise ValueError(f"Input must contain {len(self.columns)} finite numeric features")
        raw = pd.DataFrame([values], columns=self.columns)
        if self.raw_pipeline:
            return raw
        if getattr(self.scaler, "n_features_in_", len(self.columns)) != len(self.columns):
            raise ValueError("Scaler feature count does not match model feature count")
        scaler_columns = list(getattr(self.scaler, "feature_names_in_", self.columns))
        if scaler_columns != self.columns:
            raise ValueError("Scaler feature order does not match saved model features")
        input_data = raw if hasattr(self.scaler, "feature_names_in_") else raw.to_numpy()
        raw = pd.DataFrame(self.scaler.transform(input_data), columns=self.columns)
        return raw if hasattr(self.model, "feature_names_in_") else raw.to_numpy()

    def score(self, features):
        data = self.prepare(features)
        labels = np.asarray(self.model.predict(data)).reshape(-1)
        if labels.shape != (1,) or labels[0] not in (0, 1):
            raise ValueError("Model must return one binary prediction")
        probabilities = np.asarray(self.model.predict_proba(data), dtype=float)
        classes = np.asarray(getattr(self.model, "classes_", []))
        fraud_columns = np.flatnonzero(classes == 1)
        if probabilities.shape != (1, len(classes)) or len(fraud_columns) != 1:
            raise ValueError("Model probability output must include fraud class 1")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("Model returned an invalid probability")
        if not np.allclose(probabilities.sum(axis=1), 1.0):
            raise ValueError("Model probability values must sum to one")
        return int(labels[0]), float(probabilities[0, fraud_columns[0]])

    def explain(self, features):
        """Return actual attributions; unsupported estimators fail rather than invent them."""
        data = self.prepare(features)
        model = self.model
        if self.raw_pipeline:
            if not hasattr(model, "steps"):
                raise ValueError("Registered model must contain its raw-input preprocessing pipeline")
            for _, step in model.steps[:-1]:
                if step is None or step == "passthrough":
                    continue
                if hasattr(step, "transform"):
                    data = step.transform(data)
                elif hasattr(step, "fit_resample"):
                    # SMOTE and other training-only samplers are skipped at inference.
                    continue
                else:
                    raise ValueError("Unsupported inference preprocessing step")
            model = model.steps[-1][1]
        if hasattr(model, "coef_") and np.asarray(model.coef_).shape == (1, len(self.columns)):
            # Exact linear SHAP in log-odds relative to an explicit zero reference.
            values = np.asarray(data)[0] * np.asarray(model.coef_)[0]
            base_value = float(np.asarray(model.intercept_).reshape(-1)[0])
            if np.asarray(model.classes_)[1] != 1:
                values = -values
                base_value = -base_value
            method = "linear_log_odds_zero_reference"
        else:
            import shap
            explainer = shap.TreeExplainer(model)
            values = np.asarray(explainer.shap_values(data))
            if values.shape == (1, len(self.columns), 2):
                values = values[:, :, 1]
            values = values.reshape(-1)
            base = np.asarray(explainer.expected_value).reshape(-1)
            base_value = float(base[-1])
            method = "tree_shap_raw_output"
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) != len(self.columns) or not np.isfinite(values).all() or not np.isfinite(base_value):
            raise ValueError("Model explanation has invalid shape or nonfinite values")
        return values.tolist(), base_value, method


def load_bundle(model_dir=None):
    directory = artifact_path(model_dir or os.getenv("MODEL_DIR", "models"))
    model_default = "xgb_model.joblib" if (directory / "xgb_model.joblib").is_file() else "logistic_model.joblib"
    model_path = artifact_path(os.getenv("MODEL_PATH", str(directory / model_default)))
    scaler = joblib.load(artifact_path(os.getenv("SCALER_PATH", str(directory / "scaler.joblib"))))
    columns = list(joblib.load(artifact_path(os.getenv("COLUMNS_PATH", str(directory / "columns.joblib")))))
    if not columns or len(set(columns)) != len(columns) or not all(isinstance(c, str) and c for c in columns):
        raise ValueError("Model feature names must be unique, nonempty strings")
    uri = os.getenv("MLFLOW_MODEL_URI")
    if uri:
        # Remote loading is opt-in. Registered models must include raw-input preprocessing.
        import mlflow.sklearn
        model = mlflow.sklearn.load_model(uri)
        steps = getattr(model, "named_steps", {})
        remote_scaler = steps.get("scaler")
        if remote_scaler is None or not hasattr(remote_scaler, "n_features_in_"):
            raise ValueError("Registered model must include a fitted raw-input scaler pipeline")
        if getattr(remote_scaler, "n_features_in_") != len(columns):
            raise ValueError("Registered pipeline scaler feature count does not match saved feature names")
        remote_columns = list(getattr(remote_scaler, "feature_names_in_", columns))
        if remote_columns != columns:
            raise ValueError("Registered pipeline feature order does not match saved feature names")
        bundle = InferenceBundle(model, scaler, columns, source="mlflow", raw_pipeline=True)
    else:
        bundle = InferenceBundle(joblib.load(model_path), scaler, columns)
    if getattr(bundle.model, "n_features_in_", len(columns)) != len(columns):
        raise ValueError("Model feature count does not match saved feature names")
    # Catch misaligned or unusable artifacts at startup, before readiness can turn green.
    bundle.score([0.] * len(columns))
    return bundle
