"""One preprocessing and scoring contract for HTTP requests and explanations."""
from dataclasses import dataclass
from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class InvalidFeatures(ValueError):
    """A submitted raw transaction violates the serving input contract."""



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
    drift_baseline: dict | None = None
    baseline_error: str | None = None

    @property
    def explanation_columns(self):
        names = list(self.columns)
        if self.raw_pipeline:
            for _, step in self.model.steps[:-1]:
                if step is None or (isinstance(step, str) and step == "passthrough"):
                    continue
                if hasattr(step, "transform"):
                    if not hasattr(step, "get_feature_names_out"):
                        raise ValueError("Pipeline transformer must expose its output feature names")
                    names = list(step.get_feature_names_out(names))
                elif not hasattr(step, "fit_resample"):
                    raise ValueError("Unsupported inference preprocessing step")
        return names

    def prepare(self, features):
        values = np.asarray(features, dtype=float)
        if values.shape != (len(self.columns),) or not np.isfinite(values).all():
            raise InvalidFeatures(f"Input must contain {len(self.columns)} finite numeric features")
        raw = pd.DataFrame([values], columns=self.columns)
        if self.raw_pipeline:
            if "Amount" in self.columns and raw["Amount"].iloc[0] < 0:
                raise InvalidFeatures("Amount must be nonnegative")
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
                if step is None or (isinstance(step, str) and step == "passthrough"):
                    continue
                if hasattr(step, "transform"):
                    data = step.transform(data)
                elif hasattr(step, "fit_resample"):
                    # SMOTE and other training-only samplers are skipped at inference.
                    continue
                else:
                    raise ValueError("Unsupported inference preprocessing step")
            model = model.steps[-1][1]
        explanation_columns = self.explanation_columns
        if hasattr(model, "coef_") and np.asarray(model.coef_).shape == (1, len(explanation_columns)):
            # Exact linear SHAP in log-odds relative to an explicit zero reference.
            values = np.asarray(data)[0] * np.asarray(model.coef_)[0]
            base_value = float(np.asarray(model.intercept_).reshape(-1)[0])
            if np.asarray(model.classes_)[1] != 1:
                values = -values
                base_value = -base_value
            method = "linear_log_odds_zero_reference"
        else:
            import shap
            classes = np.asarray(getattr(model, "classes_", []))
            scoring_classes = np.asarray(getattr(self.model, "classes_", []))
            single_fraud_class = classes.shape == (1,) and classes[0] == 1
            binary_classes = (classes.shape == (2,) and np.count_nonzero(classes == 0) == 1
                              and np.count_nonzero(classes == 1) == 1)
            if not (single_fraud_class or binary_classes) or not np.array_equal(classes, scoring_classes):
                raise ValueError("Tree explanation requires matching binary scoring classes 0 and 1")
            fraud_index = int(np.flatnonzero(classes == 1)[0])
            explainer = shap.TreeExplainer(model)
            values = np.asarray(explainer.shap_values(data))
            base = np.asarray(explainer.expected_value).reshape(-1)
            if binary_classes and values.shape == (1, len(explanation_columns), 2) and base.shape == (2,):
                values = values[0, :, fraud_index]
                base_value = float(base[fraud_index])
            elif values.shape == (1, len(explanation_columns)) and base.shape == (1,):
                tree_output = getattr(explainer.model, "tree_output", None)
                if single_fraud_class and tree_output == "probability":
                    values = values[0]
                    base_value = float(base[0])
                elif binary_classes and tree_output == "log_odds":
                    # A single binary raw margin describes classes_[1]. Its opposite
                    # has negated contributions and base.
                    values = values[0]
                    base_value = float(base[0])
                    from sklearn.dummy import DummyClassifier
                    from sklearn.ensemble import GradientBoostingClassifier
                    if isinstance(model, GradientBoostingClassifier):
                        if (type(model) is not GradientBoostingClassifier
                                or type(model.init_) is not DummyClassifier
                                or model.init_.strategy not in {"prior", "uniform", "most_frequent", "constant"}):
                            raise ValueError("Unsupported gradient boosting initial estimator")
                        # SHAP 0.49 uses the class prior for its initial offset.
                        # Sklearn may use a different deterministic initializer
                        # or half a logit for exponential loss.
                        init = np.asarray(model._raw_predict_init(data), dtype=float).reshape(-1)
                        offset = np.asarray(explainer.model.base_offset, dtype=float).reshape(-1)
                        if (init.shape != (1,) or offset.shape != (1,)
                                or not np.isfinite(init).all() or not np.isfinite(offset).all()):
                            raise ValueError("Tree explanation has invalid initial margin shape")
                        base_value += float(init[0] - offset[0])
                    if fraud_index == 0:
                        values = -values
                        base_value = -base_value
                else:
                    raise ValueError("Unsupported single-output tree explanation")
            else:
                raise ValueError("Tree explanation has invalid shape")
            method = "tree_shap_raw_output"
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) != len(explanation_columns) or not np.isfinite(values).all() or not np.isfinite(base_value):
            raise ValueError("Model explanation has invalid shape or nonfinite values")
        return values.tolist(), base_value, method


def load_bundle(model_dir=None):
    from api.drift import load_baseline
    directory = artifact_path(model_dir or os.getenv("MODEL_DIR", "models"))
    if (directory / "pipeline.joblib").is_file():
        model_default = "pipeline.joblib"
    else:
        model_default = "xgb_model.joblib" if (directory / "xgb_model.joblib").is_file() else "logistic_model.joblib"
    model_path = artifact_path(os.getenv("MODEL_PATH", str(directory / model_default)))
    columns = list(joblib.load(artifact_path(os.getenv("COLUMNS_PATH", str(directory / "columns.joblib")))))
    if not columns or len(set(columns)) != len(columns) or not all(isinstance(c, str) and c for c in columns):
        raise ValueError("Model feature names must be unique, nonempty strings")
    uri = os.getenv("MLFLOW_MODEL_URI")
    if uri:
        import mlflow.sklearn
        model = mlflow.sklearn.load_model(uri)
    else:
        model = joblib.load(model_path)
    raw_pipeline = hasattr(model, "steps")
    if uri and not raw_pipeline:
        raise ValueError("Registered model must include its fitted raw-input preprocessing pipeline")
    if raw_pipeline:
        scaler = model.named_steps.get("scaler")
        if scaler is None or not hasattr(scaler, "n_features_in_"):
            raise ValueError("Model pipeline must include a fitted scaler")
        model_columns = list(getattr(model, "feature_names_in_", columns))
        if model_columns != columns:
            raise ValueError("Pipeline raw feature order does not match saved feature names")
    else:
        scaler = joblib.load(artifact_path(os.getenv("SCALER_PATH", str(directory / "scaler.joblib"))))
    bundle = InferenceBundle(model, scaler, columns, source="mlflow" if uri else "local", raw_pipeline=raw_pipeline)
    if getattr(bundle.model, "n_features_in_", len(columns)) != len(columns):
        raise ValueError("Model raw feature count does not match saved feature names")
    final_model = model.steps[-1][1] if raw_pipeline else model
    if getattr(final_model, "n_features_in_", len(bundle.explanation_columns)) != len(bundle.explanation_columns):
        raise ValueError("Model transformed feature count does not match explanation feature names")
    # A remote model never silently uses an unrelated local reference sample.
    baseline_path = os.getenv("DRIFT_BASELINE_PATH")
    remote_baseline = os.getenv("MLFLOW_DRIFT_BASELINE_URI")
    try:
        if remote_baseline:
            import mlflow.artifacts
            baseline_path = mlflow.artifacts.download_artifacts(artifact_uri=remote_baseline)
        elif not uri and baseline_path is None:
            baseline_path = directory / "drift_baseline.json"
        if baseline_path is not None:
            bundle.drift_baseline = load_baseline(artifact_path(baseline_path), columns)
    except Exception:
        # Drift reference failure must be visible without taking a valid scorer offline.
        bundle.baseline_error = "Configured raw training drift baseline is invalid or unavailable"
    bundle.score([0.] * len(columns))
    return bundle
