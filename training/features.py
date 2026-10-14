"""Optional 30-to-12 feature experiment fitted inside each training fold.

V1..V28 are already anonymized PCA-derived features in the public benchmark.
Reducing them again is an experiment, not an established quality improvement.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

# Keep this transformer importable by serving images without training-only SMOTE.
RAW_FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
PCA_FEATURE_NAMES = [f"V{i}" for i in range(1, 29)]


class FraudFeatureEngineer(TransformerMixin, BaseEstimator):
    """Keep Time, log-transform Amount, and project V1..V28 with fitted PCA.

    Input is a named raw-feature DataFrame with exactly the canonical 30 columns.
    Output is a DataFrame in [Time, logAmount, PC1..PCn] order with its index intact.
    Put this step before scaling and sampling in a fitted sklearn/imblearn pipeline;
    fitting it on the whole dataset before cross-validation would leak information.
    """

    def __init__(self, n_components=10, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    @staticmethod
    def _raw_frame(X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Feature engineering requires a named raw-feature DataFrame")
        if X.empty or X.columns.duplicated().any() or set(X.columns) != set(RAW_FEATURE_NAMES):
            raise ValueError("Expected exactly the 30 canonical raw feature columns, without Class")
        try:
            raw = X.loc[:, RAW_FEATURE_NAMES].apply(pd.to_numeric, errors="raise").astype(float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("Raw features must be finite numeric values") from exc
        if not np.isfinite(raw.to_numpy()).all():
            raise ValueError("Raw features must contain only finite values")
        if (raw["Amount"] < 0).any():
            raise ValueError("Amount must be nonnegative for the log1p transformation")
        return raw

    def fit(self, X, y=None):
        raw = self._raw_frame(X)
        if (isinstance(self.n_components, bool) or not isinstance(self.n_components, (int, np.integer))
                or not 1 <= self.n_components <= min(len(raw), len(PCA_FEATURE_NAMES))):
            raise ValueError("PCA components must be an integer from 1 to min(training rows, 28)")
        self.pca_ = PCA(n_components=self.n_components, svd_solver="full", random_state=self.random_state)
        self.pca_.fit(raw[PCA_FEATURE_NAMES])
        self.n_features_in_ = len(RAW_FEATURE_NAMES)
        self.feature_names_in_ = np.asarray(RAW_FEATURE_NAMES, dtype=object)
        self.n_components_ = int(self.n_components)
        return self

    def transform(self, X):
        check_is_fitted(self, ["pca_", "n_components_", "feature_names_in_"])
        raw = self._raw_frame(X)
        components = self.pca_.transform(raw[PCA_FEATURE_NAMES])
        values = np.column_stack([raw["Time"].to_numpy(), np.log1p(raw["Amount"].to_numpy()), components])
        if not np.isfinite(values).all():
            raise ValueError("Feature engineering produced nonfinite values")
        return pd.DataFrame(values, index=raw.index, columns=self.get_feature_names_out())

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["pca_", "n_components_", "feature_names_in_"])
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError("Input feature names must match the canonical raw feature order")
        return np.asarray(["Time", "logAmount"] + [f"PC{i}" for i in range(1, self.n_components_ + 1)], dtype=object)
