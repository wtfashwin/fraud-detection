import joblib
import pandas as pd
import json
import os
from pathlib import Path
import numpy as np

from training import PROJECT_ROOT
from api.inference import InferenceBundle, load_bundle

class FraudDetector:
    def __init__(self, model_path=None, scaler_path=None, features_path=None, *, model_dir=None):
        if model_path is None:
            # The shared loader prefers the authoritative raw-input pipeline.
            self.bundle = load_bundle(model_dir)
        else:
            if scaler_path is None or features_path is None:
                raise ValueError("Legacy model loading requires scaler_path and features_path")
            with open(features_path, "r") as stream:
                features = json.load(stream)
            self.bundle = InferenceBundle(joblib.load(model_path), joblib.load(scaler_path), features)
        self.model = self.bundle.model
        self.scaler = self.bundle.scaler
        self.features = list(self.bundle.columns)

    def _validate_raw_input(self, input_data):
        # Accept input dict or dataframe
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data.copy()
        else:
            raise ValueError("Input data must be dict or pandas DataFrame")

        if len(input_df) != 1:
            raise ValueError("Single prediction requires exactly one transaction")
        if input_df.columns.duplicated().any():
            raise ValueError("Duplicate input features are not allowed")
        missing = sorted(set(self.features) - set(input_df.columns))
        extra = sorted(set(input_df.columns) - set(self.features))
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        if extra:
            raise ValueError(f"Unexpected input features: {extra}")
        try:
            input_df = input_df[self.features].apply(pd.to_numeric, errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError("Input features must be numeric") from exc
        if not np.isfinite(input_df.to_numpy()).all():
            raise ValueError("Input features must contain only finite values")
        return input_df

    def predict(self, input_data):
        raw = self._validate_raw_input(input_data)
        return self.bundle.score(raw.iloc[0].to_list())

if __name__ == "__main__":
    BASE_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
    detector = FraudDetector(model_dir=BASE_DIR)

    # Example input - fill all features exactly as in training data
    sample_input = {
        "Time": 0,
        "V1": -1.3598071336738,
        "V2": -0.0727811733098497,
        "V3": 2.53634673796914,
        "V4": 1.37815522427443,
        "V5": -0.338320769942518,
        "V6": 0.462387777762292,
        "V7": 0.239598554061257,
        "V8": 0.0986979012610507,
        "V9": 0.363786969611213,
        "V10": 0.0907941719789316,
        "V11": -0.551599533260813,
        "V12": -0.617800855762348,
        "V13": -0.991389847235408,
        "V14": -0.311169353699879,
        "V15": 1.46817697209427,
        "V16": -0.470400525259478,
        "V17": 0.207971241929242,
        "V18": 0.0257905801985591,
        "V19": 0.403992960255733,
        "V20": 0.251412098239705,
        "V21": -0.018306777944153,
        "V22": 0.277837575558899,
        "V23": -0.110473910188767,
        "V24": 0.0669280749146731,
        "V25": 0.128539358273528,
        "V26": -0.189114843888824,
        "V27": 0.133558376740387,
        "V28": -0.0210530534538215,
        "Amount": 149.62
    }

    prediction, probability = detector.predict(sample_input)
    print(f"Prediction: {prediction} (1 = Fraud, 0 = Legit)")
    print(f"Fraud Probability: {probability:.4f}")
