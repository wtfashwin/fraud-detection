import joblib
import pandas as pd
import json
import os

class FraudDetector:
    def __init__(self, model_path, scaler_path, features_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(features_path, 'r') as f:
            self.features = json.load(f)

    def _validate_and_prepare_input(self, input_data):
        # Accept input dict or dataframe
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data.copy()
        else:
            raise ValueError("Input data must be dict or pandas DataFrame")

        # Filter only required features in correct order
        input_df = input_df[self.features]

        # Scale input features
        features_scaled = self.scaler.transform(input_df.values)
        return features_scaled

    def predict(self, input_data):
        features_scaled = self._validate_and_prepare_input(input_data)
        pred = self.model.predict(features_scaled)
        prob = self.model.predict_proba(features_scaled)[:, 1]
        return pred[0], prob[0]

if __name__ == "__main__":
    BASE_DIR = 'models'
    detector = FraudDetector(
        model_path=os.path.join(BASE_DIR, 'logistic_model.joblib'),
        scaler_path=os.path.join(BASE_DIR, 'scaler.joblib'),
        features_path=os.path.join(BASE_DIR, 'feature_names.json')
    )

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
