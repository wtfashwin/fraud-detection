from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load trained model and scaler
print("📥 Loading model, scaler, and columns...")
model = joblib.load('models/logistic_model.joblib')
scaler = joblib.load('models/scaler.joblib')
columns = joblib.load('models/columns.joblib')

@app.route('/', methods=['GET'])
def index():
    return jsonify({"msg": "Fraud Detection API is live 🚀"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json or request.form.to_dict()

        # Convert input to DataFrame
        input_df = pd.DataFrame([data], columns=columns)

        # Ensure numeric values (optional sanity check)
        input_df = input_df.apply(pd.to_numeric, errors='coerce')

        # Scale input
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]

        # Mark alert if high probability
        alert = prob > 0.8

        result = {
            'prediction': int(prediction),
            'fraud_probability': round(float(prob), 4),
            'alert': alert
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
