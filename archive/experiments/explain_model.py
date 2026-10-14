import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

print("📥 Loading model, scaler, and data...")

data = np.load('data/preprocessed_data.npz')
X_test, y_test = data['X_test'], data['y_test']

scaler = joblib.load('models/scaler.joblib')
model = joblib.load('models/logistic_model.joblib')


X_test_scaled = scaler.transform(X_test)

print("🔍 Setting up SHAP explainer...")

# Logistic Regression is linear, so LinearExplainer is ideal
explainer = shap.LinearExplainer(model, X_test_scaled, feature_perturbation="interventional")

print("⚡ Computing SHAP values for test set (this may take a moment)...")
shap_values = explainer.shap_values(X_test_scaled)

print("📊 Plotting SHAP summary plot...")
plt.figure(figsize=(12,8))
shap.summary_plot(shap_values, X_test_scaled, feature_names=[f'Feature_{i}' for i in range(X_test_scaled.shape[1])], show=False)
plt.tight_layout()
plt.savefig('plots/shap_summary.png')
plt.close()

print("📈 Plotting SHAP dependence plot for top features...")

# Find top 3 features with highest mean absolute SHAP value
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_features_idx = mean_abs_shap.argsort()[-3:][::-1]

for idx in top_features_idx:
    plt.figure(figsize=(8,6))
    shap.dependence_plot(idx, shap_values, X_test_scaled, feature_names=[f'Feature_{i}' for i in range(X_test_scaled.shape[1])], show=False)
    plt.tight_layout()
    plt.savefig(f'plots/shap_dependence_feature_{idx}.png')
    plt.close()

print("✅ SHAP explainability completed. Plots saved in 'plots/' folder.")
