import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import json
import os

DATA_PATH = 'data/creditcard.csv'
PROCESSED_DATA_PATH = 'data/preprocessed_data.npz'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("🔍 Checking missing values:")
print(df.isnull().sum())

# Features and labels
X = df.drop('Class', axis=1)
y = df['Class']
feature_names = X.columns.tolist()  # Save before scaling

print("🔄 Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("🔀 Splitting dataset (Train 80% / Test 20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y.values, test_size=0.2, random_state=42, stratify=y
)

print("⚠️ Class balance before SMOTE →", np.bincount(y_train))

print("🔁 Applying SMOTE to balance training data...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

print("✅ Class balance after SMOTE →", np.bincount(y_res))

print("💾 Saving preprocessed arrays...")
np.savez_compressed(PROCESSED_DATA_PATH, X_res=X_res, y_res=y_res, X_test=X_test, y_test=y_test)

print("💾 Saving scaler...")
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))

print("💾 Saving feature names...")
joblib.dump(feature_names, os.path.join(MODELS_DIR, 'columns.joblib'))
with open(os.path.join(MODELS_DIR, 'feature_names.json'), 'w') as f:
    json.dump(feature_names, f)

print("🏁 Preprocessing complete.")
