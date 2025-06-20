import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

def main():
    print("📥 Loading dataset...")
    # Change this path if your CSV is somewhere else
    df = pd.read_csv("data/creditcard.csv")  

    print("🔍 Checking missing values...")
    print(df.isnull().sum())

    print("🔄 Scaling features...")
    # Features to scale - exclude 'Class' and 'Time' optionally
    features = df.drop(columns=["Class", "Time"])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    print("🔀 Preparing final dataset...")
    X = scaled_features
    y = df["Class"].values

    print("🔀 Splitting dataset (Train 80% / Test 20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"⚠️ Class balance before SMOTE → {np.bincount(y_train)}")

    print("🤖 Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"✅ Class balance after SMOTE → {np.bincount(y_res)}")

    print("💾 Saving preprocessed arrays and scaler...")
    np.savez_compressed(
        "preprocessed_data.npz",
        X_res=X_res,
        y_res=y_res,
        X_test=X_test,
        y_test=y_test
    )
    joblib.dump(scaler, "scaler.joblib")

    print("🏁 Preprocessing complete.")

if __name__ == "__main__":
    main()
