import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False


def main():
    print("Loading dataset...")
    df = pd.read_csv(os.getenv('DATA_CSV', 'data/creditcard.csv'))

    print("Checking missing values...")
    print(df.isnull().sum())

    print("Scaling features...")
    features = df.drop(columns=["Class"], errors='ignore') 
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    print(" Preparing final dataset...")
    X = scaled_features
    y = df["Class"].values

    print("Splitting dataset (Train 80% / Test 20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Casting target variables to integer...")
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    print(f" Class balance before SMOTE → {np.bincount(y_train)}")

    print(" Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f" Class balance after SMOTE → {np.bincount(y_res)}")

    print(" Training LogisticRegression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_res, y_res)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f"Test AUC: {auc:.4f}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/logistic_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    print(" Model and scaler saved to /models")

    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
        experiment_name = os.getenv('MLFLOW_EXPERIMENT', 'fraud-detection')
        model_name = os.getenv('MLFLOW_MODEL_NAME', 'fraud-detection-model')
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            mlflow.log_metric('test_auc', float(auc))
            mlflow.sklearn.log_model(model, 'model')
            mlflow.log_artifact('models/scaler.joblib')
            print(f" Logged run to MLflow experiment '{experiment_name}'")
            
            # Register model if AUC meets threshold
            auc_threshold = float(os.getenv('MLFLOW_AUC_THRESHOLD', '0.95'))
            if auc >= auc_threshold:
                try:
                    result = mlflow.register_model(
                        f"runs:/{mlflow.active_run().info.run_id}/model",
                        model_name
                    )
                    print(f" Registered model version {result.version} (AUC {auc:.4f} >= {auc_threshold})")
                except Exception as e:
                    print(f" Model registration failed: {e}")
            else:
                print(f"ℹ Model not registered (AUC {auc:.4f} < {auc_threshold})")
    else:
        print("MLflow not available; skipping tracking")


if __name__ == "__main__":
    main()