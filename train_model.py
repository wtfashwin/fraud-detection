import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier 
from sklearn.metrics import roc_auc_score
import joblib

import mlflow
import mlflow.sklearn

def main():
    print("Loading dataset...")
    df = pd.read_csv(os.getenv('DATA_CSV', 'data/creditcard.csv'))

    print("Checking missing values...")
    print(df.isnull().sum())

    # --- 1. SPLIT FIRST (CRITICAL CHANGE) ---
    print("Splitting dataset (Train 80% / Test 20%)...")
    X = df.drop(columns=["Class"], errors='ignore').values
    y = df["Class"].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 2. SCALE AFTER SPLIT (CRITICAL CHANGE) ---
    print("Scaling features (only fitting on Train set)...")
    scaler = StandardScaler()
    
    # Fit and Transform on TRAINING data
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform on TESTING data (using parameters learned from training)
    X_test_scaled = scaler.transform(X_test) 

    print("Casting target variables to integer...")
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    # Use the scaled training data for SMOTE
    print(f" Class balance before SMOTE → {np.bincount(y_train)}")

    # --- 3. APPLY SMOTE ---
    print(" Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42, sampling_strategy='minority')
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train) # Use scaled training data

    print(f" Class balance after SMOTE → {np.bincount(y_res)}")

    # --- 4. TRAIN XGBOOST MODEL ---
    print(" Training XGBoost Classifier model...")
    # Using good starting hyperparameters for a complex fraud dataset
    model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        use_label_encoder=False, 
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_res, y_res)

    # Use the scaled test data for prediction
    preds = model.predict_proba(X_test_scaled)[:, 1] 
    auc = roc_auc_score(y_test, preds)
    print(f"Test AUC: {auc:.4f}")

    # --- 5. SAVE MODEL AND SCALER ---
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgb_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    print(" Model and scaler saved to /models")

    # --- 6. MLFLOW TRACKING (using existing logic) ---
    try:
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
        experiment_name = os.getenv('MLFLOW_EXPERIMENT', 'fraud-detection-ci') 
        model_name = os.getenv('MLFLOW_MODEL_NAME', 'fraud-detection-model')
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            mlflow.log_param('model_type', 'XGBoost') # Log the new model type
            mlflow.log_metric('test_auc', float(auc))
            mlflow.sklearn.log_model(model, 'model')
            mlflow.log_artifact('models/scaler.joblib')
            print(f" Logged run to MLflow experiment '{experiment_name}'")
            
            auc_threshold = float(os.getenv('MLFLOW_AUC_THRESHOLD', '0.95'))
            if auc >= auc_threshold:
                try:
                    result = mlflow.register_model(
                        f"runs:/{mlflow.last_active_run().info.run_id}/model",
                        model_name
                    )
                    print(f" Registered model version {result.version} (AUC {auc:.4f} >= {auc_threshold})")
                except Exception as e:
                    print(f" Model registration failed: {e}")
            else:
                print(f"ℹ Model not registered (AUC {auc:.4f} < {auc_threshold})")
                
    except Exception as e:
        print(f" MLflow Tracking Failed (likely connection error): {e}")


if __name__ == "__main__":
    main()