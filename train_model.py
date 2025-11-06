import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier 
import joblib

import mlflow
import mlflow.sklearn

def main():
    print("Loading dataset...")
    df = pd.read_csv(os.getenv('DATA_CSV', 'data/creditcard.csv'))

    print("Checking missing values...")
    print(df.isnull().sum())

    print("Splitting dataset (Train 80% / Test 20%)...")
    X = df.drop(columns=["Class"], errors='ignore').values
    y = df["Class"].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Scaling features (only fitting on Train set)...")
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    
    X_test_scaled = scaler.transform(X_test) 

    print("Casting target variables to integer...")
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    print(f" Class balance before SMOTE → {np.bincount(y_train)}")

    print(" Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42, sampling_strategy='minority')
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train) # Use scaled training data

    print(f" Class balance after SMOTE → {np.bincount(y_res)}")

    # --- 4. CROSS-VALIDATION WITH STRATIFIED K-FOLD ---
    print(" Performing Stratified K-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    neg_count = np.sum(y_res == 0)
    pos_count = np.sum(y_res == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    print(f" Scale pos weight for XGBoost: {scale_pos_weight:.2f}")

    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_res, y_res)):
        X_fold_train, X_fold_val = X_res[train_idx], X_res[val_idx]
        y_fold_train, y_fold_val = y_res[train_idx], y_res[val_idx]
        
        model_fold = XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            use_label_encoder=False, 
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight
        )
        model_fold.fit(X_fold_train, y_fold_train)
        
        preds = model_fold.predict_proba(X_fold_val)[:, 1]
        auc = roc_auc_score(y_fold_val, preds)
        cv_scores.append(auc)
        print(f"  Fold {fold+1} AUC: {auc:.4f}")
    
    print(f" CV AUC Mean: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

    print(" Training final XGBoost Classifier model...")
    model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        use_label_encoder=False, 
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )
    model.fit(X_res, y_res)

    preds = model.predict_proba(X_test_scaled)[:, 1] 
    auc = roc_auc_score(y_test, preds)
    print(f"Test AUC: {auc:.4f}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgb_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    print(" Model and scaler saved to /models")

    try:
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
        experiment_name = os.getenv('MLFLOW_EXPERIMENT', 'fraud-detection-ci') 
        model_name = os.getenv('MLFLOW_MODEL_NAME', 'fraud-detection-model')
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            mlflow.log_param('model_type', 'XGBoost') # Log the new model type
            mlflow.log_param('scale_pos_weight', scale_pos_weight)
            mlflow.log_param('cv_folds', 5)
            mlflow.log_metric('test_auc', float(auc))
            mlflow.log_metric('cv_auc_mean', float(np.mean(cv_scores)))
            mlflow.log_metric('cv_auc_std', float(np.std(cv_scores)))
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