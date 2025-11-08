import os
import warnings
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
from mlflow.models import infer_signature

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=DeprecationWarning)

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

    print(" Performing Stratified K-Fold Cross-Validation with SMOTE applied within each fold...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Calculate scale_pos_weight from original training data
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    print(f" Scale pos weight for XGBoost: {scale_pos_weight:.2f}")

    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        # Split original training data into fold train and validation
        X_fold_train_raw, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train_raw, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Apply SMOTE only to the training fold (not validation) to avoid data leakage
        print(f"  Fold {fold+1}: Applying SMOTE to training fold...")
        smote = SMOTE(random_state=42, sampling_strategy='minority')
        X_fold_train, y_fold_train = smote.fit_resample(X_fold_train_raw, y_fold_train_raw)
        print(f"    Training samples: {len(y_fold_train_raw)} → {len(y_fold_train)} (after SMOTE)")
        
        model_fold = XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            verbosity=0  # Suppress XGBoost output
        )
        model_fold.fit(X_fold_train, y_fold_train)
        
        preds = model_fold.predict_proba(X_fold_val)[:, 1]
        auc = roc_auc_score(y_fold_val, preds)
        cv_scores.append(auc)
        print(f"  Fold {fold+1} AUC: {auc:.4f}")
    
    print(f" CV AUC Mean: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

    print(" Training final XGBoost Classifier model with SMOTE on full training set...")
    # Apply SMOTE to full training set for final model
    smote = SMOTE(random_state=42, sampling_strategy='minority')
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train)
    print(f" Class balance after SMOTE → {np.bincount(y_res)}")
    
    model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        verbosity=0  # Suppress XGBoost output
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
            mlflow.log_param('model_type', 'XGBoost')
            mlflow.log_param('scale_pos_weight', scale_pos_weight)
            mlflow.log_param('cv_folds', 5)
            mlflow.log_metric('test_auc', float(auc))
            mlflow.log_metric('cv_auc_mean', float(np.mean(cv_scores)))
            mlflow.log_metric('cv_auc_std', float(np.std(cv_scores)))
            
            # Create input signature and example for better serving compatibility
            # Use a sample from test set for signature inference
            sample_input = X_test_scaled[:5]
            sample_output = model.predict_proba(sample_input)[:, 1]
            
            # Infer signature from sample data
            signature = infer_signature(sample_input, sample_output)
            
            # Create example input (first row of test set)
            input_example = X_test_scaled[:1]
            
            # Log model with signature and example
            mlflow.sklearn.log_model(
                model, 
                'model',
                signature=signature,
                input_example=input_example
            )
            mlflow.log_artifact('models/scaler.joblib')
            print(f" Logged run to MLflow experiment '{experiment_name}' with input signature and examples")
            
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