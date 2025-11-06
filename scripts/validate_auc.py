import sys
import mlflow
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np

def load_synthetic_data(size=1000):
    np.random.seed(42)  
    features = np.random.randn(size, 10)
    labels = (features[:, 0] > 0).astype(int)  
    df = pd.DataFrame(features)
    return df, labels

def validate_auc(model_uri, threshold=0.95):
    try:
        with mlflow.start_run():
            X, y = load_synthetic_data()
            model = mlflow.pyfunc.load_model(model_uri)
            preds = model.predict(X)
            if hasattr(preds, 'shape') and len(preds.shape) > 1 and preds.shape[1] > 1:
                preds = preds[:, 1]
            auc = float(roc_auc_score(y, preds))
            mlflow.log_metric("auc_score", float(auc))
            mlflow.set_tag("validation_pass", str(auc >= threshold))
            print("AUC validation complete: auc={}, threshold={}".format(auc, threshold))
            return [float(auc), bool(auc >= threshold)]
    except Exception as e:
        print("Error in validation: {}".format(e))
        return [0.0, False]

if __name__ == "__main__":
    uri = sys.argv[1] if len(sys.argv) > 1 else "models:/fraud/prod"
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95
    result = validate_auc(uri, threshold)
    if result:
        auc, passed = result
        exit_code = 0 if passed else 1
        sys.exit(exit_code)
    else:
        sys.exit(1)