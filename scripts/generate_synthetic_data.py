import os
import numpy as np
import pandas as pd
output_file_path = 'data/creditcard.csv' 
df.to_csv(output_file_path, index=False)

def generate_synthetic_data(n_samples=1000, n_features=30, fraud_ratio=0.01):
    """Generate synthetic credit card transaction data for testing."""
    np.random.seed(42)
    
    X = np.random.randn(n_samples, n_features - 2)  
    
    time = np.sort(np.random.uniform(0, 172800, n_samples)) 
    

    amount = np.exp(np.random.normal(3, 1, n_samples)) 
    
    features = np.column_stack([time, X, amount])
    
    n_frauds = int(n_samples * fraud_ratio)
    y = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, n_frauds, replace=False)
    y[fraud_indices] = 1
    
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    df = pd.DataFrame(np.column_stack([features, y]), columns=columns)
    
    return df

def main():
    os.makedirs('data', exist_ok=True)
    
    ci_samples = int(os.getenv('CI_SYNTHETIC_SAMPLES', '1000'))
    test_samples = int(os.getenv('TEST_SYNTHETIC_SAMPLES', '10000'))
    
    print(f"Generating {ci_samples} samples for CI testing...")
    ci_data = generate_synthetic_data(n_samples=ci_samples)
    ci_data.to_csv('data/synthetic_ci.csv', index=False)
    
    print(f"Generating {test_samples} samples for local testing...")
    test_data = generate_synthetic_data(n_samples=test_samples)
    test_data.to_csv('data/synthetic_test.csv', index=False)
    
if __name__ == '__main__':
    main()