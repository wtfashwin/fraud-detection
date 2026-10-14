"""Generate schema-compatible random fixtures; these do not establish fraud quality."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

output_file_path = "data/creditcard.csv"


def generate_synthetic_data(n_samples=1000, n_features=30, fraud_ratio=0.01, random_state=42):
    """Labels are random and independent of features, intentionally without real signal."""
    if n_features != 30:
        raise ValueError("The fraud feature contract requires exactly 30 features")
    if not isinstance(n_samples, int) or n_samples < 2:
        raise ValueError("n_samples must be an integer of at least 2")
    if not 0 < fraud_ratio < 1:
        raise ValueError("fraud_ratio must be between 0 and 1")
    n_frauds = int(n_samples * fraud_ratio)
    if n_frauds == 0 or n_frauds == n_samples:
        raise ValueError("Sample count and fraud_ratio must produce both classes")
    rng = np.random.default_rng(random_state)
    features = np.column_stack([
        np.sort(rng.uniform(0, 172800, n_samples)),
        rng.normal(size=(n_samples, 28)),
        np.exp(rng.normal(3, 1, n_samples)),
    ])
    labels = np.zeros(n_samples, dtype=int)
    labels[rng.choice(n_samples, n_frauds, replace=False)] = 1
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    frame = pd.DataFrame(features, columns=columns)
    frame["Class"] = labels
    return frame


def main():
    path = Path(os.getenv("SYNTHETIC_OUTPUT", output_file_path))
    samples = int(os.getenv("CI_SYNTHETIC_SAMPLES", "1000"))
    ratio = float(os.getenv("SYNTHETIC_FRAUD_RATIO", "0.1"))
    seed = int(os.getenv("RANDOM_STATE", "42"))
    frame = generate_synthetic_data(samples, fraud_ratio=ratio, random_state=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    path.with_suffix(".metadata.json").write_text(json.dumps({
        "data_kind": "synthetic", "rows": samples, "fraud_ratio": ratio,
        "random_state": seed, "labels": "random independent of features",
        "purpose": "pipeline smoke tests only; not real fraud quality evidence",
    }, indent=2) + "\n")
    print(f"Wrote {samples} synthetic smoke-test rows to {path}; no real fraud quality claim.")


if __name__ == "__main__":
    main()
