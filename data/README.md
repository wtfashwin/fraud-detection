# Local training data

Compatible fraud data uses `Time`, `V1` through `V28`, `Amount`, and binary `Class`. Supply your own authorized dataset with `DATA_CSV`; keep real transaction data out of Git.

`python scripts/generate_synthetic_data.py` writes `data/creditcard.csv` and its synthetic metadata sidecar. Both are ignored by Git. Random synthetic labels test software behavior and cannot establish fraud accuracy.

The older credit-approval sample is retained separately under `archive/data/` and is deliberately incompatible with training.
