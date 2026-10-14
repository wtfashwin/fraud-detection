# Local training data

Compatible fraud data uses `Time`, `V1` through `V28`, `Amount`, and binary `Class`. Supply your own authorized dataset with `DATA_CSV`; keep real transaction data out of Git.

`python scripts/generate_synthetic_data.py` writes `data/creditcard.csv` and its synthetic metadata sidecar. Both are ignored by Git. Random synthetic labels test software behavior and cannot establish fraud accuracy.

The older credit-approval sample is retained separately under `archive/data/` and is deliberately incompatible with training.

`python scripts/fetch_benchmark.py` downloads the fixed public OpenML 42175 v1 benchmark, verifies its checksum/schema/counts, and writes `data/benchmark/creditcard.csv` plus provenance. This directory is ignored by Git and Docker. See `docs/benchmark-results.md` for the executed holdout results and their limits.
