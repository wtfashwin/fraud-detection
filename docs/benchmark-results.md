# Public benchmark results

These are executed local experiments on OpenML **42175 version 1**, not synthetic scores. The [source dataset](https://www.openml.org/d/42175) contains 284,807 historical anonymized transactions, 492 fraud labels, the canonical 30 features, and no missing values. V1–V28 already represent PCA-derived inputs.

The source ARFF SHA256 is `62416dc7dc34cf7abda5db818e8db92e91152fb2085340f7681bd6f0fe6130ce`; the converted CSV SHA256 is `2f9cec6e0d657d30c23eb59750d7976a3498d1c2b8899a5800758c51184d2292`. `scripts/fetch_benchmark.py` verifies the fixed source and records provenance. Raw records and fitted experiment bundles stay out of Git.

Both experiments first reserved the same 20% stratified holdout with seed 42: **56,962 rows, including 98 fraud labels**. Each retained all 394 training fraud rows and a deterministic sample of 20,000 legitimate training rows. Scaling, optional PCA, and SMOTE were fitted within the five training folds. SMOTE targets 0.5; validation and holdout rows remain untouched. The final holdout did not select hyperparameters.

| Experiment | Optuna trials | Holdout ROC AUC | Average precision | Precision at 0.5 | Recall at 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original 30 features, tuned XGBoost | 10 | 0.973709 | 0.746473 | 0.332075 | 0.897959 |
| log1p(Amount), fitted PCA10, XGBoost | 1 | 0.977510 | 0.747054 | 0.215159 | 0.897959 |

Issue #8's engineered ROC AUC >0.85 and issue #10's tuned ROC AUC >0.95 both passed. The experiments use different parameters and search budgets; these results do not establish that the extra PCA improves the model. The EDA's fitted ten-component projection retains 64.766455% of training V-feature variance, which does not itself measure fraud discrimination.

The raw-feature search selected trial 8 by mean training-fold ROC AUC (0.980277). Learning rate, maximum depth, and residual class weight were searched in the requested ranges; the class-weight range is 1–2 because SMOTE already leaves a 2:1 legitimate/fraud training ratio. Applying the original large imbalance weight after oversampling would count imbalance twice.

At threshold 0.5 the tuned raw model produced 88 true positives, 10 false negatives, 177 false positives, and 56,687 true negatives. Probabilities have not been calibrated. The historical random holdout does not test time shift, current fraud patterns, deployment cohorts, production load, or decision costs. These bundles are separate experiments and are not substituted for the packaged demonstration model.

## Reproduce

Install `requirements.txt`, then choose a fresh directory under the ignored `data/benchmark/` tree. The downloader refuses to replace an existing CSV or metadata sidecar, including on a repeat run. For example:

```sh
python scripts/fetch_benchmark.py --directory data/benchmark/reproduction
python -m training.tune --data data/benchmark/reproduction/creditcard.csv \
  --output models/benchmark-optuna --trials 10 --max-majority 20000 \
  --estimators 100 --study-name benchmark-optuna-raw30-seed42 --require-target
python -m training.tune --data data/benchmark/reproduction/creditcard.csv \
  --output models/benchmark-engineered --trials 1 --max-majority 20000 \
  --estimators 100 --feature-engineering --study-name benchmark-engineered
```

The downloader buffers up to 256 MiB of source and 512 MiB of CSV in memory; parsing and conversion need additional memory. It requires `O_NOFOLLOW` and `O_NONBLOCK` to examine the source path even for a fresh download, and rejects source symlinks and nonregular files. The filesystem must honor exclusive creation with `O_CREAT|O_EXCL`. It writes metadata before CSV, and either file can be visible before writing finishes. If the command fails, inspect the files named in the error and retry in a fresh directory; do not use a partial pair. The `training.train`, `training.tune`, and `training.evaluate` commands verify a declared `csv_sha256` against the bytes they read. Reading a present sidecar requires `O_NONBLOCK` and a regular file, though a symlink to a regular file is supported; datasets with no sidecar remain supported without that flag. `training.preprocess` and `scripts/validate_auc.py` do not perform this check, while the feature engineering notebook checks a separate file read. The downloader parses verified source bytes, but another process can replace paths during or after processing or modify a newly created file while it is being written. Missing sidecars and external readers are outside the check. The two output names are not published atomically.

Models, five-row input examples, signatures, aggregate metrics, and the training-only raw drift reference are logged to local MLflow. Registration was disabled for these experiments. The safe aggregate results, parameters, provenance, and package versions are recorded in [benchmark-results.json](benchmark-results.json). The [EDA notebook](../notebooks/feature_engineering.ipynb) contains executed aggregate plots without raw transaction tables.
