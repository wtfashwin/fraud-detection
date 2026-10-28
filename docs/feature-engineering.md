# Optional feature engineering

`training.features.FraudFeatureEngineer` transforms the canonical **30 raw features** into **12 model features** by default:

| Output | Source |
| --- | --- |
| `Time` | Raw `Time`, unchanged. |
| `logAmount` | `log1p(Amount)`; `Amount` must be finite and nonnegative. |
| `PC1` through `PC10` | Fitted PCA projection of `V1` through `V28`. |

The transformer returns a DataFrame with the original row index. It accepts the exact named raw schema, restores canonical column order, and rejects missing/duplicate/extra columns and nonfinite values. `Class` is a label and cannot enter the transformer. Fit on training rows only, then reuse the fitted projection on validation, holdout, and serving requests. `transform` never refits PCA.

Put this step inside the fitted training pipeline before scaling and SMOTE, so each cross-validation fold fits its own PCA. Save the **whole fitted pipeline**, including this transformer; a final estimator and separate scaler alone cannot reproduce the 30-to-12 mapping. Serving must retain the raw 30-feature input contract and return transformed feature names for explanations. The `training.features` module must be importable when unpickling that pipeline.

V1..V28 in the public anonymized credit-card benchmark are already PCA-derived. This additional dimensionality reduction is an **opt-in experiment**. Explained variance, prettier distributions, and synthetic test results do not prove improved fraud detection. Compare baseline and engineered pipelines on the same untouched labelled holdout before choosing one. The existing bundled model is not replaced by this module.

## Notebook

`notebooks/feature_engineering.ipynb` is the historical executed notebook. It preserves the saved benchmark outputs and execution metadata. Its loader parsed the CSV path and then reopened that path to hash it, so the recorded hash alone cannot prove the parsed bytes were identical if the file changed between reads. This review did not independently observe the original execution or establish whether a change occurred.

The historical notebook records a run labeled as the [OpenML 42175 version 1 public benchmark](https://www.openml.org/d/42175). Its saved aggregate outputs report 284,807 rows, 492 fraud labels, and a 227,845-row training partition. The 10 retained PCA components explain 64.766455% of V1..V28 variance in that recorded partition. This is representation evidence, not measured fraud-detection improvement.

The notebook recorded CSV SHA256 `2f9cec6e0d657d30c23eb59750d7976a3498d1c2b8899a5800758c51184d2292`. Its embedded aggregate figures match `docs/figures/benchmark-class-balance.png`, `benchmark-amount-transform.png`, and `benchmark-pca-training-variance.png`. Raw benchmark rows are not copied into the notebook or documentation.

For future runs, use `notebooks/feature_engineering_verified.ipynb`. This copy is unexecuted and has no saved results. It requires `DATA_CSV`, reads one byte snapshot, checks a declared sidecar hash before parsing, and reports the hash of those same bytes. It retains pandas' filename-based compression inference and default parsing behavior. The input bytes are held in memory in addition to the parsed data. `DATA_KIND` takes precedence over a sidecar data kind; both are declarations, not proof of origin. Missing sidecars and sidecars without a hash remain supported without a sidecar hash check. A present sidecar must resolve to a regular file and requires `O_NONBLOCK`.

Use the new notebook's setup command to write a separate run result. Its loader is tested with synthetic files; those tests do not reproduce the historical benchmark outputs or prove model quality. The saved historical notebook and figures are not modified by these tests.

## Transformer checks

```sh
python -m pytest -q tests/test_features.py
```

Tests use explicitly synthetic fixtures to verify the raw schema, row/index preservation, log transform, fitted projection reuse, separate fitting, serialization, and rejection of invalid inputs. They do not measure fraud-model quality.
