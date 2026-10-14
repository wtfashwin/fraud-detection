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

`notebooks/feature_engineering.ipynb` requires `DATA_CSV` explicitly. It records the exact CSV path and SHA256, declared source kind, and row count, then plots observed class balance, raw/log amount distributions, and training-only PCA variance. The declared source label remains visible on every chart. Missing data produces an explicit error; no generated data is substituted.

The checked-in notebook was executed on the [OpenML 42175 version 1 public benchmark](https://www.openml.org/d/42175). It contains aggregate plots and summary counts only: 284,807 rows, 492 fraud labels, and a 227,845-row training partition. The 10 retained PCA components explain 64.766455% of V1..V28 variance in that training partition. This is representation evidence, not measured fraud-detection improvement.

The CSV SHA256 is `2f9cec6e0d657d30c23eb59750d7976a3498d1c2b8899a5800758c51184d2292`. Aggregate figures are in `docs/figures/benchmark-class-balance.png`, `benchmark-amount-transform.png`, and `benchmark-pca-training-variance.png`. Raw benchmark rows are not copied into the notebook or documentation. Run using the notebook's setup command with the intended dataset to regenerate these results. Synthetic smoke runs check execution mechanics only and must remain labeled synthetic.

## Transformer checks

```sh
python -m pytest -q tests/test_features.py
```

Tests use explicitly synthetic fixtures to verify the raw schema, row/index preservation, log transform, fitted projection reuse, separate fitting, serialization, and rejection of invalid inputs. They do not measure fraud-model quality.
