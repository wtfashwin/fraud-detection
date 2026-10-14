# Prediction input drift

The local alert fires when any raw feature's two-sample KS **statistic** exceeds 0.05 for 30 seconds. The number 0.05 is the configured statistic threshold, not the p-value cutoff. `/health` also reports the p-value for context.

This compares input distributions. It does not establish an AUC decrease, fraud increase, or a need to block predictions. Measuring live AUC requires reliable outcome labels for the evaluated cohort.

## Check the evidence

1. Read `/health` and confirm the baseline fingerprint, training-only source, sample count, and affected features. It remains HTTP 200 because distribution drift is not a process failure. `/ready` remains the dependency readiness check.
2. Inspect `fraud_drift_ks{feature="V1"}`, `fraud_drift_available`, and `fraud_drift_window_samples` in Prometheus or the **Fraud input distribution** Grafana dashboard. Metrics are calculated at scrape time from a shared Redis list, so multiple API workers see the same window.
3. Compare the active model bundle with its `drift_baseline.json`. The reference contains at most 512 raw training rows, selected before preprocessing or SMOTE; holdout rows must be excluded. Dataset SHA256 and data kind identify the source. The default legacy model has no known training baseline and honestly reports `UNAVAILABLE`.
4. Check for changes to input order, units, source systems, model configuration, or the feature pipeline. Raw inputs stay in canonical 30-feature order even when PCA produces 12 explanation features. Time naturally changes over a deployment; an alarm on Time alone needs that context.
5. If labelled outcomes are available, evaluate a separate, appropriately labelled cohort using `python -m training.evaluate`. Compare like-for-like cohorts and report uncertainty and fraud prevalence. Do not interpret the KS statistic as AUC.

## Window and unavailable states

The live window contains at most `DRIFT_MAX_SAMPLES` successfully accepted predictions (default 512), shared across API workers and bounded transactionally in Redis. It expires after one hour without accepted traffic. `DRIFT_MIN_SAMPLES` defaults to 20 and cannot be lower than 20. A model baseline fingerprint creates a distinct window.

- `UNAVAILABLE`: no usable training baseline, insufficient reference rows, or invalid configuration. No KS metric series is emitted.
- `WARMING_UP`: a valid reference exists but fewer than the required live observations are available. No KS metric series is emitted.
- `OK`: enough observations exist and every KS statistic is at or below the configured threshold.
- `DRIFT`: at least one statistic exceeds the threshold. Predictions continue.
- `ERROR`: Redis or stored observations are unavailable or invalid. No fabricated zero statistics are emitted.

Redis monitoring failures do not change a completed scoring response. The broker and persistence requirements for accepting a prediction remain enforced.

## Baseline configuration

Local complete bundles use `drift_baseline.json` beside the model artifacts. `DRIFT_BASELINE_PATH` explicitly overrides that file. Remote MLflow models are opt-in through `MLFLOW_MODEL_URI`; their baseline is also opt-in through `MLFLOW_DRIFT_BASELINE_URI` pointing to the logged `drift/drift_baseline.json` artifact, or an explicitly supplied local baseline path. A remote model never silently uses an unrelated local baseline.

Training saves the baseline in its bundle and logs it with the MLflow training run when tracking is enabled. The reference fingerprint protects against accidental content changes; it is not a proof that the source dataset or split was correct. Training and holdout separation must be verified by the training tests and provenance.

## Validate the local alert

Run `promtool test rules monitoring/tests/drift_rules.test.yml` from the repository root to verify sustained drift, the exact threshold boundary, and missing-data behavior against controlled synthetic metric series. Then load the Grafana provisioning files and inspect the managed rule `fraud-input-distribution-drift` through Grafana's local API/UI. A rule file passing a syntax check is not proof that Grafana evaluated it or that real traffic drifted.

No external notification destination is configured. Alert validation is local.

References: [SciPy two-sample KS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html), [Grafana file provisioning](https://grafana.com/docs/grafana/latest/alerting/set-up/provision-alerting-resources/file-provisioning/).
