# Bundled demo model

This directory is the single canonical copy of the packaged logistic model, scaler, and ordered feature metadata. The model and scaler were serialized with scikit-learn 1.6.1, which the serving requirements retain.

The original training data, calibration, and held-out fraud performance are unknown. A successful prediction is evidence that inference works, not that the model is suitable for financial decisions.

New training writes a complete separate bundle to `models/trained/` (ignored by Git and the default image). Select it deliberately with `MODEL_DIR`; training output is controlled by `MODEL_OUTPUT_DIR`. Never replace only the scaler or model in a serving bundle.
