"""Synthetic feature fixtures test mechanics, not fraud-detection quality."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from training.features import FraudFeatureEngineer
from training.preprocess import FEATURE_NAMES


def raw_features(rows=40, seed=23):
    generator = np.random.default_rng(seed)
    values = generator.normal(size=(rows, 30))
    values[:, 0] = np.arange(rows)
    values[:, -1] = generator.uniform(0, 1000, rows)
    return pd.DataFrame(values, columns=FEATURE_NAMES, index=np.arange(100, 100 + rows))


def test_feature_engineer_preserves_raw_schema_index_and_values():
    raw = raw_features()
    before = raw.copy(deep=True)
    transformer = FraudFeatureEngineer().fit(raw)
    result = transformer.transform(raw)
    assert result.shape == (len(raw), 12)
    assert list(result.columns) == ['Time', 'logAmount'] + [f'PC{i}' for i in range(1, 11)]
    np.testing.assert_array_equal(transformer.feature_names_in_, FEATURE_NAMES)
    assert transformer.n_features_in_ == 30
    pd.testing.assert_index_equal(result.index, raw.index)
    np.testing.assert_array_equal(result['Time'], raw['Time'])
    np.testing.assert_allclose(result['logAmount'], np.log1p(raw['Amount']))
    pd.testing.assert_frame_equal(raw, before)


def test_transform_reorders_named_raw_columns_without_changing_output():
    raw = raw_features()
    transformer = FraudFeatureEngineer().fit(raw)
    pd.testing.assert_frame_equal(transformer.transform(raw), transformer.transform(raw[raw.columns[::-1]]))
    fitted_reordered = FraudFeatureEngineer().fit(raw[raw.columns[::-1]])
    pd.testing.assert_frame_equal(transformer.transform(raw), fitted_reordered.transform(raw))
    np.testing.assert_array_equal(transformer.get_feature_names_out(), transformer.transform(raw).columns)


@pytest.mark.parametrize('phase', ['fit', 'transform'])
@pytest.mark.parametrize('value', [-0.01, np.inf, np.nan])
def test_amount_must_be_nonnegative_and_finite(phase, value):
    raw = raw_features()
    transformer = FraudFeatureEngineer().fit(raw)
    raw.loc[raw.index[0], 'Amount'] = value
    with pytest.raises(ValueError, match='Amount|finite'):
        getattr(transformer, phase)(raw)


def test_pca_fits_only_supplied_training_rows_and_never_refits_on_transform():
    raw = raw_features(60)
    train, holdout = raw.iloc[:40], raw.iloc[40:].copy()
    holdout.loc[:, 'V1'] = holdout['V1'] + 1e6
    first = FraudFeatureEngineer().fit(train)
    components_before = first.pca_.components_.copy()
    means_before = first.pca_.mean_.copy()
    result = first.transform(holdout)
    np.testing.assert_allclose(means_before, train[[f'V{i}' for i in range(1, 29)]].mean())
    np.testing.assert_array_equal(first.pca_.components_, components_before)
    np.testing.assert_array_equal(first.pca_.mean_, means_before)
    pd.testing.assert_frame_equal(result, first.transform(holdout))
    second = clone(first).fit(holdout)
    assert not np.allclose(first.pca_.mean_, second.pca_.mean_)
    assert first.pca_.n_samples_ == 40


def test_feature_engineer_roundtrips_as_part_of_saved_serving_bundle(tmp_path):
    raw = raw_features()
    transformer = FraudFeatureEngineer().fit(raw)
    destination = tmp_path / 'features.joblib'
    joblib.dump(transformer, destination)
    loaded = joblib.load(destination)
    pd.testing.assert_frame_equal(loaded.transform(raw), transformer.transform(raw))


def test_unfitted_transformer_and_incomplete_raw_contract_are_rejected():
    raw = raw_features()
    with pytest.raises(NotFittedError):
        FraudFeatureEngineer().transform(raw)
    for invalid in [raw.drop(columns='V1'), raw.assign(Class=0), raw.to_numpy()]:
        with pytest.raises(ValueError, match='DataFrame|canonical'):
            FraudFeatureEngineer().fit(invalid)


def test_duplicate_columns_and_nonfinite_pca_inputs_are_rejected():
    raw = raw_features()
    duplicated = raw.copy()
    duplicated.columns = ['Time'] + ['V1'] * 28 + ['Amount']
    with pytest.raises(ValueError, match='canonical'):
        FraudFeatureEngineer().fit(duplicated)
    raw.loc[raw.index[0], 'V8'] = np.inf
    with pytest.raises(ValueError, match='finite'):
        FraudFeatureEngineer().fit(raw)


def test_pca_dimension_requires_enough_training_rows():
    with pytest.raises(ValueError, match='rows|components'):
        FraudFeatureEngineer(n_components=10).fit(raw_features(9))
