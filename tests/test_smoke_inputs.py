import pytest

from scripts.smoke_test import csv_features


def test_csv_input_uses_model_order_and_ignores_label(tmp_path):
    path = tmp_path / 'sample.csv'
    path.write_text('Amount,Class,V1\n12.5,1,-2\n')
    assert csv_features(path, ['V1', 'Amount']) == [-2.0, 12.5]


@pytest.mark.parametrize('content', ['V1\n', 'V1\nnan\n', 'Amount\n1\n', 'V1\nno\n'])
def test_csv_input_rejects_missing_or_invalid_features(tmp_path, content):
    path = tmp_path / 'invalid.csv'
    path.write_text(content)
    with pytest.raises(ValueError):
        csv_features(path, ['V1'])
