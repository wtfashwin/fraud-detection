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


@pytest.mark.parametrize('content', [
    'V1,Amount,Amount\n-2,12.5,999\n',
    'V1,V1,Amount\n-2,999,12.5\n',
    'V1,Amount,Amount\n',
])
def test_csv_input_rejects_duplicate_headers_before_using_row_values(tmp_path, content):
    path = tmp_path / 'duplicate.csv'
    path.write_text(content)
    with pytest.raises(ValueError, match='unique|duplicate'):
        csv_features(path, ['V1', 'Amount'])


def test_canonical_csv_input_rejects_second_amount_column(tmp_path):
    names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    path = tmp_path / 'duplicate-canonical.csv'
    path.write_text(','.join(names + ['Amount']) + '\n' + ','.join(['0'] * 29 + ['100', '999']) + '\n')
    with pytest.raises(ValueError, match='unique|duplicate'):
        csv_features(path, names)
