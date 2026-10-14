import pytest

from scripts.fetch_benchmark import prepare


def test_corrupt_cached_benchmark_is_removed_for_retry(tmp_path):
    source = tmp_path / 'openml-42175.arff'
    source.write_bytes(b'incomplete download')
    with pytest.raises(ValueError, match='checksum mismatch'):
        prepare(tmp_path)
    assert not source.exists()
