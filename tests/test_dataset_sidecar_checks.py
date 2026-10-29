import hashlib
import json
import os
import signal

import pytest


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'source.csv'
    path.write_bytes(b'Time,Amount,Class\n0,1,0\n1,2,1\n')
    return path


def put_sidecar(source, kind):
    if kind == 'absent':
        return
    if kind == 'dangling':
        source.with_suffix('.metadata.json').symlink_to(source.parent / 'missing-sidecar')
        return
    if kind in {'bad-json', 'not-object'}:
        source.with_suffix('.metadata.json').write_text('{' if kind == 'bad-json' else '[]')
        return
    metadata = {'data_kind': 'synthetic', 'source': 'local test'}
    if kind == 'valid':
        metadata['csv_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest().upper()
    elif kind == 'mismatch':
        metadata['csv_sha256'] = '0' * 64
    elif kind == 'null':
        metadata['csv_sha256'] = None
    elif kind == 'number':
        metadata['csv_sha256'] = 7
    elif kind == 'short':
        metadata['csv_sha256'] = 'a' * 63
    elif kind == 'nonhex':
        metadata['csv_sha256'] = 'z' * 64
    elif kind != 'missing-hash':
        raise AssertionError(kind)
    source.with_suffix('.metadata.json').write_text(json.dumps(metadata))


@pytest.mark.parametrize('kind', ['absent', 'missing-hash', 'valid'])
def test_training_accepts_legacy_or_matching_sidecar(source, kind):
    from training.train import load_training_source
    put_sidecar(source, kind)
    frame, provenance = load_training_source(source)
    assert len(frame) == 2
    assert provenance['dataset_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    if kind != 'absent':
        assert provenance['data_kind'] == 'synthetic'


@pytest.mark.parametrize('kind', ['mismatch', 'null', 'number', 'short', 'nonhex', 'dangling', 'bad-json', 'not-object'])
def test_training_rejects_invalid_present_hash_or_dangling_sidecar(source, kind):
    from training.train import load_training_source
    put_sidecar(source, kind)
    with pytest.raises(ValueError, match='sidecar|csv_sha256'):
        load_training_source(source)


class ReachedValidation(Exception):
    pass


@pytest.mark.parametrize('kind', ['absent', 'missing-hash', 'valid'])
def test_evaluation_accepts_legacy_or_matching_sidecar_before_validation(source, monkeypatch, kind):
    from training import evaluate, preprocess
    put_sidecar(source, kind)
    monkeypatch.setenv('EVAL_DATA_CSV', str(source))
    def reached(_frame):
        raise ReachedValidation('sidecar accepted')
    monkeypatch.setattr(preprocess, 'validate_dataset', reached)
    with pytest.raises(ReachedValidation, match='sidecar accepted'):
        evaluate.main()


@pytest.mark.parametrize('kind', ['mismatch', 'null', 'number', 'short', 'nonhex', 'dangling', 'bad-json', 'not-object'])
def test_evaluation_rejects_bad_sidecar_before_validation_or_model(source, monkeypatch, kind):
    from training import evaluate, preprocess
    put_sidecar(source, kind)
    monkeypatch.setenv('EVAL_DATA_CSV', str(source))
    def should_not_run(_frame):
        raise AssertionError('evaluation reached validation before checking the sidecar')
    monkeypatch.setattr(preprocess, 'validate_dataset', should_not_run)
    with pytest.raises(ValueError, match='sidecar|csv_sha256'):
        evaluate.main()


def test_training_compares_sidecar_to_the_exact_csv_snapshot(source, monkeypatch):
    from training import dataset_provenance
    from training.train import load_training_source
    put_sidecar(source, 'valid')
    original = source.read_bytes()
    original_reader = dataset_provenance.read_dataset_sidecar
    def replace_after_snapshot(path, snapshot):
        path.write_bytes(b'Time,Amount,Class\n9,9,1\n')
        return original_reader(path, snapshot)
    monkeypatch.setattr(dataset_provenance, 'read_dataset_sidecar', replace_after_snapshot)
    frame, provenance = load_training_source(source)
    assert len(frame) == 2
    assert provenance['dataset_sha256'] == hashlib.sha256(original).hexdigest()


@pytest.mark.skipif(not hasattr(os, 'mkfifo') or not hasattr(signal, 'setitimer'), reason='requires POSIX FIFO and alarm')
@pytest.mark.parametrize('reader', ['train', 'evaluate'])
def test_fifo_sidecar_is_rejected_without_waiting(source, monkeypatch, reader):
    from training.train import load_training_source
    from training import evaluate
    sidecar = source.with_suffix('.metadata.json')
    os.mkfifo(sidecar)
    monkeypatch.setenv('EVAL_DATA_CSV', str(source))
    def timeout(*_args):
        raise TimeoutError('sidecar FIFO blocked')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 1)
    try:
        with pytest.raises(ValueError, match='regular'):
            if reader == 'train':
                load_training_source(source)
            else:
                evaluate.main()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert sidecar.exists()


def test_regular_sidecar_symlink_remains_supported(source):
    from training.train import load_training_source
    put_sidecar(source, 'valid')
    sidecar = source.with_suffix('.metadata.json')
    linked_file = source.parent / 'linked-metadata.json'
    sidecar.rename(linked_file)
    sidecar.symlink_to(linked_file)
    frame, provenance = load_training_source(source)
    assert len(frame) == 2 and provenance['data_kind'] == 'synthetic'
    assert sidecar.is_symlink()


def test_sidecar_nonblocking_flag_required_only_when_sidecar_exists(source, monkeypatch):
    from training.dataset_provenance import read_dataset_sidecar
    source_bytes = source.read_bytes()
    monkeypatch.delattr(os, 'O_NONBLOCK', raising=False)
    assert read_dataset_sidecar(source, source_bytes) == {}
    source.with_suffix('.metadata.json').write_text('{}')
    with pytest.raises(ValueError, match='O_NONBLOCK'):
        read_dataset_sidecar(source, source_bytes)
