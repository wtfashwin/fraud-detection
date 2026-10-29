import gzip
import hashlib
import io
import json
import os
import signal
import tempfile
from unittest.mock import Mock
from urllib.error import URLError
from urllib import request as urllib_request
import zipfile

import numpy as np
import pandas as pd
import pytest

from training import dataset_provenance, preprocess
from scripts.validate_auc import validate_auc


@pytest.fixture
def synthetic_csv(tmp_path):
    rows = np.arange(120)
    data = {'Time': rows.astype(float)}
    data.update({f'V{i}': np.sin(rows / 7.0 + i) for i in range(1, 29)})
    data['Amount'] = rows.astype(float) + 1
    data['Class'] = (rows % 5 == 0).astype(int)
    full = pd.DataFrame(data)
    complete = full.to_csv(index=False).encode()
    partial = full.iloc[:80].to_csv(index=False).encode()
    source = tmp_path / 'source.csv'
    source.write_bytes(partial)
    return {'source': source, 'complete': complete, 'partial': partial}


@pytest.fixture
def paths(tmp_path, monkeypatch, synthetic_csv):
    source = synthetic_csv['source']
    output = tmp_path / 'artifacts' / 'preprocessed.npz'
    models = tmp_path / 'models'
    monkeypatch.setenv('DATA_CSV', str(source))
    monkeypatch.setenv('PREPROCESSED_DATA_PATH', str(output))
    monkeypatch.setenv('PREPROCESS_OUTPUT_DIR', str(models))
    monkeypatch.delenv('MLFLOW_TRACKING_URI', raising=False)
    return {'source': source, 'output': output, 'models': models}


def write_sidecar(source, kind, *, matching_bytes=None, other_bytes=None):
    if kind == 'absent':
        return
    metadata = {'data_kind': 'synthetic'}
    if kind == 'matching':
        metadata['csv_sha256'] = hashlib.sha256(matching_bytes).hexdigest()
    elif kind == 'mismatch':
        metadata['csv_sha256'] = hashlib.sha256(other_bytes).hexdigest()
    elif kind == 'null':
        metadata['csv_sha256'] = None
    elif kind == 'malformed':
        metadata['csv_sha256'] = 'not-a-sha256'
    elif kind != 'hashless':
        raise AssertionError(kind)
    source.with_suffix('.metadata.json').write_text(json.dumps(metadata))


class SyntheticModel:
    named_steps = {'scaler': object()}
    classes_ = np.array([0, 1])

    def __init__(self):
        self.frames = []

    def predict_proba(self, frame):
        self.frames.append(frame.copy())
        return np.tile([0.5, 0.5], (len(frame), 1))


def install_model(monkeypatch):
    import mlflow.sklearn
    model = SyntheticModel()
    loader = Mock(return_value=model)
    monkeypatch.setattr(mlflow.sklearn, 'load_model', loader)
    return model, loader


def invoke(reader, paths):
    if reader == 'preprocess':
        preprocess.main()
    else:
        validate_auc('synthetic-model', threshold=0.0, data_path=paths['source'])


@pytest.mark.skipif(not hasattr(os, 'O_NONBLOCK'), reason='present sidecars require O_NONBLOCK')
@pytest.mark.parametrize('reader', ['preprocess', 'auc'])
@pytest.mark.parametrize('kind', ['mismatch', 'null', 'malformed'])
def test_bad_sidecar_stops_before_artifacts_or_model(tmp_path, monkeypatch, synthetic_csv, paths, reader, kind):
    write_sidecar(paths['source'], kind, matching_bytes=synthetic_csv['partial'], other_bytes=synthetic_csv['complete'])
    model, loader = install_model(monkeypatch)
    calls = []
    original_prepare = preprocess.prepare_data
    def capture_prepare(frame):
        calls.append(True)
        return original_prepare(frame)
    monkeypatch.setattr(preprocess, 'prepare_data', capture_prepare)
    failure = None
    try:
        invoke(reader, paths)
    except ValueError as exc:
        failure = exc
    assert not paths['output'].exists() and not paths['models'].exists()
    assert not calls and not loader.called and not model.frames
    assert failure is not None and ('csv_sha256' in str(failure) or 'sidecar' in str(failure))


@pytest.mark.parametrize('reader', ['preprocess', 'auc'])
@pytest.mark.parametrize('kind', ['absent', 'hashless', 'matching'])
def test_missing_hashless_and_matching_sidecars_work(monkeypatch, synthetic_csv, paths, reader, kind):
    if kind != 'absent' and not hasattr(os, 'O_NONBLOCK'):
        pytest.skip('present sidecars require O_NONBLOCK')
    write_sidecar(paths['source'], kind, matching_bytes=synthetic_csv['partial'])
    model, loader = install_model(monkeypatch)
    invoke(reader, paths)
    if reader == 'preprocess':
        assert paths['output'].is_file() and (paths['models'] / 'scaler.joblib').is_file()
    else:
        assert loader.call_count == 1 and len(model.frames) == 1 and len(model.frames[0]) == 80


@pytest.mark.skipif(not hasattr(os, 'O_NONBLOCK'), reason='present sidecars require O_NONBLOCK')
@pytest.mark.parametrize('reader', ['preprocess', 'auc'])
def test_parsing_uses_the_checked_byte_snapshot(monkeypatch, synthetic_csv, paths, reader):
    write_sidecar(paths['source'], 'matching', matching_bytes=synthetic_csv['partial'])
    expected = pd.read_csv(io.BytesIO(synthetic_csv['partial']))
    checked = []
    original_reader = dataset_provenance.read_dataset_sidecar
    def replace_after_check(source, snapshot):
        result = original_reader(source, snapshot)
        checked.append(hashlib.sha256(snapshot).hexdigest())
        paths['source'].write_bytes(synthetic_csv['complete'])
        return result
    monkeypatch.setattr(dataset_provenance, 'read_dataset_sidecar', replace_after_check)
    seen = []
    original_prepare = preprocess.prepare_data
    def capture_prepare(frame):
        seen.append(frame.copy())
        return original_prepare(frame)
    monkeypatch.setattr(preprocess, 'prepare_data', capture_prepare)
    model, loader = install_model(monkeypatch)
    invoke(reader, paths)
    assert checked == [hashlib.sha256(synthetic_csv['partial']).hexdigest()]
    assert paths['source'].read_bytes() == synthetic_csv['complete']
    if reader == 'preprocess':
        assert len(seen) == 1 and seen[0].equals(expected)
    else:
        assert loader.call_count == 1 and len(model.frames) == 1
        assert model.frames[0].equals(expected[preprocess.FEATURE_NAMES].astype(float))


@pytest.mark.skipif(not hasattr(os, 'mkfifo') or not hasattr(signal, 'setitimer'), reason='requires POSIX FIFO and alarm')
@pytest.mark.parametrize('reader', ['preprocess', 'auc'])
def test_fifo_sidecar_fails_without_processing(monkeypatch, paths, reader):
    if not hasattr(os, 'O_NONBLOCK'):
        pytest.skip('present sidecars require O_NONBLOCK')
    os.mkfifo(paths['source'].with_suffix('.metadata.json'))
    model, loader = install_model(monkeypatch)
    def timeout(*_args):
        raise TimeoutError('sidecar FIFO blocked')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 1)
    failure = None
    try:
        try:
            invoke(reader, paths)
        except ValueError as exc:
            failure = exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert not paths['output'].exists() and not paths['models'].exists()
    assert not loader.called and not model.frames
    assert failure is not None and 'regular' in str(failure)


@pytest.mark.parametrize('reader', ['preprocess', 'auc'])
@pytest.mark.parametrize('kind', ['absent', 'matching', 'mismatch'])
@pytest.mark.parametrize('format', ['gzip', 'zip'])
def test_compressed_local_csv_uses_raw_file_hash_and_pandas_inference(monkeypatch, synthetic_csv, paths, reader, kind, format):
    if kind != 'absent' and not hasattr(os, 'O_NONBLOCK'):
        pytest.skip('present sidecars require O_NONBLOCK')
    compressed = paths['source'].with_suffix('.csv.gz' if format == 'gzip' else '.csv.zip')
    if format == 'gzip':
        encoded = gzip.compress(synthetic_csv['partial'], mtime=0)
    else:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('synthetic.csv', synthetic_csv['partial'])
        encoded = buffer.getvalue()
    compressed.write_bytes(encoded)
    write_sidecar(compressed, kind, matching_bytes=encoded, other_bytes=synthetic_csv['complete'])
    paths['source'] = compressed
    monkeypatch.setenv('DATA_CSV', str(compressed))
    model, loader = install_model(monkeypatch)
    failure = None
    try:
        invoke(reader, paths)
    except ValueError as exc:
        failure = exc
    if kind == 'mismatch':
        assert failure is not None and 'csv_sha256' in str(failure)
        assert not paths['output'].exists() and not paths['models'].exists()
        assert not loader.called and not model.frames
    else:
        assert failure is None
        if reader == 'preprocess':
            assert paths['output'].exists()
        else:
            assert loader.call_count == 1 and len(model.frames[0]) == 80


@pytest.mark.parametrize('kind', ['binary-stream', 'text-stream', 'file-url'])
def test_auc_keeps_stream_and_file_url_inputs(monkeypatch, synthetic_csv, paths, kind):
    model, loader = install_model(monkeypatch)
    if kind == 'binary-stream':
        data = io.BytesIO(synthetic_csv['partial'])
    elif kind == 'text-stream':
        data = io.StringIO(synthetic_csv['partial'].decode())
    else:
        data = paths['source'].as_uri()
    validate_auc('synthetic-model', threshold=0.0, data_path=data)
    assert loader.call_count == 1 and len(model.frames) == 1 and len(model.frames[0]) == 80
    if hasattr(data, 'closed'):
        assert not data.closed


@pytest.mark.parametrize('compressed', [False, True])
def test_auc_uses_an_open_pathlike_stream_without_reopening(monkeypatch, synthetic_csv, paths, compressed):
    model, loader = install_model(monkeypatch)
    class PathLikeStream(io.BytesIO, os.PathLike):
        def __fspath__(self):
            suffix = '.csv.gz' if compressed else '.csv'
            return str(paths['source'].parent / ('nonexistent' + suffix))
    payload = gzip.compress(synthetic_csv['partial'], mtime=0) if compressed else synthetic_csv['partial']
    stream = PathLikeStream(payload)
    validate_auc('synthetic-model', threshold=0.0, data_path=stream)
    assert loader.call_count == 1 and len(model.frames) == 1 and len(model.frames[0]) == 80
    assert not stream.closed


@pytest.mark.skipif(not hasattr(os, 'O_NONBLOCK'), reason='present sidecars require O_NONBLOCK')
def test_auc_file_url_checks_local_sidecar(monkeypatch, synthetic_csv, paths):
    write_sidecar(paths['source'], 'mismatch', other_bytes=synthetic_csv['complete'])
    model, loader = install_model(monkeypatch)
    failure = None
    try:
        validate_auc('synthetic-model', threshold=0.0, data_path=paths['source'].as_uri())
    except ValueError as exc:
        failure = exc
    assert not loader.called and not model.frames
    assert failure is not None and 'csv_sha256' in str(failure)


@pytest.mark.skipif(not hasattr(os, 'O_NONBLOCK'), reason='present sidecars require O_NONBLOCK')
@pytest.mark.parametrize('kind', ['matching', 'mismatch'])
def test_auc_compressed_file_url_with_spaces_checks_sidecar(monkeypatch, synthetic_csv, paths, kind):
    source = paths['source'].parent / 'synthetic data.csv.gz'
    encoded = gzip.compress(synthetic_csv['partial'], mtime=0)
    source.write_bytes(encoded)
    write_sidecar(source, kind, matching_bytes=encoded, other_bytes=synthetic_csv['complete'])
    model, loader = install_model(monkeypatch)
    failure = None
    try:
        validate_auc('synthetic-model', threshold=0.0, data_path=source.as_uri())
    except ValueError as exc:
        failure = exc
    if kind == 'mismatch':
        assert failure is not None and 'csv_sha256' in str(failure)
        assert not loader.called and not model.frames
    else:
        assert failure is None and loader.call_count == 1 and len(model.frames[0]) == 80


@pytest.mark.skipif(os.name == 'nt' or not hasattr(os, 'O_NONBLOCK'), reason='requires POSIX file URL and sidecar flags')
@pytest.mark.parametrize('suffix', ['#section', '?case=1', '?case=1#section'])
@pytest.mark.parametrize('kind', ['matching', 'mismatch'])
def test_auc_file_url_query_and_fragment_checks_actual_local_file(monkeypatch, synthetic_csv, paths, suffix, kind):
    base = paths['source'].parent / 'synthetic data.csv'
    actual = base.parent / (base.name + ('?case=1' if suffix.startswith('?') else ''))
    actual.write_bytes(synthetic_csv['partial'])
    write_sidecar(actual, kind, matching_bytes=synthetic_csv['partial'], other_bytes=synthetic_csv['complete'])
    model, loader = install_model(monkeypatch)
    failure = None
    try:
        validate_auc('synthetic-model', threshold=0.0, data_path=base.as_uri() + suffix)
    except ValueError as exc:
        failure = exc
    if kind == 'mismatch':
        assert failure is not None and 'csv_sha256' in str(failure)
        assert not loader.called and not model.frames
    else:
        assert failure is None and loader.call_count == 1 and len(model.frames[0]) == 80


@pytest.mark.skipif(not hasattr(os, 'O_NONBLOCK'), reason='present sidecars require O_NONBLOCK')
def test_auc_file_url_decodes_escaped_question_mark_in_filename(monkeypatch, synthetic_csv, paths):
    source = paths['source'].parent / 'synthetic? data.csv'
    source.write_bytes(synthetic_csv['partial'])
    write_sidecar(source, 'mismatch', other_bytes=synthetic_csv['complete'])
    model, loader = install_model(monkeypatch)
    with pytest.raises(ValueError, match='csv_sha256'):
        validate_auc('synthetic-model', threshold=0.0, data_path=source.as_uri())
    assert not loader.called and not model.frames


@pytest.mark.skipif(not hasattr(os, 'O_NONBLOCK'), reason='present sidecars require O_NONBLOCK')
@pytest.mark.parametrize('kind', ['matching', 'mismatch'])
def test_auc_local_file_url_alias_checks_opened_file_sidecar(monkeypatch, synthetic_csv, paths, kind):
    write_sidecar(paths['source'], kind, matching_bytes=synthetic_csv['partial'], other_bytes=synthetic_csv['complete'])
    model, loader = install_model(monkeypatch)
    monkeypatch.setattr(urllib_request, '_safe_gethostbyname', lambda _host: '127.0.0.1')
    monkeypatch.setattr(urllib_request.FileHandler, 'get_names', lambda _self: ('127.0.0.1',))
    url = paths['source'].as_uri().replace('file:///', 'file://127.0.0.1/')
    if kind == 'mismatch':
        with pytest.raises(ValueError, match='csv_sha256'):
            validate_auc('synthetic-model', threshold=0.0, data_path=url)
        assert not loader.called and not model.frames
    else:
        validate_auc('synthetic-model', threshold=0.0, data_path=url)
        assert loader.call_count == 1 and len(model.frames[0]) == 80


def test_auc_nonlocal_file_url_keeps_urllib_rejection(monkeypatch, paths):
    model, loader = install_model(monkeypatch)
    monkeypatch.setattr(urllib_request, '_safe_gethostbyname', lambda _host: None)
    monkeypatch.setattr(urllib_request.FileHandler, 'get_names', lambda _self: ('127.0.0.1',))
    url = paths['source'].as_uri().replace('file:///', 'file://nonlocal.invalid/')
    with pytest.raises(URLError):
        validate_auc('synthetic-model', threshold=0.0, data_path=url)
    assert not loader.called and not model.frames


def test_auc_file_url_without_identifiable_opened_path_fails_closed(monkeypatch, synthetic_csv, paths):
    model, loader = install_model(monkeypatch)
    class FakeResponse(io.BytesIO):
        headers = {'Content-Encoding': 'identity'}
    monkeypatch.setattr('pandas.io.common.urlopen', lambda _request: FakeResponse(synthetic_csv['partial']))
    with pytest.raises(ValueError, match='opened file path'):
        validate_auc('synthetic-model', threshold=0.0, data_path=paths['source'].as_uri())
    assert not loader.called and not model.frames


@pytest.mark.skipif(not hasattr(os, 'O_NONBLOCK'), reason='present sidecars require O_NONBLOCK')
@pytest.mark.parametrize('kind', ['matching', 'mismatch'])
def test_auc_localhost_file_url_checks_sidecar(monkeypatch, synthetic_csv, paths, kind):
    write_sidecar(paths['source'], kind, matching_bytes=synthetic_csv['partial'], other_bytes=synthetic_csv['complete'])
    model, loader = install_model(monkeypatch)
    url = paths['source'].as_uri().replace('file:///', 'file://localhost/')
    if kind == 'mismatch':
        with pytest.raises(ValueError, match='csv_sha256'):
            validate_auc('synthetic-model', threshold=0.0, data_path=url)
        assert not loader.called and not model.frames
    else:
        validate_auc('synthetic-model', threshold=0.0, data_path=url)
        assert loader.call_count == 1 and len(model.frames[0]) == 80


@pytest.mark.parametrize('factory', [tempfile.NamedTemporaryFile, tempfile.SpooledTemporaryFile])
def test_auc_keeps_standard_library_text_streams_open(monkeypatch, synthetic_csv, paths, factory):
    model, loader = install_model(monkeypatch)
    with factory(mode='w+t', encoding='utf-8', dir=paths['source'].parent) as stream:
        stream.write('\ufeff' + synthetic_csv['partial'].decode())
        stream.seek(0)
        validate_auc('synthetic-model', threshold=0.0, data_path=stream)
        assert not stream.closed and loader.call_count == 1 and len(model.frames[0]) == 80


@pytest.mark.parametrize('factory', [tempfile.NamedTemporaryFile, tempfile.SpooledTemporaryFile])
def test_text_stream_non_ascii_matches_pandas_and_stays_open(paths, factory):
    from training.csv_snapshot import read_checked_csv
    with factory(mode='w+t', encoding='utf-8', dir=paths['source'].parent) as stream:
        stream.write('name\ncaf\u00e9\n')
        stream.seek(0)
        expected = pd.read_csv(stream)
        stream.seek(0)
        actual = read_checked_csv(stream)
        assert actual.equals(expected) and not stream.closed


@pytest.mark.parametrize('reader', ['preprocess', 'auc'])
@pytest.mark.parametrize('kind', ['absent', 'matching'])
def test_missing_nonblocking_flag_only_rejects_present_sidecar(monkeypatch, synthetic_csv, paths, reader, kind):
    write_sidecar(paths['source'], kind, matching_bytes=synthetic_csv['partial'])
    model, loader = install_model(monkeypatch)
    monkeypatch.delattr(os, 'O_NONBLOCK', raising=False)
    failure = None
    try:
        invoke(reader, paths)
    except ValueError as exc:
        failure = exc
    if kind == 'matching':
        assert failure is not None and 'O_NONBLOCK' in str(failure)
        assert not paths['output'].exists() and not paths['models'].exists()
        assert not loader.called and not model.frames
    else:
        assert failure is None
        if reader == 'preprocess':
            assert paths['output'].is_file()
        else:
            assert loader.call_count == 1 and len(model.frames) == 1


@pytest.mark.parametrize('suffix,gzip_header', [('csv', False), ('csv', True), ('csv.gz', False), ('csv.gz', True)])
def test_auc_keeps_http_url_inputs_without_network(monkeypatch, synthetic_csv, paths, suffix, gzip_header):
    model, loader = install_model(monkeypatch)
    payload = gzip.compress(synthetic_csv['partial'], mtime=0) if gzip_header or suffix.endswith('.gz') else synthetic_csv['partial']
    class FakeResponse(io.BytesIO):
        headers = {'Content-Encoding': 'gzip' if gzip_header else 'identity'}
    opened = []
    def fake_urlopen(request):
        opened.append(request.full_url)
        return FakeResponse(payload)
    monkeypatch.setattr('pandas.io.common.urlopen', fake_urlopen)
    url = f'https://example.invalid/fixture.{suffix}'
    validate_auc('synthetic-model', threshold=0.0, data_path=url)
    assert opened == [url]
    assert loader.call_count == 1 and len(model.frames) == 1 and len(model.frames[0]) == 80
