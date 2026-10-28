import hashlib
import io
import os
from pathlib import Path
import signal
import stat

import numpy as np
import pytest

from scripts import fetch_benchmark


@pytest.fixture(scope='module')
def synthetic_rows():
    dtype = [(name, 'u1') for name in fetch_benchmark.FEATURES + ['Class']]
    rows = np.zeros(284807, dtype=dtype)
    rows['Class'][:492] = 1
    return rows


@pytest.fixture
def cached_source(tmp_path, monkeypatch, synthetic_rows):
    source = tmp_path / 'openml-42175.arff'
    source.write_bytes(b'synthetic benchmark source')
    monkeypatch.setattr(fetch_benchmark, 'SHA256', hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(fetch_benchmark.arff, 'loadarff', lambda *_args, **_kwargs: (synthetic_rows, None))
    def small_csv(_frame, output, **_kwargs):
        if hasattr(output, 'write'):
            output.write('synthetic csv data\n')
        else:
            Path(output).write_bytes(b'synthetic csv data\n')
    monkeypatch.setattr(fetch_benchmark.pd.DataFrame, 'to_csv', small_csv)
    return source


def test_bad_cached_source_is_preserved(tmp_path):
    source = tmp_path / 'openml-42175.arff'
    source.write_bytes(b'user-owned invalid cache')
    with pytest.raises(ValueError, match='checksum mismatch'):
        fetch_benchmark.prepare(tmp_path)
    assert source.read_bytes() == b'user-owned invalid cache'


def test_old_partial_is_preserved_on_download_failure(tmp_path, monkeypatch):
    partial = tmp_path / 'openml-42175.arff.part'
    partial.write_bytes(b'user-owned partial')
    def fail(*_args, **_kwargs):
        raise OSError('synthetic connection failure')
    monkeypatch.setattr(fetch_benchmark.urllib.request, 'urlopen', fail)
    with pytest.raises(OSError, match='synthetic connection failure'):
        fetch_benchmark.prepare(tmp_path)
    assert partial.read_bytes() == b'user-owned partial'


def test_old_partial_is_preserved_after_successful_download(tmp_path, monkeypatch, synthetic_rows):
    partial = tmp_path / 'openml-42175.arff.part'
    partial.write_bytes(b'user-owned partial')
    payload = b'synthetic source'
    monkeypatch.setattr(fetch_benchmark, 'SHA256', hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(fetch_benchmark.urllib.request, 'urlopen', lambda *_a, **_k: io.BytesIO(payload))
    monkeypatch.setattr(fetch_benchmark.arff, 'loadarff', lambda *_a, **_k: (synthetic_rows, None))
    def small_csv(_frame, output, **_kwargs):
        output.write('synthetic csv data\n')
    monkeypatch.setattr(fetch_benchmark.pd.DataFrame, 'to_csv', small_csv)
    fetch_benchmark.prepare(tmp_path)
    assert partial.read_bytes() == b'user-owned partial'
    assert (tmp_path / 'openml-42175.arff').read_bytes() == payload


def test_source_created_during_download_is_not_overwritten(tmp_path, monkeypatch, synthetic_rows):
    source = tmp_path / 'openml-42175.arff'
    payload = b'synthetic source'
    monkeypatch.setattr(fetch_benchmark, 'SHA256', hashlib.sha256(payload).hexdigest())
    def download(*_args, **_kwargs):
        source.write_bytes(b'created by another process')
        return io.BytesIO(payload)
    monkeypatch.setattr(fetch_benchmark.urllib.request, 'urlopen', download)
    monkeypatch.setattr(fetch_benchmark.arff, 'loadarff', lambda *_args, **_kwargs: (synthetic_rows, None))
    try:
        fetch_benchmark.prepare(tmp_path)
    except (OSError, ValueError):
        pass
    assert source.read_bytes() == b'created by another process'


@pytest.mark.parametrize('name', ['creditcard.csv', 'creditcard.metadata.json'])
def test_existing_outputs_are_preserved(tmp_path, cached_source, name):
    target = tmp_path / name
    target.write_bytes(b'user-owned output')
    failure = None
    try:
        fetch_benchmark.prepare(tmp_path)
    except FileExistsError as exc:
        failure = exc
    assert target.read_bytes() == b'user-owned output'
    assert failure is not None


@pytest.mark.parametrize('name', ['creditcard.csv', 'creditcard.metadata.json'])
def test_existing_output_symlink_and_target_are_preserved(tmp_path, cached_source, name):
    other = tmp_path / 'unrelated'
    other.write_bytes(b'user-owned target')
    link = tmp_path / name
    link.symlink_to(other)
    with pytest.raises(FileExistsError):
        fetch_benchmark.prepare(tmp_path)
    assert link.is_symlink() and other.read_bytes() == b'user-owned target'


@pytest.mark.parametrize('name', ['openml-42175.arff', 'openml-42175.arff.part', 'creditcard.csv', 'creditcard.metadata.json'])
def test_dangling_symlink_is_preserved(tmp_path, monkeypatch, synthetic_rows, name):
    path = tmp_path / name
    other = tmp_path / 'missing-target'
    path.symlink_to(other)
    payload = b'synthetic source'
    monkeypatch.setattr(fetch_benchmark, 'SHA256', hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(fetch_benchmark.urllib.request, 'urlopen', lambda *_a, **_k: io.BytesIO(payload))
    monkeypatch.setattr(fetch_benchmark.arff, 'loadarff', lambda *_a, **_k: (synthetic_rows, None))
    try:
        fetch_benchmark.prepare(tmp_path)
    except (OSError, ValueError):
        pass
    assert path.is_symlink()
    assert not other.exists()


def test_valid_source_symlink_is_rejected_without_following(tmp_path, monkeypatch, synthetic_rows):
    other = tmp_path / 'user-source'
    other.write_bytes(b'synthetic source')
    source = tmp_path / 'openml-42175.arff'
    source.symlink_to(other)
    monkeypatch.setattr(fetch_benchmark, 'SHA256', hashlib.sha256(other.read_bytes()).hexdigest())
    monkeypatch.setattr(fetch_benchmark.arff, 'loadarff', lambda *_a, **_k: (synthetic_rows, None))
    with pytest.raises(OSError):
        fetch_benchmark.prepare(tmp_path)
    assert source.is_symlink() and other.read_bytes() == b'synthetic source'


@pytest.mark.parametrize('flag', ['O_NOFOLLOW', 'O_NONBLOCK'])
def test_missing_safe_open_flag_fails_closed(tmp_path, monkeypatch, cached_source, flag):
    monkeypatch.delattr(os, flag, raising=False)
    with pytest.raises(OSError, match='requires O_NOFOLLOW and O_NONBLOCK'):
        fetch_benchmark.prepare(tmp_path)
    assert cached_source.read_bytes() == b'synthetic benchmark source'
    assert not (tmp_path / 'creditcard.csv').exists()


def test_source_descriptor_closes_when_stat_fails(tmp_path, monkeypatch, cached_source):
    original_open = os.open
    opened = []
    def record_open(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        opened.append(descriptor)
        return descriptor
    def fail_stat(_descriptor):
        raise OSError('synthetic stat failure')
    monkeypatch.setattr(os, 'open', record_open)
    monkeypatch.setattr(os, 'fstat', fail_stat)
    with pytest.raises(OSError, match='synthetic stat failure'):
        fetch_benchmark.open_existing_source(cached_source)
    monkeypatch.undo()
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_publication_descriptor_closes_when_fdopen_fails(tmp_path, monkeypatch):
    target = tmp_path / 'output'
    original_open = os.open
    opened = []
    def record_open(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        opened.append(descriptor)
        return descriptor
    def fail_fdopen(*_args, **_kwargs):
        raise OSError('synthetic fdopen failure')
    monkeypatch.setattr(os, 'open', record_open)
    monkeypatch.setattr(os, 'fdopen', fail_fdopen)
    with pytest.raises(OSError, match='synthetic fdopen failure') as failure:
        fetch_benchmark.publish_exclusive(io.BytesIO(b'output'), target)
    monkeypatch.undo()
    with pytest.raises(OSError):
        os.fstat(opened[0])
    assert target.read_bytes() == b''
    assert any(str(target) in note for note in getattr(failure.value, '__notes__', []))


@pytest.mark.skipif(not hasattr(os, 'mkfifo') or not hasattr(signal, 'setitimer'), reason='requires POSIX FIFO and alarm')
def test_fifo_source_is_rejected_without_waiting(tmp_path):
    source = tmp_path / 'openml-42175.arff'
    os.mkfifo(source)
    def timeout(*_args):
        raise TimeoutError('source FIFO blocked')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 1)
    try:
        with pytest.raises((ValueError, OSError), match='regular'):
            fetch_benchmark.prepare(tmp_path)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert stat.S_ISFIFO(source.lstat().st_mode)


def test_parse_uses_verified_snapshot(tmp_path, monkeypatch, cached_source, synthetic_rows):
    original = cached_source.read_bytes()
    seen = []
    def loader(input_file):
        cached_source.write_bytes(b'changed by another process')
        value = input_file.read() if hasattr(input_file, 'read') else Path(input_file).read_bytes()
        seen.append(value.encode() if isinstance(value, str) else value)
        return synthetic_rows, None
    monkeypatch.setattr(fetch_benchmark.arff, 'loadarff', loader)
    result = fetch_benchmark.prepare(tmp_path)
    assert seen == [original]
    assert result['source_sha256'] == hashlib.sha256(original).hexdigest()


def test_repeat_preserves_previous_output(tmp_path, cached_source):
    fetch_benchmark.prepare(tmp_path)
    csv_path, metadata_path = tmp_path / 'creditcard.csv', tmp_path / 'creditcard.metadata.json'
    before = csv_path.read_bytes(), metadata_path.read_bytes()
    with pytest.raises(FileExistsError):
        fetch_benchmark.prepare(tmp_path)
    assert (csv_path.read_bytes(), metadata_path.read_bytes()) == before


def test_new_outputs_keep_normal_umask_permissions(tmp_path, cached_source):
    control = tmp_path / 'mode-control'
    control.write_bytes(b'control')
    expected_mode = stat.S_IMODE(control.stat().st_mode)
    fetch_benchmark.prepare(tmp_path)
    assert stat.S_IMODE((tmp_path / 'creditcard.csv').stat().st_mode) == expected_mode
    assert stat.S_IMODE((tmp_path / 'creditcard.metadata.json').stat().st_mode) == expected_mode


@pytest.mark.parametrize('cached', [False, True])
def test_source_reads_are_bounded(tmp_path, monkeypatch, synthetic_rows, cached):
    payload = b'a' * 17
    source = tmp_path / 'openml-42175.arff'
    if cached:
        source.write_bytes(payload)
    else:
        monkeypatch.setattr(fetch_benchmark.urllib.request, 'urlopen', lambda *_a, **_k: io.BytesIO(payload))
    monkeypatch.setattr(fetch_benchmark, 'MAX_SOURCE_BYTES', 16, raising=False)
    monkeypatch.setattr(fetch_benchmark, 'SHA256', hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(fetch_benchmark.arff, 'loadarff', lambda *_a, **_k: (synthetic_rows, None))
    failure = None
    try:
        fetch_benchmark.prepare(tmp_path)
    except ValueError as exc:
        failure = exc
    assert failure is not None and 'limit' in str(failure).lower()
    assert not (tmp_path / 'creditcard.csv').exists()
    if cached:
        assert source.read_bytes() == payload
    else:
        assert not source.exists()


def test_generated_csv_is_bounded_before_publication(tmp_path, monkeypatch, cached_source):
    monkeypatch.setattr(fetch_benchmark, 'MAX_CSV_BYTES', 10, raising=False)
    with pytest.raises(ValueError, match='limit'):
        fetch_benchmark.prepare(tmp_path)
    assert not (tmp_path / 'creditcard.csv').exists()
    assert not (tmp_path / 'creditcard.metadata.json').exists()


def test_interrupted_source_publication_reports_partial_target(tmp_path, monkeypatch):
    payload = b'synthetic source'
    source = tmp_path / 'openml-42175.arff'
    monkeypatch.setattr(fetch_benchmark, 'SHA256', hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(fetch_benchmark.urllib.request, 'urlopen', lambda *_a, **_k: io.BytesIO(payload))
    def fail_copy(_source, destination, *_args, **_kwargs):
        destination.write(b'partial')
        destination.flush()
        raise OSError('synthetic source write failure')
    monkeypatch.setattr(fetch_benchmark.shutil, 'copyfileobj', fail_copy)
    failure = None
    try:
        fetch_benchmark.prepare(tmp_path)
    except OSError as exc:
        failure = exc
    assert failure is not None and 'synthetic source write failure' in str(failure)
    assert source.read_bytes() == b'partial'
    assert any(str(source) in note for note in getattr(failure, '__notes__', []))


@pytest.mark.parametrize('stage', ['parser', 'schema', 'csv'])
def test_conversion_failure_reports_newly_published_source(tmp_path, monkeypatch, synthetic_rows, stage):
    payload = b'synthetic source'
    source = tmp_path / 'openml-42175.arff'
    monkeypatch.setattr(fetch_benchmark, 'SHA256', hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(fetch_benchmark.urllib.request, 'urlopen', lambda *_a, **_k: io.BytesIO(payload))
    def parser(_stream):
        if stage == 'parser':
            raise ValueError('synthetic parser failure')
        return (synthetic_rows[:2] if stage == 'schema' else synthetic_rows), None
    def failed_csv(*_args, **_kwargs):
        raise OSError('synthetic CSV failure')
    monkeypatch.setattr(fetch_benchmark.arff, 'loadarff', parser)
    if stage == 'csv':
        monkeypatch.setattr(fetch_benchmark.pd.DataFrame, 'to_csv', failed_csv)
    with pytest.raises((ValueError, OSError)) as failure:
        fetch_benchmark.prepare(tmp_path)
    assert source.read_bytes() == payload
    assert any(str(source) in note and 'source' in note.lower() for note in getattr(failure.value, '__notes__', []))


def test_source_path_swap_cannot_redirect_retained_writer(tmp_path, monkeypatch):
    payload = b'synthetic source'
    source = tmp_path / 'openml-42175.arff'
    unrelated = tmp_path / 'unrelated'
    unrelated.write_bytes(b'other process data')
    monkeypatch.setattr(fetch_benchmark, 'SHA256', hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(fetch_benchmark.urllib.request, 'urlopen', lambda *_a, **_k: io.BytesIO(payload))
    original_copy = fetch_benchmark.shutil.copyfileobj
    def swap_before_write(input_file, output_file, *args, **kwargs):
        source.unlink()
        source.symlink_to(unrelated)
        return original_copy(input_file, output_file, *args, **kwargs)
    monkeypatch.setattr(fetch_benchmark.shutil, 'copyfileobj', swap_before_write)
    with pytest.raises(OSError):
        fetch_benchmark.prepare(tmp_path)
    assert source.is_symlink() and unrelated.read_bytes() == b'other process data'


def test_metadata_appearing_during_conversion_prevents_csv_creation(tmp_path, monkeypatch, cached_source):
    metadata = tmp_path / 'creditcard.metadata.json'
    def racing_writer(_frame, output, **_kwargs):
        metadata.write_bytes(b'other process metadata')
        if hasattr(output, 'write'):
            output.write('synthetic csv data\n')
        else:
            Path(output).write_bytes(b'synthetic csv data\n')
    monkeypatch.setattr(fetch_benchmark.pd.DataFrame, 'to_csv', racing_writer)
    failure = None
    try:
        fetch_benchmark.prepare(tmp_path)
    except FileExistsError as exc:
        failure = exc
    assert metadata.read_bytes() == b'other process metadata'
    assert not (tmp_path / 'creditcard.csv').exists()
    assert failure is not None


def test_csv_collision_after_metadata_is_reported_and_preserved(tmp_path, monkeypatch, cached_source):
    csv_path = tmp_path / 'creditcard.csv'
    original_dumps = fetch_benchmark.json.dumps
    def concurrent_csv(*args, **kwargs):
        csv_path.write_bytes(b'other process csv')
        return original_dumps(*args, **kwargs)
    monkeypatch.setattr(fetch_benchmark.json, 'dumps', concurrent_csv)
    failure = None
    try:
        fetch_benchmark.prepare(tmp_path)
    except FileExistsError as exc:
        failure = exc
    assert csv_path.read_bytes() == b'other process csv'
    assert (tmp_path / 'creditcard.metadata.json').exists()
    assert failure is not None
    assert any('metadata' in note.lower() for note in getattr(failure, '__notes__', []))


@pytest.mark.parametrize('failed_file', ['metadata', 'csv'])
def test_interrupted_publication_reports_partial_target(tmp_path, monkeypatch, cached_source, failed_file):
    original_copy = fetch_benchmark.shutil.copyfileobj
    count = 0
    failing_call = 1 if failed_file == 'metadata' else 2
    def copy_with_failure(source, destination, *args, **kwargs):
        nonlocal count
        count += 1
        if count == failing_call:
            destination.write(b'partial')
            destination.flush()
            raise OSError('synthetic write failure')
        return original_copy(source, destination, *args, **kwargs)
    monkeypatch.setattr(fetch_benchmark.shutil, 'copyfileobj', copy_with_failure)
    failure = None
    try:
        fetch_benchmark.prepare(tmp_path)
    except OSError as exc:
        failure = exc
    path = tmp_path / ('creditcard.metadata.json' if failed_file == 'metadata' else 'creditcard.csv')
    assert failure is not None and 'synthetic write failure' in str(failure)
    assert path.read_bytes() == b'partial'
    assert any(str(path) in note for note in getattr(failure, '__notes__', []))
    if failed_file == 'metadata':
        assert not (tmp_path / 'creditcard.csv').exists()
    else:
        assert (tmp_path / 'creditcard.metadata.json').exists()


def test_csv_path_swap_cannot_redirect_retained_writer(tmp_path, monkeypatch, cached_source):
    target = tmp_path / 'creditcard.csv'
    unrelated = tmp_path / 'unrelated'
    unrelated.write_bytes(b'other process data')
    original_copy = fetch_benchmark.shutil.copyfileobj
    count = 0
    def swap_before_write(source, destination, *args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            target.unlink()
            target.symlink_to(unrelated)
        return original_copy(source, destination, *args, **kwargs)
    monkeypatch.setattr(fetch_benchmark.shutil, 'copyfileobj', swap_before_write)
    failure = None
    try:
        fetch_benchmark.prepare(tmp_path)
    except OSError as exc:
        failure = exc
    assert count == 2
    assert failure is not None
    assert target.is_symlink() and unrelated.read_bytes() == b'other process data'


def test_failed_csv_write_does_not_delete_a_replacement(tmp_path, monkeypatch, cached_source):
    target = tmp_path / 'creditcard.csv'
    replacement = tmp_path / 'unrelated'
    replacement.write_bytes(b'other process data')
    original_copy = fetch_benchmark.shutil.copyfileobj
    count = 0
    def swap_then_fail(source, destination, *args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            target.unlink()
            replacement.rename(target)
            raise OSError('synthetic write failure')
        return original_copy(source, destination, *args, **kwargs)
    monkeypatch.setattr(fetch_benchmark.shutil, 'copyfileobj', swap_then_fail)
    with pytest.raises(OSError, match='synthetic write failure'):
        fetch_benchmark.prepare(tmp_path)
    assert target.read_bytes() == b'other process data'
