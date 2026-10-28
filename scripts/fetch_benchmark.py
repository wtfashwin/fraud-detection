"""Download and verify the fixed public OpenML 42175 v1 fraud benchmark.

The dataset stays ignored locally. No credentials or OpenML account are required.
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import stat
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy.io import arff

URL = 'https://openml.org/data/v1/download/21756045/CreditCardFraudDetection.arff'
MD5 = 'f9af8fd7966f2d2b8a3e5e67158a7757'
SHA256 = '62416dc7dc34cf7abda5db818e8db92e91152fb2085340f7681bd6f0fe6130ce'
FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
MAX_SOURCE_BYTES = 256 * 1024 * 1024
MAX_CSV_BYTES = 512 * 1024 * 1024


class BoundedBuffer(io.BytesIO):
    """An in-memory file that never spills to a pathname."""

    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def write(self, data):
        if self.tell() + len(data) > self.limit:
            raise ValueError(f'In-memory data exceeds the {self.limit}-byte limit')
        return super().write(data)


def read_bounded(stream, limit):
    buffer = BoundedBuffer(limit)
    while True:
        chunk = stream.read(min(1024 * 1024, limit + 1 - buffer.tell()))
        if not chunk:
            buffer.seek(0)
            return buffer
        buffer.write(chunk)


def publish_exclusive(buffer, path):
    """Create a new file and write only through its retained descriptor."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0)
    try:
        descriptor = os.open(path, flags, 0o666)
    except FileExistsError as exc:
        raise FileExistsError(f'{path} already exists; it was not overwritten') from exc
    try:
        try:
            output = os.fdopen(descriptor, 'wb')
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        with output:
            buffer.seek(0)
            shutil.copyfileobj(buffer, output, length=1024 * 1024)
            output.flush()
            written = os.fstat(output.fileno())
            current = os.stat(path, follow_symlinks=False)
            if (written.st_dev, written.st_ino) != (current.st_dev, current.st_ino):
                raise OSError(f'{path} was replaced while being written')
    except BaseException as exc:
        exc.add_note(f'A newly created file at {path} may remain incomplete, or the path may have been replaced; inspect it before use.')
        raise


def open_existing_source(source):
    # Never follow a symlink or wait on a FIFO when examining a cache.
    if not hasattr(os, 'O_NOFOLLOW') or not hasattr(os, 'O_NONBLOCK'):
        raise OSError('Reading a cached source safely requires O_NOFOLLOW and O_NONBLOCK')
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f'Benchmark source must be a regular file: {source}')
        return os.fdopen(descriptor, 'rb')
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def prepare(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / 'openml-42175.arff'
    destination = directory / 'creditcard.csv'
    metadata_path = destination.with_suffix('.metadata.json')
    for target in (destination, metadata_path):
        if os.path.lexists(target):
            raise FileExistsError(f'{target} already exists; choose a fresh output directory')

    try:
        cached = open_existing_source(source)
    except FileNotFoundError:
        cached = None
    if cached is not None:
        with cached:
            source_buffer = read_bounded(cached, MAX_SOURCE_BYTES)
    else:
        with urllib.request.urlopen(URL, timeout=60) as response:
            source_buffer = read_bounded(response, MAX_SOURCE_BYTES)

    digest = hashlib.sha256(source_buffer.getbuffer()).hexdigest()
    if digest != SHA256:
        raise ValueError('Benchmark checksum mismatch; existing files were preserved')
    if cached is None:
        publish_exclusive(source_buffer, source)
    try:
        return convert_verified_source(source_buffer, digest, destination, metadata_path)
    except BaseException as exc:
        if cached is None:
            exc.add_note(f'The new source at {source} may remain; inspect it before use.')
        raise


def convert_verified_source(source_buffer, digest, destination, metadata_path):
    source_buffer.seek(0)
    with io.TextIOWrapper(source_buffer, encoding='utf-8') as verified_source:
        values, _ = arff.loadarff(verified_source)
    frame = pd.DataFrame(values)
    frame['Class'] = frame['Class'].astype(int)
    if (list(frame.columns) != FEATURES + ['Class'] or frame.shape != (284807, 31)
            or int(frame['Class'].sum()) != 492 or frame.isna().any().any()):
        raise ValueError('Benchmark schema, row counts, labels, or missing values changed')
    csv_buffer = BoundedBuffer(MAX_CSV_BYTES)
    with io.TextIOWrapper(csv_buffer, encoding='utf-8', newline='', write_through=True) as csv_text:
        frame.to_csv(csv_text, index=False)
        csv_text.flush()
        csv_bytes = csv_buffer.getvalue()
    metadata = {
        'kind': 'public_benchmark', 'data_kind': 'real',
        'source': 'https://www.openml.org/d/42175', 'source_url': 'https://www.openml.org/d/42175',
        'download_url': URL, 'openml_dataset_id': 42175, 'openml_version': 1,
        'source_sha256': digest, 'source_md5': MD5,
        'csv_sha256': hashlib.sha256(csv_bytes).hexdigest(),
        'rows': len(frame), 'fraud_rows': int(frame['Class'].sum()),
        'missing_values': 0, 'feature_count': len(FEATURES), 'license_metadata': 'Public',
        'retrieved_at': datetime.now(timezone.utc).isoformat(),
        'limits': 'Historical anonymized benchmark; V1-V28 already PCA. Random holdout is not production/time-shift validation.',
    }
    metadata_buffer = io.BytesIO((json.dumps(metadata, indent=2) + '\n').encode('utf-8'))
    publish_exclusive(metadata_buffer, metadata_path)
    try:
        publish_exclusive(io.BytesIO(csv_bytes), destination)
    except BaseException as exc:
        exc.add_note(f'The new metadata at {metadata_path} may remain; inspect it and {destination} before use.')
        raise
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data/benchmark')
    print(json.dumps(prepare(parser.parse_args().directory), indent=2))
