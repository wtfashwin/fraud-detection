"""Read a dataset sidecar and bind any declared CSV hash to the bytes used."""
import hashlib
import json
import os
import stat
from pathlib import Path


def read_dataset_sidecar(source, source_bytes):
    path = Path(source).with_suffix('.metadata.json')
    if not os.path.lexists(path):
        return {}
    if not hasattr(os, 'O_NONBLOCK'):
        raise ValueError(f'Invalid dataset sidecar at {path}: safe reads require O_NONBLOCK')
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, 'O_BINARY', 0))
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError('sidecar must be a regular file')
        with os.fdopen(descriptor, 'r', encoding='utf-8') as stream:
            descriptor = None
            sidecar = json.load(stream)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f'Invalid dataset sidecar at {path}: {exc}') from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if not isinstance(sidecar, dict):
        raise ValueError(f'Invalid dataset sidecar at {path}: expected a JSON object')
    if 'csv_sha256' in sidecar:
        expected = sidecar['csv_sha256']
        if (not isinstance(expected, str) or len(expected) != 64
                or any(char not in '0123456789abcdefABCDEF' for char in expected)):
            raise ValueError(f'Invalid csv_sha256 in dataset sidecar at {path}')
        if expected.lower() != hashlib.sha256(source_bytes).hexdigest():
            raise ValueError(f'csv_sha256 does not match the dataset bytes for sidecar at {path}')
    return sidecar
