"""Parse a single CSV byte snapshot, checking local sidecars when available."""
import io
import os
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request

import pandas as pd
from pandas.io import common as pandas_io
from pandas.io.common import get_handle, infer_compression, is_fsspec_url, is_url

from training import dataset_provenance


def local_path(source):
    # pandas treats an already open object as a stream even when it is path-like.
    if pd.api.types.is_file_like(source):
        return None
    if not isinstance(source, (str, os.PathLike)):
        return None
    value = os.fspath(source)
    if not isinstance(value, str):
        return None
    parts = urlsplit(value)
    if parts.scheme.lower() == 'file':
        return None
    if is_url(value) or is_fsspec_url(value):
        return None
    return Path(os.path.expanduser(value))


def read_checked_csv(source):
    path = local_path(source)
    value = os.fspath(source) if isinstance(source, (str, os.PathLike)) and not pd.api.types.is_file_like(source) else None
    if isinstance(value, str) and urlsplit(value).scheme.lower() == 'file':
        # Let urllib resolve the same local hosts, queries and fragments as pandas.
        # A file URL must expose the path of the actual opened file for the sidecar check.
        with pandas_io.urlopen(Request(value)) as response:
            opened_name = getattr(getattr(response, 'fp', None), 'name', None)
            if not isinstance(opened_name, (str, bytes, os.PathLike)):
                raise ValueError('Cannot identify the opened file path for the file URL sidecar check')
            path = Path(os.fsdecode(opened_name))
            snapshot = response.read()
            dataset_provenance.read_dataset_sidecar(path, snapshot)
            compression = ('gzip' if response.headers.get('Content-Encoding') == 'gzip'
                           else infer_compression(source, 'infer'))
    elif path is not None:
        snapshot = path.read_bytes()
        dataset_provenance.read_dataset_sidecar(path, snapshot)
        compression = infer_compression(source, 'infer')
    else:
        # pandas opens and decompresses its supported URLs and file-like objects.
        # There is no local sidecar path to associate with these inputs.
        with get_handle(source, 'rb', compression='infer', is_text=False) as handles:
            snapshot = handles.handle.read()
        compression = None
    buffer = io.StringIO(snapshot) if isinstance(snapshot, str) else io.BytesIO(snapshot)
    return pd.read_csv(buffer, compression=compression)
