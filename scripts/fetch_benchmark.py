"""Download and verify the fixed public OpenML 42175 v1 fraud benchmark.

The dataset stays ignored locally. No credentials or OpenML account are required.
"""
import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy.io import arff

URL = 'https://openml.org/data/v1/download/21756045/CreditCardFraudDetection.arff'
MD5 = 'f9af8fd7966f2d2b8a3e5e67158a7757'
SHA256 = '62416dc7dc34cf7abda5db818e8db92e91152fb2085340f7681bd6f0fe6130ce'
FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']


def prepare(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / 'openml-42175.arff'
    if not source.exists():
        partial = source.with_suffix('.arff.part')
        try:
            with urllib.request.urlopen(URL, timeout=60) as response, partial.open('wb') as output:
                shutil.copyfileobj(response, output)
            partial.replace(source)
        finally:
            partial.unlink(missing_ok=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SHA256:
        source.unlink()
        raise ValueError('Benchmark checksum mismatch; invalid cached source removed, retry the download')
    values, _ = arff.loadarff(source)
    frame = pd.DataFrame(values)
    frame['Class'] = frame['Class'].astype(int)
    if (list(frame.columns) != FEATURES + ['Class'] or frame.shape != (284807, 31)
            or int(frame['Class'].sum()) != 492 or frame.isna().any().any()):
        raise ValueError('Benchmark schema, row counts, labels, or missing values changed')
    destination = directory / 'creditcard.csv'
    frame.to_csv(destination, index=False)
    metadata = {
        'kind': 'public_benchmark', 'data_kind': 'real',
        'source': 'https://www.openml.org/d/42175', 'source_url': 'https://www.openml.org/d/42175',
        'download_url': URL, 'openml_dataset_id': 42175, 'openml_version': 1,
        'source_sha256': digest, 'source_md5': MD5,
        'csv_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        'rows': len(frame), 'fraud_rows': int(frame['Class'].sum()),
        'missing_values': 0, 'feature_count': len(FEATURES), 'license_metadata': 'Public',
        'retrieved_at': datetime.now(timezone.utc).isoformat(),
        'limits': 'Historical anonymized benchmark; V1-V28 already PCA. Random holdout is not production/time-shift validation.',
    }
    destination.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data/benchmark')
    print(json.dumps(prepare(parser.parse_args().directory), indent=2))
