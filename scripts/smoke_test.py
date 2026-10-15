"""Exercise the real HTTP API and queued explanation; no fraud-quality claims."""
import argparse
import csv
import json
import math
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def csv_features(path, feature_names):
    """Read one explicitly supplied CSV row in the serving model's raw order."""
    with Path(path).open(newline='') as stream:
        row = next(csv.DictReader(stream), None)
    if row is None or any(name not in row for name in feature_names):
        raise ValueError('CSV needs a data row and every raw model feature')
    try:
        values = [float(row[name]) for name in feature_names]
    except (TypeError, ValueError) as error:
        raise ValueError('CSV features must be numeric') from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError('CSV features must be finite')
    return values


def request(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base.rstrip('/') + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def smoke(base, timeout=60, features_csv=None):
    # The image must serve executable demo assets, not merely contain source files.
    for path, content_type in [("/demo/", "text/html"), ("/demo/app.mjs", "javascript")]:
        with urllib.request.urlopen(base.rstrip('/') + path, timeout=10) as asset:
            assert asset.status == 200 and content_type in asset.headers.get("Content-Type", ""), path
            assert asset.read(), path
    code, health = request(base, '/ready')
    assert code == 200, (code, health)
    code, info = request(base, '/model-info')
    assert code == 200 and info['feature_count'] == len(info['feature_names'])
    # Explicitly synthetic raw input: this checks transport and inference only.
    features = (csv_features(features_csv, info['feature_names']) if features_csv else
                [100.0 if name == 'Amount' else 0.0 for name in info['feature_names']])
    transaction_id = 'smoke-' + str(uuid.uuid4())
    code, result = request(base, '/predict', {'features': features, 'transaction_id': transaction_id})
    assert code == 200, (code, result)
    assert result['prediction'] in (0, 1)
    assert math.isfinite(result['score']) and 0 <= result['score'] <= 1
    assert result['explanation_status'] == 'Calculation queued'
    code, duplicate = request(base, '/predict', {'features': features, 'transaction_id': transaction_id})
    assert code == 409, (code, duplicate)
    code, invalid = request(base, '/predict', {'features': [True] * len(features)})
    assert code == 422, (code, invalid)
    deadline = time.monotonic() + timeout
    while True:
        code, explanation = request(base, '/explain/' + transaction_id)
        if code == 200:
            assert explanation['status'] == 'COMPLETED', explanation
            expected_names = info.get('explanation_feature_names', info['feature_names'])
            assert explanation['feature_names'] == expected_names
            assert len(explanation['shap_values']) == len(expected_names)
            assert all(math.isfinite(v) for v in explanation['shap_values'])
            break
        assert code == 202, (code, explanation)
        if time.monotonic() >= deadline:
            raise TimeoutError('Explanation did not complete before the smoke deadline')
        time.sleep(1)
    return {'status': 'PASS',
            'input': 'provided CSV row' if features_csv else 'synthetic transport fixture',
            'transaction_id': transaction_id, 'feature_count': len(features),
            'score': result['score'], 'prediction_response': result,
            'explanation_response': explanation}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--features-csv', type=Path)
    parser.add_argument('--output', type=Path, help='Save successful HTTP responses as JSON')
    args = parser.parse_args()
    report = smoke(args.url, args.timeout, args.features_csv)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
