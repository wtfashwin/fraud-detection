"""Exercise the real HTTP API and queued explanation; no fraud-quality claims."""
import argparse
import json
import math
import time
import urllib.error
import urllib.request
import uuid


def request(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base.rstrip('/') + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def smoke(base, timeout=60):
    code, health = request(base, '/ready')
    assert code == 200, (code, health)
    code, info = request(base, '/model-info')
    assert code == 200 and info['feature_count'] == len(info['feature_names'])
    # Explicitly synthetic raw input: this checks transport and inference only.
    features = [100.0 if name == 'Amount' else 0.0 for name in info['feature_names']]
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
            assert explanation['feature_names'] == info['feature_names']
            assert len(explanation['shap_values']) == len(features)
            assert all(math.isfinite(v) for v in explanation['shap_values'])
            break
        assert code == 202, (code, explanation)
        if time.monotonic() >= deadline:
            raise TimeoutError('Explanation did not complete before the smoke deadline')
        time.sleep(1)
    print(json.dumps({'status': 'PASS', 'input': 'synthetic transport fixture',
                      'transaction_id': transaction_id, 'feature_count': len(features),
                      'score': result['score'], 'explanation_status': explanation['status']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    smoke(args.url, args.timeout)
