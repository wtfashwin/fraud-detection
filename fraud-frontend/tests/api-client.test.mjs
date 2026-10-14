import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { test } from 'node:test';
import { normalizeApiBase, validateFeatures, validateModelInfo, validatePrediction, formatProbability, fetchJson } from '../demo/api-client.mjs';

test('feature input preserves order and rejects coercion, nonfinite values and wrong counts', () => {
  assert.deepEqual(validateFeatures('[10, -0.5, 0]', 3), [10, -0.5, 0]);
  for (const input of ['[1, "2", 3]', '[1, null, 3]', '[1, 1e309, 3]', '[1, 2]', '{"V1":1}', 'broken']) {
    assert.throws(() => validateFeatures(input, 3));
  }
});

test('API URL accepts same origin and explicit HTTP prefixes, rejecting unsafe or ambiguous URLs', () => {
  assert.equal(normalizeApiBase(''), '');
  assert.equal(normalizeApiBase(' http://localhost:8000/ '), 'http://localhost:8000');
  assert.equal(normalizeApiBase('https://example.test/api/'), 'https://example.test/api');
  for (const value of ['javascript:alert(1)', '/api', 'https://user:password@example.test', 'https://example.test/?key=value', 'https://example.test/#part']) {
    assert.throws(() => normalizeApiBase(value));
  }
});

test('metadata must provide a complete, unique ordered list', () => {
  const metadata = { feature_names: ['Time', 'Amount'], feature_count: 2, model_source: 'local' };
  assert.deepEqual(validateModelInfo(metadata), metadata);
  for (const info of [{}, { ...metadata, feature_count: 3 }, { ...metadata, feature_names: ['Time', 'Time'] }, { ...metadata, feature_names: ['', 'Amount'] }]) {
    assert.throws(() => validateModelInfo(info));
  }
});

test('prediction accepts real class probabilities and rejects malformed outcomes', () => {
  const prediction = { prediction: 1, score: 1.2e-8, transaction_id: 'synthetic-test', correlation_id: 'test', explanation_status: 'Calculation queued' };
  assert.deepEqual(validatePrediction(prediction), prediction);
  assert.equal(validatePrediction({ ...prediction, score: null }).score, null);
  for (const data of [{ ...prediction, prediction: 5 }, { ...prediction, score: 2 }, { ...prediction, score: '0.8' }, { ...prediction, transaction_id: '' }]) {
    assert.throws(() => validatePrediction(data));
  }
});

test('small probabilities never round to a false zero or certainty', () => {
  assert.equal(formatProbability(null), 'Not available');
  assert.equal(formatProbability(0), '0%');
  assert.equal(formatProbability(1), '100%');
  assert.equal(formatProbability(1.2e-8), 'Less than 0.01%');
  assert.equal(formatProbability(0.999999), 'Greater than 99.99%');
  assert.equal(formatProbability(0.125), '12.50%');
});

test('HTTP client sends the supplied values and surfaces failed dependencies', async (context) => {
  let received;
  const server = createServer(async (request, response) => {
    let body = '';
    for await (const chunk of request) body += chunk;
    received = body ? JSON.parse(body) : null;
    response.setHeader('Content-Type', 'application/json');
    if (request.url === '/failed') {
      response.writeHead(503);
      response.end(JSON.stringify({ detail: 'Explanation queue unavailable' }));
    } else {
      response.end(JSON.stringify({ echoed: received }));
    }
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  context.after(() => new Promise((resolve) => server.close(resolve)));
  const base = `http://127.0.0.1:${server.address().port}`;
  const payload = { features: [0, -1, 100] };
  assert.deepEqual(await fetchJson(`${base}/predict`, { method: 'POST', body: JSON.stringify(payload) }), { echoed: payload });
  await assert.rejects(fetchJson(`${base}/failed`), { message: 'HTTP 503: Explanation queue unavailable', status: 503 });
});
