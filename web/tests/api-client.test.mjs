import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { test } from 'node:test';
import { normalizeApiBase, validateFeatures, validateModelInfo, validatePrediction, formatProbability, fetchJson } from '../api-client.mjs';

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

for (const [name, status, body, detail] of [
  ['null body', 503, null, 'The API did not provide an error description.'],
  ['empty detail array', 422, { detail: [] }, 'The API did not provide an error description.'],
  ['empty detail string', 422, { detail: '' }, 'The API did not provide an error description.'],
  ['whitespace-only detail string', 422, { detail: '  \t\n' }, 'The API did not provide an error description.'],
  ['nonblank detail string', 503, { detail: '  Service unavailable  ' }, '  Service unavailable  '],
  ['null detail entry', 422, { detail: [null] }, 'Input: Invalid value'],
  ['non-array location', 422, { detail: [{ loc: 'body', msg: 'Invalid input' }] }, 'Input: Invalid input'],
  ['non-primitive location item', 422, { detail: [{ loc: ['body', { toString: 'invalid' }], msg: 'Invalid input' }] }, 'Input: Invalid input'],
  ['non-string message', 422, { detail: [{ loc: ['body'], msg: { toString: 'invalid' } }] }, 'body: Invalid value'],
  ['whitespace-only item message', 422, { detail: [{ loc: ['body'], msg: '  \t\n' }] }, 'body: Invalid value'],
  ['nonblank item message', 422, { detail: [{ loc: ['body'], msg: '  Invalid input  ' }] }, 'body:   Invalid input  '],
]) {
  test(`HTTP client preserves status for ${name}`, async (context) => {
    const server = createServer((_request, response) => {
      response.writeHead(status, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify(body));
    });
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    context.after(() => new Promise((resolve) => server.close(resolve)));
    const url = `http://127.0.0.1:${server.address().port}/error`;
    await assert.rejects(fetchJson(url), (error) => {
      assert.equal(error.status, status);
      assert.equal(error.message, `HTTP ${status}: ${detail}`);
      assert.doesNotMatch(error.message, /Cannot reach the API/);
      return true;
    });
  });
}

test('HTTP client preserves ordinary validation details', async (context) => {
  const server = createServer((_request, response) => {
    response.writeHead(422, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify({ detail: [{ loc: ['body', 'features', 0], msg: 'Invalid input' }] }));
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  context.after(() => new Promise((resolve) => server.close(resolve)));
  const url = `http://127.0.0.1:${server.address().port}/error`;
  await assert.rejects(fetchJson(url), { status: 422, message: 'HTTP 422: body.features.0: Invalid input' });
});

test('HTTP client preserves a successful null JSON response', async (context) => {
  const server = createServer((_request, response) => {
    response.writeHead(200, { 'Content-Type': 'application/json' });
    response.end('null');
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  context.after(() => new Promise((resolve) => server.close(resolve)));
  const url = `http://127.0.0.1:${server.address().port}/result`;
  assert.equal(await fetchJson(url), null);
});

for (const status of [409, 404]) {
  test(`HTTP client preserves HTTP ${status} for a non-JSON error body`, async (context) => {
    const server = createServer((_request, response) => {
      response.writeHead(status, { 'Content-Type': 'text/plain' });
      response.end('not JSON');
    });
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    context.after(() => new Promise((resolve) => server.close(resolve)));
    const url = `http://127.0.0.1:${server.address().port}/error`;
    await assert.rejects(fetchJson(url), {
      status,
      message: `HTTP ${status}: The API returned a non-JSON response.`,
    });
  });
}

test('HTTP client still rejects a successful non-JSON body', async (context) => {
  const server = createServer((_request, response) => {
    response.writeHead(200, { 'Content-Type': 'text/plain' });
    response.end('not JSON');
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  context.after(() => new Promise((resolve) => server.close(resolve)));
  const url = `http://127.0.0.1:${server.address().port}/result`;
  await assert.rejects(fetchJson(url), (error) => {
    assert.equal(error.status, undefined);
    assert.equal(error.message, 'HTTP 200: The API returned a non-JSON response.');
    return true;
  });
});
