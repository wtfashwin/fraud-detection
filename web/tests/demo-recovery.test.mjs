import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';

// Run the actual module against the IDs and hidden ancestors in index.html.
// Browser APIs are represented locally; HTTP and the app's client remain real.
async function domFromHtml() {
  const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const nodes = new Map(), stack = [], voidTags = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr']);
  function node(parent = null) {
    return { parent, hidden: false, disabled: false, value: '', textContent: '', dataset: {}, listeners: {}, attributes: {},
      addEventListener(type, fn) { this.listeners[type] = fn; },
      setAttribute(key, value) { this.attributes[key] = value; },
      removeAttribute(key) { delete this.attributes[key]; },
      replaceChildren() {}, append() {}, focus() {} };
  }
  for (const match of html.matchAll(/<(\/)?([a-z][\w-]*)\b([^>]*)>/giu)) {
    const [, closing, rawTag, attributes] = match, tag = rawTag.toLowerCase();
    if (closing) {
      while (stack.length && stack.pop().tag !== tag) {}
      continue;
    }
    const n = node(stack.at(-1)?.node ?? null), id = attributes.match(/\bid="([^"]+)"/u)?.[1];
    n.hidden = /(?:^|\s)hidden(?:\s|$)/u.test(attributes);
    n.disabled = /(?:^|\s)disabled(?:\s|$)/u.test(attributes);
    if (id) nodes.set(id, n);
    if (!voidTags.has(tag) && !attributes.trimEnd().endsWith('/')) stack.push({ tag, node: n });
  }
  return { getElementById(id) { assert.ok(nodes.has(id), `Missing element ${id}`); return nodes.get(id); },
    createElement() { return node(); } };
}

const visible = (node) => { for (let n = node; n; n = n.parent) if (n.hidden) return false; return true; };
const pause = () => new Promise((resolve) => setTimeout(resolve, 5));
let instance = 0;

async function startApp(context, { uuid = null, mode = 'success', explanations = {}, modelUnavailable = false } = {}) {
  const calls = [], saved = new Map(Object.entries(explanations));
  let releasePrediction;
  const predictionGate = new Promise((resolve) => { releasePrediction = resolve; });
  const server = createServer(async (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    if (req.url === '/model-info') {
      if (modelUnavailable) { res.writeHead(503); return res.end(JSON.stringify({ detail: 'Model unavailable' })); }
      return res.end(JSON.stringify({ feature_names: ['Amount'], feature_count: 1, model_source: 'synthetic' }));
    }
    if (req.method === 'POST' && req.url === '/predict') {
      let body = ''; for await (const chunk of req) body += chunk;
      const data = JSON.parse(body); calls.push({ method: 'POST', path: req.url, data });
      if (mode === 'held') await predictionGate;
      if (mode === 'lost') {
        if (saved.has(data.transaction_id)) { res.writeHead(409); return res.end(JSON.stringify({ detail: 'Transaction ID already exists; use a new ID' })); }
        saved.set(data.transaction_id, { transaction_id: data.transaction_id, status: 'PENDING' }); req.socket.destroy(); return;
      }
      if (mode === 'conflict') { res.writeHead(409); return res.end(JSON.stringify({ detail: 'Transaction ID already exists; use a new ID' })); }
      saved.set(data.transaction_id, { transaction_id: data.transaction_id, status: 'COMPLETED', shap_values: [1], feature_names: ['Amount'], base_value: 0, explanation_method: 'synthetic', created_at: '2026-01-01T00:00:00Z' });
      return res.end(JSON.stringify({ transaction_id: mode === 'wrong-id' ? 'other-id' : data.transaction_id, prediction: 1, score: 0.8, correlation_id: 'synthetic-correlation', explanation_status: 'Calculation queued' }));
    }
    if (req.method === 'GET' && req.url.startsWith('/explain/')) {
      const id = decodeURIComponent(req.url.slice('/explain/'.length)); calls.push({ method: 'GET', path: req.url, id });
      if (!saved.has(id)) { res.writeHead(404); return res.end(JSON.stringify({ detail: 'SHAP explanation not found' })); }
      const result = saved.get(id); if (result.status === 'PENDING') res.writeHead(202);
      return res.end(JSON.stringify(result));
    }
    res.writeHead(500); return res.end(JSON.stringify({ detail: 'Unexpected test request' }));
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  context.after(() => new Promise((resolve) => server.close(resolve)));
  const oldDocument = Object.getOwnPropertyDescriptor(globalThis, 'document');
  const oldCrypto = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  context.after(() => {
    if (oldDocument) Object.defineProperty(globalThis, 'document', oldDocument); else delete globalThis.document;
    if (oldCrypto) Object.defineProperty(globalThis, 'crypto', oldCrypto); else delete globalThis.crypto;
  });
  const document = await domFromHtml();
  Object.defineProperty(globalThis, 'document', { configurable: true, value: document });
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value: uuid ? { randomUUID: typeof uuid === 'function' ? uuid : () => uuid } : {} });
  const get = (id) => document.getElementById(id);
  get('api-base').value = `http://127.0.0.1:${server.address().port}`;
  await import(`../app.mjs?test=${++instance}`);
  for (let i = 0; i < 400 && !['ready', 'failed'].includes(get('connection-status').dataset.state); i++) await pause();
  assert.equal(get('connection-status').dataset.state, modelUnavailable ? 'failed' : 'ready', 'Unexpected test API connection state');
  const input = (id, value) => { get(id).value = value; get(id).listeners.input?.({}); };
  const submit = () => get('transaction').listeners.submit({ preventDefault() {} });
  const lookup = () => get('explanation-button').listeners.click({ preventDefault() {} });
  return { get, input, submit, lookup, calls, releasePrediction, setMode: (value) => { mode = value; } };
}

test('browser without randomUUID requires an ID before sending', async (context) => {
  const app = await startApp(context); app.input('features', '[0]');
  await app.submit();
  assert.equal(app.calls.filter((call) => call.method === 'POST').length, 0);
  assert.match(app.get('input-error').textContent, /transaction ID/i);
  assert.equal(app.get('transaction-id').attributes['aria-invalid'], 'true');
});

test('invalid supplied IDs do not reach the prediction API', async (context) => {
  const app = await startApp(context); app.input('features', '[0]'); app.input('transaction-id', '../bad');
  await app.submit();
  assert.equal(app.calls.length, 0);
  assert.equal(app.get('transaction-id').attributes['aria-invalid'], 'true');
});

test('filling the example preserves an entered ID for recovery', async (context) => {
  const app = await startApp(context); app.input('transaction-id', 'previous-id');
  app.get('sample-button').listeners.click({});
  assert.equal(app.get('transaction-id').value, 'previous-id');
});

test('filling the example clears a confirmed ID even after editing the features', async (context) => {
  const ids = ['11111111-2222-4333-8444-555555555555', 'aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee'];
  const app = await startApp(context, { uuid: () => ids.shift() }); app.input('features', '[0]');
  await app.submit();
  assert.equal(app.get('result').hidden, false);
  app.input('features', '[1]');
  assert.equal(app.get('result').hidden, true);
  app.get('sample-button').listeners.click({});
  assert.equal(app.get('transaction-id').value, '');
  await app.submit();
  assert.equal(app.get('result').hidden, false);
  assert.deepEqual(app.calls.filter((call) => call.method === 'POST').map((call) => call.data.transaction_id),
    ['11111111-2222-4333-8444-555555555555', 'aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee']);
});

test('local feature validation does not forget an already confirmed ID', async (context) => {
  const app = await startApp(context); app.input('features', '[0]'); app.input('transaction-id', 'confirmed-id');
  await app.submit();
  assert.equal(app.get('result').hidden, false);
  app.input('features', 'invalid json');
  await app.submit();
  assert.equal(app.calls.filter((call) => call.method === 'POST').length, 1);
  assert.match(app.get('input-error').textContent, /valid JSON/i);
  app.get('sample-button').listeners.click({});
  assert.equal(app.get('transaction-id').value, '');
});

test('the example retains an ID after a lost response', async (context) => {
  const app = await startApp(context, { mode: 'lost' }); app.input('features', '[0]'); app.input('transaction-id', 'uncertain-id');
  await app.submit();
  app.get('sample-button').listeners.click({});
  assert.equal(app.get('transaction-id').value, 'uncertain-id');
});

test('a user entered ID survives filling the example after a previous success', async (context) => {
  const app = await startApp(context); app.input('features', '[0]'); app.input('transaction-id', 'confirmed-id');
  await app.submit();
  app.input('transaction-id', 'user-entered-id');
  app.get('sample-button').listeners.click({});
  assert.equal(app.get('transaction-id').value, 'user-entered-id');
});

test('reentering an ID after success keeps the user entered value', async (context) => {
  const app = await startApp(context); app.input('features', '[0]'); app.input('transaction-id', 'confirmed-id');
  await app.submit();
  app.input('transaction-id', 'confirmed-id');
  app.get('sample-button').listeners.click({});
  assert.equal(app.get('transaction-id').value, 'confirmed-id');
});

test('the example retains an ID after a later failed request', async (context) => {
  const app = await startApp(context); app.input('features', '[0]'); app.input('transaction-id', 'confirmed-id');
  await app.submit();
  app.setMode('conflict');
  await app.submit();
  app.get('sample-button').listeners.click({});
  assert.equal(app.get('transaction-id').value, 'confirmed-id');
});

test('changing the API base clears the previous confirmation but preserves the ID', async (context) => {
  const other = createServer((req, res) => {
    res.setHeader('Content-Type', 'application/json');
    return res.end(JSON.stringify({ feature_names: ['Amount'], feature_count: 1, model_source: 'other synthetic' }));
  });
  await new Promise((resolve) => other.listen(0, '127.0.0.1', resolve));
  context.after(() => new Promise((resolve) => other.close(resolve)));
  const app = await startApp(context); app.input('features', '[0]'); app.input('transaction-id', 'confirmed-id');
  await app.submit();
  app.input('api-base', `http://127.0.0.1:${other.address().port}`);
  await app.get('connection-form').listeners.submit({ preventDefault() {} });
  app.get('sample-button').listeners.click({});
  assert.equal(app.get('transaction-id').value, 'confirmed-id');
});

test('a lost response retains the sent ID and allows a visible lookup of that ID', async (context) => {
  const app = await startApp(context, { mode: 'lost' }); app.input('features', '[0]'); app.input('transaction-id', 'my-transaction');
  await app.submit();
  assert.ok(app.calls.filter((call) => call.method === 'POST').length >= 1);
  assert.ok(app.calls.filter((call) => call.method === 'POST').every((call) => call.data.transaction_id === 'my-transaction'));
  assert.equal(app.get('transaction-id').value, 'my-transaction');
  assert.match(app.get('input-error').textContent, /may (?:still )?have been processed|already exists/i);
  assert.equal(app.get('result').hidden, true);
  assert.ok(visible(app.get('explanation-button')));
  assert.equal(app.get('explanation-button').disabled, false);
  await app.lookup();
  assert.equal(app.calls.find((call) => call.method === 'GET').id, 'my-transaction');
  assert.match(app.get('explanation-message').textContent, /my-transaction/);
  assert.match(app.get('explanation-message').textContent, /pending/i);
});

test('generated UUID is sent, shown and usable for a completed explanation', async (context) => {
  const id = '11111111-2222-4333-8444-555555555555';
  const app = await startApp(context, { uuid: id }); app.input('features', '[0]');
  await app.submit();
  assert.equal(app.calls[0].data.transaction_id, id);
  assert.equal(app.get('transaction-id').value, id);
  assert.equal(app.get('result').hidden, false);
  assert.equal(app.get('result-id').textContent, id);
  await app.lookup();
  assert.equal(app.calls[1].id, id);
  assert.match(app.get('explanation-message').textContent, new RegExp(id));
  assert.equal(JSON.parse(app.get('explanation-response').textContent).status, 'COMPLETED');
});

test('duplicate ID leads to lookup without urging an immediate new ID', async (context) => {
  const app = await startApp(context, { mode: 'conflict' }); app.input('features', '[0]'); app.input('transaction-id', 'already-used');
  await app.submit();
  assert.equal(app.get('transaction-id').value, 'already-used');
  assert.doesNotMatch(app.get('input-error').textContent, /enter a new|use a new/i);
  assert.match(app.get('input-error').textContent, /explanation/i);
  assert.ok(visible(app.get('explanation-button')));
  await app.lookup();
  assert.equal(app.calls[1].id, 'already-used');
  assert.match(app.get('explanation-message').textContent, /not found/i);
  assert.doesNotMatch(app.get('explanation-message').textContent, /was not processed|did not process/i);
});

test('a mismatched response ID does not replace the ID sent', async (context) => {
  const app = await startApp(context, { mode: 'wrong-id' }); app.input('features', '[0]'); app.input('transaction-id', 'original-id');
  await app.submit();
  assert.equal(app.get('transaction-id').value, 'original-id');
  assert.equal(app.get('result').hidden, true);
  assert.match(app.get('input-error').textContent, /may (?:still )?have been processed/i);
  await app.lookup();
  assert.equal(app.calls[1].id, 'original-id');
});

test('the form cannot be edited while a prediction is in flight', async (context) => {
  const app = await startApp(context, { mode: 'held' }); app.input('features', '[0]'); app.input('transaction-id', 'held-id');
  const submission = app.submit();
  try {
    for (let i = 0; i < 400 && app.calls.length === 0; i++) await pause();
    assert.equal(app.calls[0].data.transaction_id, 'held-id');
    assert.equal(app.get('transaction-id').disabled, true);
    assert.equal(app.get('features').disabled, true);
    assert.equal(app.get('score-button').disabled, true);
  } finally { app.releasePrediction(); await submission; }
  assert.equal(app.get('transaction-id').value, 'held-id');
  assert.equal(app.get('transaction-id').disabled, false);
});

test('lookup works after reload and clears old results when ID changes', async (context) => {
  const app = await startApp(context, { explanations: { 'saved-id': { transaction_id: 'saved-id', status: 'PENDING' } } });
  assert.equal(app.get('result').hidden, true);
  app.input('transaction-id', 'saved-id');
  assert.ok(visible(app.get('explanation-button')));
  assert.equal(app.get('explanation-button').disabled, false);
  await app.lookup();
  assert.equal(app.calls[0].id, 'saved-id');
  assert.match(app.get('explanation-message').textContent, /saved-id/);
  app.input('transaction-id', 'another-id');
  assert.equal(app.get('explanation-message').textContent, '');
  assert.equal(app.get('explanation-response').hidden, true);
  app.input('transaction-id', '');
  assert.equal(app.get('explanation-button').disabled, true);
});

test('saved explanation remains retrievable when model metadata is unavailable', async (context) => {
  const app = await startApp(context, { modelUnavailable: true, explanations: { 'saved-id': { transaction_id: 'saved-id', status: 'PENDING' } } });
  app.input('transaction-id', 'saved-id');
  assert.ok(visible(app.get('explanation-button')));
  assert.equal(app.get('explanation-button').disabled, false);
  await app.lookup();
  assert.equal(app.calls[0].id, 'saved-id');
  assert.match(app.get('explanation-message').textContent, /pending/i);
});

test('an invalid changed API base cannot query the previous API', async (context) => {
  const app = await startApp(context, { explanations: { 'saved-id': { transaction_id: 'saved-id', status: 'PENDING' } } });
  app.input('transaction-id', 'saved-id');
  app.input('api-base', 'not a URL');
  assert.equal(app.get('explanation-button').disabled, true);
  await app.get('connection-form').listeners.submit({ preventDefault() {} });
  assert.equal(app.get('explanation-button').disabled, true);
  await app.lookup();
  assert.deepEqual(app.calls, []);
});

test('lookup uses the newly validated API base when its model is unavailable', async (context) => {
  const received = [];
  const other = createServer((req, res) => {
    received.push(req.url); res.setHeader('Content-Type', 'application/json');
    if (req.url === '/model-info') { res.writeHead(503); return res.end(JSON.stringify({ detail: 'Model unavailable' })); }
    if (req.url === '/explain/saved-id') { res.writeHead(202); return res.end(JSON.stringify({ transaction_id: 'saved-id', status: 'PENDING' })); }
    res.writeHead(404); return res.end(JSON.stringify({ detail: 'Not found' }));
  });
  await new Promise((resolve) => other.listen(0, '127.0.0.1', resolve));
  context.after(() => new Promise((resolve) => other.close(resolve)));
  const app = await startApp(context); app.input('transaction-id', 'saved-id');
  app.input('api-base', `http://127.0.0.1:${other.address().port}`);
  await app.get('connection-form').listeners.submit({ preventDefault() {} });
  assert.equal(app.get('connection-status').dataset.state, 'failed');
  assert.equal(app.get('explanation-button').disabled, false);
  await app.lookup();
  assert.deepEqual(received, ['/model-info', '/explain/saved-id']);
  assert.deepEqual(app.calls, []);
});
