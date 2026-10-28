// Execute the actual browser modules; only DOM, clock and HTTP boundaries are doubled.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

class Element {
  constructor(id, hidden = false) {
    Object.assign(this, { id, hidden, disabled: false, value: '', textContent: '', dataset: {}, children: [] });
    this.listeners = new Map();
    this.attributes = new Map();
    this.parents = [];
  }
  addEventListener(type, callback) { this.listeners.set(type, callback); }
  setAttribute(name, value) { this.attributes.set(name, value); }
  removeAttribute(name) { this.attributes.delete(name); }
  replaceChildren(...children) { this.children = children; }
  append(...children) { this.children.push(...children); }
  focus() { this.focused = true; }
}

export async function createDemo({ apiBase = '', uuid = true, cryptoSource, predictions = [], explanations = [] } = {}) {
  const markup = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  const nodes = new Map(), stack = [];
  const voidTags = new Set(['meta', 'link', 'input', 'br', 'img', 'source', 'track', 'wbr', 'hr', 'area', 'base', 'embed', 'param', 'col']);
  for (const match of markup.matchAll(/<\/?([a-z][a-z0-9-]*)\b[^>]*>/gi)) {
    const tag = match[1].toLowerCase();
    if (match[0].startsWith('</')) {
      const index = stack.findLastIndex((node) => node.tag === tag);
      if (index >= 0) stack.splice(index);
    } else {
      const id = /\bid="([^"]+)"/.exec(match[0])?.[1];
      if (id) {
        const node = new Element(id, /\shidden(?:\s|>)/.test(match[0]));
        node.parents = stack.map((parent) => parent.id).filter(Boolean);
        nodes.set(id, node);
      }
      if (!voidTags.has(tag)) stack.push({ tag, id });
    }
  }
  const node = (id) => { assert.ok(nodes.has(id), `Missing actual DOM ID: ${id}`); return nodes.get(id); };
  const requests = [], timers = new Map();
  const context = vm.createContext({ AbortController, URL,
    document: { getElementById: node, createElement: () => new Element() },
    crypto: cryptoSource === undefined ? (uuid ? { randomUUID: () => 'generated-fixture-id' } : undefined) : cryptoSource,
    setTimeout(callback) { const handle = {}; timers.set(handle, callback); return handle; },
    clearTimeout(handle) { timers.delete(handle); } });
  const response = (status, data, nonJson = false) => ({ status, ok: status >= 200 && status < 300,
    json: async () => { if (nonJson) throw new SyntaxError('Fixed non-JSON response'); return JSON.parse(JSON.stringify(data)); } });
  context.fetch = async (url, options = {}) => {
    const request = { url, method: options.method || 'GET', payload: options.body ? JSON.parse(options.body) : null };
    requests.push(request);
    if (request.method === 'GET' && url.endsWith('/model-info')) {
      return response(200, { feature_names: ['Time', 'V1', 'Amount'], feature_count: 3, model_source: 'fixed fixture' });
    }
    if (request.method === 'POST' && url.endsWith('/predict')) {
      let fixture = predictions.shift() || {};
      if (fixture.timeout) {
        return new Promise((resolve, reject) => options.signal.addEventListener('abort', () => reject(new Error('Fixed aborted transport')), { once: true }));
      }
      if (fixture.deferred) fixture = await fixture.deferred;
      const id = request.payload.transaction_id || 'server-generated-fixture-id';
      return response(fixture.status || 200, fixture.data || { transaction_id: id, prediction: 0, score: 0.25,
        correlation_id: '00000000-0000-4000-8000-000000000001', explanation_status: 'Calculation queued' }, fixture.nonJson);
    }
    if (request.method === 'GET' && url.includes('/explain/')) {
      let fixture = explanations.shift() || {};
      if (fixture.deferred) fixture = await fixture.deferred;
      const id = decodeURIComponent(url.split('/explain/')[1]);
      return response(fixture.status || 202, fixture.data || { transaction_id: id, status: 'PENDING' });
    }
    throw new Error(`Unexpected in-memory request: ${request.method} ${url}`);
  };
  node('api-base').value = apiBase;
  const client = new vm.SourceTextModule(readFileSync(new URL('../api-client.mjs', import.meta.url), 'utf8'), { context });
  const app = new vm.SourceTextModule(readFileSync(new URL('../app.mjs', import.meta.url), 'utf8'), { context });
  await app.link(async (specifier) => { assert.equal(specifier, './api-client.mjs'); return client; });
  await app.evaluate();
  const flush = async () => { for (let turn = 0; turn < 20; turn++) await Promise.resolve(); };
  await flush();
  assert.equal(node('connection-status').dataset.state, 'ready');
  const fire = (id, event) => node(id).listeners.get(event)?.({ preventDefault() {} });
  return { node, requests, flush,
    visible(id) { return !node(id).hidden && node(id).parents.every((parent) => !node(parent).hidden); },
    input(id, value) { node(id).value = value; return fire(id, 'input'); },
    submit(id = 'submitted-fixture-id', features = '[0, 0, 100]') {
      node('transaction-id').value = id; node('features').value = features; return fire('transaction', 'submit');
    },
    submitCurrent: () => fire('transaction', 'submit'),
    lookup: () => fire('explanation-button', 'click'),
    reconnect: () => fire('connection-form', 'submit'),
    expireTimeouts() { for (const callback of [...timers.values()]) callback(); },
    posts: () => requests.filter((request) => request.method === 'POST'),
    lookups: () => requests.filter((request) => request.url.includes('/explain/')) };
}

export const ambiguous = { status: 503, data: { detail: { transaction_id: 'fixture', status: 'DISPATCH_UNCONFIRMED',
  message: 'Explanation dispatch is unconfirmed; check saved status before retrying' } } };
