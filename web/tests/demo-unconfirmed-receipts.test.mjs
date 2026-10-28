import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import { test } from 'node:test';
import { createDemo } from './demo-harness.mjs';

if (typeof vm.SourceTextModule !== 'function') {
  test('unconfirmed receipt checks in an isolated module runner', (context) => {
    const childEnv = { ...process.env };
    delete childEnv.NODE_TEST_CONTEXT;
    const child = spawnSync(process.execPath, ['--experimental-vm-modules', '--max-old-space-size=128', '--test', '--test-reporter=tap', fileURLToPath(import.meta.url)],
      { env: childEnv, encoding: 'utf8', timeout: 20000, maxBuffer: 1024 * 1024 });
    assert.equal(child.error, undefined, child.error?.message);
    assert.equal(child.status, 0, `${child.stdout}\n${child.stderr}`);
    assert.match(child.stdout, /# tests [1-9]\d*/, 'The VM child must report executed tests.');
    context.diagnostic(child.stdout.trim());
  });
} else {
  const unconfirmed = () => ({ status: 503, data: { detail: 'Synthetic queue unavailable' } });
  const rows = (demo) => demo.node('unconfirmed-receipt-list').children;
  const rowFor = (demo, id, api) => {
    const item = rows(demo).find((row) => row.children.some((child) => child.textContent.includes(id) && child.textContent.includes(api)));
    assert.ok(item, `The receipt for ${id} on ${api} must remain visible.`);
    return item;
  };
  const action = (row, label) => {
    const button = row.children.find((child) => child.textContent === label);
    assert.ok(button, `Missing ${label} button`);
    return button.listeners.get('click')({ preventDefault() {} });
  };
  const status = (row) => row.children.find((child) => child.attributes.get('role') === 'status');
  const output = (row) => row.children.find((child) => child.id === '');

  test('filling another example retains the unconfirmed ID and original API without reusing it for scoring', async () => {
    const demo = await createDemo({ apiBase: 'https://first.invalid/api', predictions: [unconfirmed()] });
    await demo.submit('unconfirmed-A');
    demo.node('sample-button').listeners.get('click')();
    assert.equal(demo.node('transaction-id').value, '');
    assert.equal(demo.lookups().length, 0);
    const row = rowFor(demo, 'unconfirmed-A', 'https://first.invalid/api');
    assert.equal(demo.visible('unconfirmed-receipts'), true);
    assert.equal(demo.posts().length, 1);
    await action(row, 'Check original API');
    assert.equal(demo.lookups()[0].url, 'https://first.invalid/api/explain/unconfirmed-A');
    assert.match(status(row).textContent, /first\.invalid/);
    assert.equal(demo.posts().length, 1);
  });

  test('a second unconfirmed request and API switch do not overwrite the first receipt', async () => {
    const demo = await createDemo({ apiBase: 'https://first.invalid/api', predictions: [unconfirmed(), unconfirmed()] });
    await demo.submit('unconfirmed-A');
    demo.input('api-base', 'https://second.invalid/api');
    await demo.reconnect();
    await demo.submit('unconfirmed-B');
    const first = rowFor(demo, 'unconfirmed-A', 'https://first.invalid/api');
    rowFor(demo, 'unconfirmed-B', 'https://second.invalid/api');
    assert.equal(rows(demo).length, 2);
    assert.equal(demo.lookups().length, 0);
    await action(first, 'Check original API');
    assert.equal(demo.lookups()[0].url, 'https://first.invalid/api/explain/unconfirmed-A');
    assert.match(status(first).textContent, /first\.invalid/);
    assert.equal(demo.posts().length, 2);
  });

  test('the same ID on two different APIs remains two separate original API receipts', async () => {
    const demo = await createDemo({ apiBase: 'https://first.invalid/api', predictions: [unconfirmed(), unconfirmed()] });
    await demo.submit('shared-ID');
    demo.input('api-base', 'https://second.invalid/api');
    await demo.reconnect();
    await demo.submit('shared-ID');
    const first = rowFor(demo, 'shared-ID', 'https://first.invalid/api');
    const second = rowFor(demo, 'shared-ID', 'https://second.invalid/api');
    assert.equal(rows(demo).length, 2);
    await action(first, 'Check original API');
    await action(second, 'Check original API');
    assert.deepEqual(demo.lookups().map((request) => request.url), [
      'https://first.invalid/api/explain/shared-ID',
      'https://second.invalid/api/explain/shared-ID',
    ]);
    assert.equal(demo.posts().length, 2);
  });

  test('receipt actions name the ID and original API for screen reader navigation', async () => {
    const demo = await createDemo({ apiBase: 'https://first.invalid/api', predictions: [unconfirmed(), unconfirmed()] });
    await demo.submit('shared-ID');
    demo.input('api-base', 'https://second.invalid/api');
    await demo.reconnect();
    await demo.submit('shared-ID');
    const labels = [];
    for (const api of ['https://first.invalid/api', 'https://second.invalid/api']) {
      const row = rowFor(demo, 'shared-ID', api);
      for (const visible of ['Check original API', 'Remove from page']) {
        const button = row.children.find((child) => child.textContent === visible);
        assert.ok(button);
        const label = button.attributes.get('aria-label');
        assert.equal(label, `${visible} for ID shared-ID. Original API: ${api}.`);
        labels.push(label);
      }
    }
    assert.equal(new Set(labels).size, 4);
  });

  test('same-origin blank and explicit API URLs cannot bypass the retained ID guard', async () => {
    for (const [firstApi, secondApi] of [['', 'https://page.invalid/'], ['https://page.invalid/', '']]) {
      const demo = await createDemo({ apiBase: firstApi, predictions: [unconfirmed()] });
      await demo.submit('shared-ID');
      demo.input('api-base', secondApi);
      await demo.reconnect();
      demo.input('features', '[1,2,3]');
      await demo.submitCurrent();
      assert.equal(demo.posts().length, 1, `${firstApi} and ${secondApi} address the same page origin`);
      assert.match(demo.node('input-error').textContent, /recovery receipt/i);
      assert.equal(rows(demo).length, 1);
    }
  });

  test('a distinct path or origin is not treated as the original page API', async () => {
    for (const otherApi of ['https://page.invalid/other', 'https://other.invalid']) {
      const demo = await createDemo({ predictions: [unconfirmed(), unconfirmed()] });
      await demo.submit('shared-ID');
      demo.input('api-base', otherApi);
      await demo.reconnect();
      await demo.submit('shared-ID');
      assert.equal(demo.posts().length, 2);
      rowFor(demo, 'shared-ID', 'https://page.invalid');
      rowFor(demo, 'shared-ID', otherApi);
      assert.equal(rows(demo).length, 2);
    }
  });

  test('a response redirect does not make separately selected API bases equivalent', async () => {
    const demo = await createDemo({ apiBase: 'https://redirect.invalid', predictions: [
      { ...unconfirmed(), redirectedTo: 'https://page.invalid/predict' }, unconfirmed(),
    ] });
    await demo.submit('shared-ID');
    demo.input('api-base', '');
    await demo.reconnect();
    await demo.submit('shared-ID');
    assert.equal(demo.posts().length, 2);
    rowFor(demo, 'shared-ID', 'https://redirect.invalid');
    rowFor(demo, 'shared-ID', 'https://page.invalid');
    assert.equal(rows(demo).length, 2);
  });

  test('removing one receipt is deliberate and does not contact either API', async () => {
    const demo = await createDemo({ predictions: [unconfirmed(), unconfirmed()] });
    await demo.submit('unconfirmed-A');
    await demo.submit('unconfirmed-B');
    const first = rowFor(demo, 'unconfirmed-A', 'https://page.invalid');
    await action(first, 'Remove from page');
    assert.equal(rows(demo).length, 1);
    rowFor(demo, 'unconfirmed-B', 'https://page.invalid');
    assert.equal(demo.posts().length, 2);
    assert.equal(demo.lookups().length, 0);
  });

  test('a receipt never displays an explanation for a different ID', async () => {
    const demo = await createDemo({ predictions: [unconfirmed()], explanations: [{ data: { transaction_id: 'other-ID', status: 'COMPLETED', private_value: 919 } }] });
    await demo.submit('unconfirmed-A');
    const row = rowFor(demo, 'unconfirmed-A', 'https://page.invalid');
    await action(row, 'Check original API');
    assert.match(status(row).textContent, /missing or different transaction ID/i);
    assert.ok(!row.children.some((child) => child.textContent.includes('919')));
  });

  test('edited features cannot reuse an ID while its unconfirmed receipt remains', async () => {
    const demo = await createDemo({ apiBase: 'https://first.invalid/api', predictions: [unconfirmed()] });
    await demo.submit('unconfirmed-A', '[0,0,100]');
    demo.input('features', '[1,2,3]');
    await demo.submitCurrent();
    assert.equal(demo.posts().length, 1);
    assert.match(demo.node('input-error').textContent, /recovery receipt/i);
    assert.match(demo.node('input-error').textContent, /not sent/i);
    const first = rowFor(demo, 'unconfirmed-A', 'https://first.invalid/api');
    assert.equal(demo.node('transaction-id').value, 'unconfirmed-A');
    await action(first, 'Remove from page');
    await demo.submitCurrent();
    assert.equal(demo.posts().length, 2);
    assert.equal(demo.posts()[1].payload.transaction_id, 'unconfirmed-A');
    assert.deepEqual(demo.posts()[1].payload.features, [1, 2, 3]);
  });

  test('a programmatic edit during a pending POST cannot expose another transaction error', async () => {
    let release;
    const deferred = new Promise((resolve) => { release = resolve; });
    const demo = await createDemo({ predictions: [{ deferred }] });
    const pending = demo.submit('unconfirmed-A');
    await demo.flush();
    demo.input('features', '[1,2,3]');
    release({ status: 503, data: { detail: { transaction_id: 'other-ID', status: 'FAILED', message: 'Unrelated private detail' } } });
    await pending;
    assert.match(demo.node('input-error').textContent, /missing or different transaction ID/i);
    assert.doesNotMatch(demo.node('input-error').textContent, /Unrelated private detail/);
    rowFor(demo, 'unconfirmed-A', 'https://page.invalid');
    assert.equal(demo.posts().length, 1);
  });

  test('a valid response from the original API is not displayed after the API input changes', async () => {
    let release;
    const deferred = new Promise((resolve) => { release = resolve; });
    const demo = await createDemo({ apiBase: 'https://first.invalid/api', predictions: [{ deferred }] });
    const pending = demo.submit('unconfirmed-A');
    await demo.flush();
    demo.input('api-base', 'https://second.invalid/api');
    release({ data: { transaction_id: 'unconfirmed-A', prediction: 1, score: 0.75,
      correlation_id: '00000000-0000-4000-8000-000000000001', explanation_status: 'Calculation queued' } });
    await pending;
    assert.equal(demo.node('result').hidden, true);
    assert.equal(demo.node('result-id').textContent, '');
    const row = rowFor(demo, 'unconfirmed-A', 'https://first.invalid/api');
    assert.ok(row.children.some((child) => /result.*not displayed.*input/i.test(child.textContent)));
    assert.equal(demo.lookups().length, 0);
  });

  test('a request already sent at the receipt boundary can still be recorded after its input changes', async () => {
    let release;
    const deferred = new Promise((resolve) => { release = resolve; });
    const demo = await createDemo({ predictions: [...Array.from({ length: 9 }, unconfirmed), { deferred }] });
    for (let i = 1; i <= 9; i++) await demo.submit(`unconfirmed-${i}`);
    const pending = demo.submit('unconfirmed-10');
    await demo.flush();
    demo.input('features', '[1,2,3]');
    release({ data: { transaction_id: 'unconfirmed-10', prediction: 1, score: 0.75,
      correlation_id: '00000000-0000-4000-8000-000000000001', explanation_status: 'Calculation queued' } });
    await pending;
    assert.equal(demo.posts().length, 10);
    assert.equal(rows(demo).length, 10);
    const row = rowFor(demo, 'unconfirmed-10', 'https://page.invalid');
    assert.ok(row.children.some((child) => /result.*not displayed.*input/i.test(child.textContent)));
    await demo.submit('unconfirmed-11');
    assert.equal(demo.posts().length, 10);
    rowFor(demo, 'unconfirmed-1', 'https://page.invalid');
  });

  test('the receipt limit blocks another distinct POST without evicting earlier IDs', async () => {
    const demo = await createDemo({ predictions: Array.from({ length: 11 }, unconfirmed) });
    for (let i = 1; i <= 10; i++) await demo.submit(`unconfirmed-${i}`);
    assert.equal(demo.posts().length, 10);
    await demo.submit('unconfirmed-11');
    assert.equal(demo.posts().length, 10);
    assert.match(demo.node('input-error').textContent, /remove.*receipt|receipt.*limit/i);
    assert.equal(rows(demo).length, 10);
    const first = rowFor(demo, 'unconfirmed-1', 'https://page.invalid');
    rowFor(demo, 'unconfirmed-10', 'https://page.invalid');
    await action(first, 'Remove from page');
    await demo.submit('unconfirmed-11');
    assert.equal(demo.posts().length, 11);
    assert.equal(rows(demo).length, 10);
    rowFor(demo, 'unconfirmed-11', 'https://page.invalid');
  });

  test('a late lookup response cannot recreate an explicitly removed receipt', async () => {
    let release;
    const deferred = new Promise((resolve) => { release = resolve; });
    const demo = await createDemo({ predictions: [unconfirmed()], explanations: [{ deferred }] });
    await demo.submit('unconfirmed-A');
    const first = rowFor(demo, 'unconfirmed-A', 'https://page.invalid');
    const pending = action(first, 'Check original API');
    await demo.flush();
    await action(first, 'Remove from page');
    release({ data: { transaction_id: 'unconfirmed-A', status: 'COMPLETED', private_value: 919 } });
    await pending;
    assert.equal(rows(demo).length, 0);
    assert.equal(demo.visible('unconfirmed-receipts'), false);
  });

  test('a fresh page session does not restore earlier in-memory receipts', async () => {
    const first = await createDemo({ predictions: [unconfirmed()] });
    await first.submit('unconfirmed-A');
    rowFor(first, 'unconfirmed-A', 'https://page.invalid');
    const fresh = await createDemo();
    assert.equal(rows(fresh).length, 0);
    assert.equal(fresh.visible('unconfirmed-receipts'), false);
    assert.equal(fresh.posts().length, 0);
  });
}
