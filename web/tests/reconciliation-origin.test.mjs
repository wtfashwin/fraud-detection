import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import { test } from 'node:test';
import { createDemo } from './demo-harness.mjs';

if (typeof vm.SourceTextModule !== 'function') {
  test('origin and unconfirmed request checks in an isolated module runner', (context) => {
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
const warningCount = (text) => (text.match(/\b(?:may|might|could) (?:still |already )?have (?:already )?been processed\b/giu) || []).length;

test('an unconfirmed scoring error explains that processing may have occurred', async () => {
  const demo = await createDemo({ apiBase: 'https://first.invalid/api/', predictions: [
    { status: 503, data: { detail: 'Synthetic queue unavailable' } },
  ] });
  await demo.submit('request-one');
  assert.match(demo.node('input-error').textContent, /Synthetic queue unavailable/);
  assert.match(demo.node('input-error').textContent, /may (?:still )?have been processed/i);
  assert.equal(warningCount(demo.node('input-error').textContent), 1);
  assert.equal(demo.posts().length, 1);
  assert.equal(demo.visible('explanation-button'), true);
  assert.equal(demo.visible('unconfirmed-receipts'), true);
});

test('a mismatched prediction ID warns of uncertainty without exposing unrelated data', async () => {
  const demo = await createDemo({ predictions: [{ data: {
    transaction_id: 'unrelated-id', prediction: 1, score: 0.75,
    correlation_id: '00000000-0000-4000-8000-000000000001', explanation_status: 'Calculation queued',
  } }] });
  await demo.submit('request-one');
  assert.equal(demo.node('result').hidden, true);
  assert.match(demo.node('input-error').textContent, /different transaction ID/i);
  assert.match(demo.node('input-error').textContent, /may (?:still )?have been processed/i);
  assert.equal(warningCount(demo.node('input-error').textContent), 1);
  assert.doesNotMatch(demo.node('input-error').textContent, /unrelated-id/);
  assert.equal(demo.posts().length, 1);
});

test('a timed out request warns about processing without sending a second prediction', async () => {
  const demo = await createDemo({ predictions: [{ timeout: true }] });
  const pending = demo.submit('request-one');
  await demo.flush();
  demo.expireTimeouts();
  await pending;
  assert.match(demo.node('input-error').textContent, /may (?:still )?have been processed/i);
  assert.equal(warningCount(demo.node('input-error').textContent), 1);
  assert.match(demo.node('input-error').textContent, /15 seconds/);
  assert.match(demo.node('input-error').textContent, /transaction ID before retrying/i);
  assert.equal(demo.posts().length, 1);
  assert.equal(demo.visible('unconfirmed-receipts'), true);
});

test('a network failure keeps actionable detail and one processing warning', async () => {
  const demo = await createDemo({ predictions: [{ networkFailure: true }] });
  await demo.submit('request-one');
  assert.match(demo.node('input-error').textContent, /Cannot reach the API/i);
  assert.equal(warningCount(demo.node('input-error').textContent), 1);
  assert.equal(demo.posts().length, 1);
  assert.equal(demo.visible('unconfirmed-receipts'), true);
});

test('a duplicate conflict keeps recovery guidance and one processing warning', async () => {
  const demo = await createDemo({ predictions: [{ status: 409, data: { detail: 'Transaction ID already exists; use a new ID' } }] });
  await demo.submit('request-one');
  assert.match(demo.node('input-error').textContent, /check the saved explanation/i);
  assert.match(demo.node('input-error').textContent, /new ID only for a different transaction/i);
  assert.equal(warningCount(demo.node('input-error').textContent), 1);
  assert.equal(demo.posts().length, 1);
});

test('a backend processing warning remains visible alongside the application caution', async () => {
  const demo = await createDemo({ predictions: [{ status: 503, data: { detail: 'This request might already have been processed. Check its saved status.' } }] });
  await demo.submit('request-one');
  assert.match(demo.node('input-error').textContent, /This request might already have been processed/i);
  assert.match(demo.node('input-error').textContent, /Check its saved status/i);
  assert.match(demo.node('input-error').textContent, /The request may have been processed\./);
  assert.equal(demo.posts().length, 1);
});

test('a negated backend warning cannot suppress the application caution', async () => {
  const demo = await createDemo({ predictions: [{ status: 503, data: { detail: 'It is false that this request may have been processed.' } }] });
  await demo.submit('request-one');
  assert.match(demo.node('input-error').textContent, /It is false that this request may have been processed/);
  assert.match(demo.node('input-error').textContent, /The request may have been processed\./);
  assert.equal(demo.posts().length, 1);
});

test('HTTP text imitating the timeout warning cannot suppress the application caution', async () => {
  const demo = await createDemo({ predictions: [{ status: 503, data: { detail: 'A submitted request may still have been processed; check its transaction ID before retrying.' } }] });
  await demo.submit('request-one');
  assert.match(demo.node('input-error').textContent, /A submitted request may still have been processed/);
  assert.match(demo.node('input-error').textContent, /The request may have been processed\./);
  assert.equal(demo.posts().length, 1);
});

test('a warning about a different transaction does not replace the current request warning', async () => {
  const demo = await createDemo({ predictions: [{ status: 503, data: { detail: 'A different transaction may have been processed.' } }] });
  await demo.submit('request-one');
  assert.match(demo.node('input-error').textContent, /A different transaction may have been processed/i);
  assert.match(demo.node('input-error').textContent, /The request may have been processed/i);
  assert.equal(demo.posts().length, 1);
});

test('a failed response after an input change warns about the previous request', async () => {
  let release;
  const deferred = new Promise((resolve) => { release = resolve; });
  const demo = await createDemo({ predictions: [{ deferred }] });
  const pending = demo.submit('request-one');
  await demo.flush();
  demo.input('features', '[1, 0, 100]');
  release({ status: 503, data: { detail: 'Synthetic queue unavailable' } });
  await pending;
  assert.match(demo.node('input-error').textContent, /previous request may have been processed/i);
  assert.match(demo.node('input-error').textContent, /original API receipt/i);
  assert.equal(demo.posts().length, 1);
  assert.equal(demo.visible('unconfirmed-receipts'), true);
});

for (const [name, explanation] of [
  ['pending', { status: 202, data: { transaction_id: 'request-one', status: 'PENDING' } }],
  ['completed', { data: { transaction_id: 'request-one', status: 'COMPLETED', shap_values: [1], feature_names: ['Amount'], base_value: 0, explanation_method: 'synthetic', created_at: '2026-01-01T00:00:00Z' } }],
  ['failed', { data: { transaction_id: 'request-one', status: 'FAILED' } }],
  ['missing', { status: 404, data: { detail: 'SHAP explanation not found' } }],
  ['service error', { status: 503, data: { detail: 'Synthetic explanation unavailable' } }],
]) {
  test(`original ${name} lookup identifies the submitted ID and original API`, async () => {
    const demo = await createDemo({ apiBase: 'https://first.invalid/api/', explanations: [explanation] });
    await demo.submit('request-one');
    await demo.lookup();
    assert.equal(demo.lookups()[0].url, 'https://first.invalid/api/explain/request-one');
    assert.match(demo.node('explanation-message').textContent, /request-one/);
    assert.match(demo.node('explanation-message').textContent, /Original API: https:\/\/first\.invalid\/api/);
    assert.doesNotMatch(demo.node('explanation-message').textContent, /responding server is/i);
    assert.equal(demo.posts().length, 1);
  });
}

test('a same-origin original lookup labels the page origin without claiming the responding server', async () => {
  const demo = await createDemo();
  await demo.submit('request-one');
  await demo.lookup();
  assert.match(demo.node('explanation-message').textContent, /request-one/);
  assert.match(demo.node('explanation-message').textContent, /Original API: https:\/\/page\.invalid/i);
  assert.doesNotMatch(demo.node('explanation-message').textContent, /responding server is/i);
});

test('a same-origin manual lookup displays the selected page origin', async () => {
  const demo = await createDemo({ pageOrigin: 'http://127.0.0.1:8000' });
  demo.input('manual-transaction-id', 'manual-one');
  await demo.manualLookup();
  assert.match(demo.node('manual-explanation-message').textContent, /Selected API: http:\/\/127\.0\.0\.1:8000/);
  assert.equal(demo.posts().length, 0);
});

test('a same-origin unconfirmed receipt and its action names include the page origin', async () => {
  const demo = await createDemo({ predictions: [{ status: 503, data: { detail: 'Synthetic unavailable' } }] });
  await demo.submit('request-one');
  const row = demo.node('unconfirmed-receipt-list').children[0];
  assert.ok(row.children.some((child) => child.textContent.includes('Original API: https://page.invalid')));
  const buttons = row.children.filter((child) => child.attributes.has('aria-label'));
  assert.equal(buttons.length, 2);
  for (const button of buttons) assert.match(button.attributes.get('aria-label'), /Original API: https:\/\/page\.invalid/);
});

test('opaque or unsafe page origins fall back without exposing credentials or paths', async () => {
  for (const pageOrigin of [null, 'null', 'file://', 'https://user:secret@invalid.test', 'https://page.invalid/path?token=secret']) {
    const demo = await createDemo({ pageOrigin, predictions: [{ status: 503, data: { detail: 'Synthetic unavailable' } }] });
    await demo.submit('request-one');
    const row = demo.node('unconfirmed-receipt-list').children[0];
    const receiptText = row.children.map((child) => child.textContent).join(' ');
    const actionNames = row.children.filter((child) => child.attributes.has('aria-label')).map((child) => child.attributes.get('aria-label')).join(' ');
    assert.match(receiptText, /Original API: this server/i);
    assert.match(actionNames, /Original API: this server/i);
    await demo.lookup();
    assert.match(demo.node('explanation-message').textContent, /Original API: this server/i);
    demo.input('manual-transaction-id', 'manual-one');
    await demo.manualLookup();
    assert.match(demo.node('manual-explanation-message').textContent, /Selected API: this server/i);
    assert.doesNotMatch(`${receiptText} ${actionNames} ${demo.node('explanation-message').textContent} ${demo.node('manual-explanation-message').textContent}`, /secret|token=/i);
  }
});

test('the in-flight original lookup names its source and leaves manual lookup separate', async () => {
  let release;
  const deferred = new Promise((resolve) => { release = resolve; });
  const demo = await createDemo({ apiBase: 'https://first.invalid/api', explanations: [{ deferred }] });
  await demo.submit('request-one');
  demo.input('manual-transaction-id', 'another-id');
  const pending = demo.lookup();
  await demo.flush();
  assert.match(demo.node('explanation-message').textContent, /request-one/);
  assert.match(demo.node('explanation-message').textContent, /Original API: https:\/\/first\.invalid\/api/);
  assert.equal(demo.node('manual-explanation-button').disabled, true);
  release({ data: { transaction_id: 'request-one', status: 'PENDING' } });
  await pending;
  assert.equal(demo.node('manual-transaction-id').value, 'another-id');
  assert.equal(demo.node('manual-explanation-button').disabled, false);
  assert.equal(demo.posts().length, 1);
});
}
