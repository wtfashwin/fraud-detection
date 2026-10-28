import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import { test } from 'node:test';
import { createDemo } from './demo-harness.mjs';

if (typeof vm.SourceTextModule !== 'function') {
  test('manual lookup checks in an isolated module runner', (context) => {
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
  const completed = (id, value) => ({ transaction_id: id, status: 'COMPLETED', shap_values: [value],
    feature_names: ['Amount'], base_value: 0, explanation_method: 'synthetic', created_at: '2026-01-01T00:00:00Z' });

  test('a known ID can be deliberately checked on the selected API without scoring', async () => {
    const demo = await createDemo({ apiBase: 'https://first.invalid/api/' });
    demo.input('manual-transaction-id', 'saved-id');
    assert.equal(demo.node('manual-explanation-button').disabled, false);
    assert.equal(demo.lookups().length, 0);
    await demo.manualLookup();
    assert.equal(demo.lookups().length, 1);
    assert.equal(demo.lookups()[0].url, 'https://first.invalid/api/explain/saved-id');
    assert.equal(demo.posts().length, 0);
    assert.equal(demo.node('result').hidden, true);
    assert.match(demo.node('manual-explanation-message').textContent, /pending/i);
    assert.match(demo.node('manual-explanation-message').textContent, /Selected API: https:\/\/first\.invalid\/api/);
  });

  test('manual lookup is available when model metadata is unavailable', async () => {
    const demo = await createDemo({ apiBase: 'https://first.invalid/api', modelInfoStatus: 503 });
    demo.input('manual-transaction-id', 'saved-id');
    assert.equal(demo.node('score-button').disabled, true);
    assert.equal(demo.node('manual-explanation-button').disabled, false);
    await demo.manualLookup();
    assert.equal(demo.lookups().length, 1);
    assert.equal(demo.lookups()[0].url, 'https://first.invalid/api/explain/saved-id');
    assert.equal(demo.posts().length, 0);
  });

  test('changing API clears old manual output and uses the new API after reconnect', async () => {
    const demo = await createDemo({ apiBase: 'https://first.invalid/api', explanations: [
      { data: completed('saved-id', 1) }, { data: completed('saved-id', 2) },
    ] });
    demo.input('manual-transaction-id', 'saved-id');
    await demo.manualLookup();
    assert.equal(demo.lookups().length, 1);
    assert.equal(JSON.parse(demo.node('manual-explanation-response').textContent).shap_values[0], 1);
    demo.input('api-base', 'https://second.invalid/service');
    assert.equal(demo.node('manual-explanation-response').hidden, true);
    assert.equal(demo.node('manual-explanation-response').textContent, '');
    assert.equal(demo.node('manual-explanation-button').disabled, true);
    await demo.manualLookup();
    assert.equal(demo.lookups().length, 1);
    await demo.reconnect();
    await demo.manualLookup();
    assert.equal(demo.lookups()[1].url, 'https://second.invalid/service/explain/saved-id');
    assert.equal(JSON.parse(demo.node('manual-explanation-response').textContent).shap_values[0], 2);
    assert.match(demo.node('manual-explanation-message').textContent, /Selected API: https:\/\/second\.invalid\/service/);
  });

  test('editing the manual ID clears the prior result without making another request', async () => {
    const demo = await createDemo({ explanations: [{ data: completed('saved-id', 1) }] });
    demo.input('manual-transaction-id', 'saved-id');
    await demo.manualLookup();
    assert.equal(demo.lookups().length, 1);
    demo.input('manual-transaction-id', 'next-id');
    assert.equal(demo.node('manual-explanation-response').hidden, true);
    assert.equal(demo.node('manual-explanation-response').textContent, '');
    assert.equal(demo.node('manual-explanation-message').textContent, '');
    assert.equal(demo.lookups().length, 1);
  });

  test('a late response cannot restore output for a changed manual ID', async () => {
    let release;
    const deferred = new Promise((resolve) => { release = resolve; });
    const demo = await createDemo({ explanations: [{ deferred }] });
    demo.input('manual-transaction-id', 'saved-id');
    const pending = demo.manualLookup();
    await demo.flush();
    assert.equal(demo.lookups().length, 1);
    demo.input('manual-transaction-id', 'next-id');
    release({ data: completed('saved-id', 7) });
    await pending;
    assert.equal(demo.node('manual-explanation-response').hidden, true);
    assert.equal(demo.node('manual-explanation-response').textContent, '');
    assert.equal(demo.node('manual-explanation-message').textContent, '');
    assert.equal(demo.node('manual-explanation-button').disabled, false);
  });

  test('a late response from an old API cannot appear after the API input changes', async () => {
    let release;
    const deferred = new Promise((resolve) => { release = resolve; });
    const demo = await createDemo({ apiBase: 'https://first.invalid/api', explanations: [{ deferred }] });
    demo.input('manual-transaction-id', 'saved-id');
    const pending = demo.manualLookup();
    await demo.flush();
    assert.equal(demo.lookups().length, 1);
    demo.input('api-base', 'https://second.invalid/api');
    release({ data: completed('saved-id', 7) });
    await pending;
    assert.equal(demo.node('manual-explanation-response').hidden, true);
    assert.equal(demo.node('manual-explanation-response').textContent, '');
    assert.equal(demo.node('manual-explanation-message').textContent, '');
    assert.equal(demo.node('manual-explanation-button').disabled, true);
  });

  test('manual lookup rejects a successful explanation for a different ID', async () => {
    const demo = await createDemo({ explanations: [{ data: completed('other-id', 99) }] });
    demo.input('manual-transaction-id', 'saved-id');
    await demo.manualLookup();
    assert.equal(demo.node('manual-explanation-response').hidden, true);
    assert.match(demo.node('manual-explanation-message').textContent, /different transaction ID/i);
    assert.doesNotMatch(demo.node('manual-explanation-message').textContent, /99/);
  });

  test('manual lookup redacts a mismatched structured 404 instead of calling it missing', async () => {
    const demo = await createDemo({ explanations: [{ status: 404, data: { detail: {
      transaction_id: 'other-id', status: 'FAILED', message: 'unrelated private detail',
    } } }] });
    demo.input('manual-transaction-id', 'saved-id');
    await demo.manualLookup();
    assert.equal(demo.node('manual-explanation-response').hidden, true);
    assert.match(demo.node('manual-explanation-message').textContent, /missing or different transaction ID/i);
    assert.doesNotMatch(demo.node('manual-explanation-message').textContent, /unrelated private detail|not found/i);
  });

  test('ordinary 404 does not claim the original request was unprocessed', async () => {
    const demo = await createDemo({ explanations: [{ status: 404, data: { detail: 'SHAP explanation not found' } }] });
    demo.input('manual-transaction-id', 'saved-id');
    await demo.manualLookup();
    assert.match(demo.node('manual-explanation-message').textContent, /no saved explanation/i);
    assert.match(demo.node('manual-explanation-message').textContent, /does not establish/i);
  });

  test('matching structured 404 retains its status detail rather than claiming an absent record', async () => {
    const demo = await createDemo({ explanations: [{ status: 404, data: { detail: {
      transaction_id: 'saved-id', status: 'FAILED', message: 'Synthetic processing failure',
    } } }] });
    demo.input('manual-transaction-id', 'saved-id');
    await demo.manualLookup();
    assert.match(demo.node('manual-explanation-message').textContent, /Synthetic processing failure/);
    assert.doesNotMatch(demo.node('manual-explanation-message').textContent, /no saved explanation/i);
  });

  test('manual lookup preserves actionable generic 404 details', async () => {
    for (const detail of ['Endpoint not found. Check API base URL.', 'Account not found. Contact administrator.']) {
      const demo = await createDemo({ explanations: [{ status: 404, data: { detail } }] });
      demo.input('manual-transaction-id', 'saved-id');
      await demo.manualLookup();
      assert.ok(demo.node('manual-explanation-message').textContent.includes(detail));
      assert.doesNotMatch(demo.node('manual-explanation-message').textContent, /no saved explanation/i);
    }
  });

  test('a generic service error retains its actionable detail', async () => {
    const demo = await createDemo({ explanations: [{ status: 503, data: { detail: 'Synthetic storage unavailable' } }] });
    demo.input('manual-transaction-id', 'saved-id');
    await demo.manualLookup();
    assert.match(demo.node('manual-explanation-message').textContent, /Synthetic storage unavailable/);
  });

  test('an unrecognized successful status is not presented as a saved explanation', async () => {
    const demo = await createDemo({ explanations: [{ data: { transaction_id: 'saved-id', status: 'UNKNOWN', value: 99 } }] });
    demo.input('manual-transaction-id', 'saved-id');
    await demo.manualLookup();
    assert.equal(demo.node('manual-explanation-response').hidden, true);
    assert.match(demo.node('manual-explanation-message').textContent, /unrecognized.*status/i);
    assert.doesNotMatch(demo.node('manual-explanation-message').textContent, /saved response received/i);
  });

  test('a manual lookup does not replace the original submitted request context', async () => {
    const demo = await createDemo({ apiBase: 'https://first.invalid/api', predictions: [{ status: 503, data: { detail: 'Queue unavailable' } }] });
    await demo.submit('submitted-id');
    demo.input('manual-transaction-id', 'other-saved-id');
    await demo.manualLookup();
    await demo.lookup();
    assert.deepEqual(demo.lookups().map((r) => r.url), [
      'https://first.invalid/api/explain/other-saved-id',
      'https://first.invalid/api/explain/submitted-id',
    ]);
    assert.equal(demo.posts().length, 1);
  });
}
