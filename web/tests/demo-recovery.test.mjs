import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import { test } from 'node:test';
import { createDemo, ambiguous } from './demo-harness.mjs';

function onlyLookup(demo, url) {
  assert.equal(demo.lookups().length, 1);
  assert.equal(demo.lookups()[0].url, url);
}

// Plain `node --test` remains supported. The exact-module VM feature is enabled
// only in this bounded, reaped test child; no global Node flag is required.
if (typeof vm.SourceTextModule !== 'function') {
  test('actual demo recovery checks in an isolated module runner', (context) => {
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
  test('successful scoring keeps deliberate pending-status lookup without polling', async () => {
    const demo = await createDemo();
    await demo.submit('success-id');
    assert.equal(demo.node('result').hidden, false);
    assert.equal(demo.node('explanation-button').disabled, false);
    assert.equal(demo.visible('explanation-button'), true);
    assert.equal(demo.lookups().length, 0);
    await demo.lookup();
    onlyLookup(demo, '/explain/success-id');
    assert.match(demo.node('explanation-message').textContent, /pending/i);
    assert.equal(demo.posts().length, 1);
  });

  test('ambiguous confirmation exposes original ID and base lookup without a score', async () => {
    const demo = await createDemo({ apiBase: 'https://fixture.invalid/api/', predictions: [ambiguous] });
    await demo.submit('uncertain-id');
    assert.equal(demo.node('result').hidden, true);
    assert.equal(demo.node('transaction-id').value, 'uncertain-id');
    assert.equal(demo.node('explanation-button').disabled, false);
    assert.equal(demo.visible('explanation-button'), true, 'Lookup must be outside every hidden score ancestor');
    assert.equal(demo.lookups().length, 0);
    await demo.lookup();
    onlyLookup(demo, 'https://fixture.invalid/api/explain/uncertain-id');
    assert.equal(demo.posts().length, 1);
    assert.equal(demo.node('result').hidden, true);
  });

  test('timed-out confirmation retains explicit lookup and never retries the POST', async () => {
    const demo = await createDemo({ predictions: [{ timeout: true }] });
    const submission = demo.submit('timeout-id');
    await demo.flush();
    demo.expireTimeouts();
    await submission;
    assert.match(demo.node('input-error').textContent, /may still have been processed/);
    assert.equal(demo.visible('explanation-button'), true);
    assert.equal(demo.node('explanation-button').disabled, false);
    await demo.lookup();
    onlyLookup(demo, '/explain/timeout-id');
    assert.equal(demo.posts().length, 1);
  });

  test('a non-JSON failure retains the same known-ID recovery path', async () => {
    const demo = await createDemo({ predictions: [{ status: 500, nonJson: true }] });
    await demo.submit('non-json-id');
    assert.equal(demo.node('result').hidden, true);
    assert.equal(demo.visible('explanation-button'), true);
    await demo.lookup();
    onlyLookup(demo, '/explain/non-json-id');
  });

  test('a later failed request clears a prior score and looks up only the new ID', async () => {
    const demo = await createDemo({ predictions: [{}, ambiguous] });
    await demo.submit('old-success');
    await demo.submit('new-uncertain');
    assert.equal(demo.node('result').hidden, true);
    assert.equal(demo.node('probability').textContent, '');
    assert.equal(demo.node('raw-response').textContent, '');
    await demo.lookup();
    onlyLookup(demo, '/explain/new-uncertain');
    assert.equal(demo.posts().length, 2);
  });

  for (const field of ['features', 'transaction-id', 'api-base']) {
    test(`${field} edits clear old score and recovery context`, async () => {
      const demo = await createDemo();
      await demo.submit('old-id');
      demo.input(field, field === 'api-base' ? 'https://other.invalid' : 'changed');
      assert.equal(demo.node('result').hidden, true);
      assert.equal(demo.node('explanation-button').disabled, true);
      assert.equal(demo.visible('explanation-button'), false);
      assert.equal(demo.node('probability').textContent, '');
      await demo.lookup();
      assert.equal(demo.lookups().length, 0);
    });
  }

  test('reconnecting clears prior model recovery context', async () => {
    const demo = await createDemo();
    await demo.submit('old-id');
    await demo.reconnect();
    assert.equal(demo.node('result').hidden, true);
    assert.equal(demo.visible('explanation-button'), false);
    await demo.lookup();
    assert.equal(demo.lookups().length, 0);
  });

  test('blank ID without UUID support never invents a recovery query', async () => {
    const demo = await createDemo({ uuid: false, predictions: [ambiguous] });
    await demo.submit('');
    assert.equal(demo.posts()[0].payload.transaction_id, undefined);
    assert.equal(demo.node('transaction-id').value, '');
    assert.equal(demo.visible('explanation-button'), false);
    await demo.lookup();
    assert.equal(demo.lookups().length, 0);
  });

  test('a generated known ID supports recovery after failed confirmation', async () => {
    const demo = await createDemo({ predictions: [ambiguous] });
    await demo.submit('');
    assert.equal(demo.posts()[0].payload.transaction_id, 'generated-fixture-id');
    assert.equal(demo.visible('explanation-button'), true);
    await demo.lookup();
    onlyLookup(demo, '/explain/generated-fixture-id');
  });

  test('successful server-generated ID still enables the normal explicit lookup', async () => {
    const demo = await createDemo({ uuid: false });
    await demo.submit('');
    assert.equal(demo.node('transaction-id').value, 'server-generated-fixture-id');
    assert.equal(demo.visible('explanation-button'), true);
    await demo.lookup();
    onlyLookup(demo, '/explain/server-generated-fixture-id');
  });

  test('validated success associates lookup with the returned ID instead of the submitted ID', async () => {
    const demo = await createDemo({ predictions: [{ data: {
      transaction_id: 'returned-id-B', prediction: 1, score: 0.75,
      correlation_id: '00000000-0000-4000-8000-000000000001', explanation_status: 'Calculation queued',
    } }] });
    await demo.submit('submitted-id-A');
    assert.equal(demo.node('result-id').textContent, 'returned-id-B');
    assert.equal(demo.node('transaction-id').value, 'returned-id-B');
    await demo.lookup();
    onlyLookup(demo, '/explain/returned-id-B');
    assert.equal(demo.lookups().some((request) => request.url.endsWith('/submitted-id-A')), false);
  });

  test('a returned ID outside lookup rules cannot retain an earlier request context', async () => {
    for (const returnedId of ['bad/id', 'x'.repeat(129)]) {
      const demo = await createDemo({ predictions: [{}, { data: {
        transaction_id: returnedId, prediction: 1, score: 0.75,
        correlation_id: '00000000-0000-4000-8000-000000000001', explanation_status: 'Calculation queued',
      } }] });
      await demo.submit('prior-success');
      await demo.submit('submitted-id-A');
      assert.equal(demo.node('result-id').textContent, returnedId);
      assert.equal(demo.node('probability').textContent, '75.00%');
      assert.equal(demo.node('explanation-button').disabled, true);
      assert.equal(demo.visible('explanation-button'), false);
      await demo.lookup();
      assert.equal(demo.lookups().length, 0);
    }
  });

  test('invalid input clears recovery and never submits or queries', async () => {
    const demo = await createDemo();
    await demo.submit('old-id');
    await demo.submit('new-id', '[1]');
    assert.equal(demo.posts().length, 1);
    assert.equal(demo.visible('explanation-button'), false);
    await demo.lookup();
    assert.equal(demo.lookups().length, 0);
  });

  test('invalid transaction IDs never become a POST or lookup context', async () => {
    for (const id of ['bad/id', 'x'.repeat(129)]) {
      const demo = await createDemo();
      await demo.submit(id);
      assert.equal(demo.posts().length, 0);
      assert.equal(demo.node('transaction-id').attributes.get('aria-invalid'), 'true');
      assert.equal(demo.visible('explanation-button'), false);
      await demo.lookup();
      assert.equal(demo.lookups().length, 0);
    }
  });

  for (const fixture of [
    { status: 404, data: { detail: 'SHAP explanation not found' }, message: /No saved explanation/ },
    { status: 503, data: { detail: { status: 'FAILED', message: 'Explanation calculation failed' } }, message: /FAILED/ },
    { status: 200, data: { transaction_id: 'recover-id', status: 'COMPLETED', shap_values: [0, 0, 0], feature_names: ['Time', 'V1', 'Amount'], base_value: 0, explanation_method: 'fixed fixture' }, message: /Saved response/ },
  ]) {
    test(`explicit ${fixture.status} saved-status response never restores a prediction`, async () => {
      const demo = await createDemo({ predictions: [ambiguous], explanations: [fixture] });
      await demo.submit('recover-id');
      await demo.lookup();
      assert.equal(demo.lookups().length, 1);
      assert.match(demo.node('explanation-message').textContent, fixture.message);
      assert.equal(demo.node('result').hidden, true);
      assert.equal(demo.node('probability').textContent, '');
      assert.equal(demo.posts().length, 1);
    });
  }

  test('busy requests prevent a second POST or concurrent lookup', async () => {
    let resolve;
    const deferred = new Promise((done) => { resolve = done; });
    const demo = await createDemo({ predictions: [{ deferred }] });
    const submission = demo.submit('busy-id');
    await demo.flush();
    assert.equal(demo.node('score-button').disabled, true);
    await demo.submitCurrent();
    await demo.lookup();
    assert.equal(demo.posts().length, 1);
    assert.equal(demo.lookups().length, 0);
    resolve(ambiguous);
    await submission;
    await demo.lookup();
    onlyLookup(demo, '/explain/busy-id');
    assert.equal(demo.posts().length, 1);
  });

  test('an explicit lookup in progress cannot issue a second GET', async () => {
    let resolve;
    const deferred = new Promise((done) => { resolve = done; });
    const demo = await createDemo({ predictions: [ambiguous], explanations: [{ deferred }] });
    await demo.submit('lookup-id');
    const lookup = demo.lookup();
    await demo.flush();
    assert.equal(demo.node('explanation-button').disabled, true);
    await demo.lookup();
    assert.equal(demo.lookups().length, 1);
    resolve({ status: 202, data: { transaction_id: 'lookup-id', status: 'PENDING' } });
    await lookup;
    assert.equal(demo.lookups().length, 1);
    assert.equal(demo.posts().length, 1);
    assert.equal(demo.node('result').hidden, true);
  });
}
