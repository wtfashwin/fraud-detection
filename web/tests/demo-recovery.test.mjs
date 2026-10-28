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

  test('blank ID without a generator requires a manual ID before recovery can be needed', async () => {
    const demo = await createDemo({ uuid: false, predictions: [{ status: 503, data: { detail: 'Queue unavailable' } }] });
    await demo.submit('');
    assert.equal(demo.posts().length, 0);
    assert.match(demo.node('input-error').textContent, /enter.*transaction ID/i);
    assert.equal(demo.node('transaction-id').value, '');
    assert.equal(demo.visible('explanation-button'), false);
    await demo.lookup();
    assert.equal(demo.lookups().length, 0);
    await demo.submit('manual-recovery-id');
    assert.equal(demo.posts()[0].payload.transaction_id, 'manual-recovery-id');
    await demo.lookup();
    onlyLookup(demo, '/explain/manual-recovery-id');
  });

  test('a generated known ID supports recovery after failed confirmation', async () => {
    const demo = await createDemo({ predictions: [ambiguous] });
    await demo.submit('');
    assert.equal(demo.posts()[0].payload.transaction_id, 'generated-fixture-id');
    assert.equal(demo.visible('explanation-button'), true);
    await demo.lookup();
    onlyLookup(demo, '/explain/generated-fixture-id');
  });

  test('a manual ID without a generator enables the normal explicit lookup', async () => {
    const demo = await createDemo({ uuid: false });
    await demo.submit('manual-id');
    assert.equal(demo.node('transaction-id').value, 'manual-id');
    assert.equal(demo.visible('explanation-button'), true);
    await demo.lookup();
    onlyLookup(demo, '/explain/manual-id');
  });

  test('blank ID cannot retain an earlier lookup and an invalid returned ID cannot replace a new manual ID', async () => {
    const demo = await createDemo({ uuid: false, predictions: [{}, { data: {
      transaction_id: 'bad/id', prediction: 1, score: 0.75,
      correlation_id: '00000000-0000-4000-8000-000000000001', explanation_status: 'Calculation queued',
    } }] });
    await demo.submit('prior-success');
    assert.equal(demo.node('result-id').textContent, 'prior-success');
    await demo.submit('');
    assert.equal(demo.posts().length, 1);
    assert.equal(demo.node('result').hidden, true);
    assert.equal(demo.node('explanation-button').disabled, true);
    assert.equal(demo.visible('explanation-button'), false);
    await demo.lookup();
    assert.equal(demo.lookups().length, 0);
    await demo.submit('next-id');
    assert.equal(demo.posts().length, 2);
    assert.equal(demo.posts()[1].payload.transaction_id, 'next-id');
    assert.equal(demo.node('result').hidden, true);
    assert.match(demo.node('input-error').textContent, /different transaction ID/i);
    await demo.lookup();
    onlyLookup(demo, '/explain/next-id');
  });

  test('a mismatched submitted ID rejects the prediction and retains its original recovery context', async () => {
    const demo = await createDemo({ apiBase: 'https://fixture.invalid/api/', predictions: [{ data: {
      transaction_id: 'returned-id-B', prediction: 1, score: 0.75,
      correlation_id: '00000000-0000-4000-8000-000000000001', explanation_status: 'Calculation queued',
    } }] });
    await demo.submit('submitted-id-A');
    assert.equal(demo.posts()[0].payload.transaction_id, 'submitted-id-A');
    assert.equal(demo.node('result').hidden, true);
    assert.equal(demo.node('result-id').textContent, '');
    assert.equal(demo.node('probability').textContent, '');
    assert.equal(demo.node('transaction-id').value, 'submitted-id-A');
    assert.match(demo.node('input-error').textContent, /different transaction ID/i);
    assert.equal(demo.visible('explanation-button'), true);
    assert.equal(demo.lookups().length, 0);
    await demo.lookup();
    onlyLookup(demo, 'https://fixture.invalid/api/explain/submitted-id-A');
  });

  test('a mismatched browser-generated ID keeps recovery bound to that generated ID', async () => {
    const demo = await createDemo({ predictions: [{ data: {
      transaction_id: 'returned-id-B', prediction: 1, score: 0.75,
      correlation_id: '00000000-0000-4000-8000-000000000001', explanation_status: 'Calculation queued',
    } }] });
    await demo.submit('');
    assert.equal(demo.posts()[0].payload.transaction_id, 'generated-fixture-id');
    assert.equal(demo.node('result').hidden, true);
    assert.equal(demo.node('transaction-id').value, 'generated-fixture-id');
    assert.match(demo.node('input-error').textContent, /different transaction ID/i);
    await demo.lookup();
    onlyLookup(demo, '/explain/generated-fixture-id');
  });

  test('matching browser-generated ID still shows the prediction and supports lookup', async () => {
    const demo = await createDemo();
    await demo.submit('');
    assert.equal(demo.posts()[0].payload.transaction_id, 'generated-fixture-id');
    assert.equal(demo.node('result').hidden, false);
    assert.equal(demo.node('result-id').textContent, 'generated-fixture-id');
    await demo.lookup();
    onlyLookup(demo, '/explain/generated-fixture-id');
  });

  test('a mismatched invalid returned ID cannot replace the submitted recovery context', async () => {
    for (const returnedId of ['bad/id', 'x'.repeat(129)]) {
      const demo = await createDemo({ predictions: [{}, { data: {
        transaction_id: returnedId, prediction: 1, score: 0.75,
        correlation_id: '00000000-0000-4000-8000-000000000001', explanation_status: 'Calculation queued',
      } }] });
      await demo.submit('prior-success');
      await demo.submit('submitted-id-A');
      assert.equal(demo.node('result').hidden, true);
      assert.equal(demo.node('result-id').textContent, '');
      assert.equal(demo.node('probability').textContent, '');
      assert.equal(demo.node('transaction-id').value, 'submitted-id-A');
      assert.match(demo.node('input-error').textContent, /different transaction ID/i);
      assert.equal(demo.visible('explanation-button'), true);
      await demo.lookup();
      onlyLookup(demo, '/explain/submitted-id-A');
    }
  });

  test('mismatched or absent IDs in explanation responses never display unrelated status or values', async () => {
    for (const fixture of [
      { status: 202, data: { transaction_id: 'other-id', status: 'PENDING' } },
      { status: 200, data: { transaction_id: 'other-id', status: 'COMPLETED', shap_values: [3], feature_names: ['Amount'], base_value: 0 } },
      { status: 202, data: { status: 'PENDING' } },
    ]) {
      const demo = await createDemo({ apiBase: 'https://fixture.invalid/api/', predictions: [ambiguous], explanations: [fixture] });
      await demo.submit('recover-id-A');
      await demo.lookup();
      onlyLookup(demo, 'https://fixture.invalid/api/explain/recover-id-A');
      assert.equal(demo.node('explanation-response').hidden, true);
      assert.match(demo.node('explanation-message').textContent, /different transaction ID/i);
      assert.equal(demo.node('result').hidden, true);
      assert.equal(demo.visible('explanation-button'), true);
      await demo.lookup();
      assert.equal(demo.lookups().length, 2);
      assert.equal(demo.lookups()[1].url, 'https://fixture.invalid/api/explain/recover-id-A');
      assert.match(demo.node('explanation-message').textContent, /pending/i);
      assert.equal(demo.posts().length, 1);
    }
  });

  test('structured prediction status errors with missing or different IDs keep the original recovery context', async () => {
    for (const detail of [
      { transaction_id: 'other-id', status: 'DISPATCH_UNCONFIRMED', message: 'Queue state unknown' },
      { status: 'DISPATCH_UNCONFIRMED', message: 'Queue state unknown' },
      { transaction_id: null, status: 'dispatch_unconfirmed', message: 'Queue state unknown' },
    ]) {
      const demo = await createDemo({ apiBase: 'https://fixture.invalid/api/', predictions: [{ status: 503, data: { detail } }] });
      await demo.submit('wanted-id');
      assert.equal(demo.node('result').hidden, true);
      assert.match(demo.node('input-error').textContent, /missing or different transaction ID/i);
      assert.doesNotMatch(demo.node('input-error').textContent, /other-id|DISPATCH_UNCONFIRMED/);
      assert.equal(demo.node('transaction-id').value, 'wanted-id');
      await demo.lookup();
      onlyLookup(demo, 'https://fixture.invalid/api/explain/wanted-id');
    }
  });

  test('matching structured prediction status and generic errors remain informative', async () => {
    for (const [detail, expected] of [
      [{ transaction_id: 'wanted-id', status: 'DISPATCH_UNCONFIRMED', message: 'Queue state unknown' }, /DISPATCH_UNCONFIRMED/],
      ['Explanation storage is unavailable', /Explanation storage is unavailable/],
      [{ message: 'Explanation storage is unavailable' }, /Explanation storage is unavailable/],
      [{ status: 'UNAVAILABLE', message: 'Explanation storage is unavailable' }, /Explanation storage is unavailable/],
    ]) {
      const demo = await createDemo({ predictions: [{ status: 503, data: { detail } }] });
      await demo.submit('wanted-id');
      assert.match(demo.node('input-error').textContent, expected);
      assert.equal(demo.node('result').hidden, true);
      await demo.lookup();
      onlyLookup(demo, '/explain/wanted-id');
    }
  });

  test('a 409 with an unrelated transaction ID never advises retrying with a new ID', async () => {
    const demo = await createDemo({ predictions: [{ status: 409, data: {
      detail: { transaction_id: 'other-id', status: 'FAILED', message: 'Duplicate ID' },
    } }] });
    await demo.submit('wanted-id');
    assert.match(demo.node('input-error').textContent, /missing or different transaction ID/i);
    assert.doesNotMatch(demo.node('input-error').textContent, /other-id|Enter a new transaction ID/i);
    assert.equal(demo.node('result').hidden, true);
    await demo.lookup();
    onlyLookup(demo, '/explain/wanted-id');
    assert.equal(demo.posts().length, 1);
  });

  test('an ordinary 409 advises checking saved status before retrying', async () => {
    const demo = await createDemo({ predictions: [{ status: 409, data: { detail: 'Transaction ID already exists' } }] });
    await demo.submit('wanted-id');
    assert.match(demo.node('input-error').textContent, /check.*saved/i);
    assert.match(demo.node('input-error').textContent, /new ID.*different transaction/i);
    assert.equal(demo.node('result').hidden, true);
    await demo.lookup();
    onlyLookup(demo, '/explain/wanted-id');
  });

  test('a structured prediction error is associated with the required manual ID', async () => {
    const demo = await createDemo({ uuid: false, predictions: [{ status: 503, data: {
      detail: { transaction_id: 'server-id', status: 'FAILED', message: 'Queue unavailable' },
    } }] });
    await demo.submit('');
    assert.equal(demo.posts().length, 0);
    await demo.submit('server-id');
    assert.equal(demo.posts()[0].payload.transaction_id, 'server-id');
    assert.match(demo.node('input-error').textContent, /server-id/);
    assert.equal(demo.visible('explanation-button'), true);
    await demo.lookup();
    onlyLookup(demo, '/explain/server-id');
  });

  test('structured explanation status errors with missing or different IDs never display the unrelated detail', async () => {
    for (const detail of [
      { transaction_id: 'other-id', status: 'FAILED', message: 'Explanation failed' },
      { status: 'FAILED', message: 'Explanation failed' },
      { transaction_id: null, status: 'failed', message: 'Explanation failed' },
    ]) {
      const demo = await createDemo({ apiBase: 'https://fixture.invalid/api/', predictions: [ambiguous],
        explanations: [{ status: 503, data: { detail } }] });
      await demo.submit('wanted-id');
      await demo.lookup();
      onlyLookup(demo, 'https://fixture.invalid/api/explain/wanted-id');
      assert.equal(demo.node('explanation-response').hidden, true);
      assert.match(demo.node('explanation-message').textContent, /missing or different transaction ID/i);
      assert.doesNotMatch(demo.node('explanation-message').textContent, /other-id|FAILED/);
      assert.equal(demo.visible('explanation-button'), true);
    }
  });

  test('matching structured explanation status and generic 503 or 404 errors remain informative', async () => {
    for (const fixture of [
      { status: 503, data: { detail: { transaction_id: 'wanted-id', status: 'FAILED', message: 'Explanation failed' } }, expected: /FAILED/ },
      { status: 503, data: { detail: 'Explanation storage is unavailable' }, expected: /Explanation storage is unavailable/ },
      { status: 503, data: { detail: { message: 'Explanation storage is unavailable' } }, expected: /Explanation storage is unavailable/ },
      { status: 503, data: { detail: { status: 'UNAVAILABLE', message: 'Explanation storage is unavailable' } }, expected: /Explanation storage is unavailable/ },
      { status: 404, data: { detail: 'SHAP explanation not found' }, expected: /No saved explanation/ },
      { status: 404, data: { detail: { message: 'SHAP explanation not found' } }, expected: /No saved explanation/ },
      { status: 404, data: { detail: { transaction_id: 'wanted-id', status: 'FAILED' } }, expected: /No saved explanation/ },
    ]) {
      const demo = await createDemo({ predictions: [ambiguous], explanations: [fixture] });
      await demo.submit('wanted-id');
      await demo.lookup();
      assert.match(demo.node('explanation-message').textContent, fixture.expected);
      assert.doesNotMatch(demo.node('explanation-message').textContent, /other-id/);
      assert.equal(demo.node('explanation-response').hidden, true);
    }
  });

  test('structured 404 identity mismatch or absence is not reported as a missing saved explanation', async () => {
    for (const detail of [
      { transaction_id: 'other-id', status: 'FAILED' },
      { status: 'FAILED', message: 'No saved explanation' },
    ]) {
      const demo = await createDemo({ apiBase: 'https://fixture.invalid/api/', predictions: [ambiguous],
        explanations: [{ status: 404, data: { detail } }] });
      await demo.submit('wanted-id');
      await demo.lookup();
      onlyLookup(demo, 'https://fixture.invalid/api/explain/wanted-id');
      assert.match(demo.node('explanation-message').textContent, /missing or different transaction ID/i);
      assert.doesNotMatch(demo.node('explanation-message').textContent, /other-id|No saved explanation/i);
      assert.equal(demo.node('explanation-response').hidden, true);
      assert.equal(demo.visible('explanation-button'), true);
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
    { status: 503, data: { detail: { transaction_id: 'recover-id', status: 'FAILED', message: 'Explanation calculation failed' } }, message: /FAILED/ },
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
