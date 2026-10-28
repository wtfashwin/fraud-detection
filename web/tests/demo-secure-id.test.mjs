import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import { test } from 'node:test';
import { createDemo } from './demo-harness.mjs';

// Exercise the actual demo module through the same bounded VM runner as recovery tests.
if (typeof vm.SourceTextModule !== 'function') {
  test('secure ID checks in an isolated module runner', (context) => {
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
  const expectedBytesId = 'tx-000102030405060708090a0b0c0d0e0f';
  function fixedBytes(bytes) {
    for (let i = 0; i < bytes.length; i++) bytes[i] = i;
    return bytes;
  }
  const failedConfirmation = { status: 503, data: { detail: 'Queue unavailable' } };

  test('a browser UUID is submitted and retained for explicit recovery', async () => {
    const demo = await createDemo({ cryptoSource: { randomUUID: () => 'primary-id' }, predictions: [failedConfirmation] });
    await demo.submit('');
    assert.equal(demo.posts().length, 1);
    assert.equal(demo.posts()[0].payload.transaction_id, 'primary-id');
    assert.equal(demo.node('transaction-id').value, 'primary-id');
    assert.equal(demo.node('result').hidden, true);
    await demo.lookup();
    assert.equal(demo.lookups()[0].url, '/explain/primary-id');
  });

  test('random bytes supply a known ID when UUID is unavailable', async () => {
    const demo = await createDemo({ apiBase: 'https://fixture.invalid/api/', cryptoSource: { getRandomValues: fixedBytes }, predictions: [failedConfirmation] });
    await demo.submit('');
    assert.equal(demo.posts().length, 1);
    assert.equal(demo.posts()[0].payload.transaction_id, expectedBytesId);
    assert.equal(demo.node('transaction-id').value, expectedBytesId);
    await demo.lookup();
    assert.equal(demo.lookups()[0].url, `https://fixture.invalid/api/explain/${expectedBytesId}`);
  });

  test('random bytes are used when UUID generation throws', async () => {
    const demo = await createDemo({ cryptoSource: { randomUUID() { throw new Error('unavailable'); }, getRandomValues: fixedBytes } });
    await demo.submit('');
    assert.equal(demo.posts().length, 1);
    assert.equal(demo.posts()[0].payload.transaction_id, expectedBytesId);
    assert.equal(demo.node('result').hidden, false);
  });

  test('an invalid generated UUID falls back to random bytes', async () => {
    const demo = await createDemo({ cryptoSource: { randomUUID: () => 'invalid/id', getRandomValues: fixedBytes } });
    await demo.submit('');
    assert.equal(demo.posts().length, 1);
    assert.equal(demo.posts()[0].payload.transaction_id, expectedBytesId);
    assert.equal(demo.node('result').hidden, false);
  });

  test('missing generators require a manual ID before any POST', async () => {
    const demo = await createDemo({ cryptoSource: null });
    await demo.submit('');
    assert.equal(demo.posts().length, 0);
    assert.match(demo.node('input-error').textContent, /transaction ID/i);
    assert.equal(demo.node('transaction-id').attributes.get('aria-invalid'), 'true');
    assert.equal(demo.visible('explanation-button'), false);
  });

  test('throwing generators require a manual ID before any POST', async () => {
    const demo = await createDemo({ cryptoSource: { randomUUID() { throw new Error('uuid unavailable'); }, getRandomValues() { throw new Error('bytes unavailable'); } } });
    await demo.submit('');
    assert.equal(demo.posts().length, 0);
    assert.match(demo.node('input-error').textContent, /transaction ID/i);
  });

  test('a valid manual ID works when generators are unavailable', async () => {
    const demo = await createDemo({ cryptoSource: null, predictions: [failedConfirmation] });
    await demo.submit('manual-id');
    assert.equal(demo.posts().length, 1);
    assert.equal(demo.posts()[0].payload.transaction_id, 'manual-id');
    await demo.lookup();
    assert.equal(demo.lookups()[0].url, '/explain/manual-id');
  });

  test('a manual ID does not call available generators', async () => {
    const demo = await createDemo({ cryptoSource: { randomUUID() { throw new Error('must not generate'); }, getRandomValues() { throw new Error('must not generate'); } } });
    await demo.submit('provided-id');
    assert.equal(demo.posts()[0].payload.transaction_id, 'provided-id');
    assert.equal(demo.node('result').hidden, false);
  });

  test('a conflict after an unconfirmed submission prompts a saved status check before retrying', async () => {
    const duplicate = { status: 409, data: { detail: 'Transaction ID already exists; use a new ID' } };
    const demo = await createDemo({ predictions: [failedConfirmation, duplicate] });
    await demo.submit('uncertain-id');
    await demo.submit('uncertain-id');
    const message = demo.node('input-error').textContent;
    assert.match(message, /409/);
    assert.match(message, /check.*saved/i);
    assert.match(message, /new ID.*different transaction/i);
    assert.doesNotMatch(message, /enter a new transaction ID before scoring again/i);
    assert.doesNotMatch(message, /already exists; use a new ID/i);
    assert.doesNotMatch(message, /was saved|has been saved/i);
    await demo.lookup();
    assert.equal(demo.lookups()[0].url, '/explain/uncertain-id');
  });

  test('a conflict naming a different transaction suppresses unrelated details and advice', async () => {
    const mismatch = { status: 409, data: { detail: { transaction_id: 'other-id', status: 'FAILED', message: 'unrelated details' } } };
    const demo = await createDemo({ predictions: [mismatch] });
    await demo.submit('expected-id');
    const message = demo.node('input-error').textContent;
    assert.match(message, /missing or different transaction ID/i);
    assert.doesNotMatch(message, /other-id|unrelated details|new ID/i);
    await demo.lookup();
    assert.equal(demo.lookups()[0].url, '/explain/expected-id');
  });

  test('a generic conflict preserves actionable API detail and cautious guidance', async () => {
    const conflict = { status: 409, data: { detail: 'Account locked. Contact administrator.' } };
    const demo = await createDemo({ predictions: [conflict] });
    await demo.submit('expected-id');
    const message = demo.node('input-error').textContent;
    assert.match(message, /Account locked\. Contact administrator\./);
    assert.match(message, /check.*saved/i);
    assert.match(message, /new ID.*different transaction/i);
    assert.doesNotMatch(message, /was saved|has been saved/i);
  });

  test('a matching structured conflict preserves its actionable detail', async () => {
    const conflict = { status: 409, data: { detail: { transaction_id: 'expected-id', status: 'FAILED', message: 'Account locked. Contact administrator.' } } };
    const demo = await createDemo({ predictions: [conflict] });
    await demo.submit('expected-id');
    const message = demo.node('input-error').textContent;
    assert.match(message, /Account locked\. Contact administrator\./);
    assert.match(message, /check.*saved/i);
  });

  test('an ordinary non-conflict service error retains its detail', async () => {
    const demo = await createDemo({ predictions: [failedConfirmation] });
    await demo.submit('known-id');
    assert.match(demo.node('input-error').textContent, /Queue unavailable/);
    assert.doesNotMatch(demo.node('input-error').textContent, /409|different transaction/i);
  });
}
