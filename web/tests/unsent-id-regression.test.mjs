import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import { test } from 'node:test';
import { createDemo } from './demo-harness.mjs';

if (typeof vm.SourceTextModule !== 'function') {
  test('transaction ID checks in an isolated module runner', (context) => {
    const childEnv = { PATH: process.env.PATH ?? '' };
    if (process.env.NODE_NO_WARNINGS !== undefined) childEnv.NODE_NO_WARNINGS = process.env.NODE_NO_WARNINGS;
    delete childEnv.NODE_TEST_CONTEXT;
    const child = spawnSync(process.execPath,
      ['--experimental-vm-modules', '--max-old-space-size=128', '--test', '--test-isolation=none',
        '--test-reporter=tap', fileURLToPath(import.meta.url)],
      { env: childEnv, encoding: 'utf8', timeout: 12000, maxBuffer: 1024 * 1024 });
    assert.equal(child.error, undefined, child.error?.message);
    assert.equal(child.status, 0, `${child.stdout}\n${child.stderr}`);
    assert.match(child.stdout, /# tests [1-9]\d*/, 'The VM child must report executed tests.');
    context.diagnostic(child.stdout.trim());
  });
} else {
  test('filling an example preserves an ID entered before any submission', async () => {
    const demo = await createDemo();
    demo.input('transaction-id', 'unsent-123');
    assert.equal(demo.posts().length, 0);

    demo.node('sample-button').listeners.get('click')();

    assert.equal(demo.node('transaction-id').value, 'unsent-123');
    assert.equal(demo.posts().length, 0);
    assert.equal(demo.lookups().length, 0);
  });

  test('filling an example clears a successfully submitted ID without making a receipt', async () => {
    const demo = await createDemo();
    await demo.submit('confirmed-123');
    assert.equal(demo.node('result').hidden, false);
    assert.equal(demo.posts().length, 1);

    demo.node('sample-button').listeners.get('click')();

    assert.equal(demo.node('transaction-id').value, '');
    assert.equal(demo.node('unconfirmed-receipt-list').children.length, 0);
    assert.equal(demo.posts().length, 1);
    assert.equal(demo.lookups().length, 0);
  });

  test('filling an example clears an unconfirmed ID but keeps its original API receipt', async () => {
    const demo = await createDemo({
      apiBase: 'https://first.invalid/api',
      predictions: [{ status: 503, data: { detail: 'Synthetic queue unavailable' } }],
    });
    await demo.submit('unconfirmed-123');
    assert.equal(demo.posts().length, 1);

    demo.node('sample-button').listeners.get('click')();

    assert.equal(demo.node('transaction-id').value, '');
    const rows = demo.node('unconfirmed-receipt-list').children;
    assert.equal(rows.length, 1);
    assert.ok(rows[0].children.some((child) =>
      child.textContent.includes('unconfirmed-123') &&
      child.textContent.includes('Original API: https://first.invalid/api')));
    assert.equal(demo.posts().length, 1);
    assert.equal(demo.lookups().length, 0);
  });

  test('an unsent ID keeps its original surrounding spaces', async () => {
    const demo = await createDemo();
    demo.input('transaction-id', '  unsent-spaced  ');
    demo.node('sample-button').listeners.get('click')();
    assert.equal(demo.node('transaction-id').value, '  unsent-spaced  ');
    assert.equal(demo.posts().length, 0);
  });

  test('a successfully attempted ID is cleared after feature edits', async () => {
    const demo = await createDemo();
    await demo.submit('confirmed-after-edit');
    demo.input('features', '[1, 2, 3]');
    demo.node('sample-button').listeners.get('click')();
    assert.equal(demo.node('transaction-id').value, '');
    assert.equal(demo.posts().length, 1);
  });

  test('a generated ID is cleared after feature edits', async () => {
    const demo = await createDemo({ cryptoSource: { randomUUID: () => 'generated-123' } });
    await demo.submit('');
    assert.equal(demo.posts()[0].payload.transaction_id, 'generated-123');
    demo.input('features', '[1, 2, 3]');
    demo.node('sample-button').listeners.get('click')();
    assert.equal(demo.node('transaction-id').value, '');
    assert.equal(demo.posts().length, 1);
  });

  test('earlier attempted IDs are recognized after another attempt and spaced retyping', async () => {
    const demo = await createDemo();
    await demo.submit('first-attempt');
    await demo.submit('second-attempt');
    demo.input('transaction-id', '  first-attempt  ');
    demo.node('sample-button').listeners.get('click')();
    assert.equal(demo.node('transaction-id').value, '');
    assert.equal(demo.posts().length, 2);
  });

  test('a validation failure does not mark an unsent ID as attempted', async () => {
    const demo = await createDemo();
    demo.input('transaction-id', '  invalid-feature-id  ');
    demo.input('features', 'not valid JSON');
    await demo.submitCurrent();
    assert.equal(demo.posts().length, 0);
    assert.match(demo.node('input-error').textContent, /valid JSON/i);
    demo.node('sample-button').listeners.get('click')();
    assert.equal(demo.node('transaction-id').value, '  invalid-feature-id  ');
  });

  test('replacing invalid features clears their message and invalid marker', async () => {
    const demo = await createDemo();
    demo.input('transaction-id', 'fresh-id');
    demo.input('features', 'not valid JSON');
    await demo.submitCurrent();
    assert.match(demo.node('input-error').textContent, /valid JSON/i);
    assert.equal(demo.node('features').attributes.get('aria-invalid'), 'true');
    demo.node('sample-button').listeners.get('click')();
    assert.equal(demo.node('input-error').textContent, '');
    assert.equal(demo.node('features').attributes.has('aria-invalid'), false);
    assert.equal(demo.node('transaction-id').value, 'fresh-id');
    assert.equal(demo.posts().length, 0);
  });

  test('an unchanged invalid ID keeps its validation message and marker', async () => {
    const demo = await createDemo();
    demo.input('transaction-id', '  invalid/id  ');
    demo.input('features', '[0, 0, 100]');
    await demo.submitCurrent();
    const message = demo.node('input-error').textContent;
    assert.match(message, /ID/);
    assert.equal(demo.node('transaction-id').attributes.get('aria-invalid'), 'true');
    demo.node('sample-button').listeners.get('click')();
    assert.equal(demo.node('input-error').textContent, message);
    assert.equal(demo.node('transaction-id').attributes.get('aria-invalid'), 'true');
    assert.equal(demo.node('transaction-id').value, '  invalid/id  ');
    assert.equal(demo.posts().length, 0);
  });

  test('editing features keeps the message for an unchanged invalid ID', async () => {
    const demo = await createDemo();
    demo.input('transaction-id', 'invalid/id');
    demo.input('features', '[0, 0, 100]');
    await demo.submitCurrent();
    const message = demo.node('input-error').textContent;
    assert.match(message, /ID/);
    demo.input('features', '[1, 0, 100]');
    assert.equal(demo.node('input-error').textContent, message);
    assert.equal(demo.node('transaction-id').attributes.get('aria-invalid'), 'true');
    demo.node('sample-button').listeners.get('click')();
    assert.equal(demo.node('input-error').textContent, message);
    assert.equal(demo.node('transaction-id').value, 'invalid/id');
  });

  test('editing an ID keeps the message for unchanged invalid features', async () => {
    const demo = await createDemo();
    demo.input('transaction-id', 'initial-id');
    demo.input('features', 'invalid JSON');
    await demo.submitCurrent();
    const message = demo.node('input-error').textContent;
    assert.match(message, /valid JSON/i);
    assert.equal(demo.node('features').attributes.get('aria-invalid'), 'true');
    demo.input('transaction-id', 'different-id');
    assert.equal(demo.node('input-error').textContent, message);
    assert.equal(demo.node('features').attributes.get('aria-invalid'), 'true');
    assert.equal(demo.node('transaction-id').attributes.has('aria-invalid'), false);
    assert.equal(demo.posts().length, 0);
  });

  test('clearing an attempted ID also clears its stale invalid marker', async () => {
    const demo = await createDemo({ predictions: [{ status: 503, data: { detail: 'Synthetic unavailable' } }] });
    await demo.submit('unconfirmed-repeat');
    await demo.submit('unconfirmed-repeat');
    assert.equal(demo.posts().length, 1);
    assert.equal(demo.node('transaction-id').attributes.get('aria-invalid'), 'true');
    assert.match(demo.node('input-error').textContent, /receipt/i);
    demo.node('sample-button').listeners.get('click')();
    assert.equal(demo.node('transaction-id').value, '');
    assert.equal(demo.node('transaction-id').attributes.has('aria-invalid'), false);
    assert.equal(demo.node('input-error').textContent, '');
    assert.equal(demo.node('unconfirmed-receipt-list').children.length, 1);
  });

  test('the receipt limit does not mark the rejected ID as attempted', async () => {
    const failed = () => ({ status: 503, data: { detail: 'Synthetic unavailable' } });
    const demo = await createDemo({ predictions: Array.from({ length: 10 }, failed) });
    for (let index = 0; index < 10; index++) await demo.submit(`failed-${index}`);
    assert.equal(demo.posts().length, 10);
    await demo.submit('  blocked-by-limit  ');
    assert.match(demo.node('input-error').textContent, /receipt.*limit|limit.*receipt/i);
    assert.equal(demo.posts().length, 10);
    demo.node('sample-button').listeners.get('click')();
    assert.equal(demo.node('transaction-id').value, '  blocked-by-limit  ');
    assert.match(demo.node('input-error').textContent, /receipt.*limit|limit.*receipt/i);
    assert.equal(demo.node('transaction-id').attributes.get('aria-invalid'), 'true');
  });

  test('an attempted ID is still recognized after switching APIs and reconnecting', async () => {
    const demo = await createDemo({ apiBase: 'https://first.invalid/api' });
    await demo.submit('first-api-id');
    demo.input('api-base', 'https://second.invalid/api');
    await demo.reconnect();
    demo.input('transaction-id', '  first-api-id  ');
    demo.node('sample-button').listeners.get('click')();
    assert.equal(demo.node('transaction-id').value, '');
    assert.equal(demo.posts().length, 1);
  });

  test('a new unsent ID stays intact after switching APIs and reconnecting', async () => {
    const demo = await createDemo({ apiBase: 'https://first.invalid/api' });
    await demo.submit('first-api-id');
    demo.input('api-base', 'https://second.invalid/api');
    await demo.reconnect();
    demo.input('transaction-id', '  new-second-api-id  ');
    demo.node('sample-button').listeners.get('click')();
    assert.equal(demo.node('transaction-id').value, '  new-second-api-id  ');
    assert.equal(demo.posts().length, 1);
  });

  test('a hidden late success keeps a new unsent ID and the original receipt', async () => {
    let release;
    const deferred = new Promise((resolve) => { release = resolve; });
    const demo = await createDemo({
      apiBase: 'https://first.invalid/api',
      predictions: [{ deferred }],
    });
    const pending = demo.submit('attempted-A');
    await demo.flush();
    assert.equal(demo.posts().length, 1);
    assert.equal(demo.node('transaction-id').disabled, true);
    // Simulate a programmatic change while the disabled request is pending.
    demo.input('transaction-id', '  unsent-B  ');
    release({});
    await pending;

    const rows = demo.node('unconfirmed-receipt-list').children;
    assert.equal(rows.length, 1);
    assert.ok(rows[0].children.some((child) =>
      child.textContent.includes('attempted-A') &&
      child.textContent.includes('Original API: https://first.invalid/api')));
    assert.ok(rows[0].children.some((child) => /not displayed/i.test(child.textContent)));
    assert.equal(demo.node('result').hidden, true);

    demo.node('sample-button').listeners.get('click')();
    assert.equal(demo.node('transaction-id').value, '  unsent-B  ');
    assert.equal(demo.node('unconfirmed-receipt-list').children.length, 1);
    assert.equal(demo.posts().length, 1);

    demo.input('transaction-id', '  attempted-A  ');
    demo.node('sample-button').listeners.get('click')();
    assert.equal(demo.node('transaction-id').value, '');
    assert.equal(demo.node('unconfirmed-receipt-list').children.length, 1);
    assert.equal(demo.posts().length, 1);
  });

  test('reconnecting clears stale ID feedback but resubmission revalidates the retained ID', async () => {
    const demo = await createDemo();
    demo.input('transaction-id', 'invalid/id');
    demo.input('features', '[0, 0, 100]');
    await demo.submitCurrent();
    const message = demo.node('input-error').textContent;
    assert.match(message, /ID/);
    assert.equal(demo.node('transaction-id').attributes.get('aria-invalid'), 'true');
    await demo.reconnect();
    assert.equal(demo.node('input-error').textContent, '');
    assert.equal(demo.node('transaction-id').attributes.has('aria-invalid'), false);
    assert.equal(demo.node('transaction-id').value, 'invalid/id');
    demo.node('sample-button').listeners.get('click')();
    assert.equal(demo.node('input-error').textContent, '');
    assert.equal(demo.node('transaction-id').attributes.has('aria-invalid'), false);
    assert.equal(demo.node('transaction-id').value, 'invalid/id');
    await demo.submitCurrent();
    assert.equal(demo.node('input-error').textContent, message);
    assert.equal(demo.node('transaction-id').attributes.get('aria-invalid'), 'true');
    assert.equal(demo.posts().length, 0);
  });

  test('reconnecting clears stale feature feedback while retaining the input until Fill', async () => {
    const demo = await createDemo();
    demo.input('transaction-id', 'unsent-id');
    demo.input('features', 'invalid JSON');
    await demo.submitCurrent();
    const message = demo.node('input-error').textContent;
    assert.match(message, /valid JSON/i);
    assert.equal(demo.node('features').attributes.get('aria-invalid'), 'true');
    await demo.reconnect();
    assert.equal(demo.node('input-error').textContent, '');
    assert.equal(demo.node('features').attributes.has('aria-invalid'), false);
    assert.equal(demo.node('features').value, 'invalid JSON');
    demo.node('sample-button').listeners.get('click')();
    assert.equal(demo.node('input-error').textContent, '');
    assert.equal(demo.node('features').attributes.has('aria-invalid'), false);
    assert.equal(demo.node('transaction-id').value, 'unsent-id');
    assert.equal(demo.posts().length, 0);
  });

  test('switching APIs clears the prior receipt conflict without dropping its receipt', async () => {
    const demo = await createDemo({
      apiBase: 'https://first.invalid/api',
      predictions: [{ status: 503, data: { detail: 'Synthetic unavailable' } }],
    });
    await demo.submit('same-id');
    await demo.submit('same-id');
    assert.equal(demo.posts().length, 1);
    assert.match(demo.node('input-error').textContent, /receipt/i);
    assert.equal(demo.node('transaction-id').attributes.get('aria-invalid'), 'true');
    demo.input('api-base', 'https://second.invalid/api');
    await demo.reconnect();
    assert.equal(demo.node('input-error').textContent, '');
    assert.equal(demo.node('transaction-id').attributes.has('aria-invalid'), false);
    assert.equal(demo.node('transaction-id').value, 'same-id');
    assert.equal(demo.node('unconfirmed-receipt-list').children.length, 1);
    assert.ok(demo.node('unconfirmed-receipt-list').children[0].children.some((child) =>
      child.textContent.includes('Original API: https://first.invalid/api')));
    await demo.submitCurrent();
    assert.equal(demo.posts().length, 2);
    assert.equal(demo.posts()[1].url, 'https://second.invalid/api/predict');
    assert.equal(demo.node('unconfirmed-receipt-list').children.length, 1);
  });

  test('reconnecting after a network failure still clears the error and keeps its receipt', async () => {
    const demo = await createDemo({ predictions: [{ networkFailure: true }] });
    await demo.submit('network-failed');
    assert.match(demo.node('input-error').textContent, /Cannot reach the API/i);
    assert.equal(demo.node('unconfirmed-receipt-list').children.length, 1);
    await demo.reconnect();
    assert.equal(demo.node('input-error').textContent, '');
    assert.equal(demo.node('unconfirmed-receipt-list').children.length, 1);
    assert.equal(demo.posts().length, 1);
  });
}
