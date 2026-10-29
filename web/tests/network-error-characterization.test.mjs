import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import { test } from 'node:test';
import { createDemo } from './demo-harness.mjs';

if (typeof vm.SourceTextModule !== 'function') {
  test('network error checks in an isolated module runner', (context) => {
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
  for (const field of ['features', 'transaction-id']) {
    test(`editing ${field} after a network failure clears the error and keeps the receipt`, async () => {
      const demo = await createDemo({ predictions: [{ networkFailure: true }] });
      await demo.submit('network-failed');
      assert.match(demo.node('input-error').textContent, /Cannot reach the API/i);
      assert.equal(demo.node('features').attributes.has('aria-invalid'), false);
      assert.equal(demo.node('transaction-id').attributes.has('aria-invalid'), false);
      assert.equal(demo.node('unconfirmed-receipt-list').children.length, 1);
      demo.input(field, field === 'features' ? '[1, 0, 100]' : 'different-id');
      assert.equal(demo.node('input-error').textContent, '');
      assert.equal(demo.node('unconfirmed-receipt-list').children.length, 1);
      assert.equal(demo.posts().length, 1);
    });
  }
}
