import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import { test } from 'node:test';
import { createDemo } from './demo-harness.mjs';

if (typeof vm.SourceTextModule !== 'function') {
  test('direction isolation checks in an isolated module runner', (context) => {
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
  const detail = 'Synthetic\u202e';
  const failure = (status = 503) => ({ status, data: { detail } });
  function assertIsolated(node, fixedPattern) {
    assert.equal(node.children.length, 2);
    const [fixed, external] = node.children;
    assert.equal(external.tagName, 'BDI');
    assert.equal(external.attributes.get('dir'), 'auto');
    assert.ok(external.textContent.includes(detail));
    assert.equal(fixed.tagName, 'BDI');
    assert.equal(fixed.attributes.get('dir'), 'ltr');
    assert.match(fixed.textContent, fixedPattern);
    assert.equal(node.textContent, fixed.textContent + external.textContent);
  }

  test('untrusted errors cannot change the direction of fixed recovery context', async () => {
    const demo = await createDemo({ apiBase: 'https://first.invalid/api', predictions: [failure()], explanations: [failure(), failure(), failure()] });
    await demo.submit('score-id');
    assertIsolated(demo.node('input-error'), /The request may have been processed/);
    await demo.lookup();
    assertIsolated(demo.node('explanation-message'), /Transaction ID: score-id\. Original API: https:\/\/first\.invalid\/api/);
    demo.input('manual-transaction-id', 'manual-id');
    await demo.manualLookup();
    assertIsolated(demo.node('manual-explanation-message'), /Transaction ID: manual-id\. Selected API: https:\/\/first\.invalid\/api/);
    const row = demo.node('unconfirmed-receipt-list').children[0];
    await row.children.find((child) => child.textContent === 'Check original API').listeners.get('click')({ preventDefault() {} });
    const status = row.children.find((child) => child.attributes.get('role') === 'status');
    assertIsolated(status, /Transaction ID: score-id\. Original API: https:\/\/first\.invalid\/api/);
  });

  test('untrusted conflict details cannot change fixed retry guidance', async () => {
    const demo = await createDemo({ predictions: [failure(409)] });
    await demo.submit('score-id');
    assertIsolated(demo.node('input-error'), /The request may have been processed.*Check the saved explanation.*new ID only for a different transaction/);
  });

  test('fixed recovery guidance precedes an intact long server detail on every surface', async () => {
    const longDetail = `${'synthetic'.repeat(1500)}\u202e`;
    const longFailure = { status: 503, data: { detail: longDetail } };
    const demo = await createDemo({ apiBase: 'https://first.invalid/api', predictions: [longFailure], explanations: [longFailure, longFailure, longFailure] });
    function assertPriority(node, prefix) {
      assert.ok(node.textContent.startsWith(prefix));
      assert.ok(node.textContent.includes(longDetail));
      assert.match(node.textContent, /\. HTTP 503:/);
    }
    await demo.submit('score-id');
    assertPriority(demo.node('input-error'), 'The request may have been processed.');
    await demo.lookup();
    assertPriority(demo.node('explanation-message'), 'Transaction ID: score-id. Original API: https://first.invalid/api.');
    demo.input('manual-transaction-id', 'manual-id');
    await demo.manualLookup();
    assertPriority(demo.node('manual-explanation-message'), 'Transaction ID: manual-id. Selected API: https://first.invalid/api.');
    const row = demo.node('unconfirmed-receipt-list').children[0];
    await row.children.find((child) => child.textContent === 'Check original API').listeners.get('click')({ preventDefault() {} });
    assertPriority(row.children.find((child) => child.attributes.get('role') === 'status'), 'Transaction ID: score-id. Original API: https://first.invalid/api.');
  });

  test('conflict guidance precedes its intact server detail', async () => {
    const demo = await createDemo({ predictions: [failure(409)] });
    await demo.submit('score-id');
    const message = demo.node('input-error').textContent;
    assert.ok(message.startsWith('The request may have been processed.'));
    assert.ok(message.includes(detail));
    assert.match(message, /different transaction\. HTTP 409:/);
  });

  test('conflict processing caution comes before retry guidance and intact server detail', async () => {
    const demo = await createDemo({ predictions: [failure(409)] });
    await demo.submit('score-id');
    const node = demo.node('input-error');
    assert.ok(node.textContent.startsWith('The request may have been processed.'));
    assert.ok(node.textContent.indexOf('Check the saved explanation') > node.textContent.indexOf('The request may have been processed.'));
    assert.equal(node.children[1].textContent, `HTTP 409: ${detail}`);
    assert.equal(demo.posts().length, 1);
  });
}
