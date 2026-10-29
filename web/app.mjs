import { normalizeApiBase, validateFeatures, validateModelInfo, validatePrediction, formatProbability, fetchJson, RequestTimeoutError } from './api-client.mjs';

const element = (id) => document.getElementById(id);
let modelInfo = null;
let apiBase = '';
let apiBaseValid = false;
let prediction = null;
let requestContext = null;
let busy = false;
let manualRevision = 0;
let scoringRevision = 0;
const attemptedTransactionIds = new Set();
const recoveryReceipts = [];
const maxRecoveryReceipts = 10;
const validTransactionId = (value) => typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/.test(value);
const transactionErrorStates = new Set(['PENDING', 'PROCESSING', 'QUEUED', 'COMPLETED', 'FAILED', 'DISPATCH_UNCONFIRMED']);

function recoveryApiKey(base) {
  const pageOrigin = globalThis.location?.origin;
  return base || (typeof pageOrigin === 'string' && /^https?:\/\//.test(pageOrigin) ? pageOrigin : '');
}

function displayApi(base) {
  if (base) return base;
  try {
    const pageOrigin = globalThis.location?.origin;
    const parsed = new URL(pageOrigin);
    if ((parsed.protocol === 'http:' || parsed.protocol === 'https:') && parsed.origin === pageOrigin) return pageOrigin;
  } catch { /* A missing or opaque page origin cannot identify an API. */ }
  return 'this server';
}

function generateTransactionId() {
  const source = globalThis.crypto;
  try {
    if (typeof source?.randomUUID === 'function') {
      const id = source.randomUUID();
      if (validTransactionId(id)) return id;
    }
  } catch { /* Try the other secure browser generator. */ }
  try {
    if (typeof source?.getRandomValues === 'function') {
      const bytes = new Uint8Array(16);
      source.getRandomValues(bytes);
      return `tx-${Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('')}`;
    }
  } catch { /* Require an entered ID if neither method works. */ }
  return null;
}

function hasUnverifiedTransactionError(error, expectedId) {
  const detail = error?.detail;
  if (!expectedId || !detail || typeof detail !== 'object' || Array.isArray(detail)) return false;
  const hasId = Object.prototype.hasOwnProperty.call(detail, 'transaction_id');
  const hasTransactionStatus = typeof detail.status === 'string' && transactionErrorStates.has(detail.status.toUpperCase());
  return (hasId || hasTransactionStatus) && detail.transaction_id !== expectedId;
}

function validateExplanation(explanation, expectedId) {
  if (explanation?.transaction_id !== expectedId) {
    throw new Error('The API returned an explanation with a missing or different transaction ID.');
  }
  if (!['PENDING', 'COMPLETED', 'FAILED'].includes(explanation.status)) {
    throw new Error('The API returned an unrecognized explanation status. No result is displayed.');
  }
}

function explanationError(error, expectedId) {
  if (hasUnverifiedTransactionError(error, expectedId)) {
    return 'The API returned an error with a missing or different transaction ID.';
  }
  if (error.status === 404 && error.detail === 'SHAP explanation not found') {
    return 'No saved explanation was found. This does not establish whether the request was processed.';
  }
  return error.message;
}

function setMessage(id, message) {
  const node = element(id);
  node.textContent = message;
  node.hidden = !message;
}

function setContextMessage(node, message, context) {
  const detail = document.createElement('bdi');
  detail.setAttribute('dir', 'auto');
  detail.textContent = message;
  const fixed = document.createElement('bdi');
  fixed.setAttribute('dir', 'ltr');
  const fixedText = context.trim();
  fixed.textContent = fixedText ? `${fixedText} ` : '';
  node.replaceChildren(fixed, detail);
  node.hidden = false;
}

function updateControls() {
  element('connect-button').disabled = busy;
  element('api-base').disabled = busy;
  element('score-button').disabled = busy || !modelInfo;
  element('sample-button').disabled = busy || !modelInfo;
  element('features').disabled = busy;
  element('transaction-id').disabled = busy;
  element('manual-transaction-id').disabled = busy;
  element('manual-explanation-button').disabled = busy || !apiBaseValid || !validTransactionId(element('manual-transaction-id').value.trim());
  element('explanation-button').disabled = busy || !requestContext;
  element('explanation-panel').hidden = !requestContext;
  element('result-panel').setAttribute('aria-busy', String(busy));
  element('score-button').textContent = busy ? 'Please wait…' : 'Score transaction';
  for (const receipt of recoveryReceipts) {
    receipt.check.disabled = busy;
    receipt.remove.disabled = busy;
  }
}

function renderRecoveryReceipts() {
  element('unconfirmed-receipt-list').replaceChildren(...recoveryReceipts.map((receipt) => receipt.item));
  element('unconfirmed-receipts').hidden = recoveryReceipts.length === 0;
  updateControls();
}

function rememberRecoveryReceipt(context, outcome) {
  if (!context || recoveryReceipts.some((receipt) => receipt.transactionId === context.transactionId && recoveryApiKey(receipt.apiBase) === recoveryApiKey(context.apiBase))) return;
  const item = document.createElement('li');
  const description = document.createElement('p');
  const reason = document.createElement('p');
  const check = document.createElement('button');
  const remove = document.createElement('button');
  const status = document.createElement('p');
  const response = document.createElement('pre');
  const originalApi = displayApi(context.apiBase);
  description.textContent = `ID: ${context.transactionId}. Original API: ${originalApi}.`;
  reason.textContent = outcome === 'hidden'
    ? 'A valid response was received, but the result was not displayed because the input changed while it was pending.'
    : 'The prediction response was not confirmed. This does not establish whether the request was saved.';
  check.type = 'button';
  check.setAttribute('class', 'secondary');
  check.textContent = 'Check original API';
  check.setAttribute('aria-label', `Check original API for ID ${context.transactionId}. Original API: ${originalApi}.`);
  remove.type = 'button';
  remove.setAttribute('class', 'secondary');
  remove.textContent = 'Remove from page';
  remove.setAttribute('aria-label', `Remove from page for ID ${context.transactionId}. Original API: ${originalApi}.`);
  status.setAttribute('role', 'status');
  status.hidden = true;
  response.hidden = true;
  item.append(description, reason, check, remove, status, response);
  const receipt = { ...context, item, check, remove, status, response };
  const showReceiptMessage = (message) => setContextMessage(status, message, ` Transaction ID: ${context.transactionId}. Original API: ${originalApi}.`);

  check.addEventListener('click', async () => {
    if (busy || !recoveryReceipts.includes(receipt)) return;
    busy = true;
    updateControls();
    showReceiptMessage('Checking saved status.');
    response.textContent = '';
    response.hidden = true;
    try {
      const explanation = await fetchJson(`${receipt.apiBase}/explain/${encodeURIComponent(receipt.transactionId)}`);
      if (!recoveryReceipts.includes(receipt)) return;
      validateExplanation(explanation, receipt.transactionId);
      if (explanation.status === 'PENDING') {
        showReceiptMessage('The API reports this explanation is pending.');
      } else {
        response.textContent = JSON.stringify(explanation, null, 2);
        response.hidden = false;
        showReceiptMessage(explanation.status === 'FAILED'
          ? 'The API reports explanation processing failed.'
          : 'Saved response received.');
      }
    } catch (error) {
      if (!recoveryReceipts.includes(receipt)) return;
      showReceiptMessage(explanationError(error, receipt.transactionId));
    } finally {
      busy = false;
      updateControls();
    }
  });

  remove.addEventListener('click', () => {
    const index = recoveryReceipts.indexOf(receipt);
    if (index < 0) return;
    recoveryReceipts.splice(index, 1);
    renderRecoveryReceipts();
  });
  recoveryReceipts.push(receipt);
  renderRecoveryReceipts();
}

function clearResult() {
  scoringRevision += 1;
  prediction = null;
  requestContext = null;
  element('result').hidden = true;
  element('empty-result').hidden = false;
  for (const id of ['prediction-label', 'probability', 'result-id', 'correlation-id', 'explanation-status', 'raw-response']) {
    element(id).textContent = '';
  }
  element('explanation-response').hidden = true;
  setMessage('explanation-message', '');
  setMessage('request-status', '');
  updateControls();
}

function clearManualResult() {
  manualRevision += 1;
  element('manual-explanation-response').hidden = true;
  element('manual-explanation-response').textContent = '';
  setMessage('manual-explanation-message', '');
}

async function connect(event) {
  event?.preventDefault();
  if (busy) return;
  modelInfo = null;
  apiBaseValid = false;
  clearResult();
  clearManualResult();
  busy = true;
  updateControls();
  setMessage('connection-error', '');
  setMessage('input-error', '');
  element('features').removeAttribute('aria-invalid');
  element('transaction-id').removeAttribute('aria-invalid');
  element('connection-status').textContent = 'Connecting…';
  element('connection-status').dataset.state = 'loading';
  element('feature-order').replaceChildren();
  element('feature-count').textContent = '';
  try {
    apiBase = normalizeApiBase(element('api-base').value);
    apiBaseValid = true;
    modelInfo = validateModelInfo(await fetchJson(`${apiBase}/model-info`));
    element('docs-link').href = `${apiBase}/docs`;
    element('feature-count').textContent = `(${modelInfo.feature_count} values)`;
    for (const name of modelInfo.feature_names) {
      const item = document.createElement('li');
      item.textContent = name;
      element('feature-order').append(item);
    }
    element('features').placeholder = `Enter ${modelInfo.feature_count} numbers in the order shown below.`;
    element('connection-status').textContent = `Model connected · ${modelInfo.model_source || 'source not supplied'} · ${modelInfo.feature_count} features`;
    element('connection-status').dataset.state = 'ready';
  } catch (error) {
    modelInfo = null;
    element('connection-status').textContent = 'Connection unavailable';
    element('connection-status').dataset.state = 'failed';
    setMessage('connection-error', error.message);
  } finally {
    busy = false;
    updateControls();
  }
}

element('connection-form').addEventListener('submit', connect);
element('api-base').addEventListener('input', () => {
  modelInfo = null;
  apiBaseValid = false;
  clearResult();
  clearManualResult();
  element('connection-status').textContent = 'URL changed. Reconnect to load this model.';
  element('connection-status').dataset.state = 'loading';
  element('feature-order').replaceChildren();
  element('feature-count').textContent = '';
});

element('manual-transaction-id').addEventListener('input', () => {
  clearManualResult();
  updateControls();
});

element('sample-button').addEventListener('click', () => {
  if (!modelInfo || busy) return;
  const values = modelInfo.feature_names.map((name) => name === 'Amount' ? 100 : 0);
  element('features').value = JSON.stringify(values);
  element('features').removeAttribute('aria-invalid');
  const transactionInput = element('transaction-id');
  if (attemptedTransactionIds.has(transactionInput.value.trim())) {
    transactionInput.value = '';
    transactionInput.removeAttribute('aria-invalid');
  }
  // The example is an arbitrary input vector; it carries no expected class or score.
  element('sample-note').textContent = modelInfo.feature_names.includes('Amount')
    ? 'Synthetic example: zeros for all features except Amount = 100. This is invented input for testing the request flow, with no expected fraud outcome.'
    : 'Synthetic example: zeros for all features. This is invented input for testing the request flow, with no expected fraud outcome.';
  element('sample-note').hidden = false;
  if (transactionInput.getAttribute('aria-invalid') !== 'true') setMessage('input-error', '');
  clearResult();
  element('features').focus();
});

for (const id of ['features', 'transaction-id']) {
  element(id).addEventListener('input', () => {
    clearResult();
    const other = id === 'features' ? 'transaction-id' : 'features';
    if (element(other).getAttribute('aria-invalid') !== 'true') setMessage('input-error', '');
    element(id).removeAttribute('aria-invalid');
    if (id === 'features') element('sample-note').hidden = true;
  });
}

element('transaction').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!modelInfo || busy) return;
  clearResult();
  setMessage('input-error', '');
  element('features').removeAttribute('aria-invalid');
  element('transaction-id').removeAttribute('aria-invalid');
  let payload;
  const submittedBase = apiBase;
  let invalidField = 'features';
  try {
    const features = validateFeatures(element('features').value, modelInfo.feature_count);
    invalidField = 'transaction-id';
    let transactionId = element('transaction-id').value.trim();
    if (!transactionId) {
      transactionId = generateTransactionId();
      if (!transactionId) throw new Error('Enter a transaction ID. Your browser cannot generate one automatically.');
      element('transaction-id').value = transactionId;
    }
    if (transactionId && !validTransactionId(transactionId)) {
      throw new Error('Use 1–128 letters, numbers, dots, underscores, colons, or hyphens for the ID. Start with a letter or number.');
    }
    if (recoveryReceipts.some((receipt) => receipt.transactionId === transactionId && recoveryApiKey(receipt.apiBase) === recoveryApiKey(submittedBase))) {
      throw new Error('A recovery receipt exists for this ID and API. Check its original API, then save details and remove the receipt before reusing this ID. This request was not sent.');
    }
    if (recoveryReceipts.length >= maxRecoveryReceipts) {
      throw new Error('Session receipt limit reached. Save details and remove a receipt before scoring a different transaction. This request was not sent.');
    }
    payload = { features, ...(transactionId ? { transaction_id: transactionId } : {}) };
    requestContext = transactionId ? { transactionId, apiBase: submittedBase } : null;
  } catch (error) {
    setMessage('input-error', error.message);
    element(invalidField).setAttribute('aria-invalid', 'true');
    element(invalidField).focus();
    return;
  }
  const submittedContext = requestContext;
  const submittedRevision = scoringRevision;
  busy = true;
  updateControls();
  setMessage('request-status', 'Scoring with the connected API…');
  try {
    attemptedTransactionIds.add(payload.transaction_id);
    const response = validatePrediction(await fetchJson(`${submittedBase}/predict`, { method: 'POST', body: JSON.stringify(payload) }));
    if (payload.transaction_id && response.transaction_id !== payload.transaction_id) {
      throw new Error('The API returned a different transaction ID.');
    }
    if (scoringRevision !== submittedRevision) {
      rememberRecoveryReceipt(submittedContext, 'hidden');
      return;
    }
    prediction = response;
    requestContext = validTransactionId(prediction.transaction_id)
      ? { transactionId: prediction.transaction_id, apiBase: submittedBase }
      : null;
    element('prediction-label').textContent = prediction.prediction === 1 ? 'Flagged as fraud · class 1' : 'Not flagged as fraud · class 0';
    element('prediction-label').dataset.fraud = String(prediction.prediction === 1);
    element('probability').textContent = formatProbability(prediction.score);
    element('result-id').textContent = prediction.transaction_id;
    element('transaction-id').value = prediction.transaction_id;
    element('correlation-id').textContent = prediction.correlation_id;
    element('explanation-status').textContent = prediction.explanation_status;
    element('raw-response').textContent = JSON.stringify(prediction, null, 2);
    element('empty-result').hidden = true;
    element('result').hidden = false;
    setMessage('request-status', 'Prediction received. The result below is the API response.');
    element('result-title').focus();
  } catch (error) {
    rememberRecoveryReceipt(submittedContext, 'unconfirmed');
    prediction = null;
    setMessage('request-status', '');
    const unverified = hasUnverifiedTransactionError(error, submittedContext?.transactionId);
    if (scoringRevision !== submittedRevision) {
      setMessage('input-error', unverified
        ? 'The API returned an error with a missing or different transaction ID for the previous request. It may have been processed. See its original API receipt.'
        : 'The previous request may have been processed, but its response was not confirmed after inputs changed. Check its original API receipt.');
      return;
    }
    const conflict = error.detail === 'Transaction ID already exists; use a new ID'
      ? 'HTTP 409: The API reported a conflict.' : error.message;
    const isConflict = !unverified && error.status === 409 && Boolean(requestContext);
    const message = unverified ? 'The API returned an error with a missing or different transaction ID.' : isConflict ? conflict : error.message;
    const advice = isConflict ? ' Check the saved explanation before deciding whether to retry. Use a new ID only for a different transaction.' : '';
    const warning = error instanceof RequestTimeoutError ? '' : ' The request may have been processed.';
    setContextMessage(element('input-error'), message, `${warning}${advice}`);
    setMessage('explanation-message', requestContext
      ? `No valid prediction was received. Check saved status for ${requestContext.transactionId}; this does not confirm that it was saved.`
      : '');
  } finally {
    busy = false;
    updateControls();
  }
});

element('explanation-button').addEventListener('click', async () => {
  if (!requestContext || busy) return;
  const lookup = requestContext;
  const originalApi = displayApi(lookup.apiBase);
  const showLookupMessage = (message) => setContextMessage(element('explanation-message'), message, ` Transaction ID: ${lookup.transactionId}. Original API: ${originalApi}.`);
  busy = true;
  updateControls();
  showLookupMessage('Checking for a saved explanation…');
  element('explanation-response').hidden = true;
  try {
    const explanation = await fetchJson(`${lookup.apiBase}/explain/${encodeURIComponent(lookup.transactionId)}`);
    validateExplanation(explanation, lookup.transactionId);
    if (explanation.status === 'PENDING') {
      showLookupMessage('The API reports this explanation is pending. Check again later.');
    } else {
      element('explanation-response').textContent = JSON.stringify(explanation, null, 2);
      element('explanation-response').hidden = false;
      showLookupMessage(explanation.status === 'FAILED' ? 'The API reports explanation processing failed.' : 'Saved response received from the API. Method and units are included when supplied.');
    }
  } catch (error) {
    showLookupMessage(explanationError(error, lookup.transactionId));
  } finally {
    busy = false;
    updateControls();
  }
});

element('manual-explanation-button').addEventListener('click', async () => {
  const id = element('manual-transaction-id').value.trim();
  if (busy || !apiBaseValid || !validTransactionId(id)) return;
  const lookupBase = apiBase;
  const selectedApi = displayApi(lookupBase);
  const showMessage = (message) => setContextMessage(element('manual-explanation-message'), message, ` Transaction ID: ${id}. Selected API: ${selectedApi}.`);
  clearManualResult();
  const revision = manualRevision;
  busy = true;
  updateControls();
  showMessage('Checking for a saved explanation…');
  try {
    const explanation = await fetchJson(`${lookupBase}/explain/${encodeURIComponent(id)}`);
    if (revision !== manualRevision) return;
    validateExplanation(explanation, id);
    if (explanation.status === 'PENDING') {
      showMessage('The API reports this explanation is pending. Check again later.');
    } else {
      element('manual-explanation-response').textContent = JSON.stringify(explanation, null, 2);
      element('manual-explanation-response').hidden = false;
      showMessage(explanation.status === 'FAILED' ? 'The API reports explanation processing failed.' : 'Saved response received. Method and units are included when supplied.');
    }
  } catch (error) {
    if (revision !== manualRevision) return;
    showMessage(explanationError(error, id));
  } finally {
    busy = false;
    updateControls();
  }
});

connect();
