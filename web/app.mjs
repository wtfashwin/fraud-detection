import { normalizeApiBase, validateFeatures, validateModelInfo, validatePrediction, formatProbability, fetchJson } from './api-client.mjs';

const element = (id) => document.getElementById(id);
let modelInfo = null;
let apiBase = '';
let apiBaseValid = false;
let prediction = null;
let requestContext = null;
let busy = false;
let manualRevision = 0;
const validTransactionId = (value) => typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/.test(value);
const transactionErrorStates = new Set(['PENDING', 'PROCESSING', 'QUEUED', 'COMPLETED', 'FAILED', 'DISPATCH_UNCONFIRMED']);

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
}

function clearResult() {
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
  element('transaction-id').value = '';
  // The example is an arbitrary input vector; it carries no expected class or score.
  element('sample-note').textContent = modelInfo.feature_names.includes('Amount')
    ? 'Synthetic example: zeros for all features except Amount = 100. This is invented input for testing the request flow, with no expected fraud outcome.'
    : 'Synthetic example: zeros for all features. This is invented input for testing the request flow, with no expected fraud outcome.';
  element('sample-note').hidden = false;
  setMessage('input-error', '');
  clearResult();
  element('features').focus();
});

for (const id of ['features', 'transaction-id']) {
  element(id).addEventListener('input', () => {
    clearResult();
    setMessage('input-error', '');
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
    payload = { features, ...(transactionId ? { transaction_id: transactionId } : {}) };
    requestContext = transactionId ? { transactionId, apiBase: submittedBase } : null;
  } catch (error) {
    setMessage('input-error', error.message);
    element(invalidField).setAttribute('aria-invalid', 'true');
    element(invalidField).focus();
    return;
  }
  busy = true;
  updateControls();
  setMessage('request-status', 'Scoring with the connected API…');
  try {
    const response = validatePrediction(await fetchJson(`${submittedBase}/predict`, { method: 'POST', body: JSON.stringify(payload) }));
    if (payload.transaction_id && response.transaction_id !== payload.transaction_id) {
      throw new Error('The API returned a different transaction ID.');
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
    prediction = null;
    setMessage('request-status', '');
    const unverified = hasUnverifiedTransactionError(error, requestContext?.transactionId);
    const conflict = error.detail === 'Transaction ID already exists; use a new ID'
      ? 'HTTP 409: The API reported a conflict.' : error.message;
    const message = unverified
      ? 'The API returned an error with a missing or different transaction ID.'
      : error.status === 409 && requestContext
        ? `${conflict} Check the saved explanation before deciding whether to retry. Use a new ID only for a different transaction.`
        : error.message;
    setMessage('input-error', message);
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
  busy = true;
  updateControls();
  setMessage('explanation-message', 'Checking for a saved explanation…');
  element('explanation-response').hidden = true;
  try {
    const explanation = await fetchJson(`${lookup.apiBase}/explain/${encodeURIComponent(lookup.transactionId)}`);
    validateExplanation(explanation, lookup.transactionId);
    if (explanation.status === 'PENDING') {
      setMessage('explanation-message', 'The API reports this explanation is pending. Check again later.');
    } else {
      element('explanation-response').textContent = JSON.stringify(explanation, null, 2);
      element('explanation-response').hidden = false;
      setMessage('explanation-message', explanation.status === 'FAILED' ? 'The API reports explanation processing failed.' : 'Saved response received from the API. Method and units are included when supplied.');
    }
  } catch (error) {
    setMessage('explanation-message', explanationError(error, lookup.transactionId));
  } finally {
    busy = false;
    updateControls();
  }
});

element('manual-explanation-button').addEventListener('click', async () => {
  const id = element('manual-transaction-id').value.trim();
  if (busy || !apiBaseValid || !validTransactionId(id)) return;
  const lookupBase = apiBase;
  const selectedApi = lookupBase || 'this server';
  const showMessage = (message) => setMessage('manual-explanation-message', `${message} Selected API: ${selectedApi}.`);
  clearManualResult();
  const revision = manualRevision;
  busy = true;
  updateControls();
  showMessage(`Checking for a saved explanation for ${id}…`);
  try {
    const explanation = await fetchJson(`${lookupBase}/explain/${encodeURIComponent(id)}`);
    if (revision !== manualRevision) return;
    validateExplanation(explanation, id);
    if (explanation.status === 'PENDING') {
      showMessage(`The API reports the explanation for ${id} is pending. Check again later.`);
    } else {
      element('manual-explanation-response').textContent = JSON.stringify(explanation, null, 2);
      element('manual-explanation-response').hidden = false;
      showMessage(explanation.status === 'FAILED' ? `The API reports explanation processing failed for ${id}.` : `Saved response received for ${id}. Method and units are included when supplied.`);
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
