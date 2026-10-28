import { normalizeApiBase, validateFeatures, validateModelInfo, validatePrediction, formatProbability, fetchJson } from './api-client.mjs';

const element = (id) => document.getElementById(id);
let modelInfo = null;
let apiBase = '';
let apiBaseValid = false;
let prediction = null;
let confirmedTransactionId = null;
let busy = false;
const transactionIdPattern = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/;

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
  element('explanation-button').disabled = busy || !apiBaseValid || !transactionIdPattern.test(element('transaction-id').value.trim());
  element('result-panel').setAttribute('aria-busy', String(busy));
  element('score-button').textContent = busy ? 'Please wait…' : 'Score transaction';
}

function clearResult() {
  prediction = null;
  element('result').hidden = true;
  element('empty-result').hidden = false;
  element('explanation-response').hidden = true;
  setMessage('explanation-message', '');
  setMessage('request-status', '');
  updateControls();
}

async function connect(event) {
  event?.preventDefault();
  if (busy) return;
  modelInfo = null;
  apiBaseValid = false;
  clearResult();
  busy = true;
  updateControls();
  setMessage('connection-error', '');
  setMessage('input-error', '');
  element('connection-status').textContent = 'Connecting…';
  element('connection-status').dataset.state = 'loading';
  element('feature-order').replaceChildren();
  element('feature-count').textContent = '';
  try {
    const nextBase = normalizeApiBase(element('api-base').value);
    if (nextBase !== apiBase) confirmedTransactionId = null;
    apiBase = nextBase;
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
  confirmedTransactionId = null;
  clearResult();
  element('connection-status').textContent = 'URL changed. Reconnect to load this model.';
  element('connection-status').dataset.state = 'loading';
  element('feature-order').replaceChildren();
  element('feature-count').textContent = '';
});

element('sample-button').addEventListener('click', () => {
  if (!modelInfo || busy) return;
  const values = modelInfo.feature_names.map((name) => name === 'Amount' ? 100 : 0);
  element('features').value = JSON.stringify(values);
  if (confirmedTransactionId && element('transaction-id').value.trim() === confirmedTransactionId) {
    element('transaction-id').value = '';
    confirmedTransactionId = null;
  }
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
    if (id === 'transaction-id') confirmedTransactionId = null;
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
  let invalidField = 'features';
  try {
    const features = validateFeatures(element('features').value, modelInfo.feature_count);
    invalidField = 'transaction-id';
    let transactionId = element('transaction-id').value.trim();
    if (transactionId && !transactionIdPattern.test(transactionId)) {
      throw new Error('Use 1–128 letters, numbers, dots, underscores, colons, or hyphens for the ID. Start with a letter or number.');
    }
    if (!transactionId && typeof globalThis.crypto?.randomUUID === 'function') {
      transactionId = crypto.randomUUID();
    }
    if (!transactionId) throw new Error('Enter a unique transaction ID. This browser cannot generate one automatically.');
    element('transaction-id').value = transactionId;
    payload = { features, transaction_id: transactionId };
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
    confirmedTransactionId = null;
    prediction = validatePrediction(await fetchJson(`${apiBase}/predict`, { method: 'POST', body: JSON.stringify(payload) }));
    if (prediction.transaction_id !== payload.transaction_id) throw new Error('The API returned a different transaction ID.');
    confirmedTransactionId = payload.transaction_id;
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
    confirmedTransactionId = null;
    setMessage('request-status', '');
    const id = element('transaction-id').value.trim();
    if (error.status === 409) {
      setMessage('input-error', `This transaction ID already exists. Check the saved explanation for ${id} before deciding what to do next.`);
    } else if (error.status === 422) {
      setMessage('input-error', error.message);
    } else {
      setMessage('input-error', `${error.message} This request may have been processed. Check the saved explanation for ${id} before retrying.`);
    }
  } finally {
    busy = false;
    updateControls();
  }
});

element('explanation-button').addEventListener('click', async () => {
  const id = element('transaction-id').value.trim();
  if (!apiBaseValid || busy || !transactionIdPattern.test(id)) return;
  const lookupBase = apiBase;
  const sourceUrl = lookupBase ? new URL(lookupBase) : null;
  const lookupSource = sourceUrl ? `${sourceUrl.origin}${sourceUrl.pathname.replace(/\/+$/, '')}` : 'this server';
  const showLookupMessage = (message) => setMessage('explanation-message', `${message} Selected API: ${lookupSource}.`);
  busy = true;
  updateControls();
  showLookupMessage(`Checking for a saved explanation for ${id}…`);
  element('explanation-response').hidden = true;
  try {
    const explanation = await fetchJson(`${lookupBase}/explain/${encodeURIComponent(id)}`);
    if (explanation.transaction_id !== id) throw new Error('The API returned an explanation for a different transaction ID.');
    if (explanation.status === 'PENDING') {
      showLookupMessage(`The API reports the explanation for ${id} is pending. Check again later.`);
    } else {
      element('explanation-response').textContent = JSON.stringify(explanation, null, 2);
      element('explanation-response').hidden = false;
      showLookupMessage(explanation.status === 'FAILED' ? `The API reports explanation processing failed for ${id}.` : `Saved response received for ${id}. Method and units are included when supplied.`);
    }
  } catch (error) {
    showLookupMessage(error.status === 404 ? `Saved explanation not found for ${id}. This does not establish whether the request was processed.` : `Could not retrieve the explanation for ${id}. ${error.message}`);
  } finally {
    busy = false;
    updateControls();
  }
});

connect();
