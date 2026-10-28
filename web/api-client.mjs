export function normalizeApiBase(value) {
  const trimmed = value.trim();
  if (!trimmed) return '';
  let url;
  try { url = new URL(trimmed); } catch { throw new Error('Enter a full HTTP or HTTPS API URL, or leave it blank for this server.'); }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new Error('Use an HTTP or HTTPS URL without credentials, a query, or a fragment.');
  }
  return url.href.replace(/\/+$/, '');
}

export function validateFeatures(text, count) {
  let values;
  try { values = JSON.parse(text); } catch { throw new Error('Enter a valid JSON array of numbers, such as [0, 1.5, -2].'); }
  if (!Array.isArray(values) || values.length !== count) throw new Error(`Enter exactly ${count} numbers in the feature order below.`);
  if (!values.every((value) => typeof value === 'number' && Number.isFinite(value))) {
    throw new Error('Every feature must be a finite number. Strings, null, and infinite values are not accepted.');
  }
  return values;
}

export function validateModelInfo(info) {
  const names = info?.feature_names;
  if (!Array.isArray(names) || !names.length || names.length !== info.feature_count ||
      !names.every((name) => typeof name === 'string' && name.trim()) || new Set(names).size !== names.length) {
    throw new Error('The API returned incomplete feature metadata. Check the model bundle before scoring.');
  }
  return info;
}

export function validatePrediction(data) {
  if (![0, 1].includes(data?.prediction) || typeof data.transaction_id !== 'string' || !data.transaction_id ||
      typeof data.correlation_id !== 'string' || typeof data.explanation_status !== 'string' ||
      (data.score !== null && (typeof data.score !== 'number' || !Number.isFinite(data.score) || data.score < 0 || data.score > 1))) {
    throw new Error('The API returned an invalid prediction response. No result is displayed.');
  }
  return data;
}

export function formatProbability(score) {
  if (score === null) return 'Not available';
  if (score === 0) return '0%';
  if (score === 1) return '100%';
  if (score < 0.0001) return 'Less than 0.01%';
  if (score > 0.9999) return 'Greater than 99.99%';
  return `${(score * 100).toFixed(2)}%`;
}

function errorDetail(detail) {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((item) => `${item.loc?.join('.') || 'Input'}: ${item.msg || 'Invalid value'}`).join('; ');
  return detail ? JSON.stringify(detail) : 'The API did not provide an error description.';
}

export class RequestTimeoutError extends Error {}

export async function fetchJson(url, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(url, {
      ...options,
      headers: { Accept: 'application/json', ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers },
      signal: controller.signal,
    });
    let data;
    try { data = await response.json(); } catch { throw new Error(`HTTP ${response.status}: The API returned a non-JSON response.`); }
    if (!response.ok) {
      const error = new Error(`HTTP ${response.status}: ${errorDetail(data.detail)}`);
      error.status = response.status;
      error.detail = data.detail;
      throw error;
    }
    return data;
  } catch (error) {
    if (controller.signal.aborted) throw new RequestTimeoutError('The API did not respond within 15 seconds. A submitted request may still have been processed; check its transaction ID before retrying.');
    if (error instanceof TypeError) throw new Error('Cannot reach the API. Check the URL, server, and browser access settings, then reconnect.');
    throw error;
  } finally { clearTimeout(timer); }
}
