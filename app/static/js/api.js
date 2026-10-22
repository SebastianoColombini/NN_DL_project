// Thin wrappers around the HTTP API; errors carry the server's `detail`.

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function request(url, options) {
  let res;
  try {
    res = await fetch(url, options);
  } catch {
    throw new ApiError('Cannot reach the local server. Is it still running?', 0);
  }
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON body */
  }
  if (!res.ok) {
    let detail = body && body.detail;
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg || JSON.stringify(d)).join('; ');
    throw new ApiError(detail || `Request failed (HTTP ${res.status})`, res.status);
  }
  return body;
}

export const api = {
  health: () => request('/api/health'),
  models: () => request('/api/models'),
  samples: () => request('/api/samples'),
  predict: (form) => request('/api/predict', { method: 'POST', body: form }),
};
