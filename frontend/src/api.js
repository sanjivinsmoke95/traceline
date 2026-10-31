// Thin fetch wrapper. All data comes from the TraceLine API; nothing is invented client-side.
const BASE = '/api/v1';

export class ApiError extends Error {
  constructor(status, body) {
    super(body?.error?.message || `HTTP ${status}`);
    this.status = status;
    this.code = body?.error?.code;
  }
}

export const API_DOWN = 'The TraceLine backend did not respond. The web app only shows data from the API (port 8000), so it must be running with PostgreSQL: start it with ./run.sh, or see "Run it" in the README.';

/** A dead backend shows up as a failed fetch, or as the dev proxy's empty 5xx; either way, say what is missing instead of "HTTP 500". */
async function req(path, opts = {}) {
  let res;
  try {
    res = await fetch(BASE + path, { headers: { 'Content-Type': 'application/json', 'X-Actor': 'investigator' }, ...opts });
  } catch (e) {
    if (e?.name === 'AbortError') throw e;
    throw new ApiError(0, { error: { code: 'api_unreachable', message: API_DOWN } });
  }
  const type = res.headers.get('content-type') || '';
  const body = type.includes('json') ? await res.json().catch(() => null) : await res.text();
  if (!res.ok) {
    if (res.status >= 500 && !body?.error) throw new ApiError(res.status, { error: { code: 'api_unreachable', message: API_DOWN } });
    throw new ApiError(res.status, body);
  }
  return body;
}

const q = (o = {}) => {
  const p = new URLSearchParams(Object.entries(o).filter(([, v]) => v !== undefined && v !== null && v !== ''));
  const s = p.toString();
  return s ? `?${s}` : '';
};

export const api = {
  dashboard: () => req('/dashboard'),
  cases: () => req('/cases'),
  case: (id) => req(`/cases/${id}`),
  evidence: (id) => req(`/cases/${id}/evidence`),
  actors: (id) => req(`/cases/${id}/actors`),
  scenarios: (id) => req(`/cases/${id}/reconstruction/scenarios`),
  buildScenario: (id, body) => req(`/cases/${id}/reconstruction/scenarios`, { method: 'POST', body: JSON.stringify(body) }),
  timeline: (id, scenario) => req(`/cases/${id}/timeline${q({ scenario_id: scenario })}`),
  incidentEvents: (id, scenario) => req(`/cases/${id}/incident-events${q({ scenario_id: scenario })}`),
  incidentNarrative: (id, scenario) => req(`/cases/${id}/incident-narrative${q({ scenario_id: scenario })}`),
  storyboard: (id, scenario) => req(`/cases/${id}/storyboard${q({ scenario_id: scenario })}`),
  incidentFrames: (id, scenario, eventId) => req(`/cases/${id}/incident-frames${q({ scenario_id: scenario, event_id: eventId })}`),
  associations: (id, scenario) => req(`/cases/${id}/associations${q({ scenario_id: scenario })}`),
  addAssociation: (id, body) => req(`/cases/${id}/associations`, { method: 'POST', body: JSON.stringify(body) }),
  completeness: (id, scenario) => req(`/cases/${id}/reconstruction/completeness${q({ scenario_id: scenario })}`),
  summary: (eventId) => req(`/events/${eventId}/summary`),
  contradictions: (id, s) => req(`/cases/${id}/contradictions${q({ scenario_id: s })}`),
  corroborations: (id, s) => req(`/cases/${id}/corroborations${q({ scenario_id: s })}`),
  gaps: (id, s) => req(`/cases/${id}/gaps${q({ scenario_id: s })}`),
  findings: (id, s) => req(`/cases/${id}/findings${q({ scenario_id: s })}`),
  review: (fid, status) => req(`/findings/${fid}`, { method: 'PATCH', body: JSON.stringify({ review_status: status }) }),
  trajectories: (id, s, frame) => req(`/cases/${id}/trajectories${q({ scenario_id: s, frame })}`),
  seek: (id, ms, s) => req(`/cases/${id}/seek${q({ case_ms: ms, scenario_id: s })}`),
  overlays: (evId, from, to) => req(`/evidence/${evId}/overlays${q({ from_ms: from, to_ms: to })}`),
  sensorRecords: (evId, from, to) => req(`/evidence/${evId}/sensor-records${q({ limit: 5000, from_ms: from, to_ms: to, fields: 'speed,accel_long,heading' })}`),
  event: (id) => req(`/events/${id}`),
  frames: (evId, from, to) => req(`/evidence/${evId}/frames${q({ from_ms: from, to_ms: to, sampled: false, limit: 30 })}`),
  statement: (id) => req(`/statements/${id}`),
  statements: (witnessId) => req(`/witnesses/${witnessId}/statements`),
  report: (id, scenario, format) => req(`/cases/${id}/reports`, { method: 'POST', body: JSON.stringify({ scenario_id: scenario || null, ...(format ? { format } : {}) }) }),
  reportById: (rid) => req(`/reports/${rid}`),
  witnesses: (id) => req(`/cases/${id}/witnesses`),
  claims: (stId) => req(`/statements/${stId}/claims`),
  reports: (id) => req(`/cases/${id}/reports`),
  reportMd: (rid) => req(`/reports/${rid}/markdown`),
  createCase: (body) => req('/cases', { method: 'POST', body: JSON.stringify(body) }),
  patchCase: (id, body) => req(`/cases/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
  sync: (id) => req(`/cases/${id}/sync`),
  patchSource: (evId, body) => req(`/evidence/${evId}/source`, { method: 'PATCH', body: JSON.stringify(body) }),
  setClock: (evId, body) => req(`/evidence/${evId}/clock`, { method: 'PUT', body: JSON.stringify(body) }),
  upload: async (caseId, file, kind, synthetic) => {
    const fd = new FormData();
    fd.append('file', file); fd.append('kind', kind); fd.append('synthetic', String(synthetic));
    let res;
    try { res = await fetch(`${BASE}/cases/${caseId}/evidence`, { method: 'POST', body: fd, headers: { 'X-Actor': 'investigator' } }); } catch { throw new ApiError(0, { error: { code: 'api_unreachable', message: API_DOWN } }); }
    const body = await res.json().catch(() => null);
    if (!res.ok) throw new ApiError(res.status, body || (res.status >= 500 ? { error: { code: 'api_unreachable', message: API_DOWN } } : null));
    return body;
  },
  jobs: (id) => req(`/cases/${id}/jobs`),
  sensors: (id) => req(`/cases/${id}/sensors`),
  sensorPreview: (evId, body) => req(`/evidence/${evId}/sensor-preview`, { method: 'POST', body: JSON.stringify(body) }),
  registerSensor: (id, body) => req(`/cases/${id}/sensors`, { method: 'POST', body: JSON.stringify(body) }),
  retryJob: (jobId) => req(`/jobs/${jobId}/retry`, { method: 'POST' }),
  addWitness: (id, label) => req(`/cases/${id}/witnesses`, { method: 'POST', body: JSON.stringify({ label }) }),
  addStatement: (wid, text, synthetic) => req(`/witnesses/${wid}/statements`, { method: 'POST', body: JSON.stringify({ text, synthetic }) }),
  fullFrameUrl: (frameId) => `${BASE}/frames/${frameId}/image?full=true`,
  modelRuns: (evId) => req(`/evidence/${evId}/model-runs`),
  tracks: (evId) => req(`/evidence/${evId}/tracks`),
  track: (pk) => req(`/tracks/${pk}`),
  streamUrl: (evId) => `${BASE}/evidence/${evId}/stream`,
  pdfUrl: (rid) => `${BASE}/reports/${rid}/pdf`,
};
