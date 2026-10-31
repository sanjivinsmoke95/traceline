/** The "possible accident moment": picked from stored events only, with a plain description of what the stored data shows around it.
 *  It is a CANDIDATE: image-space overlap or a sensor flag, never a confirmed physical contact, and no sentence here says what caused it. */
import { formatTime } from './time';

const COLL = new Set(['possible_collision']);
const secs = (ms) => `${(ms / 1000).toFixed(1)} s`;

const sceneScore = (e) => (e.observations?.scene_cue ? (e.observations.scene_cue.strength === 'strong' ? 1 : 0.5) : 0) + (e.observations?.collision_signals?.length || 0) * 0.2;
const dropOf = (e) => sceneScore(e) + Math.max(0, ...((e.observations?.abrupt_speed_changes || []).map((c) => (c.relative_speed_before > 0 ? (c.relative_speed_before - c.relative_speed_after) / c.relative_speed_before : 0))));

/** events: placed timeline events; crbs: corroborations. Returns null when nothing in the stored data points to a moment. */
export function pickMoment(events = [], crbs = []) {
  const placed = events.filter((e) => e.start_ms != null && e.category !== 'inference');
  const byId = Object.fromEntries(placed.map((e) => [e.id, e]));
  const cams = placed.filter((e) => COLL.has(e.event_type));
  const flags = placed.filter((e) => e.event_type === 'collision_flag');
  const agreed = crbs.find((c) => c.rule_id === 'COR-COLL-01' && c.events.some((i) => byId[i]));
  let lead; let basis;
  if (agreed) { lead = agreed.events.map((i) => byId[i]).find((e) => e && COLL.has(e.event_type)) || agreed.events.map((i) => byId[i]).find(Boolean); basis = 'camera and sensor agree'; }
  else if (flags.length) { lead = [...flags].sort((a, b) => a.start_ms - b.start_ms)[0]; basis = 'vehicle sensor flag only'; }
  else if (cams.length) { lead = [...cams].sort((a, b) => dropOf(b) - dropOf(a) || a.start_ms - b.start_ms)[0]; basis = 'camera analysis only'; }
  if (!lead) return null;
  const cam = lead.source_type?.startsWith('video') ? lead : cams.find((e) => Math.abs(e.start_ms - lead.start_ms) <= 1500);
  const evidenceId = cam?.source_id || lead.source_id;
  const trackIds = cam?.observations?.track_ids || lead.observations?.track_ids || [];
  return { ms: lead.start_ms, endMs: lead.end_ms ?? lead.start_ms, uncertaintyMs: lead.uncertainty_ms || 0, basis, candidates: cams.length, eventId: lead.id, event: lead, videoEvent: cam || null,
    evidenceId, trackIds, trackKeys: trackIds.map((t) => `${evidenceId}:${t}`), mediaMs: cam?.observations?.source_time_ms?.[0] ?? null };
}

const diag = (b) => Math.hypot(b[2] - b[0], b[3] - b[1]);

/** Mean image-space speed (anchor pixels per second divided by the box diagonal) of one track in a media-time window. null if fewer than two points. */
export function windowSpeed(points = [], from, to) {
  const p = points.filter((x) => x.time_ms >= from && x.time_ms <= to);
  if (p.length < 2) return null;
  const v = [];
  for (let i = 1; i < p.length; i++) {
    const dt = (p[i].time_ms - p[i - 1].time_ms) / 1000;
    if (dt <= 0) continue;
    v.push(Math.hypot(p[i].anchor[0] - p[i - 1].anchor[0], p[i].anchor[1] - p[i - 1].anchor[1]) / Math.max(diag(p[i - 1].bbox), 1e-6) / dt);
  }
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}

/** Plain description of one track's image-space motion before / at / after the moment (media ms). Returns { before, at, after, text, kind }. */
export function describeMotion(track, atMediaMs) {
  const pts = track?.points || [];
  const before = windowSpeed(pts, atMediaMs - 2500, atMediaMs - 600);
  const at = windowSpeed(pts, atMediaMs - 600, atMediaMs + 600);
  const after = windowSpeed(pts, atMediaMs + 600, atMediaMs + 2500);
  const f = (x) => x.toFixed(2);
  const name = `${track?.class || 'object'} #${track?.track_id}`;
  if (before == null || at == null) return { before, at, after, kind: 'unknown', text: `${name}: too few stored positions around this moment to describe its movement.` };
  let kind = 'steady'; let text;
  if (before >= 0.1 && at > before * 1.3 && at - before >= 0.25) { kind = 'sped_up'; text = `${name} moved faster just before this moment (image-space pace ${f(before)} to ${f(at)} vehicle-lengths per second).`; }
  else if (before >= 0.15 && at < before * 0.7) { kind = 'slowed'; text = `${name} was already slowing at this moment (image-space pace ${f(before)} to ${f(at)}).`; }
  else if (before < 0.15 && at < 0.15) { kind = 'stationary'; text = `${name} was nearly stationary in the image around this moment.`; }
  else text = `${name} kept a similar image-space pace (${f(before)} to ${f(at)}).`;
  if (after != null && before >= 0.15 && after < before * 0.3) text += ` Afterwards it nearly stopped (${f(after)}).`;
  else if (after != null && at >= 0.15 && after > at * 1.5) text += ` Afterwards it moved faster (${f(after)}).`;
  return { before, at, after, kind, text };
}

/** Stored events near the moment, for the "at this moment" list. Case-clock times. */
export function nearbyEvidence(events = [], moment) {
  if (!moment) return { before: [], around: [] };
  const rel = events.filter((e) => e.start_ms != null && e.id !== moment.eventId && e.category !== 'inference' && !['vehicle_enters_scene', 'vehicle_leaves_scene', 'potential_interaction', 'sensor_gap'].includes(e.event_type));
  const around = rel.filter((e) => Math.abs(e.start_ms - moment.ms) <= 1500).sort((a, b) => a.start_ms - b.start_ms);
  const before = rel.filter((e) => e.start_ms < moment.ms - 1500 && e.start_ms >= moment.ms - 4000).sort((a, b) => a.start_ms - b.start_ms);
  return { before: before.slice(-5), around: around.slice(0, 8) };
}

export const momentLabel = (m) => `Possible accident at ${formatTime(m.ms)}`;
export const momentBasis = (m) => `${m.basis}${m.candidates > 1 ? `; strongest of ${m.candidates} camera candidates` : ''}`;
export { secs };
