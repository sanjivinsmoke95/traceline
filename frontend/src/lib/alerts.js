/** Event policy and the deterministic playback trigger. Pure functions; the player, timeline and alert all read one clock (the store's currentMs).
 *  Only validated event records trigger anything: a raw detection or a box overlap alone never does.
 *    tier 'accident'  : possible_collision with an abrupt movement change, or a sensor collision flag   -> may pause playback
 *    tier 'attention' : braking, slowing, stopping, direction change, near miss, weak collision cue       -> listed, can alert if the policy says so
 *    tier 'info'      : enters / leaves the scene, steady maneuver                                       -> never alerts */
export const RANK = { info: 0, attention: 1, accident: 2 };
export const TIER_LABEL = { info: 'Informational', attention: 'Attention', accident: 'Possible accident' };
export const DEFAULT_POLICY = { autoPause: true, minTier: 'accident', cooldownMs: 3000 };
const STRONG = (e) => (e.observations?.abrupt_speed_changes?.length || 0) > 0;

export function tierOf(e) {
  if (e.category === 'inference' || e.start_ms == null) return 'info';
  switch (e.event_type) {
    case 'collision_flag': return e.source_type === 'sensor' ? 'accident' : 'attention';
    case 'possible_collision': return STRONG(e) ? 'accident' : 'attention'; // a collision cue without a movement change is a weak cue
    case 'possible_near_miss': case 'hard_braking': case 'acceleration_spike': case 'vehicle_slowing': case 'vehicle_stopped':
    case 'vehicle_direction_change': case 'potential_interaction': return 'attention';
    default: return 'info';
  }
}
export const isWeakCollisionCue = (e) => e.event_type === 'possible_collision' && !STRONG(e);

/** Strength used to pick the lead of a cluster: the biggest relative speed drop. */
const drop = (e) => Math.max(0, ...(e.observations?.abrupt_speed_changes || []).map((c) => (c.relative_speed_before > 0 ? (c.relative_speed_before - c.relative_speed_after) / c.relative_speed_before : 0)));

/** Placed events at or above the policy tier, merged into clusters so one incident pauses once. Sorted by time. */
export function buildTriggers(events, policy = DEFAULT_POLICY) {
  const min = RANK[policy.minTier] ?? 2;
  const list = events.filter((e) => e.start_ms != null && RANK[tierOf(e)] >= min).sort((a, b) => a.start_ms - b.start_ms || (a.id < b.id ? -1 : 1));
  const out = [];
  for (const e of list) {
    const last = out[out.length - 1];
    if (last && e.start_ms - last.startMs <= (policy.cooldownMs ?? 3000)) { last.members.push(e); if (drop(e) > drop(last.lead)) last.lead = e; }
    else out.push({ startMs: e.start_ms, lead: e, members: [e] });
  }
  return out.map((c) => ({ ...c, atMs: c.lead.start_ms }));
}

export const createTriggerState = () => ({ fired: new Set() });

/** One clock step. Returns { fire, state }. `fire` is a trigger to alert on, only for continuous forward playback:
 *  - a backward seek forgets the triggers after the new position, so replaying can alert again;
 *  - a forward jump (scrubbing, clicking a card) marks the skipped triggers as seen without alerting;
 *  - a fired trigger never fires again until the playhead has gone back before it. */
export function advance(triggers, state, prevMs, curMs, { maxStepMs = 500 } = {}) {
  const fired = new Set(state.fired);
  if (curMs < prevMs - 100) { // a real backward seek; smaller moves are the decoder landing a few ms before a requested time
    triggers.forEach((t) => { if (t.atMs > curMs) fired.delete(t.atMs); });
    return { fire: null, state: { fired } };
  }
  const crossed = triggers.filter((t) => t.atMs > prevMs && t.atMs <= curMs && !fired.has(t.atMs));
  if (!crossed.length) return { fire: null, state: { fired } };
  if (curMs - prevMs > maxStepMs) { crossed.forEach((t) => fired.add(t.atMs)); return { fire: null, state: { fired } }; }
  const t = crossed[0];
  crossed.forEach((c) => fired.add(c.atMs));
  return { fire: t, state: { fired } };
}

/** Plain explanation built from the event record only. Always says it is a possible event and not a confirmed collision. */
export function alertText(e) {
  const abrupt = e.observations?.abrupt_speed_changes?.length > 0;
  if (e.event_type === 'collision_flag') return 'A vehicle sensor flagged a possible impact. This is a possible accident event, not a confirmed collision, and it requires investigator review.';
  if (e.event_type === 'possible_collision') return `Vehicle paths appear to overlap in the image${abrupt ? ', followed by an abrupt movement change' : ''}. This is a possible accident event, not a confirmed collision, and it requires investigator review.`;
  return `${e.title}. This is a cue to review, not a confirmed collision.`;
}

/** Observations behind the alert, from the record (image-space pace is relative to each vehicle's size, not a real speed). */
export function supportingObservations(e, nearby = []) {
  const o = e.observations || {};
  const lines = [];
  (o.abrupt_speed_changes || []).forEach((c) => lines.push(`Track #${c.track_id}: movement pace fell from ${c.relative_speed_before?.toFixed?.(2)} to ${c.relative_speed_after?.toFixed?.(2)} (image space)`));
  const SIGNAL = { approach: 'the two objects closed on each other quickly', arrest: 'their relative motion was stopped or deflected right after contact', rest: 'they stayed together afterwards',
    shape: 'one object changed shape abruptly (rolled, flipped or fell)', cue: 'the weak crash-cue model fired on the pair over several frames' };
  if (o.collision_signals?.length) lines.push(`Why it was flagged: ${o.collision_signals.map((k) => SIGNAL[k] || k).join('; ')}`);
  if (o.min_gap_in_diagonals != null) lines.push(`Closest approach in the image: ${o.min_gap_in_diagonals} vehicle diagonals`);
  if (o.max_box_overlap_iou != null) lines.push(`Largest box overlap: ${o.max_box_overlap_iou} (image-space overlap is not contact)`);
  nearby.forEach((n) => lines.push(`${n.title}`));
  return lines;
}

export const DECISIONS = { reviewed: 'Reviewed', relevant: 'Confirmed as relevant', not_relevant: 'Not relevant', inconclusive: 'Inconclusive' };

const EXPLAIN = {
  vehicle_enters_scene: 'A road user first appears in this camera view.',
  vehicle_leaves_scene: 'A tracked road user is no longer seen (it may have left or been lost).',
  vehicle_slowing: 'A tracked vehicle appears to slow down (image-space movement).',
  vehicle_stopped: 'A tracked vehicle appears to stop.',
  vehicle_direction_change: 'A tracked vehicle changes direction in the image.',
  vehicle_maneuver_observed: 'The overall movement of a tracked vehicle across this view.',
  potential_interaction: 'Two tracked boxes come close or overlap in the image.',
  possible_near_miss: 'Two road users come close in the image without a clear overlap.',
  hard_braking: 'The vehicle sensor recorded hard braking.',
  acceleration_spike: 'The vehicle sensor recorded a sharp acceleration change.',
  collision_flag: 'The vehicle sensor raised a collision flag.',
  possible_collision: 'Paths overlap in the image; a strong cue also shows an abrupt movement change.',
};
/** One plain sentence for a timeline card, from the event type; witness claims show their own quote. */
export function explainEvent(e) {
  if (e.category === 'witness_claim') return e.observations?.quote ? `Witness says: “${e.observations.quote}”` : 'A witness statement.';
  if (e.event_type === 'possible_collision' && !(e.observations?.abrupt_speed_changes?.length)) return 'Paths overlap in the image, but no abrupt movement change supports it: a weak cue.';
  return EXPLAIN[e.event_type] || e.title;
}
