/** The short brief shown first: one headline, up to three plain lines, one caveat. Built only from stored records; nothing is invented.
 *  Everything else (factors, tracks, scores, raw counts) stays available but folded away. */
import { formatTime } from './time';
import { gapTitle } from './findings';

const near = (e, ms, within = 3000) => e.start_ms != null && Math.abs(e.start_ms - ms) <= within;
const pct = (x) => `${Math.round((x || 0) * 100)}%`;

export function buildBrief({ moment = null, events = [], gaps = [], nameOf = (x) => x, cueRan = false } = {}) {
  const caveat = 'Flagged by software. A person must confirm it.';
  const lines = [];
  const missing = gaps.length ? `Missing: ${gapTitle(gaps[0].gap_type).toLowerCase()}${gaps.length > 1 ? ` (+${gaps.length - 1} more)` : ''}.` : null;
  if (!moment) {
    return {
      tone: 'neutral', ms: null, caveat: 'This does not show that no accident happened.',
      headline: cueRan ? 'No accident was detected in the analysed footage.' : 'The accident check did not run on this footage, so nothing was flagged.',
      lines: missing ? [missing] : [],
    };
  }
  const cluster = events.filter((e) => e.event_type === 'possible_collision' && near(e, moment.ms));
  const scene = cluster.map((e) => e.observations?.scene_cue).find(Boolean);
  const motion = cluster.find((e) => e.observations?.collision_signals?.length);
  if (scene) lines.push(`The accident-scene model flagged ${scene.frames} frames in a row (peak ${pct(scene.peak_confidence)}).`);
  if (motion) {
    const s = motion.observations.collision_signals;
    lines.push(s.includes('approach') && s.includes('arrest') ? 'Two vehicles closed on each other and their movement stopped.' : 'Vehicle movement near this moment is consistent with an impact.');
  } else if (scene) lines.push('Vehicle movement alone does not confirm contact.');
  else lines.push('Two vehicle boxes overlapped and one slowed. This is a weak cue.');
  if (missing) lines.push(missing);
  const cam = moment.evidenceId ? nameOf(moment.evidenceId) : null;
  return { tone: 'alert', ms: moment.ms, caveat, lines: lines.slice(0, 3), headline: `Possible accident at ${formatTime(moment.ms)}${cam ? ` on ${cam}` : ''}. Not confirmed.` };
}
