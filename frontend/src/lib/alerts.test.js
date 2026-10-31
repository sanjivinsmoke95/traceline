import { describe, expect, it } from 'vitest';
import { DEFAULT_POLICY, advance, alertText, buildTriggers, createTriggerState, explainEvent, supportingObservations, tierOf } from './alerts';

const E = (id, type, ms, extra = {}) => ({ id, event_type: type, category: 'machine_detection', start_ms: ms, end_ms: ms, uncertainty_ms: 100, source_type: 'video_cctv', source_id: 'ev1', title: `${type} ${id}`, observations: {}, ...extra });
const strong = (id, ms, drop = 0.6) => E(id, 'possible_collision', ms, { observations: { abrupt_speed_changes: [{ track_id: 3, relative_speed_before: 1, relative_speed_after: 1 - drop }] } });
const run = (triggers, positions, state = createTriggerState()) => { const fired = []; let prev = positions[0]; for (const cur of positions.slice(1)) { const r = advance(triggers, state, prev, cur); state = r.state; if (r.fire) fired.push([cur, r.fire.lead.id]); prev = cur; } return { fired, state }; };
const frames = (a, b, step = 40) => { const out = []; for (let t = a; t <= b; t += step) out.push(t); return out; };

describe('event policy', () => {
  it('only a collision cue with an abrupt movement change, or a sensor collision flag, is a possible accident', () => {
    expect(tierOf(strong('a', 1000))).toBe('accident');
    expect(tierOf(E('w', 'possible_collision', 1000))).toBe('attention'); // weak cue: overlap alone
    expect(tierOf(E('s', 'collision_flag', 1000, { source_type: 'sensor' }))).toBe('accident');
    expect(tierOf(E('p', 'potential_interaction', 1000))).toBe('attention');
    expect(tierOf(E('i', 'vehicle_enters_scene', 1000))).toBe('info');
    expect(tierOf(E('u', 'possible_collision', null))).toBe('info');
  });
  it('weak vehicle detections never produce a trigger by default', () => {
    const t = buildTriggers([E('w', 'possible_collision', 1000), E('p', 'potential_interaction', 1500), E('i', 'vehicle_enters_scene', 200), E('b', 'hard_braking', 800, { source_type: 'sensor' })]);
    expect(t).toEqual([]);
  });
  it('the policy is configurable: attention events can trigger too', () => {
    expect(buildTriggers([E('b', 'hard_braking', 800)], { ...DEFAULT_POLICY, minTier: 'attention' })).toHaveLength(1);
  });
  it('nearby cues merge into one trigger led by the strongest', () => {
    const t = buildTriggers([strong('a', 4600, 0.3), strong('b', 5200, 0.8), strong('c', 5900, 0.4), strong('far', 12000)]);
    expect(t.map((x) => x.lead.id)).toEqual(['b', 'far']);
    expect(t[0].members).toHaveLength(3);
  });
});

describe('playback trigger (the acceptance scenarios)', () => {
  const triggers = buildTriggers([strong('a', 4600), strong('b', 9000)]);
  it('1. a video with no accident candidate plays on without any alert', () => {
    expect(run(buildTriggers([E('i', 'vehicle_enters_scene', 1000)]), frames(0, 12000)).fired).toEqual([]);
  });
  it('2. playing through a qualifying event fires it once, at the first clock tick at or after its time', () => {
    const { fired } = run(triggers, frames(0, 5000));
    expect(fired).toHaveLength(1);
    expect(fired[0][1]).toBe('a');
    expect(fired[0][0]).toBeGreaterThanOrEqual(4600);
    expect(fired[0][0] - 4600).toBeLessThan(40); // within one frame step
  });
  it('8. after the alert, resuming from the event does not fire it again', () => {
    const first = run(triggers, frames(0, 4620));
    expect(first.fired).toHaveLength(1);
    const resumed = run(triggers, [4600, 4590, ...frames(4630, 8000)], first.state); // decoder lands a few ms before the requested time
    expect(resumed.fired).toEqual([]);
  });
  it('10. several events fire in chronological order, each once', () => {
    expect(run(triggers, frames(0, 10000)).fired.map((f) => f[1])).toEqual(['a', 'b']);
  });
  it('9. seeking back before the event and playing again fires it again', () => {
    const first = run(triggers, frames(0, 5000));
    const back = run(triggers, [5000, 3000, ...frames(3040, 5000)], first.state);
    expect(back.fired.map((f) => f[1])).toEqual(['a']);
  });
  it('seeking back to after the event does not re-arm it', () => {
    const first = run(triggers, frames(0, 5000));
    expect(run(triggers, [5000, 4800, ...frames(4840, 6000)], first.state).fired).toEqual([]);
  });
  it('a forward jump over an event (scrubbing or clicking a card) does not alert, and does not alert later either', () => {
    const r = run(triggers, [3000, 7000, ...frames(7040, 8000)]);
    expect(r.fired).toEqual([]);
  });
  it('a jump that lands exactly on the event time does not alert either', () => {
    expect(run(triggers, [3000, 4600, ...frames(4640, 5000)]).fired).toEqual([]);
  });
  it('switching camera or scenario resets the trigger state', () => {
    const first = run(triggers, frames(0, 5000));
    const reset = run(triggers, frames(0, 5000), createTriggerState());
    expect(first.fired).toHaveLength(1);
    expect(reset.fired).toHaveLength(1);
  });
  it('13. unsynchronized events (no case time) are never triggers', () => {
    expect(buildTriggers([E('u', 'possible_collision', null, { observations: { abrupt_speed_changes: [{ relative_speed_before: 1, relative_speed_after: 0.1 }] } })])).toEqual([]);
  });
});

describe('alert text', () => {
  it('always says possible, never confirmed, and never assigns a cause', () => {
    for (const e of [strong('a', 1000), E('s', 'collision_flag', 1000, { source_type: 'sensor' }), E('w', 'possible_collision', 1000)]) {
      const t = alertText(e);
      expect(t).toMatch(/possible|cue/i);
      expect(t).toMatch(/not a confirmed collision/);
      expect(t).not.toMatch(/caus|fault|blame|responsib|speeding/i);
    }
    expect(alertText(strong('a', 1))).toMatch(/abrupt movement change/);
    expect(explainEvent(E('w', 'possible_collision', 1))).toMatch(/weak cue/);
  });
});

describe('collision signals', () => {
  it('lists why a possible collision was flagged, in plain words', () => {
    const lines = supportingObservations({ observations: { collision_signals: ['approach', 'arrest', 'cue'] } });
    expect(lines.join(' ')).toMatch(/closed on each other quickly.*stopped or deflected.*weak crash-cue/);
  });
});
