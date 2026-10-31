import { describe, expect, it } from 'vitest';
import { buildBrief } from './brief';

const scene = { id: 's', event_type: 'possible_collision', start_ms: 1200, observations: { scene_cue: { frames: 10, peak_confidence: 0.94 } } };
const motion = { id: 'm', event_type: 'possible_collision', start_ms: 800, observations: { collision_signals: ['approach', 'arrest', 'shape'] } };
const moment = { ms: 1200, evidenceId: 'ev1' };

describe('buildBrief', () => {
  it('is a headline, at most three lines and a caveat', () => {
    const b = buildBrief({ moment, events: [scene, motion], gaps: [{ gap_type: 'no_camera_calibration' }, { gap_type: 'x' }], nameOf: () => 'cctv 3.mp4' });
    expect(b.headline).toBe('Possible accident at 00:01.200 on cctv 3.mp4. Not confirmed.');
    expect(b.lines).toEqual(['The accident-scene model flagged 10 frames in a row (peak 94%).', 'Two vehicles closed on each other and their movement stopped.', 'Missing: no camera calibration (+1 more).']);
    expect(b.caveat).toMatch(/person must confirm/);
  });
  it('says movement does not confirm contact when only the scene cue fired', () => {
    expect(buildBrief({ moment, events: [scene] }).lines).toContain('Vehicle movement alone does not confirm contact.');
  });
  it('never claims there was no accident', () => {
    const b = buildBrief({ cueRan: true });
    expect(b.headline).toBe('No accident was detected in the analysed footage.');
    expect(b.caveat).toMatch(/does not show that no accident happened/);
    expect(buildBrief({ cueRan: false }).headline).toMatch(/did not run/);
  });
});
