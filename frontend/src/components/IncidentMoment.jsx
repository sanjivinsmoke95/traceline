import { useMemo } from 'react';
import { AlertTriangle, Play, Crosshair } from 'lucide-react';
import { Badge, Button } from '../ui-kit';
import { useWorkspace } from '../store';
import { formatTime, formatUncertainty } from '../lib/time';
import { momentBasis, momentLabel, nearbyEvidence } from '../lib/moment';

const KIND_TONE = { sped_up: 'text-warning', slowed: 'text-fg-2', steady: 'text-fg-2', stationary: 'text-fg-3', unknown: 'text-fg-3' };

/** The possible accident moment, highlighted, with what the stored data shows around it. Image-space motion is relative to each vehicle's own size and the camera;
 *  it is not a real speed. Nothing here says what caused the event. */
export default function IncidentMoment({ moment, events, nameOf, motions }) {
  const near = useMemo(() => nearbyEvidence(events, moment), [events, moment]);
  if (!moment) {
    return (
      <section aria-label="Possible accident moment" className="rounded-xl border border-line bg-surface p-4">
        <h2 className="t-eyebrow text-fg-3">Possible accident moment</h2>
        <p className="t-small mt-1 text-fg-2">No moment in the stored data points to a collision: no camera collision candidate and no sensor collision flag. That is a result, not a failure.</p>
      </section>
    );
  }
  const go = (ms, play) => { const st = useWorkspace.getState(); st.setPlaying(false); st.requestSeek(ms); if (play) setTimeout(() => useWorkspace.getState().setPlaying(true), 150); };
  return (
    <section aria-label="Possible accident moment" className="rounded-xl border-2 border-danger/70 bg-danger/5 p-4">
      <div className="flex flex-wrap items-center gap-3">
        <AlertTriangle className="h-5 w-5 text-danger" aria-hidden />
        <h2 className="text-lg font-semibold text-fg">{momentLabel(moment)}</h2>
        <Badge variant="warning">candidate, contact not confirmed</Badge>
        <span className="ml-auto flex gap-2">
          <Button size="sm" variant="outline" icon={Crosshair} onClick={() => go(moment.ms, false)}>Go to moment</Button>
          <Button size="sm" variant="primary" icon={Play} onClick={() => go(Math.max(0, moment.ms - 1500), true)}>Play 1.5 s before</Button>
        </span>
      </div>
      <p className="t-caption mt-1 text-fg-3">Case clock {formatTime(moment.ms)} {formatUncertainty(moment.uncertaintyMs)} · {momentBasis(moment)}{moment.evidenceId ? ` · ${nameOf(moment.evidenceId)}` : ''}</p>

      <h3 className="t-eyebrow mt-3 text-fg-3">At this moment</h3>
      <ul className="mt-1 space-y-1.5" aria-label="At this moment">
        <li className="t-small text-fg">{moment.event.title}</li>
        {motions === null && <li className="t-caption text-fg-3">Reading the involved vehicles' stored positions…</li>}
        {(motions || []).map((m, i) => <li key={i} className={`t-small ${KIND_TONE[m.kind]}`}>{m.text}</li>)}
        {near.around.map((e) => <li key={e.id} className="t-small text-fg-2"><span className="mono text-fg-3">{formatTime(e.start_ms)}</span> {e.title}</li>)}
      </ul>
      {near.before.length > 0 && (
        <>
          <h3 className="t-eyebrow mt-3 text-fg-3">In the seconds before</h3>
          <ul className="mt-1 space-y-1" aria-label="Before the moment">{near.before.map((e) => <li key={e.id} className="t-small text-fg-2"><span className="mono text-fg-3">{formatTime(e.start_ms)}</span> {e.title}</li>)}</ul>
        </>
      )}
      <p className="t-caption mt-3 text-fg-3">Pace figures are image-space (relative to each vehicle's size, camera-dependent), not real speeds. TraceLine reports what the stored data shows around this moment; it does not say that any vehicle caused it or who is responsible.</p>
    </section>
  );
}

/** One line above the video: where the possible accident is, with a jump. The playhead turns red inside it (see Scrubber and Timeline). */
export function MomentBar({ moment }) {
  const near = useWorkspace((s) => !!s.incident && Math.abs(s.currentMs - s.incident.ms) <= 600);
  if (!moment) return null;
  return (
    <div role="status" aria-label="Accident moment" className={`mb-2 flex flex-wrap items-center gap-2 rounded-lg border px-3 py-1.5 ${near ? 'border-danger bg-danger/15' : 'border-danger/40 bg-danger/5'}`}>
      <AlertTriangle className="h-4 w-4 text-danger" aria-hidden />
      <span className="t-small font-medium text-fg">{near ? 'You are at the possible accident moment' : momentLabel(moment)}</span>
      <span className="t-caption text-fg-3">Not confirmed</span>
      <Button size="sm" variant="ghost" className="ml-auto" onClick={() => { const st = useWorkspace.getState(); st.setPlaying(false); st.requestSeek(moment.ms); }}>Go to moment</Button>
    </div>
  );
}
