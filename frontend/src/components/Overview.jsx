import { useState } from 'react';
import { Alert, Badge, Button } from '../ui-kit';
import { ChevronDown, Crosshair, Play } from 'lucide-react';
import { ANALYSIS_LABEL, classOfEvent, sourceName } from '../lib/overview';
import { formatTime, formatUncertainty } from '../lib/time';
import { ClassBadge, EvidenceLine, FindingRow } from './FindingParts';
import IncidentMoment from './IncidentMoment';
import { useWorkspace } from '../store';

const TONE = { alert: 'border-l-danger', neutral: 'border-l-line' };

/** One headline, up to three plain lines, one caveat, and the two actions a reader needs. */
function Brief({ brief, onGo }) {
  return (
    <div className={`rounded-xl border border-line border-l-4 ${TONE[brief.tone]} bg-surface p-4`}>
      <p className="text-[1.0625rem] font-semibold leading-snug text-fg">{brief.headline}</p>
      {brief.lines.length > 0 && <ul className="mt-2 space-y-1">{brief.lines.map((l, i) => <li key={i} className="t-small text-fg-2">{l}</li>)}</ul>}
      <p className="t-caption mt-2 text-fg-3">{brief.caveat}</p>
      {brief.ms != null && <div className="mt-3 flex flex-wrap gap-2"><Button size="sm" variant="primary" icon={Play} onClick={() => onGo(Math.max(0, brief.ms - 1500), true)}>Play 1.5 s before</Button><Button size="sm" variant="outline" icon={Crosshair} onClick={() => onGo(brief.ms, false)}>Go to moment</Button></div>}
    </div>
  );
}

const Stat = ({ label, value }) => (
  <div className="rounded-lg border border-line bg-surface-2/40 px-3 py-2"><div className="font-display text-xl leading-none tabular-nums">{value}</div><div className="t-caption mt-1 text-fg-3">{label}</div></div>
);
const H = ({ children, tone = 'text-fg-3' }) => <h3 className={`t-eyebrow mb-2 ${tone}`}>{children}</h3>;

function Factor({ f, events, onPick, onOpenTrail }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="rounded-lg border border-line">
      <button type="button" aria-expanded={open} onClick={() => setOpen(!open)} className="flex w-full items-start gap-2 px-3 py-2.5 text-left">
        <span className="min-w-0 flex-1"><span className="t-small block font-medium text-fg">{f.title}</span><span className="t-caption mt-0.5 block text-fg-3">{f.status}</span></span>
        <ChevronDown className={`mt-1 h-4 w-4 shrink-0 text-fg-3 transition ${open ? 'rotate-180' : ''}`} aria-hidden />
      </button>
      {open && (
        <div className="space-y-3 border-t border-line px-3 py-3">
          <p className="t-small text-fg-2">{f.summary}</p>
          <p className="flex flex-wrap items-center gap-2"><ClassBadge value={f.classification} /><span className="t-caption text-fg-3">Verification: {f.status}</span></p>
          {f.supporting.length > 0 && <div><h4 className="t-eyebrow mb-1 text-fg-3">Evidence</h4><ul className="space-y-1.5">{f.supporting.map((r) => <EvidenceLine key={r.eventId + r.role} r={r} events={events} onPick={onPick} onOpenTrail={onOpenTrail} />)}</ul></div>}
          {f.contradicting.length > 0 && <div><h4 className="t-eyebrow mb-1 text-fg-3">Contradicting evidence</h4><ul className="space-y-1.5">{f.contradicting.map((r) => <EvidenceLine key={r.eventId + r.role} r={r} events={events} onPick={onPick} onOpenTrail={onOpenTrail} />)}</ul></div>}
          {f.supporting.length + f.contradicting.length === 0 && <p className="t-caption text-fg-3">No stored evidence exists for this: that is the finding.</p>}
        </div>
      )}
    </li>
  );
}

/** The default inspector view: what happened, the factors behind it (each expandable to its evidence), who may be involved, what is unknown and what to do next.
 *  Raw detections, tracks, scores and counts are folded under Technical details. Built from stored records; no model writes any of it. */
export default function Overview({ brief, model, executive, moment, motions, events, designation, activeId, nameOf = (x) => x, onSelectFinding, onSeeAll, onPick, onOpenTrail, onTechnical }) {
  const { overview: o } = model;
  const x = executive;
  const a = o.analysis;
  const imported = ['imported-synthetic', 'imported-external', 'mixed'].includes(a.detection);
  const go = (ms, play) => { const st = useWorkspace.getState(); st.setPlaying(false); st.requestSeek(ms); if (play) setTimeout(() => useWorkspace.getState().setPlaying(true), 150); };
  const sel = activeId ? model.events[activeId] : null;
  return (
    <div className="space-y-6" aria-label="Investigation overview">
      <section aria-label="What happened">
        <H>What happened?</H>
        {brief ? <Brief brief={brief} onGo={go} /> : (x ? <div className="space-y-2">{x.sentences.map((t, i) => <p key={i} className={i === 0 ? 'text-[1.0625rem] leading-7 text-fg' : 't-small text-fg-2'}>{t}</p>)}</div> : <p className="t-small text-fg-3">{o.headline}</p>)}
        {imported && <Alert variant="warning" className="mt-3" title="Imported boxes, not detector output">The boxes in this case were supplied from outside the detector. Movement and events are real rule output over those boxes, but they do not show how the detector performs on this footage.</Alert>}
        {designation === 'synthetic' && !imported && <p className="t-caption mt-2 text-warning">Synthetic demonstration case.</p>}
      </section>

      <details className="rounded-lg border border-line px-3 py-2" aria-label="More detail" open={!brief}>
        <summary className="t-small cursor-pointer text-fg-2">More detail: factors, who is involved, what is uncertain, next steps</summary>
        <div className="mt-4 space-y-6">
      {x && (
        <section aria-label="Contributing factors">
          <H>Factors the evidence points to</H>
          {x.factors.length === 0 ? <p className="t-small text-fg-3">No factor is recorded. The stored data does not show what preceded this moment.</p> : <ul className="space-y-2">{x.factors.map((f) => <Factor key={f.id} f={f} events={model.events} onPick={onPick} onOpenTrail={onOpenTrail} />)}</ul>}
          <p className="t-caption mt-2 text-fg-3">Observations to examine. A movement before an event is not shown to be its cause.</p>
        </section>
      )}

      {x && (
        <section aria-label="Potentially involved">
          <H>Potentially involved</H>
          {x.entities.length === 0 ? <p className="t-small text-fg-3">No vehicle or person is tied to the moment.</p> : (
            <ul className="space-y-1.5">{x.entities.map((e) => <li key={e.key} className="flex items-center gap-2 rounded-lg border border-line px-3 py-2"><span className="t-small flex-1 text-fg">{e.label}<span className="t-caption block text-fg-3">{e.detail}</span></span><Badge variant={e.status.startsWith('confirmed') ? 'success' : 'neutral'}>{e.status}</Badge></li>)}</ul>
          )}
          {x.counts.statements > 0 && <p className="t-caption mt-2 text-fg-3">{x.counts.statements} witness statement{x.counts.statements > 1 ? 's' : ''}, {x.counts.claims} claim{x.counts.claims === 1 ? '' : 's'} extracted.</p>}
        </section>
      )}

      {x && (
        <section aria-label="Evidence overview">
          <H>Evidence</H>
          <div className="flex flex-wrap gap-2">
            <Badge variant="neutral">{x.counts.videos} recording{x.counts.videos === 1 ? '' : 's'}</Badge>
            <Badge variant="neutral">{x.counts.sensors ? `${x.counts.sensors} sensor log${x.counts.sensors > 1 ? 's' : ''}` : 'no sensor log'}</Badge>
            <Badge variant="neutral">{x.counts.statements ? `${x.counts.statements} statement${x.counts.statements > 1 ? 's' : ''}` : 'no witness statement'}</Badge>
          </div>
        </section>
      )}

      <section aria-label="Unresolved questions">
        <H tone="text-warning">What remains uncertain</H>
        {o.uncertain.length === 0 ? <p className="t-small text-fg-3">No gaps or open questions were recorded.</p> : (
          <ul className="space-y-1.5">{o.uncertain.slice(0, 6).map((f) => (
            <li key={f.id}><button type="button" onClick={() => onSelectFinding(f)} className="flex w-full items-start gap-2 rounded-lg px-2 py-1.5 text-left transition hover:bg-surface-2"><i className="mt-1.5 h-2 w-2 shrink-0 rounded-full bg-warning" aria-hidden /><span className="t-small text-fg-2">{f.title}</span></button></li>
          ))}</ul>
        )}
        {o.uncertain.length > 6 && <Button size="sm" variant="ghost" onClick={onSeeAll}>{o.uncertain.length - 6} more in Findings</Button>}
      </section>

      <section aria-label="Recommended actions">
        <H>Recommended next steps</H>
        {!x || x.actions.length === 0 ? <p className="t-small text-fg-3">Nothing is flagged for immediate review.</p> : (
          <ol className="space-y-1.5">{x.actions.map((t, i) => (
            <li key={i}><button type="button" disabled={!t.findingId} onClick={() => t.findingId && onSelectFinding(model.byId[t.findingId])} className="flex w-full items-start gap-2 rounded-lg px-2 py-1.5 text-left transition hover:bg-surface-2 disabled:cursor-default"><span className="mono t-caption mt-0.5 text-accent-text">{i + 1}</span><span className="t-small text-fg">{t.text}</span></button></li>
          ))}</ol>
        )}
        {o.unreviewed > 0 && <p className="t-caption mt-2 text-fg-3">{o.unreviewed} finding{o.unreviewed > 1 ? 's' : ''} not yet reviewed.</p>}
      </section>

        </div>
      </details>

      <details className="rounded-lg border border-line px-3 py-2" aria-label="Raw analysis data">
        <summary className="t-small cursor-pointer text-fg-2">Raw analysis data</summary>
        <div className="mt-3 space-y-4">
          <div className="grid grid-cols-5 gap-2"><Stat label="Sources" value={o.stats.sources} /><Stat label="Tracked" value={o.stats.tracked} /><Stat label="Key events" value={o.stats.significant} /><Stat label="Conflicts" value={o.stats.contradictions} /><Stat label="Gaps" value={o.stats.gaps} /></div>
          <p className="t-caption text-fg-3">Analysis: <span className="text-fg-2">{ANALYSIS_LABEL[a.detection]}</span>{a.models.length > 0 && <span className="mono"> ({a.models.join(', ')})</span>}{a.otherRuns.length > 0 && <> · earlier run{a.otherRuns.length > 1 ? 's' : ''} not used for tracking: <span className="mono">{a.otherRuns.join(', ')}</span></>}{a.failed > 0 && <Badge variant="danger" className="ml-2">{a.failed} job failed</Badge>}</p>
          <IncidentMoment moment={moment} events={events || []} nameOf={nameOf} motions={motions} />
          {sel && (
            <div className="rounded-lg border border-accent/40 bg-accent/5 p-3" aria-label="Selected moment">
              <h4 className="t-eyebrow text-accent-text">Selected event</h4>
              <p className="t-small mt-1 text-fg">{sel.title}</p>
              <p className="mono t-caption mt-0.5 text-fg-3">{sel.start_ms != null ? `case ${formatTime(sel.start_ms)} ${formatUncertainty(sel.uncertainty_ms)}` : 'no time on the case clock'} · {sourceName(sel)} · {classOfEvent(sel)}</p>
              <Button size="sm" variant="ghost" className="mt-1" onClick={onTechnical}>Evidence layers</Button>
            </div>
          )}
          {o.found.length > 0 && <div><h4 className="t-eyebrow mb-1 text-fg-3">All findings ranked</h4><ul className="space-y-2">{o.found.map((f) => <li key={f.id}><FindingRow f={f} onSelect={onSelectFinding} /></li>)}</ul></div>}
        </div>
      </details>
      <p className="t-caption text-fg-3">Observed = measured by a machine · Reported = witness testimony · Inferred = rule-derived possibility · Unknown = missing evidence.</p>
    </div>
  );
}
