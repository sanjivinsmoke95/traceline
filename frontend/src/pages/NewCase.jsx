import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import Shell from '../layout';
import { Alert, Badge, Button, Card, Field, Input, LoadingState } from '../ui-kit';
import { Check } from 'lucide-react';
import { api } from '../api';
import { useCaseContext } from '../caseContext';
import CaseBanner from '../components/CaseBanner';
import EvidenceUploader from '../components/EvidenceUploader';
import JobStages from '../components/JobStages';
import { ACTIVE, STAGE_LABEL } from '../lib/jobs';

const STEPS = ['Investigation', 'Evidence', 'Processing'];

function Stepper({ step }) {
  return (
    <ol className="mb-6 flex flex-wrap items-center gap-2" aria-label="Guided flow">
      {STEPS.map((s, i) => (
        <li key={s} aria-current={i === step ? 'step' : undefined} className={`flex items-center gap-2 rounded-full border px-3 py-1 text-[0.8125rem] ${i === step ? 'border-accent bg-accent/10 text-accent-text' : 'border-line text-fg-3'}`}>
          {i < step ? <Check className="h-3.5 w-3.5" aria-hidden /> : <span className="mono">{i + 1}</span>}{s}
        </li>
      ))}
    </ol>
  );
}

/** Step 1. Creates the case from a title. Incident time, location, time zone and reference are not asked for here. */
function Details({ initial, onCreated }) {
  const [title, setTitle] = useState(initial.title);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const submit = async (e) => {
    e.preventDefault(); setBusy(true); setErr(null);
    try {
      const c = await api.createCase({ title: title.trim(), designation: 'real' });
      onCreated(c);
    } catch (x) { setErr(x.message); setBusy(false); }
  };
  return (
    <Card as="form" onSubmit={submit} className="space-y-4" aria-label="Create investigation">
      <Field label="Case title"><Input required autoFocus value={title} onChange={(e) => setTitle(e.target.value)} disabled={busy} placeholder="e.g. Junction collision, 4th Ave" /></Field>
      <Button type="submit" variant="primary" loading={busy} disabled={!title.trim()}>Create investigation</Button>
      {err && <Alert variant="danger" title="Could not create the case">{err}</Alert>}
    </Card>
  );
}

/** Step 3. Polls the real job list. Stages appear only when the backend has queued them. */
function Processing({ c }) {
  const nav = useNavigate();
  const [jobs, setJobs] = useState(null);
  const [evidence, setEvidence] = useState([]);
  const [phase, setPhase] = useState('running');
  const [err, setErr] = useState(null);
  const built = useRef(false);

  useEffect(() => {
    let stop = false;
    let timer;
    const tick = async () => {
      try {
        const [j, ev] = await Promise.all([api.jobs(c.id), api.evidence(c.id)]);
        if (stop) return;
        setJobs(j.items); setEvidence(ev.items);
        const failed = j.items.find((x) => x.status === 'failed');
        const active = j.items.some((x) => ACTIVE.includes(x.status));
        const hasVideo = ev.items.some((e) => e.kind === 'cctv' || e.kind === 'dashcam');
        const tracked = j.items.some((x) => x.stage === 'track' && x.status === 'completed');
        if (failed) { setErr(`${STAGE_LABEL[failed.stage] || failed.stage} failed: ${failed.error || 'see job log'}`); setPhase('failed'); return; }
        if (!active && j.items.length && (tracked || !hasVideo)) { setPhase('ready'); return; }
      } catch (x) { if (!stop) { setErr(x.message); setPhase('failed'); } return; }
      timer = setTimeout(tick, 1500);
    };
    tick();
    return () => { stop = true; clearTimeout(timer); };
  }, [c.id]);

  const build = useCallback(async () => {
    if (built.current) return;
    built.current = true;
    setPhase('building');
    try { await api.buildScenario(c.id, { name: 'Baseline' }); nav(`/reconstruction/${c.id}`); } catch (x) { built.current = false; setErr(x.message); setPhase('failed'); }
  }, [c.id, nav]);
  useEffect(() => { if (phase === 'ready') build(); }, [phase, build]);

  const retry = async (job) => { setErr(null); setPhase('running'); try { await api.retryJob(job.id); } catch (x) { setErr(x.message); setPhase('failed'); } };
  const failedJob = (jobs || []).find((j) => j.status === 'failed');

  if (!jobs) return <LoadingState rows={3} label="Reading processing jobs" />;
  return (
    <div className="space-y-4" aria-label="Processing">
      <p className="t-small text-fg-2" role="status">
        {phase === 'building' ? 'Building the reconstruction. The backend reports no progress for this one request.' : phase === 'failed' ? 'Processing stopped.' : 'Processing runs on this machine. Each stage appears when the backend queues it.'}
      </p>
      {evidence.length === 0 ? <p className="t-small text-fg-3">No evidence has been uploaded to this case.</p> : evidence.map((e) => (
        <Card key={e.id} className="space-y-3">
          <div className="flex items-center gap-2"><span className="t-small min-w-0 flex-1 truncate font-medium">{e.original_filename}</span><Badge variant="neutral">{e.kind}</Badge></div>
          <JobStages jobs={jobs} evidenceId={e.id} empty="No stage has been queued for this file yet." />
        </Card>
      ))}
      {jobs.some((j) => !j.evidence_id) && <Card><p className="t-eyebrow mb-2 text-fg-3">Case-level</p><JobStages jobs={jobs.filter((j) => !j.evidence_id)} /></Card>}
      {err && <Alert variant="danger" title="Something went wrong">{err}</Alert>}
      {phase === 'failed' && (
        <div className="flex gap-2">
          {failedJob && <Button variant="outline" onClick={() => retry(failedJob)}>Retry the failed stage</Button>}
          <Button as={Link} to={`/evidence/${c.id}`} variant="ghost">Open the evidence page</Button>
        </div>
      )}
    </div>
  );
}

/** Guided flow: create the investigation, add evidence with source details, watch real processing, open the reconstruction. `?case=` resumes a case. */
export default function NewCase() {
  const [params, setParams] = useSearchParams();
  const select = useCaseContext((s) => s.select);
  const [c, setC] = useState(null);
  const [loading, setLoading] = useState(!!params.get('case'));
  const [step, setStep] = useState(params.get('case') ? 1 : 0);
  const [err, setErr] = useState(null);
  const [dups, setDups] = useState([]);
  const initial = { title: params.get('title') || '', designation: ['real', 'staged', 'synthetic'].includes(params.get('designation')) ? params.get('designation') : 'real' };

  useEffect(() => {
    const id = params.get('case');
    if (!id) return;
    api.case(id).then((x) => { setC(x); select(x); setLoading(false); }).catch((x) => { setErr(x.message); setLoading(false); setStep(0); });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const created = (x) => { setC(x); select(x); setStep(1); setParams({ case: x.id }, { replace: true }); };

  return (
    <Shell crumbs={[{ label: 'New investigation' }]}>
      <div className="mx-auto max-w-3xl">
        <h1 className="mb-1 text-2xl font-semibold">New investigation</h1>
        <p className="t-small mb-4 text-fg-2">Describe the case, add the evidence you have, then watch it being processed. Detection runs on your machine; nothing is sent elsewhere.</p>
        <Stepper step={step} />
        {c && <CaseBanner designation={c.designation} />}
        {err && <Alert variant="danger" className="mb-4" title="Could not resume this case">{err}</Alert>}
        {loading ? <LoadingState rows={3} label="Loading the case" /> : step === 0 ? <Details initial={initial} onCreated={created} /> : step === 1 ? (
          <div className="space-y-4">
            <Card><p className="t-small"><span className="font-medium">{c.title}</span> <span className="mono t-caption text-fg-3">{c.code}</span></p></Card>
            <Card><EvidenceUploader simple caseId={c.id} designation={c.designation} submitLabel="Upload and analyse" onDone={({ uploaded }) => { setDups(uploaded.filter((u) => u.duplicate).map((u) => u.evidence.original_filename || u.evidence.id)); setStep(2); }} /></Card>
            {(c.counts?.evidence_total ?? 0) > 0 && <Button variant="outline" onClick={() => setStep(2)}>Continue to processing</Button>}
          </div>
        ) : (<>{dups.length > 0 && <Alert variant="info" className="mb-4" title="Already in this case">{dups.join(', ')} {dups.length === 1 ? 'was' : 'were'} already in this case and was not added again. Identical files are not counted as separate corroboration.</Alert>}<Processing c={c} /></>)}
      </div>
    </Shell>
  );
}
