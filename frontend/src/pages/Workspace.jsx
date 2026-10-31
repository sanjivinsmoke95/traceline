import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { Alert, Badge, Button, Checkbox, ErrorState, IconButton, LoadingState, SectionHeader, Select, Tabs, useToast } from '../ui-kit';
import { FileText, Film, Gauge, PanelLeftClose, PanelLeftOpen, Users } from 'lucide-react';
import Shell from '../layout';
import { api } from '../api';
import { useWorkspace } from '../store';
import { useCaseContext } from '../caseContext';
import { byStart, findActiveEvent, resolveActive } from '../lib/active';
import { ACTOR_COLORS, CATEGORY_STYLE } from '../fmt';
import { formatTime } from '../lib/time';
import { ANALYSIS_LABEL, buildModel, effectiveDesignation, isSignificant } from '../lib/overview';
import DesignationBadge from '../components/DesignationBadge';
import Panel from '../components/Panel';
import Player from '../components/Player';
import Timeline from '../components/Timeline';
import Trajectory from '../components/Trajectory';
import Sensor from '../components/Sensor';
import Evidence from '../components/Evidence';
import Overview from '../components/Overview';
import TechnicalPanel from '../components/TechnicalPanel';
import { FindingDetail, FindingRow } from '../components/FindingParts';
import { FrameDrawer, SensorDrawer, StatementDrawer } from '../components/EvidenceDrawers';
import { IncidentCards, IncidentGallery, IncidentOverview, IncidentVehicles } from '../components/IncidentAnalysis';
import CaseBanner from '../components/CaseBanner';
import CaseMeta from '../components/CaseMeta';
import IncidentMoment, { MomentBar } from '../components/IncidentMoment';
import { pickMoment } from '../lib/moment';
import { buildExecutive } from '../lib/executive';
import { useAutoPause, useMotions } from '../hooks';
import AccidentAlert from '../components/AccidentAlert';
import IncidentSequence from '../components/IncidentSequence';
import { buildBrief } from '../lib/brief';
import CrashCuePanel, { useCueSummaries } from '../components/CrashCue';
import InvestigationFindings from '../components/InvestigationFindings';
import { Associations, CameraFilter, EventClocks, EventTrace, HypothesisExplanation, KeyObservations, Overlaps, PhaseInspector, Scenarios, SensorPanelNote, evidenceLabel, sourceName } from '../components/Reconstruction';

const NAV_KEY = 'traceline.nav';
/** The navigation rail starts collapsed on screens narrower than 1600 px so the video stays the main element; the choice is remembered. */
const readNav = () => { try { const v = localStorage.getItem(NAV_KEY); if (v) return v === 'open'; } catch { /* storage unavailable */ } return window.innerWidth >= 1600; };

function EventTab({ event, nameOf, board }) {
  if (!event) return <p className="t-small text-fg-3">Select an event on the timeline to inspect it.</p>;
  return (
    <div className="space-y-3">
      <div>
        <p className="t-small font-medium text-fg">{event.title}</p>
        <p className="t-caption mt-1 flex flex-wrap items-center gap-2 text-fg-3"><Badge variant="neutral">{evidenceLabel(event)}</Badge>{event.start_ms != null ? `case time ${formatTime(event.start_ms)}` : 'Not placed on the case clock'}</p>
        <EventTrace event={event} nameOf={nameOf} />
      </div>
      <EventClocks event={event} />
      <PhaseInspector event={event} board={board} />
    </div>
  );
}

/** Alert policy: auto-pause can be turned off without turning detection off (the alert then appears as a notice and playback continues). */
function AutoPauseControls() {
  const policy = useWorkspace((s) => s.policy);
  return (
    <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1" aria-label="Accident alert settings">
      <Checkbox label="Pause automatically at a possible accident" checked={policy.autoPause} onChange={() => useWorkspace.getState().setPolicy({ autoPause: !policy.autoPause })} />
      <label className="t-caption flex items-center gap-2 text-fg-3">Alert on
        <Select className="h-8" value={policy.minTier} onChange={(e) => useWorkspace.getState().setPolicy({ minTier: e.target.value })} options={[{ value: 'accident', label: 'Possible accidents only' }, { value: 'attention', label: 'Attention events too' }]} />
      </label>
    </div>
  );
}

export default function Workspace() {
  const { caseId } = useParams();
  const toast = useToast();
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [tab, setTab] = useState('overview');
  const [scenarios, setScenarios] = useState([]);
  const [scenarioId, setScenarioId] = useState(null);
  const [trail, setTrail] = useState(null);
  const [incident, setIncident] = useState(null);
  const [narrative, setNarrative] = useState(null);
  const [frames, setFrames] = useState([]);
  const [board, setBoard] = useState(undefined);
  const [camera, setCamera] = useState('all');
  const [assoc, setAssoc] = useState(undefined);
  const [complete, setComplete] = useState(undefined);
  const [navOpen, setNavOpen] = useState(readNav);
  const selectedActor = useWorkspace((s) => s.selectedActorId);
  const activeId = useWorkspace((s) => s.activeEventId);
  const primaryId = useWorkspace((s) => s.primaryVideoId);
  const findingId = useWorkspace((s) => s.selectedFindingId);

  const load = useCallback(async (sid) => {
    try {
      setErr(null);
      const [c, ev, actors, sc, jobs] = await Promise.all([api.case(caseId), api.evidence(caseId), api.actors(caseId), api.scenarios(caseId), api.jobs(caseId).catch(() => ({ items: [] }))]);
      setScenarios(sc.items);
      useCaseContext.getState().select(c);
      const videosEv = ev.items.filter((e) => e.kind === 'cctv' || e.kind === 'dashcam');
      const runs = Object.fromEntries(await Promise.all(videosEv.map(async (v) => [v.id, (await api.modelRuns(v.id).catch(() => ({ items: [] }))).items])));
      const chosen = sid || sc.items.find((s) => s.is_baseline)?.id || sc.items[0]?.id;
      if (!chosen) return setData({ c, ev: ev.items, actors: actors.items, jobs: jobs.items, runs, none: true });
      const [tl, cons, crbs, gaps, finds, trajs, seek0, inc, nar] = await Promise.all([
        api.timeline(caseId, chosen), api.contradictions(caseId, chosen), api.corroborations(caseId, chosen), api.gaps(caseId, chosen),
        api.findings(caseId, chosen), api.trajectories(caseId, chosen), api.seek(caseId, 0, chosen),
        api.incidentEvents(caseId, chosen), api.incidentNarrative(caseId, chosen),
      ]);
      setScenarioId(chosen);
      setCamera('all');
      setAssoc(undefined);
      setComplete(undefined);
      api.associations(caseId, chosen).then(setAssoc).catch(() => setAssoc(null));
      api.completeness(caseId, chosen).then(setComplete).catch(() => setComplete(null));
      setIncident(inc);
      setNarrative(nar);
      useWorkspace.getState().reset(caseId, chosen);
      setTab('overview');
      setData({ c, ev: ev.items, actors: actors.items, jobs: jobs.items, runs, tl, cons: cons.items, crbs: crbs.items, gaps: gaps.items, finds: finds.items, trajs: trajs.items, seek0 });
    } catch (e) { setErr(e); }
  }, [caseId]);
  useEffect(() => { load(); }, [load]);

  const events = useMemo(() => (data?.tl ? Object.values(data.tl.lanes).flat().sort(byStart) : []), [data]);
  const shifts = useMemo(() => Object.fromEntries((data?.seek0?.targets || []).map((t) => [t.evidence_id, t.media_ms])), [data]);
  const videos = useMemo(() => (data?.ev || []).filter((e) => (e.kind === 'cctv' || e.kind === 'dashcam') && e.video?.playback_ready), [data]);
  const sensorEv = (data?.ev || []).find((e) => e.kind === 'sensor_csv');
  const range = useMemo(() => (data?.tl?.span_ms?.[0] != null ? [Math.max(0, data.tl.span_ms[0] - 500), data.tl.span_ms[1] + 500] : [0, 1000]), [data]);
  const trackActors = useMemo(() => {
    const m = {};
    data?.trajs?.forEach((t) => { if (t.actor_id && t.track_ref) m[`${t.track_ref.evidence_id}:${t.track_ref.track_id}`] = { id: t.actor_id, index: data.actors.findIndex((a) => a.id === t.actor_id) }; });
    return m;
  }, [data]);
  const actorLabels = useMemo(() => Object.fromEntries((data?.actors || []).map((a) => [a.id, a.label])), [data]);
  const model = useMemo(() => (data && !data.none ? buildModel({ events, unplaced: data.tl.unplaced, cons: data.cons, crbs: data.crbs, gaps: data.gaps, finds: data.finds, trajs: data.trajs, actors: data.actors, evidence: data.ev, modelRuns: data.runs, jobs: data.jobs, shifts }) : null), [data, events, shifts]);
  const cues = useCueSummaries(videos, data?.runs);
  const keyEvents = useMemo(() => events.filter(isSignificant), [events]);
  const moment = useMemo(() => (data && !data.none ? pickMoment(events, data.crbs) : null), [data, events]);
  const brief = useMemo(() => (data && !data.none ? buildBrief({ moment, events, gaps: data.gaps || [], nameOf: (id) => sourceName((data.ev || []).find((e) => e.id === id)), cueRan: videos.some((v) => (data.runs?.[v.id] || []).some((r) => r.stage === 'accident_cue' && r.status === 'completed')) }) : null), [data, moment, events, videos]);
  const placedEvents = useMemo(() => events.filter((e) => e.start_ms != null), [events]);
  useAutoPause(placedEvents, `${scenarioId}:${primaryId}`);
  const alertState = useWorkspace((st) => st.alert);
  const alertEvent = alertState ? model?.events[alertState.eventId] : null;
  const alertCard = useMemo(() => (alertEvent && board ? board.strips.flatMap((st) => st.items).find((i) => i.event_id === alertEvent.id) : null), [alertEvent, board]);
  const motions = useMotions(moment, data?.trajs || []);
  const executive = useMemo(() => (model ? buildExecutive({ model, moment, motions: motions || [], events, evidence: data.ev, actors: data.actors, trajs: data.trajs, gaps: data.gaps, nameOf: (id) => sourceName((data.ev || []).find((e) => e.id === id)) }) : null), [model, moment, motions, events, data]);
  useEffect(() => { useWorkspace.getState().setIncident(moment ? { ms: moment.ms, trackKeys: moment.trackKeys } : null); }, [moment]);
  const allVideos = useMemo(() => (data?.ev || []).filter((e) => e.kind === 'cctv' || e.kind === 'dashcam' || e.kind === 'other'), [data]);
  const nameOf = useCallback((id) => sourceName((data?.ev || []).find((e) => e.id === id)), [data]);
  const shown = useMemo(() => {
    const allOverlaps = data?.tl?.overlaps || [];
    if (camera === 'all' || !data?.tl) return { tl: data?.tl, keyEvents, gaps: data?.gaps || [], overlaps: allOverlaps, items: incident?.items || [], unplaced: incident?.unplaced || [] };
    const only = (e) => e.source_id === camera;
    return {
      tl: { ...data.tl, lanes: Object.fromEntries(Object.entries(data.tl.lanes).map(([k, v]) => [k, v.filter(only)])) },
      keyEvents: keyEvents.filter(only),
      overlaps: allOverlaps.filter((o) => o.evidence_ids.includes(camera)),
      gaps: (data.gaps || []).filter((g) => g.scope?.evidence_id === camera),
      items: (incident?.items || []).filter((i) => i.evidence_id === camera),
      unplaced: (incident?.unplaced || []).filter((i) => i.evidence_id === camera),
    };
  }, [camera, data, keyEvents, incident]);

  useEffect(() => {
    if (!events.length) return undefined;
    return useWorkspace.subscribe((s, p) => {
      if (s.currentMs === p.currentMs) return;
      const id = resolveActive({ computed: findActiveEvent(events, s.currentMs), pinnedId: s.pinnedId, pinnedAt: s.pinnedAt, currentMs: s.currentMs });
      if (id !== s.activeEventId) s.setActive(id);
    });
  }, [events]);

  const pick = useCallback((e) => {
    const st = useWorkspace.getState();
    st.setPlaying(false);
    if (e.start_ms != null) { st.requestSeek(e.start_ms); st.selectEvent(e.id, e.start_ms); } else st.selectEvent(e.id, null);
    if (e.actor_id) st.selectActor(e.actor_id);
  }, []);
  const pickById = useCallback((id) => { const e = model?.events[id]; if (e) pick(e); else useWorkspace.getState().selectEvent(id, null); }, [model, pick]);

  /** Selecting a finding: seek to its interval, highlight its events and interval on the timeline, select its vehicle, show it in the inspector. */
  const selectFinding = useCallback((f) => {
    if (!f) return;
    const st = useWorkspace.getState();
    st.setPlaying(false);
    const ids = [...f.supporting, ...f.contradicting].map((r) => r.eventId);
    st.setFinding(f.id, ids, f.interval ? { start: f.interval.start, end: f.interval.end } : null);
    if (f.interval) { st.requestSeek(f.interval.at); const first = ids.find((i) => model.events[i]?.start_ms != null); if (first) st.selectEvent(first, f.interval.at); }
    if (f.actorIds?.[0]) st.selectActor(f.actorIds[0]);
    setTab('findings');
  }, [model]);
  const clearFinding = () => useWorkspace.getState().setFinding(null, [], null);
  const review = async (id, status) => {
    try { await api.review(id, status); toast({ title: status === 'reviewed' ? 'Marked as reviewed' : status === 'disputed' ? 'Marked as disputed' : 'Review reset', type: 'success' }); const f = await api.findings(caseId, scenarioId); setData((d) => ({ ...d, finds: f.items })); }
    catch (e) { toast({ title: 'Could not save the review', description: e.message, type: 'error' }); }
  };
  const stepEvent = useCallback((dir) => {
    const i = keyEvents.findIndex((e) => e.id === useWorkspace.getState().activeEventId);
    const n = keyEvents[Math.min(Math.max((i < 0 ? (dir > 0 ? -1 : keyEvents.length) : i) + dir, 0), keyEvents.length - 1)];
    if (n) pick(n);
  }, [keyEvents, pick]);
  useEffect(() => {
    const k = (e) => { if (/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return; if (e.key === ']') stepEvent(1); if (e.key === '[') stepEvent(-1); };
    window.addEventListener('keydown', k);
    return () => window.removeEventListener('keydown', k);
  }, [stepEvent]);
  const addAssociation = useCallback(async (body) => {
    await api.addAssociation(caseId, { scenario_id: scenarioId, ...body });
    setAssoc(await api.associations(caseId, scenarioId));
    api.completeness(caseId, scenarioId).then(setComplete).catch(() => {});
  }, [caseId, scenarioId]);
  const seekToFrame = useCallback((ref) => {
    const st = useWorkspace.getState();
    st.setPlaying(false);
    const shift = shifts[ref.evidence_id];
    if (shift == null) { st.setPrimaryVideo(ref.evidence_id); st.requestSeek(ref.time_ms); } else st.requestSeek(ref.time_ms - shift);
  }, [shifts]);
  const seekTo = useCallback((ms) => { const st = useWorkspace.getState(); st.setPlaying(false); st.requestSeek(ms); }, []);
  useEffect(() => {
    setBoard(undefined);
    if (!scenarioId) return undefined;
    let off = false;
    api.storyboard(caseId, scenarioId).then((b) => { if (!off) setBoard(b); }).catch(() => { if (!off) setBoard(null); });
    return () => { off = true; };
  }, [caseId, scenarioId]);
  useEffect(() => {
    const chosen = incident?.items?.find((item) => item.id === activeId) || incident?.unplaced?.find((item) => item.id === activeId) || incident?.items?.[0];
    if (!scenarioId || !chosen) { setFrames([]); return undefined; }
    let cancel = false;
    api.incidentFrames(caseId, scenarioId, chosen.id).then((result) => { if (!cancel) setFrames(result.items || []); }).catch(() => { if (!cancel) setFrames([]); });
    return () => { cancel = true; };
  }, [activeId, incident, scenarioId, caseId]);
  const toggleNav = () => setNavOpen((o) => { try { localStorage.setItem(NAV_KEY, o ? 'closed' : 'open'); } catch { /* storage unavailable */ } return !o; });
  const openTrail = (kind, ev) => setTrail({ kind, ev });

  const crumbs = [{ label: data?.c?.title || 'Workspace' }];
  if (err) return <Shell crumbs={crumbs}><ErrorState title={err.code === 'CASE_NOT_FOUND' ? 'This case does not exist' : 'Could not load the case'} description={err.message} onRetry={() => load()} /></Shell>;
  if (!data) return <Shell crumbs={crumbs} wide><LoadingState rows={6} label="Loading workspace" /></Shell>;
  const { c } = data;
  const eff = effectiveDesignation(c.designation, data.ev);
  const trailEvidence = trail && data.ev.find((e) => e.id === trail.ev.source_id);
  const selected = model && findingId ? model.byId[findingId] : null;
  const reviewed = (data.finds || []).filter((f) => f.review_status === 'reviewed').length;
  const analysis = model?.overview.analysis;

  const NavRail = data.none ? null : (
    <Panel as="nav" className="h-fit p-3" aria-label="Case navigation">
      <div className="mb-2 flex items-center justify-between"><h2 className="t-eyebrow text-fg-3">Investigation</h2><IconButton icon={PanelLeftClose} label="Collapse navigation" size="sm" onClick={toggleNav} /></div>
      <dl className="t-caption space-y-1 rounded-lg bg-surface-2/50 p-2.5 text-fg-3">
        <div className="flex justify-between gap-2"><dt>Detections</dt><dd className="text-right text-fg-2">{analysis ? ANALYSIS_LABEL[analysis.detection].replace(' run on the uploaded video', '').replace('Boxes from ', '') : 'n/a'}</dd></div>
        <div className="flex justify-between gap-2"><dt>Processing</dt><dd className="text-fg-2">{analysis?.running ? 'running' : analysis?.failed ? `${analysis.failed} failed` : 'complete'}</dd></div>
        <div className="flex justify-between gap-2"><dt>Reviewed</dt><dd className="text-fg-2">{reviewed} / {data.finds.length}</dd></div>
        <div className="flex justify-between gap-2"><dt>Clock</dt><dd className="text-fg-2">{data.seek0.unsynchronized.length ? `${data.seek0.unsynchronized.length} source(s) unrelated` : 'all related'}</dd></div>
      </dl>
      <h2 className="t-eyebrow mb-1.5 mt-4 text-fg-3">Sources</h2>
      <ul className="space-y-1">
        {data.ev.filter((e) => ['cctv', 'dashcam', 'sensor_csv'].includes(e.kind)).map((e) => {
          const isVideo = e.kind !== 'sensor_csv';
          return (
            <li key={e.id}>
              <button type="button" disabled={!isVideo || !e.video?.playback_ready} aria-pressed={primaryId === e.id} onClick={() => useWorkspace.getState().setPrimaryVideo(e.id)}
                      className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left transition hover:bg-surface-2 aria-pressed:bg-accent/10 disabled:cursor-default disabled:hover:bg-transparent">
                {isVideo ? <Film className="h-4 w-4 shrink-0 text-fg-3" aria-hidden /> : <Gauge className="h-4 w-4 shrink-0 text-fg-3" aria-hidden />}
                <span className="min-w-0 flex-1"><span className="t-small block truncate text-fg">{e.original_filename}</span><span className="t-caption text-fg-3">{e.kind}{shifts[e.id] != null ? '' : ' · not synchronized'}</span></span>
              </button>
            </li>
          );
        })}
      </ul>
      <h2 className="t-eyebrow mb-1.5 mt-4 flex items-center gap-1.5 text-fg-3"><Users className="h-3.5 w-3.5" aria-hidden />Vehicles and people</h2>
      {data.actors.length === 0 ? <p className="t-caption text-fg-3">No identities confirmed.</p> : (
        <ul className="space-y-0.5">{data.actors.map((a, i) => (
          <li key={a.id}><button type="button" aria-pressed={selectedActor === a.id} onClick={() => useWorkspace.getState().selectActor(selectedActor === a.id ? null : a.id)} className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left transition hover:bg-surface-2 aria-pressed:bg-accent/10">
            <i className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: ACTOR_COLORS[i % ACTOR_COLORS.length] }} aria-hidden /><span className="t-small flex-1 truncate">{a.label}</span></button></li>
        ))}</ul>
      )}
      <h2 className="t-eyebrow mb-1.5 mt-4 text-fg-3">Key events</h2>
      <ul className="max-h-60 space-y-0.5 overflow-auto pr-1" aria-label="Key events">
        {keyEvents.map((e) => (
          <li key={e.id}><button type="button" aria-pressed={activeId === e.id} onClick={() => { pick(e); setTab('overview'); }} className="flex w-full items-start gap-2 rounded-lg px-2 py-1 text-left transition hover:bg-surface-2 aria-pressed:bg-accent/10">
            <i className="mt-1.5 h-2 w-2 shrink-0 rounded-sm" style={{ background: CATEGORY_STYLE[e.category]?.color }} aria-hidden /><span className="min-w-0"><span className="mono t-caption block text-fg-3">{formatTime(e.start_ms)}</span><span className="t-caption block text-fg-2">{e.title}</span></span></button></li>
        ))}
        {keyEvents.length === 0 && <li className="t-caption text-fg-3">No key events.</li>}
      </ul>
    </Panel>
  );

  return (
    <Shell crumbs={crumbs} wide actions={<Button as={Link} to={`/reports/${caseId}`} size="sm" variant="outline" icon={FileText}>Report</Button>}>
      <div className="mb-3 flex flex-wrap items-center gap-3">
        {!navOpen && <IconButton icon={PanelLeftOpen} label="Show navigation" variant="outline" size="sm" onClick={toggleNav} />}
        <h1 className="text-xl font-semibold">{c.title}</h1>
        <DesignationBadge designation={eff.value} />
        <span className="mono t-caption text-fg-3">{c.code}</span>
        {scenarios.length > 1 && (
          <label className="t-caption ml-auto flex items-center gap-2 text-fg-3">Scenario
            <Select className="h-9" value={scenarioId || ''} onChange={(e) => load(e.target.value)} options={scenarios.map((s) => ({ value: s.id, label: `${s.name}${s.is_baseline ? ' (baseline)' : ''}` }))} />
          </label>
        )}
      </div>
      <CaseBanner designation={c.designation} warning={eff.warning} />
      <div className="mb-3"><CaseMeta c={c} /></div>

      {data.none ? (
        <ErrorState title="No reconstruction yet" description="The pipeline has not produced a reconstruction for this case. Open the dashboard to check processing jobs." />
      ) : (
        <>
          <div className={`grid grid-cols-[minmax(0,1fr)] gap-4 [@media(max-height:1100px)]:h-[calc(100dvh-9.5rem)] [@media(max-height:1100px)]:min-h-0 [@media(max-height:1100px)]:overflow-hidden ${navOpen ? 'xl:grid-cols-[250px_minmax(0,1fr)_440px]' : 'xl:grid-cols-[minmax(0,1fr)_440px]'}`}>
            <div className="order-1 min-w-0 space-y-4 xl:order-2 [@media(max-height:1100px)]:flex [@media(max-height:1100px)]:h-full [@media(max-height:1100px)]:min-h-0 [@media(max-height:1100px)]:flex-col [@media(max-height:1100px)]:gap-3 [@media(max-height:1100px)]:space-y-0 [@media(max-height:1100px)]:overflow-hidden">
              <Panel as="section" className="p-3 [@media(max-height:1100px)]:max-h-[55%] [@media(max-height:1100px)]:shrink-0 [@media(max-height:1100px)]:overflow-y-auto" aria-label="Evidence video">
                <AccidentAlert alert={alertState} event={alertEvent} events={events} cameraName={alertEvent ? nameOf(alertEvent.source_id) : null} card={alertCard} caseId={caseId}
                               onOpenFrame={(e) => openTrail('frame', e)} onDetails={(e) => { pick(e); setTab('event'); }} onEvidence={(e) => { pick(e); setTab('evidence'); }}
                               onResume={() => { const st = useWorkspace.getState(); st.setAlert(null); st.setPlaying(true); }} onDismiss={() => useWorkspace.getState().setAlert(null)} onSeekFrame={seekToFrame} />
                <AutoPauseControls />
                <MomentBar moment={moment} />
                <CameraFilter videos={allVideos} value={camera} onChange={(id) => { setCamera(id); const v = videos.find((x) => x.id === id); if (v) useWorkspace.getState().setPrimaryVideo(id); }} />
                <Player cues={cues} videos={videos} shifts={shifts} trackActors={trackActors} actorLabels={actorLabels} range={range} events={keyEvents} onSelectEvent={(e) => { pick(e); setTab('overview'); }} />
                <p className="t-caption mt-2 text-fg-3" title="Percentages on boxes are the detector's score for that box, not the chance of an event. Nothing is drawn between sampled frames.">Stored detections on sampled frames ({analysis ? ANALYSIS_LABEL[analysis.detection].toLowerCase() : ''}). Hover for what box labels mean.</p>
                <div className="mt-3"><CrashCuePanel videos={videos} cues={cues} onSeek={(evId, ms) => seekToFrame({ evidence_id: evId, time_ms: ms })} /></div>
                <div className="mt-3">
                  <IncidentOverview narrative={narrative} provenance={incident?.provenance} />
                  <HypothesisExplanation data={narrative} nameOf={nameOf} />
                  <IncidentVehicles actors={data.actors} items={[...shown.items, ...shown.unplaced]} />
                  <KeyObservations events={shown.keyEvents} activeId={activeId} onSelect={(e) => { pick(e); setTab('overview'); }} nameOf={nameOf} />
                  <IncidentGallery items={frames} />
                </div>
              </Panel>
              <Panel as="aside" className="min-w-0 p-4 xl:min-h-[28rem] [@media(max-height:1100px)]:min-h-0 [@media(max-height:1100px)]:flex-1 [@media(max-height:1100px)]:overflow-y-auto" aria-label="Inspector">
                <div className="-mx-1 overflow-x-auto px-1"><Tabs value={tab} onChange={setTab} label="Inspector" items={[{ id: 'overview', label: 'Overview' }, { id: 'event', label: 'Event' }, { id: 'findings', label: 'Findings', count: model.findings.length }, { id: 'evidence', label: 'Evidence' }, { id: 'technical', label: 'Technical' }]} /></div>
                <div className="mt-4">
                  {tab === 'overview' && <Overview brief={brief} model={model} executive={executive} moment={moment} motions={motions} events={events} designation={eff.value} activeId={activeId} nameOf={nameOf} onSelectFinding={selectFinding} onSeeAll={() => setTab('findings')} onPick={pick} onOpenTrail={openTrail} onTechnical={() => setTab('technical')} />}
                  {tab === 'event' && <EventTab event={model.events[activeId]} nameOf={nameOf} board={board} />}
                  {tab === 'findings' && (selected
                    ? <FindingDetail f={selected} events={model.events} onPick={pick} onOpenTrail={openTrail} onReview={review} onBack={clearFinding} />
                    : model.findings.length === 0 ? <p className="t-small text-fg-3">No findings were produced for this reconstruction.</p>
                      : <ul className="space-y-2">{model.findings.map((f) => <li key={f.id}><FindingRow f={f} selected={false} onSelect={selectFinding} /></li>)}</ul>)}
                  {tab === 'evidence' && (
                    <div className="space-y-4">
                      {selected && <section><h4 className="t-eyebrow mb-1.5 text-fg-3">For the selected finding</h4><FindingDetail f={{ ...selected, summary: selected.title, uncertainty: null, action: null }} events={model.events} onPick={pick} onOpenTrail={openTrail} /></section>}
                      <Evidence caseId={caseId} evidence={data.ev} />
                    </div>
                  )}
                  {tab === 'technical' && <TechnicalPanel model={model} events={events} videos={videos} modelRuns={data.runs} shifts={shifts} unsynchronized={data.seek0.unsynchronized} gaps={data.gaps} activeId={activeId} onPick={pick} onSelectEvent={pickById} onOpenTrail={openTrail} />}
                </div>
              </Panel>
            </div>

            <Panel as="section" className="order-2 min-w-0 p-3 xl:order-3 xl:sticky xl:top-20 xl:max-h-[calc(100dvh-6.5rem)] xl:self-start xl:overflow-y-auto [@media(max-height:1100px)]:max-h-full [@media(max-height:1100px)]:overflow-y-auto" aria-label="Timeline">
              <SectionHeader title="Timeline" description={`${shown.items.length} incident events · ${shown.keyEvents.length} key events of ${events.length} placed · times on the case clock`} className="!mb-3" />
              <IncidentSequence events={shown.keyEvents} unplaced={data.tl.unplaced || []} board={board} activeId={activeId} nameOf={nameOf} caseId={caseId} onSelect={pick} onSeekFrame={seekToFrame} onOpenFrame={(e) => openTrail('frame', e)} />
              <Timeline tl={shown.tl} gaps={shown.gaps} overlaps={shown.overlaps} onSelect={(e) => { pick(e); if (tab !== 'technical' && tab !== 'findings') setTab('overview'); }} onSeek={seekTo} />
            </Panel>


            {navOpen && <div className="order-3 xl:order-1 [@media(max-height:1100px)]:max-h-full [@media(max-height:1100px)]:overflow-y-auto">{NavRail}</div>}
          </div>

          <InvestigationFindings caseId={caseId} cons={data.cons} crbs={data.crbs} gaps={shown.gaps} finds={data.finds} events={events} nameOf={nameOf} onReview={review} onSeek={seekTo} onPickEvent={pickById}
                              onSeekMedia={(evId, ms) => { window.scrollTo({ top: 0, behavior: 'smooth' }); seekToFrame({ evidence_id: evId, time_ms: ms }); }} />
          <Overlaps overlaps={shown.overlaps} nameOf={nameOf} onSeek={seekTo} />
          <Scenarios scenarios={scenarios} scenarioId={scenarioId} onSelect={(id) => load(id)} hypotheses={data.tl.hypotheses || []} completeness={complete} onPickEvent={pickById} />
          <details className="mt-4 rounded-lg border border-line p-3">
            <summary className="t-small cursor-pointer text-fg">Technical details</summary>
            <div className="mt-3 space-y-5">
              <section aria-label="Cross-camera association">
                <h3 className="t-eyebrow mb-2 text-fg-3">Cross-camera association</h3>
                <Associations state={assoc} nameOf={nameOf} videos={allVideos.filter((v) => v.kind !== 'other')} onAdd={addAssociation} />
              </section>
              <div className="grid gap-4 lg:grid-cols-2">
                <div>
                  <h3 className="t-eyebrow mb-2 text-fg-3">Trajectories</h3>
                  <Trajectory trajs={data.trajs} actors={data.actors} selected={selectedActor} onSelect={(id) => useWorkspace.getState().selectActor(id)} />
                </div>
                <div aria-label="Sensor panel">
                  <h3 className="t-eyebrow mb-2 text-fg-3">Sensor telemetry</h3>
                  {sensorEv ? <Sensor evidence={sensorEv} shift={shifts[sensorEv.id]} label={sensorEv.original_filename} events={(data.tl.lanes.sensor || []).filter((e) => ['hard_braking', 'collision_flag'].includes(e.event_type))} onSelect={pick} />
                    : <SensorPanelNote hasSensor={false} />}
                </div>
              </div>
            </div>
          </details>
        </>
      )}
      <p className="t-caption mt-6 text-fg-3">TraceLine supports an investigation. It does not determine fault or legal responsibility.</p>

      <FrameDrawer open={trail?.kind === 'frame'} onClose={() => setTrail(null)} evidence={trailEvidence} event={trail?.ev} />
      <StatementDrawer open={trail?.kind === 'statement'} onClose={() => setTrail(null)} event={trail?.ev} />
      <SensorDrawer open={trail?.kind === 'sensor'} onClose={() => setTrail(null)} evidence={trailEvidence} event={trail?.ev} />
    </Shell>
  );
}
