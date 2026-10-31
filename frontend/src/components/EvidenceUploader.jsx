import FilePicker from './FilePicker';
import { useState } from 'react';
import { Alert, Badge, Button, Card, Field, Input, Segmented, Textarea } from '../ui-kit';
import { api } from '../api';
import SourceMetaFields from './SourceMetaFields';
import { EMPTY_META, toClockBody, toSourceBody, validateMeta } from '../lib/sourceMeta';

const guessKind = (name) => (/dash/i.test(name) ? 'dashcam' : 'cctv');

/** Multi-file video upload with per-source metadata, plus an optional sensor CSV and witness statement.
 *  Every file is declared real or synthetic from the case designation, never defaulted. A real or staged case never sends synthetic=true. */
export default function EvidenceUploader({ caseId, designation, onDone, submitLabel = 'Upload evidence', simple = false }) {
  const synthetic = designation === 'synthetic';
  const mixed = designation === 'mixed';
  const [files, setFiles] = useState([]);
  const [sensor, setSensor] = useState(null);
  const [statement, setStatement] = useState('');
  const [errors, setErrors] = useState({});
  const [log, setLog] = useState([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  const choose = (list) => {
    const picked = [...list];
    const seen = new Set();
    const next = picked.filter((f) => { const k = `${f.name}:${f.size}`; if (seen.has(k)) return false; seen.add(k); return true; });
    setFiles((old) => next.map((file) => old.find((o) => o.key === `${file.name}:${file.size}`) || { key: `${file.name}:${file.size}`, file, meta: { ...EMPTY_META, camera_kind: guessKind(file.name) } }));
    setErrors({});
  };
  const setMeta = (key) => (meta) => setFiles((l) => l.map((f) => (f.key === key ? { ...f, meta } : f)));
  const say = (m) => setLog((l) => [...l, m]);

  const submit = async (e) => {
    e.preventDefault();
    const all = {};
    files.forEach((f) => { const x = validateMeta(f.meta); if (Object.keys(x).length) all[f.key] = x; });
    setErrors(all);
    if (Object.keys(all).length) return;
    setBusy(true); setErr(null); setLog([]);
    const uploaded = [];
    try {
      for (const f of files) {
        say(`Uploading ${f.file.name}…`);
        const r = await api.upload(caseId, f.file, f.meta.camera_kind, synthetic);
        const ev = r.evidence;
        uploaded.push({ evidence: ev, duplicate: !!r.duplicate });
        if (r.duplicate) { say(`${f.file.name} is already in this case. It was not added again and its stored details were left as they are.`); continue; }
        await api.patchSource(ev.id, toSourceBody(f.meta));
        const body = toClockBody(f.meta);
        if (body) { await api.setClock(ev.id, body); say(`Source details and clock saved for ${f.file.name}`); }
      }
      if (sensor) { say('Uploading sensor file…'); const r = await api.upload(caseId, sensor, 'sensor_csv', synthetic); uploaded.push({ evidence: r.evidence, duplicate: !!r.duplicate }); }
      if (statement.trim()) {
        const w = await api.addWitness(caseId, 'Witness 1');
        await api.addStatement(w.id, statement, synthetic);
        say('Statement stored');
      }
      onDone?.({ uploaded });
    } catch (x) {
      setErr(x.message);
      setBusy(false);
    }
  };

  if (mixed) return <Alert variant="warning" title="Mixed cases cannot take new uploads here">Declare each file's data type through the API. Real and synthetic evidence belong in separate cases.</Alert>;

  return (
    <form onSubmit={submit} className="space-y-4" aria-label="Upload evidence">
      <Field label="Videos" hint={simple ? 'Add one or more videos, then say whether each is a dashcam or a CCTV view.' : 'Several files can be added. Dashcam = moving camera, CCTV = fixed camera. Each file keeps its own clock until you state how they relate.'}>
        <FilePicker accept="video/*" multiple disabled={busy} label="Choose video files" hint="MP4, MOV or similar" files={files.map((f) => f.file)} onChange={choose} />
      </Field>
      {files.map((f, i) => (
        <Card key={f.key} padded={false} className="space-y-3 p-4" aria-label={`Source ${f.file.name}`}>
          <div className="flex items-center gap-2"><span className="t-small min-w-0 flex-1 truncate font-medium">{f.file.name}</span><Badge variant="neutral">{(f.file.size / 1e6).toFixed(1)} MB</Badge></div>
          {simple ? (
            <Field label="Camera view">
              <Segmented label={`Camera view for ${f.file.name}`} value={f.meta.camera_kind} onChange={(v) => setMeta(f.key)({ ...f.meta, camera_kind: v })}
                         items={[{ id: 'dashcam', label: 'Dashcam view' }, { id: 'cctv', label: 'CCTV view' }]} />
            </Field>
          ) : <SourceMetaFields value={f.meta} onChange={setMeta(f.key)} errors={errors[f.key]} disabled={busy} idPrefix={`up-${i}`} filename={f.file.name} />}
        </Card>
      ))}
      {!simple && <Field label="Sensor CSV" optional hint="After upload, map its columns and units on the Evidence page. Until then no sensor reading is used."><FilePicker accept=".csv" disabled={busy} label="Choose CSV file" hint="Optional" files={sensor ? [sensor] : []} onChange={(fs) => setSensor(fs[0] || null)} /></Field>}
      <Field label="Witness statement" optional><Textarea rows={3} value={statement} onChange={(e) => setStatement(e.target.value)} disabled={busy} /></Field>
      <div className="flex flex-wrap items-center gap-3">
        <Button type="submit" variant="primary" loading={busy} disabled={!files.length && !sensor && !statement.trim()}>{busy ? 'Uploading…' : submitLabel}</Button>
        {synthetic && <Badge variant="warning">will be labelled synthetic</Badge>}
      </div>
      {log.length > 0 && <ul className="t-caption space-y-0.5 text-fg-3" role="status">{log.map((l, i) => <li key={i}>{l}</li>)}</ul>}
      {err && <Alert variant="danger" title="Upload stopped">{err}</Alert>}
    </form>
  );
}
