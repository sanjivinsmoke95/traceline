"""Database-backed job queue. States: pending -> validating/processing -> completed | failed | partially_completed.

Stages: validate (every upload) -> preprocess (valid videos: frame index, sampled thumbnails, playback copy).
extract_interval adds thumbnails for an investigator-selected interval at a higher rate.
"""
import contextvars
import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone

from sqlalchemy import func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.llm import featherless
from app.models import (Case, Detection, EvidenceFile, ExtractedFrame, ModelRun, ProcessingJob, VehicleTrack, VideoMetadata, Witness,
                        WitnessClaim, WitnessStatement, SourceEvent)
from app.services import audit, detection, tracking, video_events
from app.services import witness as wit
from app.services import video as vid
from app.services.weights import UnsafeWeights, load_yolo
from app.services.validators import VALIDATOR_VERSION, validate
from app.storage import derived_dir, safe_resolve

STAGE_VALIDATE = "validate"
STAGE_PREPROCESS = "preprocess"
STAGE_INTERVAL = "extract_interval"
STAGE_DETECT = "detect"
STAGE_TRACK = "track"
STAGE_CLAIMS = "extract_claims"
MAX_ATTEMPTS = 3
ACTIVE = ("pending", "validating", "processing")
VIDEO_KINDS = ("dashcam", "cctv")


def _now() -> datetime:
    return datetime.now(timezone.utc)


# Set by the runner for the duration of one stage. Handlers report completed work units / planned work units.
# 1.0 is written only when the stage run finishes, so an in-flight value never means "done".
_progress_sink: contextvars.ContextVar = contextvars.ContextVar("traceline_job_progress", default=None)


def publish_progress(fraction: float) -> None:
    """Publish the fraction of this stage's planned work that has already finished. Not a clock."""
    cb = _progress_sink.get()
    if cb is None:
        return
    cb(min(0.99, max(0.0, float(fraction))))


def _progress_token(factory: sessionmaker[Session], job_id: str):
    def sink(fraction: float) -> None:
        try:
            with factory() as ps:
                ps.execute(update(ProcessingJob).where(
                    ProcessingJob.id == job_id, ProcessingJob.status.in_(("validating", "processing"))
                ).values(progress=round(fraction, 4)))
                ps.commit()
        except Exception:  # noqa: BLE001 - a progress write must not fail the stage
            return

    return _progress_sink.set(sink)


def params_hash(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


def preprocess_params(settings: Settings, sample_fps: float | None = None, max_frames: int | None = None) -> dict:
    return {"sample_fps": float(sample_fps or settings.sample_fps), "max_frames": int(max_frames or settings.max_frames),
            "thumb_width": settings.thumb_width, "version": vid.PREPROCESS_VERSION}


def detect_params(settings: Settings) -> dict:
    return {"detector": settings.detector_model, "imgsz": settings.detector_imgsz, "conf": settings.detector_conf,
            "accident": settings.accident_model or None, "accident_conf": settings.accident_conf, "frames": "sampled",
            "version": detection.DETECTION_VERSION}


def enqueue(session: Session, *, case_id: str, evidence_id: str | None, stage: str, input_hash: str | None, params: dict) -> ProcessingJob:
    job = ProcessingJob(case_id=case_id, evidence_id=evidence_id, stage=stage, status="pending", input_hash=input_hash,
                        params=params, params_hash=params_hash(params))
    session.add(job)
    session.flush()
    return job


def _rel(settings: Settings, p) -> str:
    return str(p.resolve().relative_to(settings.storage_path.resolve()))


def _upsert_frames(session: Session, rows: list[dict]) -> None:
    """Frames are keyed by (evidence, frame_no). A frame once sampled stays sampled; new thumbnails/metrics fill in."""
    for i in range(0, len(rows), 2000):
        stmt = pg_insert(ExtractedFrame).values(rows[i:i + 2000])
        ex = stmt.excluded
        session.execute(stmt.on_conflict_do_update(
            index_elements=["evidence_id", "frame_no"],
            set_={"sampled": ExtractedFrame.sampled | ex.sampled,
                  "thumbnail_path": func.coalesce(ex.thumbnail_path, ExtractedFrame.thumbnail_path),
                  "blur": func.coalesce(ex.blur, ExtractedFrame.blur),
                  "brightness": func.coalesce(ex.brightness, ExtractedFrame.brightness)}))


# ---------------------------------------------------------------------------------------------------- validate
def handle_validate(session: Session, settings: Settings, job: ProcessingJob) -> str | None:
    ev = session.get(EvidenceFile, job.evidence_id)
    if ev is None:
        raise LookupError(f"evidence {job.evidence_id} not found")
    path = safe_resolve(settings.storage_path, ev.stored_path)
    if not path.is_file():
        raise FileNotFoundError("stored file is missing from the evidence store")
    result = validate(ev.kind, path)
    ev.validation = result
    ev.status = "valid" if result["ok"] else "invalid"
    if ev.kind in (*VIDEO_KINDS, "other") and result["ok"]:
        m = result["metadata"]
        row = session.get(VideoMetadata, ev.id) or VideoMetadata(evidence_id=ev.id)
        row.duration_ms, row.fps_nominal = m["duration_ms"], m["fps_nominal"]
        row.fps_measured = m.get("fps_measured") or m.get("fps_average")
        row.width, row.height, row.codec, row.rotation = m["width"], m["height"], m["codec"], m["rotation"]
        row.frame_count = m.get("frame_count_declared") or m.get("packet_count")
        if m.get("creation_time_claimed"):
            row.creation_time_claimed = datetime.fromisoformat(m["creation_time_claimed"])  # a claim, not a synchronized time
        session.add(row)
        if ev.kind in VIDEO_KINDS:
            enqueue(session, case_id=ev.case_id, evidence_id=ev.id, stage=STAGE_PREPROCESS, input_hash=ev.sha256, params=preprocess_params(settings))
    audit.record(session, actor="system", action="evidence_validated", case_id=ev.case_id, target_type="evidence", target_id=ev.id,
                 details={"status": ev.status, "errors": len(result["errors"]), "warnings": len(result["warnings"])})
    job.result = {"evidence_status": ev.status, "errors": result["errors"], "warnings": result["warnings"]}
    return None


# ---------------------------------------------------------------------------------------------------- preprocess
def _video_for(session: Session, settings: Settings, job: ProcessingJob):
    ev = session.get(EvidenceFile, job.evidence_id)
    if ev is None or ev.kind not in VIDEO_KINDS:
        raise ValueError("preprocessing applies to dashcam/cctv evidence only")
    if ev.status != "valid":
        raise ValueError(f"evidence status is '{ev.status}'; only valid videos are preprocessed")
    path = safe_resolve(settings.storage_path, ev.stored_path)
    if not path.is_file():
        raise FileNotFoundError("stored file is missing from the evidence store")
    return ev, session.get(VideoMetadata, ev.id), path


def handle_preprocess(session: Session, settings: Settings, job: ProcessingJob) -> str | None:
    ev, vm, path = _video_for(session, settings, job)
    params = job.params or preprocess_params(settings)
    ph = params_hash(params)
    prior = session.execute(select(ModelRun).where(ModelRun.case_id == ev.case_id, ModelRun.stage == STAGE_PREPROCESS,
                                                   ModelRun.input_sha256 == ev.sha256, ModelRun.params_hash == ph,
                                                   ModelRun.status == "completed")).scalars().first()
    if prior is not None:  # idempotent: same bytes + same parameters = same outputs (partial runs are redone, not reused)
        job.result = {"reused_model_run": prior.id, "note": "identical input and parameters were already processed"}
        return None

    run = ModelRun(case_id=ev.case_id, stage=STAGE_PREPROCESS, model_id="ffmpeg+opencv",
                   model_version=f"ffmpeg {vid.ffmpeg_version()}; opencv {vid.cv2.__version__}", params=params, params_hash=ph,
                   input_sha256=ev.sha256, device="cpu", started_at=_now(), status="running")
    session.add(run)
    session.flush()
    out = derived_dir(settings.storage_path, ev.case_id, ev.id, run.id)

    pts = vid.probe_frame_pts(path)
    if not pts:
        raise ValueError("no frame timestamps could be read from the video")
    rel = [(t - pts[0]) * 1000 for t in pts]
    period = statistics.median([b - a for a, b in zip(rel, rel[1:])]) if len(rel) > 1 else 0.0
    idx, truncated = vid.choose_sampled(rel, params["sample_fps"], params["max_frames"])
    # Planned units: every container frame, plus the playback copy. The fraction is frames grabbed / (frames + 1).
    dec = vid.decode_selected(path, set(idx), out / "thumbs", params["thumb_width"], rel,
                              progress=lambda done, total: publish_progress(done / (total + 1)),
                              rotation=vm.rotation if vm else None)

    rows = []
    for i, t in enumerate(rel):
        f = dec["frames"].get(i)
        rows.append({"evidence_id": ev.id, "frame_no": i, "time_ms": round(t), "sampled": f is not None,
                     "thumbnail_path": _rel(settings, out / "thumbs" / f["thumb"]) if f else None,
                     "blur": f["blur"] if f else None, "brightness": f["brightness"] if f else None})
    _upsert_frames(session, rows)

    warnings, partial = [], False
    if dec["decoded_count"] != len(pts):
        warnings.append(f"decoder produced {dec['decoded_count']} frames but the container lists {len(pts)}")
        partial = True
    if period and dec["max_pts_deviation_ms"] > period / 2:
        warnings.append(f"decoder timing differs from container timestamps by up to {dec['max_pts_deviation_ms']} ms")
        partial = True
    if truncated:
        warnings.append(f"sampling stopped at max_frames={params['max_frames']}; later frames are indexed but have no thumbnails")
        partial = True
    missing = sorted(set(idx) - set(dec["frames"]))
    if missing:
        warnings.append(f"{len(missing)} sampled frame(s) could not be decoded (first: {missing[:5]})")
        partial = True

    if vm is not None:
        vm.pts_start_ms = pts[0] * 1000
    if dec["decoded_count"] >= len(pts):
        publish_progress(len(pts) / (len(pts) + 1))
    try:
        pb = vid.make_playback(path, out / "playback.mp4", pts[0] * 1000, vm.rotation if vm else None)
        if vm is not None:
            vm.playback_path = _rel(settings, out / "playback.mp4")
        if not pb["timing_preserved"]:
            warnings.append("playback copy timing does not match the original; overlays may drift")
            partial = True
    except Exception as exc:  # noqa: BLE001 - frames are still useful without a playback copy
        pb = {"error": str(exc)[:200]}
        warnings.append("playback copy could not be created; the original can still be streamed")
        partial = True

    status = "partially_completed" if partial else "completed"
    run.metrics = {"frames_indexed": len(pts), "frames_sampled": len(dec["frames"]), "decoded_count": dec["decoded_count"],
                   "pts_start_ms": round(pts[0] * 1000, 3), "median_frame_step_ms": round(period, 3),
                   "max_pts_deviation_ms": dec["max_pts_deviation_ms"], "playback": pb, "warnings": warnings}
    run.status, run.finished_at = status, _now()
    job.result = {"model_run_id": run.id, "frames_indexed": len(pts), "frames_sampled": len(dec["frames"]), "warnings": warnings, "playback": pb}
    audit.record(session, actor="system", action="video_preprocessed", case_id=ev.case_id, target_type="evidence", target_id=ev.id,
                 details={"model_run": run.id, "status": status, "frames": len(pts), "sampled": len(dec["frames"])})
    if dec["frames"] and (job.params or {}).get("chain_detect", True):
        enqueue(session, case_id=ev.case_id, evidence_id=ev.id, stage=STAGE_DETECT, input_hash=ev.sha256, params=detect_params(settings))
    return status if partial else None


def handle_interval(session: Session, settings: Settings, job: ProcessingJob) -> str | None:
    """Higher-rate thumbnails for [from_ms, to_ms]. Requires an existing frame index; never changes frame times."""
    ev, vm, path = _video_for(session, settings, job)
    p = job.params or {}
    frames = session.execute(select(ExtractedFrame).where(ExtractedFrame.evidence_id == ev.id).order_by(ExtractedFrame.frame_no)).scalars().all()
    if not frames:
        raise ValueError("video has no frame index yet; run preprocessing first")
    inside = [f for f in frames if p["from_ms"] <= f.time_ms <= p["to_ms"]]
    if p.get("sample_fps"):
        sub, _ = vid.choose_sampled([float(f.time_ms) for f in inside], p["sample_fps"], 100_000)
        wanted = {inside[i].frame_no for i in sub}
    else:
        wanted = {f.frame_no for f in inside}
    run = ModelRun(case_id=ev.case_id, stage=STAGE_INTERVAL, model_id="ffmpeg+opencv", model_version=f"opencv {vid.cv2.__version__}",
                   params=p, params_hash=params_hash(p), input_sha256=ev.sha256, device="cpu", started_at=_now(), status="running")
    session.add(run)
    session.flush()
    out = derived_dir(settings.storage_path, ev.case_id, ev.id, run.id)
    indexed = len(frames) or 1
    dec = vid.decode_selected(path, wanted, out / "thumbs", settings.thumb_width, [float(f.time_ms) for f in frames],
                              progress=lambda done, total: publish_progress(done / indexed),
                              rotation=vm.rotation if vm else None)
    rows = [{"evidence_id": ev.id, "frame_no": f.frame_no, "time_ms": f.time_ms, "sampled": True,
             "thumbnail_path": _rel(settings, out / "thumbs" / dec["frames"][f.frame_no]["thumb"]),
             "blur": dec["frames"][f.frame_no]["blur"], "brightness": dec["frames"][f.frame_no]["brightness"]}
            for f in frames if f.frame_no in dec["frames"]]
    _upsert_frames(session, rows)
    run.metrics = {"frames_requested": len(wanted), "frames_extracted": len(rows)}
    run.status, run.finished_at = "completed", _now()
    job.result = {"model_run_id": run.id, "frames_requested": len(wanted), "frames_extracted": len(rows)}
    return None if len(rows) == len(wanted) else "partially_completed"


def handle_detect(session: Session, settings: Settings, job: ProcessingJob) -> str | None:
    """Run the detector (required) and the accident cue (optional) over the sampled frames. One model run per model."""
    ev, vm, path = _video_for(session, settings, job)
    p = job.params or detect_params(settings)
    frames = session.execute(select(ExtractedFrame).where(ExtractedFrame.evidence_id == ev.id, ExtractedFrame.sampled.is_(True))
                             .order_by(ExtractedFrame.frame_no)).scalars().all()
    if not frames:
        raise ValueError("no sampled frames; run preprocessing first")
    by_no = {f.frame_no: f for f in frames}
    device = detection.pick_device(settings.detector_device)
    passes = [("detector", STAGE_DETECT, p["detector"], detection.COCO_KEEP, p["conf"])]
    if p.get("accident"):
        passes.append(("accident_cue", "accident_cue", p["accident"], detection.ACCIDENT_KEEP, p["accident_conf"]))

    import ultralytics

    result: dict = {"device": device, "frames": len(frames), "warnings": []}
    partial = False
    n_passes = len(passes)
    n_frames = len(frames)
    for pass_i, (role, stage, name, keep, conf) in enumerate(passes):
        rp = {"model": name, "imgsz": p["imgsz"], "conf": conf, "frames": "sampled", "version": detection.DETECTION_VERSION}
        ph = params_hash(rp)
        prior = session.execute(select(ModelRun).where(ModelRun.case_id == ev.case_id, ModelRun.stage == stage, ModelRun.input_sha256 == ev.sha256,
                                                       ModelRun.params_hash == ph, ModelRun.status == "completed")).scalars().first()
        if prior is not None:
            result[role] = {"reused_model_run": prior.id}
            publish_progress((pass_i + 1) / n_passes)
            continue
        try:
            model, entry = load_yolo(name, str(settings.models_path))
        except (UnsafeWeights, FileNotFoundError) as exc:
            if role == "detector":
                raise
            result[role] = {"skipped": str(exc)}
            result["warnings"].append(f"accident cue not run: {exc}")
            partial = True
            continue
        run = ModelRun(case_id=ev.case_id, stage=stage, model_id=name, model_version=f"sha256:{entry['sha256'][:12]}; ultralytics {ultralytics.__version__}",
                       params=rp, params_hash=ph, input_sha256=ev.sha256, device=device, started_at=_now(), status="running")
        session.add(run)
        session.flush()
        stats: dict = {}
        rows, by_class = [], {}

        def _inferred(done, pass_i=pass_i):
            if n_frames:
                publish_progress((pass_i + done / n_frames) / n_passes)

        for idx, boxes in detection.detect_stream(vid.iter_frames(path, set(by_no), vm.rotation if vm else None), model, imgsz=p["imgsz"], conf=conf,
                                                   device=device, keep=keep, stats=stats, progress=_inferred):
            for b in boxes:
                by_class[b.class_name] = by_class.get(b.class_name, 0) + 1
                rows.append({"frame_id": by_no[idx].id, "model_run_id": run.id, "class_name": b.class_name, "confidence": b.confidence,
                             "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "is_accident_cue": role == "accident_cue"})
        for i in range(0, len(rows), 5000):
            session.execute(insert(Detection), rows[i:i + 5000])
        if stats["frames"] != len(frames):
            result["warnings"].append(f"{role}: {stats['frames']} of {len(frames)} sampled frames could be decoded")
            partial = True
        run.metrics = stats | {"by_class": by_class}
        run.status, run.finished_at = ("partially_completed" if stats["frames"] != len(frames) else "completed"), _now()
        result[role] = {"model_run_id": run.id, "boxes": stats["boxes"], "by_class": by_class, "ms_per_frame": stats["ms_per_frame"]}
    job.result = result
    audit.record(session, actor="system", action="detection_run", case_id=ev.case_id, target_type="evidence", target_id=ev.id,
                 details={"device": device, "partial": partial})
    enqueue(session, case_id=ev.case_id, evidence_id=ev.id, stage=STAGE_TRACK, input_hash=ev.sha256, params={"version": tracking.TRACKING_VERSION})
    return "partially_completed" if partial else None


def _latest_run(session: Session, ev: EvidenceFile, stage: str) -> ModelRun | None:
    return session.execute(select(ModelRun).where(ModelRun.case_id == ev.case_id, ModelRun.input_sha256 == ev.sha256, ModelRun.stage == stage,
                                                  ModelRun.status.in_(("completed", "partially_completed")))
                           .order_by(ModelRun.finished_at.desc())).scalars().first()


def handle_track(session: Session, settings: Settings, job: ProcessingJob) -> str | None:
    """Associate the latest detector run's boxes into per-video tracks, then post-process (gaps, flags, candidates)."""
    ev, vm, path = _video_for(session, settings, job)
    det_run = _latest_run(session, ev, STAGE_DETECT)
    if det_run is None:
        raise ValueError("no finished detector run; run detection first")
    pre = _latest_run(session, ev, STAGE_PREPROCESS)
    fps = float((pre.params or {}).get("sample_fps", settings.sample_fps)) if pre else settings.sample_fps
    p = tracking.default_params(ev.kind, fps) | {k: v for k, v in (job.params or {}).items() if k != "version"}
    p["detector_run"] = det_run.id
    p["video_events"] = video_events.VIDEO_EVENTS_VERSION
    ph = params_hash(p)
    prior = session.execute(select(ModelRun).where(ModelRun.case_id == ev.case_id, ModelRun.stage == STAGE_TRACK, ModelRun.input_sha256 == ev.sha256,
                                                   ModelRun.params_hash == ph, ModelRun.status == "completed")).scalars().first()
    if prior is not None:
        job.result = {"reused_model_run": prior.id}
        return None

    sampled = session.execute(select(ExtractedFrame).where(ExtractedFrame.evidence_id == ev.id, ExtractedFrame.sampled.is_(True))
                              .order_by(ExtractedFrame.frame_no)).scalars().all()
    frames = {f.id: tracking.Frame(f.frame_no, f.time_ms) for f in sampled}
    dets = session.execute(select(Detection).where(Detection.model_run_id == det_run.id, Detection.is_accident_cue.is_(False),
                                                   Detection.confidence >= p["min_conf"], Detection.class_name.in_(p["classes"]))).scalars()
    for d in dets:
        fr = frames.get(d.frame_id)
        if fr is not None:
            fr.dets.append(tracking.Det(d.id, fr.frame_no, fr.time_ms, d.class_name, d.confidence, (d.x1, d.y1, d.x2, d.y2)))
    ordered = sorted(frames.values(), key=lambda f: f.frame_no)

    import trackers as _trk

    run = ModelRun(case_id=ev.case_id, stage=STAGE_TRACK, model_id=f"trackers:{p['tracker']}", model_version=f"trackers {getattr(_trk, '__version__', '2.6.1')}",
                   params=p, params_hash=ph, input_sha256=ev.sha256, device="cpu", started_at=_now(), status="running")
    session.add(run)
    session.flush()
    images = vid.iter_frames(path, {f.frame_no for f in ordered}, vm.rotation if vm else None) if p["tracker"] == "botsort" else None
    assign = tracking.associate(ordered, tracking.make_tracker(p), images,
                                progress=lambda done, total: publish_progress(done / total if total else 0.0))
    upright = vid.display_size(vm.width, vm.height, vm.rotation) if vm else None
    tracks, cands = tracking.build_tracks(ordered, assign, p, upright)
    for t in tracks:
        session.add(VehicleTrack(evidence_id=ev.id, model_run_id=run.id, track_id=t["track_id"], class_name=t["class"],
                                 first_frame=t["first_frame"], last_frame=t["last_frame"], duration_ms=t["duration_ms"],
                                 mean_confidence=t["mean_confidence"], points=t["points"],
                                 continuity=t["continuity"] | {"class_votes": t["class_votes"], "first_ms": t["first_ms"], "last_ms": t["last_ms"],
                                                               "continuation_candidates": [c for c in cands if t["track_id"] in (c["from_track"], c["to_track"])]}))
    # candidate events from the tracks (source media time); the accident cue is passed only as a weak note
    cue_run = _latest_run(session, ev, "accident_cue")
    cues = []
    if cue_run is not None:
        cues = [(f.time_ms, d.confidence) for d, f in session.execute(
            select(Detection, ExtractedFrame).join(ExtractedFrame, ExtractedFrame.id == Detection.frame_id)
            .where(Detection.model_run_id == cue_run.id)).all()]
    steps = [b.time_ms - a.time_ms for a, b in zip(ordered, ordered[1:])]
    period_ms = float(statistics.median(steps)) if steps else 1000.0 / fps
    vevents = video_events.generate(
        [{"track_id": t["track_id"], "class": t["class"], "points": t["points"], "continuity": t["continuity"]} for t in tracks],
        kind=ev.kind, period_ms=period_ms,
        video_duration_ms=float(vm.duration_ms) if vm and vm.duration_ms else float(ordered[-1].time_ms if ordered else 0), cues=cues)
    track_pk = {t["track_id"]: None for t in tracks}
    session.flush()
    for e in vevents:
        session.add(SourceEvent(case_id=ev.case_id, evidence_id=ev.id, stage="video_events", category=e["category"], event_type=e["event_type"],
                                t_start_ms=e["t_start_ms"], t_end_ms=e["t_end_ms"], uncertainty_ms=e["uncertainty_ms"],
                                source_ref=e["source_ref"] | {"evidence_id": ev.id}, observations=e["observations"] | {"camera_kind": ev.kind},
                                confidence={"timestamp": "unsynchronized", "tracking": "see track reliability"},
                                entity_hint=e["entity_hint"], rule_id=e["rule_id"], model_run_id=run.id))
    n_dets = sum(len(f.dets) for f in ordered)
    tracked = sum(len(t["points"]) for t in tracks)
    rel = Counter(t["continuity"]["reliability"] for t in tracks)
    run.metrics = {"tracks": len(tracks), "detections_considered": n_dets, "detections_tracked": tracked, "untracked": n_dets - tracked,
                   "backfilled": sum(t["continuity"]["backfilled_first_observation"] for t in tracks), "reliability": dict(rel),
                   "continuation_candidates": cands, "video_events": len(vevents)}
    run.status, run.finished_at = "completed", _now()
    job.result = {"model_run_id": run.id, "tracker": p["tracker"], "detector_run": det_run.id} | {k: run.metrics[k] for k in ("tracks", "detections_considered", "detections_tracked", "untracked", "backfilled", "reliability")} | {"continuation_candidates": len(cands)}
    audit.record(session, actor="system", action="tracking_run", case_id=ev.case_id, target_type="evidence", target_id=ev.id,
                 details={"tracker": p["tracker"], "tracks": len(tracks)})
    return None


class ExtractionFailed(Exception):
    pass


def maybe_enqueue_claims(session: Session, settings: Settings, case: Case, statement: WitnessStatement, explicit: bool = False) -> ProcessingJob | None:
    """Auto-queue claim extraction only when the case allows external inference and a key is configured.
    An explicit request is always queued: without a usable model the job falls back to deterministic extraction."""
    if not case.allow_external_inference or not (explicit or settings.featherless_configured):
        return None
    return enqueue(session, case_id=case.id, evidence_id=statement.evidence_id, stage=STAGE_CLAIMS,
                   input_hash=hashlib.sha256(statement.text_original.encode()).hexdigest(), params={"statement_id": statement.id})


def store_fallback_claims(session: Session, case: Case, st: WitnessStatement, reason: str) -> dict:
    """Rule-based sentence claims with exact spans of the immutable statement; nothing is interpreted, dated or linked to evidence."""
    text = st.text_original
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    rp = {"method": wit.FALLBACK_METHOD, "version": wit.FALLBACK_VERSION}
    run = ModelRun(case_id=case.id, stage=STAGE_CLAIMS, model_id=wit.FALLBACK_METHOD, model_version=wit.FALLBACK_VERSION, params=rp,
                   params_hash=params_hash(rp), input_sha256=text_hash, device="local", started_at=_now(), status="running")
    session.add(run)
    session.flush()
    stored = 0
    for a, b in wit.split_claim_spans(text):
        quote = text[a:b]
        session.add(WitnessClaim(statement_id=st.id, model_run_id=run.id, char_start=a, char_end=b, source_text=quote, subject=None, action=None,
                                 direction=None, time_description=None, certainty="reported_as_direct_observation", extraction_status="valid",
                                 normalized={"location": None, "time_exactness": "none", "hedges": [], "event_type_hint": None,
                                             "interpretation_by": wit.FALLBACK_METHOD, "extraction_method": wit.FALLBACK_METHOD,
                                             "code_derived_maneuver": None, "direction_check": "not_comparable"}))
        stored += 1
    info = {"extraction_method": wit.FALLBACK_METHOD, "llm_used": False, "fallback_reason": reason[:300], "claims_stored": stored,
            "model_run_id": run.id}
    run.metrics, run.status, run.finished_at = info, "completed", _now()
    audit.record(session, actor="system", action="claims_extracted_fallback", case_id=case.id, target_type="statement", target_id=st.id, details=info)
    return info


def handle_extract_claims(session: Session, settings: Settings, job: ProcessingJob) -> str | None:
    st = session.get(WitnessStatement, (job.params or {}).get("statement_id"))
    if st is None:
        raise LookupError("statement not found")
    if not settings.featherless_configured:
        case = session.get(Case, session.get(Witness, st.witness_id).case_id)
        job.result = store_fallback_claims(session, case, st, "FEATHERLESS_API_KEY is not configured")
        return None
    try:
        return _extract_with_llm(session, settings, job, st)
    except (featherless.LlmError, featherless.NotConfigured, ExtractionFailed) as exc:
        session.rollback()
        st = session.get(WitnessStatement, st.id)
        case = session.get(Case, session.get(Witness, st.witness_id).case_id)
        job.result = store_fallback_claims(session, case, st, f"{type(exc).__name__}: {exc}")
        return None


def _extract_with_llm(session: Session, settings: Settings, job: ProcessingJob, st: WitnessStatement) -> str | None:
    witness = session.get(Witness, st.witness_id)
    case = session.get(Case, witness.case_id)
    text = st.text_original
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    model = settings.featherless_text_model
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if settings.featherless_disable_thinking and model.startswith("Qwen/Qwen3") else None
    rp = {"prompt_version": wit.PROMPT_VERSION, "model": model, "max_tokens": 1200, "thinking": "disabled" if extra_body else "model_default"}
    ph = params_hash(rp)
    prior = session.execute(select(ModelRun).where(ModelRun.case_id == case.id, ModelRun.stage == STAGE_CLAIMS, ModelRun.input_sha256 == text_hash,
                                                   ModelRun.params_hash == ph, ModelRun.status.in_(("completed", "partially_completed")))).scalars().first()
    if prior is not None:
        job.result = {"reused_model_run": prior.id}
        return None

    run = ModelRun(case_id=case.id, stage=STAGE_CLAIMS, model_id=f"featherless:{model}", model_version=wit.PROMPT_VERSION, params=rp, params_hash=ph,
                   input_sha256=text_hash, device="remote", started_at=_now(), status="running")
    session.add(run)
    session.flush()
    client = featherless.make_client(settings)
    ref = {"statement_sha256": text_hash, "prompt_version": wit.PROMPT_VERSION}

    def call(messages, purpose, extra=None):
        try:
            return client.chat(session, case=case, model=model, messages=messages, purpose=purpose, input_ref=ref | (extra or {}), model_run_id=run.id,
                               extra_body=extra_body)
        except featherless.LlmError as exc:  # keep the failed run and its ledger entries (what was tried, how often), then fail the job
            run.status, run.finished_at, run.metrics = "failed", _now(), {"error": str(exc)[:300]}
            session.commit()
            raise

    first = call(wit.build_messages(text), "witness_claims")
    repaired, tokens = False, [first.tokens_in or 0, first.tokens_out or 0]
    try:
        parsed = wit.parse(first.content)
        client.record_validation(session, first.call_id, True)
    except ValueError as err:
        client.record_validation(session, first.call_id, False)
        second = call(wit.repair_messages(text, first.content, str(err)), "witness_claims_repair", {"repair": True})
        tokens = [tokens[0] + (second.tokens_in or 0), tokens[1] + (second.tokens_out or 0)]
        try:
            parsed = wit.parse(second.content)
            client.record_validation(session, second.call_id, True)
            repaired = True
        except ValueError as err2:
            client.record_validation(session, second.call_id, False)
            run.status, run.finished_at = "failed", _now()
            run.metrics = {"error": str(err2)[:300], "attempts": 2}
            session.commit()  # keep the failed run and ledger entries; the job itself is marked failed by the runner
            raise ExtractionFailed(f"model output invalid after one repair attempt: {err2}")

    taken: list[tuple[int, int]] = []
    stored, rejected = [], []
    for cand in parsed.claims:
        span = wit.locate(text, cand.source_text, taken)
        if span is None:
            rejected.append({"reason": "quote not found verbatim in the statement", "quote_head": cand.source_text[:120]})
            continue
        taken.append(span)
        hint = wit.event_type_hint(cand)
        derived = wit.derive_maneuver(cand.source_text)
        c = WitnessClaim(statement_id=st.id, model_run_id=run.id, char_start=span[0], char_end=span[1], source_text=text[span[0]:span[1]],
                         subject=cand.subject, action=cand.action, direction=cand.direction, time_description=cand.time_description,
                         certainty=cand.certainty, extraction_status="repaired" if repaired else "valid",
                         normalized={"location": cand.location, "time_exactness": cand.time_exactness, "hedges": cand.hedges,
                                     "event_type_hint": hint, "interpretation_by": f"featherless:{model}", "prompt_version": wit.PROMPT_VERSION,
                                     "code_derived_maneuver": derived, "direction_check": wit.direction_check(cand.direction, derived)})
        session.add(c)
        stored.append(c)
    run.metrics = {"claims_stored": len(stored), "claims_rejected": len(rejected), "rejected": rejected, "repaired": repaired,
                   "tokens_in": tokens[0], "tokens_out": tokens[1], "cached_first_call": first.cached}
    run.status, run.finished_at = ("partially_completed" if rejected else "completed"), _now()
    job.result = {"model_run_id": run.id, "claims_stored": len(stored), "claims_rejected": len(rejected), "repaired": repaired,
                  "warnings": [f"{len(rejected)} claim(s) rejected: quote not found verbatim"] if rejected else []}
    audit.record(session, actor="system", action="claims_extracted", case_id=case.id, target_type="statement", target_id=st.id,
                 details={"model_run": run.id, "stored": len(stored), "rejected": len(rejected), "repaired": repaired})
    return "partially_completed" if rejected else None


HANDLERS: dict[str, Callable[[Session, Settings, ProcessingJob], str | None]] = {
    STAGE_VALIDATE: handle_validate, STAGE_PREPROCESS: handle_preprocess, STAGE_INTERVAL: handle_interval, STAGE_DETECT: handle_detect, STAGE_TRACK: handle_track, STAGE_CLAIMS: handle_extract_claims}
PARAMS = {STAGE_VALIDATE: {"validator_version": VALIDATOR_VERSION}}


# ---------------------------------------------------------------------------------------------------- runner
def run_pending(factory: sessionmaker[Session], settings: Settings, limit: int = 10) -> int:
    """Run up to `limit` pending jobs, oldest first. Safe to call concurrently (rows are claimed with SKIP LOCKED)."""
    done = 0
    for _ in range(limit):
        with factory() as s:
            job = s.execute(select(ProcessingJob).where(ProcessingJob.status == "pending").order_by(ProcessingJob.created_at)
                            .limit(1).with_for_update(skip_locked=True)).scalar_one_or_none()
            if job is None:
                return done
            job.status = "validating" if job.stage == STAGE_VALIDATE else "processing"
            job.started_at, job.attempts = _now(), job.attempts + 1
            job_id, stage = job.id, job.stage
            s.commit()
        with factory() as s:
            job = s.get(ProcessingJob, job_id)
            token = _progress_token(factory, job_id)
            try:
                try:
                    outcome = HANDLERS[stage](s, settings, job)
                    job.status, job.progress, job.error = outcome or "completed", 1.0, None
                except Exception as exc:  # noqa: BLE001 - a failed job must be recorded, not crash the worker
                    s.rollback()
                    job = s.get(ProcessingJob, job_id)
                    job.status, job.error = "failed", {"type": type(exc).__name__, "message": str(exc)[:300]}
                job.finished_at = _now()
                s.commit()
            finally:
                _progress_sink.reset(token)
        done += 1
    return done


def retry(session: Session, job: ProcessingJob, actor: str) -> ProcessingJob:
    if job.status not in ("failed", "partially_completed"):
        raise ValueError("only failed or partially completed jobs can be retried")
    if job.attempts >= MAX_ATTEMPTS:
        raise ValueError(f"job already attempted {job.attempts} times")
    job.status, job.error, job.progress, job.finished_at = "pending", None, 0.0, None
    audit.record(session, actor=actor, action="job_retry", case_id=job.case_id, target_type="job", target_id=job.id)
    return job


__all__ = ["enqueue", "run_pending", "retry", "preprocess_params", "detect_params", "STAGE_DETECT", "STAGE_TRACK", "STAGE_CLAIMS", "maybe_enqueue_claims", "STAGE_VALIDATE", "STAGE_PREPROCESS", "STAGE_INTERVAL", "PARAMS"]
