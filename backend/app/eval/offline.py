"""Offline (database-free) run of the real analysis path on one video, with an on-disk cache of the expensive model outputs.

It calls the same services as the job runner (frame PTS index, sampling grid, detector, accident cue, tracker, track
post-processing, event generation) so threshold work can be done on cached raw detections without re-running the models.
Raw detections are cached with their model provenance and are never edited. Nothing here writes to the evidence store.
"""
import hashlib
import json
import statistics
from pathlib import Path

from app.services import detection, tracking, video_events
from app.services import video as vid
from app.services.validators import validate

CACHE_VERSION = "1"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def frame_index(path: Path, sample_fps: float, max_frames: int = 3000) -> dict:
    """Real container PTS (relative to the first frame), the sampled subset, and the video's own metadata."""
    meta = validate("cctv", path)
    pts = vid.probe_frame_pts(path)
    if not pts:
        raise ValueError("no frame timestamps could be read")
    rel = [(t - pts[0]) * 1000 for t in pts]
    sampled, truncated = vid.choose_sampled(rel, sample_fps, max_frames)
    m = meta.get("metadata", {})
    return {"pts_start_ms": pts[0] * 1000, "rel_ms": rel, "sampled": sampled, "truncated": truncated, "rotation": m.get("rotation"),
            "width": m.get("width"), "height": m.get("height"), "duration_ms": m.get("duration_ms"), "valid": meta.get("ok"), "errors": meta.get("errors", []),
            "fps_nominal": m.get("fps_nominal"), "fps_measured": m.get("fps_measured") or m.get("fps_average")}


def detect_cached(path: Path, idx: dict, cache_dir: Path, *, models_dir: str, detector: str = "yolo26s", accident: str | None = "accident_yolov8s",
                  imgsz: int = 640, conf: float = 0.15, accident_conf: float = 0.25, device: str = "cpu") -> dict:
    """Raw detections per sampled frame, cached by (file hash, model, conf, imgsz, sampled frames)."""
    from app.services.weights import load_yolo

    sha = file_sha256(path)
    key = hashlib.sha256(json.dumps([sha, detector, accident, imgsz, conf, accident_conf, idx["sampled"][:5], len(idx["sampled"]), detection.DETECTION_VERSION]).encode()).hexdigest()[:16]
    cache = cache_dir / f"{path.stem}.{key}.detections.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    wanted = set(idx["sampled"])
    out: dict = {"video_sha256": sha, "frames": {}, "cue": {}, "provenance": {"detector": detector, "accident_cue": accident, "imgsz": imgsz, "conf": conf,
                                                                              "accident_conf": accident_conf, "device": device, "version": detection.DETECTION_VERSION}}
    passes = [("frames", detector, detection.COCO_KEEP, conf)] + ([("cue", accident, detection.ACCIDENT_KEEP, accident_conf)] if accident else [])
    for slot, name, keep, c in passes:
        model, entry = load_yolo(name, models_dir)
        out["provenance"][f"{slot}_weights_sha256"] = entry["sha256"]
        stats: dict = {}
        for i, boxes in detection.detect_stream(vid.iter_frames(path, wanted, idx["rotation"]), model, imgsz=imgsz, conf=c, device=device, keep=keep, stats=stats):
            out[slot][str(i)] = [{"cls": b.class_name, "conf": b.confidence, "box": [b.x1, b.y1, b.x2, b.y2]} for b in boxes]
        out["provenance"][f"{slot}_ms_per_frame"] = stats.get("ms_per_frame")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def track(path: Path, idx: dict, dets: dict, kind: str, *, sample_fps: float, min_conf: float = 0.15, overrides: dict | None = None) -> tuple[list[dict], list[dict], dict]:
    """ByteTrack (cctv) or BoT-SORT (dashcam) over the cached detections, then the standard post-processing."""
    p = tracking.default_params(kind, sample_fps) | (overrides or {})
    frames, det_id = [], 0
    for i in idx["sampled"]:
        fr = tracking.Frame(i, round(idx["rel_ms"][i]))
        for d in dets["frames"].get(str(i), []):
            if d["conf"] >= min_conf and d["cls"] in p["classes"]:
                det_id += 1
                fr.dets.append(tracking.Det(det_id, i, fr.time_ms, d["cls"], d["conf"], tuple(d["box"])))
        frames.append(fr)
    images = vid.iter_frames(path, {f.frame_no for f in frames}, idx["rotation"]) if p["tracker"] == "botsort" else None
    assign = tracking.associate(frames, tracking.make_tracker(p), images)
    size = vid.display_size(idx["width"], idx["height"], idx["rotation"]) if idx.get("width") else None
    tracks, cands = tracking.build_tracks(frames, assign, p, size)
    return tracks, cands, p


def period_ms(idx: dict) -> float:
    t = [idx["rel_ms"][i] for i in idx["sampled"]]
    steps = [b - a for a, b in zip(t, t[1:])]
    return float(statistics.median(steps)) if steps else 200.0


def cue_list(dets: dict, idx: dict) -> list[tuple[float, float]]:
    return [(idx["rel_ms"][int(i)], b["conf"]) for i, boxes in dets["cue"].items() for b in boxes]


def run_clip(path: Path, kind: str, cache_dir: Path, *, models_dir: str, sample_fps: float = 5.0, event_overrides: dict | None = None,
             tracker_overrides: dict | None = None, detector_conf: float = 0.15) -> dict:
    idx = frame_index(path, sample_fps)
    dets = detect_cached(path, idx, cache_dir, models_dir=models_dir, conf=detector_conf)
    tracks, cands, tparams = track(path, idx, dets, kind, sample_fps=sample_fps, overrides=tracker_overrides)
    per = period_ms(idx)
    events = video_events.generate(
        [{"track_id": t["track_id"], "class": t["class"], "points": t["points"], "continuity": t["continuity"]} for t in tracks],
        kind=kind, period_ms=per, video_duration_ms=float(idx["duration_ms"] or idx["rel_ms"][-1]), cues=cue_list(dets, idx), overrides=event_overrides)
    return {"index": idx, "detections": dets, "tracks": tracks, "continuation_candidates": cands, "tracker_params": tparams, "period_ms": per, "events": events}
