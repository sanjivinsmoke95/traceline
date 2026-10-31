"""Phase 4: weight safety, detection metrics, detector + accident-cue stage, detections API."""
import hashlib
import json
import os
import pickle
import shutil
import zipfile

import cv2
import numpy as np
import pytest
from sqlalchemy import select

from app.eval.detection_metrics import ap50, average_precision, iou, match, precision_recall_at
from app.models import Detection, ModelRun
from app.services import weights
from app.services.detection import COCO_KEEP
from tests.conftest import MODELS, needs_accident, needs_weights


# ------------------------------------------------------------------------------------------ metrics
def test_iou_basics():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9


def test_ap_values():
    gts = {"a": [[0, 0, 10, 10]], "b": [[0, 0, 10, 10]]}
    assert 0.99 <= ap50([("a", 0.9, [0, 0, 10, 10]), ("b", 0.8, [0, 0, 10, 10])], gts) <= 1.0  # 0.995: 101-point convention
    # one of two objects found, at perfect precision: AP must be ~0.5, not credited up to recall 1.0
    half = ap50([("a", 0.9, [0, 0, 10, 10])], gts)
    assert 0.49 <= half <= 0.51
    assert ap50([("a", 0.9, [50, 50, 60, 60])], gts) == 0.0  # only a false positive
    assert ap50([], gts) == 0.0
    # a duplicate box on the same object counts once
    conf, tp, n = match([("a", 0.9, [0, 0, 10, 10]), ("a", 0.8, [0, 0, 10, 10])], gts)
    assert tp.tolist() == [True, False] and n == 2
    assert precision_recall_at(conf, tp, n, 0.85) == (1.0, 0.5)
    assert np.isnan(average_precision(np.array([True]), 0))


def test_ap_matches_ultralytics_on_random_curves():
    from ultralytics.utils.metrics import compute_ap

    rng = np.random.default_rng(0)
    for _ in range(200):
        n_gt = int(rng.integers(1, 40))
        tp = rng.random(int(rng.integers(0, 80))) < rng.random()
        tp[np.cumsum(tp) > n_gt] = False  # cannot have more hits than objects
        if tp.size == 0:
            assert average_precision(tp, n_gt) == 0.0
            continue
        ctp = np.cumsum(tp)
        ref, _, _ = compute_ap(ctp / n_gt, ctp / np.arange(1, len(tp) + 1))
        assert abs(average_precision(tp, n_gt) - ref) < 1e-9


# ------------------------------------------------------------------------------------------ weight safety
class _Evil:
    def __reduce__(self):  # would run a shell command if this pickle were ever loaded
        return (os.system, (f"touch {self.marker}",))


def _malicious_checkpoint(path, marker):
    evil = _Evil()
    evil.marker = marker
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", pickle.dumps(evil, protocol=2))


def _temp_lock(monkeypatch, tmp_path, entries):
    lock = tmp_path / "models.lock.json"
    lock.write_text(json.dumps(entries))
    monkeypatch.setattr(weights, "LOCK_FILE", lock)


def test_malicious_weights_are_refused_without_executing(tmp_path, monkeypatch):
    marker = tmp_path / "pwned"
    ckpt = tmp_path / "evil.pt"
    _malicious_checkpoint(ckpt, marker)
    assert any("system" in r for r in weights.disallowed(weights.pickle_globals(ckpt)))
    _temp_lock(monkeypatch, tmp_path, {"evil": {"file": "evil.pt", "kind": "detector", "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()}})
    with pytest.raises(weights.UnsafeWeights, match="disallowed"):
        weights.verified_path("evil", tmp_path)
    assert not marker.exists()  # the payload never ran


def test_hash_mismatch_unknown_and_missing_are_refused(tmp_path, monkeypatch):
    f = tmp_path / "m.pt"
    f.write_bytes(b"not a checkpoint")
    _temp_lock(monkeypatch, tmp_path, {"m": {"file": "m.pt", "kind": "detector", "sha256": "0" * 64},
                                       "gone": {"file": "gone.pt", "kind": "detector", "sha256": "0" * 64}})
    with pytest.raises(weights.UnsafeWeights, match="SHA-256"):
        weights.verified_path("m", tmp_path)
    with pytest.raises(weights.UnsafeWeights, match="not in models.lock"):
        weights.verified_path("other", tmp_path)
    with pytest.raises(FileNotFoundError):
        weights.verified_path("gone", tmp_path)


@needs_weights
def test_installed_weights_pass_verification():
    for name, entry in weights.load_lock().items():
        if (MODELS / entry["file"]).is_file():
            path, _ = weights.verified_path(name, MODELS)
            assert path.is_file()


# ------------------------------------------------------------------------------------------ helpers
def make_case(client):
    return client.post("/api/v1/cases", json={"title": "Detection test", "designation": "synthetic"}).json()


def upload(client, case_id, path, kind="cctv"):
    r = client.post(f"/api/v1/cases/{case_id}/evidence", files={"file": (path.name, path.read_bytes())}, data={"kind": kind, "synthetic": "true"})
    assert r.status_code == 201, r.text
    return r.json()["evidence"]["id"]


def stage_jobs(client, case_id, stage):
    return [j for j in client.get(f"/api/v1/cases/{case_id}/jobs").json()["items"] if j["stage"] == stage]


# ------------------------------------------------------------------------------------------ detection stage
@needs_weights
def test_pipeline_detects_bus_and_pedestrians_with_traceable_runs(client, bus_video):
    case = make_case(client)
    eid = upload(client, case["id"], bus_video)
    statuses = {j["stage"]: j["status"] for j in client.get(f"/api/v1/cases/{case['id']}/jobs").json()["items"]}
    assert statuses["validate"] == statuses["preprocess"] == "completed" and statuses["detect"] in ("completed", "partially_completed")

    det = client.get(f"/api/v1/evidence/{eid}/detections").json()
    classes = {d["class"] for f in det["items"] for d in f["detections"]}
    assert {"bus", "pedestrian"} <= classes and classes <= set(COCO_KEEP.values()) | {"accident_cue"}
    w, h = det["frame_size"]["width"], det["frame_size"]["height"]
    sampled = {f["frame_no"] for f in client.get(f"/api/v1/evidence/{eid}/frames").json()["items"]}
    for f in det["items"]:
        assert f["frame_no"] in sampled
        for d in f["detections"]:
            x1, y1, x2, y2 = d["bbox"]
            assert -1 <= x1 < x2 <= w + 1 and -1 <= y1 < y2 <= h + 1 and d["confidence"] >= 0.25
            assert d["category"] == "machine_detection"

    runs = {r["stage"]: r for r in client.get(f"/api/v1/evidence/{eid}/model-runs").json()["items"]}
    assert runs["detect"]["model_id"] == "yolo26s" and runs["detect"]["model_version"].startswith("sha256:646f8bc3fe0a")
    assert runs["detect"]["params"]["conf"] == client.settings.detector_conf


@needs_weights
def test_stored_boxes_match_direct_inference_on_the_same_frame(client, bus_video):
    from ultralytics import YOLO

    case = make_case(client)
    eid = upload(client, case["id"], bus_video)
    det = client.get(f"/api/v1/evidence/{eid}/detections?min_conf=0.5").json()
    target = det["items"][len(det["items"]) // 2]
    cap = cv2.VideoCapture(str(bus_video))
    for _ in range(target["frame_no"] + 1):
        ok, img = cap.read()
    direct = YOLO(str(MODELS / "yolo26s.pt")).predict(img, imgsz=640, conf=0.5, device="cpu", verbose=False)[0]
    ref = sorted((COCO_KEEP[direct.names[int(c)]], [round(v) for v in b]) for b, c in zip(direct.boxes.xyxy.tolist(), direct.boxes.cls.tolist())
                 if direct.names[int(c)] in COCO_KEEP)
    ours = sorted((d["class"], [round(v) for v in d["bbox"]]) for d in target["detections"] if not d["is_accident_cue"])
    assert [c for c, _ in ours] == [c for c, _ in ref]
    for (_, a), (_, b) in zip(ours, ref):
        assert max(abs(x - y) for x, y in zip(a, b)) <= 3  # same frame, same boxes (device rounding aside)


@needs_weights
def test_empty_scene_records_zero_detections(client, sample_video):
    case = make_case(client)
    eid = upload(client, case["id"], sample_video)
    job = stage_jobs(client, case["id"], "detect")[0]
    assert job["status"] in ("completed", "partially_completed") and job["result"]["detector"]["boxes"] == 0
    det = client.get(f"/api/v1/evidence/{eid}/detections").json()
    assert det["items"] == [] and det["frames_considered"] == 10  # absence is recorded, not inferred


@needs_accident
def test_accident_cue_is_a_separate_flagged_run(client, accident_video):
    case = make_case(client)
    eid = upload(client, case["id"], accident_video)
    det = client.get(f"/api/v1/evidence/{eid}/detections").json()
    cues = [d for f in det["items"] for d in f["detections"] if d["is_accident_cue"]]
    assert cues and all(d["class"] == "accident_cue" for d in cues)
    assert "never establishes a collision" in det["note"]
    stages = {r["stage"] for r in det["model_runs"]}
    assert stages == {"detect", "accident_cue"}
    without = client.get(f"/api/v1/evidence/{eid}/detections?include_accident_cue=false").json()
    assert not any(d["is_accident_cue"] for f in without["items"] for d in f["detections"])
    only = client.get(f"/api/v1/evidence/{eid}/detections?classes=accident_cue").json()
    assert all(d["class"] == "accident_cue" for f in only["items"] for d in f["detections"])


@needs_weights
def test_accident_cue_disabled_or_missing(client, bus_video, tmp_path):
    client.settings.accident_model = None  # disabled by configuration: not partial
    case = make_case(client)
    upload(client, case["id"], bus_video)
    job = stage_jobs(client, case["id"], "detect")[0]
    assert job["status"] == "completed" and "accident_cue" not in job["result"]

    models = tmp_path / "models_no_accident"  # configured but weights absent: detector still runs, job is partial
    models.mkdir()
    shutil.copy(MODELS / "yolo26s.pt", models / "yolo26s.pt")
    client.settings.accident_model, client.settings.models_dir = "accident_yolov8s", models
    case2 = make_case(client)
    upload(client, case2["id"], bus_video)
    job2 = stage_jobs(client, case2["id"], "detect")[0]
    assert job2["status"] == "partially_completed" and "skipped" in job2["result"]["accident_cue"]
    assert job2["result"]["detector"]["boxes"] > 0


def test_missing_detector_fails_detect_but_keeps_earlier_outputs(client, sample_video, tmp_path):
    client.settings.models_dir = tmp_path / "empty_models"
    case = make_case(client)
    eid = upload(client, case["id"], sample_video)
    job = stage_jobs(client, case["id"], "detect")[0]
    assert job["status"] == "failed" and job["error"]["type"] == "FileNotFoundError"
    assert stage_jobs(client, case["id"], "preprocess")[0]["status"] == "completed"
    assert client.get(f"/api/v1/evidence/{eid}/stream").status_code == 200
    r = client.get(f"/api/v1/evidence/{eid}/detections")
    assert r.status_code == 409 and r.json()["error"]["code"] == "DETECTIONS_NOT_READY"


@needs_weights
def test_detect_rerun_is_idempotent_and_filters_work(client, bus_video):
    case = make_case(client)
    eid = upload(client, case["id"], bus_video)
    again = client.post(f"/api/v1/cases/{case['id']}/process", json={"stages": ["detect"]}).json()["jobs"][0]
    res = client.get(f"/api/v1/jobs/{again['id']}").json()["result"]
    assert "reused_model_run" in res["detector"]
    loose = client.get(f"/api/v1/evidence/{eid}/detections?min_conf=0.15").json()
    strict = client.get(f"/api/v1/evidence/{eid}/detections?min_conf=0.8").json()
    n = lambda d: sum(len(f["detections"]) for f in d["items"])  # noqa: E731
    assert n(loose) >= n(strict)
    window = client.get(f"/api/v1/evidence/{eid}/detections?from_ms=1000&to_ms=2000").json()
    assert all(1000 <= f["time_ms"] <= 2000 for f in window["items"])
    buses = client.get(f"/api/v1/evidence/{eid}/detections?classes=bus").json()
    assert {d["class"] for f in buses["items"] for d in f["detections"]} == {"bus"}


def test_detr_accident_cue_failure_is_partial_not_fatal(client, bus_video, monkeypatch):
    """The DETR cue is optional: if it cannot load, the detector result is still stored and the job is partially completed."""
    from app.services import accident_detr

    def boom(*a, **k):
        raise OSError("model not available offline")
    monkeypatch.setattr(accident_detr.AccidentDetr, "load", classmethod(lambda cls, *a, **k: boom()))
    client.settings.accident_backend = "detr"
    case = make_case(client)
    upload(client, case["id"], bus_video)
    job = stage_jobs(client, case["id"], "detect")[0]
    assert job["status"] == "partially_completed"
    assert "skipped" in job["result"]["accident_cue"] and job["result"]["detector"]["boxes"] > 0


def test_detr_accident_cue_runs_with_a_stub_and_is_stored_as_cue_boxes(client, bus_video, monkeypatch):
    from app.services import accident_detr

    class Stub:
        def accident_boxes(self, image, conf):
            return [(0.9, [5.0, 5.0, 60.0, 60.0])]
    monkeypatch.setattr(accident_detr.AccidentDetr, "load", classmethod(lambda cls, *a, **k: Stub()))
    client.settings.accident_backend = "detr"
    case = make_case(client)
    upload(client, case["id"], bus_video)
    job = stage_jobs(client, case["id"], "detect")[0]
    assert job["status"] == "completed" and job["result"]["accident_cue"]["boxes"] == job["result"]["frames"]
    with client.factory() as s:
        run = s.execute(select(ModelRun).where(ModelRun.case_id == case["id"], ModelRun.stage == "accident_cue")).scalars().one()
        assert run.model_id == "accident_detr" and run.model_version.startswith("revision:")
        assert s.execute(select(Detection).where(Detection.model_run_id == run.id, Detection.is_accident_cue.is_(True))).scalars().first() is not None
