"""Train and evaluate a learned collision classifier on the same pair features the rules use (app.services.collision.FEATURES).

Library use (Colab notebook) or CLI on stored tracks of a running API:

    python scripts/train_collision_classifier.py --api http://127.0.0.1:8000 --truth truth.json --out /tmp/collision_clf

`truth.json`: {"<video id>": {"kind": "dashcam|cctv", "windows": [[start_s, end_s], ...]}}. A video with an empty list is a negative.
Evaluation is leave-one-video-out: a candidate is never scored by a model that saw any frame of its video, which is the minimum for
a number that means anything. The rules (`hit` flag) are scored the same way as the baseline. With few videos the result is a smoke
test of the pipeline, not an accuracy claim; report it with the number of videos and windows next to it.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services import collision, video_events  # noqa: E402


def candidates(tracks, *, kind, period_ms, duration_ms, cue_boxes=None):
    """Every touching pair the rules considered, with its feature vector and whether the rules flagged it."""
    dbg = []
    cues = [(t, c) for t, _b, c in (cue_boxes or [])]
    video_events.generate(tracks, kind=kind, period_ms=period_ms, video_duration_ms=duration_ms, cues=cues, cue_boxes=cue_boxes, overrides={"_debug": dbg})
    return [{"t0": d["t0"] / 1000, "t1": d["t1"] / 1000, "tracks": d["tracks"], "rules": bool(d["hit"]), "x": collision.features(d["ev"], d["max_iou"])}
            for d in dbg if d["touching"]]


def label(c, windows, pad=1.0):
    return int(any(c["t1"] >= w[0] - pad and c["t0"] <= w[1] + pad for w in windows))


def groups(times, gap=2.0):
    out = []
    for t in sorted(times):
        out.append([t]) if not out or t - out[-1][-1] > gap else out[-1].append(t)
    return [(g[0], g[-1]) for g in out]


def score_groups(flagged, windows, pad=1.0):
    gs = groups([c["t0"] for c in flagged])
    ok = lambda g: any(g[1] >= w[0] - pad and g[0] <= w[1] + pad for w in windows)  # noqa: E731
    return {"windows": len(windows), "found": sum(any(g[1] >= w[0] - pad and g[0] <= w[1] + pad for g in gs) for w in windows),
            "true_groups": sum(map(ok, gs)), "false_groups": sum(not ok(g) for g in gs)}


def model():
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.1, subsample=0.8, random_state=0)


def fit(X, y):
    w = np.where(np.array(y) == 1, max(1.0, (len(y) - sum(y)) / max(sum(y), 1)) ** 0.5, 1.0)  # soften class imbalance, do not flip it
    m = model()
    m.fit(np.array(X), np.array(y), sample_weight=w)
    return m


def pick_threshold(m, data, vids):
    """Threshold that maximises group-level F1 on the *training* videos only."""
    best = (0.0, 0.5)
    for thr in np.linspace(0.2, 0.9, 15):
        tp = fp = fn = 0
        for v in vids:
            cs = data[v]["cands"]
            flagged = [c for c in cs if m.predict_proba([c["x"]])[0][1] >= thr] if cs else []
            s = score_groups(flagged, data[v]["windows"])
            tp += s["true_groups"]; fp += s["false_groups"]; fn += s["windows"] - s["found"]
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        best = max(best, (f1, float(thr)))
    return best[1]


def leave_one_video_out(data):
    rows, rules_rows = [], []
    for held in data:
        train = [v for v in data if v != held]
        X = [c["x"] for v in train for c in data[v]["cands"]]
        y = [label(c, data[v]["windows"]) for v in train for c in data[v]["cands"]]
        if len(set(y)) < 2:
            rows.append({"video": held, "skipped": "training videos contain only one class"}); continue
        m = fit(X, y)
        thr = pick_threshold(m, data, train)
        cs = data[held]["cands"]
        flagged = [c for c in cs if m.predict_proba([c["x"]])[0][1] >= thr]
        rows.append({"video": held, "threshold": round(thr, 2)} | score_groups(flagged, data[held]["windows"]))
        rules_rows.append({"video": held} | score_groups([c for c in cs if c["rules"]], data[held]["windows"]))
    return rows, rules_rows


def summarize(rows):
    r = [x for x in rows if "skipped" not in x]
    return {k: sum(x[k] for x in r) for k in ("windows", "found", "true_groups", "false_groups")} | {"videos": len(r)}


def load_from_api(api, truth):
    from scripts.replay_events import load
    data = {}
    for vid, t in truth.items():
        ev, tracks, _cues, cue_boxes, period, v = load(api, vid)
        tr = [{"track_id": x["track_id"], "class": x["class"], "points": x["points"], "continuity": x.get("continuity")} for x in tracks]
        data[vid] = {"windows": t["windows"], "cands": candidates(tr, kind=t.get("kind", ev["kind"]), period_ms=period, duration_ms=float(v.get("duration_ms") or 0), cue_boxes=cue_boxes)}
    return data


def tracks_from_video(path, weights="yolo26s.pt", tracker="bytetrack.yaml", classes=(2, 3, 5, 7, 0)):
    """Detect and track a video with Ultralytics (Colab). Returns (tracks in the pipeline's point format, period_ms, duration_ms)."""
    from ultralytics import YOLO
    import cv2
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0; n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    names = {0: "pedestrian", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    tracks = {}
    stride = max(1, round(fps / 5))  # about 5 sampled frames per second, like the pipeline
    for i, r in enumerate(YOLO(weights).track(str(path), stream=True, persist=True, tracker=tracker, classes=list(classes), vid_stride=stride, verbose=False)):
        if r.boxes is None or r.boxes.id is None:
            continue
        frame_no, t_ms = i * stride, i * stride / fps * 1000
        for b, tid, cls, conf in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            tracks.setdefault(tid, {"track_id": tid, "class": names.get(cls, "car"), "points": []})["points"].append(
                {"frame_no": frame_no, "time_ms": t_ms, "bbox": b, "anchor": [(b[0] + b[2]) / 2, b[3]], "confidence": conf})
    return [t for t in tracks.values() if len(t["points"]) >= 3], stride / fps * 1000, n / fps * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    truth = json.loads(Path(a.truth).read_text())
    data = load_from_api(a.api, truth)
    rows, rules = leave_one_video_out(data)
    print(json.dumps({"learned": summarize(rows), "rules": summarize(rules), "per_video": rows}, indent=1))
    if a.out:
        import joblib
        X = [c["x"] for v in data for c in data[v]["cands"]]; y = [label(c, data[v]["windows"]) for v in data for c in data[v]["cands"]]
        m = fit(X, y)
        thr = pick_threshold(m, data, list(data))
        Path(a.out).mkdir(parents=True, exist_ok=True)
        joblib.dump(m, Path(a.out) / "collision_clf.joblib")
        (Path(a.out) / "collision_clf.json").write_text(json.dumps({"features": collision.FEATURES, "threshold": thr, "videos": len(data), "positives": int(sum(y)), "candidates": len(y),
                                                                  "note": "trained on a small development set; not validated for use"}, indent=1))


if __name__ == "__main__":
    main()
