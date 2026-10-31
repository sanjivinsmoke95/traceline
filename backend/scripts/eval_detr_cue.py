"""Run the DETR accident-scene model on labelled clips and compare its cue with the current accident-cue model.

    python scripts/eval_detr_cue.py [--api URL] [--conf 0.48]

Same labelled windows as eval_collision_rules.py (author's visual judgement; a development set, not a benchmark).
For each clip: persistent DETR cue moments (>=3 sampled frames within 1.2 s), whether each overlaps a labelled window, and the
first-detection delay against the window start.
"""
import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_collision_rules import TRUTH  # noqa: E402
from scripts.replay_events import load  # noqa: E402
from app.services.accident_detr import AccidentDetr  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontend"))

def summarize(items, gap=1200, min_frames=3):
    """Same grouping as frontend/src/lib/cues.js: frames within gap ms form a cluster; a cluster of >= min_frames is a moment."""
    fr = sorted(it["time_ms"] for it in items if any(b.get("is_accident_cue") for b in it.get("boxes", [])))
    pk = {it["time_ms"]: max(b["confidence"] for b in it["boxes"] if b.get("is_accident_cue")) for it in items if any(b.get("is_accident_cue") for b in it.get("boxes", []))}
    cl = []
    for t in fr:
        if cl and t - cl[-1][-1] <= gap: cl[-1].append(t)
        else: cl.append([t])
    return {"moments": [{"startMs": c[0], "endMs": c[-1], "frames": len(c), "peak": max(pk[t] for t in c)} for c in cl if len(c) >= min_frames]}  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--conf", type=float, default=0.48)
    a = ap.parse_args()
    det = AccidentDetr.load()
    for eid, (label, wins) in TRUTH.items():
        path = Path(f"/tmp/sheets/{eid}.mp4")
        if not path.exists():
            urllib.request.urlretrieve(f"{a.api}/api/v1/evidence/{eid}/stream", path)
        cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 25; stride = max(1, round(fps / 5)); i = 0; samples = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % stride == 0:
                boxes = det.accident_boxes(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), a.conf)
                samples.append({"time_ms": i / fps * 1000, "boxes": [{"is_accident_cue": True, "confidence": c, "bbox": b} for c, b in boxes]})
            i += 1
        cap.release()
        s = summarize(samples)
        _, _, cues, cue_boxes, _, _ = load(a.api, eid)
        old = summarize([{"time_ms": t, "boxes": [{"is_accident_cue": True, "confidence": c}]} for t, c in cues])
        def hit(m): return any(m["endMs"] / 1000 >= w[0] - 1 and m["startMs"] / 1000 <= w[1] + 1 for w in wins)
        print(f"\n{eid[:11]} {label}\n  windows {wins}")
        for name, sm in (("DETR", s), ("current cue", old)):
            ms = sm["moments"]
            print(f"  {name:<11} moments={len(ms)} on-window={sum(map(hit, ms))} off-window={sum(not hit(m) for m in ms)} "
                  + " ".join(f"[{m['startMs']/1000:.1f}-{m['endMs']/1000:.1f}s pk{m['peak']:.2f}]" for m in ms[:6]))


if __name__ == "__main__":
    main()
