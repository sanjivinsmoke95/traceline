"""Replay stored tracks of real videos through app.services.video_events.generate and print the collision-related events.

    python scripts/replay_events.py [--api http://127.0.0.1:8000] EVIDENCE_ID [EVIDENCE_ID ...]

Reads tracks and accident-cue boxes from the running API (nothing is written), so old and new rule versions can be compared on
identical stored detections. Labels are given by the caller in --truth (evidence_id=collision|none), never inferred.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services import video_events  # noqa: E402


def get(api, path):
    with urllib.request.urlopen(api + "/api/v1" + path) as r:
        return json.load(r)


def load(api, evidence_id):
    ev = get(api, f"/evidence/{evidence_id}")
    v = ev.get("video") or {}
    tracks = [get(api, f"/tracks/{t['id']}") for t in get(api, f"/evidence/{evidence_id}/tracks")["items"]]
    ov = get(api, f"/evidence/{evidence_id}/overlays?from_ms=0&to_ms={int(v.get('duration_ms') or 0) + 1000}")
    cues = [(f["time_ms"], b["confidence"]) for f in ov["items"] for b in f["boxes"] if b.get("is_accident_cue")]
    cue_boxes = [(f["time_ms"], b["bbox"], b["confidence"]) for f in ov["items"] for b in f["boxes"] if b.get("is_accident_cue")]
    times = sorted({f["time_ms"] for f in ov["items"]})
    steps = [b - a for a, b in zip(times, times[1:])]
    period = sorted(steps)[len(steps) // 2] if steps else 200.0
    return ev, tracks, cues, cue_boxes, period, v


def run(api, evidence_id, **kw):
    ev, tracks, cues, cue_boxes, period, v = load(api, evidence_id)
    args = dict(kind=ev["kind"], period_ms=period, video_duration_ms=float(v.get("duration_ms") or 0), cues=cues)
    if "cue_boxes" in video_events.generate.__code__.co_varnames:
        args["cue_boxes"] = cue_boxes
    return ev, video_events.generate([{"track_id": t["track_id"], "class": t["class"], "points": t["points"], "continuity": t.get("continuity")} for t in tracks], **args | kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evidence", nargs="+")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--truth", action="append", default=[], help="evidence_id=collision|none")
    a = ap.parse_args()
    truth = dict(t.split("=") for t in a.truth)
    print(f"rules {video_events.VIDEO_EVENTS_VERSION}")
    for eid in a.evidence:
        ev, events = run(a.api, eid)
        col = [e for e in events if e["event_type"] == "possible_collision"]
        near = [e for e in events if e["event_type"] == "possible_near_miss"]
        print(f"{eid} {ev['original_filename']:<20} kind={ev['kind']:<8} truth={truth.get(eid, '?'):<9} possible_collision={len(col)} near_miss={len(near)} events={len(events)}")
        for e in col[:6]:
            o = e["observations"]
            print(f"    collision {e['t_start_ms']/1000:.1f}-{e['t_end_ms']/1000:.1f}s tracks={e.get('track_ids')} signals={o.get('collision_signals')} score={o.get('collision_score')}")


if __name__ == "__main__":
    main()
