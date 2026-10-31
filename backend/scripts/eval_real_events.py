"""Score collision-type alerts on the labelled REAL clips (eval_data/real_clips.json). No database, no API.

    python scripts/eval_real_events.py --tag after [--clips candidate-04 ...] [--event-version legacy|current]

Raw detections are cached under research/backend_accuracy/cache (git-ignored), so threshold changes re-run in seconds.
`--event-version legacy` re-creates the pre-audit rules (video_events v2) so before/after numbers come from the same detections.
Real and synthetic results are never mixed: this script reads real clips only; synthetic scenarios are scored by eval_scenarios.py.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_settings  # noqa: E402
from app.eval import event_metrics, offline  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1]
OUT = BACKEND.parent / "research" / "backend_accuracy"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="current")
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--event-version", choices=["legacy", "current"], default="current")
    ap.add_argument("--sample-fps", type=float, default=5.0)
    a = ap.parse_args()
    labels = json.loads((BACKEND / "eval_data" / "real_clips.json").read_text())
    root = Path(os.environ.get(labels["media_root_env"], labels["media_root_default"]))
    models = str(get_settings().models_path)
    cache = OUT / "cache"
    if a.event_version == "legacy":
        from app.services import video_events_legacy as ve
    else:
        from app.services import video_events as ve
    offline.video_events = ve  # run_clip resolves the generator through this module attribute

    results, durations, summary = {}, {}, {}
    for clip in labels["clips"]:
        if a.clips and clip["id"] not in a.clips:
            continue
        path = root / clip["media"]
        if not path.is_file():
            print(f"{clip['id']:28s} MISSING {path}")
            continue
        t0 = time.time()
        r = offline.run_clip(path, clip["kind"], cache, models_dir=models, sample_fps=a.sample_fps)
        results[clip["id"]] = r["events"]
        durations[clip["id"]] = (r["index"]["duration_ms"] or 0) / 1000
        c = Counter(e["event_type"] for e in r["events"])
        tiers = Counter(((e.get("observations") or {}).get("evidence_model") or {}).get("tier", "n/a") for e in r["events"])
        summary[clip["id"]] = {"events": dict(c), "tiers": dict(tiers), "tracks": len(r["tracks"]), "seconds": round(time.time() - t0, 1)}
        print(f"{clip['id']:28s} tracks={len(r['tracks']):3d} collision={c.get('possible_collision', 0):2d} near_miss={c.get('possible_near_miss', 0):2d} "
              f"interaction={c.get('potential_interaction', 0):3d} tiers={dict(tiers)}", flush=True)
    scored = event_metrics.score(results, labels, durations)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"real_events_{a.tag}.json"
    out.write_text(json.dumps({"designation": "REAL footage, hand-labelled, tiny n; see caveat", "event_version": a.event_version, "video_events_version": ve.VIDEO_EVENTS_VERSION,
                               "sample_fps": a.sample_fps, "scores": scored, "per_clip_event_counts": summary}, indent=2, default=str))
    s = scored
    print(f"\nTP={s['counts']['tp']} FP={s['counts']['fp']} FN={s['counts']['fn']}  precision={s['precision']['value']} recall={s['recall']['value']} f1={s['f1']}")
    print(f"FP alerts per scored clip={s['false_positive_alerts_per_scored_clip']}  negative clips with any alert={s['negative_clips_with_any_alert']['k']}/{s['negative_clips_with_any_alert']['n']}")
    print(f"timing error ms: {s['timing_error_ms']}")
    print("unscored positives:", {k: v['alerts'] for k, v in s["unscored_positive_clips"].items()})
    print("written", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
