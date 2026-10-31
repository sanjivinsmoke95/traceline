"""Evaluate the collision rules on stored tracks of real clips against hand-labelled collision windows.

    python scripts/eval_collision_rules.py [--api URL] [--dump]

TRUTH is the author's visual judgement of each clip (seconds of media time); it is a small development set, not a benchmark.
A flagged possible_collision is a true positive if it overlaps a labelled window (+/- 1 s), otherwise a false positive. Flags
within 2 s of each other count once. Numbers here describe these six clips only, and the rules were tuned on them.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.replay_events import run  # noqa: E402

TRUTH = {  # evidence id: (label, [(start_s, end_s)])
    "ev_ec6a62b1acd7": ("dashcam, car flipped across junction", [(1.0, 3.5)]),
    "ev_0ff2527eb7f6": ("CCTV, van hits SUV", [(1.0, 2.4)]),
    "ev_920f39c56b91": ("dashcam highway, motorcycle falls, then normal traffic", [(2.0, 4.6)]),
    "ev_cc326f33fcbe": ("dashcam, two crashes", [(12.5, 17.5), (28.0, 33.0)]),
    "ev_7cdf73000e1f": ("CCTV junction, two cars touch", [(4.0, 6.2)]),
    "ev_35e81a7e8f9a": ("CCTV street, queued traffic, no accident", []),
}


def groups(times, gap=2.0):
    out = []
    for t in sorted(times):
        if out and t - out[-1][-1] <= gap:
            out[-1].append(t)
        else:
            out.append([t])
    return [(g[0], g[-1]) for g in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()
    tp = fp = hit_w = tot_w = 0
    for eid, (label, wins) in TRUTH.items():
        dbg = []
        _, events = run(a.api, eid, overrides={"_debug": dbg})
        col = [(e["t_start_ms"] / 1000, e["t_end_ms"] / 1000) for e in events if e["event_type"] == "possible_collision"]
        gs = groups([c[0] for c in col])
        ok = lambda g: any(g[1] >= w[0] - 1 and g[0] <= w[1] + 1 for w in wins)  # noqa: E731
        g_tp, g_fp = [g for g in gs if ok(g)], [g for g in gs if not ok(g)]
        found = [w for w in wins if any(g[1] >= w[0] - 1 and g[0] <= w[1] + 1 for g in gs)]
        tp += len(g_tp); fp += len(g_fp); hit_w += len(found); tot_w += len(wins)
        print(f"{eid[:11]} {label[:48]:<48} windows {len(found)}/{len(wins)}  flagged groups {len(gs)} (true {len(g_tp)}, false {len(g_fp)})")
        if a.dump:
            for d in sorted(dbg, key=lambda d: d["t0"]):
                inw = any(d["t1"] / 1000 >= w[0] - 1 and d["t0"] / 1000 <= w[1] + 1 for w in wins)
                e = d["ev"]
                print(f"    {'TRUE ' if inw else 'false'} {'HIT ' if d['hit'] else '    '} {d['t0']/1000:5.1f}-{d['t1']/1000:5.1f}s tr{d['tracks']} iou={d['max_iou']:.2f} sig={','.join(e['signals'])} "
                      f"app={e['details'].get('approach', {}).get('closing_speed')} arr={e['details'].get('arrest', {}).get('relative_speed_after')}/{e['details'].get('arrest', {}).get('relative_speed_before')} shp={e['details'].get('shape', {}).get('max_aspect_change')}")
    print(f"\nwindows detected {hit_w}/{tot_w}   flagged groups: {tp} true, {fp} false")


if __name__ == "__main__":
    main()
