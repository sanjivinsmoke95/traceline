"""Collision evidence for one pair of tracks, from image-space kinematics. Deterministic; nothing here is a fact about the world.

Why not "two boxes overlap and something slowed down": in dense traffic, and on any moving camera, boxes overlap constantly
(one car occludes another) and every track's image speed changes when the camera's own vehicle brakes. Those two cues alone
flag ordinary traffic. A collision is an exchange of momentum between two bodies, which shows up in the *pair*:

  approach  the two objects close on each other quickly before contact (a steady following gap does not),
  arrest    their relative motion collapses or is deflected right after contact (the closing is stopped),
  rest      they stay together after contact instead of passing through/over each other,
  shape     one box changes shape abruptly after contact (rolled, flipped or fallen vehicle, rider on the ground),
  cue       the weak accident-cue model fires on the pair over several frames.

All quantities are relative motion between the two objects, in units of the larger box width, so the camera's own translation
cancels to first order and nothing depends on resolution. A possible collision needs contact plus either the kinematic triple
(approach + arrest + rest) or an abrupt shape change with arrest or approach, or the cue agreeing with one kinematic signal.
Overlap, even with an abrupt speed change of one track, is never enough.
"""
import math

import numpy as np

DEFAULTS = {
    "pre_window_ms": 1000, "post_window_ms": 1000, "approach_min_closing": 0.5, "approach_min_speed": 0.5, "approach_closing_share": 0.6,
    "arrest_min_pre_speed": 0.6, "arrest_ratio": 0.4, "deflect_deg": 70.0, "rest_max_dist": 1.3, "shape_change": 0.4,
    "cue_min_frames": 3, "cue_pad_ms": 600, "cue_min_overlap": 0.1, "min_history": 4, "max_plausible_speed": 3.5,
}


def _w(b): return max(b[2] - b[0], 1e-6)
def _h(b): return max(b[3] - b[1], 1e-6)
def _c(b): return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = _w(a) * _h(a) + _w(b) * _h(b) - inter
    return inter / union if union > 0 else 0.0


def pair_series(pa: dict, pb: dict) -> dict | None:
    """pa, pb: {frame_no: point}. Relative position of A to B in units of the larger box width, per common frame."""
    frames = sorted(set(pa) & set(pb))
    if len(frames) < 3:
        return None
    t, rx, ry = [], [], []
    for f in frames:
        ba, bb = pa[f]["bbox"], pb[f]["bbox"]
        s = max(_w(ba), _w(bb))
        (ax, ay), (bx, by) = _c(ba), _c(bb)
        t.append(pa[f]["time_ms"]); rx.append((ax - bx) / s); ry.append((ay - by) / s)
    return {"frames": frames, "t": np.array(t, float), "r": np.stack([rx, ry], 1)}


def _rel_speed(s):
    """Relative speed between consecutive samples (box widths per second) with its midpoint time and vector."""
    dt = np.diff(s["t"]) / 1000.0
    dr = np.diff(s["r"], axis=0)
    ok = dt > 0
    u = np.zeros_like(dr)
    u[ok] = dr[ok] / dt[ok][:, None]
    return (s["t"][:-1] + s["t"][1:]) / 2, u


def _aspect_shift(pts: dict, frames: list[int], t0: float, p: dict) -> float:
    pre = [_h(pts[f]["bbox"]) / _w(pts[f]["bbox"]) for f in frames if t0 - p["pre_window_ms"] <= pts[f]["time_ms"] <= t0]
    post = [_h(pts[f]["bbox"]) / _w(pts[f]["bbox"]) for f in frames if t0 <= pts[f]["time_ms"] <= t0 + p["post_window_ms"]]
    if len(pre) < 2 or not post:
        return 0.0
    base = float(np.median(pre))
    return max(abs(x / base - 1.0) for x in post)


def assess(pa: dict, pb: dict, t_contact_ms: float, t_end_ms: float, cue_boxes: list | None = None, overrides: dict | None = None) -> dict:
    """Signals for the pair whose boxes first touch at t_contact_ms. Returns {signals, score, details, abrupt}."""
    p = DEFAULTS | (overrides or {})
    out = {"signals": [], "details": {}, "abrupt": None}
    s = pair_series(pa, pb)
    if s is None:
        return out
    t, r = s["t"], s["r"]
    dist = np.hypot(r[:, 0], r[:, 1])
    mid, u = _rel_speed(s)
    mag = np.hypot(u[:, 0], u[:, 1]) if len(u) else np.array([])

    pre_i = [i for i in range(len(t)) if t_contact_ms - p["pre_window_ms"] <= t[i] <= t_contact_ms]
    if len(pre_i) >= p["min_history"]:
        closed = float(dist[pre_i[0]] - dist[pre_i[-1]])
        steps = np.diff(dist[pre_i])
        share = float(np.mean(steps < 0)) if len(steps) else 0.0
        dur = (t[pre_i[-1]] - t[pre_i[0]]) / 1000.0
        closing_speed = closed / dur if dur > 0 else 0.0
        out["details"]["approach"] = {"closed_widths": round(float(closed), 2), "closing_speed": round(float(closing_speed), 2), "closing_share": round(float(share), 2)}
        out["details"]["approach"]["plausible"] = bool(closing_speed <= p["max_plausible_speed"])
        if closing_speed <= p["max_plausible_speed"] and closed >= p["approach_min_closing"] and closing_speed >= p["approach_min_speed"] and share >= p["approach_closing_share"]:
            out["signals"].append("approach")

    pre_m = [i for i in range(len(mid)) if t_contact_ms - 600 <= mid[i] <= t_contact_ms]
    post_m = [i for i in range(len(mid)) if t_contact_ms + 200 <= mid[i] <= t_contact_ms + p["post_window_ms"]]
    if pre_m and post_m:
        pre_u, post_u = float(np.mean(mag[pre_m])), float(np.mean(mag[post_m]))
        pv, qv = u[pre_m].mean(0), u[post_m].mean(0)
        turn = math.degrees(math.acos(float(np.clip(np.dot(pv, qv) / (np.linalg.norm(pv) * np.linalg.norm(qv) + 1e-9), -1, 1))))
        out["details"]["arrest"] = {"relative_speed_before": round(float(pre_u), 2), "relative_speed_after": round(float(post_u), 2), "deflection_deg": round(float(turn), 1)}
        if p["arrest_min_pre_speed"] <= pre_u <= p["max_plausible_speed"] and (post_u <= pre_u * p["arrest_ratio"] or (turn >= p["deflect_deg"] and np.linalg.norm(qv) >= 0.4)):
            out["signals"].append("arrest")
            out["abrupt"] = {"relative_speed_before": round(float(pre_u), 2), "relative_speed_after": round(float(post_u), 2)}

    after = [i for i in range(len(t)) if t_contact_ms + 600 <= t[i] <= t_contact_ms + p["post_window_ms"] + 200]
    if after:
        d_after = float(np.median(dist[after]))
        out["details"]["rest"] = {"distance_after_widths": round(float(d_after), 2)}
        if d_after <= p["rest_max_dist"]:
            out["signals"].append("rest")

    shifts = [_aspect_shift(pa, s["frames"], t_contact_ms, p), _aspect_shift(pb, s["frames"], t_contact_ms, p)]
    out["details"]["shape"] = {"max_aspect_change": round(float(max(shifts)), 2)}
    if max(shifts) >= p["shape_change"]:
        out["signals"].append("shape")

    if cue_boxes:
        lo, hi = t_contact_ms - p["cue_pad_ms"], t_end_ms + p["cue_pad_ms"]
        hits = set()
        for (ct, cb, _conf) in cue_boxes:
            if not lo <= ct <= hi:
                continue
            f = min(s["frames"], key=lambda ff: abs(pa[ff]["time_ms"] - ct))
            ba, bb = pa[f]["bbox"], pb[f]["bbox"]
            union = [min(ba[0], bb[0]), min(ba[1], bb[1]), max(ba[2], bb[2]), max(ba[3], bb[3])]
            if _iou(cb, union) >= p["cue_min_overlap"]:
                hits.add(round(ct))
        out["details"]["cue"] = {"frames": len(hits)}
        if len(hits) >= p["cue_min_frames"]:
            out["signals"].append("cue")
    out["score"] = len(out["signals"])
    return out


def is_collision(signals: list[str]) -> bool:
    """Contact is checked by the caller. Overlap with one track slowing, a shape change alone, or staying close are not sufficient:
    the pair must approach and then have its relative motion arrested, backed by a second independent cue, or the weak cue model must
    persist over several frames on the pair together with one kinematic signal."""
    k = set(signals)
    core = {"approach", "arrest"} <= k and bool(k & {"shape", "cue", "rest"})
    cue = "cue" in k and bool(k & {"approach", "arrest", "shape"})
    return core or cue
