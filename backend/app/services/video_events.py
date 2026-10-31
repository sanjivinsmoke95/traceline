"""Candidate events from tracks (image space). Everything here is a machine detection: 'possible_*', 'apparent', never a fact.

Speeds are bottom-centre displacement in pixels per second, compared as a fraction of the object's own box diagonal so the
rules do not depend on resolution or distance from the camera. Direction-based rules use the sense of rotation (left/right
relative to the vehicle's own travel), which a non-mirrored fixed camera preserves. On a moving camera (dashcam) the image
motion includes the ego vehicle's own motion, so those events are marked `relative_to_moving_camera` and are not used to
judge another party's maneuver.
"""
import math

import numpy as np

from app.services import collision

VIDEO_EVENTS_VERSION = "3"
DEFAULTS = {"moving_rel_speed": 0.5, "stopped_rel_speed": 0.15, "stop_min_s": 1.0, "slow_drop": 0.5, "slow_window_s": 0.8,
            "turn_deg": 30.0, "turn_window_s": 1.6, "min_turn_rel_speed": 0.3, "maneuver_min_path_diag": 3.0,
            "interaction_iou": 0.05, "interaction_gap_diag": 0.08, "near_miss_gap_diag": 0.5, "collision_drop": 0.6,
            "collision_lookahead_s": 0.8, "cue_window_s": 0.5, "cue_min_conf": 0.4, "entry_margin_s": 0.5,
            "segment_gap_factor": 1.5, "common_slowdown_share": 0.5}


def _diag(b) -> float:
    return math.hypot(b[2] - b[0], b[3] - b[1])


def _speeds(points: list[dict]) -> list[float]:
    """Smoothed relative speed (diagonals per second) at each point; first point copies the second."""
    raw = []
    for a, b in zip(points, points[1:]):
        dt = (b["time_ms"] - a["time_ms"]) / 1000
        d = math.hypot(b["anchor"][0] - a["anchor"][0], b["anchor"][1] - a["anchor"][1]) / max(_diag(a["bbox"]), 1e-6)
        raw.append(d / dt if dt > 0 else 0.0)
    if not raw:
        return [0.0] * len(points)
    sm = [float(np.mean(raw[max(0, i - 1):i + 2])) for i in range(len(raw))]  # 3-point mean
    return [sm[0]] + sm


def _heading(points, i, span) -> float | None:
    a = points[max(0, i - span)]
    b = points[min(len(points) - 1, i + span)]
    dx, dy = b["anchor"][0] - a["anchor"][0], b["anchor"][1] - a["anchor"][1]
    return None if math.hypot(dx, dy) < 1e-6 else math.atan2(dy, dx)


def _signed_turn(h1: float, h2: float) -> float:
    """Degrees, positive = clockwise on screen (image y points down) = a right turn for the vehicle."""
    return math.degrees((h2 - h1 + math.pi) % (2 * math.pi) - math.pi)


def track_maneuver(points: list[dict], diag: float, p: dict) -> dict | None:
    if len(points) < 6:
        return None
    path = sum(math.hypot(b["anchor"][0] - a["anchor"][0], b["anchor"][1] - a["anchor"][1]) for a, b in zip(points, points[1:]))
    if path < p["maneuver_min_path_diag"] * diag:
        return None
    # headings are read from the moving part only: a vehicle that comes to rest has identical trailing points (no direction)
    step = [math.hypot(b["anchor"][0] - a["anchor"][0], b["anchor"][1] - a["anchor"][1]) for a, b in zip(points, points[1:])]
    moving = [i for i, d in enumerate(step) if d >= 0.02 * diag]
    if len(moving) < 4:
        return None
    points = points[moving[0]:moving[-1] + 2]
    k = max(2, len(points) // 4)
    h1 = _heading(points, k // 2, k // 2)
    h2 = _heading(points, len(points) - 1 - k // 2, k // 2)
    if h1 is None or h2 is None:
        return None
    change = _signed_turn(h1, h2)
    kind = "straight" if abs(change) < p["turn_deg"] else ("right" if change > 0 else "left")
    return {"maneuver": kind, "net_heading_change_deg": round(change, 1), "path_length_diagonals": round(path / diag, 1)}


def _end_note(track: dict) -> tuple[str, str]:
    """Why a track stopped. A gap is a missed detection. The image border is a possible exit. Neither is fault."""
    c = track.get("continuity") or {}
    if c.get("gaps"):
        return "missed_detection", "missed detection: sampled frames are missing inside the track, so the id may have been lost rather than the object leaving"
    if c.get("ends_at_frame_border"):
        return "frame_border", "track ends at the image border; the object may have left the frame. This is not a confirmed exit"
    if c:
        return "missed_detection", "track ended in open frame with no gap recorded; the detector stopped reporting this id (possible missed detection, not a confirmed exit)"
    return "track_ended", "track ended; the object may have been lost rather than left"


def _relation(max_iou: float, min_gap: float, p: dict) -> str:
    """Overlap of the boxes is not the same thing as a small gap between them, and neither is physical contact."""
    if max_iou >= p["interaction_iou"] or min_gap <= p["interaction_gap_diag"]:
        return "box_overlap"
    return "proximity"


def _collapse_overlapping(events: list[dict], join_ms: float) -> list[dict]:
    """One event per type and track set when the time spans meet. Separate types (slowing, stopped) stay separate."""
    groups: list[dict] = []
    for e in events:
        key = (e["event_type"], tuple(sorted(e["observations"].get("track_ids") or [])))
        hit = None
        for g in groups:
            if g["key"] == key and e["t_start_ms"] <= g["t_end"] + join_ms and g["t_start"] <= e["t_end_ms"] + join_ms:
                hit = g
                break
        if hit is None:
            groups.append({"key": key, "event": e, "t_start": e["t_start_ms"], "t_end": e["t_end_ms"]})
            continue
        hit["t_start"] = min(hit["t_start"], e["t_start_ms"])
        hit["t_end"] = max(hit["t_end"], e["t_end_ms"])
        base, extra = hit["event"], e
        if (extra["observations"].get("max_box_overlap_iou") or 0) > (base["observations"].get("max_box_overlap_iou") or 0):
            base["observations"]["max_box_overlap_iou"] = extra["observations"].get("max_box_overlap_iou")
        if "min_gap_in_diagonals" in extra["observations"]:
            prev = base["observations"].get("min_gap_in_diagonals")
            gap = extra["observations"]["min_gap_in_diagonals"]
            if prev is None or gap < prev:
                base["observations"]["min_gap_in_diagonals"] = gap
        frames = sorted(set((base["source_ref"].get("frames") or []) + (extra["source_ref"].get("frames") or [])))
        times = sorted(set((base["source_ref"].get("frame_times_ms") or []) + (extra["source_ref"].get("frame_times_ms") or [])))
        if frames:
            base["source_ref"]["frames"] = [frames[0], frames[-1]]
        if times:
            base["source_ref"]["frame_times_ms"] = [times[0], times[-1]]
    out = []
    for g in groups:
        e = g["event"]
        e["t_start_ms"], e["t_end_ms"] = g["t_start"], g["t_end"]
        out.append(e)
    return out


def _own_change(tr, spd, t0, p):
    """An abrupt change of the track's own motion after t0: its image speed falls by collision_drop, or its heading turns sharply."""
    pts = tr["points"]
    if spd:
        pre = [s for x, s in zip(pts, spd) if t0 - 1000 <= x["time_ms"] < t0]
        post = [s for x, s in zip(pts, spd) if t0 <= x["time_ms"] <= t0 + 1000]
        if pre and post and np.mean(pre) >= p["moving_rel_speed"] and np.mean(post[len(post) // 2:] or post) <= np.mean(pre) * (1 - p["collision_drop"]):
            return {"track_id": tr["track_id"], "kind": "speed_drop", "before": round(float(np.mean(pre)), 2), "after": round(float(np.mean(post[len(post) // 2:] or post)), 2)}
    before = [x for x in pts if t0 - 1000 <= x["time_ms"] <= t0]
    after = [x for x in pts if t0 <= x["time_ms"] <= t0 + 1000]
    if len(before) >= 3 and len(after) >= 3:
        h1 = math.atan2(before[-1]["anchor"][1] - before[0]["anchor"][1], before[-1]["anchor"][0] - before[0]["anchor"][0])
        h2 = math.atan2(after[-1]["anchor"][1] - after[0]["anchor"][1], after[-1]["anchor"][0] - after[0]["anchor"][0])
        moved = math.hypot(before[-1]["anchor"][0] - before[0]["anchor"][0], before[-1]["anchor"][1] - before[0]["anchor"][1]) / max(_diag(before[0]["bbox"]), 1e-6)
        turn = abs(_signed_turn(h1, h2))
        if moved >= 0.3 and turn >= 45:
            return {"track_id": tr["track_id"], "kind": "heading_change", "degrees": round(float(turn), 1)}
    return None


def _common_slowdown(tracks, sp_cache, skip, t0, p):
    """Share of the other moving tracks whose image speed falls by collision_drop around t0, and how many were moving."""
    moving = dropped = 0
    for tr in tracks:
        if tr["track_id"] in skip:
            continue
        spd = sp_cache.get(tr["track_id"])
        if not spd:
            continue
        pts = tr["points"]
        pre = [s for x, s in zip(pts, spd) if t0 - 1000 <= x["time_ms"] < t0]
        post = [s for x, s in zip(pts, spd) if t0 + 200 <= x["time_ms"] <= t0 + 1000]
        if len(pre) >= 2 and post and np.mean(pre) >= p["moving_rel_speed"]:
            moving += 1
            dropped += int(np.mean(post) <= np.mean(pre) * (1 - p["collision_drop"]))
    return (dropped / moving if moving else 0.0), moving


def generate(tracks: list[dict], *, kind: str, period_ms: float, video_duration_ms: float, cues: list[tuple[float, float]] | None = None,
             cue_boxes: list | None = None, overrides: dict | None = None) -> list[dict]:
    """tracks: [{track_id, class, points:[{frame_no,time_ms,bbox,anchor,confidence}]}] -> candidate events (source media time)."""
    p = DEFAULTS | (overrides or {})
    moving_cam = kind == "dashcam"
    out: list[dict] = []
    cues = [c for c in (cues or []) if c[1] >= p["cue_min_conf"]]
    sp_cache: dict[int, list[float]] = {}

    def add(etype, rule, cat, a_pt, b_pt, track_ids, obs, tracks_involved):
        obs = dict(obs) | {"track_ids": track_ids, "coordinate_frame": "image_pixels", "relative_to_moving_camera": moving_cam}
        out.append({"event_type": etype, "rule_id": rule, "category": cat, "t_start_ms": a_pt["time_ms"], "t_end_ms": b_pt["time_ms"],
                    "uncertainty_ms": period_ms / 2, "observations": obs,
                    "source_ref": {"kind": "video_frames", "frames": [a_pt["frame_no"], b_pt["frame_no"]], "frame_times_ms": [a_pt["time_ms"], b_pt["time_ms"]]},
                    "entity_hint": {"tracks": [{"track_id": t["track_id"], "class": t["class"]} for t in tracks_involved]}})

    for t in tracks:
        pts = t["points"]
        if len(pts) < 3:
            continue
        diag = float(np.median([_diag(x["bbox"]) for x in pts]))
        sp = sp_cache[t["track_id"]] = _speeds(pts)
        tid = [t["track_id"]]
        # entering / leaving the scene (a track that exists at the very start/end of the video did not enter/leave during it)
        if pts[0]["time_ms"] > p["entry_margin_s"] * 1000:
            add("vehicle_enters_scene", "VID-ENTER-01", "machine_detection", pts[0], pts[0], tid, {"class": t["class"]}, [t])
        if pts[-1]["time_ms"] < video_duration_ms - p["entry_margin_s"] * 1000:
            reason, note = _end_note(t)
            add("vehicle_leaves_scene", "VID-LEAVE-01", "machine_detection", pts[-1], pts[-1], tid,
                {"class": t["class"], "end_reason": reason, "note": note}, [t])
        # slowing: mean speed after a window is far below the mean before it, starting from real movement
        w = max(2, int(round(p["slow_window_s"] * 1000 / period_ms)))
        i, last_end = w, -1
        while i + w <= len(pts) - 1:
            before, after = float(np.mean(sp[i - w:i])), float(np.mean(sp[i:i + w]))
            if before >= p["moving_rel_speed"] and after <= before * (1 - p["slow_drop"]) and i > last_end:
                add("vehicle_slowing", "VID-SLOW-01", "machine_detection", pts[i - w], pts[i + w - 1], tid,
                    {"motion": "slowing", "relative_speed_before": round(before, 2), "relative_speed_after": round(after, 2),
                     "drop": round(1 - after / before, 2), "unit": "box diagonals per second",
                     "note": "speed dropped and the track is still moving; this is slowing, not a stop"}, [t])
                last_end = i + w
                i += w
            else:
                i += 1
        # stopped
        run = None
        for k, s in enumerate(sp + [p["moving_rel_speed"] * 10]):
            slow = s <= p["stopped_rel_speed"] and k < len(pts)
            if slow and run is None and max(sp[:k] or [0]) >= p["moving_rel_speed"]:
                run = k
            elif not slow and run is not None:
                if (pts[k - 1]["time_ms"] - pts[run]["time_ms"]) / 1000 >= p["stop_min_s"]:
                    add("vehicle_stopped", "VID-STOP-01", "machine_detection", pts[run], pts[k - 1], tid,
                        {"motion": "stopped", "relative_speed": round(float(np.mean(sp[run:k])), 3),
                         "note": "speed stayed near zero after the object had been moving; this is a stop, not only slowing"}, [t])
                run = None
        # direction change
        span = max(1, int(round(p["turn_window_s"] * 1000 / period_ms / 2)))
        ep = None
        for k in range(span, len(pts) - span):
            h1, h2 = _heading(pts, k - span // 2, span), _heading(pts, k + span // 2, span)
            ok = h1 is not None and h2 is not None and sp[k] >= p["min_turn_rel_speed"]
            d = _signed_turn(h1, h2) if ok else 0.0
            if abs(d) >= p["turn_deg"]:
                if ep is None or abs(d) > abs(ep[2]):
                    ep = (ep[0] if ep else k - span, k + span, d)
            elif ep is not None and abs(d) < p["turn_deg"] / 2:
                a, b, dd = ep
                add("vehicle_direction_change", "VID-TURN-01", "machine_detection", pts[max(a, 0)], pts[min(b, len(pts) - 1)], tid,
                    {"heading_change_deg": round(dd, 1), "turn_sense": "right" if dd > 0 else "left"}, [t])
                ep = None
        if ep is not None:
            a, b, dd = ep
            add("vehicle_direction_change", "VID-TURN-01", "machine_detection", pts[max(a, 0)], pts[min(b, len(pts) - 1)], tid,
                {"heading_change_deg": round(dd, 1), "turn_sense": "right" if dd > 0 else "left"}, [t])
        # overall maneuver of the track (fixed cameras only: on a moving camera it describes relative image motion)
        if not moving_cam and t["class"] != "pedestrian":
            m = track_maneuver(pts, diag, p)
            if m:
                add("vehicle_maneuver_observed", "VID-MANEUVER-01", "machine_detection", pts[0], pts[-1], tid, m, [t])

    # interactions between tracks of the same camera (box overlap/proximity in the image)
    for ai in range(len(tracks)):
        for bi in range(ai + 1, len(tracks)):
            a, b = tracks[ai], tracks[bi]
            fa = {x["frame_no"]: x for x in a["points"]}
            common = [f for f in sorted(fa) if f in {x["frame_no"] for x in b["points"]}]
            if not common:
                continue
            fb = {x["frame_no"]: x for x in b["points"]}
            near, flags = [], []
            for f in common:
                ba, bb = fa[f]["bbox"], fb[f]["bbox"]
                ix = max(0, min(ba[2], bb[2]) - max(ba[0], bb[0]))
                iy = max(0, min(ba[3], bb[3]) - max(ba[1], bb[1]))
                inter = ix * iy
                union = (ba[2] - ba[0]) * (ba[3] - ba[1]) + (bb[2] - bb[0]) * (bb[3] - bb[1]) - inter
                iou_v = inter / union if union > 0 else 0.0
                gx = max(0, max(ba[0], bb[0]) - min(ba[2], bb[2]))
                gy = max(0, max(ba[1], bb[1]) - min(ba[3], bb[3]))
                gap = math.hypot(gx, gy) / max(_diag(ba), _diag(bb))  # scaled by the larger object (a pedestrian beside a car is judged by the car's size)
                near.append((f, iou_v, gap))
                flags.append(iou_v >= p["interaction_iou"] or gap <= p["near_miss_gap_diag"])
            k = 0
            while k < len(common):
                if not flags[k]:
                    k += 1
                    continue
                j = k
                join_ms = period_ms * p["segment_gap_factor"]
                while j + 1 < len(common) and flags[j + 1] and fa[common[j + 1]]["time_ms"] - fa[common[j]]["time_ms"] <= join_ms:
                    j += 1
                seg = near[k:j + 1]
                pa0, pa1 = fa[seg[0][0]], fa[seg[-1][0]]
                max_iou = max(s[1] for s in seg)
                min_gap = min(s[2] for s in seg)
                touching = max_iou >= p["interaction_iou"] or min_gap <= p["interaction_gap_diag"]
                tids = [a["track_id"], b["track_id"]]
                relation = _relation(max_iou, min_gap, p)
                obs = {"relation": relation, "max_box_overlap_iou": round(max_iou, 3), "min_gap_in_diagonals": round(min_gap, 3),
                       "note": "image-space proximity or box overlap is not physical contact or distance"}
                add("potential_interaction", "VID-INTER-01", "machine_detection", pa0, pa1, tids, obs, [a, b])
                # a collision needs more than overlap: approach, arrest of the relative motion, staying together, shape change or a
                # persistent cue (see app.services.collision). Overlap plus one track slowing is ordinary traffic and is not flagged.
                # the contact moment is where the boxes actually touch, not where a wider proximity segment starts: score every touching
                # frame as a candidate contact and keep the best supported one
                touch = [fr for fr, iou_v, gap in seg if iou_v >= p["interaction_iou"] or gap <= p["interaction_gap_diag"]]
                best = None
                for fr in touch[:80]:
                    t_c = fa[fr]["time_ms"]
                    cand = collision.assess(fa, fb, t_c, min(pa1["time_ms"], t_c + 1000), cue_boxes, p)
                    key = (collision.is_collision(cand["signals"]), cand.get("score", 0), -t_c)
                    if best is None or key > best[0]:
                        best = (key, cand, fr)
                ev = best[1] if best else collision.assess(fa, fb, pa0["time_ms"], pa1["time_ms"], cue_boxes, p)
                if best:
                    # if most other tracks slow down in the same moment (the camera's own vehicle braking, a queue stopping), the pair's
                    # slowdown says nothing about the pair: discount its arrest
                    share, n_other = _common_slowdown(tracks, sp_cache, (a["track_id"], b["track_id"]), fa[best[2]]["time_ms"], p)
                    ev["details"]["common_slowdown"] = {"share": round(float(share), 2), "other_tracks": int(n_other)}
                    if n_other >= 3 and share >= p["common_slowdown_share"] and "arrest" in ev["signals"]:
                        ev["signals"] = [x for x in ev["signals"] if x != "arrest"]
                        ev["abrupt"] = None
                    # relative motion that merely converges (one car merging into another's lane, following at the same pace) is not an
                    # impact: at least one of the two must change its own motion abruptly
                    own = [_own_change(tr, sp_cache.get(tr["track_id"]), fa[best[2]]["time_ms"], p) for tr in (a, b)]
                    ev["details"]["own_motion_change"] = [o for o in own if o]
                    if "arrest" in ev["signals"] and not any(o and (o["kind"] == "speed_drop" or "shape" in ev["signals"]) for o in own):
                        ev["signals"] = [x for x in ev["signals"] if x != "arrest"]
                        ev["abrupt"] = None
                if best:
                    pa0 = fa[best[2]]
                hit = touching and collision.is_collision(ev["signals"])
                if p.get("_debug") is not None:
                    p["_debug"].append({"tracks": [a["track_id"], b["track_id"]], "t0": pa0["time_ms"], "t1": pa1["time_ms"], "touching": touching, "max_iou": max_iou, "hit": hit, "ev": ev})
                drops = []
                if ev["abrupt"]:
                    drops = [{"track_id": a["track_id"], "paired_with": b["track_id"], "basis": "relative motion of the pair"} | ev["abrupt"]]
                evidence = {"collision_signals": ev["signals"], "collision_score": ev.get("score", 0), "collision_details": ev["details"]}
                cue_hits = [c for c in cues if pa0["time_ms"] - p["cue_window_s"] * 1000 <= c[0] <= pa1["time_ms"] + p["cue_window_s"] * 1000]
                if hit:
                    add("possible_collision", "VID-COLL-02", "machine_detection", pa0, pa1, tids,
                        obs | evidence | {"relation": "box_overlap", "abrupt_speed_changes": drops, "accident_cue_nearby": bool(cue_hits),
                                          "accident_cue_note": "the accident-cue model is a weak supporting signal and never decides this event"}, [a, b])
                elif not touching and "approach" in ev["signals"] and "arrest" in ev["signals"]:
                    add("possible_near_miss", "VID-NEAR-02", "machine_detection", pa0, pa1, tids,
                        obs | evidence | {"relation": "proximity", "abrupt_speed_changes": drops, "accident_cue_nearby": bool(cue_hits),
                                          "accident_cue_note": "the accident-cue model is a weak supporting signal and never decides this event"}, [a, b])
                k = j + 1
    return _collapse_overlapping(sorted(out, key=lambda e: (e["t_start_ms"], e["event_type"])), period_ms * p["segment_gap_factor"])
