"""Phase 9 (video side): rules over synthetic tracks. Pure, no models involved."""
import math

import pytest

from app.services import video_events as V

PERIOD = 200  # 5 fps


def track(tid, cls, fn, n, start=0, w=60, h=40):
    pts = []
    for k in range(n):
        x, y = fn(k)
        pts.append({"frame_no": start + k, "time_ms": (start + k) * PERIOD, "bbox": [x - w / 2, y - h, x + w / 2, y], "anchor": [x, y], "confidence": 0.9})
    return {"track_id": tid, "class": cls, "points": pts}


def gen(tracks, kind="cctv", dur=20_000, **kw):
    return V.generate(tracks, kind=kind, period_ms=PERIOD, video_duration_ms=dur, **kw)


def of(events, etype):
    return [e for e in events if e["event_type"] == etype]


def test_straight_track_has_one_straight_maneuver_and_nothing_else():
    ev = gen([track(1, "car", lambda k: (50 + 25 * k, 200), 40)], dur=8000)
    assert [e["event_type"] for e in ev] == ["vehicle_maneuver_observed"]
    m = ev[0]["observations"]
    assert m["maneuver"] == "straight" and abs(m["net_heading_change_deg"]) < 5 and m["coordinate_frame"] == "image_pixels"
    assert ev[0]["source_ref"]["frames"] == [0, 39] and ev[0]["uncertainty_ms"] == PERIOD / 2


def left_turn(k):
    # drives east then turns to travel north (up the screen) = a LEFT turn for the vehicle
    return (50 + 25 * k, 200) if k < 20 else (50 + 25 * 20, 200 - 25 * (k - 20))


def right_turn(k):
    return (50 + 25 * k, 200) if k < 20 else (50 + 25 * 20, 200 + 25 * (k - 20))


@pytest.mark.parametrize("fn,sense", [(left_turn, "left"), (right_turn, "right")])
def test_turn_sense_matches_the_vehicles_own_left_right(fn, sense):
    ev = gen([track(1, "car", fn, 40)], dur=8000)
    m = of(ev, "vehicle_maneuver_observed")[0]["observations"]
    assert m["maneuver"] == sense and abs(abs(m["net_heading_change_deg"]) - 90) < 15
    turns = of(ev, "vehicle_direction_change")
    assert len(turns) == 1 and turns[0]["observations"]["turn_sense"] == sense and abs(turns[0]["observations"]["heading_change_deg"]) >= 30


def test_moving_camera_never_reports_a_maneuver():
    ev = gen([track(1, "car", left_turn, 40)], kind="dashcam", dur=8000)
    assert of(ev, "vehicle_maneuver_observed") == []
    assert all(e["observations"]["relative_to_moving_camera"] for e in ev)


def test_slowing_then_stopping():
    def fn(k):
        x = 0.0
        for i in range(k):
            x += 25 if i < 15 else max(0.0, 25 - 6 * (i - 14))
        return (50 + x, 200)

    ev = gen([track(1, "car", fn, 40)], dur=8000)
    slow, stop = of(ev, "vehicle_slowing"), of(ev, "vehicle_stopped")
    assert len(slow) == 1 and slow[0]["observations"]["drop"] >= 0.5 and slow[0]["rule_id"] == "VID-SLOW-01"
    assert len(stop) == 1 and stop[0]["t_start_ms"] >= slow[0]["t_start_ms"] and stop[0]["t_end_ms"] - stop[0]["t_start_ms"] >= 1000


def test_constant_speed_and_a_parked_object_produce_no_slowing_or_stop():
    ev = gen([track(1, "car", lambda k: (50 + 25 * k, 200), 40), track(2, "car", lambda k: (300, 100), 40)], dur=8000)
    assert of(ev, "vehicle_slowing") == [] and of(ev, "vehicle_stopped") == []  # never moved => not a "stop"


def test_entry_exit_only_when_not_present_at_video_edges():
    t_mid = track(1, "car", lambda k: (50 + 25 * k, 200), 20, start=10)  # appears at 2 s, gone at 6 s of a 20 s video
    ev = gen([t_mid])
    assert [e["event_type"] for e in ev if e["event_type"].startswith("vehicle_e") or e["event_type"].startswith("vehicle_l")] == ["vehicle_enters_scene", "vehicle_leaves_scene"]
    assert gen([track(1, "car", lambda k: (50 + 25 * k, 200), 100)], dur=100 * PERIOD - PERIOD) == gen([track(1, "car", lambda k: (50 + 25 * k, 200), 100)], dur=99 * PERIOD)


def crossing_pair():
    """A drives east along y=200 and reaches x=350 at k=10; B drives south along x=350 and reaches y=200 at k=10: they meet there."""
    a = track(1, "car", lambda k: (50 + 30 * k, 200), 40)
    b = track(2, "car", lambda k: (350, 30 * k - 100), 40)
    return a, b


def stop_dead_at_contact(a, b):
    for tr in (a, b):  # both come to rest at the contact point
        tr["points"] = tr["points"][:11] + [dict(tr["points"][10], frame_no=11 + i, time_ms=(11 + i) * PERIOD) for i in range(20)]


def test_collision_candidate_needs_overlap_and_an_abrupt_change():
    a, b = crossing_pair()
    stop_dead_at_contact(a, b)
    ev = gen([a, b], dur=8000)
    inter = of(ev, "potential_interaction")
    coll = of(ev, "possible_collision")
    assert len(inter) == 1 and len(coll) == 1 and of(ev, "possible_near_miss") == []
    o = coll[0]["observations"]
    assert o["track_ids"] == [1, 2] and o["abrupt_speed_changes"] and o["accident_cue_nearby"] is False
    assert "never decides" in o["accident_cue_note"] and "not physical contact" in o["note"]
    assert coll[0]["event_type"] == "possible_collision"  # never plain 'collision'


def test_overlap_without_any_abrupt_change_is_only_an_interaction():
    a = track(1, "car", lambda k: (50 + 30 * k, 200), 30)
    b = track(2, "car", lambda k: (350, 30 * k - 100) if k < 10 else (350 + 30 * (k - 10), 200), 30)  # B joins A's lane and both continue at the same speed
    ev = gen([a, b], dur=8000)
    assert of(ev, "potential_interaction")
    assert of(ev, "possible_collision") == [] and of(ev, "possible_near_miss") == []


def test_accident_cue_is_recorded_but_cannot_create_an_event():
    a = track(1, "car", lambda k: (50 + 30 * k, 200), 30)
    b = track(2, "car", lambda k: (900, 900), 30)
    assert gen([a, b], cues=[(2000.0, 0.9)]) and not of(gen([a, b], cues=[(2000.0, 0.99)]), "possible_collision")


def test_cue_near_a_collision_candidate_is_noted():
    a, b = crossing_pair()
    stop_dead_at_contact(a, b)
    c = of(gen([a, b], dur=8000, cues=[(2100.0, 0.7), (7000.0, 0.99)]), "possible_collision")[0]
    assert c["observations"]["accident_cue_nearby"] is True
    assert of(gen([a, b], dur=8000, cues=[(2100.0, 0.2)]), "possible_collision")[0]["observations"]["accident_cue_nearby"] is False  # below cue confidence


def test_pedestrian_near_miss_has_no_overlap_but_one_party_stops():
    car = track(1, "car", lambda k: (50 + 30 * k, 200) if k < 12 else (50 + 30 * 12, 200), 40, w=100, h=60)
    ped = track(2, "pedestrian", lambda k: (490, 200), 40, w=20, h=50)
    # the car's front edge halts at x=460, 20 px short of the pedestrian's box (480..500): close, but no overlap
    ev = gen([car, ped], dur=8000)
    near = of(ev, "possible_near_miss")
    assert len(near) == 1 and near[0]["observations"]["max_box_overlap_iou"] == 0.0 and of(ev, "possible_collision") == []
    assert "not physical contact or distance" in near[0]["observations"]["note"]


def test_short_tracks_are_ignored():
    assert gen([track(1, "car", lambda k: (50 + 25 * k, 200), 2)]) == []


def sparse(tid, fn, n, step=5, w=60, h=40, cls="car"):
    """Sampled frames whose frame numbers are several apart, as a 5 fps sample of a faster video is."""
    pts = []
    for k in range(n):
        x, y = fn(k)
        pts.append({"frame_no": k * step, "time_ms": k * PERIOD, "bbox": [x - w / 2, y - h, x + w / 2, y], "anchor": [x, y], "confidence": 0.9})
    return {"track_id": tid, "class": cls, "points": pts}


def test_sampled_frame_gaps_join_into_one_proximity_segment():
    a = sparse(1, lambda k: (100 + 10 * k, 200), 8)
    b = sparse(2, lambda k: (140 + 10 * k, 200), 8)  # boxes overlap the whole way; frame numbers jump by 5
    ev = gen([a, b], dur=8000)
    assert len(of(ev, "potential_interaction")) == 1
    assert of(ev, "possible_collision") == [] and of(ev, "possible_near_miss") == []
    assert of(ev, "potential_interaction")[0]["observations"]["relation"] == "box_overlap"


def test_a_real_time_gap_stays_two_segments_and_the_gap_factor_is_configurable():
    def burst(tid, k0, n):
        pts = []
        for k in range(n):
            i = k0 + k
            x = 100 + 5 * i
            pts.append({"frame_no": i * 5, "time_ms": i * PERIOD, "bbox": [x - 30, 160, x + 30, 200], "anchor": [x, 200], "confidence": 0.9})
        return {"track_id": tid, "class": "car", "points": pts}

    def pair(k0):
        return burst(1, k0, 4), burst(2, k0, 4)

    a1, b1 = pair(0)
    a2, b2 = pair(20)  # 3.2 s later, far beyond one sample period
    a = {"track_id": 1, "class": "car", "points": a1["points"] + a2["points"]}
    b = {"track_id": 2, "class": "car", "points": b1["points"] + b2["points"]}
    assert len(of(gen([a, b], dur=20000), "potential_interaction")) == 2
    tight = gen([sparse(1, lambda k: (100, 200), 6), sparse(2, lambda k: (130, 200), 6)], dur=8000, overrides={"segment_gap_factor": 0.1})
    assert len(of(tight, "potential_interaction")) == 6  # samples 200 ms apart no longer join


def test_sparse_overlap_and_a_stop_is_still_one_collision_candidate():
    a, b = crossing_pair()
    stop_dead_at_contact(a, b)
    for tr in (a, b):
        tr["points"] = [{**p, "frame_no": p["frame_no"] * 5} for p in tr["points"]]
    ev = gen([a, b], dur=8000)
    assert len(of(ev, "potential_interaction")) == 1 and len(of(ev, "possible_collision")) == 1
    assert of(ev, "possible_collision")[0]["observations"]["relation"] == "box_overlap"


def test_slowing_is_not_a_stop_and_a_stop_is_not_only_slowing():
    def fn(k):
        x = 0.0
        for i in range(k):
            x += 25 if i < 15 else max(0.0, 25 - 6 * (i - 14))
        return (50 + x, 200)

    ev = gen([track(1, "car", fn, 40)], dur=8000)
    slow, stop = of(ev, "vehicle_slowing")[0], of(ev, "vehicle_stopped")[0]
    assert slow["observations"]["motion"] == "slowing" and "not a stop" in slow["observations"]["note"]
    assert stop["observations"]["motion"] == "stopped" and "not only slowing" in stop["observations"]["note"]
    assert slow["rule_id"] != stop["rule_id"]


def test_leave_reason_distinguishes_a_missed_detection_from_the_frame_border():
    base = track(1, "car", lambda k: (50 + 25 * k, 200), 20, start=10)
    gapped = dict(base, continuity={"gaps": [{"after_frame": 12, "missing_sampled_frames": 2}], "ends_at_frame_border": False})
    bordered = dict(base, continuity={"gaps": [], "ends_at_frame_border": True})
    lost = dict(base, continuity={"gaps": [], "ends_at_frame_border": False})
    assert of(gen([gapped]), "vehicle_leaves_scene")[0]["observations"]["end_reason"] == "missed_detection"
    assert of(gen([bordered]), "vehicle_leaves_scene")[0]["observations"]["end_reason"] == "frame_border"
    assert of(gen([lost]), "vehicle_leaves_scene")[0]["observations"]["end_reason"] == "missed_detection"
    assert "not a confirmed exit" in of(gen([bordered]), "vehicle_leaves_scene")[0]["observations"]["note"]


def test_confident_sustained_scene_cue_raises_a_possible_accident_without_tracks():
    cue = [(t, [10, 10, 200, 150], c, i) for i, (t, c) in enumerate([(1000, 0.8), (1200, 0.9), (1400, 0.88), (1600, 0.7)])]
    ev = gen([track(1, "car", lambda k: (50 + 25 * k, 200), 40)], dur=8000, cue_boxes=cue, overrides={"scene_cue_min_peak": 0.75})
    scene = [e for e in of(ev, "possible_collision") if e["rule_id"] == "VID-SCENE-01"]
    assert len(scene) == 1
    o = scene[0]["observations"]
    assert scene[0]["t_start_ms"] == 1000 and o["scene_cue"]["strength"] == "strong" and o["track_ids"] == [] and o["motion_analysis_agrees"] is False
    assert "does not show" in o["note"]


def test_weak_or_brief_scene_cues_raise_nothing():
    ov = {"scene_cue_min_peak": 0.75}
    brief = [(1000, [0, 0, 9, 9], 0.9, 0), (1200, [0, 0, 9, 9], 0.9, 1)]
    weak = [(t, [0, 0, 9, 9], 0.6, i) for i, t in enumerate(range(1000, 2000, 200))]
    for cues in (brief, weak):
        assert [e for e in gen([], dur=8000, cue_boxes=cues, overrides=ov) if e["rule_id"] == "VID-SCENE-01"] == []
    # and without the DETR switch (the weak YOLO cue) a cue never creates an event by itself
    assert [e for e in gen([], dur=8000, cue_boxes=[(t, [0, 0, 9, 9], 0.99, i) for i, t in enumerate(range(1000, 2000, 200))]) if e["rule_id"] == "VID-SCENE-01"] == []
