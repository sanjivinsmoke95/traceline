"""Collision evidence: overlap, close following and one car slowing are ordinary traffic; an approach that is arrested is not."""
from app.services import collision
from tests.test_video_events import PERIOD, gen, of, track


def run(a, b):
    dbg = []
    events = gen([a, b], overrides={"_debug": dbg})
    return events, dbg


def test_following_in_a_queue_is_not_a_collision():
    # two cars close together, one behind the other, both creeping along at the same pace and slowing together
    a = track(1, "car", lambda k: (100 + 6 * k, 200), 40, w=80, h=50)
    b = track(2, "car", lambda k: (100 + 6 * k - 70, 215), 40, w=80, h=50)
    events, _ = run(a, b)
    assert of(events, "possible_collision") == []


def test_overlap_while_one_track_slows_alone_is_not_enough():
    # B stops (as if for a red light) next to a moving A that never changes course: they overlap but there was no approach
    a = track(1, "car", lambda k: (50 + 20 * k, 200), 40)
    b = track(2, "car", lambda k: (400, 200 - 3) if k > 5 else (400 + 40 * (5 - k), 197), 40)
    events, _ = run(a, b)
    assert of(events, "possible_collision") == []


def test_approach_then_both_stop_dead_is_flagged_with_its_reasons():
    a = track(1, "car", lambda k: (50 + 12 * min(k, 14), 200), 30)
    b = track(2, "car", lambda k: (330 - 12 * min(k, 14), 200), 30)
    events, dbg = run(a, b)
    hit = of(events, "possible_collision")
    assert hit, dbg
    o = hit[0]["observations"]
    assert {"approach", "arrest"} <= set(o["collision_signals"]) and o["abrupt_speed_changes"]
    assert "never decides" in o["accident_cue_note"]


def test_unphysical_jumps_are_tracking_discontinuities_not_motion():
    s = collision.DEFAULTS["max_plausible_speed"]
    a = track(1, "car", lambda k: (50 + 400 * k, 200), 12)
    b = track(2, "car", lambda k: (900 - 400 * k, 200), 12)
    events, dbg = run(a, b)
    assert of(events, "possible_collision") == [] and s > 0


def test_features_are_a_fixed_length_vector_with_minus_one_for_unknown():
    _, dbg = run(track(1, "car", lambda k: (50 + 12 * min(k, 14), 200), 30), track(2, "car", lambda k: (330 - 12 * min(k, 14), 200), 30))
    v = collision.features(dbg[0]["ev"], dbg[0]["max_iou"])
    assert len(v) == len(collision.FEATURES) and all(isinstance(x, float) for x in v)
    assert collision.features({}) [:3] == [-1.0, -1.0, -1.0]
