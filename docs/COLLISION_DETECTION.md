# Collision detection (video rules v3)

**Problem.** Rules v2 raised `possible_collision` whenever two boxes overlapped by 5% IoU (or sat within 8% of a box diagonal) and either
track's image speed later fell by 60%. In dense traffic and on any moving camera that is almost always true: one car occludes another, and
when the camera's own vehicle brakes every track "slows down". A highway dashcam clip produced 29 collision events.

**Now** (`backend/app/services/collision.py`, wired in `video_events.generate`). A possible collision needs contact *and* evidence of an
impact between the two bodies, measured on the pair's relative motion (units of the larger box width, so camera translation largely
cancels):

| signal | meaning |
|---|---|
| approach | the pair closed by >= 0.5 widths at >= 0.5 widths/s in the second before contact, mostly monotonic |
| arrest | relative speed fell to <= 40% right after contact (or was deflected >= 70 deg) |
| rest | the pair stayed within 1.3 widths afterwards |
| shape | one box's aspect ratio changed >= 40% (flip, roll, fallen rider) |
| cue | the weak accident-cue model fired on the pair in >= 3 frames |

Collision = approach + arrest + one of (shape, cue, rest), or cue + one kinematic signal. Then three guards:
- the **contact moment** is the best-supported touching frame, not the start of a proximity segment;
- relative speeds above 3.5 widths/s are tracking jumps, not motion;
- if >= 50% of the other moving tracks slow down at the same time (ego braking, a queue stopping), the pair's arrest is discounted;
- arrest must come with an abrupt change of a vehicle's *own* motion (speed drop, or a heading change together with a shape change);
  one car merging into another's lane and matching speed is not an impact.

Every event carries `collision_signals`, `collision_score` and `collision_details`; the UI lists the reasons. The cue model still never
decides an event on its own, and nothing here assigns fault.

## Evaluation (development set, read this before trusting it)
`python scripts/eval_collision_rules.py` replays stored tracks of six real clips against collision windows the author labelled by eye.

| | rules v2 | rules v3 |
|---|---|---|
| labelled collision windows detected | 3 / 6 | 6 / 6 |
| flagged groups that were true / false | 3 / 3 | 7 / 2 |

The rules were tuned on these same six clips, so the numbers are optimistic. They are not a benchmark. Known remaining false flags: a
stationary far-away pair in a queued-traffic clip and one close pass at a junction. Not handled: a collision involving the camera's own
vehicle with no tracked partner (needs global image motion), and tracks lost at impact (cctv 3 is detected only because both tracks survived).

## To make it properly accurate
Hand-tuned thresholds cannot be validated on six clips. The signals above are the feature set for a learned classifier; training it needs
labelled collision/non-collision pairs (e.g. the ACCIDENT, CADP or DoTA datasets) and a held-out split of *videos*, which is the
Colab work on branch `colab-ml-model`.

Existing cases keep the events stored by v2 until their analysis is re-run (`VIDEO_EVENTS_VERSION` is now "3").
