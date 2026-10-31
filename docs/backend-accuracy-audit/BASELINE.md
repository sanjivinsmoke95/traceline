---
cursor:
  subagentId: "bc-12778e8b-a568-56e6-8762-e31d12c18c74"
---

# Backend accuracy audit: BASELINE (before any code change)

Branch `feat/backend-accuracy-audit`, cut from `feat/traceline-investigation-flow-v2` at `2288949`. Written before the first behavioural change. Everything below was either read from code, run, or measured on the licensed real clips; each item says which. "Passing unit tests" is not accuracy: the existing suite proves invariants (traceability, wording, schema), and none of it measures how often the pipeline is right on real footage.

## 1. Environment actually used

| Item | State |
|---|---|
| Python / venv | 3.12, `virtualenv` (ensurepip missing); `requirements.txt` + `requirements-vision.txt` (torch 2.15 CPU, ultralytics 8.4.193, trackers 2.6.1) |
| PostgreSQL 16 | installed locally for the run (apt); `traceline` + `traceline_test` DBs; `.env` is git-ignored |
| Detector weights | `yolo26s`, `yolo26n`, `accident_yolov8s` fetched with `scripts/fetch_models.py --include-accident`, hash-pinned, git-ignored |
| GPU | none (CPU, 4 cores) |
| Featherless API key | **not available**. Every live-LLM path is exercised only against a simulated HTTP transport |
| Accident-model validation split | not installed (1 test skips) |

## 2. Existing test suite, before changes

| Run | Result |
|---|---|
| No weights, no scenario packages | 241 passed, 29 skipped |
| With weights (`fetch_models.py`) | 251 passed, 19 skipped |
| With weights and `scripts/build_scenarios.py` | **267 passed, 3 skipped** (2 are frontend-owned browser tests, 1 needs the accident validation split) |

## 3. Pipeline trace and status

Legend: **Works** = behaves as designed and is checked. **Partial** = works with a stated hole. **Unreliable** = measured to be wrong often enough to mislead. **Untested** = no test or measurement. **Missing** = not implemented.

| Stage | Where | Status | Evidence / hole |
|---|---|---|---|
| Upload + validation | `services/ingest.py`, `validators.py`, `filetypes.py` | Works | Content sniffing, hash, duplicate rejection, corrupt files rejected (tests). |
| Video metadata + frame index from container PTS | `services/video.py`, `jobs.handle_preprocess` | Works | `time_ms` = real PTS relative to first frame, `pts_start_ms` kept separately; gap/VFR tests pass. Decoder-vs-container timing cross-checked. |
| Playback copy re-timed to 0 | `video.make_playback` | Works | `timing_preserved` verified per file. |
| Full-resolution still by time | `video.extract_full_frame`, `api/video.py` | **Unreliable (reproduced)** | Stored `time_ms` is rounded to whole ms; `-ss` takes the first frame at or after that time, so when the rounding went up the **next frame** is returned. On a 30 fps clip 20 of 60 frames (every third) came back as the following frame. The thumbnail (sequential decode) is right, the "full" image is not. |
| Detection (YOLO26s) | `services/detection.py`, `jobs.handle_detect` | Partial | Raw boxes + model run (weights sha, params) stored. One storage threshold (0.15) for every lighting condition. Video decoded once **per model** (detector, cue, then BoT-SORT again). No detector precision/recall measured on traffic footage in this repo (no labelled boxes for the real clips). |
| Accident-cue model | `detection.py` | Partial | Weak by design (own-split P 0.586 / R 0.524). Used only as a **time** match: any cue box anywhere in the frame within ±0.5 s counts as "nearby". On `candidate-05` it fires on nearly every sampled frame at conf ~0.7 (151 cue frames) because two crashed cars sit still for the whole clip, i.e. it reports aftermath, not onset. |
| Tracking (ByteTrack / BoT-SORT) | `services/tracking.py` | Partial | Per-video ids, gaps/flags/continuation candidates stored, nothing merged. Car+truck boxes on the same object (IoU > 0.9) are tracked as two objects. No ID-switch / IDF1 measured on any real clip (no tracking labels). |
| Event extraction | `services/video_events.py` v2 | **Unreliable (measured)** | See section 4. |
| Clock sync | `services/sync.py` | Works (synthetic) / Untested (real) | Only declared edges and trusted clock records synchronize; embedded creation time and proposals never do; cycles checked. `propose_anchor_edges` picks its "best" pair by smallest bound, which is not a meaningful criterion. No real multi-camera offset error measured (Accid3nD has per-frame timestamps that could be used). |
| Timeline / seek | `api/reconstruction.py` | Works (API) | Seek and per-source time chain exist (`clocks_for`: media, container PTS, wall clock, case, synchronized). |
| Witness claims, live extraction | `llm/featherless.py`, `services/witness.py`, `jobs.handle_extract_claims`, `api/claims.py` | Partial | Strict schema, verbatim-span location, one repair, ledger, cache, retries. **No key => `POST /statements/{id}/extract` returns 503 `LLM_NOT_CONFIGURED`** and nothing is extracted: there is no deterministic fallback and the manual path is the only alternative. No model-availability check (a wrong model name only fails at call time). |
| Witness claims, verification | `engine_rules.py`, `api/claims.py` | Partial | Claim endpoints always return `corroboration_status: "not_assessed"`. Verification statuses (supported / partially supported / contradicted / not independently verified / insufficient) do not exist as a per-claim field. Claim records have no witness viewpoint, no observation/inference/hearsay split beyond `certainty`, no relative-time anchor type. |
| Correlation / contradictions | `engine_rules.correlate` | Works (synthetic) | Rules COR-BRAKE-01, COR-COLL-01, COR-MANEUVER-01, CON-ACTION-01, CON-QTY-01, CON-COLL-CLAIM-01; independence of sources checked. Weakness: COR-COLL-01 and witness time placement anchor on `possible_collision`, a machine candidate that is wrong most of the time (section 4). |
| Evidence gaps | `engine_rules.find_gaps` | Partial | 10 gap types with advice. No gap for: coverage ends before/starts after the candidate moment, vehicle leaves coverage, witness conflict, aftermath-only footage. `covers()` contains dead logic (`and False`). |
| Hypotheses | `reconstruction_view.hypotheses` | Partial | Only one inference rule (INF-BRAKE-PRIOR-01) produces alternatives. No multi-hypothesis sequences with ordered events, assumptions, missing evidence. |
| Cross-camera association | `api/reconstruction.py` associations | Partial | Created `uncertain`; investigator can propose one. No confirm / reject / manual-link decision, no audit trail of a decision, no temporal-compatibility field. |
| Keyframes / storyboard | `services/storyboard.py` | Partial | before/during/after as references to stored sampled frames; **the time difference between the event and the "before"/"after" frame is not disclosed**; no keyframe type on an event; relies on the full-frame still bug above. |
| Narrative | `services/incident.py` | Works | Deterministic wording gates (banned phrases, "possible collision" never bare, contact never claimed). Built from whatever events exist, so a false alert is narrated as "a possible collision was identified". |
| Report | `services/report.py` | Works | Banned-phrase scan; synthetic marked. |
| Copilot | (none) | **Missing** | No copilot endpoint. `featherless_vision_model` is configured and reported by `/health` but is never called anywhere, so no image content is ever sent to any model. |
| Re-running tracking | `jobs.handle_track` + `reconstruct.build` | **Unreliable (by code; test follows)** | `build()` loads *all* `SourceEvent` rows of the case but only the tracks of the *latest* track run. Track numbers are per-run, so an event from an older run is mapped onto an unrelated track of the newer run, and events are duplicated. |
| Duplicate jobs / repeat processing | `jobs.py` | Partial | Idempotent by (sha256, params hash) for preprocess/detect/track/claims. Two tracking jobs with different params for one video both persist their events. |
| Model reuse | `weights.load_yolo` (`lru_cache`) | Works | Loaded once per process. |

## 4. Event / accident detection, measured on real footage (the highest priority)

Reproducible with `backend/scripts/eval_real_events.py --event-version legacy` (frozen copy of v2 rules in `app/services/video_events_legacy.py`). Same cached raw detections are used before and after.

Labels: `backend/eval_data/real_clips.json`. Positives with a located onset (manual, from decoded frames): `candidate-04` (visible collision, onset 4.75–4.92 s) and `accid3nd_south_50mm` (yellow car hits the shoulder/van at ≈3.80 s; this supersedes the acquisition report's "no collision found"). Five more clips carry only the dataset's clip-level "accident" label with no located impact and are not scored. Ten clips are negatives: 3 Accid3nD views without a visible collision, the 6 nuScenes cameras of a hard yield with no collision, and comma2k19 normal driving. n is tiny, labels are by one annotator, thresholds were not tuned on them at this point.

| Metric (legacy v2 rules, 17 real clips) | Value |
|---|---|
| Collision alerts (`possible_collision`) in scored clips | TP 1, **FP 21**, FN 1 |
| Precision | **0.045** (1/22; Wilson 95% 0.008–0.218) |
| Recall | 0.5 (1/2; very wide interval) |
| Negative clips with at least one collision alert | **7 of 10** |
| False-positive alerts per scored clip | 1.75 |
| Timing error on the one matched alert | 66 ms (n = 1: not an estimate) |
| Unscored positive clips | candidate-02: 2 alerts, candidate-03: 1, candidate-05: 1, candidate-06: 0, primary-demo: 2 (locations unknown) |

Failure patterns seen in the raw events (examples from `research/backend_accuracy/real_events_before_v2.json`):

1. **Proximity is promoted to a collision.** An alert needs only: two tracks' boxes within 0.5 box-diagonals (or IoU ≥ 0.05) at the same sampled frame **plus** a speed drop in either track. Many alerts have `max_box_overlap_iou` 0.0.
2. **Alert spans are meaningless.** A "possible collision" on `primary-demo` runs 0.97 s → 29.8 s (two queued cars). Segments are merged across the whole video.
3. **One object counted as two.** Car and truck boxes on the same vehicle give IoU 0.97–0.99 between two track ids ("collision" at IoU 0.994 on `candidate-04`, 0.992 on `accid3nd_north_50mm`). This is a duplicate detection, not an interaction.
4. **Depth ordering ignored.** Perspective overlap in dense highway traffic (`accid3nd_north_50mm`: 8 alerts, 15 near misses on a clip with no visible collision).
5. **Moving camera.** Image-space speed on a dashcam includes ego motion; nuScenes (a yield, no collision) produced alerts on 4 of 6 cameras, comma2k19 (60 s of normal driving) produced 3.
6. **Missed the real one in one case.** On `accid3nd_south_50mm` all alerts are at 7.4–7.8 s; the visible impact at ≈3.8 s is not flagged (the van is a stopped object on the shoulder and the yellow car is occluded in part).
7. **Volume.** 1 093 stored events for a 15 s clip (previous report), 943 of them `potential_interaction`; the narrative layer then has to merge them.

Other event facts: there is no event tier (informational / attention / possible accident); every event has `category = machine_detection`. There is no supporting-vs-contradicting observation list, no review status carried with the event, no defined meaning of `uncertainty_ms` beyond "half the sample period", and no keyframe reference per event.

## 5. Timestamps and sync, baseline

- Real PTS is used end to end (index, sampling grid, detections, tracks, events). VFR clips in the set (`candidate-03/04/06`, all six nuScenes cameras) process without special-casing; nominal fps is never used to compute time.
- `source media ts` (relative PTS), `container PTS start`, `wall clock claimed` (`creation_time_claimed`, not trusted), `case ts` and sync bound are separate fields in the API (`clocks_for`). Good.
- Hole: the still-image path (above) breaks event-to-frame mapping for the full-size image.
- Unmeasured: seek accuracy on VFR with large gaps, clock offset error against Accid3nD's per-frame epoch timestamps (offsets of the four cameras are known to < 20 ms and could serve as an offset-error test).

## 6. Witness side, baseline

- No Featherless key here, so live behaviour is **not measured**. All claim-extraction tests use simulated responses and prove the guards (fabricated or paraphrased quotes rejected, prompt injection ignored, repair once, ledger, policy gate), not extraction accuracy.
- No labelled witness-statement set exists in the repo with known claim labels beyond the four scripted scenario statements; an extraction/verification accuracy number "cannot yet be established" until one is written (a synthetic set can be written and will be labelled synthetic).

## 7. What this baseline rules out claiming

- No claim that accident detection is accurate: measured precision on the small real set was 0.045.
- No claim about tracking accuracy, detector mAP on traffic video, or witness-extraction accuracy: no labels / no key.
- Synthetic scenario results (`research/phase14`) test the reconstruction logic with oracle boxes and are kept apart from the real-footage numbers above.
