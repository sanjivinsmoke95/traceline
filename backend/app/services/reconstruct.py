"""Reconstruction: turns source events, clock relations, identity decisions and witness claims into one scenario's canonical events,
contradictions, corroborations, gaps, trajectories, summaries and findings.

A scenario is an immutable set of assumptions (clock offsets, excluded relations, reference clock, identity links to ignore). Building
one never changes the evidence or other scenarios. Everything stored here is derived and can be rebuilt; nothing is ever marked as fact.
"""
import re
from collections import defaultdict
from datetime import datetime, timezone
from statistics import median

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import Range
from sqlalchemy.orm import Session

from app.db import new_id
from app.models import (Actor, ActorLink, Camera, Case, Contradiction, Corroboration, Event, EventSource, EventSummary, EvidenceFile,
                        ExtractedFrame, GapRecord, InvestigationFinding, ModelRun, ReconstructionScenario, SensorRecord, SensorSource,
                        SourceAnnotation, SourceEvent, Trajectory, VehicleAssociation, VehicleTrack, VideoMetadata, Witness, WitnessClaim, WitnessStatement)
from app.models.engine import ClockRecord, SyncEdge
from app.config import get_settings
from app.llm.featherless import FeatherlessClient
from app.services import audit, engine_rules as rules, geometry, incident, sync
from app.services.reconstruction_view import associations_from_links
from app.services.sources import annotation_edges

ENGINE_VERSION = "1"
TIME_WORDS = [
    (re.compile(r"\b(just |right |shortly )?before (the |it )?(collision|crash|impact|accident|hit)", re.I), (-3000, 0)),
    (re.compile(r"\b(at the (moment|instant)|as (they|it|the cars?) (hit|collided)|when (they|it|the cars?) (hit|collided|crashed))", re.I), (-500, 500)),
    (re.compile(r"\b(just |right |shortly )?(after|following) (the |it )?(collision|crash|impact|accident)", re.I), (0, 3000)),
]
TITLES = {
    "vehicle_enters_scene": "{a} enters the scene", "vehicle_leaves_scene": "{a} leaves the scene (or the track ended)",
    "vehicle_slowing": "{a} appears to slow", "vehicle_stopped": "{a} appears to stop", "vehicle_direction_change": "{a} changes direction",
    "vehicle_maneuver_observed": "{a} appears to {m} (image space)", "potential_interaction": "Possible interaction between tracked road users",
    "possible_collision": "Possible collision between tracked vehicles", "possible_near_miss": "Possible near miss between tracked road users",
    "hard_braking": "Hard braking recorded for {a}", "acceleration_spike": "Sharp acceleration change recorded for {a}",
    "collision_flag": "Collision flag recorded by sensor of {a}", "stop": "{a} stops (sensor)", "heading_change": "Heading change recorded for {a}",
    "sensor_gap": "Sensor recording gap",
}


def _now():
    return datetime.now(timezone.utc)


def _title(et: str, actor_label: str | None, obs: dict) -> str:
    a = actor_label or "A tracked vehicle"
    if et == "possible_collision" and obs.get("relation") == "accident_scene_cue":
        return "Possible accident: the scene looks like an accident (model cue)"
    m = {"left": "turn left", "right": "turn right", "straight": "continue straight"}.get(obs.get("maneuver"), "move")
    return TITLES.get(et, et.replace("_", " ").capitalize() + (f" ({a})" if actor_label else "")).format(a=a, m=m)[:240]


def baseline_assumptions() -> dict:
    return {"edge_offsets": {}, "exclude_edges": [], "reference": None, "unconfirm_links": []}


def _links(db: Session, case_id: str, ignore: set[str]) -> tuple[dict, dict, dict]:
    """confirmed links by ref -> actor, candidate links by ref -> [actor]"""
    acts = {a.id: a for a in db.execute(select(Actor).where(Actor.case_id == case_id)).scalars()}
    conf, cand = {}, defaultdict(list)
    for l in db.execute(select(ActorLink).where(ActorLink.actor_id.in_(list(acts) or [""]))).scalars():
        if l.id in ignore:
            continue
        if l.status == "confirmed":
            conf[(l.ref_type, l.ref_id)] = l.actor_id
        elif l.status == "candidate":
            cand[(l.ref_type, l.ref_id)].append(l.actor_id)
    return acts, conf, cand


def _place_claim(norm: dict, anchors: list[dict], desc: str) -> tuple | None:
    """Only phrases anchored to another event are placed; 'around 10:30' or 'a few seconds' never become timestamps."""
    for pat, (lo, hi) in TIME_WORDS:
        if pat.search(desc or ""):
            if not anchors:
                return None
            a = min(anchors, key=lambda e: e["t_start"])
            return a["group"], a["t_start"] + lo, a["t_start"] + hi, a["unc"] + 1000
    return None


def build(db: Session, case: Case, scn: ReconstructionScenario, who: str) -> dict:
    asm = baseline_assumptions() | (scn.assumptions or {})
    # ---- evidence, sync, identity
    evs = {e.id: e for e in db.execute(select(EvidenceFile).where(EvidenceFile.case_id == case.id, EvidenceFile.status == "valid")).scalars()}
    media = {i: e for i, e in evs.items() if e.kind in ("dashcam", "cctv", "sensor_csv")}
    labels = {i: f"{('CCTV' if e.kind == 'cctv' else 'dashcam' if e.kind == 'dashcam' else 'sensor log' if e.kind == 'sensor_csv' else e.kind)} “{e.original_filename}”" for i, e in evs.items()}
    clocks = [{"evidence_id": c.evidence_id, "utc_zero": c.utc_zero, "bound_ms": c.bound_ms, "basis": c.basis}
              for c in db.execute(select(ClockRecord).where(ClockRecord.case_id == case.id)).scalars()]
    edges = [{"id": e.id, "a_evidence": e.a_evidence, "b_evidence": e.b_evidence, "offset_ms": e.offset_ms, "bound_ms": e.bound_ms, "basis": e.basis, "status": e.status}
             for e in db.execute(select(SyncEdge).where(SyncEdge.case_id == case.id)).scalars()]
    ann_rows = db.execute(select(SourceAnnotation).where(SourceAnnotation.evidence_id.in_(list(media) or [""]))).scalars()
    edges.extend(annotation_edges([
        {"evidence_id": a.evidence_id, "clock_offset_ms": a.clock_offset_ms, "clock_offset_reference_id": a.clock_offset_reference_id}
        for a in ann_rows if a.clock_offset_reference_id in media and a.evidence_id in media
    ]))
    solved = sync.solve([{"id": i, "kind": e.kind} for i, e in media.items()], clocks, edges, reference=asm["reference"], edge_offsets=asm["edge_offsets"],
                        exclude_edges=set(asm["exclude_edges"]))
    lk = sync.lookup(solved)
    actors, conf, cand = _links(db, case.id, set(asm["unconfirm_links"]))
    alabel = {i: a.label for i, a in actors.items()}

    tracks_pk: dict[tuple, VehicleTrack] = {}
    track_rows: list[VehicleTrack] = []
    for ev_id, e in media.items():
        if e.kind in ("dashcam", "cctv"):
            run = db.execute(select(ModelRun).where(ModelRun.case_id == case.id, ModelRun.input_sha256 == e.sha256, ModelRun.stage == "track",
                                                    ModelRun.status.in_(("completed", "partially_completed"))).order_by(ModelRun.finished_at.desc())).scalars().first()
            if run:
                for t in db.execute(select(VehicleTrack).where(VehicleTrack.model_run_id == run.id)).scalars():
                    tracks_pk[(ev_id, t.track_id)] = t
                    track_rows.append(t)
    src_sensor = {s.id: s for s in db.execute(select(SensorSource).where(SensorSource.case_id == case.id)).scalars()}

    # ---- canonical events from source events
    events: list[dict] = []
    by_se: dict[str, dict] = {}
    for se in db.execute(select(SourceEvent).where(SourceEvent.case_id == case.id)).scalars():
        ev = evs.get(se.evidence_id)
        if ev is None or se.t_start_ms is None:
            continue
        hint = se.entity_hint or {}
        tr = [tracks_pk.get((se.evidence_id, t["track_id"])) for t in hint.get("tracks", [])]
        trk_actors = [conf.get(("track", t.id)) if t else None for t in tr]
        if se.stage == "sensors":
            actor = conf.get(("sensor_source", hint.get("sensor_source"))) or None
            stype, ckind = "sensor", "sensor"
        else:
            actor = trk_actors[0] if len(trk_actors) == 1 else None
            stype = "video_cctv" if ev.kind == "cctv" else "video_dashcam"
            ckind = ev.kind
        iv = sync.to_case_interval(lk, se.evidence_id, se.t_start_ms, se.t_end_ms, se.uncertainty_ms)
        obs = dict(se.observations or {}) | {"source_time_ms": [se.t_start_ms, se.t_end_ms]}
        if len(trk_actors) > 1:
            obs["actors"] = [a for a in trk_actors if a]
        d = {"id": new_id("evt")(), "source_event_id": se.id, "category": se.category, "event_type": se.event_type, "actor_id": actor, "evidence_id": se.evidence_id,
             "source_type": stype, "camera_kind": ckind, "observations": obs, "synthetic": ev.synthetic, "group": iv["group"] if iv else None,
             "t_start": iv["start_ms"] if iv else None, "t_end": iv["end_ms"] if iv else None, "unc": iv["uncertainty_ms"] if iv else se.uncertainty_ms,
             "rule_id": se.rule_id, "source_ref": se.source_ref, "confidence": dict(se.confidence or {}), "model_run_id": se.model_run_id,
             "tracks": hint.get("tracks", []), "candidates": [c for t in tr if t for c in cand.get(("track", t.id), [])] or None,
             "witness_label": None}
        if len(trk_actors) == 1 and actor is None and tr[0] is not None:
            d["candidates"] = cand.get(("track", tr[0].id)) or None
        d["title"] = _title(se.event_type, alabel.get(actor), obs)
        events.append(d)
        by_se[se.id] = d

    # ---- witness claims
    anchors = [e for e in events if e["event_type"] in ("possible_collision", "collision_flag") and e["t_start"] is not None]
    wit_events = []
    for c, st, w in db.execute(select(WitnessClaim, WitnessStatement, Witness).join(WitnessStatement, WitnessClaim.statement_id == WitnessStatement.id)
                               .join(Witness, WitnessStatement.witness_id == Witness.id).where(Witness.case_id == case.id,
                                                                                               WitnessClaim.extraction_status.in_(("valid", "repaired", "manual")))):
        n = c.normalized or {}
        place = _place_claim(n, anchors, c.time_description or "")
        et = n.get("event_type_hint") or "witness_statement"
        actor = conf.get(("claim", c.id))
        obs = {"quote": c.source_text, "subject": c.subject, "action": c.action, "direction": c.direction, "time_description": c.time_description,
               "certainty": c.certainty, "location": n.get("location"), "code_derived_maneuver": n.get("code_derived_maneuver"),
               "direction_check": n.get("direction_check"), "hedges": n.get("hedges"), "time_exactness": n.get("time_exactness", "none"),
               "claim_id": c.id, "statement_id": st.id, "extraction_status": c.extraction_status,
               "time_placement": f"placed relative to a collision event from the phrase “{c.time_description}”" if place else "not placed: no exact or anchored time in the statement"}
        d = {"id": new_id("evt")(), "source_event_id": None, "category": "witness_claim", "event_type": et, "actor_id": actor, "evidence_id": st.evidence_id,
             "source_type": "witness", "camera_kind": "witness", "observations": obs, "synthetic": st.synthetic, "group": place[0] if place else None,
             "t_start": place[1] if place else None, "t_end": place[2] if place else None, "unc": place[3] if place else 0.0, "rule_id": "WIT-CLAIM-01",
             "source_ref": {"kind": "statement_span", "char_start": c.char_start, "char_end": c.char_end}, "confidence": {"certainty": c.certainty, "extraction": c.extraction_status},
             "model_run_id": c.model_run_id, "tracks": [], "candidates": cand.get(("claim", c.id)) or None, "witness_label": w.label,
             "title": f"{w.label}: “{c.source_text[:110]}”"}
        wit_events.append(d)
    events += wit_events

    # ---- sensor speeds & coverage for the rules
    speeds, cover = defaultdict(list), {}
    for s in src_sensor.values():
        m = lk.get(s.evidence_id)
        recs = db.execute(select(SensorRecord).where(SensorRecord.sensor_source_id == s.id, SensorRecord.valid.is_(True)).order_by(SensorRecord.row_no)).scalars().all()
        if m and recs:
            cover[s.evidence_id] = (recs[0].time_ms + m["offset"], recs[-1].time_ms + m["offset"], m["group"])
            actor = conf.get(("sensor_source", s.id)) or s.actor_id
            for r in recs:
                v = r.values["si"].get("speed")
                if v is not None and actor:
                    speeds[actor].append((r.time_ms + m["offset"], v))
    ctx = {"actors": {i: {"label": l} for i, l in alabel.items()}, "evidence_labels": labels, "speeds": speeds, "sensor_cover": cover}
    corr = rules.correlate(events, ctx)

    # ---- inference events (after premises exist)
    # ---- persist events
    prov_base = {"engine_version": ENGINE_VERSION, "rules_version": rules.RULES_VERSION, "scenario_id": scn.id, "built_by": who}
    for d in events:
        synced = d["group"] is not None
        quality = d["confidence"] | {"timestamp": f"synchronized, ±{d['unc']:.0f} ms" if synced else ("not placed" if d["category"] == "witness_claim" else "unsynchronized (own clock)")}
        db.add(Event(id=d["id"], case_id=case.id, scenario_id=scn.id, category=d["category"], event_type=d["event_type"], title=d["title"], actor_id=d["actor_id"],
                     actor_candidates=d["candidates"], start_case_ms=round(d["t_start"]) if synced and d["t_start"] is not None else None,
                     end_case_ms=round(d["t_end"]) if synced and d["t_end"] is not None else None, uncertainty_ms=round(d["unc"]),
                     uncertainty_basis=("clock_bound+sampling" if synced and d["category"] != "witness_claim" else "witness_phrase" if synced else "unsynchronized"),
                     sync_group_id=None, source_type=d["source_type"], source_id=d["evidence_id"], observations=d["observations"] | {"actors": d["observations"].get("actors", [])},
                     confidence=quality, provenance=prov_base | {"source_event_id": d["source_event_id"], "rule_id": d["rule_id"], "model_run_id": d["model_run_id"],
                                                                  "sync_group": d["group"]}, synthetic=d["synthetic"]))
    db.flush()
    for d in events:
        ref = d["source_ref"]
        if d["evidence_id"] is None:
            continue
        if ref.get("kind") == "video_frames":
            fr = ref["frames"]
            db.add(EventSource(event_id=d["id"], evidence_id=d["evidence_id"], kind="video_frames", frames=Range(int(fr[0]), int(fr[1]) + 1)))
        elif ref.get("kind") == "statement_span":
            db.add(EventSource(event_id=d["id"], evidence_id=d["evidence_id"], kind="statement_span", char_span=Range(ref["char_start"], ref["char_end"])))
        else:
            rows = ref.get("rows") or ref.get("row_range")
            db.add(EventSource(event_id=d["id"], evidence_id=d["evidence_id"], kind="sensor_rows" if rows else "metadata",
                               rows=Range(int(rows[0]), int(rows[1]) + 1) if rows else None))
    db.flush()
    inf_events = []
    for inf in corr["inferences"]:
        e = {"id": new_id("evt")(), "source_event_id": None, "category": "inference", "event_type": inf["type"], "actor_id": inf["actor_id"], "evidence_id": None,
             "source_type": "engine", "camera_kind": "engine", "observations": {"alternatives": inf["alternatives"], "premises": inf["premises"]}, "synthetic": any(x["synthetic"] for x in events),
             "group": inf["group"], "t_start": inf["t_start"], "t_end": inf["t_end"], "unc": inf["unc"], "rule_id": inf["rule_id"], "source_ref": {}, "confidence": {"basis": "rule-based; see premises"},
             "model_run_id": None, "tracks": [], "candidates": None, "witness_label": None, "title": inf["statement"][:240], "statement": inf["statement"]}
        db.add(Event(id=e["id"], case_id=case.id, scenario_id=scn.id, category="inference", event_type=e["event_type"], title=e["title"], actor_id=e["actor_id"],
                     start_case_ms=round(e["t_start"]), end_case_ms=round(e["t_end"]), uncertainty_ms=round(e["unc"]), uncertainty_basis="derived_from_premises", source_type="engine",
                     observations={"alternatives": inf["alternatives"]}, inference={"premises": inf["premises"], "alternatives": inf["alternatives"], "rule_id": inf["rule_id"], "statement": inf["statement"]},
                     confidence={"basis": "rule-based; inference, not established fact"}, provenance=prov_base | {"rule_id": inf["rule_id"]}, synthetic=e["synthetic"]))
        inf_events.append(e)
    db.flush()
    index = {e["id"]: e for e in events + inf_events}

    # ---- contradictions / corroborations rows
    con_rows, crb_rows = [], []
    for c in corr["contradictions"]:
        if c["event_a"] == c["event_b"]:
            continue
        row = Contradiction(case_id=case.id, scenario_id=scn.id, rule_id=c["rule_id"], event_a=c["event_a"], event_b=c["event_b"], related_events=c["related"], strength=c["strength"],
                            basis=c["basis"] | {"benign_explanations": c["benign_explanations"]}, explanation_template=c["explanation_template"], unresolved_questions=c["unresolved_questions"])
        db.add(row)
        con_rows.append(row)
    for c in corr["corroborations"]:
        row = Corroboration(case_id=case.id, scenario_id=scn.id, rule_id=c["rule_id"], events=c["events"], independent=c["independent"], independence_reason=c["independence_reason"],
                            temporal_relation=c["temporal_relation"], explanation_template=c["explanation_template"])
        db.add(row)
        crb_rows.append(row)
    db.flush()
    for c, row in zip([c for c in corr["contradictions"] if c["event_a"] != c["event_b"]], con_rows):
        c["id"] = row.id
    for c, row in zip(corr["corroborations"], crb_rows):
        c["id"] = row.id
    if actors:
        link_rows = db.execute(select(ActorLink).where(
            ActorLink.actor_id.in_(list(actors)), ActorLink.status == "confirmed", ActorLink.ref_type == "track")).scalars()
        for payload in associations_from_links(
            [{"pk": t.id, "evidence_id": t.evidence_id, "track_id": t.track_id} for t in track_rows],
            [{"id": link.id, "actor_id": link.actor_id, "ref_id": link.ref_id} for link in link_rows],
            [{"id": row.id, "event_a": row.event_a, "event_b": row.event_b} for row in con_rows],
            {item["id"]: item["evidence_id"] for item in events},
        ):
            db.add(VehicleAssociation(
                case_id=case.id, scenario_id=scn.id, status="uncertain", tracks=payload["tracks"],
                supporting=payload["supporting"], contradicting=payload["contradicting"], basis=payload["basis"], created_by=who,
            ))

    # ---- gaps
    facts = _gap_facts(db, case, media, lk, solved, events, tracks_pk, track_rows, conf, cand, corr, labels, evs)
    gaps = rules.find_gaps(facts)
    gap_rows = []
    for g in gaps:
        row = GapRecord(case_id=case.id, scenario_id=scn.id, gap_type=g["gap_type"], scope=g["scope"], details=g["details"], recommendation=g["recommendation"], refs=g["refs"])
        db.add(row)
        gap_rows.append(row)
    db.flush()

    # ---- trajectories
    n_traj = _trajectories(db, case, scn, track_rows, conf, lk, evs, media)

    # ---- summaries
    questions = corr["questions"]
    for e in events + inf_events:
        e["title"] = e["title"]
        content = rules.build_summary(e, index, corr, questions, gaps, ctx)
        db.add(EventSummary(event_id=e["id"], scenario_id=scn.id, content=content, validated=True))

    # ---- findings (for investigator review; never decisions)
    n_find = 0
    def finding(kind, refs, text):
        nonlocal n_find
        db.add(InvestigationFinding(case_id=case.id, scenario_id=scn.id, kind=kind, ref_ids=refs, text=text[:2000]))
        n_find += 1
    for c in corr["contradictions"]:
        if c.get("id"):
            finding("contradiction", [c["id"]], c["explanation_template"])
    for c in corr["corroborations"]:
        finding("corroboration", [c["id"]], c["explanation_template"])
    for g, row in zip(gaps, gap_rows):
        finding("gap", [row.id], f"{g['gap_type'].replace('_', ' ')}: {g['recommendation']}")
    for q in questions:
        finding("open_question", q["refs"] or [q["event"]], q["text"])
    for e in inf_events:
        finding("inference", [e["id"]], e["statement"])
    db.flush()
    settings = get_settings()
    narrative = incident.narrate(db, case, scn, FeatherlessClient(settings), settings)
    scn_counts = {"events": len(events) + len(inf_events), "contradictions": len(con_rows), "corroborations": len(crb_rows), "gaps": len(gap_rows), "trajectories": n_traj,
                  "open_questions": len(questions), "findings": n_find, "inferences": len(inf_events), "sync_groups": len(solved["groups"]),
                  "unsynchronized_sources": solved["unsynchronized"], "sync_inconsistencies": len(solved["inconsistencies"])}
    scn.assumptions = asm | {"build": scn_counts, "sync_solution": solved, "engine_version": ENGINE_VERSION, "rules_version": rules.RULES_VERSION,
                             "incident_narrative": narrative}
    audit.record(db, actor=who, action="reconstruction_built", case_id=case.id, target_type="scenario", target_id=scn.id, details=scn_counts | {"unsynchronized_sources": len(solved["unsynchronized"])})
    return scn_counts


def _gap_facts(db, case, media, lk, solved, events, tracks_pk, track_rows, conf, cand, corr, labels, evs) -> dict:
    videos = {}
    for ev_id, e in media.items():
        if e.kind not in ("dashcam", "cctv"):
            continue
        frames = db.execute(select(ExtractedFrame.time_ms, ExtractedFrame.blur, ExtractedFrame.brightness).where(ExtractedFrame.evidence_id == ev_id, ExtractedFrame.sampled.is_(True))
                            .order_by(ExtractedFrame.time_ms)).all()
        times = [f[0] for f in frames]
        m = lk.get(ev_id)
        off = m["offset"] if m else None
        vm = db.get(VideoMetadata, ev_id)
        cam = db.get(Camera, vm.camera_id) if vm and vm.camera_id else None
        blur = [f[1] for f in frames if f[1] is not None]
        bright = [f[2] for f in frames if f[2] is not None]
        step = median([b - a for a, b in zip(times, times[1:])]) if len(times) > 2 else 0
        def covers(lo, hi, times=times, off=off, step=step):
            if off is None:
                return False
            return any(lo <= t + off <= hi for t in times) or (bool(times) and times[0] + off <= lo and hi <= times[-1] + off and False)
        videos[ev_id] = {"times": times, "to_case": (lambda t, off=off: t + off if off is not None else None), "label": labels[ev_id], "brightness": median(bright) if bright else None,
                         "blur": median(blur) if blur else None, "static": e.kind == "cctv", "calibrated": bool(cam and cam.calibration and cam.calibration.get("usable_for_metric")), "covers": covers}
    in_events = set()
    for e in events:
        for t in e.get("tracks", []):
            in_events.add((e["evidence_id"], t["track_id"]))
    tracks = []
    for t in track_rows:
        c = t.continuity or {}
        pk_link = conf.get(("track", t.id))
        tracks.append({"evidence_id": t.evidence_id, "track_id": t.track_id, "class": t.class_name, "actor_id": pk_link, "in_events": (t.evidence_id, t.track_id) in in_events,
                       "observations": c.get("observations", 0), "reliability": c.get("reliability"), "continuations": c.get("continuation_candidates", []),
                       "candidates": cand.get(("track", t.id), [])})
    claims = []
    for e in events:
        if e["category"] != "witness_claim" or e["observations"].get("code_derived_maneuver") is None and e["event_type"] not in ("vehicle_speed_claim", "collision_claim", "vehicle_braked"):
            continue
        if e["id"] in corr["compared_claims"]:
            continue
        reasons = []
        if e["t_start"] is None:
            reasons.append("the statement gives no time that can be placed against the other evidence")
        if not e["actor_id"]:
            reasons.append("the vehicle described is not confirmed as any known vehicle")
        if e["event_type"] == "vehicle_speed_claim":
            reasons.append("no number or no sensor on the vehicle to compare with")
        if not reasons:
            reasons.append("no other evidence addresses what this statement describes")
        claims.append({"id": e["id"], "witness_label": e["witness_label"], "quote": e["observations"].get("quote", ""), "reasons": reasons, "compared": False})
    return {"videos": videos, "events": events, "labels": labels, "unsynchronized": solved["unsynchronized"], "inconsistencies": solved["inconsistencies"], "tracks": tracks, "claims": claims}


def _trajectories(db, case, scn, track_rows, conf, lk, evs, media) -> int:
    n = 0
    for t in track_rows:
        obs = (t.continuity or {}).get("observations", len(t.points))
        actor = conf.get(("track", t.id))
        if obs < 5 or (not actor and t.class_name not in ("car", "truck", "bus", "motorcycle", "pedestrian", "bicycle")):
            continue
        ev = evs[t.evidence_id]
        vm = db.get(VideoMetadata, t.evidence_id)
        cam = db.get(Camera, vm.camera_id) if vm and vm.camera_id else None
        if cam is None:
            cam = Camera(case_id=case.id, kind=ev.kind, label=f"{ev.kind}: {ev.original_filename}")
            db.add(cam)
            db.flush()
            if vm:
                vm.camera_id = cam.id
        m = lk.get(t.evidence_id)
        off = m["offset"] if m else None
        pts_img = [{"frame_no": p["frame_no"], "source_ms": p["time_ms"], "case_ms": round(p["time_ms"] + off) if off is not None else None, "x": round(p["anchor"][0], 1),
                    "y": round(p["anchor"][1], 1), "backfilled": p.get("backfilled", False)} for p in t.points]
        gaps = (t.continuity or {}).get("gaps", [])
        ref = {"evidence_id": t.evidence_id, "track_pk": t.id, "track_id": t.track_id, "class": t.class_name, "missing_segments": gaps, "reliability": (t.continuity or {}).get("reliability"),
               "moving_camera": ev.kind == "dashcam"}
        note = ("Image pixels of a moving camera: positions include the camera's own movement. Not a ground trajectory." if ev.kind == "dashcam"
                else "Image pixels; no real-world distance is implied.")
        db.add(Trajectory(case_id=case.id, scenario_id=scn.id, actor_id=actor, track_ref=ref, frame="image", camera_id=cam.id, points=pts_img, calibration_note=note))
        n += 1
        cal = cam.calibration if cam and ev.kind == "cctv" else None
        if cal and cal.get("usable_for_metric"):
            pts_g = []
            for p in pts_img:
                gx, gy = geometry.project(cal["H"], p["x"], p["y"])
                pts_g.append({"frame_no": p["frame_no"], "case_ms": p["case_ms"], "x": round(gx, 3), "y": round(gy, 3),
                              "uncertainty": round(geometry.ground_uncertainty(cal["H"], p["x"], p["y"], 3.0, cal["rms_error_ground"]), 3),
                              "inside_calibrated_region": geometry.inside(cal["valid_region"], p["x"], p["y"])})
            db.add(Trajectory(case_id=case.id, scenario_id=scn.id, actor_id=actor, track_ref=ref, frame="ground_plane", camera_id=cam.id, points=pts_g,
                              calibration_note=f"Ground plane in {cal.get('units', 'm')} from a homography fit to {cal['n_points']} reference points (rms {cal['rms_error_ground']} {cal.get('units', 'm')}); "
                                               "valid only on the calibrated flat plane, near the reference points; each point carries its own uncertainty. Not a measurement of speed or distance of record."))
            n += 1
    return n
