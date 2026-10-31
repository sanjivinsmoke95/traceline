"""Investigator read models for one scenario.

The list is derived from stored events. Merging does not delete those events.
A narrative is stored when the scenario is built. Playback reads that document
and does not call Featherless. A failed model call stores no model sentences.
"""
import json
import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.llm.featherless import LlmError, NotConfigured, PolicyBlocked
from app.models import Case, Event, EventSummary, ExtractedFrame, GapRecord, ReconstructionScenario, SourceEvent
from app.services.reconstruction_view import hypotheses as hypothesis_rows
from app.services.report import BANNED

# Strings already produced by engine_rules.build_summary. Copied, not rewritten.
_UNSYNC = "This event comes from a source that is not synchronized with any other; its time is its own."
_PIXELS = "Positions and movement are in image pixels; no real-world distance or speed is implied."
_MOVING = "The camera was moving; image motion includes the camera's own movement."
_CUE = "The accident-cue model is a weak supporting signal and does not decide this event."
_IDENTITY = "The vehicle in this event is not confirmed as any known vehicle."

_INFERRED = {"possible_collision", "possible_near_miss"}
_OBS_KEYS = (
    "track_ids", "class", "relation", "max_box_overlap_iou", "min_gap_in_diagonals",
    "abrupt_speed_changes", "relative_speed_before", "relative_speed_after", "drop",
    "motion", "end_reason", "note", "coordinate_frame", "relative_to_moving_camera",
    "accident_cue_nearby", "accident_cue_note", "unit",
    "scene_cue", "motion_analysis_agrees", "collision_signals", "collision_score",
)
_ITEM_KEYS = (
    "id", "event_type", "classification", "measurement_classification", "title",
    "start_ms", "end_ms", "uncertainty_ms", "uncertainty_basis", "synchronized",
    "evidence_id", "track_ids", "track_classes", "frames", "frame_times_ms",
    "observations", "why_generated", "uncertainty", "display", "merged_from",
)
_METHOD = re.compile(r"does not determine fault|assigns no fault", re.I)
_PLATE = re.compile(r"\b(licen[cs]e plate|number plate|plate)\b", re.I)
_VISUAL = re.compile(r"visually supported contact", re.I)


class NarrativeInvalid(Exception):
    """The model text failed a check. Nothing from it is stored."""


def _rule_id(event_type: str, obs: dict, given: str | None) -> str | None:
    if given:
        return given
    return {
        "vehicle_enters_scene": "VID-ENTER-01", "vehicle_leaves_scene": "VID-LEAVE-01",
        "vehicle_slowing": "VID-SLOW-01", "vehicle_stopped": "VID-STOP-01",
        "vehicle_direction_change": "VID-TURN-01", "vehicle_maneuver_observed": "VID-MANEUVER-01",
        "potential_interaction": "VID-INTER-01", "possible_near_miss": "VID-NEAR-01",
        "possible_collision": "VID-COLL-01",
    }.get(event_type)


def _uncertainty(row: dict) -> list[str]:
    if row.get("uncertainty_lines"):
        return list(row["uncertainty_lines"])
    obs = row.get("observations") or {}
    lines = []
    if row.get("start_ms") is None:
        lines.append(_UNSYNC)
    else:
        lines.append(f"Time is known to within ±{(row.get('uncertainty_ms') or 0) / 1000:.2f} s (clock relation and sampling).")
    if obs.get("coordinate_frame") == "image_pixels":
        lines.append(_PIXELS)
    if obs.get("relative_to_moving_camera"):
        lines.append(_MOVING)
    if obs.get("accident_cue_note"):
        lines.append(_CUE)
    if not row.get("actor_id") and row.get("category") not in ("inference",) and row.get("source_type") != "sensor":
        lines.append(_IDENTITY)
    return lines


def _why(row: dict) -> dict:
    obs = row.get("observations") or {}
    et = row["event_type"]
    touching = et == "possible_collision" or (et == "potential_interaction" and obs.get("relation") == "box_overlap")
    drop = bool(obs.get("abrupt_speed_changes")) or et in ("possible_collision", "possible_near_miss")
    return {"rule_id": _rule_id(et, obs, row.get("rule_id")), "touching": touching, "abrupt_speed_drop": drop}


def _display(event_type: str) -> str:
    if event_type == "visually_supported_contact":
        return "critical"
    if event_type == "possible_collision":
        return "amber"
    return "neutral"


def _classification(event_type: str) -> str:
    return "inferred" if event_type in _INFERRED or event_type == "visually_supported_contact" else "observed"


def _origin(row: dict) -> str:
    return row.get("source_event_id") or row["id"]


def _span_end(row: dict) -> int | None:
    if row.get("end_ms") is not None:
        return row["end_ms"]
    return row.get("start_ms")


def _sample_period(row: dict) -> int:
    """uncertainty_ms is half the sample period."""
    return int(row.get("uncertainty_ms") or 0) * 2


def merge_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Merge only same evidence, type, and track set, when the gap is at most one sample period.

    Unplaced rows stay unmerged. Continuation candidates are different track ids, so they stay apart.
    """
    placed = sorted((r for r in rows if r.get("start_ms") is not None), key=lambda r: (r["start_ms"], r["id"]))
    unplaced = [r for r in rows if r.get("start_ms") is None]
    groups: list[dict] = []
    for row in placed:
        key = (row.get("evidence_id"), row["event_type"], tuple(sorted(row.get("track_ids") or [])))
        end = _span_end(row)
        hit = None
        for group in groups:
            if group["key"] != key:
                continue
            limit = max(_sample_period(row), _sample_period(group["rep"]))
            if row["start_ms"] <= group["end"] + limit and group["start"] <= (end if end is not None else row["start_ms"]) + limit:
                hit = group
                break
        if hit is None:
            groups.append({"key": key, "rep": row, "members": [row], "start": row["start_ms"], "end": end if end is not None else row["start_ms"]})
            continue
        hit["members"].append(row)
        hit["start"] = min(hit["start"], row["start_ms"])
        hit["end"] = max(hit["end"], end if end is not None else row["start_ms"])
        if (row["start_ms"], row["id"]) < (hit["rep"]["start_ms"], hit["rep"]["id"]):
            hit["rep"] = row
    return groups, unplaced


def _frames_of(members: list[dict]) -> tuple[list, list]:
    frames, times = [], []
    for row in members:
        frames.extend(row.get("frames") or [])
        times.extend(row.get("frame_times_ms") or [])
    frames = [f for f in frames if f is not None]
    times = [t for t in times if t is not None]
    if not frames:
        return [], []
    return [min(frames), max(frames)], ([min(times), max(times)] if times else [])


def _project(group_or_row: dict, *, members: list[dict] | None = None) -> dict:
    row = group_or_row
    members = members or [row]
    obs = dict(row.get("observations") or {})
    for extra in members:
        eo = extra.get("observations") or {}
        if (eo.get("max_box_overlap_iou") or 0) > (obs.get("max_box_overlap_iou") or 0):
            obs["max_box_overlap_iou"] = eo.get("max_box_overlap_iou")
        if eo.get("abrupt_speed_changes") and not obs.get("abrupt_speed_changes"):
            obs["abrupt_speed_changes"] = eo["abrupt_speed_changes"]
    frames, times = _frames_of(members)
    start = min(m["start_ms"] for m in members) if members[0].get("start_ms") is not None else None
    ends = [_span_end(m) for m in members if _span_end(m) is not None]
    item = {
        "id": row["id"],
        "event_type": row["event_type"],
        "classification": _classification(row["event_type"]),
        "measurement_classification": "observed",
        "title": row.get("title") or row["event_type"].replace("_", " "),
        "start_ms": start,
        "end_ms": max(ends) if ends else None,
        "uncertainty_ms": max(int(m.get("uncertainty_ms") or 0) for m in members),
        "uncertainty_basis": row.get("uncertainty_basis") or "sampling",
        "synchronized": start is not None,
        "evidence_id": row.get("evidence_id"),
        "track_ids": list(row.get("track_ids") or []),
        "track_classes": list(row.get("track_classes") or []),
        "frames": frames,
        "frame_times_ms": times,
        "observations": {k: obs[k] for k in _OBS_KEYS if k in obs},
        "why_generated": _why(row | {"observations": obs}),
        "uncertainty": _uncertainty(row | {"start_ms": start, "uncertainty_ms": max(int(m.get("uncertainty_ms") or 0) for m in members)}),
        "display": _display(row["event_type"]),
        "merged_from": [ _origin(m) for m in members ],
    }
    return {k: item[k] for k in _ITEM_KEYS}


def contact_status(items: list[dict], gaps: list[dict]) -> str:
    types = {i["event_type"] for i in items if i.get("evidence_id")}
    if "visually_supported_contact" in types:
        return "visually_supported_contact"
    gap_types = {g.get("gap_type") for g in gaps}
    if gap_types == {"no_visual_coverage_at_candidate_collision"} and "visually_supported_contact" not in types:
        return "insufficient_evidence"
    if "possible_collision" in types:
        return "possible_collision"
    return "insufficient_evidence"


def _headline(status: str, items: list[dict], gaps: list[dict]) -> str:
    types = {i["event_type"] for i in items if i.get("evidence_id")}
    gap_types = {g.get("gap_type") for g in gaps}
    if status == "visually_supported_contact":
        return "A stored event records visually supported contact."
    if "possible_near_miss" in types and "possible_collision" not in types and status != "possible_collision":
        return "A possible near miss was identified. No collision was identified in the analysed evidence."
    if "possible_collision" in types and gap_types == {"no_visual_coverage_at_candidate_collision"} and status == "insufficient_evidence":
        return "A possible collision was stored, and visual coverage at that time is missing. Contact is not confirmed."
    if status == "possible_collision":
        return "A possible collision was identified from image-space rules. Contact is not confirmed."
    return "No collision or near miss was identified in the analysed evidence."


def _sentence(text: str, classification: str, event_ids: list[str], evidence_ids: list[str]) -> dict:
    return {"text": text, "classification": classification, "event_ids": event_ids, "evidence_ids": evidence_ids}


def deterministic_summary(items: list[dict], unplaced: list[dict], gaps: list[dict], evidence_ids: list[str]) -> dict:
    status = contact_status(items + unplaced, gaps)
    sentences = []
    if not evidence_ids and not items and not unplaced:
        sentences.append(_sentence("No evidence file is stored for this scenario, so no contact claim is supported.", "insufficient", [], []))
    elif not any(i["event_type"] == "possible_collision" for i in items + unplaced):
        sentences.append(_sentence("No possible-collision event was stored for this scenario.", "insufficient", [], list(evidence_ids)))
    proximity = [i for i in items + unplaced if i["event_type"] == "potential_interaction" and i.get("evidence_id")]
    if proximity:
        item = proximity[0]
        sentences.append(_sentence(
            f"Proximity in the image was recorded ({item['id']}). Overlap is not a collision.",
            "observed", [item["id"]], [item["evidence_id"]],
        ))
    for item in items + unplaced:
        if not item.get("evidence_id"):
            sentences.append(_sentence(
                f"Event {item['id']} has no evidence file, so it does not support a contact claim.",
                "insufficient", [item["id"]], [],
            ))
            continue
        ev = [item["evidence_id"]]
        if item["event_type"] == "possible_collision" and (item.get("observations") or {}).get("relation") == "accident_scene_cue":
            sc = item["observations"].get("scene_cue") or {}
            sentences.append(_sentence(
                f"A possible collision ({item['id']}) was flagged because the accident-scene model reported an accident over {sc.get('frames', 'several')} sampled "
                f"frames (peak {round(100 * sc.get('peak_confidence', 0))}%). It does not show which vehicles touched. Contact is not confirmed.",
                "inferred", [item["id"]], ev,
            ))
        elif item["event_type"] == "possible_collision":
            sentences.append(_sentence(
                f"A possible collision ({item['id']}) was recorded from image-space boxes and a speed drop. Contact is not confirmed.",
                "inferred", [item["id"]], ev,
            ))
        elif item["event_type"] == "possible_near_miss":
            sentences.append(_sentence(
                f"A possible near miss ({item['id']}) was recorded. It is not contact.",
                "inferred", [item["id"]], ev,
            ))
        elif item["event_type"] == "collision_flag":
            sentences.append(_sentence(
                f"A sensor collision flag was recorded ({item['id']}). It is a sensor reading, not visual contact.",
                "observed", [item["id"]], ev,
            ))
        elif item["event_type"] == "visually_supported_contact":
            sentences.append(_sentence(
                f"Visually supported contact was recorded on a stored event ({item['id']}).",
                "observed", [item["id"]], ev,
            ))
    return {"headline": _headline(status, items + unplaced, gaps), "sentences": sentences, "contact_status": status}


def build_view(rows: list[dict], gaps: list[dict] | None = None, evidence_ids: list[str] | None = None) -> dict:
    gaps = gaps or []
    groups, loose = merge_rows(rows)
    items = [_project(g["rep"], members=g["members"]) for g in groups]
    unplaced = [_project(r) for r in loose]
    evidence = list(dict.fromkeys(evidence_ids if evidence_ids is not None else [i["evidence_id"] for i in items + unplaced if i.get("evidence_id")]))
    summary = deterministic_summary(items, unplaced, gaps, evidence)
    return {
        "contact_status": summary["contact_status"],
        "items": items,
        "unplaced": unplaced,
        "deterministic": {"headline": summary["headline"], "sentences": summary["sentences"]},
        "provenance": "real-detector" if any(r.get("category") == "machine_detection" for r in rows) else "none",
    }


def _bare_collision(text: str) -> bool:
    scrubbed = re.sub(r"possible\s+collision", "", text, flags=re.I)
    scrubbed = re.sub(r"\bno collision\b", "", scrubbed, flags=re.I)
    scrubbed = re.sub(r"\bnot a collision\b", "", scrubbed, flags=re.I)
    scrubbed = re.sub(r"overlap is not a collision", "", scrubbed, flags=re.I)
    return re.search(r"\bcollision\b", scrubbed, re.I) is not None


def _bad_contact(text: str, visual: bool) -> bool:
    if _VISUAL.search(text):
        return not visual
    scrubbed = re.sub(r"contact is not confirmed", "", text, flags=re.I)
    scrubbed = re.sub(r"not physical contact", "", scrubbed, flags=re.I)
    scrubbed = re.sub(r"\bnot contact\b", "", scrubbed, flags=re.I)
    scrubbed = re.sub(r"not visual contact", "", scrubbed, flags=re.I)
    return re.search(r"\b(contact|impact)\b", scrubbed, re.I) is not None


def _is_method(text: str) -> bool:
    return _METHOD.search(text) is not None and not re.search(r"\b(collision|impact|contact|vehicle|track|proximity)\b", text, re.I)


def validate_sentences(sentences: list, items: list[dict], hypotheses: list[dict] | None = None) -> None:
    """Reject the whole narrative. Do not rewrite a sentence."""
    by_id = {i["id"]: i for i in items}
    for hyp in hypotheses or []:
        by_id[hyp["id"]] = {"id": hyp["id"], "evidence_id": None, "evidence_ids": list(hyp.get("evidence_ids") or []), "event_type": "hypothesis"}
    visual = any(i["event_type"] == "visually_supported_contact" for i in items)
    if not isinstance(sentences, list) or not sentences:
        raise NarrativeInvalid("validation_failed")
    for sentence in sentences:
        if not isinstance(sentence, dict):
            raise NarrativeInvalid("validation_failed")
        text = sentence.get("text")
        if not isinstance(text, str) or not text.strip():
            raise NarrativeInvalid("validation_failed")
        if any(re.search(p, text, re.I) for p in BANNED):
            raise NarrativeInvalid("validation_failed")
        names_a_model = re.search(r"\b(make|model)\b", text, re.I) and not re.search(r"accident-cue model", text, re.I)
        if _PLATE.search(text) or names_a_model:
            raise NarrativeInvalid("validation_failed")
        event_ids = sentence.get("event_ids") or []
        evidence_ids = sentence.get("evidence_ids") or []
        if not isinstance(event_ids, list) or not isinstance(evidence_ids, list):
            raise NarrativeInvalid("validation_failed")
        if _is_method(text):
            continue
        if not event_ids or not evidence_ids:
            raise NarrativeInvalid("validation_failed")
        cited = []
        for eid in event_ids:
            if eid not in by_id:
                raise NarrativeInvalid("validation_failed")
            cited.append(by_id[eid])
        allowed_evidence = {c["evidence_id"] for c in cited if c.get("evidence_id")}
        for cited_row in cited:
            allowed_evidence.update(cited_row.get("evidence_ids") or [])
        if set(evidence_ids) != allowed_evidence or not allowed_evidence:
            raise NarrativeInvalid("validation_failed")
        if _bare_collision(text) and not any(c["event_type"] == "visually_supported_contact" for c in cited):
            raise NarrativeInvalid("validation_failed")
        if _bad_contact(text, visual and any(c["event_type"] == "visually_supported_contact" for c in cited)):
            raise NarrativeInvalid("validation_failed")
        kind = sentence.get("classification")
        if kind not in ("observed", "inferred", "insufficient"):
            raise NarrativeInvalid("validation_failed")
        if any(c["event_type"] in _INFERRED for c in cited) and kind == "observed" and re.search(r"possible\s+(collision|near miss)", text, re.I):
            raise NarrativeInvalid("validation_failed")


def parse_model_json(content: str) -> list:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise NarrativeInvalid("validation_failed") from exc
    sentences = data.get("sentences") if isinstance(data, dict) else None
    if not isinstance(sentences, list):
        raise NarrativeInvalid("validation_failed")
    return sentences


HYPOTHESIS_NOTE = "Stored rule alternatives with evidence ids. No probability is assigned. A model sentence is kept only inside ai_narrative after those ids are checked."


def hypothesis_explanation(db: Session, scn: ReconstructionScenario) -> dict:
    events = list(db.execute(select(Event).where(Event.scenario_id == scn.id)).scalars())
    plain = [{
        "id": event.id, "category": event.category, "source_type": event.source_type, "source_id": event.source_id,
        "title": event.title, "inference": event.inference, "provenance": event.provenance or {},
        "observations": event.observations or {}, "start_ms": event.start_case_ms, "end_ms": event.end_case_ms,
    } for event in events]
    ready, _unsupported = hypothesis_rows(plain, {item["id"]: item for item in plain})
    return {"items": ready, "note": HYPOTHESIS_NOTE}


def _model_payload(view: dict, explanation: dict | None = None) -> dict:
    events = []
    for item in view["items"] + view["unplaced"]:
        events.append({
            "id": item["id"], "event_type": item["event_type"], "evidence_id": item["evidence_id"],
            "start_ms": item["start_ms"], "end_ms": item["end_ms"], "track_ids": item["track_ids"],
            "track_classes": item["track_classes"], "frames": item["frames"], "rule_id": item["why_generated"]["rule_id"],
            "observations": item["observations"],
        })
    hyps = []
    for item in (explanation or {}).get("items") or []:
        hyps.append({"id": item["id"], "statement": item["statement"], "alternatives": item["alternatives"],
                     "evidence_ids": item["evidence_ids"], "probability": None})
    return {"events": events, "hypotheses": hyps, "headline": view["deterministic"]["headline"], "contact_status": view["contact_status"]}


def narrate(db: Session, case: Case, scn: ReconstructionScenario, client, settings) -> dict:
    """One text-model call after the scenario exists. Failure keeps the deterministic summary."""
    view = view_for_scenario(db, scn)
    explanation = hypothesis_explanation(db, scn)
    base = {
        "contact_status": view["contact_status"],
        "deterministic": view["deterministic"],
        "hypothesis_explanation": explanation,
        "ai_narrative_status": "unavailable",
        "ai_narrative_reason": "request_failed",
        "ai_narrative": None,
    }
    payload = _model_payload(view, explanation)
    messages = [
        {"role": "system", "content": (
            "Narrate only the JSON evidence. Every sentence that states a time, vehicle, movement, proximity, or contact "
            "must cite event_ids and evidence_ids copied from that JSON. Do not invent events, timestamps, plates, makes, or models. "
            "A possible_collision is a possible collision, never a collision, impact, or contact. "
            "Name visually supported contact only when that event_type is present. Do not assign fault. "
            "Hypothesis ids in the JSON may be cited in event_ids. Copy evidence_ids from the cited rows exactly. Do not assign a probability. "
            "Return JSON {\"sentences\":[{\"text\",\"classification\",\"event_ids\",\"evidence_ids\"}]}."
        )},
        {"role": "user", "content": json.dumps(payload, sort_keys=True)},
    ]
    try:
        result = client.chat(
            db, case=case, model=settings.featherless_text_model, messages=messages,
            purpose="incident_narrative", input_ref=payload, response_format={"type": "json_object"}, max_tokens=1200,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}} if settings.featherless_disable_thinking else None,
        )
    except PolicyBlocked:
        base["ai_narrative_reason"] = "policy_blocked"
        return base
    except NotConfigured:
        base["ai_narrative_reason"] = "not_configured"
        return base
    except LlmError:
        base["ai_narrative_reason"] = "request_failed"
        return base
    except Exception:
        base["ai_narrative_reason"] = "request_failed"
        return base
    try:
        sentences = parse_model_json(result.content)
        validate_sentences(sentences, view["items"] + view["unplaced"], explanation["items"])
    except NarrativeInvalid:
        client.record_validation(db, result.call_id, False)
        base["ai_narrative_reason"] = "validation_failed"
        return base
    client.record_validation(db, result.call_id, True)
    base["ai_narrative_status"] = "ready"
    base["ai_narrative_reason"] = None
    base["ai_narrative"] = {"model": settings.featherless_text_model, "model_run_id": str(result.call_id), "sentences": sentences}
    return base


def _row_from_event(event: Event, source: SourceEvent | None, summary: EventSummary | None) -> dict | None:
    if event.category in ("inference", "witness_claim"):
        return None
    obs = dict(event.observations or {})
    hint = (source.entity_hint if source else None) or {}
    tracks = hint.get("tracks") or []
    track_ids = obs.get("track_ids") or [t.get("track_id") for t in tracks if t.get("track_id") is not None]
    classes = obs.get("track_classes") or [t.get("class") for t in tracks if t.get("class")]
    ref = (source.source_ref if source else None) or {}
    frames = obs.get("frames") or ref.get("frames") or []
    times = obs.get("frame_times_ms") or ref.get("frame_times_ms") or []
    prov = event.provenance or {}
    lines = None
    if summary is not None and isinstance(summary.content, dict) and summary.content.get("uncertainty"):
        lines = list(summary.content["uncertainty"])
    return {
        "id": event.id, "source_event_id": prov.get("source_event_id"), "event_type": event.event_type,
        "category": event.category, "title": event.title, "start_ms": event.start_case_ms, "end_ms": event.end_case_ms,
        "uncertainty_ms": event.uncertainty_ms, "uncertainty_basis": event.uncertainty_basis,
        "evidence_id": event.source_id, "track_ids": list(track_ids), "track_classes": list(classes),
        "frames": list(frames), "frame_times_ms": list(times), "observations": obs,
        "rule_id": prov.get("rule_id") or (source.rule_id if source else None),
        "actor_id": event.actor_id, "source_type": event.source_type, "uncertainty_lines": lines,
    }


def view_for_scenario(db: Session, scn: ReconstructionScenario) -> dict:
    events = list(db.execute(select(Event).where(Event.scenario_id == scn.id)).scalars())
    summaries = {s.event_id: s for s in db.execute(select(EventSummary).where(EventSummary.scenario_id == scn.id)).scalars()}
    source_ids = [((e.provenance or {}).get("source_event_id")) for e in events]
    source_ids = [i for i in source_ids if i]
    sources = {}
    if source_ids:
        sources = {s.id: s for s in db.execute(select(SourceEvent).where(SourceEvent.id.in_(source_ids))).scalars()}
    rows = []
    for event in events:
        row = _row_from_event(event, sources.get((event.provenance or {}).get("source_event_id")), summaries.get(event.id))
        if row is not None:
            rows.append(row)
    gaps = [{"gap_type": g.gap_type} for g in db.execute(select(GapRecord).where(GapRecord.scenario_id == scn.id)).scalars()]
    evidence_ids = list(dict.fromkeys(r["evidence_id"] for r in rows if r.get("evidence_id")))
    return build_view(rows, gaps, evidence_ids)


def read_narrative(db: Session, scn: ReconstructionScenario) -> dict:
    """Stored at build time. A missing document is not a new model call."""
    stored = (scn.assumptions or {}).get("incident_narrative")
    if isinstance(stored, dict) and stored.get("deterministic"):
        doc = {"scenario_id": scn.id, **stored}
        doc.setdefault("hypothesis_explanation", hypothesis_explanation(db, scn))
        return doc
    view = view_for_scenario(db, scn)
    return {
        "scenario_id": scn.id,
        "contact_status": view["contact_status"],
        "deterministic": view["deterministic"],
        "hypothesis_explanation": hypothesis_explanation(db, scn),
        "ai_narrative_status": "unavailable",
        "ai_narrative_reason": "request_failed",
        "ai_narrative": None,
    }


def frames_for(db: Session, item: dict) -> list[dict]:
    wanted = []
    for number in item.get("frames") or []:
        if number not in wanted:
            wanted.append(number)
    if not item.get("evidence_id") or not wanted:
        return []
    found = {f.frame_no: f for f in db.execute(select(ExtractedFrame).where(
        ExtractedFrame.evidence_id == item["evidence_id"], ExtractedFrame.frame_no.in_(wanted))).scalars()}
    items = []
    for number in wanted:
        frame = found.get(number)
        if frame is None:
            continue
        items.append({
            "evidence_id": item["evidence_id"],
            "frame_id": frame.id,
            "frame_no": frame.frame_no,
            "time_ms": frame.time_ms,
            "thumbnail_url": f"/api/v1/frames/{frame.id}/image" if frame.thumbnail_path else None,
            "full_url": f"/api/v1/frames/{frame.id}/image?full=true",
        })
    return items


def rows_from_generated(events: list[dict], evidence_id: str = "ev_generated") -> list[dict]:
    """Wrap video_events.generate output so the investigator list can be asserted without a database."""
    rows = []
    for index, event in enumerate(events):
        obs = event["observations"]
        hint = event.get("entity_hint") or {}
        rows.append({
            "id": f"evt_{index}", "source_event_id": f"se_{index}", "event_type": event["event_type"],
            "category": event.get("category") or "machine_detection", "title": event["event_type"].replace("_", " "),
            "start_ms": event["t_start_ms"], "end_ms": event["t_end_ms"], "uncertainty_ms": event["uncertainty_ms"],
            "uncertainty_basis": "sampling", "evidence_id": evidence_id,
            "track_ids": list(obs.get("track_ids") or []),
            "track_classes": [t.get("class") for t in hint.get("tracks") or [] if t.get("class")],
            "frames": list((event.get("source_ref") or {}).get("frames") or []),
            "frame_times_ms": list((event.get("source_ref") or {}).get("frame_times_ms") or []),
            "observations": obs, "rule_id": event.get("rule_id"), "source_type": "video_cctv",
        })
    return rows
