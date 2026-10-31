"""Scoring of collision-type alerts against hand-labelled windows. Pure functions; no I/O.

Honest-by-construction rules:
- an alert is only a true or false positive where the clip's content was reviewed well enough to say so (labels.fully_reviewed or a
  negative clip); alerts in positives with no located window are listed separately and never counted either way;
- every number carries its counts, so a reader can see how small n is; a Wilson interval is attached;
- when there is nothing to score the value is None and the caller must print "cannot be established".
"""
import math

ALERT_TYPES = ("possible_collision",)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)


def _ratio(k: int, n: int) -> dict:
    return {"value": round(k / n, 3) if n else None, "k": k, "n": n, "wilson95": wilson(k, n)}


def _is_alert(e: dict, types=ALERT_TYPES) -> bool:
    return e["event_type"] in types


def match_window(alert: dict, window: dict, before: float, after: float) -> bool:
    onset = window.get("onset_ms")
    if onset is None:
        lo, hi = window["onset_range_ms"]
        onset_lo, onset_hi = lo, hi
    else:
        onset_lo = onset_hi = onset
    end = alert.get("t_end_ms") if alert.get("t_end_ms") is not None else alert["t_start_ms"]
    return alert["t_start_ms"] <= onset_hi + after and end >= onset_lo - before


def score(results: dict[str, list[dict]], labels: dict, duration_s: dict[str, float] | None = None, types=ALERT_TYPES) -> dict:
    """results: clip id -> events. labels: parsed eval_data/real_clips.json."""
    cfg = labels["scoring"]["match_window_ms"]
    tp = fp = fn = 0
    errors: list[float] = []
    per_clip, unscored = {}, {}
    neg_clips = neg_with_alert = 0
    fp_alerts_total = 0
    reviewed_minutes = 0.0
    for clip in labels["clips"]:
        cid = clip["id"]
        if cid not in results:
            continue
        alerts = [e for e in results[cid] if _is_alert(e, types)]
        wins = [w for w in clip.get("event_windows", []) if w["type"] == "visible_collision"]
        no_collision_windows = [w for w in clip.get("event_windows", []) if w["type"] != "visible_collision"]
        entry = {"alerts": len(alerts), "windows": len(wins)}
        scored = bool(wins) or clip["accident_present"] is False
        if clip["accident_present"] and not wins:
            unscored[cid] = {"alerts": len(alerts), "reason": "accident present per source label but impact not located by annotator"}
            per_clip[cid] = entry | {"scored": False}
            continue
        used = set()
        c_tp = c_fn = 0
        for w in wins:
            hits = [i for i, a in enumerate(alerts) if match_window(a, w, cfg["before"], cfg["after"])]
            if hits:
                c_tp += 1
                used.update(hits)
                best = min(hits, key=lambda i: abs(alerts[i]["t_start_ms"] - (w.get("onset_ms") or sum(w["onset_range_ms"]) / 2)))
                ref = w.get("onset_ms") or sum(w["onset_range_ms"]) / 2
                errors.append(abs(alerts[best]["t_start_ms"] - ref))
            else:
                c_fn += 1
        c_fp = len(alerts) - len(used)
        tp, fn, fp = tp + c_tp, fn + c_fn, fp + c_fp
        fp_alerts_total += c_fp
        if duration_s and cid in duration_s:
            reviewed_minutes += duration_s[cid] / 60
        if clip["accident_present"] is False:
            neg_clips += 1
            neg_with_alert += 1 if alerts else 0
        per_clip[cid] = entry | {"scored": scored, "tp": c_tp, "fn": c_fn, "fp": c_fp, "braking_only_window": bool(no_collision_windows)}
    n_scored_clips = sum(1 for v in per_clip.values() if v.get("scored"))
    prec = _ratio(tp, tp + fp)
    rec = _ratio(tp, tp + fn)
    f1 = None
    if prec["value"] is not None and rec["value"] is not None and (prec["value"] + rec["value"]) > 0:
        f1 = round(2 * prec["value"] * rec["value"] / (prec["value"] + rec["value"]), 3)
    return {
        "alert_types": list(types),
        "counts": {"tp": tp, "fp": fp, "fn": fn, "scored_clips": n_scored_clips, "negative_clips": neg_clips},
        "precision": prec, "recall": rec, "f1": f1,
        "false_positive_alerts_per_scored_clip": round(fp_alerts_total / n_scored_clips, 3) if n_scored_clips else None,
        "false_positive_alerts_per_scored_minute": round(fp_alerts_total / reviewed_minutes, 3) if reviewed_minutes else None,
        "negative_clips_with_any_alert": _ratio(neg_with_alert, neg_clips),
        "timing_error_ms": {"n": len(errors), "mae": round(sum(errors) / len(errors), 1) if errors else None, "values": [round(x, 1) for x in errors]},
        "per_clip": per_clip, "unscored_positive_clips": unscored,
        "caveat": "Tiny, hand-labelled set; thresholds were chosen while looking at these same clips, so these numbers are optimistic and do not estimate performance on other footage.",
    }
