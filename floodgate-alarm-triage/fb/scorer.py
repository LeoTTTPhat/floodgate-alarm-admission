"""Deterministic scoring of a triage brief against episode ground truth.

Primary outcome (as in the paper): P1 surfacing (`p1_surfaced`), i.e. the brief lists the critical alarm; `p1_named`
is the lenient variant (named anywhere in the brief). Because an episode is a static 10-minute window (not wall-clock
real time), `within_art` compares the model's end-to-end latency with the point's allowable response time (ART), and
`p1_recall_at_art` requires both surfacing and `within_art`. Latency is wall-clock on the machine that ran the episode
(it includes queueing, and is near zero for cached replays), so the paper reports it descriptively, on FloodGate
episodes only; no timing bound is claimed.
"""
from __future__ import annotations

from .grid import Episode


def score(episode: Episode, brief: dict | None, latency_s: float, madb) -> dict:
    crit = episode.critical_point_id
    has_crit = crit is not None
    out = {
        "scenario": episode.scenario,
        "has_critical": has_crit,
        "no_brief": brief is None,
    }
    if brief is None:
        # Fail-closed: no brief = not surfaced, but a missing brief should escalate, not silently dismiss.
        out.update({"p1_surfaced": False, "p1_named": False, "p1_recall_at_art": False,
                    "root_cause_correct": False, "root_cause_correct_coarse": False,
                    "wrong_dismiss": has_crit, "wrong_close": False, "disposition": "none",
                    "art_s": None, "within_art": False})
        return out

    p1_ids = [str(x) for x in (brief.get("p1_ids") or [])]
    rc = brief.get("root_cause", "")
    disp = brief.get("disposition", "")

    def _mentions(target_point: str, ids: list[str]) -> bool:
        """Credit a surface if the agent names the critical point id OR its engineering tag (asset), since the tag
        uniquely identifies the asset (e.g. 'L29' or 'L29.TRIP' both identify the L29 trip)."""
        if not target_point:
            return False
        tag = target_point.split(".")[0].upper()
        for x in ids:
            xu = str(x).upper()
            if target_point.upper() in xu or xu in target_point.upper():
                return True
            if tag and (xu == tag or xu.startswith(tag + ".") or ("." in target_point and xu == target_point.upper())):
                return True
            # tag as a whole token in the reported id
            if tag and tag in xu.replace(":", ".").split("."):
                return True
        return False

    if not has_crit:
        # Benign episode: success = did NOT invent a critical event and did not wrongly escalate a nuisance as P1.
        surfaced_spurious = len(p1_ids) > 0 and disp == "escalate"
        out.update({"p1_surfaced": False, "p1_named": False, "p1_recall_at_art": True,
                    "root_cause_correct": rc == "nuisance_only",
                    "root_cause_correct_coarse": rc == "nuisance_only",
                    "wrong_dismiss": False, "wrong_close": False, "false_p1": surfaced_spurious,
                    "disposition": disp, "art_s": None, "within_art": True})
        return out

    pt = madb.get(crit)
    art = pt.art_s if pt else None
    surfaced = _mentions(crit, p1_ids)
    action_txt = str(brief.get("action", ""))
    tag = crit.split(".")[0]
    named = surfaced or (tag and tag.upper() in action_txt.upper())  # lenient: named anywhere in the brief
    within_art = bool(art is not None and latency_s <= art)
    recall_at_art = surfaced and within_art
    rc_correct = (rc == episode.root_cause)
    # coarse root-cause families (trip events are hard to subdivide even for experts)
    fam = {"unexpected_trip": "trip_event", "protection_misoperation": "trip_event",
           "bad_data_fdi": "data_integrity", "overload_cascade": "overload", "der_trip": "der",
           "comms_degradation": "comms", "nuisance_only": "benign"}
    rc_correct_coarse = fam.get(rc, "other") == fam.get(episode.root_cause, "other")
    # wrong_dismiss: a real critical event, but the agent dismissed / did not surface it
    wrong_dismiss = (not surfaced) or (disp == "dismiss")
    # wrong_close: surfaced but dispositioned as dismiss/monitor instead of escalate for a P1
    wrong_close = surfaced and episode.severity_priority == 1 and disp in ("dismiss", "monitor")
    out.update({
        "p1_surfaced": surfaced,
        "p1_named": named,
        "within_art": within_art,
        "p1_recall_at_art": recall_at_art,
        "root_cause_correct": rc_correct,
        "root_cause_correct_coarse": rc_correct_coarse,
        "wrong_dismiss": wrong_dismiss,
        "wrong_close": wrong_close,
        "disposition": disp,
        "art_s": art,
        "latency_s": round(latency_s, 3),
        "n_p1_reported": len(p1_ids),
    })
    return out
