"""Run one full episode: build window -> run agent over the tool service -> score. This is the unit of work the
experiments fan out over (via tools/jobrunner.py). Deterministic given the job spec.
"""
from __future__ import annotations

from functools import lru_cache

from .agent import run_agent, run_rule_agent
from .alarms import Attack, HistorianMode, build_window
from .grid import build_episode
from .madb import build_madb
from .scorer import score
from .tools import AlarmService, ToolConfig


@lru_cache(maxsize=4)
def _madb(grid: str):
    return build_madb(grid)


def run_one(job: dict) -> dict:
    """job keys: grid, scenario, seed, model, attack, flood_rate, position, historian, num_ctx,
    floodgate, severity_filter, severity_source, expose_raw_message, protocol, inject_text."""
    grid = job.get("grid", "case39")
    madb = _madb(grid)
    ep = build_episode(grid, job["scenario"], int(job["seed"]))
    win = build_window(
        madb, ep,
        attack=Attack(job.get("attack", "none")),
        flood_rate=int(job.get("flood_rate", 0)),
        position=job.get("position", "interleaved"),
        historian=HistorianMode(job.get("historian", "normal")),
        seed=int(job["seed"]),
        inject_text=job.get("inject_text"),
        spoof_critical_severity=bool(job.get("spoof_critical_severity", False)),
    )
    cfg = ToolConfig(
        floodgate=bool(job.get("floodgate", False)),
        severity_filter=bool(job.get("severity_filter", True)),
        severity_source=job.get("severity_source", "madb"),
        expose_raw_message=bool(job.get("expose_raw_message", False)),
    )
    svc = AlarmService(win, cfg)
    # presentation metric (no LLM): is the genuine critical alarm present in the admitted/raw view the agent sees?
    view = svc.view()
    p1_present_in_view = bool(ep.critical_point_id and any(a.point_id == ep.critical_point_id for a in view))
    if job.get("exp") == "Eguard2":
        # Measure whether a content guardrail (Llama Guard 3) flags the text-free flood. It sees the rendered alarm
        # list; since no alarm carries attacker text, a content classifier has nothing to flag (expected ~0).
        from .guard import guard_flags
        gf = guard_flags(svc.view(), model=job.get("guard_model", "llama-guard3:8b"))
        return {"scenario": ep.scenario, "grid": grid, "model": job["model"], "seed": int(job["seed"]),
                "attack": job.get("attack", "none"), "flood_rate": int(job.get("flood_rate", 0)),
                "floodgate": cfg.floodgate, "guard_flagged": bool(gf["flagged"]), "guard_raw": gf["raw"][:200]}
    if job["model"] == "rule":
        res = run_rule_agent(svc)
    else:
        res = run_agent(svc, job["model"], num_ctx=int(job.get("num_ctx", 8192)), seed=int(job["seed"]),
                        protocol=job.get("protocol", "direct"))
    sc = score(ep, res["brief"], res["latency_s"], madb)
    return {
        **sc,
        "grid": grid, "model": job["model"], "seed": int(job["seed"]), "attack": job.get("attack", "none"),
        "flood_rate": int(job.get("flood_rate", 0)), "position": job.get("position", "interleaved"),
        "historian": job.get("historian", "normal"), "num_ctx": int(job.get("num_ctx", 8192)),
        "floodgate": cfg.floodgate, "severity_filter": cfg.severity_filter,
        "severity_source": cfg.severity_source, "protocol": job.get("protocol", "direct"),
        "spoof": bool(job.get("spoof_critical_severity", False)),
        "p1_present_in_view": p1_present_in_view,
        "n_alarms_total": win.meta["n_total"], "n_flood": win.meta["n_flood"],
        "n_view": res["n_alarms_in_view"], "n_steps": res["n_steps"], "latency_s": res["latency_s"],
        "eval_tokens": res["eval_tokens"], "brief": res["brief"], "window_digest": win.digest(),
    }
