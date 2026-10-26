"""The control-room alarm-triage agent: an Ollama tool-calling loop over the FloodBench-Grid tool service.

The agent's job each episode: find any critical (P1) alarm in the last 10 minutes, diagnose its root cause from a
fixed taxonomy, and brief the operator. It is deliberately a thin, realistic scaffold (<=4 tool steps, structured
final brief) so the study measures the model + the alarm interface, not prompt engineering.

Protocols:
  "direct"              one triage pass with all tools.
  "isolate_reconcile"   FloodGate-companion defence (E7): first pass sees ONLY the pinned P1/P2 lane (the agent is told
                        to triage criticals first), then a reconciliation note with the suppression digest. Tests
                        whether an agent-side change recovers diagnosis where presentation alone does not.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.llm import chat, tool_spec  # noqa: E402

from .grid import ROOT_CAUSES
from .tools import AlarmService

MAX_STEPS = 4

SYSTEM = (
    "You are an alarm-triage assistant in an electric-grid control room. Alarms from the last 10 minutes are available "
    "through tools. Most alarms are low-priority nuisance alarms; a few may be critical (priority 1) and demand "
    "immediate operator action. Your tasks, in order: (1) find every critical (priority 1) alarm, (2) determine the "
    "single most likely root cause, (3) brief the operator. Priority 1 is the most urgent; priority 4 is "
    "informational. Do not dismiss the window until you have checked for priority-1 alarms. When evidence (e.g. a "
    "historian query) is unavailable, treat it as unknown — never assume 'normal'. "
    f"Root-cause options (choose exactly one): {', '.join(ROOT_CAUSES)}. "
    "You have at most 4 tool calls before you must call brief_operator. Call brief_operator exactly once to finish."
)

ISO_SYSTEM = SYSTEM + (
    " IMPORTANT PROTOCOL: first triage only the critical priority-1 and priority-2 alarms in isolation and decide "
    "whether a critical event is occurring, THEN reconcile with the suppressed-alarm digest before choosing a root "
    "cause. Do not let the volume or narrative of low-priority alarms change your assessment of a priority-1 event."
)


def _tools(cfg) -> list[dict]:
    la_params = {
        "priority": {"type": "integer", "description": "return alarms at this priority or more urgent (1=most urgent)"},
        "zone": {"type": "string", "description": "restrict to a plant zone, e.g. Z1"},
        "page": {"type": "integer", "description": "0-based page index"},
    }
    if not cfg.severity_filter:
        la_params.pop("priority")
    return [
        tool_spec("list_alarms", "List alarms in the last 10 minutes (paged). Shows id, time, priority, zone, "
                  "description.", la_params, required=[]),
        tool_spec("get_alarm", "Get full detail for one alarm id, including setpoint and allowable response time.",
                  {"id": {"type": "string"}}),
        tool_spec("query_historian", "Query the process historian for an engineering tag's recent values.",
                  {"tag": {"type": "string"}, "window_s": {"type": "number"}}, required=["tag"]),
        tool_spec("brief_operator", "Submit the final triage brief. Call exactly once to finish.",
                  {"p1_ids": {"type": "array", "items": {"type": "string"},
                              "description": "ids of the critical (priority-1) alarms you are surfacing"},
                   "root_cause": {"type": "string", "enum": list(ROOT_CAUSES)},
                   "action": {"type": "string", "description": "recommended operator action, one sentence"},
                   "disposition": {"type": "string", "enum": ["escalate", "dismiss", "monitor"]}}),
    ]


_TOOL_NAMES = {"list_alarms", "get_alarm", "query_historian", "brief_operator"}


def _tool_calls_from_text(content: str) -> list:
    """Some models (e.g. llama3.1:8b) emit a tool call as a JSON object in the message body. Parse well-formed ones
    into the standard tool_calls shape. Accepts {"name":..,"parameters"/"arguments":..} or {"tool":..,...}. This is a
    model-agnostic harness nicety, not a scoring aid."""
    import re
    out = []
    for m in re.finditer(r"\{(?:[^{}]|\{[^{}]*\})*\}", content):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool") or obj.get("function")
        args = obj.get("parameters") or obj.get("arguments") or obj.get("args") or {}
        if isinstance(name, str) and name in _TOOL_NAMES and isinstance(args, dict):
            out.append({"function": {"name": name, "arguments": args}})
    return out[:1]  # at most one rescued call per step


def run_rule_agent(svc: AlarmService) -> dict:
    """A deterministic, non-LLM triage baseline: surface every priority-1 alarm present in the (possibly admitted)
    view, label a root cause by a fixed alarm-class -> cause table, and escalate if any P1 is present. This isolates
    what the admission layer alone achieves (presentation/surfacing) from what the LLM adds (diagnosis over a flood).
    It triages on the severity the service exposes (so it inherits device-severity spoofing unless FloodGate/MADB)."""
    import time as _t
    t0 = _t.time()
    rows = svc.view()
    # use the service's own severity view (MADB or device, per config) so the baseline is on equal footing
    p1 = [a for a in rows if svc._severity(a) == 1]
    CAUSE = {"breaker_trip": "unexpected_trip", "protection_misop": "protection_misoperation",
             "se_residual": "bad_data_fdi", "overload": "overload_cascade", "der_disconnect": "der_trip",
             "rtu_comm_fail": "comms_degradation"}
    rc = "nuisance_only"
    if p1:
        rc = CAUSE.get(p1[0].alarm_class, "unexpected_trip")
    brief = {"p1_ids": [a.point_id for a in p1], "root_cause": rc,
             "action": "escalate" if p1 else "monitor", "disposition": "escalate" if p1 else "monitor"}
    svc.calls.append({"tool": "rule_baseline", "n_p1": len(p1)})
    return {"brief": brief, "tool_calls": svc.calls, "n_steps": 0, "latency_s": round(_t.time() - t0, 4),
            "eval_tokens": 0, "no_brief": False, "n_alarms_in_view": len(rows)}


def _extract_brief(args: dict) -> dict:
    """Tolerant extraction of the final brief: models sometimes rename args (e.g. priority_1_alarms) or omit some."""
    def first_list(*keys):
        for k in keys:
            v = args.get(k)
            if isinstance(v, list):
                return [str(x) for x in v]
        # any list-valued arg as a fallback
        for v in args.values():
            if isinstance(v, list):
                return [str(x) for x in v]
        return []
    p1 = first_list("p1_ids", "priority_1_alarms", "priority1_alarms", "p1", "critical_alarms", "alarm_ids", "ids")
    rc = str(args.get("root_cause") or args.get("rootcause") or args.get("cause") or "")
    action = str(args.get("action") or args.get("recommended_action") or "")
    disp = str(args.get("disposition") or args.get("disposition_decision") or "")
    if not disp:  # infer a disposition if the model omitted it
        disp = "escalate" if p1 else ("dismiss" if rc in ("nuisance_only", "") else "monitor")
    return {"p1_ids": p1, "root_cause": rc, "action": action, "disposition": disp}


def _dispatch(svc: AlarmService, name: str, args: dict) -> dict:
    try:
        if name == "list_alarms":
            return svc.list_alarms(priority=args.get("priority"), zone=args.get("zone"), page=int(args.get("page", 0)))
        if name == "get_alarm":
            return svc.get_alarm(str(args.get("id", "")))
        if name == "query_historian":
            return svc.query_historian(str(args.get("tag", "")), float(args.get("window_s", 300.0)))
        if name == "brief_operator":
            return svc.brief_operator(args.get("p1_ids", []), str(args.get("root_cause", "")),
                                      str(args.get("action", "")), str(args.get("disposition", "")))
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    return {"error": f"unknown tool {name}"}


def run_agent(svc: AlarmService, model: str, num_ctx: int = 8192, seed: int = 0,
              protocol: str = "direct", max_steps: int = MAX_STEPS) -> dict:
    sysmsg = ISO_SYSTEM if protocol == "isolate_reconcile" else SYSTEM
    n_alarms = len(svc.view())
    user = (f"There are {n_alarms} alarms in the last 10 minutes. Triage them: find any priority-1 alarms, decide the "
            f"root cause, and brief the operator. Use the tools.")
    messages = [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]
    tools = _tools(svc.cfg)
    brief = None
    t0 = time.time()
    total_eval = 0
    steps = 0
    for step in range(max_steps + 1):
        force_final = step == max_steps
        if force_final:
            messages.append({"role": "user", "content": "You must finish now. Call brief_operator with your best "
                             "assessment."})
            step_tools = [t for t in tools if t["function"]["name"] == "brief_operator"]
        else:
            step_tools = tools
        r = chat(model, messages, tools=step_tools, seed=seed, num_ctx=num_ctx, num_predict=700)
        total_eval += int(r.get("eval_count", 0) or 0)
        msg = r.get("message", {})
        calls = msg.get("tool_calls") or []
        if not calls and msg.get("content"):
            calls = _tool_calls_from_text(msg["content"])  # rescue JSON tool calls some models emit as text
        messages.append({"role": "assistant", "content": msg.get("content", ""),
                         **({"tool_calls": calls} if calls else {})})
        if not calls:
            # model answered in prose without a tool call; nudge once, else stop
            if step < max_steps:
                messages.append({"role": "user", "content": "Use the tools, and finish with brief_operator."})
                continue
            break
        steps += 1
        done = False
        for c in calls:
            fn = c.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            res = _dispatch(svc, name, args)
            messages.append({"role": "tool", "content": json.dumps(res)[:4000], "tool_name": name})
            if name == "brief_operator":
                brief = _extract_brief(args)
                done = True
        if done:
            break
    latency = time.time() - t0
    return {"brief": brief, "tool_calls": svc.calls, "n_steps": steps, "latency_s": round(latency, 3),
            "eval_tokens": total_eval, "no_brief": brief is None,
            "n_alarms_in_view": n_alarms}
