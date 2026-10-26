"""Alarm/historian tool service for the triage agent (in-process; mirrors the MCP tool surface).

The same four tools are exposed to the LLM whether or not FloodGate is in front of the stream. Two deployment
factors are modelled because they decide whether the flood even bites:
  * severity_filter : whether `list_alarms` accepts a `priority` argument (an affordance a competent agent would use)
  * severity_source : "madb" (trustworthy, rationalized) or "device" (spoofable by whoever annunciates the point)

Without FloodGate, `list_alarms` returns the raw chronological stream in pages; the critical alarm can be buried by
volume or position. With FloodGate, the service serves the admission-controlled view (see fb.floodgate).
"""
from __future__ import annotations

from dataclasses import dataclass

from .alarms import Alarm, AlarmWindow, HistorianUnavailable, query_historian
from .madb import MADB

PAGE_SIZE = 50
MAX_LIST = 60  # a single list_alarms call never returns more than this many rows (context safety on the tool side)


@dataclass
class ToolConfig:
    floodgate: bool = False
    severity_filter: bool = True       # does list_alarms accept a priority arg?
    severity_source: str = "madb"      # "madb" | "device"  (device severity is spoofable)
    expose_raw_message: bool = False   # include device raw_message text (content-injection sanity check)
    page_size: int = PAGE_SIZE


class AlarmService:
    def __init__(self, win: AlarmWindow, cfg: ToolConfig):
        self.win = win
        self.cfg = cfg
        self.madb: MADB = win.madb
        self.calls: list[dict] = []     # audit log of tool calls
        self._view: list[Alarm] | None = None
        self._digest: dict | None = None

    # --- the admission-controlled or raw view the tools serve from ---
    def view(self) -> list[Alarm]:
        if self._view is None:
            if self.cfg.floodgate:
                from .floodgate import FloodGate
                self._view, self._digest = FloodGate().admit(self.win)
            else:
                self._view = self.win.sorted_alarms()
                self._digest = None
        return self._view

    def _severity(self, a: Alarm) -> int:
        # FloodGate always re-derives severity from the MADB (see fb.floodgate), so device spoofing cannot reach it.
        if self.cfg.severity_source == "device" and not self.cfg.floodgate:
            return a.dev_priority()  # device-reported severity, which the attacker can spoof
        return self.madb.get(a.point_id).priority if self.madb.get(a.point_id) else a.priority

    def _fmt(self, a: Alarm) -> dict:
        d = {"id": a.point_id, "t": round(a.ts), "priority": self._severity(a),
             "zone": a.zone, "desc": a.description}
        if self.cfg.expose_raw_message and not self.cfg.floodgate:
            d["message"] = a.raw_message
        return d

    # --- tools ---
    def list_alarms(self, priority: int | None = None, zone: str | None = None, page: int = 0) -> dict:
        self.calls.append({"tool": "list_alarms", "priority": priority, "zone": zone, "page": page})
        rows = self.view()
        # Treat empty / 0 / "null"/"none"/"all"/"*" filters as "no filter" (models send these for "unset").
        _NONE = (None, "", 0, "0", "null", "none", "all", "any", "*")
        if str(priority).lower() not in {str(x).lower() for x in _NONE} and self.cfg.severity_filter:
            try:
                rows = [a for a in rows if self._severity(a) <= int(priority)]
            except (TypeError, ValueError):
                pass
        if zone is not None and str(zone).lower() not in {str(x).lower() for x in _NONE}:
            rows = [a for a in rows if a.zone == zone]
        total = len(rows)
        ps = self.cfg.page_size
        start = max(0, int(page)) * ps
        chunk = rows[start:start + min(ps, MAX_LIST)]
        out = {"total_matching": total, "page": page, "page_size": ps,
               "returned": len(chunk), "alarms": [self._fmt(a) for a in chunk]}
        if not self.cfg.severity_filter and priority is not None:
            out["note"] = "priority filtering is not supported by this alarm server; results are chronological"
        if self.cfg.floodgate and self._digest:
            out["suppressed_digest"] = self._digest
        return out

    def get_alarm(self, id: str) -> dict:
        self.calls.append({"tool": "get_alarm", "id": id})
        for a in self.view():
            if a.point_id == id:
                pt = self.madb.get(id)
                return {"id": id, "t": round(a.ts), "priority": self._severity(a), "zone": a.zone,
                        "class": a.alarm_class, "desc": a.description,
                        "setpoint": pt.setpoint if pt else "", "cause_hint": pt.cause_hint if pt else "",
                        "allowable_response_s": pt.art_s if pt else None}
        return {"id": id, "error": "not present in current view"}

    def query_historian(self, tag: str, window_s: float = 300.0) -> dict:
        self.calls.append({"tool": "query_historian", "tag": tag, "window_s": window_s})
        try:
            return query_historian(self.win, tag, window_s)
        except HistorianUnavailable as e:
            # tri-state: unavailable is NOT "no problem"
            return {"tag": tag, "available": False, "error": str(e),
                    "note": "historian unavailable; treat as unknown, do not infer 'normal'"}

    def brief_operator(self, p1_ids, root_cause: str, action: str, disposition: str) -> dict:
        self.calls.append({"tool": "brief_operator", "p1_ids": p1_ids, "root_cause": root_cause,
                           "action": action, "disposition": disposition})
        return {"accepted": True}
