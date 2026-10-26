"""Physics layer for FloodBench-Grid.

Produces physically consistent ground truth for each episode: a base power-flow state, one injected *critical event*
(a contingency), the post-event state, and the historian time series the agent can query. Everything is derived from
pandapower built-in grids with no downloads, and is deterministic given (grid, scenario, seed).

Critical events (the P1/P2 the agent must catch and diagnose):
  C1  unauthorized breaker opening of a heavily loaded line  -> P1 "unexpected trip", may cause downstream overload
  C2  false-data injection on a tie-line measurement         -> P2 state-estimator residual (no real physical change)
  C3  protection mis-operation tripping a generator/line     -> P1 protection operation
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandapower as pp
import pandapower.networks as pn

warnings.filterwarnings("ignore")

# Fixed root-cause taxonomy the agent must choose from (and the scorer grades against).
ROOT_CAUSES = (
    "unexpected_trip",          # C1
    "protection_misoperation",  # C3
    "bad_data_fdi",             # C2
    "overload_cascade",         # secondary consequence
    "der_trip",
    "comms_degradation",
    "nuisance_only",            # no genuine critical event (benign)
)

CRITICAL = {
    "C1": {"root_cause": "unexpected_trip", "priority": 1,
           "action": "confirm trip, assess N-1, prepare redispatch/reclose per policy"},
    "C2": {"root_cause": "bad_data_fdi", "priority": 2,
           "action": "flag SE bad data, cross-check redundant metering, do not act on single residual"},
    "C3": {"root_cause": "protection_misoperation", "priority": 1,
           "action": "verify no fault, investigate relay mis-operation, restore per policy"},
}


@dataclass
class Episode:
    grid: str
    scenario: str          # C1 | C2 | C3 | BENIGN
    seed: int
    critical_point_id: str | None   # the MADB point id of the genuine critical alarm (None for BENIGN)
    root_cause: str
    correct_action: str
    affected_tag: str
    severity_priority: int
    # physics-derived facts the historian exposes:
    pre_loading: dict = field(default_factory=dict)    # line tag -> loading %
    post_loading: dict = field(default_factory=dict)
    pre_vm: dict = field(default_factory=dict)          # bus name -> vm pu
    post_vm: dict = field(default_factory=dict)
    secondary_overloads: list = field(default_factory=list)  # tags that overload after C1
    notes: str = ""


@lru_cache(maxsize=8)
def _solved_base(grid: str):
    net = getattr(pn, grid)()
    pp.runpp(net, numba=True)
    return net


def _bus_name(net, b: int) -> str:
    v = net.bus.at[b, "name"] if "name" in net.bus else None
    return str(v) if v else f"B{b}"


def _loading(net) -> dict:
    return {f"L{i}": round(float(net.res_line.at[i, "loading_percent"]), 2) for i in net.line.index}


def _vm(net) -> dict:
    return {_bus_name(net, b): round(float(net.res_bus.at[b, "vm_pu"]), 4) for b in net.bus.index}


def build_episode(grid: str, scenario: str, seed: int) -> Episode:
    rng = np.random.default_rng(seed + 100003 * {"C1": 1, "C2": 2, "C3": 3, "BENIGN": 0}[scenario])
    base = _solved_base(grid)
    pre_loading, pre_vm = _loading(base), _vm(base)

    if scenario == "BENIGN":
        return Episode(grid, scenario, seed, None, "nuisance_only",
                       "monitor; no critical event present", "-", 0,
                       pre_loading, pre_loading, pre_vm, pre_vm, [], "no genuine critical event")

    if scenario == "C1":
        # Open a heavily loaded line (unauthorized). Pick among the top-loaded lines (deterministic by seed).
        order = sorted(base.line.index, key=lambda i: -pre_loading[f"L{i}"])
        cand = order[: max(3, len(order) // 5)]
        ln = int(cand[rng.integers(len(cand))])
        tag = f"L{ln}"
        net = getattr(pn, grid)()
        net.line.at[ln, "in_service"] = False
        secondary = []
        post_loading, post_vm = pre_loading, pre_vm
        try:
            pp.runpp(net, numba=True)
            post_loading, post_vm = _loading(net), _vm(net)
            secondary = [t for t, v in post_loading.items() if v > 100.0 and t != tag]
        except pp.LoadflowNotConverged:
            secondary = ["<diverged: islanding/collapse>"]
        return Episode(grid, scenario, seed, f"{tag}.TRIP", "unexpected_trip",
                       CRITICAL["C1"]["action"], tag, 1, pre_loading, post_loading, pre_vm, post_vm, secondary,
                       f"unauthorized opening of {tag} (pre-load {pre_loading[tag]}%)")

    if scenario == "C2":
        # FDI on a line's flow measurement -> SE residual alarm on an endpoint bus. No real physical change.
        order = sorted(base.line.index, key=lambda i: -pre_loading[f"L{i}"])
        ln = int(order[rng.integers(min(5, len(order)))])
        bus = int(base.line.at[ln, "from_bus"])
        name = _bus_name(base, bus)
        return Episode(grid, scenario, seed, f"{name}.SE_RES", "bad_data_fdi",
                       CRITICAL["C2"]["action"], name, 2, pre_loading, pre_loading, pre_vm, pre_vm, [],
                       f"injected residual on {name} (measurement of L{ln}); physical state unchanged")

    if scenario == "C3":
        # Protection mis-operation: trip a generator (transmission) or a line (if no gen).
        if hasattr(base, "gen") and len(base.gen):
            g = int(base.gen.index[rng.integers(len(base.gen))])
            tag = f"G{g}"
            net = getattr(pn, grid)()
            net.gen.at[g, "in_service"] = False
            post_loading, post_vm, secondary = pre_loading, pre_vm, []
            try:
                pp.runpp(net, numba=True)
                post_loading, post_vm = _loading(net), _vm(net)
                secondary = [t for t, v in post_loading.items() if v > 100.0]
            except pp.LoadflowNotConverged:
                secondary = ["<diverged>"]
            return Episode(grid, scenario, seed, f"{tag}.PROT", "protection_misoperation",
                           CRITICAL["C3"]["action"], tag, 1, pre_loading, post_loading, pre_vm, post_vm, secondary,
                           f"spurious protection trip of {tag} (no fault present)")
        else:
            order = sorted(base.line.index, key=lambda i: -pre_loading[f"L{i}"])
            ln = int(order[rng.integers(min(5, len(order)))])
            tag = f"L{ln}"
            return Episode(grid, scenario, seed, f"{tag}.TRIP", "protection_misoperation",
                           CRITICAL["C3"]["action"], tag, 1, pre_loading, pre_loading, pre_vm, pre_vm, [],
                           f"spurious protection trip of {tag} (no fault present)")

    raise ValueError(scenario)


if __name__ == "__main__":
    for sc in ("C1", "C2", "C3", "BENIGN"):
        e = build_episode("case39", sc, 1)
        print(sc, e.critical_point_id, e.root_cause, "| secondary overloads:", e.secondary_overloads[:5],
              "| note:", e.notes)
