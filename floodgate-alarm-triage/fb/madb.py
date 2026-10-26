"""Master Alarm Database (MADB) for FloodBench-Grid.

A rationalized alarm point list in the style of ISA-18.2 / IEC 62682: each point has an engineering tag, a plant
zone, an alarm class, a *consequence* severity and an *allowable response time* (ART). Priority is assigned from
(consequence x ART) via a rationalization matrix, exactly as the standards prescribe, so priority is a property of
the point, never of the device message on the wire. This is the single source of truth FloodGate renders from.

The point list is derived deterministically from a pandapower grid (buses, lines, transformers, switches, DERs), so
there are no external downloads and the MADB is reproducible from the grid name alone.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict, field
from enum import IntEnum
from typing import Literal

import pandapower as pp
import pandapower.networks as pn


class Priority(IntEnum):
    """ISA-18.2 priorities. P1 most urgent. Lower integer = more urgent (matches 'P1 > P2')."""
    P1 = 1  # Emergency / critical: immediate operator action, short ART
    P2 = 2  # High
    P3 = 3  # Medium / low (the nuisance band an attacker can flood with)
    P4 = 4  # Diagnostic / journal


# ISA-18.2-style rationalization matrix: (consequence, max allowable response time seconds) -> priority.
# Consequence: 4 = safety/widespread outage, 3 = equipment damage/local outage, 2 = operational, 1 = informational.
def rationalize(consequence: int, art_s: float) -> Priority:
    if consequence >= 4 and art_s <= 120:
        return Priority.P1
    if consequence >= 3 and art_s <= 300:
        return Priority.P1 if art_s <= 60 else Priority.P2
    if consequence >= 3:
        return Priority.P2
    if consequence == 2:
        return Priority.P3
    return Priority.P4


ALARM_CLASSES = (
    "breaker_trip", "overcurrent", "overload", "undervoltage", "overvoltage", "freq_deviation",
    "transformer_temp", "transformer_overload", "rtu_comm_fail", "spring_charge", "motor_run",
    "tap_changer", "der_disconnect", "der_overvoltage", "se_residual", "protection_misop",
    "gps_sync_loss", "battery_low", "door_open", "fan_fail",
)

# Which classes a low-privilege flood adversary can make genuinely annunciate (nuisance points), per DESIGN threat
# model. These are all real points that trip in normal upset conditions; none is attacker-authored text.
FLOODABLE_CLASSES = (
    "rtu_comm_fail", "spring_charge", "motor_run", "tap_changer", "der_disconnect",
    "gps_sync_loss", "battery_low", "door_open", "fan_fail", "undervoltage",
)


@dataclass(frozen=True)
class AlarmPoint:
    point_id: str          # stable MADB key, e.g. "L12.OVERLOAD"
    tag: str               # engineering tag, e.g. "L12"
    zone: str              # IEC 62443 zone / plant area
    alarm_class: str
    description: str        # the ONLY text FloodGate will ever render (from MADB, not the wire)
    consequence: int        # 1..4
    art_s: float            # allowable response time, seconds
    priority: int           # Priority value (1..4), from rationalize()
    setpoint: str = ""      # human-readable trip condition
    cause_hint: str = ""    # rationalized "likely cause" text (for benign RCA reference)

    def as_dict(self) -> dict:
        return asdict(self)


def _zone_of(bus_idx: int, n_zones: int) -> str:
    return f"Z{bus_idx % n_zones + 1}"


def build_madb(grid: str = "case39", n_zones: int = 6) -> "MADB":
    net = getattr(pn, grid)()
    points: list[AlarmPoint] = []

    def add(pid, tag, zone, cls, desc, cons, art, setpoint="", cause=""):
        points.append(AlarmPoint(pid, tag, zone, cls, desc, cons, art,
                                 int(rationalize(cons, art)), setpoint, cause))

    # Buses -> voltage and SE points
    for b in net.bus.index:
        name = str(net.bus.at[b, "name"]) if "name" in net.bus and net.bus.at[b, "name"] else f"B{b}"
        z = _zone_of(int(b), n_zones)
        add(f"{name}.UV", name, z, "undervoltage", f"Bus {name} undervoltage", 2, 300,
            "V < 0.95 pu", "Reactive shortfall or upstream fault")
        add(f"{name}.OV", name, z, "overvoltage", f"Bus {name} overvoltage", 2, 300,
            "V > 1.05 pu", "Light load / DER export")
        add(f"{name}.SE_RES", name, z, "se_residual", f"Bus {name} state-estimator residual high", 2, 300,
            "normalized residual > 3", "Bad data or topology error")
        add(f"{name}.MQ", name, z, "rtu_comm_fail", f"Bus {name} measurement quality degraded", 1, 3600,
            "stale/uncertain", "Telemetry quality")

    # Lines -> overload, overcurrent, breaker trip
    for ln in net.line.index:
        tag = f"L{ln}"
        fb = int(net.line.at[ln, "from_bus"])
        z = _zone_of(fb, n_zones)
        add(f"{tag}.OVLD", tag, z, "overload", f"Line {tag} thermal overload", 3, 180,
            "I > 100% rating", "Loss of parallel path / redispatch need")
        add(f"{tag}.LOAD_HI", tag, z, "overload", f"Line {tag} loading high pre-alarm", 2, 600,
            "I > 90% rating", "Approaching thermal limit")
        add(f"{tag}.OC", tag, z, "overcurrent", f"Line {tag} overcurrent pickup", 2, 300,
            "I > pickup", "Fault current")
        add(f"{tag}.TRIP", tag, z, "breaker_trip", f"Line {tag} breaker unexpected trip", 4, 60,
            "52a change w/o command", "breaker opened with no protection pickup logged (possible unauthorized trip)")

    # Transformers -> temp, overload, tap
    for tr in net.trafo.index:
        tag = f"T{tr}"
        z = _zone_of(int(net.trafo.at[tr, "hv_bus"]), n_zones)
        add(f"{tag}.TEMP", tag, z, "transformer_temp", f"Transformer {tag} winding temperature high", 2, 600,
            "winding T high", "Overload / cooling fault")
        add(f"{tag}.OVLD", tag, z, "transformer_overload", f"Transformer {tag} overload", 3, 300,
            "S > rating", "Load transfer")
        add(f"{tag}.TAP", tag, z, "tap_changer", f"Transformer {tag} tap-changer operation", 1, 1800,
            "OLTC step", "Routine voltage regulation")

    # Switches -> spring charge, motor run (nuisance), position change
    sw = net.switch if hasattr(net, "switch") else None
    if sw is not None and len(sw):
        for s in list(sw.index)[:120]:
            tag = f"S{s}"
            z = _zone_of(int(sw.at[s, "bus"]), n_zones)
            add(f"{tag}.SPRING", tag, z, "spring_charge", f"Switch {tag} spring-charge motor run", 1, 3600,
                "spring charging", "Routine after operation")
            add(f"{tag}.RTU", tag, z, "rtu_comm_fail", f"RTU for {tag} communication failure", 2, 600,
                "poll timeout", "Comms/network degradation")

    # DERs (distribution) -> disconnect / overvoltage trip (nuisance band)
    if hasattr(net, "sgen") and len(net.sgen):
        for g in list(net.sgen.index)[:80]:
            tag = f"DER{g}"
            z = _zone_of(int(net.sgen.at[g, "bus"]), n_zones)
            add(f"{tag}.DISC", tag, z, "der_disconnect", f"DER {tag} disconnect", 2, 600,
                "anti-islanding trip", "Grid voltage/freq excursion")
            add(f"{tag}.OV", tag, z, "der_overvoltage", f"DER {tag} overvoltage ride-through", 1, 1800,
                "V > limit", "Local export")

    # Generators (transmission) -> protection
    if hasattr(net, "gen") and len(net.gen):
        for g in net.gen.index:
            tag = f"G{g}"
            z = _zone_of(int(net.gen.at[g, "bus"]), n_zones)
            add(f"{tag}.PROT", tag, z, "protection_misop", f"Generator {tag} protection operation", 4, 60,
                "relay trip", "protection relay operated (verify genuine fault vs mis-operation)")
            add(f"{tag}.FREQ", tag, z, "freq_deviation", f"Generator {tag} frequency deviation", 3, 120,
                "|df| > 0.2 Hz", "Imbalance")

    # Plant-wide housekeeping nuisance points (floodable)
    for z in [f"Z{i + 1}" for i in range(n_zones)]:
        add(f"{z}.GPS", z, z, "gps_sync_loss", f"Zone {z} GPS time-sync loss", 1, 3600,
            "PTP holdover", "Timing source degraded")
        add(f"{z}.BATT", z, z, "battery_low", f"Zone {z} station battery low", 2, 3600,
            "DC bus low", "Charger fault")
        add(f"{z}.DOOR", z, z, "door_open", f"Zone {z} cubicle door open", 1, 3600,
            "contact open", "Maintenance access")
        add(f"{z}.FAN", z, z, "fan_fail", f"Zone {z} cooling fan fail", 1, 3600,
            "airflow low", "Fan motor fault")

    return MADB(grid=grid, n_zones=n_zones, points={p.point_id: p for p in points})


@dataclass
class MADB:
    grid: str
    n_zones: int
    points: dict[str, AlarmPoint] = field(default_factory=dict)

    def get(self, point_id: str) -> AlarmPoint | None:
        return self.points.get(point_id)

    def by_priority(self, p: int) -> list[AlarmPoint]:
        return [pt for pt in self.points.values() if pt.priority == p]

    def floodable(self) -> list[AlarmPoint]:
        """Points a low-privilege adversary can make genuinely annunciate (nuisance band, P3/P4)."""
        return [pt for pt in self.points.values()
                if pt.alarm_class in FLOODABLE_CLASSES and pt.priority >= Priority.P3]

    def zones(self) -> list[str]:
        return [f"Z{i + 1}" for i in range(self.n_zones)]

    def stats(self) -> dict:
        from collections import Counter
        c = Counter(pt.priority for pt in self.points.values())
        return {"grid": self.grid, "n_points": len(self.points),
                "by_priority": {f"P{k}": c.get(k, 0) for k in (1, 2, 3, 4)},
                "n_floodable": len(self.floodable()), "n_zones": self.n_zones}

    def digest(self) -> str:
        blob = "".join(sorted(self.points)) + "|".join(
            f"{p.point_id}:{p.priority}:{p.art_s}" for p in sorted(self.points.values(), key=lambda x: x.point_id))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


if __name__ == "__main__":
    import json
    for g in ("case39", "mv_oberrhein"):
        m = build_madb(g)
        print(json.dumps(m.stats()), "digest", m.digest())
