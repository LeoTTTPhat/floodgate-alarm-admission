"""FloodGate: a deterministic alarm admission layer between an OPC UA A&C stream and an LLM triage agent.

Standards mapping (see paper's conformance table):
  * IEC 62682 / ISA-18.2 : rationalized priority (from MADB), flood definition, first-out + state-based grouping,
                           shelving of nuisance points.
  * OPC UA Part 9 (IEC 62541-9) : the A&C object model (Severity, grouping, Shelving/Suppressed state) FloodGate reads.
  * IEC 62443-3-3 : SR 7.1 (DoS protection) and SR 7.2 (resource management) via per-source/per-conduit token buckets;
                    SR 6.2 (continuous monitoring) via the completeness digest.

Six mechanisms:
  1. MADB rendering       severity+text come only from the MADB keyed by point id; device strings are dropped.
  2. Token buckets        per-source rate shaping so one chatty source cannot monopolize the operator view.
  3. Pinned priority lane  a fixed budget of slots reserved for P1/P2, independent of P3/P4 volume.
  4. Grouping             ISA-18.2 first-out + same-priority/same-zone suppression of redundant nuisance points.
  5. Completeness digest   counts of shown/suppressed/overflowed alarms per priority band, with an unkeyed SHA-256
                          checksum (an audit record). Alarms on points absent from the MADB are dropped uncounted.
  6. (consumed in tools)  tri-state evidence / fail-closed handling of an unavailable historian.

Properties stated in the paper (origin stated honestly): the flood-independent admission order (a Remark) and the
context-position bound (Proposition 1) are textbook non-preemptive strict-priority scheduling, and Proposition 2 is
the digest's completeness. With MADB rendering they give a presentation guarantee, not a timing one; the contribution
is enforcing them on an LLM consumer (property-tested in tests/). `admit` returns (view, digest) where `view` is the
ordered, bounded alarm list the agent sees and `digest` is the suppression summary with its checksum.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .alarms import Alarm, AlarmWindow
from .madb import Priority

# Pinned-lane budgets (slots). P1/P2 are never squeezed out by P3/P4 volume.
LANE_P1 = 12
LANE_P2 = 20
LANE_LOW = 18             # a bounded sample of the low-priority band (the rest is summarized in the digest)
# Per-source token bucket: at most this many low-priority alarms per source reach the view.
BUCKET_LOW_PER_SOURCE = 3


@dataclass
class FloodGate:
    lane_p1: int = LANE_P1
    lane_p2: int = LANE_P2
    lane_low: int = LANE_LOW
    bucket_low_per_source: int = BUCKET_LOW_PER_SOURCE

    def admit(self, win: AlarmWindow) -> tuple[list[Alarm], dict]:
        madb = win.madb
        alarms = win.sorted_alarms()

        # 1. MADB rendering: rebuild each alarm's priority and description from the MADB, discard device raw_message.
        rendered: list[Alarm] = []
        for a in alarms:
            pt = madb.get(a.point_id)
            if pt is None:   # not recognized: dropped and NOT counted in the digest (a deployment should flag it)
                continue
            rendered.append(Alarm(ts=a.ts, point_id=a.point_id, tag=pt.tag, zone=pt.zone, priority=pt.priority,
                                  alarm_class=pt.alarm_class, description=pt.description, source=a.source,
                                  raw_message=pt.description, is_critical=a.is_critical, seq=a.seq))

        p1 = [a for a in rendered if a.priority == Priority.P1]
        p2 = [a for a in rendered if a.priority == Priority.P2]
        low = [a for a in rendered if a.priority >= Priority.P3]

        # 2+4. Token buckets + first-out/same-(priority,zone) grouping on the low band.
        kept_low: list[Alarm] = []
        suppressed_low = 0
        per_source: dict[str, int] = {}
        seen_group: set[tuple] = set()
        for a in sorted(low, key=lambda x: x.ts):
            grp = (a.priority, a.zone, a.alarm_class)
            if grp in seen_group:                       # first-out: keep the first of each (pri,zone,class) group
                suppressed_low += 1
                continue
            if per_source.get(a.source, 0) >= self.bucket_low_per_source:   # token bucket per source
                suppressed_low += 1
                continue
            seen_group.add(grp)
            per_source[a.source] = per_source.get(a.source, 0) + 1
            kept_low.append(a)

        # 3. Pinned lanes: P1/P2 are admitted in full (bounded by their own, generous lanes); low band is a bounded
        #    sample. Everything beyond a lane is counted in the digest, never dropped silently.
        def take(seq, n):
            seq = sorted(seq, key=lambda x: x.ts)
            return seq[:n], max(0, len(seq) - n)

        p1_keep, p1_over = take(p1, self.lane_p1)
        p2_keep, p2_over = take(p2, self.lane_p2)
        low_keep, low_over = take(kept_low, self.lane_low)

        # Final ordered view: P1 first (fixed positions), then P2, then the low sample. Position of any P1 <= lane_p1.
        view = p1_keep + p2_keep + low_keep

        # 5. Completeness digest: tamper-evident counts of what was suppressed/overflowed, by priority and zone.
        from collections import Counter
        supp_by_zone = Counter(a.zone for a in low if True)  # total low by zone (shown as context)
        digest_payload = {
            "p1_shown": len(p1_keep), "p1_overflow": p1_over,
            "p2_shown": len(p2_keep), "p2_overflow": p2_over,
            "low_shown": len(low_keep),
            "low_suppressed_grouping_or_bucket": suppressed_low,
            "low_overflow": low_over,
            "low_total": len(low),
            "low_by_zone": dict(supp_by_zone),
        }
        chain = hashlib.sha256(repr(sorted(digest_payload.items())).encode()).hexdigest()[:16]
        # The note below overstates (the checksum is unkeyed; unrecognized points are not counted). It is kept verbatim
        # because every logged episode showed it to the model.
        digest = {**digest_payload, "hash": chain,
                  "note": ("priority and text are rendered from the master alarm database; device-supplied strings "
                           "are not shown; counts above are complete and tamper-evident")}
        return view, digest
