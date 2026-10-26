"""Time one FloodGate admission pass and record it in results/admission_timing.json.

The timing depends on the machine (and the window on the Python hash seed), so it is measured once here and read by
analyze.py, which therefore stays deterministic. Usage: python experiments/time_admission.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

from fb.alarms import Attack, build_window   # noqa: E402
from fb.floodgate import FloodGate           # noqa: E402
from fb.grid import build_episode            # noqa: E402
from fb.madb import build_madb               # noqa: E402

RATE, REPEATS = 1000, 200   # the highest flood rate of the study


def main():
    win = build_window(build_madb("case39"), build_episode("case39", "C1", 0), Attack.A1, RATE, seed=0)
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        FloodGate().admit(win)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    out = {"median_ms": 1000 * ts[len(ts) // 2], "n_alarms": len(win.alarms), "flood_rate": RATE,
           "repeats": REPEATS, "python": sys.version.split()[0]}
    (HERE.parent / "results" / "admission_timing.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
