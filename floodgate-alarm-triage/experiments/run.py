"""Run one experiment (or 'all') via the resumable job runner. Launch under the memory guard:

    cd floodgate-alarm-triage
    PYTHONPATH=<repo> ../.venv/bin/python ../tools/memguard.py run --workers 6 -- \
        ../.venv/bin/python experiments/run.py E1

Results stream to results/<exp>.jsonl (one line per finished episode). Re-running skips finished jobs.
The per-call worker budget is taken from memguard (logs/memguard.workers); --workers here is only the initial pool.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent))

from tools.jobrunner import run_jobs  # noqa: E402
from fb.episode import run_one        # noqa: E402
from experiments.jobs import REGISTRY  # noqa: E402

RESULTS = HERE.parent / "results"


def main():
    names = sys.argv[1:] or ["E1"]
    if names == ["all"]:
        names = list(REGISTRY)
    workers = int(os.environ.get("CSI_WORKERS", "6"))
    for name in names:
        if name not in REGISTRY:
            print(f"unknown experiment {name}; known: {list(REGISTRY)}")
            continue
        jobs = REGISTRY[name]()
        out = RESULTS / f"{name}.jsonl"
        print(f"=== {name}: {len(jobs)} jobs -> {out}", flush=True)
        run_jobs(jobs, run_one, out, workers=workers)
    print("done", flush=True)


if __name__ == "__main__":
    main()
