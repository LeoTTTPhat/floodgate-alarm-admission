"""Resumable, memory-aware job runner for CSI-26 experiments.

* Every job has a unique ``id``.  Finished jobs are appended (one JSON line) to
  ``out_path``; on restart they are skipped, so a run stopped by
  ``tools/memguard.py`` resumes where it stopped.
* Concurrency is dynamic: before starting each job the runner re-reads the
  worker budget written by memguard (``logs/memguard.workers``; falls back to
  ``CSI_WORKERS`` or the ``workers`` argument) and refuses to start new work
  while the project footprint is above the soft limit.

    from tools.jobrunner import run_jobs
    run_jobs(jobs, fn, "results/exp1.jsonl", workers=4)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import memguard  # noqa: E402

_lock = threading.Lock()


def done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    ids = set()
    for ln in out_path.read_text().splitlines():
        try:
            rec = json.loads(ln)
            if rec.get("ok", True):
                ids.add(rec["id"])
        except Exception:
            continue
    return ids


def _budget(default: int) -> int:
    try:
        return max(1, int(memguard.WORKERS_FILE.read_text().strip()))
    except Exception:
        return max(1, int(os.environ.get("CSI_WORKERS", default)))


def run_jobs(jobs: Iterable[dict], fn: Callable[[dict], dict], out_path: str | Path,
             workers: int = 4, check_every: int = 1, quiet: bool = False) -> int:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    finished = done_ids(out)
    todo = [j for j in jobs if j["id"] not in finished]
    total = len(todo)
    if not quiet:
        print(f"[jobrunner] {out.name}: {len(finished)} done, {total} to run", flush=True)
    if not todo:
        return 0
    max_pool = max(workers, 32)
    pending: set[Future] = set()
    n_done = 0
    t0 = time.time()

    def _wrap(job: dict) -> dict:
        try:
            res = fn(job)
            rec = {"id": job["id"], "ok": True, **res}
        except Exception as e:  # keep going, record failure (retried on resume)
            rec = {"id": job["id"], "ok": False, "error": repr(e),
                   "tb": traceback.format_exc()[-2000:]}
        with _lock:
            with out.open("a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=max_pool) as ex:
        it = iter(todo)
        exhausted = False
        while not exhausted or pending:
            budget = _budget(workers)
            while not exhausted and len(pending) < budget:
                fp = memguard.footprint()
                if fp["footprint_gb"] >= memguard.SOFT_GB or fp["sys_available_gb"] < memguard.SYS_FLOOR_GB:
                    if not pending:
                        time.sleep(5)  # wait for memory to come back
                        continue
                    break
                try:
                    job = next(it)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(ex.submit(_wrap, job))
            if not pending:
                continue
            fin, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            n_done += len(fin)
            if fin and not quiet and (n_done % max(1, total // 20) == 0 or n_done == total):
                rate = n_done / max(1e-9, time.time() - t0)
                print(f"[jobrunner] {out.name}: {n_done}/{total} ({rate:.2f}/s, workers={budget})", flush=True)
    return total
