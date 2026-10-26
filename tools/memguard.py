#!/usr/bin/env python3
"""Memory guard for the CSI-26 project.

The project must never hold more than LIMIT_GB (default 200 GB) of RAM.  The
guard measures the project's *footprint*:

    footprint = RSS of every process whose command line or cwd lies under the
                project root (including children of the supervised command)
              + memory of the Ollama models this project registered as in use
                (``_env/ollama_models.txt``; counted even though the Ollama
                server is shared, which is conservative)

It also watches the machine as a whole: if system *available* memory drops
below SYS_FLOOR_GB the run is treated as over-limit too, so that the project
never pushes the machine into swap even when other projects are busy.

Usage
-----
    memguard.py status                    # print footprint, exit 1 if over limit
    memguard.py run [--workers N] -- CMD  # supervise CMD (stop + slow restart)

Supervision ("stop and restart slowly")
---------------------------------------
* CMD is started in its own process group with ``CSI_WORKERS`` = current
  worker budget, and the budget is also written to ``logs/memguard.workers`` so
  a running job runner can re-read it (see ``tools/jobrunner.py``).
* Above SOFT (80 % of the limit) the budget is cut by one worker at a time.
* At or above LIMIT the whole process group is stopped (SIGTERM, then SIGKILL
  after 30 s), the Ollama models we registered are unloaded, the guard waits
  for memory to settle, halves the worker budget and restarts CMD.  The job
  runner is resumable (finished jobs are skipped), so nothing is lost.
* After a restart the budget is raised again by one worker every RAMP_S seconds
  while the footprint stays below 60 % of the limit (slow ramp-up).
* Every event is appended to ``logs/memguard.log`` and to ``plans.md``-friendly
  ``logs/memguard_events.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
WORKERS_FILE = LOG_DIR / "memguard.workers"
EVENTS = LOG_DIR / "memguard_events.jsonl"
TEXT_LOG = LOG_DIR / "memguard.log"
OLLAMA_REG = ROOT / "_env" / "ollama_models.txt"

GB = 1024 ** 3
LIMIT_GB = float(os.environ.get("CSI_MEM_LIMIT_GB", "200"))
SOFT_GB = 0.80 * LIMIT_GB
RAMP_OK_GB = 0.60 * LIMIT_GB
SYS_FLOOR_GB = float(os.environ.get("CSI_SYS_FLOOR_GB", "48"))
POLL_S = 3.0
RAMP_S = float(os.environ.get("CSI_RAMP_S", "120"))
COOLDOWN_S = 60.0


def _log(event: str, **kw) -> None:
    rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **kw}
    with EVENTS.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    with TEXT_LOG.open("a") as f:
        f.write(f"[{rec['t']}] {event} " + " ".join(f"{k}={v}" for k, v in kw.items()) + "\n")
    print(f"[memguard] {event} {kw}", file=sys.stderr, flush=True)


def _registered_ollama_models() -> set[str]:
    if not OLLAMA_REG.exists():
        return set()
    return {ln.strip() for ln in OLLAMA_REG.read_text().splitlines() if ln.strip() and not ln.startswith("#")}


def _ollama_endpoints() -> list[str]:
    ep = ROOT / "_env" / "ollama_endpoints.txt"
    hosts = []
    if ep.exists():
        hosts = [ln.strip() for ln in ep.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return hosts or ["http://localhost:11434"]


def ollama_footprint_gb() -> tuple[float, list[str]]:
    """Memory of the models THIS project loaded, summed across the project's Ollama endpoints (its dedicated
    parallel server). The shared server on :11434 is other projects' and is not counted."""
    reg = _registered_ollama_models()
    if not reg:
        return 0.0, []
    tot, names = 0.0, []
    for host in _ollama_endpoints():
        try:
            with urllib.request.urlopen(f"{host}/api/ps", timeout=3) as r:
                data = json.load(r)
        except Exception:
            continue
        for m in data.get("models", []):
            if m.get("name") in reg or m.get("model") in reg:
                tot += float(m.get("size", 0)) / GB
                names.append(m.get("name"))
    return tot, names


def project_processes(extra_roots: tuple[int, ...] = ()) -> list[psutil.Process]:
    me = os.getpid()
    root = str(ROOT)
    procs: dict[int, psutil.Process] = {}
    try:
        pids = psutil.pids()
    except Exception:
        pids = []
    for pid in pids:
        if pid == me:
            continue
        # Per-process access is wrapped so a process dying mid-scan (or an unreadable cmdline, which can raise
        # SystemError on macOS) never crashes the guard loop.
        try:
            p = psutil.Process(pid)
            cmd = " ".join(p.cmdline() or [])
            cwd = p.cwd() or ""
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, SystemError, OSError, Exception):
            continue
        if root in cmd or cwd.startswith(root):
            # Ollama server/runner memory is counted via /api/ps (model sizes); don't double-count its RSS here.
            low = cmd.lower()
            if "ollama" in low or "llama_server" in low or "llama-server" in low:
                continue
            procs[pid] = p
    for pid in extra_roots:
        try:
            parent = psutil.Process(pid)
            for c in [parent] + parent.children(recursive=True):
                procs[c.pid] = c
        except psutil.NoSuchProcess:
            pass
    procs.pop(me, None)
    return list(procs.values())


def project_rss_gb(extra_roots: tuple[int, ...] = ()) -> float:
    tot = 0
    for p in project_processes(extra_roots):
        try:
            tot += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return tot / GB


def footprint(extra_roots: tuple[int, ...] = ()) -> dict:
    rss = project_rss_gb(extra_roots)
    oll, names = ollama_footprint_gb()
    vm = psutil.virtual_memory()
    return {
        "project_rss_gb": round(rss, 2),
        "ollama_gb": round(oll, 2),
        "ollama_models": names,
        "footprint_gb": round(rss + oll, 2),
        "sys_available_gb": round(vm.available / GB, 2),
        "limit_gb": LIMIT_GB,
    }


def over_limit(fp: dict) -> bool:
    return fp["footprint_gb"] >= LIMIT_GB or fp["sys_available_gb"] < SYS_FLOOR_GB


def write_workers(n: int) -> None:
    WORKERS_FILE.write_text(str(n))


def unload_registered_ollama() -> None:
    for name in _registered_ollama_models():
        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=json.dumps({"model": name, "keep_alive": 0}).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=30).read()
        except Exception:
            pass


def stop_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(30):
        if proc.poll() is not None:
            return
        time.sleep(1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def supervise(cmd: list[str], workers: int, max_restarts: int) -> int:
    restarts = 0
    max_workers = workers
    while True:
        write_workers(workers)
        env = dict(os.environ, CSI_WORKERS=str(workers), CSI_MEMGUARD="1")
        _log("start", workers=workers, cmd=" ".join(cmd), restart=restarts)
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, start_new_session=True)
        last_ramp = time.time()
        killed = False
        while proc.poll() is None:
            time.sleep(POLL_S)
            fp = footprint((proc.pid,))
            if over_limit(fp):
                _log("LIMIT_HIT_stopping", **fp, workers=workers)
                stop_group(proc)
                unload_registered_ollama()
                killed = True
                break
            if fp["footprint_gb"] >= SOFT_GB and workers > 1:
                workers -= 1
                write_workers(workers)
                last_ramp = time.time()
                _log("soft_limit_reduce_workers", **fp, workers=workers)
            elif (fp["footprint_gb"] < RAMP_OK_GB and workers < max_workers
                  and time.time() - last_ramp > RAMP_S):
                workers += 1
                write_workers(workers)
                last_ramp = time.time()
                _log("ramp_up_workers", **fp, workers=workers)
        if not killed:
            rc = proc.returncode
            _log("exit", rc=rc)
            return rc
        restarts += 1
        if restarts > max_restarts:
            _log("giving_up", restarts=restarts)
            return 3
        workers = max(1, workers // 2)
        max_workers = max(1, max_workers - 1)
        _log("cooldown", seconds=COOLDOWN_S, next_workers=workers)
        time.sleep(COOLDOWN_S)
        while over_limit(footprint()):
            time.sleep(10)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("status")
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--max-restarts", type=int, default=20)
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.mode == "status":
        fp = footprint()
        print(json.dumps(fp, indent=1))
        return 1 if over_limit(fp) else 0
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("run needs a command after --")
    return supervise(cmd, a.workers, a.max_restarts)


if __name__ == "__main__":
    sys.exit(main())
