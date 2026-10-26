"""Cached Ollama chat client with tool calling (shared by all CSI-26 experiments).

* Every request is keyed by a hash of (model, messages, tools, options); responses are stored in a SQLite cache
  (``_env/llm_cache.sqlite``), so reruns and resumed runs are free and exactly reproducible.
* Each model used is registered in ``_env/ollama_models.txt`` so that ``tools/memguard.py`` counts its memory
  against the project's 200 GB cap.
* Deterministic decoding by default (temperature 0, fixed seed).

    from tools.llm import chat
    r = chat("qwen2.5:14b-instruct-q4_K_M", [{"role": "user", "content": "hi"}], tools=[...])
    r["message"]["content"], r["message"].get("tool_calls")
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "_env"
ENV.mkdir(exist_ok=True)
CACHE_DB = Path(os.environ.get("CSI_LLM_CACHE", ENV / "llm_cache.sqlite"))
REG = ENV / "ollama_models.txt"
def _default_host() -> str:
    # Prefer the project's dedicated (parallel) server if its endpoint file names one.
    ep = ENV / "ollama_endpoints.txt"
    if ep.exists():
        for ln in ep.read_text().splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                return ln
    return "http://localhost:11434"


HOST = os.environ.get("OLLAMA_HOST_URL", _default_host())

_tls = threading.local()
_reg_lock = threading.Lock()
_registered: set[str] = set()


def _db() -> sqlite3.Connection:
    con = getattr(_tls, "con", None)
    if con is None:
        con = sqlite3.connect(CACHE_DB, timeout=60, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT, model TEXT, t REAL)")
        _tls.con = con
    return con


def _register(model: str) -> None:
    with _reg_lock:
        if model in _registered:
            return
        existing = set(REG.read_text().split()) if REG.exists() else set()
        if model not in existing:
            with REG.open("a") as f:
                f.write(model + "\n")
        _registered.add(model)


def _key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def chat(model: str, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.0,
         seed: int = 0, num_ctx: int = 8192, num_predict: int = 1024, fmt: str | dict | None = None,
         think: bool | None = None, use_cache: bool = True, retries: int = 4, timeout: int = 600,
         keep_alive: str = "20m") -> dict:
    _register(model)
    options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx, "num_predict": num_predict}
    payload = {"model": model, "messages": messages, "options": options, "stream": False}
    if tools:
        payload["tools"] = tools
    if fmt is not None:
        payload["format"] = fmt
    if think is not None:
        payload["think"] = think
    k = _key(payload)
    if use_cache:
        row = _db().execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        if row:
            out = json.loads(row[0])
            out["_cached"] = True
            return out
    body = json.dumps({**payload, "keep_alive": keep_alive}).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{HOST}/api/chat", data=body, headers={"Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.load(r)
            out["_latency_s"] = time.time() - t0
            if use_cache:
                _db().execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?)", (k, json.dumps(out), model, time.time()))
                _db().commit()
            out["_cached"] = False
            return out
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"ollama chat failed after {retries} tries: {last!r}")


def tool_spec(name: str, description: str, params: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": params, "required": required or list(params)}}}
