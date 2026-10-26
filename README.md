# FloodBench-Grid and FloodGate: reproduction artifact

This folder reproduces the FloodBench-Grid and FloodGate study. It contains:

- the FloodBench-Grid testbed;
- the FloodGate admission layer;
- the LLM triage assistant (prompts and tool surface);
- every experiment definition;
- the 5,633 logged episodes;
- the property-test suite;
- the analysis pipeline that turns the logs into every reported number, table and chart.

It needs no dataset downloads and no paid API.

## Contents

| Path | Contents |
|---|---|
| `floodgate-alarm-triage/fb/madb.py` | Rationalized master alarm database (ISA-18.2 style) built from a pandapower grid |
| `floodgate-alarm-triage/fb/grid.py` | Physically consistent critical events: C1 unexpected trip, C2 bad data, C3 protection mis-operation |
| `floodgate-alarm-triage/fb/alarms.py` | Alarm streams, the four flood attacks and matched random control, severity spoofing, historian modes |
| `floodgate-alarm-triage/fb/floodgate.py` | FloodGate: MADB rendering, per-source token buckets, pinned P1 lane, grouping, completeness digest |
| `floodgate-alarm-triage/fb/tools.py` | MCP-style tools: `list_alarms`, `get_alarm`, `query_historian`, `brief_operator` |
| `floodgate-alarm-triage/fb/agent.py` | Ollama tool-calling triage assistant (system prompt and tool loop) |
| `floodgate-alarm-triage/fb/scorer.py`, `episode.py`, `guard.py` | Deterministic scoring, one-episode runner, Llama Guard check |
| `floodgate-alarm-triage/experiments/jobs.py` | Every experiment as a list of jobs (parameters and seeds) |
| `floodgate-alarm-triage/experiments/run.py` | Resumable experiment driver (one JSON line per finished episode) |
| `floodgate-alarm-triage/experiments/analyze.py` | Statistics, every reported number, the result tables and the chart data |
| `floodgate-alarm-triage/experiments/time_admission.py` | Times one FloodGate admission pass; writes `results/admission_timing.json` |
| `floodgate-alarm-triage/tests/test_invariants.py` | Property tests for the testbed and FloodGate's guarantees (no LLM needed) |
| `floodgate-alarm-triage/results/` | The logged episodes (`<experiment>.jsonl`), `summary.json` and `admission_timing.json` |
| `tools/` | Cached Ollama client (`llm.py`), resumable job runner (`jobrunner.py`), memory guard (`memguard.py`) |
| `requirements.txt` | Exact Python package versions of the environment that produced the results |

## 1. Set up

Python 3.12 is required (the results were produced with CPython 3.12.12).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 2. Run the property tests (no LLM, about 20 s)

```bash
cd floodgate-alarm-triage
../.venv/bin/python -m pytest tests -q
```

All 47 tests should pass.

## 3. Regenerate every reported number from the logs (no LLM, about 15 s)

```bash
cd floodgate-alarm-triage
../.venv/bin/python experiments/analyze.py --out outputs
```

This reads `results/*.jsonl` and writes:

- `results/summary.json`: every statistic (rates with Wilson intervals, bootstrap intervals, seed-matched McNemar
  tests).
- `outputs/`: every reported number, the result tables, and the data plotted in every chart
  (`outputs/figures/data/`). This folder is created by this step:
  - the result tables are the experiment ledger, the per-model FloodGate comparison, the dose-response and the
    spoofing table;
  - the same logs always give byte-identical files.

The admission-pass timing is read from `results/admission_timing.json`. Running `experiments/time_admission.py`
re-measures it on your machine and changes only that number.

## 4. Re-run the experiments (needs Ollama and GPU time)

Install [Ollama](https://ollama.com) and pull the models below. Then, from `floodgate-alarm-triage/`:

```bash
mkdir -p ../logged && mv results/*.jsonl ../logged/   # keep the logged episodes; the runner skips finished jobs
export CSI_SYS_FLOOR_GB=4      # see "Memory guard"; without this the runner waits on machines with < 48 GB free
export PYTHONHASHSEED=0        # repeatable alarm streams across processes (see "Determinism")
../.venv/bin/python experiments/run.py E5      # one experiment, or: all
../.venv/bin/python experiments/analyze.py --out outputs
```

Each finished episode is appended to `results/<experiment>.jsonl`. A stopped run continues where it left off. Model
responses are cached in `_env/llm_cache.sqlite` (created on the first call; path set by `CSI_LLM_CACHE`), keyed by
model, messages, tools and options, so repeated calls are not recomputed. The Ollama endpoint defaults to
`http://localhost:11434` (set `OLLAMA_HOST_URL` to change it). `CSI_WORKERS` (default 6) sets how many episodes run in
parallel.

| Model (Ollama tag) | Size, quantization | Used in |
|---|---|---|
| `qwen3-coder:30b` | 30.5B MoE, Q4_K_M | E1, E1nf, E1aff, E1ctx, E2, E2b, E3, E5, E6, E7, E8, E9, Espoof, EspoofNF |
| `mistral-small3.1:24b` | 24.0B, Q4_K_M | E1, E1nf, E1aff, E1ctx, E2, E3, E4, E5, E6, E7, E8, E9, Eguard, Espoof, EspoofNF, Emvob |
| `qwen2.5:14b-instruct-q4_K_M` | 14.8B, Q4_K_M | E1, E1nf, E1aff, E2, E3, E4, E5, E6, E7, E8, E9, Espoof, EspoofNF, Emvob |
| `qwen2.5:7b` | 7.6B, Q4_K_M | E1, E1nf, E1aff, E2, E3, E5, E7, E8, E9, Espoof, EspoofNF |
| `llama3.1:8b-instruct-q4_K_M` | 8.0B, Q4_K_M | E1, Eguard |
| `qwen3.8:latest` (parent `qwen3.8:27b-q4_K_M`) | 27.3B, Q4_K_M | E2b |
| `llama-guard3:8b` | 8.0B, Q4_K_M | Eguard2 |

E9 also runs `rule`, a deterministic non-LLM baseline that needs no model.

### Memory guard

Before starting each episode, the job runner checks memory. It waits while the machine has less available RAM than
`CSI_SYS_FLOOR_GB`. The default is 48 GB, set for the original workstation, so on smaller machines set a lower value as
shown above. For long unattended sweeps, `tools/memguard.py` can also supervise the run. It starts the command from
the artifact root, so run it from there:

```bash
.venv/bin/python tools/memguard.py run --workers 4 -- .venv/bin/python floodgate-alarm-triage/experiments/run.py all
```

It caps the run's RAM at `CSI_MEM_LIMIT_GB` (default 200). If the cap is reached it stops the run and restarts it with
fewer workers; finished episodes are never repeated.

## Experiments

| Experiment | Episodes | What it tests |
|---|---:|---|
| E1 | 1,380 | Flood dose-response with the severity filter: rate x position x scenario x model, plus no-flood controls |
| E1nf | 288 | The same dose-response without the severity filter |
| E1aff | 288 | Whether the severity filter and its source (MADB or device) change the outcome |
| E1ctx | 144 | Whether a 32k context rescues recall under flood |
| E2 | 432 | Decoy-narrative and grouping-capture floods vs a matched-volume random control |
| E2b | 57 | The decoy-vs-random contrast with the filter off, on the two larger models |
| E3 | 288 | Historian degradation: normal, empty, timeout, partial |
| E4 | 180 | Defense baselines vs no defense and FloodGate |
| E5 | 720 | FloodGate vs no defense across the main attack grid, seed-matched |
| E6 | 288 | Six fixed flood designs aimed at FloodGate (attack class, rate, position) |
| E7 | 288 | FloodGate with an isolate-then-reconcile protocol on the decoy-narrative case |
| E8 | 240 | Contingency windows without an attack flood, and no-event windows (false escalations) |
| E9 | 160 | Non-LLM rule baseline vs the LLMs, with and without FloodGate |
| Eguard | 48 | Content-guardrail check with an injected-text channel, with and without FloodGate |
| Eguard2 | 16 | Llama Guard 3 detection rate over the flood, with a text-injection positive control |
| Espoof | 512 | Severity spoofing: severity source x spoof x FloodGate |
| EspoofNF | 256 | Spoofing with the severity filter off: the effect of the displayed severity alone |
| Emvob | 48 | The spoofing contrast on the mv_oberrhein distribution grid |

Parameters and seeds of every job are in `experiments/jobs.py`. Scenarios: C1 unexpected trip, C2 bad data,
C3 protection mis-operation.

## Log format

`results/<experiment>.jsonl` holds one JSON object per finished episode. Its `id` encodes the job parameters; the
other fields are the outcome (for example `p1_surfaced`, `root_cause_correct`, `disposition`, the assistant's `brief`).
Eight lines record failed jobs (`"ok": false`): seven qwen3.8 calls in E2b that returned HTTP 500, and one timeout in
E6 that a later rerun completed. The analysis excludes them. Absolute file paths in their tracebacks were replaced by
`<repo>/`.

## Determinism

- **Alarm streams.** Episode generation is deterministic within a process. The flood generator seeds its random
  stream with Python's built-in `hash()` of the attack name (`fb/alarms.py`, line 112), and Python randomizes string
  hashes per process unless `PYTHONHASHSEED` is set. The logged runs did not set it, so a re-run regenerates
  statistically equivalent alarm streams, not identical ones. Paired FloodGate/no-FloodGate comparisons were generated
  within one process, so both arms of a pair share the same stream in nearly all cases:
  - all pairs in E5, Espoof, EspoofNF and Emvob;
  - 143 of 144 pairs in E6 (the exception is the timed-out job, re-run in a new process; both of its arms surfaced
    the critical alarm);
  - 75 of 80 pairs in E9.

  Set `PYTHONHASHSEED` to any fixed value to make new runs repeatable.
- **Model outputs.** Decoding uses temperature 0 and a fixed seed, but local LLM inference is not bitwise
  deterministic under batching. Results are therefore reported as statistics across seeds and models.
- **Timing.** `latency_s` and the fields derived from it (`within_art`, `p1_recall_at_art`) are wall-clock
  measurements from the original machine, whose GPU was shared with other jobs.
