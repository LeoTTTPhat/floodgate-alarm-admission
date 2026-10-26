"""Analysis pipeline for FloodBench-Grid.

Reads results/*.jsonl, computes the statistics for each experiment, and writes:
  * results/summary.json        machine-readable aggregates
  * OUT/macros.tex              every reported number, as \\newcommand macros
  * OUT/tables/*.tex            the generated tables (experiment ledger, FloodGate comparison, dose-response, spoofing)
  * OUT/figures/data/*.dat      the data behind every chart
OUT is paper/ by default; `python experiments/analyze.py --out DIR` writes these files to DIR instead.
Stats: Wilson 95% CIs for proportions, Newcombe 95% CIs for differences of proportions, a bootstrap CI for the
latency median, and seed-matched exact McNemar tests for FloodGate-vs-none. Percentages are printed as whole numbers
(rounded half up) in both text and tables, with k/n wherever the paper reports a proportion.
Surfacing-type outcomes are analysed on the two scenarios that contain a priority-1 critical alarm (C1, C3); C2's
residual alarm is rationalized below P1 and enters only the data-integrity analysis.
Everything degrades gracefully if an experiment's results are missing or partial (so it can run mid-sweep).
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
PAPER = Path(sys.argv[sys.argv.index("--out") + 1]).resolve() if "--out" in sys.argv else ROOT / "paper"
PAPER.mkdir(parents=True, exist_ok=True)

NA = (float("nan"), float("nan"), float("nan"), 0)   # rate() tuple for a missing cell


def seed_of(r: dict) -> str:
    """Seed for pairing: prefer an explicit field, else parse the 's=<seed>' token from the job id."""
    if "seed" in r:
        return str(r["seed"])
    for tok in str(r.get("id", "")).split("|"):
        if tok.startswith("s="):
            return tok[2:]
    return "?"


def load(exp: str) -> list[dict]:
    p = RESULTS / f"{exp}.jsonl"
    if not p.exists():
        return []
    rows = []
    for ln in p.read_text().splitlines():
        try:
            r = json.loads(ln)
            if r.get("ok", True):
                rows.append(r)
        except Exception:
            continue
    return rows


def n_failed(exp: str) -> int:
    """Jobs whose only logged lines are failures (a failure later rerun successfully is not counted)."""
    p = RESULTS / f"{exp}.jsonl"
    if not p.exists():
        return 0
    ok, bad = set(), set()
    for ln in p.read_text().splitlines():
        try:
            r = json.loads(ln)
        except Exception:
            continue
        (ok if r.get("ok", True) else bad).add(r["id"])
    return len(bad - ok)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def ratek(k: int, n: int) -> tuple[float, float, float, int]:
    return (*wilson(k, n), n)


def rate(rows, field) -> tuple[float, float, float, int]:
    vals = [bool(r.get(field)) for r in rows if field in r]
    return ratek(sum(vals), len(vals))


def half_up(x: float) -> int:
    """Round half up (62.5 -> 63); the convention for every printed percentage."""
    return int(Decimal(repr(float(x))).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def kn(t) -> str:
    """'k/n' from a rate() tuple (p, lo, hi, n)."""
    if not t or not isinstance(t[3], int) or t[3] == 0:
        return "--"
    return f"{round(t[0] * t[3])}/{t[3]}"


def newcombe(t1, t2) -> tuple[float, float, float]:
    """t1 - t2 for two independent proportions with Newcombe's hybrid-score 95% CI, in percentage points."""
    p1, l1, u1, _ = t1
    p2, l2, u2, _ = t2
    d = p1 - p2
    return (100 * d, 100 * (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)),
            100 * (d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)))


def boot_median(x, nboot=2000, seed=0):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(x, len(x), replace=True)) for _ in range(nboot)]
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def mcnemar(pairs):
    """pairs: list of (none_success, fg_success) booleans. Returns (b, c, p_value) exact binomial."""
    b = sum(1 for a, d in pairs if a and not d)   # none ok, fg fail
    c = sum(1 for a, d in pairs if not a and d)   # none fail, fg ok
    n = b + c
    if n == 0:
        return b, c, 1.0
    from math import comb
    k = min(b, c)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return b, c, p


def paired(rows, keyfn, armfn, field="p1_surfaced"):
    """Seed-matched pairs (arm False, arm True) of a boolean outcome."""
    idx = defaultdict(dict)
    for r in rows:
        idx[keyfn(r)][armfn(r)] = bool(r.get(field))
    return [(v[False], v[True]) for v in idx.values() if False in v and True in v]


def logistic_dose(rates, recall):
    """Fit recall ~ logistic(a + b*log10(1+rate)); return the flood tolerance = rate where recall crosses 0.9."""
    try:
        import statsmodels.api as sm
        x = np.log10(1 + np.asarray(rates, float))
        y = np.asarray(recall, float)
        X = sm.add_constant(x)
        m = sm.Logit(y, X).fit(disp=0)
        a, b = m.params
        if b >= -1e-6:
            return None
        lx = (math.log(0.9 / 0.1) - a) / b
        return float(10 ** lx - 1)
    except Exception:
        return None


# ---------------- per-experiment analyses ----------------
def grp_rate(rows, keyfn, field):
    g = defaultdict(list)
    for r in rows:
        g[keyfn(r)].append(r)
    return {k: rate(v, field) for k, v in g.items()}


ALL_MACROS = [
    "EAffFilterDevice", "EAffFilterMadb", "EAffNoFilter", "ECtxEightK", "ECtxThirtyTwoK",
    "EEightUtilFG", "EEightUtilNone", "EFiveMcNemarFGonly", "EFiveMcNemarNoneOnly", "EFiveMcNemarPairs",
    "EFiveMcNemarP", "EFiveSurfFG", "EFiveSurfNone", "EOneSurfFloodMax", "EOneSurfNoFlood",
    "EOneLowMin", "EOneLowMax", "EOneHighMin", "EOnePosBefore", "EOnePosInter", "EOnePosAfter",
    "ESevenFGdirectRC", "ESevenFGisoRC", "ESixFGSurf", "ESixNoneSurf", "ESixMcNemarP", "ESixMcNemarFGonly",
    "ESixMcNemarNoneOnly", "ESixMcNemarPairs", "NSixDesigns",
    "EThreeWDEmpty", "EThreeWDNormal", "EThreeWDTimeout", "EThreeWDPartial", "EThreeDiff", "EThreeDiffLo",
    "EThreeDiffHi", "ETwoDecoyRC", "ETwoRandomRC", "NEpisodes", "NModelsCore", "NFailed",
    "ENfSurfNoFlood", "ENfSurfFloodMax",
    "ESpoofDevSpoofNone", "ESpoofDevSpoofFG", "ESpoofDevCleanNone", "ESpoofMadbSpoofNone",
    "ESpoofMcNemarP", "ESpoofMcNemarFGonly", "ESpoofMcNemarNoneOnly", "ESpoofMcNemarPairs",
    "ETwoDecoySurf", "ETwoRandomSurf", "ETwoGroupSurf", "ETwoDiffDecoy", "ETwoDiffDecoyLo", "ETwoDiffDecoyHi",
    "ETwoDiffGroup", "ETwoDiffGroupLo", "ETwoDiffGroupHi",
    "EEightFalsePNone", "EEightFalsePFG", "EEightNoEventNone", "EEightNoEventFG", "EEightNoBriefNone", "EEightNoBriefFG", "EFourNone", "EFourFilter",
    "EFourFG", "CTwoDataIntegrity",
    "ENineRuleSurfFG", "ENineRuleSurfNone", "ENineLLMSurfFG", "ENineLLMSurfNone", "ENinePresentFG",
    "ENineRulePresentFG", "ENineLLMdiagRC", "ENineRuleDiagRC", "ENineLowNone", "ENineLowFG",
    "ESpoofNFcleanNone", "ESpoofNFcleanFG", "ESpoofNFspoofNone", "ESpoofNFspoofFG", "ESpoofNFViewMin", "ESpoofNFViewMax",
    "ENFMcNemarP", "ENFMcNemarFGonly", "ENFMcNemarNoneOnly", "ENFMcNemarPairs",
    "ETwoBdecoySurf", "ETwoBrandomSurf", "ETwoBdecoyRC", "ETwoBrandomRC", "ETwoBDiff", "ETwoBDiffLo", "ETwoBDiffHi",
    "ETwoBFailed",
    "EmvobSpoofNone", "EmvobSpoofFG", "EguardFlagRate", "EguardFlood", "EguardInj",
    "EFiveLatMed", "EFiveLatMedLo", "EFiveLatMedHi", "EFiveLatPNF", "EFiveLatN", "EFiveART", "EFiveWithinART",
    "EFiveLowNone", "EFiveLowFG", "EFiveRestNone", "EFiveRestFG", "EFiveEscNone", "EFiveEscFG",
    "EOneWeakNamed", "EOneWeakSurf",
    "NListed", "NListedDismissed", "NListedEscalated", "NListedMonitor", "NListedOther",
    "EscMistral", "EscQwenFourteen", "EscQwenSeven", "EscCoder",
    "CtxDefault", "CtxLarge", "SpoofRate", "FloodMultiple", "FloodMax",
    "NMadbPoints", "MadbLowPct", "NMadbFloodable", "NMadbPOne", "NMadbPTwo", "NMadbPThree", "NMadbPFour",
    "NMvobBus", "NMvobSgen", "NPropTests",
    "LaneOne", "LaneTwo", "LaneLow", "BucketLow", "PageSize", "MaxSteps", "BgMin", "BgMax", "FloodThreshold",
    "PairsSameFive", "PairsTotalFive", "PairsSameSpoof", "PairsTotalSpoof", "PairsSameNF", "PairsTotalNF",
    "PairsSameSix", "PairsTotalSix", "ESixSameMcNemarP", "ESixSameMcNemarFGonly", "ESixSameMcNemarNoneOnly",
    "NEpisodesCore", "AdmitMs", "AdmitAlarms", "NModelsCoreWord", "NExperimentsWord", "NModelsHighWord",
]

REGISTRY_NAMES = ["E1", "E1nf", "E1aff", "E1ctx", "E2", "E2b", "E3", "E4", "E5", "E6", "E7", "E8",
                  "Eguard", "Eguard2", "Espoof", "EspoofNF", "E9", "Emvob"]
# Experiments whose consumer is an LLM (Eguard2 runs the guard classifier only).
LLM_EXPS = [e for e in REGISTRY_NAMES if e != "Eguard2"]
CORE_ORDER = ["mistral-small3.1:24b", "qwen2.5:14b-instruct-q4_K_M", "qwen2.5:7b", "qwen3-coder:30b"]
WEAK = "llama3.1:8b-instruct-q4_K_M"
P1_SCEN = ("C1", "C3")  # the scenarios with a priority-1 critical alarm; C2 is analysed for data integrity only


def analyze():
    S = {}
    macros = {k: "--" for k in ALL_MACROS}  # every paper-referenced macro is always defined

    def M(name, val, fmt="int"):
        """Store a number: whole numbers rounded half up by default, or an explicit format string."""
        if val is None or (isinstance(val, float) and math.isnan(val)):
            macros[name] = "--"
        elif fmt == "int":
            macros[name] = str(half_up(val))
        else:
            macros[name] = fmt.format(val)

    def R(name, t):
        """A proportion as <name> (whole percent) and <name>KN (k/n)."""
        M(name, 100 * t[0] if t and isinstance(t[3], int) and t[3] else float("nan"))
        macros[name + "KN"] = kn(t)

    def MP(name, p):
        macros[name] = _p_tex(p)

    # ---- E1 dose-response (MADB-sourced filter) ----
    e1 = load("E1")
    if e1:
        crit = [r for r in e1 if r.get("has_critical") and r["scenario"] in P1_SCEN]
        core = [r for r in crit if r["model"] != WEAK]
        byrate = grp_rate(core, lambda r: r["flood_rate"], "p1_surfaced")
        S["E1_surfacing_by_rate"] = {str(k): byrate[k] for k in sorted(byrate)}
        R("EOneSurfNoFlood", byrate.get(0, NA))
        R("EOneSurfFloodMax", byrate.get(max(byrate), NA))
        M("FloodMax", max(byrate), "{:d}")
        S["E1_by_model"] = {}
        for mdl in sorted(set(r["model"] for r in crit)):
            mr = [r for r in crit if r["model"] == mdl]
            br = grp_rate(mr, lambda r: r["flood_rate"], "p1_surfaced")
            tol = logistic_dose([k for k in sorted(br)], [br[k][0] for k in sorted(br)])
            S["E1_by_model"][mdl] = {"by_rate": {str(k): br[k] for k in sorted(br)}, "flood_tolerance": tol}
        # spread across rates: the lowest core model versus the other three
        per = {m: [v[0] for v in S["E1_by_model"][m]["by_rate"].values()] for m in CORE_ORDER if m in S["E1_by_model"]}
        if per:
            low = min(per, key=lambda m: np.mean(per[m]))
            S["E1_lowest_model"] = low
            M("EOneLowMin", 100 * min(per[low]))
            M("EOneLowMax", 100 * max(per[low]))
            M("EOneHighMin", 100 * min(min(v) for m, v in per.items() if m != low))
        # critical-alarm position (before / interleaved / after the flood), pooled over flood rates > 0
        pos = grp_rate([r for r in core if r["flood_rate"] > 0], lambda r: r["position"], "p1_surfaced")
        S["E1_position"] = {k: pos[k] for k in pos}
        R("EOnePosBefore", pos.get("before", NA))
        R("EOnePosInter", pos.get("interleaved", NA))
        R("EOnePosAfter", pos.get("after", NA))
        weak = [r for r in crit if r["model"] == WEAK]
        if weak:
            S["E1_weak_named_vs_surf"] = {"named": rate(weak, "p1_named"), "surfaced": rate(weak, "p1_surfaced")}
            R("EOneWeakNamed", S["E1_weak_named_vs_surf"]["named"])
            R("EOneWeakSurf", S["E1_weak_named_vs_surf"]["surfaced"])
        c2 = [r for r in e1 if r["scenario"] == "C2" and r["model"] != WEAK]
        if c2:
            S["E1_C2_dataintegrity_by_rate"] = {
                str(k): v for k, v in grp_rate(c2, lambda r: r["flood_rate"], "root_cause_correct_coarse").items()}
        M("CtxDefault", Counter(r.get("num_ctx") for r in e1).most_common(1)[0][0], "{:d}")

    # ---- E1nf no-filter dose-response ----
    e1nf = load("E1nf")
    if e1nf:
        crit = [r for r in e1nf if r.get("has_critical") and r["scenario"] in P1_SCEN]
        byrate = grp_rate(crit, lambda r: r["flood_rate"], "p1_surfaced")
        S["E1nf_surfacing_by_rate"] = {str(k): byrate[k] for k in sorted(byrate)}
        R("ENfSurfNoFlood", byrate.get(0, NA))
        R("ENfSurfFloodMax", byrate.get(max(byrate), NA))
        S["E1nf_by_model"] = {
            m: {str(k): v for k, v in sorted(grp_rate([r for r in crit if r["model"] == m],
                                                      lambda r: r["flood_rate"], "p1_surfaced").items())}
            for m in sorted({r["model"] for r in crit})}

    # ---- Espoof severity spoofing (filter on) ----
    esp = load("Espoof")
    if esp:
        def spk(r):
            return (r["severity_source"], "spoof" if r.get("spoof") else "clean",
                    "fg" if r["floodgate"] else "none")
        g = grp_rate(esp, spk, "p1_surfaced")
        S["Espoof_surf"] = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in g.items()}
        R("ESpoofDevSpoofNone", g.get(("device", "spoof", "none"), NA))
        R("ESpoofDevSpoofFG", g.get(("device", "spoof", "fg"), NA))
        R("ESpoofDevCleanNone", g.get(("device", "clean", "none"), NA))
        R("ESpoofMadbSpoofNone", g.get(("madb", "spoof", "none"), NA))
        pairs = paired([r for r in esp if r["severity_source"] == "device" and r.get("spoof")],
                       lambda r: (r["model"], r["scenario"], seed_of(r)), lambda r: r["floodgate"])
        b, c, p = mcnemar(pairs)
        S["Espoof_mcnemar"] = {"b_none_only": b, "c_fg_only": c, "p": p, "n_pairs": len(pairs)}
        MP("ESpoofMcNemarP", p)
        M("ESpoofMcNemarFGonly", c, "{:d}")
        M("ESpoofMcNemarNoneOnly", b, "{:d}")
        M("ESpoofMcNemarPairs", len(pairs), "{:d}")

    # ---- E1aff affordance: severity filter / source ----
    e1a = load("E1aff")
    if e1a:
        def affkey(r):
            return ("filter" if r["severity_filter"] else "nofilter", r["severity_source"])
        aff = grp_rate([r for r in e1a if r.get("has_critical") and r["scenario"] in P1_SCEN], affkey, "p1_surfaced")
        S["E1aff"] = {f"{k[0]}_{k[1]}": v for k, v in aff.items()}
        R("EAffNoFilter", aff.get(("nofilter", "madb"), NA))
        R("EAffFilterMadb", aff.get(("filter", "madb"), NA))
        R("EAffFilterDevice", aff.get(("filter", "device"), NA))

    # ---- E1ctx 32k ----
    e1c = load("E1ctx")
    if e1c:
        ctx = grp_rate([r for r in e1c if r.get("has_critical") and r["scenario"] in P1_SCEN],
                       lambda r: r["num_ctx"], "p1_surfaced")
        S["E1ctx"] = {str(k): ctx[k] for k in sorted(ctx)}
        R("ECtxEightK", ctx.get(min(ctx), NA))
        R("ECtxThirtyTwoK", ctx.get(max(ctx), NA))
        M("CtxLarge", max(ctx), "{:d}")

    # ---- E2 structure vs matched random control ----
    e2 = [r for r in load("E2") if r["scenario"] in P1_SCEN]
    if e2:
        by = grp_rate(e2, lambda r: r["attack"], "root_cause_correct_coarse")
        surf = grp_rate(e2, lambda r: r["attack"], "p1_surfaced")
        S["E2_rc_by_attack"] = {k: by[k] for k in by}
        S["E2_surf_by_attack"] = {k: surf[k] for k in surf}
        R("ETwoDecoyRC", by.get("decoy_narrative", NA))
        R("ETwoRandomRC", by.get("random_control", NA))
        R("ETwoDecoySurf", surf.get("decoy_narrative", NA))
        R("ETwoRandomSurf", surf.get("random_control", NA))
        R("ETwoGroupSurf", surf.get("grouping_capture", NA))
        for nm, atk in [("Decoy", "decoy_narrative"), ("Group", "grouping_capture")]:
            if atk in surf and "random_control" in surf:
                d, lo, hi = newcombe(surf[atk], surf["random_control"])
                M(f"ETwoDiff{nm}", d)
                M(f"ETwoDiff{nm}Lo", lo)
                M(f"ETwoDiff{nm}Hi", hi)

    # ---- E3 evidence starvation: missed critical alarms (brief omits it or dismisses it) ----
    e3 = [r for r in load("E3") if r["scenario"] in P1_SCEN]
    if e3:
        wd = grp_rate(e3, lambda r: r["historian"], "wrong_dismiss")
        S["E3_wrong_dismiss_by_historian"] = {k: wd[k] for k in wd}
        for nm, h in [("Normal", "normal"), ("Empty", "empty"), ("Timeout", "timeout"), ("Partial", "partial")]:
            R(f"EThreeWD{nm}", wd.get(h, NA))
        if "empty" in wd and "normal" in wd:
            d, lo, hi = newcombe(wd["empty"], wd["normal"])
            M("EThreeDiff", d)
            M("EThreeDiffLo", lo)
            M("EThreeDiffHi", hi)

    # ---- E5 FloodGate with the MADB-sourced filter (McNemar, seed-matched) ----
    e5 = [r for r in load("E5") if r["scenario"] in P1_SCEN]
    if e5:
        surf = grp_rate(e5, lambda r: r["floodgate"], "p1_surfaced")
        S["E5_surf_by_fg"] = {("fg" if k else "none"): v for k, v in surf.items()}
        R("EFiveSurfNone", surf.get(False, NA))
        R("EFiveSurfFG", surf.get(True, NA))
        pairs = paired(e5, lambda r: (r["model"], r["scenario"], r["attack"], r["flood_rate"], seed_of(r)),
                       lambda r: r["floodgate"])
        b, c, p = mcnemar(pairs)
        S["E5_mcnemar"] = {"b_none_only": b, "c_fg_only": c, "p": p, "n_pairs": len(pairs)}
        MP("EFiveMcNemarP", p)
        M("EFiveMcNemarFGonly", c, "{:d}")
        M("EFiveMcNemarNoneOnly", b, "{:d}")
        M("EFiveMcNemarPairs", len(pairs), "{:d}")
        S["E5_by_model"] = {m: {("fg" if k else "none"): v for k, v in
                                grp_rate([r for r in e5 if r["model"] == m], lambda r: r["floodgate"],
                                         "p1_surfaced").items()} for m in sorted({r["model"] for r in e5})}
        low = min(S["E5_by_model"], key=lambda m: S["E5_by_model"][m]["none"][0])
        S["E5_lowest_model"] = low
        R("EFiveLowNone", S["E5_by_model"][low]["none"])
        R("EFiveLowFG", S["E5_by_model"][low]["fg"])
        rest = grp_rate([r for r in e5 if r["model"] != low], lambda r: r["floodgate"], "p1_surfaced")
        R("EFiveRestNone", rest.get(False, NA))
        R("EFiveRestFG", rest.get(True, NA))
        for arm, nm in [(False, "None"), (True, "FG")]:   # listed AND escalated
            rows = [r for r in e5 if r["floodgate"] == arm]
            k = sum(1 for r in rows if r.get("p1_surfaced") and r.get("disposition") == "escalate")
            R(f"EFiveEsc{nm}", ratek(k, len(rows)))

    # ---- E7 isolate-then-reconcile on the decoy flood ----
    e7 = load("E7")
    if e7:
        def pk(r):
            return ("fg" if r["floodgate"] else "nofg") + "_" + r.get("protocol", "direct")
        rc = grp_rate([r for r in e7 if r["attack"] == "decoy_narrative"], pk, "root_cause_correct_coarse")
        S["E7_decoy_rc"] = {k: rc[k] for k in rc}
        R("ESevenFGdirectRC", rc.get("fg_direct", NA))
        R("ESevenFGisoRC", rc.get("fg_isolate_reconcile", NA))

    # ---- E8 non-attack contingency floods and no-event windows ----
    e8 = load("E8")
    if e8:
        cont = [r for r in e8 if r["scenario"] == "C1"]
        rc = grp_rate(cont, lambda r: r["floodgate"], "root_cause_correct_coarse")
        S["E8_contingency_rc_by_fg"] = {("fg" if k else "none"): v for k, v in rc.items()}
        R("EEightUtilNone", rc.get(False, NA))
        R("EEightUtilFG", rc.get(True, NA))
        noev = [r for r in e8 if r["scenario"] == "BENIGN"]
        if noev:
            for arm, nm in [(False, "None"), (True, "FG")]:
                rows = [r for r in noev if r["floodgate"] == arm]
                # a missing brief cannot escalate, so it counts as no false escalation (the scorer leaves it unset)
                fp = ratek(sum(bool(r.get("false_p1")) for r in rows), len(rows))
                nb = ratek(sum(bool(r.get("no_brief")) for r in rows), len(rows))
                ok = rate(rows, "root_cause_correct_coarse")       # 'nuisance only' verdict; no brief = not given
                S.setdefault("E8_noevent", {})[nm] = {"false_p1": fp, "no_brief": nb, "nuisance_verdict": ok,
                                                      "no_brief_models": dict(Counter(
                                                          r["model"] for r in rows if r.get("no_brief")))}
                R(f"EEightFalseP{nm}", fp)
                R(f"EEightNoBrief{nm}", nb)
                R(f"EEightNoEvent{nm}", ok)

    # ---- E4 defense baselines (parse the 'defense' label from the id: ...|d=<name>|...) ----
    def defense_of(r):
        if r.get("defense"):
            return r["defense"]
        for tok in str(r.get("id", "")).split("|"):
            if tok.startswith("d="):
                return tok[2:]
        return "?"
    e4 = load("E4")
    if e4:
        bl = grp_rate(e4, defense_of, "p1_surfaced")
        S["E4_surf_by_defense"] = {k: bl[k] for k in bl}
        S["E4_models"] = sorted({r["model"] for r in e4})
        R("EFourNone", bl.get("none", NA))
        R("EFourFilter", bl.get("severity_filter", NA))
        R("EFourFG", bl.get("floodgate", NA))

    # ---- E6 fixed sweep of flood designs aimed at FloodGate ----
    e6 = load("E6")
    if e6:
        fg = grp_rate(e6, lambda r: r["floodgate"], "p1_surfaced")
        S["E6_surf_by_fg"] = {("fg" if k else "none"): v for k, v in fg.items()}
        R("ESixFGSurf", fg.get(True, NA))
        R("ESixNoneSurf", fg.get(False, NA))
        pairs = paired(e6, lambda r: (r["model"], r["scenario"], r["attack"], r["flood_rate"], r.get("position"),
                                      seed_of(r)), lambda r: r["floodgate"])
        b, c, p = mcnemar(pairs)
        S["E6_mcnemar"] = {"b_none_only": b, "c_fg_only": c, "p": p, "n_pairs": len(pairs)}
        MP("ESixMcNemarP", p)
        M("ESixMcNemarFGonly", c, "{:d}")
        M("ESixMcNemarNoneOnly", b, "{:d}")
        M("ESixMcNemarPairs", len(pairs), "{:d}")
        M("NSixDesigns", len({(r["attack"], r["flood_rate"], r.get("position")) for r in e6}), "{:d}")
        S["E6_by_model"] = {m: {("fg" if k else "none"): v for k, v in
                                grp_rate([r for r in e6 if r["model"] == m], lambda r: r["floodgate"],
                                         "p1_surfaced").items()} for m in sorted({r["model"] for r in e6})}

    # ---- C2 data-integrity recall (from E1), reported as a fragility finding ----
    e1c2 = [r for r in load("E1") if r["scenario"] == "C2" and r["model"] != WEAK]
    if e1c2:
        di = rate(e1c2, "root_cause_correct_coarse")
        S["C2_dataintegrity"] = di
        R("CTwoDataIntegrity", di)

    # ---- E9: non-LLM rule baseline vs LLM; presentation vs surfacing ----
    e9 = load("E9")
    if e9:
        def a9(r):
            return ("rule" if r["model"] == "rule" else "llm", "fg" if r["floodgate"] else "none")
        surf = grp_rate(e9, a9, "p1_surfaced")
        pres = grp_rate(e9, a9, "p1_present_in_view")
        S["E9_surf"] = {f"{k[0]}_{k[1]}": v for k, v in surf.items()}
        S["E9_present"] = {f"{k[0]}_{k[1]}": v for k, v in pres.items()}
        rc = grp_rate([r for r in e9 if r["model"] != "rule"], lambda r: "llm", "root_cause_correct_coarse")
        rcr = grp_rate([r for r in e9 if r["model"] == "rule"], lambda r: "rule", "root_cause_correct_coarse")
        R("ENineRuleSurfFG", surf.get(("rule", "fg"), NA))
        R("ENineRuleSurfNone", surf.get(("rule", "none"), NA))
        R("ENineLLMSurfFG", surf.get(("llm", "fg"), NA))
        R("ENineLLMSurfNone", surf.get(("llm", "none"), NA))
        R("ENinePresentFG", pres.get(("llm", "fg"), NA))
        R("ENineRulePresentFG", pres.get(("rule", "fg"), NA))
        R("ENineLLMdiagRC", rc.get("llm", NA))
        R("ENineRuleDiagRC", rcr.get("rule", NA))
        low = S.get("E5_lowest_model")
        if low:
            lw = grp_rate([r for r in e9 if r["model"] == low], lambda r: r["floodgate"], "p1_surfaced")
            R("ENineLowNone", lw.get(False, NA))
            R("ENineLowFG", lw.get(True, NA))
            misses = {arm: (surf[("llm", arm)][3] - round(surf[("llm", arm)][0] * surf[("llm", arm)][3]))
                      for arm in ("none", "fg") if ("llm", arm) in surf}
            low_misses = {("fg" if k else "none"): v[3] - round(v[0] * v[3]) for k, v in lw.items()}
            S["E9_all_llm_misses_from_lowest_model"] = all(misses[a] == low_misses.get(a) for a in misses)

    # ---- EspoofNF: spoofing with NO filter (the full stream is listed page by page, with the shown priority) ----
    enf = load("EspoofNF")
    if enf:
        def k(r):
            return ("spoof" if r.get("spoof") else "clean", "fg" if r["floodgate"] else "none")
        g = grp_rate(enf, k, "p1_surfaced")
        S["EspoofNF_surf"] = {f"{a}_{b}": v for (a, b), v in g.items()}
        R("ESpoofNFcleanNone", g.get(("clean", "none"), NA))
        R("ESpoofNFcleanFG", g.get(("clean", "fg"), NA))
        R("ESpoofNFspoofNone", g.get(("spoof", "none"), NA))
        R("ESpoofNFspoofFG", g.get(("spoof", "fg"), NA))
        clean = [r for r in enf if not r.get("spoof")]
        pairs = paired(clean, lambda r: (r["model"], r["scenario"], seed_of(r)), lambda r: r["floodgate"])
        b, c, p = mcnemar(pairs)
        S["EspoofNF_clean_mcnemar"] = {"b_none_only": b, "c_fg_only": c, "p": p, "n_pairs": len(pairs)}
        MP("ENFMcNemarP", p)
        M("ENFMcNemarFGonly", c, "{:d}")
        M("ENFMcNemarNoneOnly", b, "{:d}")
        M("ENFMcNemarPairs", len(pairs), "{:d}")
        S["EspoofNF_clean_by_model"] = {m: {("fg" if kk else "none"): v for kk, v in
                                            grp_rate([r for r in clean if r["model"] == m], lambda r: r["floodgate"],
                                                     "p1_surfaced").items()} for m in sorted({r["model"] for r in clean})}
        M("SpoofRate", Counter(r["flood_rate"] for r in enf).most_common(1)[0][0], "{:d}")
        nv = [r["n_view"] for r in enf if not r["floodgate"]]  # alarms listed without FloodGate (several pages)
        M("ESpoofNFViewMin", min(nv), "{:d}")
        M("ESpoofNFViewMax", max(nv), "{:d}")

    # ---- E2b: decoy vs random without a filter, on the two larger models ----
    e2b = [r for r in load("E2b") if r["scenario"] in P1_SCEN]
    if e2b:
        g = grp_rate(e2b, lambda r: r["attack"], "p1_surfaced")
        rcg = grp_rate(e2b, lambda r: r["attack"], "root_cause_correct_coarse")
        S["E2b_surf"] = {kk: g[kk] for kk in g}
        R("ETwoBdecoySurf", g.get("decoy_narrative", NA))
        R("ETwoBrandomSurf", g.get("random_control", NA))
        R("ETwoBdecoyRC", rcg.get("decoy_narrative", NA))
        R("ETwoBrandomRC", rcg.get("random_control", NA))
        if "decoy_narrative" in g and "random_control" in g:
            d, lo, hi = newcombe(g["decoy_narrative"], g["random_control"])
            M("ETwoBDiff", d)
            M("ETwoBDiffLo", lo)
            M("ETwoBDiffHi", hi)
        M("ETwoBFailed", n_failed("E2b"), "{:d}")

    # ---- Emvob: spoofing on mv_oberrhein ----
    emv = load("Emvob")
    if emv:
        sp = grp_rate([r for r in emv if r.get("spoof") is not None],
                      lambda r: ("spoof" if r.get("spoof") else "clean", "fg" if r["floodgate"] else "none"),
                      "p1_surfaced")
        S["Emvob_surf"] = {f"{a}_{b}": v for (a, b), v in sp.items()}
        R("EmvobSpoofNone", sp.get(("spoof", "none"), NA))
        R("EmvobSpoofFG", sp.get(("spoof", "fg"), NA))

    # ---- Eguard2: Llama Guard 3 on flood windows and on windows with injected instruction text ----
    eg2 = load("Eguard2")
    if eg2:
        flagged = rate(eg2, "guard_flagged")
        S["Eguard2_flag_rate"] = flagged
        R("EguardFlagRate", flagged)
        R("EguardFlood", rate([r for r in eg2 if "inj=True" not in r["id"]], "guard_flagged"))
        R("EguardInj", rate([r for r in eg2 if "inj=True" in r["id"]], "guard_flagged"))

    # ---- latency (descriptive; no timing bound is claimed) ----
    # FloodGate episodes on the P1 scenarios only. The no-FloodGate E5 episodes repeat E1 conditions and were served
    # from the LLM response cache (near-zero wall-clock), so pooling them would understate latency; cache hits
    # (< 1 s, far below any multi-step local-LLM episode) are excluded defensively. Episodes ran concurrently on one
    # shared inference server, so these wall-clock figures include queueing.
    lat_rows = [r for r in load("E5") if r["floodgate"] and r["scenario"] in P1_SCEN
                and isinstance(r.get("latency_s"), (int, float)) and r["latency_s"] > 1.0]
    if lat_rows:
        lat = np.array([r["latency_s"] for r in lat_rows], float)
        arts = sorted({r.get("art_s") for r in lat_rows})
        within = rate(lat_rows, "within_art")
        med, lo, hi = boot_median(lat)
        S["E5_latency"] = {"median": med, "median_ci": [lo, hi], "p95": float(np.percentile(lat, 95)),
                           "n": len(lat_rows), "art_s": arts, "within_art": within}
        M("EFiveLatMed", med)
        M("EFiveLatMedLo", lo)
        M("EFiveLatMedHi", hi)
        M("EFiveLatPNF", float(np.percentile(lat, 95)))
        M("EFiveLatN", len(lat_rows), "{:d}")
        M("EFiveART", arts[0] if len(arts) == 1 else float("nan"), "{:d}")
        R("EFiveWithinART", within)

    # ---- dispositions of briefs that listed the critical alarm (all LLM experiments, C1/C3) ----
    listed = [r for e in LLM_EXPS for r in load(e)
              if r["scenario"] in P1_SCEN and r.get("model") != "rule" and r.get("p1_surfaced")]
    if listed:
        valid = ("escalate", "monitor", "dismiss")  # anything else (e.g. free text in the field) is invalid
        disp = Counter(r.get("disposition") if r.get("disposition") in valid else "invalid" for r in listed)
        S["listed_dispositions"] = dict(disp)
        M("NListed", len(listed), "{:d}")
        M("NListedDismissed", disp.get("dismiss", 0), "{:d}")
        M("NListedEscalated", disp.get("escalate", 0), "{:d}")
        M("NListedMonitor", disp.get("monitor", 0), "{:d}")
        M("NListedOther", disp.get("invalid", 0), "{:d}")
        esc = {}
        for m, nm in zip(CORE_ORDER, ["Mistral", "QwenFourteen", "QwenSeven", "Coder"]):
            ml = [r for r in listed if r["model"] == m]
            esc[m] = ratek(sum(r.get("disposition") == "escalate" for r in ml), len(ml))
            R(f"Esc{nm}", esc[m])
        S["escalation_among_listed_by_model"] = esc

    # ---- do both arms of each tested pair share one alarm stream? (equal raw alarm counts) ----
    def same_stream(rows, keyfn):
        idx = defaultdict(dict)
        for r in rows:
            idx[keyfn(r)][bool(r["floodgate"])] = r.get("n_alarms_total")
        prs = [v for v in idx.values() if len(v) == 2]
        return sum(v[False] == v[True] for v in prs), len(prs)
    checks = {
        "Five": (e5, lambda r: (r["model"], r["scenario"], r["attack"], r["flood_rate"], seed_of(r))),
        "Spoof": ([r for r in esp if r["severity_source"] == "device" and r.get("spoof")],
                  lambda r: (r["model"], r["scenario"], seed_of(r))),
        "NF": ([r for r in enf if not r.get("spoof")], lambda r: (r["model"], r["scenario"], seed_of(r))),
        "Six": (e6, lambda r: (r["model"], r["scenario"], r["attack"], r["flood_rate"], r.get("position"),
                               seed_of(r))),
    }
    S["pairs_same_stream"] = {}
    for nm, (rows, keyfn) in checks.items():
        if rows:
            same, total = same_stream(rows, keyfn)
            S["pairs_same_stream"][nm] = [same, total]
            M(f"PairsSame{nm}", same, "{:d}")
            M(f"PairsTotal{nm}", total, "{:d}")
    if e6:   # E6 again, without pairs whose arms had different streams (an arm re-run in a new process)
        idx6 = defaultdict(dict)
        for r in e6:
            idx6[checks["Six"][1](r)][bool(r["floodgate"])] = r
        mixed = [v for v in idx6.values() if len(v) == 2 and v[False].get("n_alarms_total") != v[True].get("n_alarms_total")]
        S["E6_mixed_stream_pairs"] = [{"id_none": v[False]["id"], "surfaced": [v[False].get("p1_surfaced"),
                                       v[True].get("p1_surfaced")]} for v in mixed]
        same6 = [(bool(v[False].get("p1_surfaced")), bool(v[True].get("p1_surfaced"))) for v in idx6.values()
                 if len(v) == 2 and v[False].get("n_alarms_total") == v[True].get("n_alarms_total")]
        b, c, p = mcnemar(same6)
        S["E6_mcnemar_same_stream"] = {"b_none_only": b, "c_fg_only": c, "p": p, "n_pairs": len(same6)}
        MP("ESixSameMcNemarP", p)
        M("ESixSameMcNemarFGonly", c, "{:d}")
        M("ESixSameMcNemarNoneOnly", b, "{:d}")

    # ---- global counts (ALL experiments) ----
    allrows = sum((load(e) for e in REGISTRY_NAMES), [])
    S["n_episodes_total"] = len(allrows)
    S["n_episodes_by_experiment"] = {e: len(load(e)) for e in REGISTRY_NAMES}
    M("NEpisodes", len(allrows), "{:d}")
    M("NEpisodesCore", sum(r.get("model") in CORE_ORDER for r in allrows), "{:d}")   # episodes of the four core models
    M("NModelsCore", len(CORE_ORDER), "{:d}")
    M("NFailed", sum(n_failed(e) for e in REGISTRY_NAMES), "{:d}")
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
             "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
    macros["NModelsCoreWord"] = words[len(CORE_ORDER)]
    macros["NExperimentsWord"] = words[len(REGISTRY_NAMES)] if len(REGISTRY_NAMES) < len(words) else str(len(REGISTRY_NAMES))
    if "E1_lowest_model" in S:   # models that stayed near ceiling at every rate with the MADB filter
        macros["NModelsHighWord"] = words[len([m for m in CORE_ORDER if m in S["E1_by_model"]]) - 1]

    _testbed_facts(S, M)
    _make_tables(S)
    _write_fig_data(S)   # TikZ/pgfplots figures read these tables
    (RESULTS / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    tex = "% auto-generated by experiments/analyze.py -- do not edit\n"
    for k in sorted(macros):
        tex += f"\\newcommand{{\\{k}}}{{{macros[k]}}}\n"
    (PAPER / "macros.tex").write_text(tex)
    print(f"analyzed {S['n_episodes_total']} episodes; {len(macros)} macros -> {PAPER.name}/macros.tex")
    for k in ["E1_surfacing_by_rate", "E5_surf_by_fg", "E5_mcnemar", "E3_wrong_dismiss_by_historian"]:
        if k in S:
            print(" ", k, json.dumps(S[k], default=lambda x: round(x, 3)))
    return S


MODEL_TAG = {"qwen3-coder:30b": "qwen3-coder:30b", "mistral-small3.1:24b": "mistral-small3.1:24b",
             "qwen2.5:14b-instruct-q4_K_M": "qwen2.5:14b", "qwen2.5:7b": "qwen2.5:7b",
             "llama3.1:8b-instruct-q4_K_M": "llama3.1:8b"}
MODEL_CODE = {"qwen3-coder:30b": "C", "mistral-small3.1:24b": "M", "qwen2.5:14b-instruct-q4_K_M": "Q14",
              "qwen2.5:7b": "Q7", "llama3.1:8b-instruct-q4_K_M": "L8", "qwen3.8:latest": "Q27", "rule": "R",
              "guard": "G"}
ATTACK_CODE = {"none": "none", "volume_dilution": "A1", "decoy_narrative": "A2", "grouping_capture": "A3",
               "evidence_starvation": "A4", "random_control": "ctrl"}
# Experiment ledger: (id, question, follow-up added after the first results?), in the order the Results use them.
# LEDGER_GROUPS names the Results subsection that reports each run of experiments (first experiment -> heading).
LEDGER_GROUPS = {
    "E1": ("Priority filtering and floods", "sec:res-filter"),
    "Espoof": ("Severity spoofing", "sec:res-spoof"),
    "E5": ("What \\textup{\\textsc{FloodGate}} adds", "sec:res-floodgate"),
    "E9": ("Presentation and the rule reference", "sec:res-rule"),
    "E2": ("Null results", "sec:res-nulls"),
    "Eguard": ("Checks", "sec:res-checks"),
}
LEDGER = [
    ("E1", "Flood volume with the MADB filter; critical-alarm position", False),
    ("E1nf", "Flood volume without a filter", True),
    ("E1aff", "Filter on or off; MADB or device severity", False),
    ("E1ctx", "8k versus 32k context", False),
    ("Espoof", "Device-severity spoofing, filter on", True),
    ("EspoofNF", "Spoofing with the filter off", True),
    ("Emvob", "Spoofing on the mv\\_oberrhein grid", True),
    ("E5", "FloodGate across the main attack grid", False),
    ("E4", "Defense baselines", False),
    ("E6", "Flood designs aimed at FloodGate", False),
    ("E8", "Contingency floods; no-event windows", False),
    ("E9", "Non-LLM rule versus LLMs; presentation", True),
    ("E2", "Structured floods versus matched control", False),
    ("E2b", "Decoy versus control, filter off", True),
    ("E3", "Historian denial (evidence starvation)", False),
    ("E7", "Isolate-then-reconcile protocol", False),
    ("Eguard", "Injected device text (sanity check)", False),
    ("Eguard2", "Llama Guard 3 on flood windows", True),
]
TABLES = PAPER / "tables"


def _ci(t):
    return f"{half_up(100*t[0])} [{half_up(100*t[1])},{half_up(100*t[2])}]" if t and not math.isnan(t[0]) else "--"


def _pct(t):
    return str(half_up(100 * t[0])) if t and not math.isnan(t[0]) else "--"


def _p_tex(p: float) -> str:
    """Format a p-value for LaTeX math mode, to two significant figures."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "\\text{--}"
    if p < 1e-3:
        exp = int(math.floor(math.log10(p)))
        mant = p / (10 ** exp)
        if round(mant, 1) >= 10:
            mant, exp = mant / 10, exp + 1
        return f"{mant:.1f}\\times10^{{{exp}}}"
    return f"{p:.2g}"


def _testbed_facts(S, M):
    """Testbed constants the paper quotes (MADB size and mix, second grid, FloodGate and harness parameters,
    property-test count), read from the code rather than typed. Skipped (macros stay '--') if the testbed cannot be
    imported."""
    import re
    import subprocess
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT.parent))
    try:
        import pandapower.networks as pn
        from fb.madb import build_madb
    except Exception as e:  # results analysis must not depend on the testbed being importable
        print("testbed facts skipped:", e)
        return
    st = build_madb("case39").stats()
    low = st["by_priority"]["P3"] + st["by_priority"]["P4"]
    net = pn.mv_oberrhein()
    S["testbed"] = {"case39_madb": st, "mv_oberrhein": {"n_bus": len(net.bus), "n_sgen": len(net.sgen)}}
    M("NMadbPoints", st["n_points"], "{:d}")
    M("MadbLowPct", 100 * low / st["n_points"])
    M("NMadbFloodable", st["n_floodable"], "{:d}")
    for p, nm in [("P1", "One"), ("P2", "Two"), ("P3", "Three"), ("P4", "Four")]:
        M(f"NMadbP{nm}", st["by_priority"][p], "{:d}")
    M("NMvobBus", len(net.bus), "{:d}")
    M("NMvobSgen", len(net.sgen), "{:d}")
    try:
        from fb.alarms import EEMUA_NORMAL_MAX, EEMUA_NORMAL_MIN, ISA_FLOOD_THRESHOLD
        from fb.floodgate import BUCKET_LOW_PER_SOURCE, LANE_LOW, LANE_P1, LANE_P2
        from fb.tools import PAGE_SIZE
        M("LaneOne", LANE_P1, "{:d}")
        M("LaneTwo", LANE_P2, "{:d}")
        M("LaneLow", LANE_LOW, "{:d}")
        M("BucketLow", BUCKET_LOW_PER_SOURCE, "{:d}")
        M("PageSize", PAGE_SIZE, "{:d}")
        M("BgMin", EEMUA_NORMAL_MIN, "{:d}")
        M("BgMax", EEMUA_NORMAL_MAX, "{:d}")
        M("FloodThreshold", ISA_FLOOD_THRESHOLD, "{:d}")
        top = max((int(k) for k in S.get("E1_surfacing_by_rate", {})), default=0)
        if top:
            M("FloodMultiple", top / ISA_FLOOD_THRESHOLD)
        from fb.agent import MAX_STEPS
        M("MaxSteps", MAX_STEPS, "{:d}")
    except Exception as e:
        print("parameter facts skipped:", e)
    tf = RESULTS / "admission_timing.json"   # measured once by experiments/time_admission.py (machine-dependent)
    if tf.exists():
        timing = json.loads(tf.read_text())
        S["admission_pass_ms"] = timing
        M("AdmitMs", timing["median_ms"], "{:.1f}")
        M("AdmitAlarms", timing["n_alarms"], "{:d}")
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    m = re.search(r"(\d+) tests? collected", out)
    if m:
        S["testbed"]["n_property_tests"] = int(m.group(1))
        M("NPropTests", int(m.group(1)), "{:d}")


def _scen_label(scens):
    s = sorted(scens)
    names = [x for x in s if x != "BENIGN"]
    label = "C1--C3" if names == ["C1", "C2", "C3"] else ", ".join(names)
    if "BENIGN" in s:
        label = (label + "; " if label else "") + "no event"
    return label


def _flood_label(rows):
    by = defaultdict(set)
    for r in rows:
        by[ATTACK_CODE.get(r.get("attack"), r.get("attack"))].add(int(r.get("flood_rate", 0)))
    order = ["A1", "A2", "A3", "A4", "ctrl", "none"]
    parts = []
    for a in sorted(by, key=lambda x: order.index(x) if x in order else 99):
        rates = sorted(by[a])
        parts.append(a if a == "none" else f"{a} " + "/".join(str(x) for x in rates))
    return "; ".join(parts)


def _filter_label(rows):
    lab = set()
    for r in rows:
        if "severity_filter" not in r:
            continue
        lab.add(("MADB" if r.get("severity_source") == "madb" else "device") if r["severity_filter"] else "off")
    return "/".join(sorted(lab, key=lambda x: ["MADB", "device", "off"].index(x))) if lab else "--"


def _make_tables(S):
    TABLES.mkdir(parents=True, exist_ok=True)
    for stale in ("e4.tex", "e5_floodgate.tex", "e9.tex"):     # superseded by fgcompare.tex and the text
        (TABLES / stale).unlink(missing_ok=True)
    tag = lambda m: f"\\texttt{{{MODEL_TAG.get(m, m)}}}"      # noqa: E731

    # Experiment ledger (design read from the logs; questions from the job definitions)
    rows_tex = []
    for exp, question, follow in LEDGER:
        rows = load(exp)
        if not rows:
            continue
        models = " ".join(MODEL_CODE.get(m, m) for m in sorted({r["model"] for r in rows},
                                                                 key=lambda m: list(MODEL_CODE).index(m)
                                                                 if m in MODEL_CODE else 99))
        fg = "/".join(sorted({"yes" if r.get("floodgate") else "no" for r in rows}, key=["no", "yes"].index))
        name = exp + ("$^\\dagger$" if follow else "")
        if exp in LEDGER_GROUPS:
            head, lab = LEDGER_GROUPS[exp]
            if rows_tex:
                rows_tex.append(r"\addlinespace[2pt]")
            rows_tex.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{head}}} (\\cref{{{lab}}})}} \\\\")
        rows_tex.append(f"{name} & {question} & {models} & {_scen_label({r['scenario'] for r in rows})} & "
                        f"{_flood_label(rows)} & {_filter_label(rows)} & {fg} & {len(rows)} \\\\")
    if rows_tex:
        failed = [f"{e}: {n_failed(e)}" for e in REGISTRY_NAMES if n_failed(e)]
        lines = [r"\begin{table}[H]\centering",
                 r"\caption{Experiments, grouped by the Results subsection that reports them. Models: C \texttt{qwen3-coder:30b}, "
                 r"M \texttt{mistral-small3.1:24b}, Q14 \texttt{qwen2.5:14b}, Q7 \texttt{qwen2.5:7b}, "
                 r"L8 \texttt{llama3.1:8b}, Q27 \texttt{qwen3.8} (27B), R non-LLM rule, G Llama Guard~3. "
                 r"Floods: attack and rate (alarms per 10\,min); ctrl is the matched-volume random control. "
                 r"Filter: severity filter on MADB or device priority, or off. FG: \textsc{FloodGate} off or on. "
                 r"$n$: episodes. $^\dagger$Follow-up experiments added after the first results.}",
                 r"\label{tab:ledger}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
                 r"\begin{tabularx}{\textwidth}{@{}l >{\raggedright\arraybackslash}X l l "
                 r">{\raggedright\arraybackslash}X l l r@{}}", r"\toprule",
                 r"Exp. & Question & Models & Scenarios & Flood & Filter & FG & $n$ \\",
                 r"\midrule", *rows_tex, r"\midrule",
                 f"\\multicolumn{{7}}{{@{{}}l}}{{Total (failed jobs excluded: {', '.join(failed) or 'none'})}} & "
                 f"{S.get('n_episodes_total', '--')} \\\\",
                 r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
        (TABLES / "ledger.tex").write_text("\n".join(lines))

    # FloodGate versus no FloodGate, per model, with and without the MADB-sourced filter
    e5m, nfm = S.get("E5_by_model", {}), S.get("EspoofNF_clean_by_model", {})
    if e5m and nfm:
        lines = [r"\begin{table}[H]\centering",
                 r"\caption{P1 surfacing without and with \textsc{FloodGate}, per model (k/n): with the MADB "
                 r"filter under the E5 attack grid, and with the filter off (EspoofNF, clean stream).}",
                 r"\label{tab:fgcompare}", r"\small", r"\setlength{\tabcolsep}{4pt}",
                 r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
                 r"& \multicolumn{2}{c}{MADB filter (E5)} & \multicolumn{2}{c}{Filter off (EspoofNF)} \\",
                 r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
                 r"Model & without & with & without & with \\", r"\midrule"]
        for m in CORE_ORDER:
            if m in e5m and m in nfm:
                lines.append(f"{tag(m)} & {kn(e5m[m].get('none'))} & {kn(e5m[m].get('fg'))} & "
                             f"{kn(nfm[m].get('none'))} & {kn(nfm[m].get('fg'))} \\\\")
        e5, nf = S.get("E5_surf_by_fg", {}), S.get("EspoofNF_surf", {})
        m5, mn = S.get("E5_mcnemar", {}), S.get("EspoofNF_clean_mcnemar", {})
        lines += [r"\midrule",
                  f"Pooled (\\%, 95\\% CI) & {_ci(e5.get('none'))} & {_ci(e5.get('fg'))} & "
                  f"{_ci(nf.get('clean_none'))} & {_ci(nf.get('clean_fg'))} \\\\",
                  f"Pairs fixed / broken & \\multicolumn{{2}}{{c}}{{{m5.get('c_fg_only', '--')} / "
                  f"{m5.get('b_none_only', '--')} of {m5.get('n_pairs', '--')}}} & "
                  f"\\multicolumn{{2}}{{c}}{{{mn.get('c_fg_only', '--')} / {mn.get('b_none_only', '--')} of "
                  f"{mn.get('n_pairs', '--')}}} \\\\",
                  f"Exact McNemar & \\multicolumn{{2}}{{c}}{{$p={_p_tex(m5.get('p'))}$}} & "
                  f"\\multicolumn{{2}}{{c}}{{$p={_p_tex(mn.get('p'))}$}} \\\\",
                  r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (TABLES / "fgcompare.tex").write_text("\n".join(lines))

    # E1 dose-response per model x rate (k/n), with a pooled row
    bm = S.get("E1_by_model", {})
    if bm:
        rates = sorted({int(k) for d in bm.values() for k in d["by_rate"]})
        lines = [r"\begin{table}[H]\centering",
                 r"\caption{P1 surfacing (k/n) by flood rate (alarms per 10\,min) and model, with the MADB filter "
                 r"(E1). Pooled row: \%, Wilson 95\% CI.}",
                 r"\label{tab:e1}", r"\small", r"\setlength{\tabcolsep}{4pt}",
                 r"\begin{tabular}{@{}l" + "c" * len(rates) + "@{}}", r"\toprule",
                 "Model & " + " & ".join(str(x) for x in rates) + r" \\", r"\midrule"]
        for mdl in CORE_ORDER:
            if mdl in bm:
                lines.append(" & ".join([tag(mdl)] + [kn(bm[mdl]["by_rate"].get(str(x))) for x in rates]) + r" \\")
        pooled = S.get("E1_surfacing_by_rate", {})
        lines += [r"\midrule", " & ".join(["Pooled"] + [_ci(pooled.get(str(x))) for x in rates]) + r" \\",
                  r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (TABLES / "e1_dose.tex").write_text("\n".join(lines))

    # Severity spoofing: filter on (Espoof) and filter off (EspoofNF)
    sp, nf = S.get("Espoof_surf", {}), S.get("EspoofNF_surf", {})
    if sp and nf:
        def cell(t):
            return f"{kn(t)} ({_pct(t)})" if t else "--"
        mc = S.get("Espoof_mcnemar", {})
        rows = [("device", "on", "no", sp.get("device_clean_none"), sp.get("device_clean_fg")),
                ("device", "on", "yes", sp.get("device_spoof_none"), sp.get("device_spoof_fg")),
                ("MADB", "on", "yes", sp.get("madb_spoof_none"), sp.get("madb_spoof_fg")),
                ("device", "off", "no", nf.get("clean_none"), nf.get("clean_fg")),
                ("device", "off", "yes", nf.get("spoof_none"), nf.get("spoof_fg"))]
        lines = [r"\begin{table}[H]\centering",
                 r"\caption{Severity spoofing: P1 surfacing, k/n (\%), without and with \textsc{FloodGate} "
                 r"(Espoof, EspoofNF).}", r"\label{tab:espoof}", r"\small", r"\setlength{\tabcolsep}{4pt}",
                 r"\begin{tabular}{@{}lllcc@{}}", r"\toprule",
                 r"Severity & Filter & Spoof & Without & With \textsc{FloodGate} \\", r"\midrule"]
        lines += [f"{a} & {b} & {c} & {cell(d)} & {cell(e)} \\\\" for a, b, c, d, e in rows]
        lines += [r"\bottomrule", r"\end{tabular}",
                  f"\\\\[3pt]{{\\footnotesize Device filter, spoofed: {mc.get('c_fg_only', '--')} pairs fixed, "
                  f"{mc.get('b_none_only', '--')} broken of {mc.get('n_pairs', '--')}; exact McNemar "
                  f"$p={_p_tex(mc.get('p'))}$.}}",
                  r"\end{table}"]
        (TABLES / "espoof.tex").write_text("\n".join(lines))


def _write_fig_data(S):
    """Emit pgfplots data tables (numeric only; labels live in the TikZ source) for every chart in the paper.
    Values are percentages with their Wilson 95% bounds, so the TikZ charts never contain hand-typed numbers."""
    D = PAPER / "figures" / "data"
    D.mkdir(parents=True, exist_ok=True)

    def pct(t):
        return (100 * t[0], 100 * t[1], 100 * t[2]) if t and not math.isnan(t[0]) else (float("nan"),) * 3

    def write(name, header, rows):
        with open(D / name, "w") as f:
            f.write(" ".join(header) + "\n")
            for r in rows:
                f.write(" ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + "\n")

    # dose-response: x = rate+1 (log axis), filter (E1) and no-filter (E1nf)
    for key, name in [("E1_surfacing_by_rate", "dose_filter.dat"), ("E1nf_surfacing_by_rate", "dose_nofilter.dat")]:
        d = S.get(key, {})
        rows = []
        for rt in sorted(d, key=lambda s: int(s)):
            m, lo, hi = pct(d[rt])
            rows.append((int(rt) + 1, m, lo, hi, d[rt][3]))
        if rows:
            write(name, ["x", "y", "lo", "hi", "n"], rows)

    # spoofing: five conditions x {no FloodGate, FloodGate}
    sp, nf = S.get("Espoof_surf", {}), S.get("EspoofNF_surf", {})
    conds = [(sp, "device_clean_none", "device_clean_fg"), (sp, "device_spoof_none", "device_spoof_fg"),
             (sp, "madb_spoof_none", "madb_spoof_fg"), (nf, "clean_none", "clean_fg"), (nf, "spoof_none", "spoof_fg")]
    rows = []
    for i, (src, a, b) in enumerate(conds):
        if src.get(a) and src.get(b):
            n_m, n_lo, n_hi = pct(src[a])
            f_m, f_lo, f_hi = pct(src[b])
            rows.append((i, n_m, n_m - n_lo, n_hi - n_m, f_m, f_m - f_lo, f_hi - f_m))
    if rows:
        write("spoof.dat", ["i", "none", "noneminus", "noneplus", "fg", "fgminus", "fgplus"], rows)

    # presentation vs surfacing (E9), rows top->bottom: rule/noFG, rule/FG, LLM/noFG, LLM/FG
    e9s, e9p = S.get("E9_surf", {}), S.get("E9_present", {})
    order = ["rule_none", "rule_fg", "llm_none", "llm_fg"]
    rows = []
    for i, k in enumerate(order):
        if e9s.get(k) and e9p.get(k):
            pm, plo, phi = pct(e9p[k])
            sm, slo, shi = pct(e9s[k])
            rows.append((len(order) - i, pm, sm, sm - slo, shi - sm))
    if rows:
        write("e9.dat", ["y", "pres", "surf", "surfminus", "surfplus"], rows)

    # nulls: (a) structure vs matched random control (E2 filter on, E2b filter off); (b) historian denial (E3)
    e2s, e2b = S.get("E2_surf_by_attack", {}), S.get("E2b_surf", {})
    rows = []
    spec = [(e2s, "decoy_narrative", 7, 1), (e2s, "grouping_capture", 6, 1), (e2s, "random_control", 5, 0),
            (e2b, "decoy_narrative", 3, 1), (e2b, "random_control", 2, 0)]
    for src, k, y, is_attack in spec:
        if src.get(k):
            m, lo, hi = pct(src[k])
            rows.append((y, m, m - lo, hi - m, is_attack))
    if rows:
        write("nulls_structure.dat", ["y", "m", "minus", "plus", "attack"], rows)
    bands = []   # x-extent of each group's random-control CI, for shading
    for src, y0, y1 in [(e2s, 4.5, 7.5), (e2b, 1.5, 3.5)]:
        if src.get("random_control"):
            m, lo, hi = pct(src["random_control"])
            bands.append((y0, y1, lo, hi))
    if bands:
        write("nulls_bands.dat", ["y0", "y1", "lo", "hi"], bands)
    e3 = S.get("E3_wrong_dismiss_by_historian", {})
    rows = []
    for i, k in enumerate(["normal", "empty", "timeout", "partial"]):
        if e3.get(k):
            m, lo, hi = pct(e3[k])
            rows.append((4 - i, m, m - lo, hi - m))
    if rows:
        write("nulls_e3.dat", ["y", "m", "minus", "plus"], rows)


if __name__ == "__main__":
    analyze()
