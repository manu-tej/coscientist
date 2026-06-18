"""Does the LLM orchestration layer earn its keep?

Setup: 5 quality checks exist, but you may run only K=2 (a budget). Each scenario is
a dataset with ONE planted artifact and a free-text description that *hints* at it in
domain language (never names the check). We compare detection recall of three routers:

    random-K  : pick K checks at random            (expected 2/5 = 40%)
    fixed-K   : always the same default K           (catches 2/5 = 40%)
    LLM-K     : read the description, pick K         (should beat both — IF context helps)

Detection counts only when the right check is BOTH selected AND fires on the data, so
this isolates the one thing the LLM could add over statistics: context-aware routing.
If LLM-K doesn't beat the baselines, the LLM layer is ceremony and we drop it.

Run:  python -m qc.value_experiment
"""
from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qc import detectors as D
from qc import synth
from qc.agent import QCScientist

AVAILABLE = ["missingness", "univariate_outlier", "covariate_shift", "batch_effect", "leakage"]
BUDGET = 2
N_PERM = 100
SUBSAMPLE = 250


@dataclass
class Scenario:
    gt_check: str
    description: str


# Two phrasings per artifact — domain language, never the check name.
SCENARIOS = [
    Scenario("batch_effect", "Samples were collected at two hospital sites whose mass-spec "
             "instruments were recalibrated between collection waves."),
    Scenario("batch_effect", "Half the cohort was processed by the night-shift lab on a "
             "different sequencer lot than the rest."),
    Scenario("covariate_shift", "The model was trained on a 2018 cohort and we are now scoring "
             "patients from a 2024 deployment population."),
    Scenario("covariate_shift", "Reference samples came from an academic center; the new batch "
             "is from community clinics with a different patient mix."),
    Scenario("missingness", "Several assay runs failed silently this quarter, so some marker "
             "readings may simply be absent for many patients."),
    Scenario("missingness", "A subset of records was migrated from a legacy LIMS where certain "
             "fields were never populated."),
    Scenario("univariate_outlier", "A few wells had pipetting errors that produced wildly "
             "extreme intensity reads for isolated measurements."),
    Scenario("univariate_outlier", "Some samples were left at room temperature and show "
             "occasional implausibly large single-feature spikes."),
    Scenario("leakage", "One column was derived post-hoc by an analyst and may inadvertently "
             "encode the clinical outcome we are trying to predict."),
    Scenario("leakage", "A feature was joined in from a downstream table that is only populated "
             "after the diagnosis is known."),
]


def _materials(gt_check: str, seed: int) -> dict:
    X0, y0, names0 = synth.load_base()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X0), size=SUBSAMPLE, replace=False)
    X, y, names = X0[idx].copy(), y0[idx].copy(), list(names0)
    batch = (rng.random(len(X)) < 0.5).astype(int)          # clean by default
    ref, new = synth.split_two(X, seed)                      # clean by default
    if gt_check == "missingness":
        X = synth.inject_missingness(X, seed, rate=0.15)
    elif gt_check == "univariate_outlier":
        X = synth.inject_outliers(X, seed, frac=0.03, magnitude=12.0)
    elif gt_check == "covariate_shift":
        new = synth.inject_covariate_shift(new, seed, strength=0.9)
    elif gt_check == "batch_effect":
        X, batch = synth.inject_batch_effect(X, seed, strength=0.9)
    elif gt_check == "leakage":
        X, names = synth.inject_leakage(X, y, names, seed)
    return {"X": X, "y": y, "names": names, "batch": batch, "ref": ref, "new": new}


def _fires(check: str, m: dict, seed: int) -> bool:
    if check == "missingness":
        return bool(D.detect_missingness(m["X"], m["names"]))
    if check == "univariate_outlier":
        return bool(D.detect_univariate_outliers(m["X"], m["names"]))
    if check == "covariate_shift":
        return any(f.check == "covariate_shift" for f in
                   D.detect_covariate_shift(m["ref"], m["new"], m["names"], seed=seed, n_perm=N_PERM))
    if check == "batch_effect":
        return any(f.check == "batch_effect" for f in
                   D.detect_batch_effect(m["X"], m["batch"], seed=seed, n_perm=N_PERM))
    if check == "leakage":
        return any(f.target == "__leaked__" for f in
                   D.detect_leakage(m["X"], m["y"], m["names"], seed=seed, auc_threshold=0.98))
    return False


def _random_recall(fired_flags: list[bool]) -> float:
    """Exact expected recall of picking BUDGET of len(AVAILABLE) checks uniformly."""
    subsets = list(itertools.combinations(range(len(AVAILABLE)), BUDGET))
    # each scenario's gt check is at a fixed index; averaging over subsets that include it
    hit = 0.0
    for i, fired in enumerate(fired_flags):
        if not fired:
            continue
        gt_idx = AVAILABLE.index(SCENARIOS[i].gt_check)
        frac_subsets_with_gt = sum(gt_idx in s for s in subsets) / len(subsets)
        hit += frac_subsets_with_gt
    return hit / len(fired_flags)


async def run(backend=None) -> dict:
    seeds = [4242 + i for i in range(len(SCENARIOS))]
    fired = [_fires(sc.gt_check, _materials(sc.gt_check, s), s) for sc, s in zip(SCENARIOS, seeds)]

    scientist = QCScientist(backend=backend, budget=BUDGET)
    fixed = set(["covariate_shift", "univariate_outlier"][:BUDGET])

    llm_hits = fixed_hits = 0
    rows = []
    for sc, s, did_fire in zip(SCENARIOS, seeds, fired):
        sel = await scientist.select_checks(sc.description, available=AVAILABLE)
        llm_hit = (sc.gt_check in sel.checks) and did_fire
        fixed_hit = (sc.gt_check in fixed) and did_fire
        llm_hits += llm_hit
        fixed_hits += fixed_hit
        rows.append((sc.gt_check, sel.source, sel.checks, llm_hit))
    n = len(SCENARIOS)
    return {
        "n": n,
        "llm_recall": llm_hits / n,
        "fixed_recall": fixed_hits / n,
        "random_recall": _random_recall(fired),
        "rows": rows,
        "fired": fired,
    }


def _print(res: dict) -> None:
    print(f"\n══ Does the QC LLM earn its keep? — budget {BUDGET}/{len(AVAILABLE)} checks, "
          f"n={res['n']} scenarios ══\n")
    print(f"  {'router':<12}{'detection recall':>18}")
    print("  " + "─" * 30)
    print(f"  {'random-K':<12}{res['random_recall']:>17.0%}")
    print(f"  {'fixed-K':<12}{res['fixed_recall']:>17.0%}")
    print(f"  {'LLM-K':<12}{res['llm_recall']:>17.0%}")
    delta = res["llm_recall"] - max(res["random_recall"], res["fixed_recall"])
    verdict = ("EARNS ITS KEEP" if delta > 0.1 else
               "MARGINAL" if delta > 0 else "DOES NOT — drop the LLM layer")
    print(f"\n  LLM lift over best baseline: {delta:+.0%}  →  {verdict}")
    print("\n  per-scenario LLM routing:")
    for gt, src, picks, hit in res["rows"]:
        mark = "✓" if hit else "✗"
        print(f"    {mark} planted={gt:<18} [{src}] picked {picks}")
    if not all(res["fired"]):
        missed = [SCENARIOS[i].gt_check for i, f in enumerate(res["fired"]) if not f]
        print(f"\n  note: detector did not fire for: {missed} (selection couldn't have helped there)")
    print()


async def _main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    import yaml
    from tools.llm import make_backend
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    backend = make_backend(cfg)
    res = await run(backend=backend)
    _print(res)


if __name__ == "__main__":
    asyncio.run(_main())
