"""Hardening study: where does the LLM router BREAK?

The earlier value experiment used clean, well-separated clues and the LLM scored
100%. That's the easy case. Here we stress it on purpose:

  explicit   — a strong clue points at the planted artifact   (control)
  vague      — a weak, generic hint                            (information-poor)
  absent     — no clue at all; generic description             (LLM should ~= baseline)
  misleading — the description points at the WRONG check       (adversarial / red herring)

and we trace a BUDGET CURVE (recall@K for K=1..4) by asking the LLM to rank all
checks once per scenario. Honest expectations: recall should fall from explicit →
vague → absent, and on `misleading` the danger sign is recall dropping BELOW the
random baseline (actively fooled). We report the fooled-rate explicitly.

Run:  python -m qc.hardening
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from qc.agent import QCScientist
from qc.value_experiment import AVAILABLE, N_PERM, _fires, _materials  # reuse the verified core

K_MAX = 4


@dataclass
class HScenario:
    gt_check: str
    level: str           # explicit | vague | absent | misleading
    description: str
    herring: str = ""    # for misleading: the check the description points at instead


_GENERIC = "A tabular biomedical dataset of patient measurements assembled for a predictive-modeling project."

# herring map for adversarial scenarios (description points here; data has gt elsewhere)
_HERRING = {
    "missingness": "batch_effect", "univariate_outlier": "covariate_shift",
    "covariate_shift": "leakage", "batch_effect": "missingness", "leakage": "univariate_outlier",
}
_HERRING_TEXT = {
    "batch_effect": "Samples were processed at two different sites on separate instruments.",
    "covariate_shift": "The training cohort is from 2018 and the scoring cohort from 2024.",
    "leakage": "One column was derived post-hoc and might encode the clinical outcome.",
    "missingness": "Several assay runs failed silently, so many readings may be absent.",
    "univariate_outlier": "A few wells had pipetting errors giving wildly extreme reads.",
}
_EXPLICIT = dict(_HERRING_TEXT)  # same strong phrasing, used as the correct clue
_VAGUE = {
    "missingness": "Data completeness has been a bit inconsistent across this study.",
    "univariate_outlier": "A handful of measurements look unusual but we're not sure why.",
    "covariate_shift": "The population we're applying this to isn't quite the original one.",
    "batch_effect": "Processing happened in a few separate operational waves.",
    "leakage": "Some columns were engineered late in the pipeline by different analysts.",
}


def build_scenarios() -> list[HScenario]:
    s: list[HScenario] = []
    for gt in AVAILABLE:
        s.append(HScenario(gt, "explicit", _EXPLICIT[gt]))
        s.append(HScenario(gt, "vague", _VAGUE[gt]))
        s.append(HScenario(gt, "absent", _GENERIC))
        s.append(HScenario(gt, "misleading", _HERRING_TEXT[_HERRING[gt]], herring=_HERRING[gt]))
    return s


async def run(backend=None) -> dict:
    scenarios = build_scenarios()
    sci = QCScientist(backend=backend)
    fixed_order = ["covariate_shift", "univariate_outlier"] + \
                  [c for c in AVAILABLE if c not in ("covariate_shift", "univariate_outlier")]

    records = []
    for i, sc in enumerate(scenarios):
        seed = 7000 + i
        fires = _fires(sc.gt_check, _materials(sc.gt_check, seed), seed)
        ranked = (await sci.rank_checks(sc.description, available=AVAILABLE)).checks
        gt_rank = ranked.index(sc.gt_check)              # 0-based position the LLM gave gt
        fooled = bool(sc.herring) and ranked.index(sc.herring) < gt_rank
        records.append({"gt": sc.gt_check, "level": sc.level, "fires": fires,
                        "gt_rank": gt_rank, "ranked": ranked, "fooled": fooled})
    return {"records": records, "fixed_order": fixed_order}


def _recall_at(records, K, source) -> float:
    hit = 0.0
    for r in records:
        if not r["fires"]:
            continue
        if source == "llm":
            hit += r["gt_rank"] < K
        elif source == "random":
            hit += K / len(AVAILABLE)
    return hit / len(records)


def _print(res: dict) -> None:
    recs = res["records"]
    levels = ["explicit", "vague", "absent", "misleading"]
    print("\n══ QC router hardening — trying to break the LLM (n=20: 5 checks × 4 conditions) ══\n")

    print("  BUDGET CURVE — recall@K on EXPLICIT scenarios (LLM vs random baseline)")
    expl = [r for r in recs if r["level"] == "explicit"]
    print(f"    {'K':>3}{'LLM':>9}{'random':>9}")
    for K in range(1, K_MAX + 1):
        llm = _recall_at(expl, K, "llm"); rnd = _recall_at(expl, K, "random")
        print(f"    {K:>3}{llm:>9.0%}{rnd:>9.0%}")

    print("\n  CLUE-STRENGTH CURVE — recall@2 by condition (the honesty test)")
    print(f"    {'condition':<12}{'LLM recall@2':>14}{'random':>9}")
    for lv in levels:
        rs = [r for r in recs if r["level"] == lv]
        llm = _recall_at(rs, 2, "llm"); rnd = _recall_at(rs, 2, "random")
        tag = "  <- should ~= baseline" if lv == "absent" else \
              "  <- danger if < baseline" if lv == "misleading" else ""
        print(f"    {lv:<12}{llm:>13.0%}{rnd:>9.0%}{tag}")

    mis = [r for r in recs if r["level"] == "misleading"]
    fooled = sum(r["fooled"] for r in mis) / len(mis)
    print(f"\n  ADVERSARIAL: on misleading clues the LLM ranked the red herring above the true "
          f"problem {fooled:.0%} of the time.")
    print("  Read: a robust router degrades gracefully (vague>absent) and isn't dragged BELOW")
    print("  random by misdirection. Below-baseline on misleading = the LLM can be fooled.\n")


async def _main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    import yaml
    from tools.llm import make_backend
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    res = await run(backend=make_backend(cfg))
    _print(res)


if __name__ == "__main__":
    asyncio.run(_main())
