"""Falsifiable evaluation: does the QC battery actually work, and is it calibrated?

Two questions, both measured on real data over seeded replicates:

  POWER       — inject a known corruption; does the right check fire? (recall)
  CALIBRATION — on genuinely-null clean data, how often does a distribution
                detector fire by chance? It must track the nominal alpha, else its
                "confidence" is a lie.

Run:  python -m qc.harness
"""
from __future__ import annotations

from dataclasses import dataclass

from qc import detectors as D
from qc import synth


@dataclass
class ScoreRow:
    name: str
    value: float
    n: int
    note: str


def _fired(flags, check: str) -> bool:
    return any(f.check == check for f in flags)


def run_power(n_reps: int = 15, alpha: float = 0.05, n_perm: int = 120) -> list[ScoreRow]:
    X, y, names = synth.load_base()
    hits = {k: 0 for k in ["batch_effect", "covariate_shift", "missingness",
                           "univariate_outlier", "leakage"]}
    for r in range(n_reps):
        s = 1000 + r
        # batch effect
        Xb, batch = synth.inject_batch_effect(X, s, strength=0.8)
        hits["batch_effect"] += _fired(D.detect_batch_effect(Xb, batch, alpha=alpha, seed=s, n_perm=n_perm), "batch_effect")
        # covariate shift: ref vs a shifted new half
        ref, new = synth.split_two(X, s)
        new = synth.inject_covariate_shift(new, s, strength=0.8)
        hits["covariate_shift"] += _fired(D.detect_covariate_shift(ref, new, names, alpha=alpha, seed=s, n_perm=n_perm), "covariate_shift")
        # missingness
        Xm = synth.inject_missingness(X, s, rate=0.15)
        hits["missingness"] += _fired(D.detect_missingness(Xm, names), "missingness")
        # univariate outliers
        Xo = synth.inject_outliers(X, s, frac=0.03, magnitude=12.0)
        hits["univariate_outlier"] += _fired(D.detect_univariate_outliers(Xo, names), "univariate_outlier")
        # leakage (on the moderate synthetic so real biomarkers don't confound)
        Xs, ys, ns = synth.make_moderate(s)
        Xl, nl = synth.inject_leakage(Xs, ys, ns, s)
        hits["leakage"] += _fired(D.detect_leakage(Xl, ys, nl, alpha=alpha, auc_threshold=0.98, seed=s), "leakage")
    return [ScoreRow(f"power:{k}", v / n_reps, n_reps, "recall on injected corruption")
            for k, v in hits.items()]


def run_calibration(n_reps: int = 24, alpha: float = 0.05, n_perm: int = 120) -> list[ScoreRow]:
    """False-positive rate of the distribution detectors on genuine nulls.

    covariate_shift: two random halves of the same data (no shift).
    batch_effect:    a random batch label uncorrelated with features.
    A calibrated detector fires <= ~alpha of the time.
    """
    X, _, names = synth.load_base()
    fp = {"covariate_shift": 0, "batch_effect": 0}
    for r in range(n_reps):
        s = 5000 + r
        ref, new = synth.split_two(X, s)
        fp["covariate_shift"] += _fired(D.detect_covariate_shift(ref, new, names, alpha=alpha, seed=s, n_perm=n_perm), "covariate_shift")
        import numpy as np
        batch = (np.random.default_rng(s).random(len(X)) < 0.5).astype(int)
        fp["batch_effect"] += _fired(D.detect_batch_effect(X, batch, alpha=alpha, seed=s, n_perm=n_perm), "batch_effect")
    return [ScoreRow(f"calibration_FPR:{k}", v / n_reps, n_reps, f"should be <= alpha={alpha}")
            for k, v in fp.items()]


def scorecard(power_reps: int = 15, calib_reps: int = 24, alpha: float = 0.05, n_perm: int = 120):
    power = run_power(power_reps, alpha, n_perm)
    calib = run_calibration(calib_reps, alpha, n_perm)
    return power, calib


def _print(power, calib, alpha=0.05) -> None:
    print(f"\n══ QC scientist — falsifiable scorecard (real breast-cancer data, alpha={alpha}) ══\n")
    print("  DETECTION POWER (higher = better; injected a known artifact)")
    for row in power:
        bar = "█" * int(round(row.value * 20))
        print(f"    {row.name:<28} {row.value:5.0%}  {bar:<20}  (n={row.n})")
    print("\n  CALIBRATION (false-positive rate on null data; must be <= alpha)")
    for row in calib:
        ok = "OK " if row.value <= alpha + 0.02 else "HOT"
        print(f"    {row.name:<28} {row.value:5.0%}  [{ok}]  (n={row.n})")
    print("\n  Read: power says 'we catch real problems'; calibration says 'we don't cry wolf'.")
    print("  Both real & measured — no LLM in the loop, no hardcoded answers.\n")


if __name__ == "__main__":
    p, c = scorecard()
    _print(p, c)
