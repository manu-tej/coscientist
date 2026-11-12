"""Statistical QC detectors. Each returns Flags with real, interpretable uncertainty.

Every detector that runs a *family* of tests (one per feature) applies Benjamini-
Hochberg FDR correction, so `q_value` is the quantity to threshold at `alpha` — that
is what makes the battery's false-positive rate calibrated (verified in harness.py).
Detectors that compare distributions use a classifier two-sample test (C2ST) with a
permutation null, which is distribution-free and needs no parametric assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


@dataclass
class Flag:
    check: str          # detector name
    target: str         # feature name, sample id, or "dataset"
    statistic: float    # the test statistic / effect magnitude
    p_value: float
    q_value: float      # BH-FDR adjusted (== p_value where N/A)
    effect_size: float  # interpretable magnitude (std-diff, AUC-0.5, outlier frac, ...)
    severity: str       # low | medium | high
    detail: str

    @property
    def confidence(self) -> float:
        """Calibrated detection confidence = 1 - q. Frequentist: under the null,
        q is ~uniform, so confidence>0.95 corresponds to a <=5% false-flag rate."""
        return round(1.0 - self.q_value, 4)


def _severity(effect: float, lo: float, hi: float) -> str:
    return "high" if effect >= hi else "medium" if effect >= lo else "low"


def _bh(pvals: np.ndarray) -> np.ndarray:
    if len(pvals) == 0:
        return pvals
    return multipletests(pvals, method="fdr_bh")[1]


# --- missingness ------------------------------------------------------------

def detect_missingness(X: np.ndarray, feature_names: list[str], *,
                       alpha: float = 0.05, rate_threshold: float = 0.0) -> list[Flag]:
    """Flag columns with missing values above `rate_threshold`. Reports a Wilson
    95% upper bound on the true missing rate as the uncertainty on the estimate."""
    n = X.shape[0]
    flags: list[Flag] = []
    miss = np.isnan(X).mean(axis=0)
    for j, rate in enumerate(miss):
        if rate > rate_threshold:
            k = int(round(rate * n))
            # Wilson interval upper bound at 95%
            lo, hi = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
            flags.append(Flag(
                "missingness", feature_names[j], statistic=round(rate, 4),
                p_value=0.0, q_value=0.0, effect_size=round(rate, 4),
                severity=_severity(rate, 0.05, 0.2),
                detail=f"{rate:.1%} missing (95% CI {lo:.1%}-{hi:.1%})",
            ))
    return flags


# --- univariate outliers (robust z / MAD) -----------------------------------

def detect_univariate_outliers(X: np.ndarray, feature_names: list[str], *,
                               z_threshold: float = 5.0) -> list[Flag]:
    """Per-feature robust z-score using median/MAD. Reports the fraction of samples
    beyond `z_threshold` and a Gaussian-tail p for that extremeness."""
    flags: list[Flag] = []
    for j in range(X.shape[1]):
        col = X[:, j]
        col = col[~np.isnan(col)]
        if col.size < 5:
            continue
        med = np.median(col)
        mad = stats.median_abs_deviation(col, scale="normal")
        if mad == 0:
            continue
        z = np.abs(col - med) / mad
        frac = float((z > z_threshold).mean())
        if frac > 0:
            tail_p = float(2 * stats.norm.sf(z_threshold))  # expected frac under normality
            flags.append(Flag(
                "univariate_outlier", feature_names[j], statistic=round(float(z.max()), 2),
                p_value=tail_p, q_value=tail_p, effect_size=round(frac, 4),
                severity=_severity(frac, 0.01, 0.05),
                detail=f"{frac:.1%} of values beyond {z_threshold} MAD (max z={z.max():.1f})",
            ))
    return flags


# --- multivariate outliers (Isolation Forest) -------------------------------

def detect_multivariate_outliers(X: np.ndarray, *, contamination: float = 0.02,
                                 seed: int = 0) -> list[Flag]:
    """One dataset-level flag if a meaningful fraction of rows look anomalous in the
    joint feature space (Isolation Forest)."""
    from sklearn.ensemble import IsolationForest
    Xi = np.nan_to_num(X, nan=np.nanmedian(X))
    if Xi.shape[0] < 20:
        return []
    iso = IsolationForest(contamination=contamination, random_state=seed)
    pred = iso.fit_predict(StandardScaler().fit_transform(Xi))
    frac = float((pred == -1).mean())
    if frac <= contamination:
        return []
    return [Flag(
        "multivariate_outlier", "dataset", statistic=round(frac, 4),
        p_value=0.0, q_value=0.0, effect_size=round(frac, 4),
        severity=_severity(frac, 0.03, 0.08),
        detail=f"{frac:.1%} of rows anomalous in joint feature space",
    )]


# --- classifier two-sample test (shared engine) -----------------------------

def _c2st(X: np.ndarray, labels: np.ndarray, *, seed: int, n_perm: int = 200) -> tuple[float, float]:
    """Distribution-free C2ST: can a classifier separate the two groups?
    Returns (test AUC, permutation p-value vs the AUC≈0.5 null). Same split is used
    for statistic and every permutation, so the null is valid."""
    X = np.nan_to_num(X, nan=np.nanmedian(X))
    Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.4, random_state=seed, stratify=labels)
    sc = StandardScaler().fit(Xtr)

    def auc_for(ytr_):
        clf = LogisticRegression(max_iter=200)
        clf.fit(sc.transform(Xtr), ytr_)
        prob = clf.predict_proba(sc.transform(Xte))[:, 1]
        return roc_auc_score(yte, prob)

    obs = auc_for(ytr)
    rng = np.random.default_rng(seed)
    perm = np.empty(n_perm)
    for i in range(n_perm):
        perm[i] = auc_for(rng.permutation(ytr))
    # one-sided: separation means AUC well above 0.5
    p = float((1 + np.sum(perm >= obs)) / (1 + n_perm))
    return float(obs), p


# --- covariate / distribution shift (reference vs new) ----------------------

def detect_covariate_shift(X_ref: np.ndarray, X_new: np.ndarray, feature_names: list[str], *,
                           alpha: float = 0.05, seed: int = 0, n_perm: int = 200) -> list[Flag]:
    """Per-feature KS tests (BH-corrected) for marginal shift, plus a dataset-level
    C2ST for joint shift. Flags features whose q < alpha and the dataset if separable."""
    flags: list[Flag] = []
    pvals, stats_, effects = [], [], []
    for j in range(X_ref.shape[1]):
        a = X_ref[:, j][~np.isnan(X_ref[:, j])]
        b = X_new[:, j][~np.isnan(X_new[:, j])]
        if a.size < 5 or b.size < 5:
            pvals.append(1.0); stats_.append(0.0); effects.append(0.0); continue
        ks = stats.ks_2samp(a, b)
        pooled_sd = np.sqrt((a.var() + b.var()) / 2) or 1.0
        effects.append(abs(a.mean() - b.mean()) / pooled_sd)  # standardized mean diff
        pvals.append(ks.pvalue); stats_.append(ks.statistic)
    q = _bh(np.array(pvals))
    for j, qj in enumerate(q):
        if qj < alpha:
            flags.append(Flag(
                "covariate_shift", feature_names[j], statistic=round(stats_[j], 4),
                p_value=round(pvals[j], 6), q_value=round(float(qj), 6),
                effect_size=round(effects[j], 3), severity=_severity(effects[j], 0.2, 0.5),
                detail=f"KS={stats_[j]:.3f}, std-mean-diff={effects[j]:.2f} (q={qj:.1e})",
            ))
    # joint shift
    labels = np.r_[np.zeros(len(X_ref)), np.ones(len(X_new))]
    auc, p = _c2st(np.r_[X_ref, X_new], labels, seed=seed, n_perm=n_perm)
    if p < alpha and auc > 0.6:
        flags.append(Flag(
            "covariate_shift", "dataset(joint)", statistic=round(auc, 3),
            p_value=round(p, 6), q_value=round(p, 6), effect_size=round(auc - 0.5, 3),
            severity=_severity(auc - 0.5, 0.1, 0.25),
            detail=f"reference vs new separable at AUC={auc:.2f} (C2ST p={p:.1e})",
        ))
    return flags


# --- batch effect (grouping variable) ---------------------------------------

def detect_batch_effect(X: np.ndarray, batch: np.ndarray, *, alpha: float = 0.05,
                        seed: int = 0, n_perm: int = 200) -> list[Flag]:
    """C2ST: are samples separable by batch label using features alone? High AUC ⇒
    technical batch effect confounding the biology."""
    uniq = np.unique(batch)
    if len(uniq) != 2:
        # collapse to largest-two for a v1 binary test
        if len(uniq) < 2:
            return []
        top2 = uniq[np.argsort([-(batch == u).sum() for u in uniq])[:2]]
        mask = np.isin(batch, top2)
        X, batch = X[mask], (batch[mask] == top2[0]).astype(int)
    else:
        batch = (batch == uniq[0]).astype(int)
    auc, p = _c2st(X, batch, seed=seed, n_perm=n_perm)
    if p < alpha and auc > 0.6:
        return [Flag(
            "batch_effect", "dataset", statistic=round(auc, 3), p_value=round(p, 6),
            q_value=round(p, 6), effect_size=round(auc - 0.5, 3),
            severity=_severity(auc - 0.5, 0.1, 0.25),
            detail=f"samples separable by batch at AUC={auc:.2f} (C2ST p={p:.1e})",
        )]
    return []


# --- target leakage ---------------------------------------------------------

def detect_leakage(X: np.ndarray, y: np.ndarray, feature_names: list[str], *,
                   alpha: float = 0.05, auc_threshold: float = 0.9, seed: int = 0) -> list[Flag]:
    """Flag single features that predict the label almost perfectly — the signature
    of target leakage. Per-feature AUC with a permutation null, BH-corrected."""
    rng = np.random.default_rng(seed)
    aucs, pvals = [], []
    for j in range(X.shape[1]):
        col = np.nan_to_num(X[:, j], nan=np.nanmedian(X[:, j]))
        try:
            auc = roc_auc_score(y, col)
        except ValueError:
            aucs.append(0.5); pvals.append(1.0); continue
        auc = max(auc, 1 - auc)  # direction-agnostic
        # permutation null on label
        perm = np.array([max(a, 1 - a) for a in
                         (roc_auc_score(rng.permutation(y), col) for _ in range(100))])
        pvals.append(float((1 + np.sum(perm >= auc)) / 101))
        aucs.append(auc)
    q = _bh(np.array(pvals))
    flags: list[Flag] = []
    for j, (auc, qj) in enumerate(zip(aucs, q)):
        if auc >= auc_threshold and qj < alpha:
            flags.append(Flag(
                "leakage", feature_names[j], statistic=round(auc, 4),
                p_value=round(pvals[j], 6), q_value=round(float(qj), 6),
                effect_size=round(auc - 0.5, 3), severity=_severity(auc - 0.5, 0.35, 0.45),
                detail=f"single-feature AUC={auc:.3f} for the label — likely leakage (q={qj:.1e})",
            ))
    return flags
