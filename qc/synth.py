"""Real base data + controlled corruption injectors for the falsifiable harness.

Base data is the Wisconsin breast-cancer set (real, biomedical, ships with sklearn,
no download). Each injector returns a corrupted copy plus the name of the check that
*should* fire — the ground truth the harness scores detection against.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification


def load_base() -> tuple[np.ndarray, np.ndarray, list[str]]:
    d = load_breast_cancer()
    return d.data.astype(float), d.target.astype(int), list(d.feature_names)


def make_moderate(seed: int, n: int = 400, n_features: int = 20) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Synthetic set whose single features are only *moderately* predictive (no real
    biomarker near AUC 1). Used as the leakage clean-baseline so a genuinely strong
    biomarker isn't mistaken for leakage."""
    X, y = make_classification(n_samples=n, n_features=n_features, n_informative=8,
                               n_redundant=4, class_sep=0.8, random_state=seed)
    return X, y, [f"f{i}" for i in range(n_features)]


def split_two(X: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two random halves of the same data — a genuine *null* for shift/batch tests."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    h = len(X) // 2
    return X[idx[:h]], X[idx[h:]]


# --- injectors (each returns the corrupted artifact + expected check) -------

def inject_batch_effect(X: np.ndarray, seed: int, *, strength: float = 1.0,
                        frac_features: float = 0.4):
    """Assign a batch label and shift a subset of features for batch B by `strength`
    standard deviations (a classic technical batch effect)."""
    rng = np.random.default_rng(seed)
    batch = (rng.random(len(X)) < 0.5).astype(int)
    Xc = X.copy()
    k = max(1, int(frac_features * X.shape[1]))
    feats = rng.choice(X.shape[1], size=k, replace=False)
    for j in feats:
        Xc[batch == 1, j] += strength * np.nanstd(X[:, j])
    return Xc, batch


def inject_covariate_shift(X_new: np.ndarray, seed: int, *, strength: float = 1.0,
                           frac_features: float = 0.4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Xc = X_new.copy()
    k = max(1, int(frac_features * X_new.shape[1]))
    for j in rng.choice(X_new.shape[1], size=k, replace=False):
        Xc[:, j] += strength * np.nanstd(X_new[:, j])
    return Xc


def inject_missingness(X: np.ndarray, seed: int, *, rate: float = 0.15):
    rng = np.random.default_rng(seed)
    Xc = X.copy()
    j = rng.integers(X.shape[1])
    mask = rng.random(len(X)) < rate
    Xc[mask, j] = np.nan
    return Xc


def inject_outliers(X: np.ndarray, seed: int, *, frac: float = 0.03, magnitude: float = 12.0):
    rng = np.random.default_rng(seed)
    Xc = X.copy()
    j = rng.integers(X.shape[1])
    rows = rng.choice(len(X), size=max(1, int(frac * len(X))), replace=False)
    Xc[rows, j] = np.nanmedian(X[:, j]) + magnitude * np.nanstd(X[:, j])
    return Xc


def inject_leakage(X: np.ndarray, y: np.ndarray, names: list[str], seed: int):
    """Append a feature that is the label plus a sliver of noise (near-perfect AUC)."""
    rng = np.random.default_rng(seed)
    leaked = y.astype(float) + rng.normal(0, 0.01, size=len(y))
    return np.c_[X, leaked], names + ["__leaked__"]
