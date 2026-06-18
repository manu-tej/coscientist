"""Tests for the QC detectors: real power on injected corruptions, calibrated
nulls, and seed-determinism. Small controlled data keeps these fast.
"""
from __future__ import annotations

import numpy as np

from qc import detectors as D


def _blob(n, d, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    return rng.normal(shift, 1.0, size=(n, d))


# --- power: injected artifacts are caught -----------------------------------

def test_missingness_detected_and_clean_is_silent():
    X = _blob(100, 4, 0)
    names = [f"f{i}" for i in range(4)]
    assert D.detect_missingness(X, names) == []          # no missing → no flags
    X[::5, 1] = np.nan
    flags = D.detect_missingness(X, names)
    assert any(f.target == "f1" for f in flags)


def test_univariate_outliers_detected():
    X = _blob(200, 3, 1)
    names = [f"f{i}" for i in range(3)]
    X[:3, 0] = 50.0                                       # blatant outliers
    flags = D.detect_univariate_outliers(X, names)
    assert any(f.target == "f0" for f in flags)


def test_covariate_shift_detected_on_real_shift():
    names = [f"f{i}" for i in range(5)]
    ref = _blob(120, 5, 2)
    new = _blob(120, 5, 3, shift=1.5)                     # strong shift
    flags = D.detect_covariate_shift(ref, new, names, seed=2)
    assert any(f.check == "covariate_shift" for f in flags)


def test_covariate_shift_silent_on_null():
    names = [f"f{i}" for i in range(5)]
    a = _blob(120, 5, 4)
    b = _blob(120, 5, 5)                                  # same distribution
    flags = D.detect_covariate_shift(a, b, names, seed=4)
    assert not any(f.check == "covariate_shift" for f in flags)


def test_batch_effect_detected():
    rng = np.random.default_rng(6)
    X = rng.normal(0, 1, size=(160, 5))
    batch = (rng.random(160) < 0.5).astype(int)
    X[batch == 1, :2] += 1.5                              # technical shift on batch B
    flags = D.detect_batch_effect(X, batch, seed=6)
    assert any(f.check == "batch_effect" for f in flags)


def test_batch_effect_silent_when_label_random():
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, size=(160, 5))
    batch = (rng.random(160) < 0.5).astype(int)           # uncorrelated with X
    flags = D.detect_batch_effect(X, batch, seed=7)
    assert not any(f.check == "batch_effect" for f in flags)


def test_leakage_detected():
    rng = np.random.default_rng(8)
    X = rng.normal(0, 1, size=(150, 4))
    y = (rng.random(150) < 0.5).astype(int)
    leaked = y + rng.normal(0, 0.01, size=150)            # near-deterministic leak
    Xl = np.c_[X, leaked]
    flags = D.detect_leakage(Xl, y, [f"f{i}" for i in range(4)] + ["leak"], seed=8)
    assert any(f.target == "leak" for f in flags)


def test_leakage_silent_on_moderate_features():
    rng = np.random.default_rng(9)
    X = rng.normal(0, 1, size=(150, 4))
    y = (rng.random(150) < 0.5).astype(int)               # features independent of y
    flags = D.detect_leakage(X, y, [f"f{i}" for i in range(4)], seed=9)
    assert flags == []


# --- confidence + determinism -----------------------------------------------

def test_flag_confidence_tracks_qvalue():
    names = [f"f{i}" for i in range(5)]
    flags = D.detect_covariate_shift(_blob(120, 5, 2), _blob(120, 5, 3, shift=1.5), names, seed=2)
    for f in flags:
        assert abs(f.confidence - round(1 - f.q_value, 4)) < 1e-9   # confidence is 4dp-rounded
        assert 0.0 <= f.confidence <= 1.0


def test_detectors_are_seed_deterministic():
    names = [f"f{i}" for i in range(5)]
    ref, new = _blob(120, 5, 2), _blob(120, 5, 3, shift=1.2)
    f1 = D.detect_covariate_shift(ref, new, names, seed=11)
    f2 = D.detect_covariate_shift(ref, new, names, seed=11)
    assert [(f.target, f.statistic, f.q_value) for f in f1] == \
           [(f.target, f.statistic, f.q_value) for f in f2]
