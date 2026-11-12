"""Tests for the literature co-occurrence method — normalization + offline safety.

All offline: the `fetch` seam is stubbed, so no network is touched.
"""
from __future__ import annotations

import math

from company import literature

PAH = "pulmonary arterial hypertension"


def test_score_saturates_and_floors():
    assert literature.literature_score("x", PAH, fetch=lambda t: 1000) == 1.0   # cap
    assert literature.literature_score("x", PAH, fetch=lambda t: 0) == 0.0       # floor
    mid = literature.literature_score("x", PAH, fetch=lambda t: 9)
    assert mid == round(math.log1p(9) / math.log1p(1000), 4)


def test_more_papers_scores_higher():
    lo = literature.literature_score("x", PAH, fetch=lambda t: 5)
    hi = literature.literature_score("x", PAH, fetch=lambda t: 500)
    assert hi > lo


def test_query_pairs_drug_and_disease():
    seen = {}
    def fetch(term):
        seen["term"] = term
        return 42
    literature.literature_score("sildenafil", PAH, fetch=fetch)
    assert "sildenafil" in seen["term"] and PAH in seen["term"]


def test_abstains_on_fetch_failure():
    def boom(term):
        raise RuntimeError("network down")
    assert literature.cooccurrence_count("x", PAH, fetch=boom) is None
    assert literature.literature_score("x", PAH, fetch=boom) is None
