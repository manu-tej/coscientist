"""Tests for the gold-set validation harness — all offline via injected scorers."""
from __future__ import annotations

import math

from company import validation


def test_gold_pairs_full_crossproduct_and_labels():
    pairs = validation.gold_pairs()
    assert len(pairs) == len(validation.GOLD) * len(validation.all_drugs())
    n_gold = sum(1 for *_, g in pairs if g)
    assert n_gold == sum(len(v) for v in validation.GOLD.values())   # sildenafil counts in 2 diseases


def test_perfect_scorer_separates_cleanly():
    def scorer(drug, disease):
        return 0.95 if drug in validation.GOLD[disease] else 0.10
    rep = validation.validate(scorer=scorer)
    assert rep.n_scored == rep.n_pairs                  # never abstains
    assert rep.auroc == 1.0
    assert rep.recall_at_1 == 1.0 and rep.recall_at_3 == 1.0
    assert rep.mean_gold > rep.mean_decoy
    assert rep.negative_control_max == 0.10


def test_inverted_scorer_is_anticoncordant():
    # a scorer that ranks decoys ABOVE truth → AUROC 0, recall 0 (the failure signature)
    def scorer(drug, disease):
        return 0.1 if drug in validation.GOLD[disease] else 0.9
    rep = validation.validate(scorer=scorer)
    assert rep.auroc == 0.0
    assert rep.recall_at_1 == 0.0


def test_abstentions_are_excluded():
    def scorer(drug, disease):
        if drug == "sildenafil":
            return None                                 # abstain on every sildenafil pair
        return 0.9 if drug in validation.GOLD[disease] else 0.2
    rep = validation.validate(scorer=scorer)
    assert rep.n_pairs - rep.n_scored == len(validation.GOLD)   # one sildenafil pair per disease


def test_literature_path_with_stubbed_network():
    # fetch sees the esearch term; return a high count only for true gold pairs
    def fetch(term: str) -> int:
        for disease, drugs in validation.GOLD.items():
            if disease in term and any(f'"{d}"' in term for d in drugs):
                return 1500
        return 4
    rep = validation.validate(fetch=fetch)
    assert rep.auroc == 1.0
    assert rep.mean_gold > rep.mean_decoy
    assert not math.isnan(rep.recall_at_1)
