import pytest
import numpy as np
from core.tournament import compute_elo_update, select_match_pairs
from core.models import Hypothesis


def make_h(id_, elo):
    return Hypothesis(
        id=id_, run_id="r1", text="t", summary="s",
        generation_method="debate", source="system", elo_rating=elo,
    )


def test_elo_update_winner_gains():
    r_a, r_b = compute_elo_update(1200.0, 1200.0, winner="a", k=32.0)
    assert r_a > 1200.0
    assert r_b < 1200.0


def test_elo_update_symmetric():
    r_a1, r_b1 = compute_elo_update(1200.0, 1200.0, winner="a", k=32.0)
    r_a2, r_b2 = compute_elo_update(1200.0, 1200.0, winner="a", k=32.0)
    assert abs(r_a1 - r_a2) < 0.01
    assert abs(r_b1 - r_b2) < 0.01


def test_elo_upset_bigger_swing():
    # Low-rated beats high-rated → bigger Elo gain than equal-rated match
    r_low, r_high = compute_elo_update(1000.0, 1500.0, winner="a", k=32.0)
    r_even_w, _ = compute_elo_update(1200.0, 1200.0, winner="a", k=32.0)
    assert r_low - 1000.0 > r_even_w - 1200.0


def test_select_pairs_prefers_similar():
    hypotheses = [make_h(f"h{i}", 1200.0) for i in range(4)]
    similarity_pairs = [("h0", "h1", 0.9), ("h0", "h2", 0.3), ("h1", "h3", 0.8)]
    pairs = select_match_pairs(hypotheses, similarity_pairs, n_pairs=2)
    pair_ids = [(a.id, b.id) for a, b in pairs]
    # h0-h1 (0.9) should be selected first, then h1-h3 (0.8)
    assert ("h0", "h1") in pair_ids or ("h1", "h0") in pair_ids


def test_select_pairs_falls_back_to_round_robin():
    hypotheses = [make_h(f"h{i}", 1200.0) for i in range(3)]
    pairs = select_match_pairs(hypotheses, [], n_pairs=2)
    assert len(pairs) == 2
