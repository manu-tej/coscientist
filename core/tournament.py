from __future__ import annotations
from core.models import Hypothesis


def compute_elo_update(
    rating_a: float, rating_b: float, winner: str, k: float = 32.0
) -> tuple[float, float]:
    expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    expected_b = 1.0 - expected_a
    score_a = 1.0 if winner == "a" else 0.0
    score_b = 1.0 - score_a
    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * (score_b - expected_b)
    return new_a, new_b


def select_match_pairs(
    hypotheses: list[Hypothesis],
    similarity_pairs: list[tuple[str, str, float]],
    n_pairs: int,
    multi_turn_threshold: float = 1350.0,
) -> list[tuple[Hypothesis, Hypothesis]]:
    h_map = {h.id: h for h in hypotheses}
    # Sort by similarity descending — prefer similar hypotheses
    sorted_pairs = sorted(similarity_pairs, key=lambda x: x[2], reverse=True)
    selected: list[tuple[Hypothesis, Hypothesis]] = []
    used: set[str] = set()
    for h1_id, h2_id, _ in sorted_pairs:
        if len(selected) >= n_pairs:
            break
        if h1_id in used or h2_id in used:
            continue
        if h1_id not in h_map or h2_id not in h_map:
            continue
        selected.append((h_map[h1_id], h_map[h2_id]))
        used.add(h1_id)
        used.add(h2_id)
    # Fill remaining with round-robin if needed (allow reuse when pool is small)
    if len(selected) < n_pairs:
        for i, h1 in enumerate(hypotheses):
            if len(selected) >= n_pairs:
                break
            for h2 in hypotheses[i + 1:]:
                if len(selected) >= n_pairs:
                    break
                # Skip pairs already covered by similarity-based selection
                already = any(
                    (a.id == h1.id and b.id == h2.id) or
                    (a.id == h2.id and b.id == h1.id)
                    for a, b in selected
                )
                if not already:
                    selected.append((h1, h2))
    return selected
