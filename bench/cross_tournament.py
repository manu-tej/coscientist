from __future__ import annotations

import random
import statistics
from typing import Awaitable, Callable

from bench.runner import BenchHypothesis
from core.tournament import compute_elo_update

_INITIAL_ELO = 1200.0

VerdictFn = Callable[[BenchHypothesis, BenchHypothesis], Awaitable[str]]


async def _adjudicate(
    a: BenchHypothesis, b: BenchHypothesis, verdict: VerdictFn, position_swap: bool
) -> str | None:
    """Return winner id, or None for a tie. With position_swap, verdict is called
    twice; both results must agree on the same winner (§7). A disagreement is
    treated as a tie — no Elo change."""
    w1 = await verdict(a, b)
    if not position_swap:
        return w1
    w2 = await verdict(a, b)
    return w1 if w1 == w2 else None


async def run_cross_tournament(
    pools: dict[str, list[BenchHypothesis]],
    verdict: VerdictFn,
    n_rounds: int = 4,
    seed: int = 0,
    position_swap: bool = False,
) -> dict[str, float]:
    """Pool all variants' hypotheses into one shared Elo tournament.
    Returns {hypothesis_id: final_elo} on a common scale."""
    everyone: list[BenchHypothesis] = [h for hs in pools.values() for h in hs]
    elo: dict[str, float] = {h.id: _INITIAL_ELO for h in everyone}
    by_id = {h.id: h for h in everyone}
    if len(everyone) < 2:
        return elo

    rng = random.Random(seed)
    ids = list(by_id.keys())
    for _ in range(n_rounds):
        rng.shuffle(ids)
        for i in range(0, len(ids) - 1, 2):
            a, b = by_id[ids[i]], by_id[ids[i + 1]]
            winner_id = await _adjudicate(a, b, verdict, position_swap)
            if winner_id is None:
                continue  # tie → no Elo change
            winner = "a" if winner_id == a.id else "b"
            new_a, new_b = compute_elo_update(elo[a.id], elo[b.id], winner)
            elo[a.id], elo[b.id] = new_a, new_b
    return elo


def variant_elo_summary(
    pools: dict[str, list[BenchHypothesis]], elo: dict[str, float]
) -> dict[str, dict]:
    """Per-variant Elo distribution (mean/median/best) on the common scale."""
    out: dict[str, dict] = {}
    for variant, hs in pools.items():
        vals = [elo[h.id] for h in hs if h.id in elo]
        if not vals:
            out[variant] = {"mean": float("nan"), "median": float("nan"),
                            "best": float("nan"), "n": 0}
            continue
        out[variant] = {"mean": statistics.mean(vals),
                        "median": statistics.median(vals),
                        "best": max(vals), "n": len(vals)}
    return out
