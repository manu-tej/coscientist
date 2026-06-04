import pytest
from bench.runner import BenchHypothesis
from bench.cross_tournament import run_cross_tournament, variant_elo_summary


def _bh(hid, elo=1200.0):
    return BenchHypothesis(id=hid, text=f"text-{hid}", summary=hid, elo_rating=elo,
                           created_at="t0")


@pytest.mark.asyncio
async def test_cross_tournament_deterministic_verdicts():
    pools = {"full": [_bh("F1"), _bh("F2")], "single_shot": [_bh("S1")]}

    async def verdict(a, b):
        rank = {"F1": 3, "F2": 2, "S1": 1}
        return a.id if rank[a.id] >= rank[b.id] else b.id

    elos = await run_cross_tournament(pools, verdict, n_rounds=4, seed=1)
    assert elos["F1"] > elos["S1"]
    summary = variant_elo_summary(pools, elos)
    assert summary["full"]["mean"] > summary["single_shot"]["mean"]
    assert "best" in summary["full"] and "median" in summary["full"]


@pytest.mark.asyncio
async def test_cross_tournament_position_swap_tie_no_update():
    pools = {"a": [_bh("X")], "b": [_bh("Y")]}
    calls = {"n": 0}

    async def flaky_verdict(a, b):
        calls["n"] += 1
        return a.id if calls["n"] % 2 == 1 else b.id

    elos = await run_cross_tournament(pools, flaky_verdict, n_rounds=1, seed=0,
                                      position_swap=True)
    assert elos["X"] == elos["Y"] == 1200.0
