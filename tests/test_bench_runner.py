import pytest
from pathlib import Path
from bench.runner import BenchHypothesis, BenchRun, trajectory_from_matches, read_run
from core.state import StateStore
from core.models import Hypothesis, TournamentMatch


@pytest.mark.asyncio
async def test_trajectory_from_matches_orders_and_filters():
    rows = [
        ("t1", "a", "b", 1216.0, 1184.0),
        ("t2", "a", "c", 1200.0, 1216.0),
    ]
    traj = trajectory_from_matches("a", rows)
    assert traj == [("t1", 1216.0), ("t2", 1200.0)]
    traj_b = trajectory_from_matches("b", rows)
    assert traj_b == [("t1", 1184.0)]


@pytest.mark.asyncio
async def test_read_run_builds_bench_hypotheses(tmp_path: Path):
    db = tmp_path / "run.db"
    store = StateStore(str(db))
    await store.init_db()
    from core.models import ResearchPlanConfig
    cfg = ResearchPlanConfig(run_id="r", goal="g", preferences="p",
                             attributes=["Novelty"], constraints="c", safety_approved=True)
    await store.save_config(cfg)
    await store.save_hypothesis(Hypothesis(id="a", run_id="r", text="Answer: B",
                                summary="s-a", generation_method="debate", source="system",
                                elo_rating=1216.0))
    await store.save_hypothesis(Hypothesis(id="b", run_id="r", text="Answer: C",
                                summary="s-b", generation_method="debate", source="system",
                                elo_rating=1184.0))
    await store.save_match_and_elos(TournamentMatch(
        id="m1", run_id="r", h1_id="a", h2_id="b", winner_id="a",
        match_type="single_turn", elo_before_h1=1200.0, elo_before_h2=1200.0,
        elo_after_h1=1216.0, elo_after_h2=1184.0))

    run = await read_run(str(db), run_id="r", goal_id="g1", variant="full",
                         n_llm_calls=10, wall_clock_s=2.0)
    assert isinstance(run, BenchRun)
    assert run.variant == "full"
    ids = {h.id for h in run.hypotheses}
    assert ids == {"a", "b"}
    ha = next(h for h in run.hypotheses if h.id == "a")
    assert ha.elo_rating == 1216.0
    assert ha.text == "Answer: B"
    assert len(ha.elo_trajectory) == 1
