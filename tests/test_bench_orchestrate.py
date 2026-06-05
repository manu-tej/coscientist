import json
import pytest
from pathlib import Path

from bench.orchestrate import concordance_from_runs
from bench.runner import BenchHypothesis, BenchRun


def _run(goal_id, hyps):
    return BenchRun(goal_id=goal_id, variant="full", hypotheses=hyps,
                    n_llm_calls=10, wall_clock_s=1.0, db_path=":memory:")


def test_concordance_from_runs_scores_against_gold(tmp_path):
    from bench.goalset import BenchGoal
    goals = [BenchGoal(id="q1", goal="...", gold_answer="B"),
             BenchGoal(id="q2", goal="...", gold_answer="C")]
    runs = [
        _run("q1", [BenchHypothesis("h1", "Answer: B", "s", 1300.0, "t0"),
                    BenchHypothesis("h2", "Answer: A", "s", 1100.0, "t0")]),
        _run("q2", [BenchHypothesis("h3", "Answer: C", "s", 1280.0, "t0")]),
    ]
    ref = {"q1": 0.25, "q2": 0.25}
    stats = concordance_from_runs(goals, runs, ref, bin_width=50, min_support=1)
    assert stats["n_rows"] == 3
    assert "spearman_rho" in stats
    assert "blue_minus_red" in stats


def test_rediscovery_from_runs_recall_and_elo_concordance():
    from bench.orchestrate import rediscovery_from_runs
    from bench.goalset import BenchGoal
    goals = [BenchGoal(id="t1", goal="...", gold_entities=["SPP1", "FAP", "LAYN"])]
    # Elo ranks track gold-entity content: high holds 2, mid holds 1, low holds 0.
    runs = [_run("t1", [
        BenchHypothesis("h1", "SPP1+ macrophages and FAP+ fibroblasts dominate", "s", 1300.0, "t0"),
        BenchHypothesis("h2", "LAYN marks exhausted T cells", "s", 1200.0, "t0"),
        BenchHypothesis("h3", "something unrelated", "s", 1100.0, "t0"),
    ])]
    r = rediscovery_from_runs(goals, runs)
    assert r["per_goal"][0]["pool_recall"] == 1.0        # SPP1, FAP, LAYN all surfaced
    assert r["per_goal"][0]["top1_recall"] == 2 / 3      # top hyp holds SPP1 + FAP
    assert r["n_hyps_scored"] == 3
    # higher-Elo hypotheses hold more gold entities → positive Elo↔recall
    assert r["elo_recall_spearman"] > 0


class _Args:
    def __init__(self, **kw):
        self.limit = 5
        self.full = False
        self.backend = "api"
        self.yes = True
        self.command = "concordance"
        for k, v in kw.items():
            setattr(self, k, v)


@pytest.mark.asyncio
async def test_run_command_proceed_unwired_command_returns_zero(capsys, monkeypatch):
    import bench.orchestrate as orch
    # A not-yet-wired command proceeds past the cost gate and returns 0 without
    # invoking the model stack (only 'concordance' runs end-to-end in v1).
    rc = await orch.run_command(_Args(yes=True, command="baseline"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "not yet wired" in out.lower()


@pytest.mark.asyncio
async def test_run_command_decline_returns_one(monkeypatch):
    import bench.orchestrate as orch
    import bench.cli as cli
    monkeypatch.setattr(cli, "_confirm", lambda est, backend, assume_yes: False)
    rc = await orch.run_command(_Args(yes=False))
    assert rc == 1
