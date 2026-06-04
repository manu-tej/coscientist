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
