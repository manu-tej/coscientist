import os
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("BENCH_INTEGRATION") != "1",
    reason="set BENCH_INTEGRATION=1 to exercise the live HuggingFace pull",
)


def test_gpqa_hf_pull_returns_biology_goals():
    from bench.datasets.gpqa import load_gpqa_hf
    goals = load_gpqa_hf(limit=5)
    assert len(goals) >= 1
    # This mirror has (problem, solution, domain): gold_answer is parsed from the
    # \boxed{X} solution; options are inline in the problem (no separate choices).
    assert all(g.gold_answer in {"A", "B", "C", "D"} for g in goals)
    assert all(g.domain == "biology" for g in goals)
    assert all("Question:" in g.goal for g in goals)
