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
    assert all(g.gold_answer for g in goals)
    assert all(g.choices for g in goals)
