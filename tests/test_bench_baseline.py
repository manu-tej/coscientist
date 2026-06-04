import pytest
from bench.baseline import single_shot, best_of_n, SINGLE_SHOT_PROMPT


@pytest.mark.asyncio
async def test_single_shot_builds_one_hypothesis():
    async def gen(prompt): return "Answer: B. Because helicase unwinds DNA."
    from bench.goalset import BenchGoal
    goal = BenchGoal(id="g1", goal="Which enzyme unwinds DNA?")
    hyps = await single_shot(goal, gen)
    assert len(hyps) == 1
    assert hyps[0].text.startswith("Answer: B")
    assert hyps[0].id.startswith("single_shot-g1")


@pytest.mark.asyncio
async def test_best_of_n_picks_highest_judge_score():
    from bench.goalset import BenchGoal
    goal = BenchGoal(id="g1", goal="propose a hypothesis")
    outputs = iter([f"hypothesis {i}" for i in range(4)])

    async def gen(prompt): return next(outputs)

    async def judge_score(text):  # higher index scores higher
        return float(text.split()[-1])

    best = await best_of_n(goal, gen, judge_score, n=4)
    assert len(best) == 1
    assert best[0].text == "hypothesis 3"


def test_single_shot_prompt_mentions_goal_placeholder():
    assert "{goal}" in SINGLE_SHOT_PROMPT
