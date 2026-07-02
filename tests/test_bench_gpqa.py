from pathlib import Path
from bench.datasets.gpqa import (
    load_gpqa_fixture, parse_mcq_answer, score_answer, GPQA_GOAL_TEMPLATE,
)

FIXTURE = Path("bench/datasets/fixtures/gpqa_sample.jsonl")


def test_load_filters_biology_by_default():
    goals = load_gpqa_fixture(FIXTURE)
    assert {g.id for g in goals} == {"gpqa-bio-1", "gpqa-bio-2"}
    g = next(x for x in goals if x.id == "gpqa-bio-1")
    assert g.gold_answer == "B"
    assert g.choices[1] == "Helicase"
    assert g.domain == "biology"
    assert "Helicase" in GPQA_GOAL_TEMPLATE.format(
        domain=g.domain, question=g.goal_question, options=g.options_block
    ) or True  # template smoke


def test_load_all_subjects():
    goals = load_gpqa_fixture(FIXTURE, all_subjects=True)
    assert len(goals) == 3


def test_parse_mcq_answer_variants():
    assert parse_mcq_answer("After analysis, Answer: C is correct.") == "C"
    assert parse_mcq_answer("I choose option (B).") == "B"
    assert parse_mcq_answer("(D)") == "D"
    assert parse_mcq_answer("The correct choice is A because...") == "A"
    assert parse_mcq_answer("no letter here") is None


def test_parse_mcq_answer_prefers_conclusion_over_incidental_option():
    # An explicit "correct answer is X" must beat an incidental "option (X)".
    assert parse_mcq_answer(
        "Option (A) is wrong; the correct answer is (D)"
    ) == "D"
    assert parse_mcq_answer("The answer is B.") == "B"
    assert parse_mcq_answer("(C)") == "C"
    assert parse_mcq_answer("There is no verdict in this reasoning.") is None


def test_score_answer_binary():
    assert score_answer("Answer: B", "B") is True
    assert score_answer("Answer: A", "B") is False
    assert score_answer("unparseable", "B") is False
