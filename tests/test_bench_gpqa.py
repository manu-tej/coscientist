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
    assert "(B) Helicase" in GPQA_GOAL_TEMPLATE.format(
        domain=g.domain, question=g.goal_question, options=g.options_block
    )


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


def test_parse_mcq_answer_ignores_article_a_and_abbreviations():
    # Regression: the article "a" after "answer is" was read as option A, and
    # "e.g." after it as option E, so a stated choice of B or D scored as A/E.
    assert parse_mcq_answer(
        "The correct answer is a matter of timing; the answer is Option (B)."
    ) == "B"
    assert parse_mcq_answer(
        "This answer is a hypothesis about kinase signalling. "
        "The answer is Option (B)."
    ) == "B"
    assert parse_mcq_answer("Answer: C. The answer is a subtle one.") == "C"
    assert parse_mcq_answer("The answer is e.g. linked to X; option (D).") == "D"
    assert parse_mcq_answer("The answer is a subtle one.") is None


def test_parse_mcq_answer_still_reads_real_choices_of_a():
    assert parse_mcq_answer("The answer is A because the enzyme ...") == "A"
    assert parse_mcq_answer("the answer is (a) because the enzyme ...") == "A"
    assert parse_mcq_answer("the answer is a) helicase") == "A"
    assert parse_mcq_answer("final answer: a") == "A"
    assert parse_mcq_answer("final answer: a.") == "A"
    assert parse_mcq_answer("a\nBecause the enzyme ...") == "A"


def test_parse_mcq_answer_reads_lowercase_choices_followed_by_a_word():
    # Only the article "a" is prose; a lowercase b, c or d after "answer is"
    # followed by a word is still a choice.
    assert parse_mcq_answer("the answer is d because the enzyme ...") == "D"
    assert parse_mcq_answer("final answer: b since helicase unwinds DNA") == "B"
    assert parse_mcq_answer("the correct answer is c because the enzyme ...") == "C"
    assert parse_mcq_answer(
        "Answer: b since helicase unwinds DNA; earlier I considered option (A)"
    ) == "B"
