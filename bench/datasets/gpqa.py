from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from bench.goalset import BenchGoal
from tools.resources import resource_path

GPQA_GOAL_TEMPLATE = (
    "Research goal: Determine the correct answer to the following "
    "graduate-level {domain} question and justify it rigorously.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Produce a hypothesis stating which option is correct and why."
)

_LETTERS = ["A", "B", "C", "D", "E", "F"]


def _options_block(choices: list[str]) -> str:
    return "\n".join(f"({_LETTERS[i]}) {c}" for i, c in enumerate(choices))


def _row_to_goal(row: dict) -> BenchGoal:
    question = row["question"]
    choices = list(row["choices"])
    options = _options_block(choices)
    goal_text = GPQA_GOAL_TEMPLATE.format(
        domain=row.get("subject", "biology").lower(),
        question=question,
        options=options,
    )
    g = BenchGoal(
        id=row["id"],
        goal=goal_text,
        domain=row.get("subject", "biology").lower(),
        gold_answer=str(row["answer"]).strip().upper(),
        choices=choices,
        metadata={"question": question, "options_block": options,
                  "subject": row.get("subject", "")},
    )
    g.goal_question = question          # type: ignore[attr-defined]
    g.options_block = options           # type: ignore[attr-defined]
    return g


def load_gpqa_fixture(path: str | Path, all_subjects: bool = False) -> list[BenchGoal]:
    """Load GPQA rows from a local jsonl fixture (offline)."""
    goals: list[BenchGoal] = []
    for line in resource_path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not all_subjects and row.get("subject", "").lower() != "biology":
            continue
        goals.append(_row_to_goal(row))
    return goals


def load_gpqa_hf(all_subjects: bool = False, limit: Optional[int] = None) -> list[BenchGoal]:
    """Load the live ungated mirror hendrydong/gpqa_diamond_mc. Network-gated;
    used by the opt-in integration test and real runs, not unit tests.

    This mirror's schema is (problem, solution, domain): `problem` is the full
    question with the (A)-(D) options inline, `solution` is `\\boxed{X}`, and
    `domain` is Physics/Chemistry/Biology. There is no separate choices list."""
    from datasets import load_dataset  # local import: heavy, network

    ds = load_dataset("hendrydong/gpqa_diamond_mc", split="test")
    goals: list[BenchGoal] = []
    for i, ex in enumerate(ds):
        domain = str(ex.get("domain", "")).strip()
        if not all_subjects and domain.lower() != "biology":
            continue
        m = re.search(r"boxed\{([A-Da-d])\}", str(ex.get("solution", "")))
        if not m:
            continue  # no gold letter → unusable for concordance scoring
        answer = m.group(1).upper()
        problem = str(ex.get("problem", ""))
        goal_text = (
            "Research goal: Determine the correct answer to the following "
            f"graduate-level {domain.lower()} question and justify it rigorously.\n\n"
            f"Question:\n{problem}\n\n"
            "Produce a hypothesis stating which option (A, B, C, or D) is correct "
            "and why. State your choice explicitly as 'Answer: <letter>'."
        )
        goals.append(BenchGoal(
            id=f"gpqa-{domain.lower()}-{i}",
            goal=goal_text,
            domain=domain.lower(),
            gold_answer=answer,
            choices=None,  # options are inline in `problem`; no separate list
            metadata={"problem": problem, "solution": str(ex.get("solution", ""))},
        ))
        if limit is not None and len(goals) >= limit:
            break
    return goals


# Ordered most-specific-first: an explicit "correct answer is X" must win over
# an incidental "option (X)" mention (models often eliminate options before
# concluding).
_ANSWER_PATTERNS = [
    r"correct (?:choice|option|answer)\s*(?:is|[:=\-])?\s*\(?([a-f])\)?\b",
    r"answer\s*(?:is|[:=\-])?\s*\(?([a-f])\)?\b",
    r"option\s*\(?([a-f])\)?\b",
    r"^\s*\(?([a-f])\)?\s*$",
]


# Two lowercase, unparenthesised matches are prose rather than a choice: the
# article "a" running straight into another word on the same line ("the answer
# is a matter of ...") and any letter running into ".x" ("e.g."). Without this
# check the article scored as option A. A lowercase b-f followed by a word ("the
# answer is d because ...") is still a choice.
_WORD_AFTER = re.compile(r"[ \t]+\w")
_ABBREVIATION_AFTER = re.compile(r"\.\w")


def _is_choice(text: str, m: re.Match) -> bool:
    start, end = m.span(1)
    letter = m.group(1)
    if letter.isupper():
        return True
    if (start > 0 and text[start - 1] == "(") or text[end:end + 1] == ")":
        return True
    if _ABBREVIATION_AFTER.match(text, end):
        return False
    return not (letter == "a" and _WORD_AFTER.match(text, end))


def parse_mcq_answer(text: str) -> Optional[str]:
    """Extract the chosen option letter from a hypothesis's text.

    Patterns are tried most-specific-first; within a matching pattern the LAST
    occurrence wins, since the final conclusion follows any option elimination.
    Matches whose letter is prose rather than a choice (see ``_is_choice``) are
    skipped.
    """
    for pat in _ANSWER_PATTERNS:
        matches = [
            m for m in re.finditer(pat, text, re.MULTILINE | re.IGNORECASE)
            if _is_choice(text, m)
        ]
        if matches:
            return matches[-1].group(1).upper()
    return None


def score_answer(hypothesis_text: str, gold_answer: str) -> bool:
    """Binary correctness: parsed letter equals gold (case-insensitive)."""
    parsed = parse_mcq_answer(hypothesis_text)
    return parsed is not None and parsed.upper() == gold_answer.strip().upper()
