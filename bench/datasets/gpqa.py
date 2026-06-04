from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from bench.goalset import BenchGoal

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
    for line in Path(path).read_text().splitlines():
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
    used by the opt-in integration test and real runs, not unit tests."""
    from datasets import load_dataset  # local import: heavy, network

    ds = load_dataset("hendrydong/gpqa_diamond_mc", split="test")
    goals: list[BenchGoal] = []
    for i, ex in enumerate(ds):
        subject = str(ex.get("subject") or ex.get("category") or "").strip()
        if not all_subjects and subject.lower() != "biology":
            continue
        choices = ex.get("choices") or ex.get("options")
        ans = ex.get("answer")
        if isinstance(ans, int):
            ans = _LETTERS[ans]
        row = {"id": ex.get("id", f"gpqa-{i}"), "subject": subject,
               "question": ex["question"], "choices": list(choices), "answer": ans}
        goals.append(_row_to_goal(row))
        if limit is not None and len(goals) >= limit:
            break
    return goals


_ANSWER_PATTERNS = [
    r"answer\s*[:\-]?\s*\(?([a-f])\)?\b",
    r"option\s*\(?([a-f])\)?\b",
    r"correct (?:choice|option|answer) is\s*\(?([a-f])\)?\b",
    r"^\s*\(?([a-f])\)?\s*$",
]


def parse_mcq_answer(text: str) -> Optional[str]:
    """Extract the chosen option letter from a hypothesis's text."""
    low = text.lower()
    for pat in _ANSWER_PATTERNS:
        m = re.search(pat, low, re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


def score_answer(hypothesis_text: str, gold_answer: str) -> bool:
    """Binary correctness: parsed letter equals gold (case-insensitive)."""
    parsed = parse_mcq_answer(hypothesis_text)
    return parsed is not None and parsed.upper() == gold_answer.strip().upper()
