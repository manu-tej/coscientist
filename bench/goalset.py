from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BenchGoal:
    id: str
    goal: str                                # research-goal text fed to the system
    domain: str = "computational biology"    # parameterizes the judge rubric
    gold_answer: Optional[str] = None        # MCQ letter (GPQA) → concordance
    gold_hypothesis: Optional[str] = None    # reference hypothesis (ResearchBench)
    gold_entities: list[str] = field(default_factory=list)  # entity-recall scoring
    choices: Optional[list[str]] = None      # MCQ options (GPQA)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "BenchGoal":
        known = {
            "id", "goal", "domain", "gold_answer", "gold_hypothesis",
            "gold_entities", "choices", "metadata",
        }
        base = {k: d[k] for k in known if k in d}
        extra = {k: v for k, v in d.items() if k not in known}
        meta = {**base.get("metadata", {}), **extra}
        base["metadata"] = meta
        return cls(**base)


def load_goalset(path: str | Path) -> list[BenchGoal]:
    """Load a .jsonl goal set, one BenchGoal per non-blank line."""
    goals: list[BenchGoal] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        goals.append(BenchGoal.from_dict(json.loads(line)))
    return goals
