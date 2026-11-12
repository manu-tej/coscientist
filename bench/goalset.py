from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tools.resources import resource_path
from bench.errors import BenchError


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
    resolved = resource_path(path)
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("row must be a JSON object")
            goals.append(BenchGoal.from_dict(row))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise BenchError(
                f"Invalid goalset row {line_number} in {path}: {exc}"
            ) from exc
    return goals


def validate_rediscovery_goals(
    goals: list[BenchGoal],
    *,
    source: str | Path = "<rediscovery goalset>",
) -> None:
    """Fail closed unless every rediscovery goal is uniquely and fully scorable."""
    if not goals:
        raise BenchError(f"Rediscovery goalset {source} contains no goals")

    seen_ids: dict[str, int] = {}
    for index, goal in enumerate(goals, start=1):
        if not isinstance(goal.id, str) or not goal.id.strip():
            raise BenchError(
                f"Rediscovery goal {index} in {source} has an empty id"
            )
        normalized_id = goal.id.strip()
        if normalized_id in seen_ids:
            raise BenchError(
                f"Rediscovery goalset {source} has duplicate id "
                f"{normalized_id!r} at rows {seen_ids[normalized_id]} and {index}"
            )
        seen_ids[normalized_id] = index

        if not isinstance(goal.goal, str) or not goal.goal.strip():
            raise BenchError(
                f"Rediscovery goal {normalized_id!r} in {source} has an empty goal"
            )
        if (
            not isinstance(goal.gold_entities, list)
            or not goal.gold_entities
            or any(
                not isinstance(entity, str) or not entity.strip()
                for entity in goal.gold_entities
            )
        ):
            raise BenchError(
                f"Rediscovery goal {normalized_id!r} in {source} must have "
                "nonempty gold_entities containing only nonempty strings"
            )
