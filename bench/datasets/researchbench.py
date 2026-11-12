from __future__ import annotations

from typing import Optional

from bench.goalset import BenchGoal

_BIO_SUBJECTS = {"biology", "cell_biology", "cell biology", "molecular biology"}


def _row_to_goal(row: dict) -> BenchGoal:
    question = str(row.get("question") or "").strip()
    background = str(row.get("background") or "").strip()
    goal_text = question if not background else f"{question}\n\nBackground: {background}"
    return BenchGoal(
        id=str(row["id"]),
        goal=goal_text,
        domain="computational biology",
        gold_hypothesis=str(row.get("hypothesis", "")).strip() or None,
        metadata={"year": row.get("year"), "subject": row.get("subject", "")},
    )


def dataframe_to_goals(df, bio_only: bool = True) -> list[BenchGoal]:
    """Convert a ResearchBench-shaped DataFrame to BenchGoals.
    Filters to biology subjects by default."""
    goals: list[BenchGoal] = []
    for row in df.to_dict(orient="records"):
        subject = str(row.get("subject", "")).strip().lower()
        if bio_only and subject not in _BIO_SUBJECTS:
            continue
        goals.append(_row_to_goal(row))
    return goals


def load_researchbench_hf(bio_only: bool = True, limit: Optional[int] = None) -> list[BenchGoal]:
    """Load ankilok/Researchbench (Parquet via HF). Network-gated."""
    import pandas as pd  # local import
    from datasets import load_dataset

    ds = load_dataset("ankilok/Researchbench", split="train")
    df = ds.to_pandas() if hasattr(ds, "to_pandas") else pd.DataFrame(ds)
    if "year" in df.columns:
        df = df[df["year"] == 2024]  # 2024-only → contamination-resistant
    goals = dataframe_to_goals(df, bio_only=bio_only)
    return goals[:limit] if limit else goals
