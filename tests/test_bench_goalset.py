import json
from pathlib import Path
from bench.goalset import BenchGoal, load_goalset


def test_benchgoal_defaults():
    g = BenchGoal(id="g1", goal="Explain X")
    assert g.domain == "computational biology"
    assert g.gold_answer is None
    assert g.gold_entities == []
    assert g.choices is None


def test_load_goalset_roundtrip(tmp_path: Path):
    rows = [
        {"id": "g1", "goal": "Explain X", "gold_answer": "B",
         "choices": ["a", "b", "c", "d"], "domain": "biology"},
        {"id": "g2", "goal": "Link Y and Z", "gold_entities": ["PI3K-Akt", "TGF-β"]},
    ]
    p = tmp_path / "goals.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    goals = load_goalset(p)
    assert len(goals) == 2
    assert goals[0].gold_answer == "B"
    assert goals[0].choices == ["a", "b", "c", "d"]
    assert goals[1].gold_entities == ["PI3K-Akt", "TGF-β"]
    assert goals[1].domain == "computational biology"  # default applied


def test_load_goalset_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "goals.jsonl"
    p.write_text('{"id":"g1","goal":"X"}\n\n   \n')
    assert len(load_goalset(p)) == 1


def test_curated_comp_bio_goldset_loads():
    from bench.goalset import load_goalset
    goals = load_goalset("bench/datasets/comp_bio_goldset.jsonl")
    assert len(goals) == 10
    assert all(g.gold_entities for g in goals)
    assert all(g.domain == "computational biology" for g in goals)
    cb1 = next(g for g in goals if g.id == "cb-1")
    assert "PI3K-Akt" in cb1.gold_entities


def test_researchbench_dataframe_to_goals():
    import pandas as pd
    from bench.datasets.researchbench import dataframe_to_goals
    df = pd.DataFrame([
        {"id": "rb-1", "subject": "Biology", "year": 2024,
         "question": "What drives X?", "background": "context",
         "hypothesis": "X is driven by gene G."},
        {"id": "rb-2", "subject": "Physics", "year": 2024,
         "question": "q", "background": "b", "hypothesis": "h"},
    ])
    goals = dataframe_to_goals(df)
    assert [g.id for g in goals] == ["rb-1"]      # biology only
    assert goals[0].gold_hypothesis == "X is driven by gene G."
    assert "What drives X?" in goals[0].goal
    assert goals[0].domain == "computational biology"


def test_researchbench_handles_null_background():
    import pandas as pd
    from bench.datasets.researchbench import dataframe_to_goals
    df = pd.DataFrame([
        {"id": "rb-3", "subject": "Biology", "year": 2024,
         "question": "What drives Y?", "background": None, "hypothesis": "Y by gene H."},
    ])
    goals = dataframe_to_goals(df)
    assert len(goals) == 1
    assert "Background" not in goals[0].goal     # null background omitted, not "None"
    assert goals[0].goal == "What drives Y?"
