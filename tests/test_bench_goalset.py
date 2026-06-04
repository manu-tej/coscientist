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
