import json
import pytest
from bench.errors import BenchError
from bench.judge import (
    Judge, RUBRIC_AXES, parse_rubric_json, weighted_total, build_judge_prompt,
)


class _FakeBackend:
    def __init__(self, payload): self.payload = payload
    async def score_text(self, system, user): return self.payload


def test_judge_rejects_same_model():
    with pytest.raises(BenchError):
        Judge(backend=_FakeBackend("{}"), judge_model="m", generator_model="m")


def test_parse_rubric_json_extracts_scores():
    payload = json.dumps({
        "novelty": {"score": 4, "justification": "fresh"},
        "feasibility": {"score": 3, "justification": "ok"},
        "correctness": {"score": 5, "justification": "sound"},
        "impact": {"score": 2, "justification": "narrow"},
    })
    scores = parse_rubric_json(payload)
    assert scores == {"novelty": 4, "feasibility": 3, "correctness": 5, "impact": 2}


def test_parse_rubric_json_tolerates_surrounding_text():
    payload = 'Here is my evaluation:\n```json\n{"novelty":{"score":3},"feasibility":{"score":3},"correctness":{"score":3},"impact":{"score":3}}\n```\nDone.'
    assert parse_rubric_json(payload)["novelty"] == 3


def test_weighted_total_default_equal_weights():
    scores = {"novelty": 4, "feasibility": 4, "correctness": 4, "impact": 4}
    assert weighted_total(scores) == 4.0


def test_build_judge_prompt_includes_field():
    p = build_judge_prompt("hypothesis text", field="computational biology")
    assert "computational biology" in p
    assert "novelty" in p.lower()


@pytest.mark.asyncio
async def test_judge_score_returns_axes_and_total():
    payload = json.dumps({a: {"score": 4} for a in RUBRIC_AXES})
    j = Judge(backend=_FakeBackend(payload), judge_model="judge-x", generator_model="gen-y")
    result = await j.score("a hypothesis", field="computational biology")
    assert result["scores"] == {a: 4 for a in RUBRIC_AXES}
    assert result["total"] == 4.0
