import pytest

from bench.errors import BenchError
from bench.orchestrate import _load_concordance_goals


def test_gpqa_fixture_fallback_is_explicit_and_provenanced(monkeypatch):
    from bench.datasets import gpqa

    monkeypatch.setattr(
        gpqa, "load_gpqa_hf",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    with pytest.raises(BenchError, match="allow-fixture-fallback"):
        _load_concordance_goals("gpqa-bio", 2)

    goals, provenance = _load_concordance_goals(
        "gpqa-bio", 2, allow_fixture_fallback=True,
    )
    assert goals
    assert provenance["fixture_fallback"] is True
    assert "fixtures/gpqa_sample.jsonl" in provenance["source"]
