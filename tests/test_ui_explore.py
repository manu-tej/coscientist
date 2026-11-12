"""Round-trip: ui.explore must recover the task id from bench.runner's filenames.

The two modules drifted once (bench added a digest + run_id to the filename; the
explorer's parser expected the old shape), which silently killed the Answer-vs-gold
tab for every new run. These tests pin them together.
"""
import uuid

import pytest

from bench.runner import benchmark_db_path
from ui import explore
from ui.explore import resolve_gold, run_family


@pytest.fixture(autouse=True)
def _no_gpqa_download(monkeypatch, tmp_path):
    # GPQA gold letters come from the Hugging Face mirror. These tests are about
    # filename round trips, so keep them off the network.
    monkeypatch.setattr(explore, "_gpqa_gold", lambda: {})
    # benchmark_db_path() creates ./bench_runs; keep it out of the working tree.
    monkeypatch.chdir(tmp_path)


def test_resolve_gold_round_trips_concordance_with_run_id():
    path = benchmark_db_path(
        "gpqa-biology-3", label="concordance", seed=0, run_id=uuid.uuid4().hex[:8]
    )
    gold = resolve_gold(path)
    assert gold["kind"] == "gpqa"
    assert gold["task"] == "gpqa-biology-3"


def test_resolve_gold_round_trips_rediscover_without_run_id():
    path = benchmark_db_path("da-1-3", label="rediscover", seed=4)
    gold = resolve_gold(path)
    assert gold["kind"] == "rediscover"
    assert gold["task"] == "da-1-3"


def test_resolve_gold_still_parses_legacy_filenames():
    assert resolve_gold("bench_runs/concordance_gpqa-biology-2_0.db")["task"] == (
        "gpqa-biology-2"
    )


def test_run_family_groups_mode_suffixed_labels():
    plain = benchmark_db_path("da-1-3", label="rediscover", seed=0)
    absolute = benchmark_db_path("da-1-3", label="rediscover-absolute", seed=0)
    assert run_family(plain) == "BiomniBench (rediscovery)"
    assert run_family(absolute) == "BiomniBench (rediscovery)"
    concordance = benchmark_db_path("gpqa-biology-1", label="concordance", seed=0)
    assert run_family(concordance) == "GPQA-bio (concordance)"
