import json

import pytest

from bench.sweep import ModelSpec, build_cells, parse_int_list


@pytest.fixture(autouse=True)
def _run_databases_in_tmp(tmp_path, monkeypatch):
    """benchmark_db_path() creates ./bench_runs, which is where real run
    databases live in a working copy. Keep this module out of it."""
    monkeypatch.chdir(tmp_path)


def test_model_spec_and_cartesian_grid_are_explicit():
    cells = build_cells(
        "1,4",
        ["claude-code:sonnet:opus", "openai-compatible:local-model"],
        "0,2",
    )
    assert len(cells) == 2 * 2 * 2
    assert {(c.provider, c.workers, c.seed) for c in cells} >= {
        ("claude-code", 1, 0),
        ("claude-code", 4, 2),
        ("openai-compatible", 1, 2),
    }
    local = next(c for c in cells if c.provider == "openai-compatible")
    assert local.model_fast == local.model_strong == "local-model"
    assert len({c.id for c in cells}) == len(cells)


def test_unambiguous_model_syntax_allows_colons_and_deduplicates_cells():
    cells = build_cells(
        "2",
        [
            "openai-compatible=llama3.1:8b",
            "openai-compatible=llama3.1:8b",
        ],
        "0",
    )
    assert len(cells) == 1
    assert cells[0].model_fast == cells[0].model_strong == "llama3.1:8b"


def test_sanitized_model_names_do_not_overwrite_each_other():
    cells = build_cells(
        "1",
        ["openai-compatible=a/b", "openai-compatible=a?b"],
        "0",
    )
    assert len(cells) == 2
    assert cells[0].id != cells[1].id


@pytest.mark.parametrize("value", ["", "claude-code", "provider:model:strong:extra"])
def test_invalid_model_spec_fails_loudly(value):
    with pytest.raises(ValueError):
        ModelSpec.parse(value)


def test_worker_counts_must_be_positive_and_are_deduplicated():
    assert parse_int_list("1,2,2,4", minimum=1) == [1, 2, 4]
    with pytest.raises(ValueError):
        parse_int_list("0,2", minimum=1)


@pytest.mark.asyncio
async def test_run_sweep_executes_every_cell_and_records_exact_selection(
    monkeypatch, tmp_path,
):
    import argparse
    import bench.orchestrate as orchestrate

    seen = []

    async def fake_rediscovery(args):
        seen.append(
            (args.provider, args.model_fast, args.model_strong, args.workers, args.seed)
        )
        return {
            "rediscovery": {
                "mean_pool_recall": 0.5,
                "mean_top1_recall": 0.25,
                "elo_recall_spearman": 0.1,
            },
            "meta": {"n_completed": 1, "n_failed": 0, "n_llm_calls": 7},
        }

    monkeypatch.setattr(orchestrate, "run_rediscovery", fake_rediscovery)
    args = argparse.Namespace(
        worker_counts="1,3",
        models=["openai-compatible=local:8b"],
        seeds="0,2",
        benchmark="rediscover",
        out=str(tmp_path / "sweep"),
    )
    report = await orchestrate.run_sweep(args)

    assert len(seen) == 4
    assert set(seen) == {
        ("openai-compatible", "local:8b", "local:8b", 1, 0),
        ("openai-compatible", "local:8b", "local:8b", 1, 2),
        ("openai-compatible", "local:8b", "local:8b", 3, 0),
        ("openai-compatible", "local:8b", "local:8b", 3, 2),
    }
    assert report["sweep"]["n_completed"] == 4
    assert all(row["n_llm_calls"] == 7 for row in report["sweep"]["cells"])


@pytest.mark.asyncio
async def test_absolute_rediscovery_sweep_uses_native_quality_key(
    monkeypatch, tmp_path,
):
    import argparse
    import bench.orchestrate as orchestrate

    async def fake_rediscovery(args):
        assert args.ranking_mode == "absolute"
        return {
            "rediscovery": {
                "ranking_mode": "absolute",
                "mean_pool_recall": 0.5,
                "mean_top1_recall": 0.25,
                "ranking_recall_spearman": 0.3,
            },
            "meta": {"n_completed": 1, "n_failed": 0, "n_scorable": 1},
        }

    monkeypatch.setattr(orchestrate, "run_rediscovery", fake_rediscovery)
    args = argparse.Namespace(
        worker_counts="1",
        models=["openai-compatible=system-model"],
        seeds="0",
        benchmark="rediscover",
        ranking_mode="absolute",
        out=str(tmp_path / "absolute-sweep"),
    )

    report = await orchestrate.run_sweep(args)

    quality = report["sweep"]["cells"][0]["quality"]
    assert quality["ranking_mode"] == "absolute"
    assert quality["ranking_recall_spearman"] == 0.3
    assert "elo_recall_spearman" not in quality


@pytest.mark.asyncio
async def test_concordance_sweep_rejects_absolute_mode_before_cells(tmp_path):
    import argparse
    from bench.errors import BenchError
    from bench.orchestrate import run_sweep

    args = argparse.Namespace(
        worker_counts="1",
        models=["openai-compatible=system-model"],
        seeds="0",
        benchmark="concordance",
        ranking_mode="absolute",
        out=str(tmp_path / "invalid-sweep"),
    )

    with pytest.raises(BenchError, match="require ranking_mode='elo'"):
        await run_sweep(args)


@pytest.mark.asyncio
async def test_run_sweep_marks_zero_scorable_cell_failed(monkeypatch, tmp_path):
    import argparse
    import bench.orchestrate as orchestrate

    async def fake_rediscovery(args):
        return {
            "rediscovery": {
                "mean_pool_recall": 0.0,
                "mean_top1_recall": 0.0,
                "elo_recall_spearman": float("nan"),
            },
            "meta": {
                "n_attempted": 1,
                "n_accepted": 1,
                "n_completed": 1,
                "n_failed": 0,
                "n_scorable": 0,
                "n_llm_calls": 4,
                "estimated_tokens": 40,
            },
        }

    monkeypatch.setattr(orchestrate, "run_rediscovery", fake_rediscovery)
    args = argparse.Namespace(
        worker_counts="1",
        models=["openai-compatible=local"],
        seeds="0",
        benchmark="rediscover",
        out=str(tmp_path / "sweep"),
    )

    report = await orchestrate.run_sweep(args)
    cell = report["sweep"]["cells"][0]

    assert report["sweep"]["n_completed"] == 0
    assert report["sweep"]["n_failed"] == 1
    assert cell["status"] == "failed"
    assert cell["n_scorable"] == 0
    assert cell["n_llm_calls"] == 4
    assert cell["estimated_tokens"] == 40


@pytest.mark.asyncio
async def test_run_sweep_honors_allow_partial_for_scorable_cells(
    monkeypatch, tmp_path,
):
    import argparse
    import bench.orchestrate as orchestrate

    async def fake_rediscovery(args):
        return {
            "rediscovery": {
                "mean_pool_recall": 0.5,
                "mean_top1_recall": 0.5,
                "elo_recall_spearman": 0.0,
            },
            "meta": {
                "n_attempted": 2,
                "n_accepted": 1,
                "n_completed": 1,
                "n_failed": 1,
                "n_scorable": 1,
                "n_llm_calls": 4,
                "estimated_tokens": 40,
            },
        }

    monkeypatch.setattr(orchestrate, "run_rediscovery", fake_rediscovery)
    args = argparse.Namespace(
        worker_counts="1",
        models=["openai-compatible=local"],
        seeds="0",
        benchmark="rediscover",
        out=str(tmp_path / "sweep"),
        allow_partial=True,
    )

    report = await orchestrate.run_sweep(args)

    assert report["sweep"]["n_completed"] == 1
    assert report["sweep"]["n_failed"] == 0
    assert report["sweep"]["cells"][0]["n_failed"] == 1


@pytest.mark.asyncio
async def test_sweep_shares_manifest_without_cell_collisions_and_reports_reuse(
    monkeypatch, tmp_path,
):
    import argparse
    import bench.manifest as manifest_module
    import bench.orchestrate as orchestrate
    import bench.runner as runner
    from bench.manifest import CacheLookup, LeaseClaim
    from bench.runner import BenchHypothesis, BenchRun

    class InMemoryManifest:
        def __init__(self):
            self.path = str(tmp_path / "shared-manifest.sqlite")
            self.runs = {}
            self.keys = []
            self.leases = {}

        async def load_complete(
            self, key, *, expected_goal, expected_tasks,
        ):
            self.keys.append(key)
            run = self.runs.get(key.id)
            return CacheLookup(
                state="hit" if run is not None else "miss",
                run=run,
                cache_key=key.id,
            )

        async def record_complete(
            self, key, run, *, expected_goal, expected_tasks, owner_token=None,
        ):
            assert self.leases.get(key.id) == owner_token
            self.runs[key.id] = run
            del self.leases[key.id]

        async def claim(self, key, *, lease_seconds, owner_token=None):
            if key.id in self.runs:
                return LeaseClaim(state="ready")
            if key.id not in self.leases:
                token = owner_token or f"owner-{key.id}"
                self.leases[key.id] = token
                return LeaseClaim(
                    state="acquired", owner_token=token, expires_at=9999999999,
                )
            return LeaseClaim(state="waiting", expires_at=9999999999)

        async def renew(self, key, *, owner_token, lease_seconds):
            return self.leases.get(key.id) == owner_token

        async def release(self, key, *, owner_token):
            if self.leases.get(key.id) != owner_token:
                return False
            del self.leases[key.id]
            return True

    shared = InMemoryManifest()

    async def shared_manifest(args):
        return shared

    model_calls = []

    async def fake_run_system(goal, **kwargs):
        model_calls.append(kwargs)
        return BenchRun(
            goal_id=goal.id,
            variant="full",
            hypotheses=[
                BenchHypothesis(
                    f"h-{len(model_calls)}",
                    "SPP1 mechanism",
                    "SPP1",
                    1200.0,
                    "now",
                )
            ],
            n_llm_calls=3,
            wall_clock_s=0.1,
            db_path=kwargs["db_path"],
            run_id=f"run-{len(model_calls)}",
            agent_task_counts={"generation": kwargs["max_tasks"]},
            agent_task_trace=[
                {
                    "priority": index,
                    "agent_type": "generation",
                    "persona": "",
                    "status": "succeeded",
                }
                for index in range(kwargs["max_tasks"])
            ],
            workers=kwargs["n_workers"],
            backend=kwargs["provider"],
            models={
                "fast": kwargs["model_fast"],
                "strong": kwargs["model_strong"],
            },
            estimated_tokens=30,
        )

    goalset = tmp_path / "goals.jsonl"
    goalset.write_text(json.dumps({
        "id": "shared-goal",
        "goal": "Find SPP1",
        "gold_entities": ["SPP1"],
    }))
    monkeypatch.setattr(orchestrate, "_manifest_for_args", shared_manifest)
    monkeypatch.setattr(runner, "run_system", fake_run_system)
    monkeypatch.setattr(
        manifest_module, "system_version", lambda: "sweep-cache-v1",
    )
    args = argparse.Namespace(
        worker_counts="1,3",
        models=["codex=fast,strong"],
        seeds="0",
        benchmark="rediscover",
        out=str(tmp_path / "sweep"),
        goals=str(goalset),
        limit=1,
        full=False,
        max_tasks=1,
        max_time=10,
        min_support=1,
        allow_partial=False,
        manifest=shared.path,
        no_cache=False,
    )

    fresh = await orchestrate.run_sweep(args)
    reused = await orchestrate.run_sweep(args)

    assert len(model_calls) == 2
    assert len(shared.runs) == 2
    assert len({key.id for key in shared.keys}) == 2
    assert len({call["db_path"] for call in model_calls}) == 2
    assert fresh["sweep"]["cache"]["misses"] == 2
    assert fresh["sweep"]["fresh_llm_calls"] == 6
    assert fresh["sweep"]["reused_llm_calls"] == 0
    assert reused["sweep"]["cache"]["hits"] == 2
    assert reused["sweep"]["fresh_llm_calls"] == 0
    assert reused["sweep"]["reused_llm_calls"] == 6
    identities = [row["execution_identity"] for row in fresh["sweep"]["cells"]]
    assert {identity["workers"] for identity in identities} == {1, 3}
    assert all(identity["system_version"] == "sweep-cache-v1" for identity in identities)


@pytest.mark.asyncio
async def test_concordance_sweep_cell_records_logistic_headline(monkeypatch, tmp_path):
    import argparse
    import bench.orchestrate as orchestrate

    async def fake_concordance(args):
        return {
            "concordance": {
                "spearman_rho": 0.2, "spearman_p": 0.6,
                "logistic_coef": 0.4, "logistic_p": 0.01,
                "logistic_n_rows": 40, "logistic_n_questions": 5,
                "top1_accuracy": 0.6,
            },
            "verdicts": {"concordance_pass": True},
            "meta": {"n_completed": 1, "n_failed": 0, "n_llm_calls": 3},
        }

    monkeypatch.setattr(orchestrate, "run_concordance", fake_concordance)
    args = argparse.Namespace(
        worker_counts="1",
        models=["openai-compatible=local:8b"],
        seeds="0",
        benchmark="concordance",
        out=str(tmp_path / "sweep"),
    )
    report = await orchestrate.run_sweep(args)

    quality = report["sweep"]["cells"][0]["quality"]
    assert quality["logistic_coef"] == 0.4
    assert quality["logistic_p"] == 0.01
    assert quality["concordance_pass"] is True
    assert quality["spearman_rho"] == 0.2          # still recorded, as descriptive
    assert quality["top1_accuracy"] == 0.6
