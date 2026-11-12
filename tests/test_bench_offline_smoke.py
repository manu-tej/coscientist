from __future__ import annotations

import json
import socket

import pytest

from bench import offline_smoke


@pytest.mark.asyncio
async def test_offline_benchmark_smoke_is_stable_complete_and_concurrent(tmp_path):
    kwargs = {
        "output_path": tmp_path / "report.json",
        "work_dir": tmp_path / "runs",
        "max_tasks": 4,
    }
    first = await offline_smoke.run_offline_smoke(**kwargs)
    first_artifact = json.loads((tmp_path / "report.json").read_text())
    second = await offline_smoke.run_offline_smoke(**kwargs)
    second_artifact = json.loads((tmp_path / "report.json").read_text())

    assert first == first_artifact == second == second_artifact
    assert first["label"] == "STRUCTURAL_OFFLINE_BENCHMARK_SMOKE"
    assert first["status"] == "passed"
    assert first["scientific_quality_benchmark"] is False
    assert first["configuration"]["network_enabled"] is False
    assert first["configuration"]["workers"] == [1, 2, 4]
    assert first["provenance"]["runtime"]["runner"] == "bench.runner.run_system"
    assert (
        first["provenance"]["runtime"]["scorer"]
        == "bench.orchestrate.rediscovery_from_runs"
    )
    assert len(first["provenance"]["dataset"]["sha256"]) == 64

    cells = {cell["workers"]: cell for cell in first["cells"]}
    assert set(cells) == {1, 2, 4}
    assert {workers: cell["max_backend_concurrency"]
            for workers, cell in cells.items()} == {1: 1, 2: 2, 4: 4}
    assert cells[1]["backend_calls_overlapped"] is False
    assert cells[2]["backend_calls_overlapped"] is True
    assert cells[4]["backend_calls_overlapped"] is True
    assert all(cell["completed_tasks"] == 4 for cell in cells.values())
    assert all(cell["agent_task_counts"] == {"generation": 4}
               for cell in cells.values())
    assert all(cell["agent_failures"] == [] for cell in cells.values())
    assert all(cell["n_hypotheses"] == 4 for cell in cells.values())
    assert all(cell["rediscovery"]["pool_recall"] == 1.0
               for cell in cells.values())
    assert all(cell["rediscovery"]["top1_recall"] == 1.0
               for cell in cells.values())
    assert all(cell["provenance"]["run_id_present"] for cell in cells.values())
    assert all(cell["provenance"]["database_existed_during_validation"]
               for cell in cells.values())
    assert all(all(cell["checks"].values()) for cell in cells.values())
    assert all(first["checks"].values())


@pytest.mark.asyncio
async def test_offline_smoke_calls_production_runner_and_rediscovery_scorer(
    monkeypatch, tmp_path,
):
    original_runner = offline_smoke.run_system
    original_scorer = offline_smoke.rediscovery_from_runs
    runner_workers: list[int] = []
    scored_workers: list[int] = []

    async def traced_runner(*args, **kwargs):
        run = await original_runner(*args, **kwargs)
        runner_workers.append(run.workers)
        return run

    def traced_scorer(goals, runs):
        scored_workers.extend(run.workers for run in runs)
        return original_scorer(goals, runs)

    monkeypatch.setattr(offline_smoke, "run_system", traced_runner)
    monkeypatch.setattr(offline_smoke, "rediscovery_from_runs", traced_scorer)

    await offline_smoke.run_offline_smoke(
        output_path=tmp_path / "report.json",
        work_dir=tmp_path / "runs",
        max_tasks=4,
    )

    assert runner_workers == [1, 2, 4]
    assert scored_workers == [1, 2, 4]


def test_offline_benchmark_smoke_network_guard_fails_closed():
    with offline_smoke._zero_network_environment():
        with pytest.raises(
            offline_smoke.OfflineBenchmarkSmokeError,
            match="network access attempted",
        ):
            socket.create_connection(("example.com", 443))


def test_offline_benchmark_smoke_command_emits_json(
    monkeypatch, tmp_path, capsys,
):
    expected = {
        "schema_version": 1,
        "label": "STRUCTURAL_OFFLINE_BENCHMARK_SMOKE",
        "status": "passed",
        "scientific_quality_benchmark": False,
    }

    async def fake_run(**kwargs):
        offline_smoke._write_json(kwargs["output_path"], expected)
        return expected

    monkeypatch.setattr(offline_smoke, "run_offline_smoke", fake_run)
    output = tmp_path / "cli.json"
    exit_code = offline_smoke.main(["--output", str(output)])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert json.loads(output.read_text()) == expected


@pytest.mark.asyncio
async def test_offline_benchmark_requires_enough_tasks_for_four_workers(tmp_path):
    with pytest.raises(ValueError, match="max_tasks must be >= 4"):
        await offline_smoke.run_offline_smoke(
            output_path=tmp_path / "report.json",
            work_dir=tmp_path / "runs",
            max_tasks=3,
        )


@pytest.mark.asyncio
async def test_offline_benchmark_uses_and_records_custom_goalset(tmp_path):
    goalset = tmp_path / "custom.jsonl"
    goalset.write_text(json.dumps({
        "id": "custom",
        "goal": "Connect custom markers",
        "gold_entities": ["CUSTOM_ALPHA", "CUSTOM_BETA"],
    }) + "\n")

    report = await offline_smoke.run_offline_smoke(
        output_path=tmp_path / "report.json",
        work_dir=tmp_path / "runs",
        max_tasks=4,
        goalset_path=goalset,
    )

    assert report["provenance"]["dataset"]["path"] == str(goalset.resolve())
    assert report["provenance"]["dataset"]["gold_entities"] == [
        "CUSTOM_ALPHA",
        "CUSTOM_BETA",
    ]
    assert all(cell["rediscovery"]["pool_recall"] == 1.0 for cell in report["cells"])


def test_offline_benchmark_artifact_replace_is_atomic(monkeypatch, tmp_path):
    output = tmp_path / "report.json"
    output.write_text('{"old": true}\n')
    original_replace = offline_smoke.Path.replace

    def fail_replace(path, target):
        if target == output:
            raise OSError("disk failure")
        return original_replace(path, target)

    monkeypatch.setattr(offline_smoke.Path, "replace", fail_replace)

    with pytest.raises(OSError, match="disk failure"):
        offline_smoke._write_json(output, {"new": True})

    assert output.read_text() == '{"old": true}\n'
    assert list(tmp_path.glob("*.tmp")) == []
