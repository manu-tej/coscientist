"""Deterministic, zero-network smoke for benchmark runtime scaling.

This command exercises the production ``bench.runner.run_system`` path,
including its Supervisor, AgentRunner, SQLite persistence, and rediscovery
scoring.  It intentionally uses a fixed in-process backend and is therefore a
runtime/integration benchmark, not evidence of scientific model quality.

Run from the repository root:

    python -m bench.offline_smoke

The default artifact is ``artifacts/bench_offline_smoke.json``.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import socket
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable
from unittest.mock import patch

from bench.goalset import BenchGoal, load_goalset
from bench.orchestrate import rediscovery_from_runs
from bench.runner import run_system
from core.models import AgentType


SCHEMA_VERSION = 1
SMOKE_LABEL = "STRUCTURAL_OFFLINE_BENCHMARK_SMOKE"
WORKER_COUNTS = (1, 2, 4)
DEFAULT_GOALSET = Path("bench/datasets/biomnibench_goldset.jsonl")
_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


class OfflineBenchmarkSmokeError(RuntimeError):
    """A required offline benchmark invariant did not hold."""


class _DeterministicBackend:
    """Minimal portable backend with a concurrency rendezvous and no I/O."""

    model_fast = "offline-deterministic-v1"
    model_strong = "offline-deterministic-v1"

    def __init__(self, expected_concurrency: int, gold_entities: Iterable[str]):
        self.expected_concurrency = expected_concurrency
        self.gold_entities = tuple(str(entity) for entity in gold_entities)
        self.call_count = 0
        self.active = 0
        self.max_active = 0
        self._rendezvous = asyncio.Event()

    async def call(self, system: str, user: str, **kwargs: Any) -> str:
        self.call_count += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.active >= self.expected_concurrency:
            self._rendezvous.set()
        try:
            if self.expected_concurrency > 1:
                try:
                    # The first wave cannot finish until all configured workers
                    # enter the backend. This turns concurrency into an observed
                    # invariant instead of a timing-dependent approximation.
                    await asyncio.wait_for(self._rendezvous.wait(), timeout=2.0)
                except TimeoutError as exc:
                    raise OfflineBenchmarkSmokeError(
                        f"{self.expected_concurrency} workers did not overlap "
                        "backend calls"
                    ) from exc
            await asyncio.sleep(0.002)
            entities = ", ".join(self.gold_entities)
            return (
                "HYPOTHESIS\n"
                f"Hypothesis: Jointly test the supplied entities: {entities}.\n"
                f"Summary: A deterministic validation hypothesis covering {entities}.\n"
                "Category: Offline benchmark fixture"
            )
        finally:
            self.active -= 1


class _NoopProximity:
    """Keep the smoke offline without loading an embedding model."""

    async def update_graph(self, hypotheses):
        return []


@contextmanager
def _zero_network_environment():
    """Disable optional search and fail immediately on every socket attempt."""
    sentinel = object()
    previous: object | str = os.environ.get("TAVILY_API_KEY", sentinel)
    os.environ["TAVILY_API_KEY"] = ""

    def blocked(*args, **kwargs):
        raise OfflineBenchmarkSmokeError(
            "network access attempted during zero-network benchmark smoke"
        )

    try:
        with (
            patch.object(socket.socket, "connect", blocked),
            patch.object(socket.socket, "connect_ex", blocked),
            patch.object(socket, "create_connection", blocked),
            patch.object(socket, "getaddrinfo", blocked),
        ):
            yield
    finally:
        if previous is sentinel:
            os.environ.pop("TAVILY_API_KEY", None)
        else:
            os.environ["TAVILY_API_KEY"] = str(previous)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OfflineBenchmarkSmokeError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _REPOSITORY_ROOT / candidate


def _fixed_goal(goalset_path: Path) -> BenchGoal:
    goals = load_goalset(goalset_path)
    _require(bool(goals), f"offline goal set is empty: {goalset_path}")
    goal = goals[0]
    _require(bool(goal.gold_entities), "offline goal has no gold entities")
    return goal


def _generation_only_weights() -> dict[AgentType, float]:
    return {
        agent_type: 0.0
        for agent_type in AgentType
        if agent_type is not AgentType.GENERATION
    }


def _score_cell(goal: BenchGoal, run) -> dict[str, Any]:
    score = rediscovery_from_runs([goal], [run])
    _require(score["n_goals"] == 1, "rediscovery scorer omitted the smoke goal")
    _require(
        len(score["per_goal"]) == 1,
        "rediscovery scorer did not return exactly one goal row",
    )
    row = score["per_goal"][0]
    return {
        "pool_recall": row["pool_recall"],
        "top1_recall": row["top1_recall"],
        "top3_recall": row["top3_recall"],
        "gold_entities": row["n_gold"],
        "hypotheses_scored": score["n_hyps_scored"],
    }


async def _run_cell(
    *,
    goal: BenchGoal,
    workers: int,
    max_tasks: int,
    seed: int,
    work_dir: Path,
    config_yaml: Path,
) -> dict[str, Any]:
    backend = _DeterministicBackend(
        expected_concurrency=workers,
        gold_entities=goal.gold_entities,
    )
    db_path = work_dir / f"workers-{workers}.db"
    run = await run_system(
        goal,
        variant="offline_runtime_smoke",
        seed=seed,
        db_path=str(db_path),
        max_tasks=max_tasks,
        max_time_seconds=30,
        weight_overrides=_generation_only_weights(),
        backend=backend,
        # This harness's whole point is zero network (see
        # _zero_network_environment below); an injected backend alone no
        # longer implies that -- it must say so explicitly.
        evidence_provider="offline",
        n_workers=workers,
        proximity_agent=_NoopProximity(),
        config_yaml=str(config_yaml),
    )
    completed_tasks = sum(run.agent_task_counts.values())
    score = _score_cell(goal, run)
    calls_overlapped = run.max_backend_concurrency > 1
    checks = {
        "task_budget_completed": completed_tasks == max_tasks,
        "agent_failures_absent": not run.agent_failures,
        "hypotheses_present": bool(run.hypotheses),
        "one_hypothesis_per_task": len(run.hypotheses) == max_tasks,
        "one_model_call_per_task": run.n_llm_calls == max_tasks,
        "gold_pool_recall_complete": score["pool_recall"] == 1.0,
        "gold_top1_recall_complete": score["top1_recall"] == 1.0,
        "sqlite_provenance_present": bool(run.run_id) and db_path.exists(),
        "configured_workers_recorded": run.workers == workers,
        "backend_accounting_reconciled": backend.call_count == run.n_llm_calls,
        "expected_call_overlap": (
            not calls_overlapped if workers == 1 else calls_overlapped
        ),
    }
    _require(
        all(checks.values()),
        f"offline benchmark cell with {workers} worker(s) failed: "
        f"{[name for name, passed in checks.items() if not passed]}",
    )
    return {
        "workers": workers,
        "configured_task_budget": max_tasks,
        "completed_tasks": completed_tasks,
        "agent_task_counts": dict(sorted(run.agent_task_counts.items())),
        "agent_failures": list(run.agent_failures),
        "hypotheses_present": bool(run.hypotheses),
        "n_hypotheses": len(run.hypotheses),
        "n_llm_calls": run.n_llm_calls,
        "max_backend_concurrency": run.max_backend_concurrency,
        "backend_calls_overlapped": calls_overlapped,
        "backend": run.backend,
        "models": dict(sorted(run.models.items())),
        "rediscovery": score,
        "provenance": {
            "run_id_present": bool(run.run_id),
            "database_file": db_path.name,
            "database_existed_during_validation": db_path.exists(),
            "variant": run.variant,
            "goal_id": run.goal_id,
        },
        "checks": checks,
    }


def _write_json(path: str | Path, report: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(
                report,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _provenance_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPOSITORY_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


async def run_offline_smoke(
    *,
    output_path: str | Path = "artifacts/bench_offline_smoke.json",
    work_dir: str | Path = "artifacts/bench_offline_smoke_runs",
    max_tasks: int = 8,
    seed: int = 0,
    goalset_path: str | Path = DEFAULT_GOALSET,
    worker_counts: Iterable[int] = WORKER_COUNTS,
) -> dict[str, Any]:
    """Run the fixed worker sweep and return/write its stable JSON report."""
    worker_counts = tuple(worker_counts)
    if worker_counts != WORKER_COUNTS:
        raise ValueError(f"worker_counts must be exactly {WORKER_COUNTS}")
    if max_tasks < max(worker_counts):
        raise ValueError(
            f"max_tasks must be >= {max(worker_counts)} to prove worker overlap"
        )

    goalset_path = _repo_path(goalset_path)
    config_yaml = _REPOSITORY_ROOT / "config.yaml"
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    goal = _fixed_goal(goalset_path)

    cells: list[dict[str, Any]] = []
    with _zero_network_environment():
        try:
            socket.create_connection(("offline-smoke.invalid", 443))
        except OfflineBenchmarkSmokeError:
            network_guard_probe_passed = True
        else:
            network_guard_probe_passed = False

        for workers in worker_counts:
            cells.append(
                await _run_cell(
                    goal=goal,
                    workers=workers,
                    max_tasks=max_tasks,
                    seed=seed,
                    work_dir=work_dir,
                    config_yaml=config_yaml,
                )
            )

    _require(network_guard_probe_passed, "network guard probe did not fail closed")
    checks = {
        "all_worker_cells_completed": len(cells) == len(WORKER_COUNTS)
        and all(cell["checks"]["task_budget_completed"] for cell in cells),
        "all_cells_failure_free": all(not cell["agent_failures"] for cell in cells),
        "all_cells_have_hypotheses": all(
            cell["hypotheses_present"] for cell in cells
        ),
        "all_cells_rediscovered_gold": all(
            cell["rediscovery"]["pool_recall"] == 1.0 for cell in cells
        ),
        "all_cells_have_sqlite_provenance": all(
            cell["checks"]["sqlite_provenance_present"] for cell in cells
        ),
        "single_worker_calls_did_not_overlap": cells[0][
            "max_backend_concurrency"
        ]
        == 1,
        "multi_worker_calls_overlapped": all(
            cell["max_backend_concurrency"] > 1 for cell in cells[1:]
        ),
        "network_guard_was_enabled": network_guard_probe_passed,
    }
    _require(all(checks.values()), "one or more offline benchmark checks failed")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "label": SMOKE_LABEL,
        "status": "passed",
        "scientific_quality_benchmark": False,
        "scope": (
            "Deterministic runtime, persistence, scoring, and generation-call "
            "concurrency smoke; not a measure of six-role scaling or LLM "
            "scientific quality."
        ),
        "configuration": {
            "workers": list(WORKER_COUNTS),
            "max_tasks_per_cell": max_tasks,
            "seed": seed,
            "network_enabled": False,
            "backend": "_DeterministicBackend",
            "model": _DeterministicBackend.model_fast,
            "scheduler_workload": "generation-only",
        },
        "provenance": {
            "dataset": {
                "path": _provenance_path(goalset_path),
                "sha256": _sha256(goalset_path),
                "goal_id": goal.id,
                "gold_entities": list(goal.gold_entities),
            },
            "runtime": {
                "runner": "bench.runner.run_system",
                "runner_sha256": _sha256(_REPOSITORY_ROOT / "bench/runner.py"),
                "scorer": "bench.orchestrate.rediscovery_from_runs",
                "scorer_sha256": _sha256(
                    _REPOSITORY_ROOT / "bench/orchestrate.py"
                ),
                "state_store": "core.state.StateStore (SQLite)",
            },
        },
        "cells": cells,
        "checks": checks,
    }
    _write_json(output_path, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic zero-network benchmark smoke at 1, 2, and 4 "
            "workers. This is not a scientific-quality model benchmark."
        )
    )
    parser.add_argument(
        "--output",
        default="artifacts/bench_offline_smoke.json",
        help="machine-readable JSON artifact path",
    )
    parser.add_argument(
        "--work-dir",
        default="artifacts/bench_offline_smoke_runs",
        help="directory for per-cell SQLite provenance",
    )
    parser.add_argument("--max-tasks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goals", default=str(DEFAULT_GOALSET))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = asyncio.run(
            run_offline_smoke(
                output_path=args.output,
                work_dir=args.work_dir,
                max_tasks=args.max_tasks,
                seed=args.seed,
                goalset_path=args.goals,
            )
        )
    except (OfflineBenchmarkSmokeError, ValueError) as exc:
        print(f"offline benchmark smoke failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
