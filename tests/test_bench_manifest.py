from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path

import aiosqlite
import pytest

from bench.goalset import BenchGoal
from bench.manifest import (
    Manifest,
    goal_fingerprint,
    make_run_cache_key,
    system_version,
)
from bench.runner import BenchmarkExecution, read_run
from bench.judge import absolute_scorer_identity
from core.models import AbsoluteScore, Hypothesis, ResearchPlanConfig
from core.state import StateStore


def _execution(**overrides) -> BenchmarkExecution:
    values = {
        "provider": "provider-x",
        "model_fast": "fast-x",
        "model_strong": "strong-x",
        "workers": 2,
        "max_tasks": 2,
        "max_time_seconds": 30,
        "config_sha256": "config-sha",
        "web_search_enabled": False,
        "evidence_max_claims": 4,
        "literature_grounded": True,
    }
    values.update(overrides)
    return BenchmarkExecution(**values)


def _absolute_execution(**overrides) -> BenchmarkExecution:
    scorer = absolute_scorer_identity(provider="judge-provider", model="judge-v1")
    scorer_json = overrides.pop(
        "absolute_scorer_json",
        json.dumps(scorer, sort_keys=True, separators=(",", ":")),
    )
    return _execution(
        ranking_mode="absolute",
        absolute_scorer_json=scorer_json,
        **overrides,
    )


async def _captured(
    tmp_path: Path,
    *,
    goal: BenchGoal | None = None,
    execution: BenchmarkExecution | None = None,
    variant: str = "full",
):
    goal = goal or BenchGoal(
        id="goal-1",
        goal="Find a testable mechanism.",
        gold_entities=["GENE1"],
        metadata={"dataset_revision": "v1"},
    )
    execution = execution or _execution()
    db_path = tmp_path / "captured.db"
    store = StateStore(str(db_path))
    await store.init_db()
    run_id = "run-1"
    await store.save_config(ResearchPlanConfig(
        run_id=run_id,
        goal=goal.goal,
        preferences="p",
        attributes=["Novelty"],
        constraints="c",
        safety_approved=True,
    ))
    await store.save_hypothesis(Hypothesis(
        id="hyp-1",
        run_id=run_id,
        text="GENE1 controls the phenotype.",
        summary="GENE1 mechanism",
        generation_method="debate",
        source="system",
        elo_rating=1200.0 if execution.ranking_mode == "absolute" else 1234.0,
    ))
    absolute_scorer = (
        json.loads(execution.absolute_scorer_json)
        if execution.ranking_mode == "absolute"
        else {}
    )
    if execution.ranking_mode == "absolute":
        await store.save_absolute_scorer(run_id, absolute_scorer)
        claim = await store.claim_absolute_score(
            "hyp-1", absolute_scorer["scorer_id"]
        )
        await store.publish_absolute_score(AbsoluteScore(
            run_id=run_id,
            hypothesis_id="hyp-1",
            scorer_id=absolute_scorer["scorer_id"],
            axis_scores={
                "novelty": 4,
                "feasibility": 3,
                "correctness": 5,
                "impact": 4,
            },
            total=4.0,
        ), owner_token=claim.owner_token)
    trace = [
        {
            "priority": index,
            "agent_type": "generation",
            "persona": "",
            "strategy": "debate",
            "status": "succeeded",
            "error": None,
        }
        for index in range(execution.max_tasks)
    ]
    run = await read_run(
        str(db_path),
        run_id=run_id,
        goal_id=goal.id,
        variant=variant,
        n_llm_calls=3,
        wall_clock_s=0.25,
        agent_task_counts={"generation": execution.max_tasks},
        agent_task_trace=trace,
        workers=execution.workers,
        backend=execution.provider,
        models={
            "fast": execution.model_fast,
            "strong": execution.model_strong,
        },
        max_backend_concurrency=execution.workers,
        estimated_tokens=123,
        expected_goal=goal.goal,
        expected_hypotheses=1,
        ranking_mode=execution.ranking_mode,
        absolute_scorer=absolute_scorer,
    )
    key = make_run_cache_key(
        goal,
        variant=variant,
        system_version_value="source-v1",
        seed=7,
        execution=execution,
    )
    return goal, execution, key, run, db_path


@pytest.mark.asyncio
async def test_completed_run_round_trips_without_backend_calls(tmp_path: Path):
    goal, execution, key, run, _ = await _captured(tmp_path)
    manifest = Manifest(str(tmp_path / "nested" / "manifest.sqlite"))
    await manifest.init()

    miss = await manifest.load_complete(
        key,
        expected_goal=goal.goal,
        expected_tasks=execution.max_tasks,
    )
    assert miss.state == "miss"

    await manifest.record_complete(
        key,
        run,
        expected_goal=goal.goal,
        expected_tasks=execution.max_tasks,
    )
    hit = await manifest.load_complete(
        key,
        expected_goal=goal.goal,
        expected_tasks=execution.max_tasks,
    )

    assert hit.state == "hit"
    assert hit.run == run
    assert hit.cache_key == key.id
    assert (tmp_path / "nested" / "manifest.sqlite").is_file()


@pytest.mark.asyncio
async def test_missing_backing_db_is_stale_and_never_recreated(tmp_path: Path):
    goal, execution, key, run, db_path = await _captured(tmp_path)
    manifest = Manifest(str(tmp_path / "manifest.sqlite"))
    await manifest.init()
    await manifest.record_complete(
        key,
        run,
        expected_goal=goal.goal,
        expected_tasks=execution.max_tasks,
    )
    db_path.unlink()

    lookup = await manifest.load_complete(
        key,
        expected_goal=goal.goal,
        expected_tasks=execution.max_tasks,
    )
    repeated = await manifest.load_complete(
        key,
        expected_goal=goal.goal,
        expected_tasks=execution.max_tasks,
    )

    assert lookup.state == repeated.state == "stale"
    assert "does not exist" in lookup.reason
    assert not db_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["zero_bytes", "same_count_mutation"])
async def test_corrupt_or_mutated_run_db_is_never_reused(
    tmp_path: Path,
    corruption: str,
):
    goal, execution, key, run, db_path = await _captured(tmp_path)
    manifest = Manifest(str(tmp_path / "manifest.sqlite"))
    await manifest.init()
    await manifest.record_complete(
        key,
        run,
        expected_goal=goal.goal,
        expected_tasks=execution.max_tasks,
    )

    if corruption == "zero_bytes":
        db_path.write_bytes(b"")
    else:
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "UPDATE hypotheses SET text='mutated' WHERE id='hyp-1'"
            )
            await db.commit()

    lookup = await manifest.load_complete(
        key,
        expected_goal=goal.goal,
        expected_tasks=execution.max_tasks,
    )
    assert lookup.state == "stale"
    assert lookup.run is None


@pytest.mark.asyncio
async def test_malformed_manifest_metadata_is_marked_stale(tmp_path: Path):
    goal, execution, key, run, _ = await _captured(tmp_path)
    manifest = Manifest(str(tmp_path / "manifest.sqlite"))
    await manifest.init()
    await manifest.record_complete(
        key,
        run,
        expected_goal=goal.goal,
        expected_tasks=execution.max_tasks,
    )
    async with aiosqlite.connect(manifest.path) as db:
        await db.execute("UPDATE manifest_v3 SET metadata_json='{'")
        await db.commit()

    lookup = await manifest.load_complete(
        key,
        expected_goal=goal.goal,
        expected_tasks=execution.max_tasks,
    )
    assert lookup.state == "stale"
    assert "JSONDecodeError" in lookup.reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda run: run.hypotheses.clear(), "no active hypotheses"),
        (
            # IMPORTANT 6: a fully-quarantined run completed cleanly -- it must
            # not be reported with the same generic message as "nothing was
            # ever generated" (which reads like a crash).
            lambda run: (run.hypotheses.clear(), setattr(run, "n_quarantined", 3)),
            "3 generated hypothesis.*quarantined by verification",
        ),
        (
            lambda run: run.agent_failures.append(
                {"agent_type": "generation", "error": "failed"}
            ),
            "task failure",
        ),
        (lambda run: run.agent_task_trace.pop(), "incomplete task trace"),
        (
            lambda run: run.agent_task_trace[0].update(status="failed"),
            "non-success",
        ),
        (lambda run: setattr(run, "workers", 9), "worker mismatch"),
        (lambda run: setattr(run, "backend", "other"), "provider mismatch"),
        (
            lambda run: setattr(
                run,
                "models",
                {"fast": "other", "strong": "strong-x"},
            ),
            "model mismatch",
        ),
    ],
)
async def test_incomplete_or_mismatched_runs_are_never_recorded(
    tmp_path: Path,
    mutation,
    message: str,
):
    goal, execution, key, original, _ = await _captured(tmp_path)
    run = copy.deepcopy(original)
    mutation(run)
    manifest = Manifest(str(tmp_path / "manifest.sqlite"))
    await manifest.init()

    with pytest.raises(ValueError, match=message):
        await manifest.record_complete(
            key,
            run,
            expected_goal=goal.goal,
            expected_tasks=execution.max_tasks,
        )

    lookup = await manifest.load_complete(
        key,
        expected_goal=goal.goal,
        expected_tasks=execution.max_tasks,
    )
    assert lookup.state == "miss"


@pytest.mark.asyncio
async def test_prompt_and_execution_changes_miss_but_scoring_changes_reuse(
    tmp_path: Path,
):
    goal, execution, key, run, _ = await _captured(tmp_path)
    manifest = Manifest(str(tmp_path / "manifest.sqlite"))
    await manifest.init()
    await manifest.record_complete(
        key,
        run,
        expected_goal=goal.goal,
        expected_tasks=execution.max_tasks,
    )

    changed_scoring = copy.deepcopy(goal)
    changed_scoring.gold_entities.append("GENE2")
    changed_scoring.metadata["dataset_revision"] = "v2"
    changed_scoring_key = make_run_cache_key(
        changed_scoring,
        variant="full",
        system_version_value="source-v1",
        seed=7,
        execution=execution,
    )
    changed_prompt = copy.deepcopy(goal)
    changed_prompt.goal = "Find a different testable mechanism."
    changed_prompt_key = make_run_cache_key(
        changed_prompt,
        variant="full",
        system_version_value="source-v1",
        seed=7,
        execution=execution,
    )
    changed_execution = _execution(model_fast="fast-y")
    changed_execution_key = make_run_cache_key(
        goal,
        variant="full",
        system_version_value="source-v1",
        seed=7,
        execution=changed_execution,
    )

    assert goal_fingerprint(changed_scoring) == goal_fingerprint(goal)
    assert (
        await manifest.load_complete(
            changed_scoring_key,
            expected_goal=changed_scoring.goal,
            expected_tasks=execution.max_tasks,
        )
    ).state == "hit"
    assert goal_fingerprint(changed_prompt) != goal_fingerprint(goal)
    assert (
        await manifest.load_complete(
            changed_prompt_key,
            expected_goal=changed_prompt.goal,
            expected_tasks=execution.max_tasks,
        )
    ).state == "miss"
    assert (
        await manifest.load_complete(
            changed_execution_key,
            expected_goal=goal.goal,
            expected_tasks=changed_execution.max_tasks,
        )
    ).state == "miss"


@pytest.mark.asyncio
async def test_absolute_execution_key_includes_mode_scorer_and_domain(tmp_path: Path):
    goal = BenchGoal(id="goal-absolute", goal="Find a mechanism.", domain="genomics")
    elo = _execution()
    absolute = _absolute_execution()
    changed_scorer = _absolute_execution(
        absolute_scorer_json=json.dumps(
            absolute_scorer_identity(
                provider="judge-provider", model="judge-v2"
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    changed_domain = copy.deepcopy(goal)
    changed_domain.domain = "immunology"

    def key(target_goal, execution):
        return make_run_cache_key(
            target_goal,
            variant="no_tournament",
            system_version_value="source-v1",
            seed=7,
            execution=execution,
        )

    assert key(goal, elo).id != key(goal, absolute).id
    assert key(goal, absolute).id != key(goal, changed_scorer).id
    assert key(goal, absolute).id != key(changed_domain, absolute).id
    # Domain is not a model input for an Elo-only system run, so legacy reuse
    # remains valid there.
    assert key(goal, elo).id == key(changed_domain, elo).id


@pytest.mark.asyncio
async def test_cached_absolute_run_round_trips_without_backend(tmp_path: Path):
    execution = _absolute_execution()
    goal, _, key, run, _ = await _captured(
        tmp_path, execution=execution, variant="no_tournament"
    )
    manifest = Manifest(str(tmp_path / "manifest.sqlite"))
    await manifest.init()

    await manifest.record_complete(
        key,
        run,
        expected_goal=goal.goal,
        expected_tasks=execution.max_tasks,
    )
    hit = await manifest.load_complete(
        key,
        expected_goal=goal.goal,
        expected_tasks=execution.max_tasks,
    )

    assert hit.state == "hit"
    assert hit.run == run
    assert hit.run.ranking_mode == "absolute"
    assert hit.run.hypotheses[0].absolute_score == 4.0
    assert hit.run.hypotheses[0].elo_rating == 1200.0
    assert hit.run.hypotheses[0].elo_trajectory == []


@pytest.mark.asyncio
async def test_manifest_rejects_run_whose_ranking_mode_mismatches_key(
    tmp_path: Path,
):
    goal, execution, _, run, _ = await _captured(tmp_path)
    absolute_key = make_run_cache_key(
        goal,
        variant="full",
        system_version_value="source-v1",
        seed=7,
        execution=_absolute_execution(),
    )
    manifest = Manifest(str(tmp_path / "manifest.sqlite"))
    await manifest.init()

    with pytest.raises(ValueError, match="ranking mode mismatch"):
        await manifest.record_complete(
            absolute_key,
            run,
            expected_goal=goal.goal,
            expected_tasks=execution.max_tasks,
        )


@pytest.mark.asyncio
async def test_absolute_score_tampering_marks_cache_stale(tmp_path: Path):
    execution = _absolute_execution()
    goal, _, key, run, db_path = await _captured(
        tmp_path, execution=execution, variant="no_tournament"
    )
    manifest = Manifest(str(tmp_path / "manifest.sqlite"))
    await manifest.init()
    await manifest.record_complete(
        key,
        run,
        expected_goal=goal.goal,
        expected_tasks=execution.max_tasks,
    )
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE absolute_scores SET total=3.5 WHERE hypothesis_id='hyp-1'"
        )
        await db.commit()

    lookup = await manifest.load_complete(
        key,
        expected_goal=goal.goal,
        expected_tasks=execution.max_tasks,
    )
    assert lookup.state == "stale"
    assert "inconsistent total" in lookup.reason


@pytest.mark.asyncio
async def test_stale_mark_is_compare_and_set_safe(tmp_path: Path):
    goal, execution, key, run, _ = await _captured(tmp_path)
    manifest = Manifest(str(tmp_path / "manifest.sqlite"))
    await manifest.init()
    await manifest.record_complete(
        key,
        run,
        expected_goal=goal.goal,
        expected_tasks=execution.max_tasks,
    )

    marked = await manifest._mark_stale(
        key,
        "validation result from an obsolete reader",
        observed_run_id=run.run_id,
        observed_run_fingerprint="obsolete-fingerprint",
    )

    assert marked is False
    lookup = await manifest.load_complete(
        key,
        expected_goal=goal.goal,
        expected_tasks=execution.max_tasks,
    )
    assert lookup.state == "hit"


@pytest.mark.asyncio
async def test_lease_claim_is_atomic_and_does_not_persist_owner_token(
    tmp_path: Path,
):
    _, _, key, _, _ = await _captured(tmp_path)
    manifest_path = tmp_path / "manifest.sqlite"
    manifest = Manifest(str(manifest_path))
    await manifest.init()

    claims = await asyncio.gather(*(
        Manifest(str(manifest_path)).claim(key, lease_seconds=30)
        for _ in range(8)
    ))

    acquired = [claim for claim in claims if claim.state == "acquired"]
    waiting = [claim for claim in claims if claim.state == "waiting"]
    assert len(acquired) == 1
    assert len(waiting) == 7
    owner_token = acquired[0].owner_token
    assert owner_token
    assert owner_token not in repr(acquired[0])
    assert all(claim.owner_token is None for claim in waiting)
    assert all(
        claim.expires_at == acquired[0].expires_at for claim in waiting
    )

    async with aiosqlite.connect(manifest_path) as db:
        async with db.execute(
            """SELECT owner_token_digest, expires_at
               FROM manifest_leases_v1 WHERE cache_key=?""",
            (key.id,),
        ) as cursor:
            row = await cursor.fetchone()
    assert row is not None
    assert row[0] == hashlib.sha256(owner_token.encode()).hexdigest()
    assert row[0] != owner_token
    assert row[1] == acquired[0].expires_at

    repeated = await manifest.claim(
        key,
        lease_seconds=30,
        owner_token=owner_token,
    )
    assert repeated.state == "acquired"
    assert repeated.owner_token == owner_token
    assert repeated.expires_at >= acquired[0].expires_at


@pytest.mark.asyncio
async def test_completed_row_wins_claim_and_clears_obsolete_lease(
    tmp_path: Path,
):
    goal, execution, key, run, _ = await _captured(tmp_path)
    manifest = Manifest(str(tmp_path / "manifest.sqlite"))
    await manifest.init()
    acquired = await manifest.claim(key, lease_seconds=30)
    assert acquired.state == "acquired"
    assert acquired.owner_token

    await manifest.record_complete(
        key,
        run,
        expected_goal=goal.goal,
        expected_tasks=execution.max_tasks,
        owner_token=acquired.owner_token,
    )
    ready = await manifest.claim(key, lease_seconds=30)

    assert ready.state == "ready"
    assert ready.owner_token is None
    assert ready.expires_at is None
    assert not await manifest.renew(
        key,
        owner_token=acquired.owner_token,
        lease_seconds=30,
    )
    async with aiosqlite.connect(manifest.path) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM manifest_leases_v1 WHERE cache_key=?",
            (key.id,),
        ) as cursor:
            assert (await cursor.fetchone())[0] == 0


@pytest.mark.asyncio
async def test_stale_lease_owner_cannot_publish_after_takeover(tmp_path: Path):
    goal, execution, key, run, _ = await _captured(tmp_path)
    manifest = Manifest(str(tmp_path / "manifest.sqlite"))
    await manifest.init()
    first = await manifest.claim(key, lease_seconds=30)
    assert first.owner_token
    async with aiosqlite.connect(manifest.path) as db:
        await db.execute(
            "UPDATE manifest_leases_v1 SET expires_at=0 WHERE cache_key=?",
            (key.id,),
        )
        await db.commit()
    second = await manifest.claim(key, lease_seconds=30)
    assert second.state == "acquired"
    assert second.owner_token

    with pytest.raises(ValueError, match="owner_token is required"):
        await manifest.record_complete(
            key,
            run,
            expected_goal=goal.goal,
            expected_tasks=execution.max_tasks,
        )
    with pytest.raises(ValueError, match="active execution lease"):
        await manifest.record_complete(
            key,
            run,
            expected_goal=goal.goal,
            expected_tasks=execution.max_tasks,
            owner_token=first.owner_token,
        )
    assert (
        await manifest.load_complete(
            key,
            expected_goal=goal.goal,
            expected_tasks=execution.max_tasks,
        )
    ).state == "miss"

    await manifest.record_complete(
        key,
        run,
        expected_goal=goal.goal,
        expected_tasks=execution.max_tasks,
        owner_token=second.owner_token,
    )
    assert (
        await manifest.load_complete(
            key,
            expected_goal=goal.goal,
            expected_tasks=execution.max_tasks,
        )
    ).state == "hit"
    assert not await manifest.release(key, owner_token=second.owner_token)


@pytest.mark.asyncio
async def test_expired_lease_is_stolen_and_mutations_are_owner_guarded(
    tmp_path: Path,
):
    _, _, key, _, _ = await _captured(tmp_path)
    manifest = Manifest(str(tmp_path / "manifest.sqlite"))
    await manifest.init()
    first = await manifest.claim(key, lease_seconds=30)
    assert first.state == "acquired"
    assert first.owner_token

    assert not await manifest.renew(
        key,
        owner_token="not-the-owner",
        lease_seconds=30,
    )
    assert not await manifest.release(key, owner_token="not-the-owner")
    assert await manifest.renew(
        key,
        owner_token=first.owner_token,
        lease_seconds=30,
    )

    async with aiosqlite.connect(manifest.path) as db:
        await db.execute(
            "UPDATE manifest_leases_v1 SET expires_at=0 WHERE cache_key=?",
            (key.id,),
        )
        await db.commit()

    contenders = await asyncio.gather(*(
        Manifest(manifest.path).claim(key, lease_seconds=30)
        for _ in range(6)
    ))
    acquired = [claim for claim in contenders if claim.state == "acquired"]
    assert len(acquired) == 1
    second = acquired[0]
    assert second.owner_token
    assert second.owner_token != first.owner_token
    assert sum(claim.state == "waiting" for claim in contenders) == 5

    assert not await manifest.renew(
        key,
        owner_token=first.owner_token,
        lease_seconds=30,
    )
    assert not await manifest.release(key, owner_token=first.owner_token)
    assert await manifest.renew(
        key,
        owner_token=second.owner_token,
        lease_seconds=30,
    )
    assert await manifest.release(key, owner_token=second.owner_token)
    assert not await manifest.release(key, owner_token=second.owner_token)
    replacement = await manifest.claim(key, lease_seconds=30)
    assert replacement.state == "acquired"


@pytest.mark.asyncio
@pytest.mark.parametrize("lease_seconds", [0, -1, float("inf"), float("nan"), True])
async def test_invalid_lease_duration_is_rejected(
    tmp_path: Path,
    lease_seconds,
):
    _, _, key, _, _ = await _captured(tmp_path)
    manifest = Manifest(str(tmp_path / "manifest.sqlite"))
    await manifest.init()

    with pytest.raises(ValueError, match="finite positive"):
        await manifest.claim(key, lease_seconds=lease_seconds)


@pytest.mark.asyncio
async def test_legacy_manifest_rows_are_not_treated_as_v3_hits(tmp_path: Path):
    path = tmp_path / "manifest.sqlite"
    async with aiosqlite.connect(path) as db:
        await db.execute(
            """CREATE TABLE manifest_v2 (
                   goal_id TEXT, variant TEXT, system_version TEXT, seed INTEGER,
                   execution_key TEXT, db_path TEXT, n_llm_calls INTEGER,
                   wall_clock_s REAL, status TEXT
               )"""
        )
        await db.execute(
            "INSERT INTO manifest_v2 VALUES (?,?,?,?,?,?,?,?,?)",
            ("goal-1", "full", "source-v1", 7, "", "old.db", 1, 1.0, "complete"),
        )
        await db.commit()
    goal = BenchGoal(id="goal-1", goal="Find a testable mechanism.")
    execution = _execution()
    key = make_run_cache_key(
        goal,
        variant="full",
        system_version_value="source-v1",
        seed=7,
        execution=execution,
    )
    manifest = Manifest(str(path))
    await manifest.init()

    lookup = await manifest.load_complete(
        key,
        expected_goal=goal.goal,
        expected_tasks=execution.max_tasks,
    )
    assert lookup.state == "miss"


def test_system_version_is_source_identity():
    version = system_version()
    assert isinstance(version, str) and len(version) >= 7


def test_system_version_binds_packaged_execution_source_digest(
    monkeypatch,
    tmp_path: Path,
):
    import bench.manifest as manifest_module

    source = tmp_path / "installed"
    (source / "bench").mkdir(parents=True)
    implementation = source / "bench" / "runner.py"
    implementation.write_text("value = 1\n")
    monkeypatch.setattr(manifest_module, "_SOURCE_ROOT", source)

    first = manifest_module.system_version()
    implementation.write_text("value = 2\n")
    second = manifest_module.system_version()

    assert first.startswith("execution-")
    assert second.startswith("execution-")
    assert first != second


def test_system_version_ignores_analysis_only_source_changes(
    monkeypatch,
    tmp_path: Path,
):
    import bench.manifest as manifest_module

    source = tmp_path / "source"
    (source / "bench").mkdir(parents=True)
    runner = source / "bench" / "runner.py"
    report = source / "bench" / "report.py"
    runner.write_text("execution = 1\n")
    report.write_text("analysis = 1\n")
    monkeypatch.setattr(manifest_module, "_SOURCE_ROOT", source)

    first = manifest_module.system_version()
    report.write_text("analysis = 2\n")
    assert manifest_module.system_version() == first
    runner.write_text("execution = 2\n")
    assert manifest_module.system_version() != first
