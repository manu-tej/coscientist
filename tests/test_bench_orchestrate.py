import asyncio
import json
import pytest
from pathlib import Path

from bench.orchestrate import concordance_from_runs
from bench.runner import BenchHypothesis, BenchRun


@pytest.fixture(autouse=True)
def _run_databases_in_tmp(tmp_path, monkeypatch):
    """benchmark_db_path() writes under ./bench_runs, which is where real run
    databases live in a working copy. Keep this module's runs in tmp_path."""
    monkeypatch.chdir(tmp_path)


def _run(goal_id, hyps):
    return BenchRun(goal_id=goal_id, variant="full", hypotheses=hyps,
                    n_llm_calls=10, wall_clock_s=1.0, db_path=":memory:")


def test_run_metadata_discloses_n_quarantined():
    """IMPORTANT 6: `n_hypotheses` in the per-run report dict only counts the
    active pool. Without n_quarantined alongside it, a hypothesis verification
    removed simply vanishes from every artifact that reads this dict --
    including the ones concordance/rediscovery recall are computed from."""
    from bench.orchestrate import _run_metadata

    run = _run("q1", [])
    run.n_quarantined = 3
    metadata = _run_metadata(run)
    assert metadata["n_quarantined"] == 3
    assert metadata["n_hypotheses"] == 0


def test_concordance_from_runs_scores_against_gold(tmp_path):
    from bench.goalset import BenchGoal
    goals = [BenchGoal(id="q1", goal="...", gold_answer="B"),
             BenchGoal(id="q2", goal="...", gold_answer="C")]
    runs = [
        _run("q1", [BenchHypothesis("h1", "Answer: B", "s", 1300.0, "t0"),
                    BenchHypothesis("h2", "Answer: A", "s", 1100.0, "t0")]),
        _run("q2", [BenchHypothesis("h3", "Answer: C", "s", 1280.0, "t0")]),
    ]
    ref = {"q1": 0.25, "q2": 0.25}
    stats = concordance_from_runs(goals, runs, ref, bin_width=50, min_support=1)
    assert stats["n_rows"] == 3
    assert "spearman_rho" in stats
    assert "blue_minus_red" in stats


def test_concordance_from_runs_stratifies_by_tournament_not_question(tmp_path):
    # Two tournaments on the same question are two strata: Elo is only
    # comparable within a tournament.
    from bench.goalset import BenchGoal
    goals = [BenchGoal(id="q1", goal="...", gold_answer="B")]
    hyps = lambda: [BenchHypothesis("h1", "Answer: B", "s", 1260.0, "t0"),
                    BenchHypothesis("h2", "Answer: A", "s", 1200.0, "t0"),
                    BenchHypothesis("h3", "Answer: B", "s", 1230.0, "t0"),
                    BenchHypothesis("h4", "Answer: A", "s", 1190.0, "t0")]
    first, second = _run("q1", hyps()), _run("q1", hyps())
    first.run_id, second.run_id = "run-1", "run-2"
    stats = concordance_from_runs(goals, [first, second], {"q1": 0.25},
                                  bin_width=50, min_support=1)
    assert stats["n_rows"] == 8
    assert stats["logistic_n_questions"] == 2


def test_rediscovery_from_runs_recall_and_elo_concordance():
    from bench.orchestrate import rediscovery_from_runs
    from bench.goalset import BenchGoal
    goals = [BenchGoal(id="t1", goal="...", gold_entities=["SPP1", "FAP", "LAYN"])]
    # Elo ranks track gold-entity content: high holds 2, mid holds 1, low holds 0.
    runs = [_run("t1", [
        BenchHypothesis("h1", "SPP1+ macrophages and FAP+ fibroblasts dominate", "s", 1300.0, "t0"),
        BenchHypothesis("h2", "LAYN marks exhausted T cells", "s", 1200.0, "t0"),
        BenchHypothesis("h3", "something unrelated", "s", 1100.0, "t0"),
    ])]
    r = rediscovery_from_runs(goals, runs)
    assert r["per_goal"][0]["pool_recall"] == 1.0        # SPP1, FAP, LAYN all surfaced
    assert r["per_goal"][0]["top1_recall"] == 2 / 3      # top hyp holds SPP1 + FAP
    assert r["n_hyps_scored"] == 3
    # higher-Elo hypotheses hold more gold entities → positive Elo↔recall
    assert r["elo_recall_spearman"] > 0


def test_rediscovery_uses_absolute_scores_without_elo_fallback():
    from bench.orchestrate import rediscovery_from_runs
    from bench.goalset import BenchGoal

    goals = [BenchGoal(id="t1", goal="...", gold_entities=["SPP1"])]
    high_elo = BenchHypothesis(
        "elo", "unrelated", "unrelated", 2000.0, "t0", absolute_score=1.0
    )
    high_absolute = BenchHypothesis(
        "absolute", "SPP1 mechanism", "SPP1", 1000.0, "t0",
        absolute_score=5.0,
    )
    run = _run("t1", [high_elo, high_absolute])
    run.ranking_mode = "absolute"

    result = rediscovery_from_runs(goals, [run])

    assert result["ranking_mode"] == "absolute"
    assert result["per_goal"][0]["top1_recall"] == 1.0
    assert "elo_recall_spearman" not in result


def test_concordance_rejects_absolute_ranking_runs():
    from bench.errors import BenchError
    from bench.goalset import BenchGoal

    run = _run("q1", [
        BenchHypothesis(
            "h", "Answer: B", "s", 1200.0, "t0", absolute_score=4.0
        )
    ])
    run.ranking_mode = "absolute"

    with pytest.raises(BenchError, match="cannot consume absolute"):
        concordance_from_runs(
            [BenchGoal(id="q1", goal="...", gold_answer="B")],
            [run],
            {},
        )


class _Args:
    def __init__(self, **kw):
        self.limit = 5
        self.full = False
        self.backend = "api"
        self.yes = True
        self.command = "concordance"
        for k, v in kw.items():
            setattr(self, k, v)


@pytest.mark.asyncio
async def test_noncooperative_cancellation_keeps_renewing_until_backend_stops():
    import bench.orchestrate as orch

    suppressed = asyncio.Event()
    finish = asyncio.Event()

    class Manifest:
        renewals = 0

        async def renew(self, key, *, owner_token, lease_seconds):
            self.renewals += 1
            return True

    async def noncooperative_backend():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            suppressed.set()
            await finish.wait()
        return "late result"

    manifest = Manifest()
    counts = orch._new_cache_counts()
    task = asyncio.create_task(orch._run_with_lease_heartbeat(
        noncooperative_backend(),
        manifest=manifest,
        key=object(),
        owner_token="owner",
        timing=orch._LeaseTiming(
            lease_seconds=0.15,
            poll_seconds=0.01,
            hard_timeout_seconds=0.02,
            wait_timeout_seconds=1.0,
        ),
        cache_counts=counts,
    ))

    await suppressed.wait()
    await asyncio.sleep(0.12)
    assert task.done() is False
    assert manifest.renewals >= 1
    finish.set()
    with pytest.raises(TimeoutError, match="cooperative owner timeout"):
        await task
    assert counts["lease_timeouts"] == 1


@pytest.mark.asyncio
async def test_run_command_unwired_command_fails_closed(capsys, monkeypatch):
    import bench.orchestrate as orch
    # An advertised-but-unwired command must never look green in CI.
    rc = await orch.run_command(_Args(yes=True, command="baseline"))
    assert rc == 2
    out = capsys.readouterr().out
    assert "unsupported benchmark command" in out.lower()


@pytest.mark.asyncio
async def test_run_command_decline_returns_one(monkeypatch):
    import bench.orchestrate as orch
    import bench.cli as cli
    monkeypatch.setattr(cli, "_confirm", lambda est, backend, assume_yes: False)
    rc = await orch.run_command(_Args(yes=False))
    assert rc == 1


@pytest.mark.asyncio
async def test_auto_billing_loads_dotenv_and_pins_effective_provider(
    monkeypatch,
):
    import bench.cli as cli
    import bench.orchestrate as orch
    import tools.llm as llm

    def load_test_environment():
        monkeypatch.setenv("COSCIENTIST_PROVIDER", "anthropic")

    captured = {}

    def decline(est, backend, assume_yes):
        captured["backend"] = backend
        return False

    monkeypatch.setattr(llm, "load_backend_environment", load_test_environment)
    monkeypatch.setattr(cli, "_confirm", decline)
    args = _Args(command="rediscover", backend="auto", provider=None, yes=False)

    assert await orch.run_command(args) == 1
    assert captured["backend"] == "api"
    assert args.provider == "anthropic"


@pytest.mark.asyncio
async def test_auto_billing_includes_separate_absolute_judge_provider(
    monkeypatch,
):
    import bench.cli as cli
    import bench.orchestrate as orch

    captured = {}

    def decline(est, backend, assume_yes):
        captured["backend"] = backend
        return False

    monkeypatch.setattr(cli, "_confirm", decline)
    args = _Args(
        command="rediscover",
        backend="auto",
        provider="claude-code",
        judge_provider="openai-compatible",
        ranking_mode="absolute",
        yes=False,
    )

    assert await orch.run_command(args) == 1
    assert captured["backend"] == "mixed"
    assert args.provider == "claude-code"


@pytest.mark.asyncio
async def test_auto_billing_labels_mixed_sweep_before_confirmation(monkeypatch):
    import bench.cli as cli
    import bench.orchestrate as orch

    captured = {}

    def decline(est, backend, assume_yes):
        captured["backend"] = backend
        return False

    monkeypatch.setattr(cli, "_confirm", decline)
    args = _Args(
        command="sweep",
        backend="auto",
        yes=False,
        worker_counts="1",
        models=["codex=gpt-5", "anthropic=claude-sonnet"],
        seeds="0",
    )

    assert await orch.run_command(args) == 1
    assert captured["backend"] == "mixed"


@pytest.mark.asyncio
async def test_manual_billing_label_warns_when_adapter_route_differs(
    capsys, monkeypatch,
):
    import bench.cli as cli
    import bench.orchestrate as orch

    monkeypatch.setattr(cli, "_confirm", lambda *args, **kwargs: False)
    args = _Args(
        command="rediscover",
        backend="api",
        provider="codex",
        yes=False,
    )

    assert await orch.run_command(args) == 1
    assert "overrides the 'subscription' route" in capsys.readouterr().out


class _ReferenceBackend:
    def __init__(self, *, block=False):
        import asyncio

        self.block = block
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def call(self, system, user, **kwargs):
        self.started.set()
        if self.block:
            await self.release.wait()
        return "Answer: B"

    async def aclose(self):
        self.closed = True


def _reference_args():
    return _Args(
        command="concordance",
        dataset="gpqa-bio",
        limit=1,
        full=False,
        max_tasks=1,
        max_time=10,
        min_support=1,
        seed=0,
        workers=1,
        provider="anthropic",
        model_fast="fast",
        model_strong="strong",
        reference_samples=1,
        reference_strong=False,
        allow_fixture_fallback=False,
    )


def _patch_reference_run(monkeypatch, backend):
    import bench.manifest as manifest
    import bench.orchestrate as orch
    import bench.runner as runner
    import tools.llm as llm
    from bench.goalset import BenchGoal

    goal = BenchGoal(id="q1", goal="Question?", gold_answer="B")
    run = BenchRun(
        goal_id="q1",
        variant="full",
        hypotheses=[
            BenchHypothesis("h1", "Answer: B", "summary", 1200.0, "now"),
        ],
        n_llm_calls=2,
        wall_clock_s=0.1,
        db_path="unused.db",
        agent_task_counts={"generation": 1},
        agent_task_trace=[{
            "priority": 0,
            "agent_type": "generation",
            "persona": "",
            "status": "succeeded",
        }],
        workers=1,
        backend="anthropic",
        models={"fast": "", "strong": ""},
    )

    async def fake_run_system(*args, **kwargs):
        return run

    monkeypatch.setattr(
        orch,
        "_load_concordance_goals",
        lambda *args, **kwargs: (
            [goal],
            {"requested": "gpqa-bio", "source": "test", "fixture_fallback": False},
        ),
    )
    monkeypatch.setattr(runner, "run_system", fake_run_system)
    monkeypatch.setattr(runner, "_load_yaml_config", lambda path: {})
    monkeypatch.setattr(
        runner,
        "configure_backend",
        lambda cfg, **kwargs: ("anthropic", {}),
    )
    monkeypatch.setattr(llm, "make_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(manifest, "system_version", lambda: "test")


@pytest.mark.asyncio
async def test_reference_backend_is_closed_and_calls_are_counted(monkeypatch):
    import bench.orchestrate as orch

    backend = _ReferenceBackend()
    _patch_reference_run(monkeypatch, backend)

    report = await orch.run_concordance(_reference_args())

    assert backend.closed is True
    assert report["meta"]["reference_llm_calls"] == 1
    assert report["meta"]["n_llm_calls"] == 3


@pytest.mark.asyncio
async def test_reference_backend_is_closed_when_sampling_is_cancelled(monkeypatch):
    import asyncio
    import bench.orchestrate as orch

    backend = _ReferenceBackend(block=True)
    _patch_reference_run(monkeypatch, backend)
    task = asyncio.create_task(orch.run_concordance(_reference_args()))
    await backend.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert backend.closed is True


def _accounting_run(
    goal_id,
    *,
    calls,
    tokens,
    hypotheses,
    failures=None,
    workers=2,
    backend="codex",
    model_fast="fast",
    model_strong="strong",
):
    return BenchRun(
        goal_id=goal_id,
        variant="full",
        hypotheses=hypotheses,
        n_llm_calls=calls,
        wall_clock_s=0.1,
        db_path=f"{goal_id}.db",
        run_id=f"run-{goal_id}",
        agent_failures=failures or [],
        agent_task_counts={"generation": 1},
        agent_task_trace=[{
            "priority": 0,
            "agent_type": "generation",
            "persona": "",
            "status": "succeeded",
        }],
        workers=workers,
        backend=backend,
        models={"fast": model_fast, "strong": model_strong},
        estimated_tokens=tokens,
    )


@pytest.mark.asyncio
async def test_rediscovery_preserves_accounting_for_rejected_returned_runs(
    monkeypatch, tmp_path,
):
    import bench.manifest as manifest
    import bench.orchestrate as orch
    import bench.runner as runner

    rows = [
        {"id": "task-failure", "goal": "g1", "gold_entities": ["SPP1"]},
        {"id": "empty", "goal": "g2", "gold_entities": ["SPP1"]},
        {"id": "accepted", "goal": "g3", "gold_entities": ["SPP1"]},
        {"id": "exception", "goal": "g4", "gold_entities": ["SPP1"]},
    ]
    goalset = tmp_path / "goals.jsonl"
    goalset.write_text("\n".join(json.dumps(row) for row in rows))
    accepted_hypothesis = BenchHypothesis(
        "h-ok", "SPP1 is enriched", "SPP1 result", 1200.0, "now",
    )
    task_failure_hypothesis = BenchHypothesis(
        "h-fail", "SPP1 candidate", "candidate", 1200.0, "now",
    )
    returned = {
        "task-failure": _accounting_run(
            "task-failure",
            calls=7,
            tokens=70,
            hypotheses=[task_failure_hypothesis],
            failures=[{"agent_type": "generation", "error": "bad output"}],
        ),
        "empty": _accounting_run(
            "empty", calls=5, tokens=50, hypotheses=[],
        ),
        "accepted": _accounting_run(
            "accepted",
            calls=3,
            tokens=30,
            hypotheses=[accepted_hypothesis],
        ),
    }

    async def fake_run_system(goal, **kwargs):
        if goal.id == "exception":
            raise RuntimeError("provider disconnected")
        return returned[goal.id]

    monkeypatch.setattr(runner, "run_system", fake_run_system)
    monkeypatch.setattr(manifest, "system_version", lambda: "test")
    args = _Args(
        command="rediscover",
        goals=str(goalset),
        limit=4,
        max_tasks=1,
        max_time=10,
        seed=0,
        workers=2,
        provider="codex",
        model_fast="fast",
        model_strong="strong",
    )

    report = await orch.run_rediscovery(args)
    meta = report["meta"]

    assert report["rediscovery"]["n_goals"] == 1
    assert meta["n_attempted"] == 4
    assert meta["n_runs_returned"] == 3
    assert meta["n_accepted"] == meta["n_completed"] == 1
    assert meta["n_failed"] == 3
    assert meta["n_scorable"] == 1
    assert meta["n_unaccounted_attempts"] == 1
    assert meta["n_llm_calls"] == 15
    assert meta["estimated_tokens"] == 150
    assert [run["accepted_for_scoring"] for run in meta["runs"]] == [
        False,
        False,
        True,
    ]
    assert meta["runs"][0]["n_llm_calls"] == 7
    assert meta["runs"][1]["estimated_tokens"] == 50
    assert [failure["run_returned"] for failure in meta["failures"]] == [
        True,
        True,
        False,
    ]


@pytest.mark.asyncio
async def test_run_rediscovery_validates_entire_goalset_before_model_calls(
    monkeypatch, tmp_path,
):
    import bench.orchestrate as orch
    import bench.runner as runner
    from bench.errors import BenchError

    goalset = tmp_path / "invalid-goals.jsonl"
    goalset.write_text(
        "\n".join(
            [
                json.dumps(
                    {"id": "same", "goal": "valid", "gold_entities": ["A"]}
                ),
                json.dumps(
                    {"id": "same", "goal": "duplicate", "gold_entities": ["B"]}
                ),
            ]
        )
    )
    called = False

    async def fake_run_system(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("model runner must not be called")

    monkeypatch.setattr(runner, "run_system", fake_run_system)
    args = _Args(
        command="rediscover",
        goals=str(goalset),
        limit=1,
        max_tasks=1,
        seed=0,
    )

    with pytest.raises(BenchError, match="duplicate id"):
        await orch.run_rediscovery(args)
    assert called is False


@pytest.mark.asyncio
async def test_concordance_preserves_accounting_for_rejected_returned_runs(
    monkeypatch,
):
    import bench.manifest as manifest
    import bench.orchestrate as orch
    import bench.runner as runner
    from bench.goalset import BenchGoal

    goals = [
        BenchGoal(id="task-failure", goal="q1", gold_answer="B"),
        BenchGoal(id="empty", goal="q2", gold_answer="B"),
        BenchGoal(id="accepted", goal="q3", gold_answer="B"),
    ]
    returned = {
        "task-failure": _accounting_run(
            "task-failure",
            calls=4,
            tokens=40,
            hypotheses=[
                BenchHypothesis(
                    "h-fail", "Answer: B", "candidate", 1200.0, "now",
                )
            ],
            failures=[{"agent_type": "ranking", "error": "bad winner"}],
        ),
        "empty": _accounting_run(
            "empty", calls=3, tokens=30, hypotheses=[],
        ),
        "accepted": _accounting_run(
            "accepted",
            calls=2,
            tokens=20,
            hypotheses=[
                BenchHypothesis(
                    "h-ok", "Answer: B", "accepted", 1200.0, "now",
                )
            ],
        ),
    }

    async def fake_run_system(goal, **kwargs):
        return returned[goal.id]

    monkeypatch.setattr(
        orch,
        "_load_concordance_goals",
        lambda *args, **kwargs: (
            goals,
            {"requested": "test", "source": "test", "fixture_fallback": False},
        ),
    )
    monkeypatch.setattr(runner, "run_system", fake_run_system)
    monkeypatch.setattr(manifest, "system_version", lambda: "test")
    args = _Args(
        command="concordance",
        dataset="gpqa-bio",
        limit=3,
        max_tasks=1,
        max_time=10,
        min_support=1,
        seed=0,
        workers=2,
        provider="codex",
        model_fast="fast",
        model_strong="strong",
        reference_samples=0,
        reference_strong=False,
        allow_fixture_fallback=False,
    )

    report = await orch.run_concordance(args)
    meta = report["meta"]

    assert report["concordance"]["n_rows"] == 1
    assert meta["n_attempted"] == meta["n_runs_returned"] == 3
    assert meta["n_accepted"] == meta["n_completed"] == 1
    assert meta["n_failed"] == 2
    assert meta["n_scorable"] == 1
    assert meta["n_unaccounted_attempts"] == 0
    assert meta["n_llm_calls"] == 9
    assert meta["estimated_tokens"] == 90
    assert [run["accepted_for_scoring"] for run in meta["runs"]] == [
        False,
        False,
        True,
    ]


@pytest.mark.asyncio
async def test_run_command_rejects_zero_scorable_rediscovery_even_when_partial_allowed(
    monkeypatch, tmp_path, capsys,
):
    import bench.cli as cli
    import bench.orchestrate as orch

    async def fake_rediscovery(args):
        return {
            "rediscovery": {
                "per_goal": [],
                "n_goals": 0,
                "mean_pool_recall": 0.0,
                "mean_top1_recall": 0.0,
                "mean_top3_recall": 0.0,
                "elo_recall_spearman": float("nan"),
                "elo_recall_p": float("nan"),
                "n_hyps_scored": 0,
            },
            "meta": {
                "n_completed": 1,
                "n_accepted": 1,
                "n_failed": 0,
                "n_scorable": 0,
            },
            "verdicts": {},
        }

    monkeypatch.setattr(cli, "_confirm", lambda *args, **kwargs: True)
    monkeypatch.setattr(orch, "run_rediscovery", fake_rediscovery)
    args = _Args(
        command="rediscover",
        backend="subscription",
        provider="codex",
        out=str(tmp_path / "zero-scorable"),
        max_tasks=1,
        allow_partial=True,
    )

    assert await orch.run_command(args) == 2
    assert "scorable=0" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_run_command_prints_absolute_rediscovery_without_elo_keys(
    monkeypatch, tmp_path, capsys,
):
    import bench.cli as cli
    import bench.orchestrate as orch

    async def fake_rediscovery(args):
        return {
            "rediscovery": {
                "ranking_mode": "absolute",
                "per_goal": [{
                    "goal_id": "g",
                    "pool_recall": 1.0,
                    "top1_recall": 1.0,
                    "top3_recall": 1.0,
                    "n_hyps": 1,
                    "n_gold": 1,
                }],
                "n_goals": 1,
                "mean_pool_recall": 1.0,
                "mean_top1_recall": 1.0,
                "mean_top3_recall": 1.0,
                "ranking_recall_spearman": 0.5,
                "ranking_recall_p": 0.1,
                "n_hyps_scored": 1,
            },
            "meta": {
                "n_completed": 1,
                "n_accepted": 1,
                "n_failed": 0,
                "n_scorable": 1,
            },
            "verdicts": {},
        }

    monkeypatch.setattr(cli, "_confirm", lambda *args, **kwargs: True)
    monkeypatch.setattr(orch, "run_rediscovery", fake_rediscovery)
    args = _Args(
        command="rediscover",
        backend="subscription",
        provider="codex",
        ranking_mode="absolute",
        out=str(tmp_path / "absolute-output"),
        max_tasks=1,
    )

    assert await orch.run_command(args) == 0
    output = capsys.readouterr().out
    assert "Absolute score↔recall" in output
    assert "Elo↔recall" not in output


async def _write_complete_benchmark_run(
    goal,
    *,
    db_path,
    max_tasks,
    n_workers,
    provider,
    model_fast,
    model_strong,
    ordinal,
    text,
    calls=4,
    tokens=40,
):
    from bench.runner import read_run
    from core.models import Hypothesis, ResearchPlanConfig
    from core.state import StateStore

    run_id = f"cached-run-{ordinal}"
    store = StateStore(db_path)
    await store.init_db()
    await store.save_config(ResearchPlanConfig(
        run_id=run_id,
        goal=goal.goal,
        preferences="test",
        attributes=["Novelty"],
        constraints="test",
        safety_approved=True,
    ))
    await store.save_hypothesis(Hypothesis(
        id=f"cached-hypothesis-{ordinal}",
        run_id=run_id,
        text=text,
        summary=text,
        generation_method="generation",
        source="system",
        elo_rating=1234.0,
    ))
    trace = [
        {
            "priority": index,
            "agent_type": "generation",
            "persona": "",
            "status": "succeeded",
        }
        for index in range(max_tasks)
    ]
    return await read_run(
        db_path,
        run_id=run_id,
        goal_id=goal.id,
        variant="full",
        n_llm_calls=calls,
        wall_clock_s=0.1,
        agent_task_counts={"generation": max_tasks},
        agent_task_trace=trace,
        workers=n_workers,
        backend=provider,
        models={"fast": model_fast, "strong": model_strong},
        max_backend_concurrency=n_workers,
        estimated_tokens=tokens,
        expected_goal=goal.goal,
        expected_hypotheses=1,
    )


@pytest.mark.asyncio
async def test_rediscovery_cache_reports_miss_hit_and_stale_without_call_loss(
    monkeypatch, tmp_path,
):
    import bench.manifest as manifest
    import bench.orchestrate as orch
    import bench.runner as runner

    goalset = tmp_path / "goals.jsonl"
    goalset.write_text(json.dumps({
        "id": "cached-goal",
        "goal": "Find an SPP1 mechanism",
        "gold_entities": ["SPP1"],
    }))
    calls = []

    async def fake_run_system(goal, **kwargs):
        calls.append(kwargs)
        return await _write_complete_benchmark_run(
            goal,
            ordinal=len(calls),
            text="SPP1 is the candidate mechanism",
            **{
                name: kwargs[name]
                for name in (
                    "db_path",
                    "max_tasks",
                    "n_workers",
                    "provider",
                    "model_fast",
                    "model_strong",
                )
            },
        )

    monkeypatch.setattr(runner, "run_system", fake_run_system)
    monkeypatch.setattr(manifest, "system_version", lambda: "cache-test-v1")
    args = _Args(
        command="rediscover",
        goals=str(goalset),
        limit=1,
        max_tasks=1,
        max_time=10,
        seed=0,
        workers=2,
        provider="codex",
        model_fast="fast",
        model_strong="strong",
        manifest=str(tmp_path / "manifest.sqlite"),
        no_cache=False,
    )

    miss = await orch.run_rediscovery(args)
    hit = await orch.run_rediscovery(args)

    assert len(calls) == 1
    assert miss["meta"]["cache"]["misses"] == 1
    assert miss["meta"]["fresh_llm_calls"] == 4
    assert miss["meta"]["reused_llm_calls"] == 0
    assert miss["meta"]["invocation_llm_calls"] == 4
    assert miss["meta"]["runs"][0]["cache_state"] == "miss"
    assert miss["meta"]["runs"][0]["execution_source"] == "fresh"

    assert hit["meta"]["cache"]["hits"] == 1
    assert hit["meta"]["n_llm_calls"] == 4
    assert hit["meta"]["fresh_llm_calls"] == 0
    assert hit["meta"]["reused_llm_calls"] == 4
    assert hit["meta"]["invocation_llm_calls"] == 0
    assert hit["meta"]["runs"][0]["cache_state"] == "hit"
    assert hit["meta"]["runs"][0]["execution_source"] == "reused"

    cached_db = Path(hit["meta"]["runs"][0]["db_path"])
    cached_db.write_bytes(b"deliberately corrupt sqlite evidence")
    stale = await orch.run_rediscovery(args)

    assert len(calls) == 2
    assert stale["meta"]["cache"]["stale"] == 1
    assert stale["meta"]["fresh_llm_calls"] == 4
    assert stale["meta"]["reused_llm_calls"] == 0
    assert stale["meta"]["runs"][0]["cache_state"] == "stale"
    assert stale["meta"]["runs"][0]["cache_stale_reason"]
    assert stale["meta"]["runs"][0]["db_path"] != str(cached_db)


@pytest.mark.asyncio
async def test_concurrent_rediscovery_misses_coalesce_to_one_paid_run(
    monkeypatch, tmp_path,
):
    import bench.manifest as manifest
    import bench.orchestrate as orch
    import bench.runner as runner

    goalset = tmp_path / "goals.jsonl"
    goalset.write_text(json.dumps({
        "id": "single-flight-goal",
        "goal": "Find an SPP1 mechanism",
        "gold_entities": ["SPP1"],
    }))
    started = asyncio.Event()
    allow_finish = asyncio.Event()
    calls = 0

    async def fake_run_system(goal, **kwargs):
        nonlocal calls
        calls += 1
        ordinal = calls
        started.set()
        await allow_finish.wait()
        return await _write_complete_benchmark_run(
            goal,
            ordinal=ordinal,
            text="SPP1 is the candidate mechanism",
            **{
                name: kwargs[name]
                for name in (
                    "db_path",
                    "max_tasks",
                    "n_workers",
                    "provider",
                    "model_fast",
                    "model_strong",
                )
            },
        )

    monkeypatch.setattr(runner, "run_system", fake_run_system)
    monkeypatch.setattr(manifest, "system_version", lambda: "lease-test-v1")

    def make_args():
        return _Args(
            command="rediscover",
            goals=str(goalset),
            limit=1,
            max_tasks=1,
            max_time=10,
            seed=0,
            workers=1,
            provider="codex",
            model_fast="fast",
            model_strong="strong",
            manifest=str(tmp_path / "manifest.sqlite"),
            no_cache=False,
            _lease_seconds=0.3,
            _lease_poll_seconds=0.05,
            _lease_hard_timeout_seconds=30.0,
            _lease_wait_timeout_seconds=35.0,
        )

    owner_task = asyncio.create_task(orch.run_rediscovery(make_args()))
    await started.wait()
    waiter_task = asyncio.create_task(orch.run_rediscovery(make_args()))
    # Keep the owner alive past one heartbeat. The waiter must still not start
    # a second model-backed execution.
    await asyncio.sleep(0.35)
    assert calls == 1
    allow_finish.set()
    owner, waiter = await asyncio.gather(owner_task, waiter_task)

    assert calls == 1
    assert owner["meta"]["n_fresh_runs"] == 1
    assert owner["meta"]["cache"]["lease_renewals"] >= 1
    assert waiter["meta"]["n_fresh_runs"] == 0
    assert waiter["meta"]["n_reused_runs"] == 1
    assert waiter["meta"]["fresh_llm_calls"] == 0
    assert waiter["meta"]["cache"]["lease_waits"] == 1
    assert waiter["meta"]["cache"]["coalesced_runs"] == 1
    assert waiter["meta"]["runs"][0]["lease_waited"] is True
    assert (
        owner["meta"]["runs"][0]["run_id"]
        == waiter["meta"]["runs"][0]["run_id"]
    )


@pytest.mark.asyncio
async def test_lease_renews_through_slow_validation_and_publication(
    monkeypatch, tmp_path,
):
    import bench.manifest as manifest
    import bench.orchestrate as orch
    import bench.runner as runner

    goalset = tmp_path / "goals.jsonl"
    goalset.write_text(json.dumps({
        "id": "slow-publication-goal",
        "goal": "Find an SPP1 mechanism",
        "gold_entities": ["SPP1"],
    }))
    publishing = asyncio.Event()
    calls = 0

    async def fake_run_system(goal, **kwargs):
        nonlocal calls
        calls += 1
        return await _write_complete_benchmark_run(
            goal,
            ordinal=calls,
            text="SPP1 slow publication",
            **{
                name: kwargs[name]
                for name in (
                    "db_path",
                    "max_tasks",
                    "n_workers",
                    "provider",
                    "model_fast",
                    "model_strong",
                )
            },
        )

    original_record = manifest.Manifest.record_complete

    async def slow_record(self, *args, **kwargs):
        publishing.set()
        await asyncio.sleep(0.4)
        return await original_record(self, *args, **kwargs)

    monkeypatch.setattr(runner, "run_system", fake_run_system)
    monkeypatch.setattr(manifest.Manifest, "record_complete", slow_record)
    monkeypatch.setattr(manifest, "system_version", lambda: "publish-lease-v1")

    def make_args():
        return _Args(
            command="rediscover",
            goals=str(goalset),
            limit=1,
            max_tasks=1,
            max_time=10,
            seed=0,
            workers=1,
            provider="codex",
            model_fast="fast",
            model_strong="strong",
            manifest=str(tmp_path / "manifest.sqlite"),
            no_cache=False,
            _lease_seconds=0.15,
            _lease_poll_seconds=0.05,
            _lease_hard_timeout_seconds=30.0,
            _lease_wait_timeout_seconds=35.0,
        )

    owner_task = asyncio.create_task(orch.run_rediscovery(make_args()))
    # Bounded: if the owner dies before publishing, this test must fail rather than
    # wedge the whole pytest run.
    await asyncio.wait_for(publishing.wait(), timeout=30)
    waiter_task = asyncio.create_task(orch.run_rediscovery(make_args()))
    await asyncio.sleep(0.25)
    assert calls == 1
    owner, waiter = await asyncio.gather(owner_task, waiter_task)

    assert calls == 1
    assert owner["meta"]["cache"]["lease_renewals"] >= 1
    assert waiter["meta"]["cache"]["coalesced_runs"] == 1
    assert waiter["meta"]["n_reused_runs"] == 1


@pytest.mark.asyncio
async def test_exception_usage_is_reported_and_execution_lease_is_released(
    monkeypatch, tmp_path,
):
    import bench.manifest as manifest
    import bench.orchestrate as orch
    import bench.runner as runner

    goalset = tmp_path / "goals.jsonl"
    goalset.write_text(json.dumps({
        "id": "usage-failure-goal",
        "goal": "Find an SPP1 mechanism",
        "gold_entities": ["SPP1"],
    }))

    async def fake_run_system(goal, **kwargs):
        sink = kwargs["usage_sink"]
        sink.n_llm_calls += 3
        sink.estimated_tokens += 30
        raise RuntimeError("provider failed after spending")

    monkeypatch.setattr(runner, "run_system", fake_run_system)
    monkeypatch.setattr(manifest, "system_version", lambda: "usage-lease-v1")
    args = _Args(
        command="rediscover",
        goals=str(goalset),
        limit=1,
        max_tasks=1,
        max_time=10,
        seed=0,
        workers=1,
        provider="codex",
        model_fast="fast",
        model_strong="strong",
        manifest=str(tmp_path / "manifest.sqlite"),
        no_cache=False,
    )

    first = await orch.run_rediscovery(args)
    second = await orch.run_rediscovery(args)

    for report in (first, second):
        meta = report["meta"]
        assert meta["failed_fresh_llm_calls"] == 3
        assert meta["fresh_system_llm_calls"] == 3
        assert meta["n_llm_calls"] == 3
        assert meta["failed_fresh_estimated_tokens"] == 30
        assert meta["failures"][0]["n_llm_calls"] == 3
        assert meta["failures"][0]["estimated_tokens"] == 30
        assert meta["cache"]["lease_owners"] == 1
        assert meta["cache"]["lease_releases"] == 1


@pytest.mark.asyncio
async def test_cancelling_owner_releases_lease_for_immediate_retry(
    monkeypatch, tmp_path,
):
    import bench.manifest as manifest
    import bench.orchestrate as orch
    import bench.runner as runner

    goalset = tmp_path / "goals.jsonl"
    goalset.write_text(json.dumps({
        "id": "cancelled-owner-goal",
        "goal": "Find an SPP1 mechanism",
        "gold_entities": ["SPP1"],
    }))
    started = asyncio.Event()
    calls = 0

    async def fake_run_system(goal, **kwargs):
        nonlocal calls
        calls += 1
        ordinal = calls
        if ordinal == 1:
            started.set()
            await asyncio.Event().wait()
        return await _write_complete_benchmark_run(
            goal,
            ordinal=ordinal,
            text="SPP1 after cancellation",
            **{
                name: kwargs[name]
                for name in (
                    "db_path",
                    "max_tasks",
                    "n_workers",
                    "provider",
                    "model_fast",
                    "model_strong",
                )
            },
        )

    monkeypatch.setattr(runner, "run_system", fake_run_system)
    monkeypatch.setattr(manifest, "system_version", lambda: "cancel-lease-v1")

    def make_args():
        return _Args(
            command="rediscover",
            goals=str(goalset),
            limit=1,
            max_tasks=1,
            max_time=10,
            seed=0,
            workers=1,
            provider="codex",
            model_fast="fast",
            model_strong="strong",
            manifest=str(tmp_path / "manifest.sqlite"),
            no_cache=False,
            _lease_seconds=5.0,
            _lease_poll_seconds=0.01,
            _lease_hard_timeout_seconds=10.0,
            _lease_wait_timeout_seconds=0.3,
        )

    owner = asyncio.create_task(orch.run_rediscovery(make_args()))
    await started.wait()
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner

    retry = await orch.run_rediscovery(make_args())
    assert calls == 2
    assert retry["meta"]["n_completed"] == 1
    assert retry["meta"]["cache"]["lease_owners"] == 1
    assert retry["meta"]["cache"]["lease_waits"] == 0


@pytest.mark.asyncio
async def test_hung_owner_is_bounded_and_retry_can_take_over(
    monkeypatch, tmp_path,
):
    import bench.manifest as manifest
    import bench.orchestrate as orch
    import bench.runner as runner

    goalset = tmp_path / "goals.jsonl"
    goalset.write_text(json.dumps({
        "id": "hung-owner-goal",
        "goal": "Find an SPP1 mechanism",
        "gold_entities": ["SPP1"],
    }))
    calls = 0

    async def fake_run_system(goal, **kwargs):
        nonlocal calls
        calls += 1
        ordinal = calls
        if ordinal == 1:
            await asyncio.Event().wait()
        return await _write_complete_benchmark_run(
            goal,
            ordinal=ordinal,
            text="SPP1 after timeout",
            **{
                name: kwargs[name]
                for name in (
                    "db_path",
                    "max_tasks",
                    "n_workers",
                    "provider",
                    "model_fast",
                    "model_strong",
                )
            },
        )

    monkeypatch.setattr(runner, "run_system", fake_run_system)
    monkeypatch.setattr(manifest, "system_version", lambda: "timeout-lease-v1")
    args = _Args(
        command="rediscover",
        goals=str(goalset),
        limit=1,
        max_tasks=1,
        max_time=10,
        seed=0,
        workers=1,
        provider="codex",
        model_fast="fast",
        model_strong="strong",
        manifest=str(tmp_path / "manifest.sqlite"),
        no_cache=False,
        _lease_seconds=0.2,
        _lease_poll_seconds=0.01,
        _lease_hard_timeout_seconds=0.05,
        _lease_wait_timeout_seconds=0.5,
    )

    timed_out = await orch.run_rediscovery(args)
    args._lease_hard_timeout_seconds = 30.0
    retry = await orch.run_rediscovery(args)

    assert timed_out["meta"]["n_failed"] == 1
    assert "owner timeout" in timed_out["meta"]["failures"][0]["error"]
    assert timed_out["meta"]["cache"]["lease_timeouts"] == 1
    assert timed_out["meta"]["cache"]["lease_releases"] == 1
    assert retry["meta"]["n_completed"] == 1
    assert calls == 2


@pytest.mark.asyncio
async def test_failed_returned_run_is_not_cached_and_keeps_fresh_accounting(
    monkeypatch, tmp_path,
):
    import bench.manifest as manifest
    import bench.orchestrate as orch
    import bench.runner as runner

    goalset = tmp_path / "goals.jsonl"
    goalset.write_text(json.dumps({
        "id": "failing-goal",
        "goal": "Find an SPP1 mechanism",
        "gold_entities": ["SPP1"],
    }))
    call_count = 0

    async def fake_run_system(goal, **kwargs):
        nonlocal call_count
        call_count += 1
        return _accounting_run(
            goal.id,
            calls=7,
            tokens=70,
            hypotheses=[
                BenchHypothesis(
                    f"failed-{call_count}",
                    "SPP1 candidate",
                    "candidate",
                    1200.0,
                    "now",
                )
            ],
            failures=[{"agent_type": "ranking", "error": "bad output"}],
            workers=kwargs["n_workers"],
            backend=kwargs["provider"],
            model_fast=kwargs["model_fast"],
            model_strong=kwargs["model_strong"],
        )

    monkeypatch.setattr(runner, "run_system", fake_run_system)
    monkeypatch.setattr(manifest, "system_version", lambda: "cache-test-v1")
    args = _Args(
        command="rediscover",
        goals=str(goalset),
        limit=1,
        max_tasks=1,
        max_time=10,
        seed=0,
        workers=1,
        provider="codex",
        model_fast="fast",
        model_strong="strong",
        manifest=str(tmp_path / "manifest.sqlite"),
        no_cache=False,
    )

    first = await orch.run_rediscovery(args)
    second = await orch.run_rediscovery(args)

    assert call_count == 2
    for report in (first, second):
        meta = report["meta"]
        assert meta["cache"]["misses"] == 1
        assert meta["cache"]["hits"] == 0
        assert meta["n_failed"] == 1
        assert meta["n_runs_returned"] == 1
        assert meta["n_unaccounted_attempts"] == 0
        assert meta["fresh_llm_calls"] == 7
        assert meta["reused_llm_calls"] == 0
        assert meta["runs"][0]["accepted_for_scoring"] is False


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.asyncio
async def test_incomplete_returned_run_fails_closed_and_retains_spend(
    monkeypatch, tmp_path, cache_enabled,
):
    import bench.manifest as manifest
    import bench.orchestrate as orch
    import bench.runner as runner

    goalset = tmp_path / "goals.jsonl"
    goalset.write_text(json.dumps({
        "id": "short-goal",
        "goal": "Find an SPP1 mechanism",
        "gold_entities": ["SPP1"],
    }))

    async def fake_run_system(goal, **kwargs):
        return BenchRun(
            goal_id=goal.id,
            variant="full",
            hypotheses=[
                BenchHypothesis(
                    "short-hypothesis",
                    "SPP1 candidate",
                    "SPP1",
                    1200.0,
                    "now",
                )
            ],
            n_llm_calls=5,
            wall_clock_s=10.0,
            db_path=kwargs["db_path"],
            run_id="short-run",
            agent_task_counts={},
            agent_task_trace=[],
            workers=kwargs["n_workers"],
            backend=kwargs["provider"],
            models={
                "fast": kwargs["model_fast"],
                "strong": kwargs["model_strong"],
            },
            estimated_tokens=50,
        )

    monkeypatch.setattr(runner, "run_system", fake_run_system)
    monkeypatch.setattr(manifest, "system_version", lambda: "cache-test-v1")
    args = _Args(
        command="rediscover",
        goals=str(goalset),
        limit=1,
        max_tasks=1,
        max_time=10,
        seed=0,
        workers=1,
        provider="codex",
        model_fast="fast",
        model_strong="strong",
        manifest=str(tmp_path / "manifest.sqlite"),
        no_cache=not cache_enabled,
    )

    report = await orch.run_rediscovery(args)
    meta = report["meta"]

    assert meta["n_completed"] == 0
    assert meta["n_failed"] == 1
    assert meta["n_runs_returned"] == 1
    assert meta["n_unaccounted_attempts"] == 0
    assert meta["fresh_llm_calls"] == 5
    assert meta["fresh_estimated_tokens"] == 50
    assert "incomplete task budget" in meta["failures"][0]["error"]
    assert meta["runs"][0]["accepted_for_scoring"] is False
    assert meta["cache"]["write_failures"] == 0
    assert meta["cache"]["uncacheable_runs"] == int(cache_enabled)


@pytest.mark.asyncio
async def test_concordance_uses_the_same_completed_run_cache_path(
    monkeypatch, tmp_path,
):
    import bench.manifest as manifest
    import bench.orchestrate as orch
    import bench.runner as runner
    from bench.goalset import BenchGoal

    goal = BenchGoal(id="cached-question", goal="Question?", gold_answer="B")
    calls = []

    async def fake_run_system(selected_goal, **kwargs):
        calls.append(kwargs)
        return await _write_complete_benchmark_run(
            selected_goal,
            ordinal=len(calls),
            text="Answer: B",
            **{
                name: kwargs[name]
                for name in (
                    "db_path",
                    "max_tasks",
                    "n_workers",
                    "provider",
                    "model_fast",
                    "model_strong",
                )
            },
        )

    monkeypatch.setattr(
        orch,
        "_load_concordance_goals",
        lambda *args, **kwargs: (
            [goal],
            {"requested": "test", "source": "test", "fixture_fallback": False},
        ),
    )
    monkeypatch.setattr(runner, "run_system", fake_run_system)
    monkeypatch.setattr(manifest, "system_version", lambda: "cache-test-v1")
    args = _Args(
        command="concordance",
        dataset="gpqa-bio",
        limit=1,
        max_tasks=1,
        max_time=10,
        min_support=1,
        seed=0,
        workers=1,
        provider="codex",
        model_fast="fast",
        model_strong="strong",
        reference_samples=0,
        reference_strong=False,
        allow_fixture_fallback=False,
        manifest=str(tmp_path / "manifest.sqlite"),
        no_cache=False,
    )

    first = await orch.run_concordance(args)
    # This changes only report/scoring behavior, not model execution identity.
    args.min_support = 2
    second = await orch.run_concordance(args)

    assert len(calls) == 1
    assert first["meta"]["cache"]["misses"] == 1
    assert second["meta"]["cache"]["hits"] == 1
    assert second["meta"]["fresh_llm_calls"] == 0
    assert second["meta"]["reused_llm_calls"] == 4
    assert second["concordance"]["top1_accuracy"] == 1.0


@pytest.mark.asyncio
async def test_execution_key_dimensions_are_forwarded_losslessly_to_runner(
    monkeypatch, tmp_path,
):
    import bench.manifest as manifest
    import bench.orchestrate as orch
    import bench.runner as runner
    from core.models import AgentType

    goalset = tmp_path / "goals.jsonl"
    goalset.write_text(json.dumps({
        "id": "settings-goal",
        "goal": "Find an SPP1 mechanism",
        "gold_entities": ["SPP1"],
    }))
    overrides = {AgentType.REFLECTION: 0.0}
    captured = {}

    async def fake_run_system(goal, **kwargs):
        captured.update(kwargs)
        return _accounting_run(
            goal.id,
            calls=1,
            tokens=10,
            hypotheses=[
                BenchHypothesis(
                    "settings-hypothesis",
                    "SPP1 mechanism",
                    "SPP1",
                    1200.0,
                    "now",
                )
            ],
            workers=kwargs["n_workers"],
            backend=kwargs["provider"],
            model_fast=kwargs["model_fast"],
            model_strong=kwargs["model_strong"],
        )

    monkeypatch.setattr(runner, "run_system", fake_run_system)
    monkeypatch.setattr(manifest, "system_version", lambda: "settings-v1")
    args = _Args(
        command="rediscover",
        goals=str(goalset),
        limit=1,
        max_tasks=1,
        max_time=10,
        seed=0,
        workers=1,
        provider="codex",
        model_fast="fast",
        model_strong="strong",
        judge_provider="openai-compatible",
        judge_model="judge-model",
        ranking_mode="absolute",
        weight_overrides=overrides,
        personas=["biologist", "chemist"],
        no_cache=True,
    )

    report = await orch.run_rediscovery(args)

    assert captured["ranking_mode"] == "absolute"
    assert captured["weight_overrides"] is overrides
    assert captured["personas"] == ("biologist", "chemist")
    assert captured["resolved_execution"].ranking_mode == "absolute"
    assert captured["resolved_execution"].weight_overrides == (
        ("reflection", 0.0),
    )
    assert captured["resolved_execution"].runtime_config_json
    identity = report["meta"]["execution_identity"]
    assert identity["system_version"] == "settings-v1"
    assert identity["seed"] == 0
    assert identity["provider"] == "codex"
    assert identity["ranking_mode"] == "absolute"
    assert identity["absolute_scorer"]["provider"] == "openai-compatible"
    assert identity["absolute_scorer"]["model"] == "judge-model"
    assert identity["weight_overrides"] == [["reflection", 0.0]]
    assert identity["personas"] == ["biologist", "chemist"]
    assert identity["max_time_seconds"] == 10
    assert identity["config_sha256"]


@pytest.mark.asyncio
async def test_direct_rediscovery_loads_environment_before_provider_resolution(
    monkeypatch, tmp_path,
):
    import bench.orchestrate as orch
    import bench.runner as runner
    import tools.llm as llm

    goalset = tmp_path / "goals.jsonl"
    goalset.write_text(json.dumps({
        "id": "environment-goal",
        "goal": "Find an SPP1 mechanism",
        "gold_entities": ["SPP1"],
    }))
    monkeypatch.delenv("COSCIENTIST_PROVIDER", raising=False)
    loaded = 0
    captured = {}

    def fake_load_environment():
        nonlocal loaded
        loaded += 1
        monkeypatch.setenv("COSCIENTIST_PROVIDER", "anthropic")

    async def fake_run_system(goal, **kwargs):
        captured.update(kwargs)
        return _accounting_run(
            goal.id,
            calls=1,
            tokens=10,
            hypotheses=[
                BenchHypothesis(
                    "environment-hypothesis",
                    "SPP1 mechanism",
                    "SPP1",
                    1200.0,
                    "now",
                )
            ],
            workers=kwargs["n_workers"],
            backend=kwargs["provider"],
            model_fast=kwargs["model_fast"],
            model_strong=kwargs["model_strong"],
        )

    monkeypatch.setattr(llm, "load_backend_environment", fake_load_environment)
    monkeypatch.setattr(runner, "run_system", fake_run_system)
    args = _Args(
        command="rediscover",
        goals=str(goalset),
        limit=1,
        max_tasks=1,
        max_time=10,
        seed=0,
        workers=1,
        provider=None,
        model_fast=None,
        model_strong=None,
        no_cache=True,
    )

    report = await orch.run_rediscovery(args)

    assert loaded == 1
    assert captured["provider"] == "anthropic"
    assert report["meta"]["provider"] == "anthropic"
