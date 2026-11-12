import pytest
from pathlib import Path
from bench.runner import (
    BenchHypothesis,
    BenchRun,
    BenchmarkUsageSink,
    benchmark_db_path,
    configure_backend,
    trajectory_from_matches,
    read_run,
)
from core.state import StateStore
from core.models import Hypothesis, TournamentMatch


def test_benchmark_db_path_does_not_interpolate_untrusted_goal_id(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # benchmark_db_path() creates ./bench_runs
    path = Path(benchmark_db_path(
        "../../../../tmp/pwn",
        label="../../rediscover",
        seed=7,
        run_id="abc123",
    ))

    assert path.parent == Path("bench_runs")
    assert "/" not in path.name
    assert ".." not in path.name
    assert path.suffix == ".db"
    assert path.name.endswith("_7_abc123.db")


@pytest.mark.asyncio
async def test_trajectory_from_matches_orders_and_filters():
    rows = [
        ("t1", "a", "b", 1216.0, 1184.0),
        ("t2", "a", "c", 1200.0, 1216.0),
    ]
    traj = trajectory_from_matches("a", rows)
    assert traj == [("t1", 1216.0), ("t2", 1200.0)]
    traj_b = trajectory_from_matches("b", rows)
    assert traj_b == [("t1", 1184.0)]


@pytest.mark.asyncio
async def test_read_run_builds_bench_hypotheses(tmp_path: Path):
    db = tmp_path / "run.db"
    store = StateStore(str(db))
    await store.init_db()
    from core.models import ResearchPlanConfig
    cfg = ResearchPlanConfig(run_id="r", goal="g", preferences="p",
                             attributes=["Novelty"], constraints="c", safety_approved=True)
    await store.save_config(cfg)
    # Seed at the match's elo_before; save_match_and_elos applies the delta
    # (elo_after - elo_before) so ratings land at 1216/1184.
    await store.save_hypothesis(Hypothesis(id="a", run_id="r", text="Answer: B",
                                summary="s-a", generation_method="debate", source="system",
                                elo_rating=1200.0))
    await store.save_hypothesis(Hypothesis(id="b", run_id="r", text="Answer: C",
                                summary="s-b", generation_method="debate", source="system",
                                elo_rating=1200.0))
    await store.save_match_and_elos(TournamentMatch(
        id="m1", run_id="r", h1_id="a", h2_id="b", winner_id="a",
        match_type="single_turn", elo_before_h1=1200.0, elo_before_h2=1200.0,
        elo_after_h1=1216.0, elo_after_h2=1184.0))

    trace = [{
        "priority": 0,
        "agent_type": "generation",
        "persona": "biologist",
        "status": "succeeded",
    }]
    run = await read_run(
        str(db),
        run_id="r",
        goal_id="g1",
        variant="full",
        n_llm_calls=10,
        wall_clock_s=2.0,
        agent_task_trace=trace,
    )
    assert isinstance(run, BenchRun)
    assert run.variant == "full"
    ids = {h.id for h in run.hypotheses}
    assert ids == {"a", "b"}
    ha = next(h for h in run.hypotheses if h.id == "a")
    assert ha.elo_rating == 1216.0
    assert ha.text == "Answer: B"
    assert len(ha.elo_trajectory) == 1
    assert run.agent_task_trace == trace


def test_backend_configuration_pins_provider_and_codex_model_tiers(monkeypatch):
    from tools.llm import make_backend

    monkeypatch.setenv("COSCIENTIST_PROVIDER", "anthropic")
    cfg = {
        "provider": "claude-code",
        "anthropic": {"model_strong": "a", "model_fast": "b"},
        "codex": {"model": "legacy"},
    }
    provider, models = configure_backend(
        cfg,
        provider="codex",
        model_fast="codex-fast",
        model_strong="codex-strong",
    )
    backend = make_backend(cfg, provider_override=provider)

    assert provider == "codex"
    assert models["model_fast"] == "codex-fast"
    assert models["model_strong"] == "codex-strong"
    assert backend.model_fast == "codex-fast"
    assert backend.model_strong == "codex-strong"


def test_absolute_execution_requires_and_records_a_distinct_judge():
    from bench.runner import resolve_benchmark_execution

    execution = resolve_benchmark_execution(
        provider="codex",
        model_fast="system-fast",
        model_strong="system-strong",
        n_workers=2,
        max_tasks=3,
        max_time_seconds=30,
        ranking_mode="absolute",
        judge_provider="openai-compatible",
        judge_model="judge-v1",
    )
    identity = execution.as_dict()

    assert identity["ranking_mode"] == "absolute"
    assert identity["absolute_scorer"]["provider"] == "openai-compatible"
    assert identity["absolute_scorer"]["model"] == "judge-v1"
    assert identity["absolute_scorer"]["scorer_id"].startswith("absolute-")

    with pytest.raises(ValueError, match="differ from both"):
        resolve_benchmark_execution(
            provider="codex",
            model_fast="system-fast",
            model_strong="system-strong",
            n_workers=2,
            max_tasks=3,
            max_time_seconds=30,
            ranking_mode="absolute",
            judge_provider="openai-compatible",
            judge_model="system-strong",
        )


class _ClosableBackend:
    def __init__(self):
        self.closed = False

    async def call(self, *args, **kwargs):
        return "unused"

    async def aclose(self):
        self.closed = True


class _UsageBackend:
    provider = "test-system"
    model_fast = "system-fast"
    model_strong = "system-strong"

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    async def call(self, *args, **kwargs):
        if self.error is not None:
            raise self.error
        return self.response


class _NoopProximity:
    async def update_graph(self, hypotheses):
        return []


@pytest.mark.asyncio
async def test_run_system_usage_sink_retains_attempted_input_when_delegate_raises(
    monkeypatch, tmp_path,
):
    import core.supervisor as supervisor_module
    from bench.goalset import BenchGoal
    from bench.runner import run_system

    async def call_backend_once(self):
        await self.runner.generation.base.client.call("abcd", "efgh")

    monkeypatch.setattr(
        supervisor_module.Supervisor, "run", call_backend_once,
    )
    sink = BenchmarkUsageSink()

    with pytest.raises(RuntimeError, match="provider failed"):
        await run_system(
            BenchGoal(id="usage-error", goal="test"),
            db_path=str(tmp_path / "usage-error.db"),
            backend=_UsageBackend(error=RuntimeError("provider failed")),
            proximity_agent=_NoopProximity(),
            n_workers=1,
            max_tasks=1,
            usage_sink=sink,
        )

    assert sink.n_llm_calls == 1
    assert sink.estimated_tokens == 2


@pytest.mark.asyncio
async def test_run_system_usage_sink_counts_successful_output(
    monkeypatch, tmp_path,
):
    import core.supervisor as supervisor_module
    from bench.goalset import BenchGoal
    from bench.runner import run_system

    async def call_backend_once(self):
        await self.runner.generation.base.client.call("abcd", "efgh")

    monkeypatch.setattr(
        supervisor_module.Supervisor, "run", call_backend_once,
    )
    sink = BenchmarkUsageSink()

    run = await run_system(
        BenchGoal(id="usage-success", goal="test"),
        db_path=str(tmp_path / "usage-success.db"),
        backend=_UsageBackend(response="hello"),
        proximity_agent=_NoopProximity(),
        n_workers=1,
        max_tasks=1,
        usage_sink=sink,
    )

    assert sink.n_llm_calls == 1
    assert sink.estimated_tokens == 4
    assert run.n_llm_calls == sink.n_llm_calls
    assert run.estimated_tokens == sink.estimated_tokens


@pytest.mark.asyncio
async def test_run_system_closes_factory_backend_when_setup_fails(tmp_path):
    from bench.goalset import BenchGoal
    from bench.runner import run_system

    backend = _ClosableBackend()
    goal = BenchGoal(id="g", goal="test")

    with pytest.raises(ValueError, match="n_workers"):
        await run_system(
            goal,
            db_path=str(tmp_path / "factory-close.db"),
            client_factory=lambda strong, fast: backend,
            proximity_agent=_NoopProximity(),
            n_workers=0,
            max_tasks=1,
        )

    assert backend.closed is True


@pytest.mark.asyncio
async def test_run_system_leaves_injected_backend_caller_owned(tmp_path):
    from bench.goalset import BenchGoal
    from bench.runner import run_system

    backend = _ClosableBackend()
    goal = BenchGoal(id="g", goal="test")

    with pytest.raises(ValueError, match="n_workers"):
        await run_system(
            goal,
            db_path=str(tmp_path / "injected-not-closed.db"),
            backend=backend,
            proximity_agent=_NoopProximity(),
            n_workers=0,
            max_tasks=1,
        )

    assert backend.closed is False


@pytest.mark.asyncio
async def test_run_system_consumes_the_resolved_execution_snapshot(
    monkeypatch, tmp_path,
):
    import agents.base as base_module
    import bench.runner as runner_module
    import core.supervisor as supervisor_module
    import tools.search as search_module
    from bench.goalset import BenchGoal
    from bench.runner import resolve_benchmark_execution, run_system

    config_path = tmp_path / "config.yaml"
    config_path.write_text(Path("config.yaml").read_text())
    monkeypatch.setenv("TAVILY_API_KEY", "search-key-at-resolution")
    execution = resolve_benchmark_execution(
        provider="claude-code",
        model_fast="fast-at-resolution",
        model_strong="strong-at-resolution",
        n_workers=1,
        max_tasks=0,
        max_time_seconds=10,
        config_yaml=str(config_path),
    )

    # Neither a later file edit nor environment rotation may drift execution
    # away from the identity already used for a cache lookup.
    config_path.write_text("not: [valid")
    monkeypatch.setenv("TAVILY_API_KEY", "changed-after-resolution")
    captured = {}

    class CapturingSearch:
        def __init__(self, *, api_key, max_results):
            captured["api_key"] = api_key
            captured["max_results"] = max_results

    original_base = base_module.BaseAgent

    class CapturingBase(original_base):
        def __init__(self, *, client, prompts_dir):
            captured["prompts_dir"] = Path(prompts_dir).resolve()
            super().__init__(client=client, prompts_dir=prompts_dir)

    async def no_work(self):
        return None

    (tmp_path / "prompts").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(base_module, "BaseAgent", CapturingBase)
    monkeypatch.setattr(search_module, "SearchTool", CapturingSearch)
    monkeypatch.setattr(supervisor_module.Supervisor, "run", no_work)
    backend = _ClosableBackend()

    run = await run_system(
        BenchGoal(id="snapshot-goal", goal="test"),
        db_path=str(tmp_path / "snapshot.db"),
        client_factory=lambda strong, fast: backend,
        proximity_agent=_NoopProximity(),
        n_workers=execution.workers,
        max_tasks=execution.max_tasks,
        max_time_seconds=execution.max_time_seconds,
        provider=execution.provider,
        model_fast=execution.model_fast,
        model_strong=execution.model_strong,
        resolved_execution=execution,
        config_yaml=str(config_path),
    )

    assert run.workers == 1
    assert run.models == {
        "fast": "fast-at-resolution",
        "strong": "strong-at-resolution",
    }
    assert captured["api_key"] == "search-key-at-resolution"
    assert captured["prompts_dir"] == (
        Path(runner_module.__file__).resolve().parent.parent / "prompts"
    )
    assert backend.closed is True


@pytest.mark.asyncio
async def test_run_system_rejects_evidence_provider_conflicting_with_resolved_execution(
    monkeypatch, tmp_path,
):
    """ALSO FIX: the resolved_execution consistency check validated
    web_search_enabled against search_api_key but had no symmetric guard for
    evidence_provider. An explicit evidence_provider that disagrees with
    resolved_execution.provider would silently ground the run on a different
    literature source than the one resolved_execution.literature_grounded
    claims it used -- a manifest misdescribing how the run was produced."""
    from bench.goalset import BenchGoal
    from bench.runner import resolve_benchmark_execution, run_system

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(Path("config.yaml").read_text())
    execution = resolve_benchmark_execution(
        provider="claude-code",
        model_fast="fast",
        model_strong="strong",
        n_workers=1,
        max_tasks=0,
        max_time_seconds=10,
        config_yaml=str(config_path),
    )

    with pytest.raises(ValueError, match="evidence_provider"):
        await run_system(
            BenchGoal(id="g", goal="test"),
            db_path=str(tmp_path / "conflict.db"),
            n_workers=execution.workers,
            max_tasks=execution.max_tasks,
            max_time_seconds=execution.max_time_seconds,
            provider=execution.provider,
            model_fast=execution.model_fast,
            model_strong=execution.model_strong,
            resolved_execution=execution,
            evidence_provider="offline",
            config_yaml=str(config_path),
        )


@pytest.mark.asyncio
async def test_run_system_closes_owned_backend_when_cancelled(
    monkeypatch, tmp_path,
):
    import asyncio
    import core.supervisor as supervisor_module
    import tools.llm as llm
    from bench.goalset import BenchGoal
    from bench.runner import run_system

    backend = _ClosableBackend()
    started = asyncio.Event()

    async def blocked_run(self):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(llm, "make_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(supervisor_module.Supervisor, "run", blocked_run)
    task = asyncio.create_task(run_system(
        BenchGoal(id="g", goal="test"),
        db_path=str(tmp_path / "cancel-close.db"),
        provider="claude-code",
        proximity_agent=_NoopProximity(),
        n_workers=1,
        max_tasks=1,
    ))
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert backend.closed is True


class _AbsoluteJudge:
    weights = {
        "novelty": 1.0,
        "feasibility": 1.0,
        "correctness": 1.0,
        "impact": 1.0,
    }

    provider = "injected"
    model = "judge-v1"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def score(self, hypothesis, *, field):
        self.calls.append((hypothesis, field))
        return next(self.responses)


def _absolute_result(value):
    return {
        "scores": {
            "novelty": value,
            "feasibility": value,
            "correctness": value,
            "impact": value,
        },
        "total": float(value),
    }


@pytest.mark.asyncio
async def test_run_system_absolute_end_to_end_usage_and_reload(
    monkeypatch, tmp_path,
):
    import core.supervisor as supervisor_module
    from bench.goalset import BenchGoal
    from bench.judge import absolute_scorer_identity
    from bench.runner import run_system
    from core.models import AgentTask, AgentType, Hypothesis

    async def seed_and_rank(self):
        for identifier, text in (("a", "low"), ("b", "high"), ("c", "mid")):
            await self.store.save_hypothesis(Hypothesis(
                id=identifier,
                run_id=self.config.run_id,
                text=text,
                summary=text,
                generation_method="debate",
                source="system",
            ))
        task = AgentTask(
            priority=0,
            agent_type=AgentType.RANKING,
            run_id=self.config.run_id,
        )
        await self.runner.run_task(task)
        await self.runner.run_task(task)

    monkeypatch.setattr(supervisor_module.Supervisor, "run", seed_and_rank)
    descriptor = absolute_scorer_identity(provider="injected", model="judge-v1")
    judge = _AbsoluteJudge([
        _absolute_result(2),
        _absolute_result(5),
        _absolute_result(3),
    ])
    sink = BenchmarkUsageSink()
    db_path = tmp_path / "absolute-run.db"

    run = await run_system(
        BenchGoal(id="absolute", goal="test", domain="molecular biology"),
        db_path=str(db_path),
        backend=_UsageBackend(response="unused"),
        proximity_agent=_NoopProximity(),
        n_workers=1,
        max_tasks=1,
        ranking_mode="absolute",
        absolute_judge=judge,
        absolute_scorer=descriptor,
        usage_sink=sink,
    )

    assert run.ranking_mode == "absolute"
    assert run.n_llm_calls == sink.n_llm_calls == 3
    assert len(judge.calls) == 3
    assert [hypothesis.id for hypothesis in run.hypotheses] == ["b", "c", "a"]
    assert [hypothesis.absolute_score for hypothesis in run.hypotheses] == [5.0, 3.0, 2.0]
    assert all(hypothesis.elo_rating == 1200.0 for hypothesis in run.hypotheses)
    assert all(not hypothesis.elo_trajectory for hypothesis in run.hypotheses)

    reopened = await read_run(
        str(db_path),
        run_id=run.run_id,
        goal_id=run.goal_id,
        variant=run.variant,
        n_llm_calls=run.n_llm_calls,
        wall_clock_s=run.wall_clock_s,
        ranking_mode="absolute",
        absolute_scorer=descriptor,
    )
    assert [hypothesis.id for hypothesis in reopened.hypotheses] == ["b", "c", "a"]
    assert reopened.absolute_scorer == descriptor


@pytest.mark.asyncio
async def test_absolute_failed_attempt_remains_in_usage_sink(
    monkeypatch, tmp_path,
):
    import core.supervisor as supervisor_module
    from bench.errors import BenchError
    from bench.goalset import BenchGoal
    from bench.judge import absolute_scorer_identity
    from bench.runner import run_system
    from core.models import AgentTask, AgentType, Hypothesis

    async def seed_and_rank(self):
        for identifier in ("a", "b"):
            await self.store.save_hypothesis(Hypothesis(
                id=identifier,
                run_id=self.config.run_id,
                text=identifier,
                summary=identifier,
                generation_method="debate",
                source="system",
            ))
        await self.runner.run_task(AgentTask(
            priority=0,
            agent_type=AgentType.RANKING,
            run_id=self.config.run_id,
        ))

    monkeypatch.setattr(supervisor_module.Supervisor, "run", seed_and_rank)
    descriptor = absolute_scorer_identity(provider="injected", model="judge-v1")
    judge = _AbsoluteJudge([
        _absolute_result(4),
        {"scores": {"novelty": 4}, "total": 4.0},
    ])
    sink = BenchmarkUsageSink()
    db_path = tmp_path / "absolute-failed.db"

    with pytest.raises(BenchError):
        await run_system(
            BenchGoal(id="absolute-failed", goal="test"),
            db_path=str(db_path),
            backend=_UsageBackend(response="unused"),
            proximity_agent=_NoopProximity(),
            n_workers=1,
            max_tasks=1,
            ranking_mode="absolute",
            absolute_judge=judge,
            absolute_scorer=descriptor,
            usage_sink=sink,
        )

    assert sink.n_llm_calls == 2
    store = StateStore(str(db_path))
    assert (await store.get_absolute_score("a")).total == 4.0
    assert await store.get_absolute_score("b") is None


@pytest.mark.parametrize(
    ("identity_overrides", "message"),
    [
        ({"panel_size": 2}, "panel_size must be 1"),
        ({"decoding_tier": "fast"}, "decoding_tier must be 'strong'"),
    ],
)
def test_absolute_execution_rejects_unimplemented_scorer_paths(
    identity_overrides, message,
):
    import json

    from bench.judge import absolute_scorer_identity
    from bench.runner import BenchmarkExecution

    scorer = absolute_scorer_identity(
        provider="injected", model="judge-v1", **identity_overrides
    )
    execution = BenchmarkExecution(
        provider="test-system",
        model_fast="system-fast",
        model_strong="system-strong",
        workers=1,
        max_tasks=1,
        max_time_seconds=10,
        config_sha256="test",
        web_search_enabled=False,
        evidence_max_claims=4,
        literature_grounded=True,
        ranking_mode="absolute",
        absolute_scorer_json=json.dumps(scorer),
    )

    with pytest.raises(ValueError, match=message):
        execution.as_dict()


@pytest.mark.asyncio
async def test_injected_absolute_judge_identity_must_match_descriptor(
    monkeypatch, tmp_path,
):
    import core.supervisor as supervisor_module
    from bench.goalset import BenchGoal
    from bench.judge import absolute_scorer_identity
    from bench.runner import run_system

    async def no_work(self):
        return None

    monkeypatch.setattr(supervisor_module.Supervisor, "run", no_work)
    judge = _AbsoluteJudge([])
    descriptor = absolute_scorer_identity(
        provider="different-provider", model=judge.model
    )

    with pytest.raises(ValueError, match="identity differs"):
        await run_system(
            BenchGoal(id="identity-mismatch", goal="test"),
            db_path=str(tmp_path / "identity-mismatch.db"),
            backend=_UsageBackend(response="unused"),
            proximity_agent=_NoopProximity(),
            n_workers=1,
            max_tasks=1,
            ranking_mode="absolute",
            absolute_judge=judge,
            absolute_scorer=descriptor,
        )


@pytest.mark.asyncio
async def test_final_absolute_scoring_obeys_overall_deadline_and_cleans_up(
    monkeypatch, tmp_path,
):
    import asyncio

    import core.supervisor as supervisor_module
    from bench.goalset import BenchGoal
    from bench.judge import absolute_scorer_identity
    from bench.runner import run_system
    from core.models import Hypothesis

    class ClosableModelBackend:
        provider = "test-system"

        def __init__(self, strong, fast):
            self.model_strong = strong
            self.model_fast = fast
            self.closed = False

        async def call(self, *args, **kwargs):
            return "unused"

        async def aclose(self):
            self.closed = True

    class BlockingJudge:
        provider = "injected"
        model = "judge-v1"
        weights = _AbsoluteJudge.weights

        def __init__(self):
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def score(self, hypothesis, *, field):
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    async def seed_only(self):
        await self.store.save_hypothesis(Hypothesis(
            id="pending",
            run_id=self.config.run_id,
            text="score me",
            summary="pending",
            generation_method="debate",
            source="system",
        ))

    monkeypatch.setattr(supervisor_module.Supervisor, "run", seed_only)
    backend = None

    def factory(strong, fast):
        nonlocal backend
        backend = ClosableModelBackend(strong, fast)
        return backend

    judge = BlockingJudge()
    descriptor = absolute_scorer_identity(
        provider=judge.provider, model=judge.model
    )
    sink = BenchmarkUsageSink()
    db_path = tmp_path / "absolute-timeout.db"

    with pytest.raises(TimeoutError):
        await run_system(
            BenchGoal(id="absolute-timeout", goal="test"),
            db_path=str(db_path),
            client_factory=factory,
            provider="claude-code",
            model_fast="system-fast",
            model_strong="system-strong",
            proximity_agent=_NoopProximity(),
            n_workers=1,
            max_tasks=1,
            max_time_seconds=0.05,
            ranking_mode="absolute",
            absolute_judge=judge,
            absolute_scorer=descriptor,
            usage_sink=sink,
        )

    assert judge.started.is_set()
    assert judge.cancelled.is_set()
    assert sink.active_calls == 0
    assert backend is not None and backend.closed is True
    store = StateStore(str(db_path))
    assert await store.get_absolute_score("pending") is None
    claim = await store.claim_absolute_score("pending", descriptor["scorer_id"])
    assert claim.state == "acquired"
    await store.release_absolute_score_claim(
        "pending", descriptor["scorer_id"], claim.owner_token
    )


async def _build_runner_for_test(bench_runner, *, evidence_provider=None):
    """Drive run_system() only as far as the AgentRunner(...) construction the
    caller has patched into a spy. Nothing that happens after that line
    (Supervisor.run, finalize_ranking, ...) matters to what the spy captured,
    so any failure past construction is swallowed here -- if construction was
    never reached, `captured` stays empty and the caller's own assertions
    fail loudly instead of this helper failing for the wrong reason.

    `evidence_provider`, when given, is forwarded to run_system() unchanged;
    when omitted, run_system() sees its default (None) -- the "injected
    backend with no stated evidence intent" shape this file's regression test
    pins.
    """
    import tempfile
    from pathlib import Path

    from bench.goalset import BenchGoal

    kwargs = {} if evidence_provider is None else {"evidence_provider": evidence_provider}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db_path = Path(tmp) / "spy.db"
        try:
            await bench_runner.run_system(
                BenchGoal(id="spy-goal", goal="test"),
                db_path=str(db_path),
                backend=_UsageBackend(response="unused"),
                proximity_agent=_NoopProximity(),
                n_workers=1,
                max_tasks=0,
                **kwargs,
            )
        except Exception:
            pass


@pytest.mark.asyncio
async def test_bench_agent_runner_receives_a_verifier(monkeypatch):
    """A benchmark that silently runs the unverified system, while the README
    claims verification, would misreport the artifacts most likely to be
    published. Assert the wiring by observing construction, not source text."""
    from bench import runner as bench_runner

    captured = {}

    class _Spy:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        async def run_task(self, task):
            return None

    monkeypatch.setattr(bench_runner, "AgentRunner", _Spy)
    await _build_runner_for_test(bench_runner)   # smallest helper that reaches construction

    assert captured.get("verifier") is not None, "bench must pass a verifier"
    assert type(captured["search"]).__name__ == "LiteratureTool", (
        "bench must ground through LiteratureTool, not a bare SearchTool"
    )


@pytest.mark.asyncio
async def test_run_system_survives_config_without_evidence_section(monkeypatch):
    """IMPORTANT 3: `cfg["evidence"]["max_claims"]` hard-subscripted and crashed
    every run_system() call for a config lacking an `evidence:` section. The
    sibling reads at resolve_benchmark_execution and coscientist.py already use
    a defensive `.get()` chain (defaults 4 and 2); only this site was unguarded."""
    from bench import runner as bench_runner

    real_cfg = bench_runner._load_yaml_config()
    stripped_cfg = {k: v for k, v in real_cfg.items() if k != "evidence"}
    assert "evidence" not in stripped_cfg
    monkeypatch.setattr(bench_runner, "_load_yaml_config", lambda *a, **k: stripped_cfg)

    captured = {}

    class _Spy:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        async def run_task(self, task):
            return None

    monkeypatch.setattr(bench_runner, "AgentRunner", _Spy)
    await _build_runner_for_test(bench_runner)   # smallest helper that reaches construction

    assert captured.get("verifier") is not None, "run_system must not crash before wiring a verifier"
    assert captured["verifier"].max_claims == 4
    assert captured["verifier"].concurrency == 2


@pytest.mark.asyncio
async def test_bench_execution_records_evidence_settings():
    """Evidence settings must enter the cache key, or a cached pre-evidence run
    is indistinguishable from a verified one."""
    from bench.runner import BenchmarkExecution
    import dataclasses

    fields = {f.name for f in dataclasses.fields(BenchmarkExecution)}
    assert "evidence_max_claims" in fields
    assert "literature_grounded" in fields


@pytest.mark.asyncio
async def test_run_system_evidence_provider_offline_uses_fixture(monkeypatch):
    """An injected backend with evidence_provider="offline" must ground on the
    deterministic fixture and construct no network-capable source -- this is
    the explicit opt-in the offline smokes rely on for their zero-network
    guarantee."""
    from bench import runner as bench_runner
    from tools.evidence import FixtureEvidenceSource

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    captured = {}

    class _Spy:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        async def run_task(self, task):
            return None

    monkeypatch.setattr(bench_runner, "AgentRunner", _Spy)
    await _build_runner_for_test(bench_runner, evidence_provider="offline")

    sources = captured["search"].sources
    assert len(sources) == 1
    assert isinstance(sources[0], FixtureEvidenceSource)


@pytest.mark.asyncio
async def test_run_system_evidence_provider_non_offline_grounds_on_pubmed(
    monkeypatch,
):
    """An explicit non-offline evidence_provider must ground on real, keyless
    PubMed -- never on the synthetic fixture -- regardless of the injected
    backend's own identity (the backend has no provider notion run_system can
    read; evidence_provider is the only signal for this)."""
    from bench import runner as bench_runner
    from tools.evidence import FixtureEvidenceSource, PubMedSource

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    captured = {}

    class _Spy:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        async def run_task(self, task):
            return None

    monkeypatch.setattr(bench_runner, "AgentRunner", _Spy)
    await _build_runner_for_test(bench_runner, evidence_provider="claude-code")

    sources = captured["search"].sources
    assert len(sources) == 1
    assert isinstance(sources[0], PubMedSource)
    assert not any(isinstance(s, FixtureEvidenceSource) for s in sources)


@pytest.mark.asyncio
async def test_run_system_injected_backend_without_evidence_provider_grounds_on_pubmed(
    monkeypatch,
):
    """Regression pin. An injected `backend=` with no `evidence_provider` at
    all is exactly the shape that, before this fix, silently produced
    synthetic FixtureEvidenceSource records for a real company/hierarchy.py
    run -- ClaimVerifier then "verified" hypotheses against data whose own
    docstring says it "carries no scientific meaning", with nothing recorded
    to flag it. The chosen fix is: fall back to real, keyless PubMed, never
    to the fixture -- an omitted evidence_provider must never look more
    "grounded" than it is."""
    from bench import runner as bench_runner
    from tools.evidence import FixtureEvidenceSource, PubMedSource

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    captured = {}

    class _Spy:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        async def run_task(self, task):
            return None

    monkeypatch.setattr(bench_runner, "AgentRunner", _Spy)
    await _build_runner_for_test(bench_runner)  # no evidence_provider at all

    sources = captured["search"].sources
    assert not any(isinstance(s, FixtureEvidenceSource) for s in sources), (
        "an injected backend with no stated evidence intent must not silently "
        "ground on synthetic fixture data"
    )
    assert any(isinstance(s, PubMedSource) for s in sources)
