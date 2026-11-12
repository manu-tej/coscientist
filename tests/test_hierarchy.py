from pathlib import Path

import pytest

from company import engine, store
from company.hierarchy import (
    BudgetedLLMBackend,
    CoScientistStageExecutor,
    PortableTokenBudget,
    StageExecutionError,
    _validate_synthesis,
)
from company.ledger import InsufficientTokens
from company.models import Candidate, GateDecision, ProgramStatus, Stage


class _PortableFakeBackend:
    """One backend contract used in place of any vendor/model during the test."""

    provider = "portable-test"
    model_fast = "portable-fast-v1"
    model_strong = "portable-strong-v1"

    def __init__(self):
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.requests = []

    async def call(self, system, user, **kwargs):
        import asyncio
        self.calls += 1
        self.requests.append((system, user))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        if "You lead the" in system or "Repair invalid" in system:
            return (
                '{"confidence": 0.82, "red_flags": [], '
                '"summary": "Independent teams converge on a testable PDE5 mechanism.", '
                '"ranked_candidates": []}'
            )
        return (
            "HYPOTHESIS\n"
            "Hypothesis: PDE5 inhibition restores cGMP signaling.\n"
            "Summary: Test PDE5-mediated vascular rescue.\n"
            "Category: Mechanism\n"
            "Verdict: PASSED\n"
            "Observation: measurable cGMP rescue\n"
            "better hypothesis: 1"
        )


class _NoopProximity:
    async def update_graph(self, hypotheses):
        return []


@pytest.mark.asyncio
async def test_portable_token_budget_refuses_before_backend_call():
    backend = _PortableFakeBackend()
    budget = PortableTokenBudget(limit=5)
    wrapped = BudgetedLLMBackend(backend, budget)

    with pytest.raises(InsufficientTokens, match="only 5 remain unreserved"):
        await wrapped.call("", "", max_tokens=6)

    assert backend.calls == 0
    assert budget.spent == 0
    assert budget.reserved == 0


@pytest.mark.asyncio
async def test_shared_token_budget_blocks_concurrent_aggregate_oversubscription():
    import asyncio

    class BlockingBackend:
        def __init__(self):
            self.calls = 0
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def call(self, *args, **kwargs):
            self.calls += 1
            self.started.set()
            await self.release.wait()
            return "ok"

    backend = BlockingBackend()
    budget = PortableTokenBudget(limit=10)
    first_team = BudgetedLLMBackend(backend, budget)
    second_team = BudgetedLLMBackend(backend, budget)

    # "abcd" is one portable prompt token, leaving a nine-token output
    # reservation. While it is in flight, the sibling team cannot reserve again.
    first = asyncio.create_task(
        first_team.call("abcd", "", max_tokens=9),
    )
    await backend.started.wait()
    second = asyncio.create_task(
        second_team.call("abcd", "", max_tokens=9),
    )
    await asyncio.sleep(0)
    assert backend.calls == 1
    assert not second.done()

    backend.release.set()
    assert await first == "ok"
    with pytest.raises(InsufficientTokens, match="only 8 remain unreserved"):
        await second
    assert budget.reserved == 0
    assert budget.spent == 2
    assert budget.remaining == 8


@pytest.mark.asyncio
async def test_cancelled_backend_call_reconciles_its_reservation():
    import asyncio

    class BlockingBackend:
        def __init__(self):
            self.started = asyncio.Event()

        async def call(self, *args, **kwargs):
            self.started.set()
            await asyncio.Event().wait()

    backend = BlockingBackend()
    budget = PortableTokenBudget(limit=10)
    wrapped = BudgetedLLMBackend(backend, budget)
    call = asyncio.create_task(
        wrapped.call("abcd", "", max_tokens=9),
    )
    await backend.started.wait()
    assert budget.reserved == 10

    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    assert budget.reserved == 0
    assert budget.spent == 10
    assert wrapped.accounted_tokens == 10


def test_team_synthesis_contract_is_fail_closed():
    assert _validate_synthesis(
        'prose {"confidence": 0.7, "red_flags": [], "summary": "grounded", '
        '"ranked_candidates": []}'
    )["confidence"] == 0.7
    with pytest.raises(ValueError):
        _validate_synthesis("")
    with pytest.raises(ValueError):
        _validate_synthesis(
            '{"confidence": 2, "red_flags": [], "summary": "bad", '
            '"ranked_candidates": []}'
        )


def test_candidate_synthesis_requires_cited_grounding():
    quantitative = {
        "quant-1": Candidate(
            "sildenafil", "pulmonary arterial hypertension",
            {"target": 0.9},
        ),
    }
    valid = _validate_synthesis(
        '{"confidence": 0.8, "red_flags": [], "summary": "grounded", '
        '"ranked_candidates": [{"drug": "sildenafil", '
        '"indication": "pulmonary arterial hypertension", '
        '"rationale": "computed target support", '
        '"evidence_refs": ["quant-1"]}]}',
        quantitative_candidates=quantitative,
        program_disease="pulmonary arterial hypertension",
        allow_candidate_ranking=True,
    )
    assert valid["ranked_candidates"][0]["drug"] == "sildenafil"

    agent_nominated = _validate_synthesis(
        '{"confidence": 0.6, "red_flags": [], "summary": "agent-grounded", '
        '"ranked_candidates": [{"drug": "bosentan", '
        '"indication": "pulmonary arterial hypertension", '
        '"rationale": "The cited hypothesis proposes endothelin blockade.", '
        '"evidence_refs": ["hypothesis-1"]}]}',
        agent_evidence={
            "hypothesis-1": (
                "Bosentan may treat pulmonary arterial hypertension through "
                "endothelin receptor blockade."
            ),
        },
        program_disease="pulmonary arterial hypertension",
        allow_candidate_ranking=True,
    )
    assert agent_nominated["ranked_candidates"][0]["drug"] == "bosentan"
    assert (
        agent_nominated["ranked_candidates"][0]["regulatory_status"]
        == "unknown"
    )
    assert agent_nominated["ranked_candidates"][0]["regulatory_live_verified"] is False

    with pytest.raises(ValueError, match="not grounded"):
        _validate_synthesis(
            '{"confidence": 0.8, "red_flags": [], "summary": "invented", '
            '"ranked_candidates": [{"drug": "ibuprofen", '
            '"indication": "pulmonary arterial hypertension", '
            '"rationale": "unsupported", "evidence_refs": ["quant-1"]}]}',
            quantitative_candidates=quantitative,
            program_disease="pulmonary arterial hypertension",
            allow_candidate_ranking=True,
        )


@pytest.mark.asyncio
async def test_company_stage_invokes_supervisor_and_persists_agent_trace(tmp_path: Path):
    pf = engine.new_company("Helix", "pulmonary arterial hypertension", seed=3)
    program = engine.add_program(
        pf, "PAH-1", "pulmonary arterial hypertension", seed=3,
    )
    backend = _PortableFakeBackend()
    executor = CoScientistStageExecutor(
        backend,
        n_workers=2,
        max_tasks=12,
        max_time_seconds=30,
        db_dir=str(tmp_path / "stage-runs"),
        proximity_agent=_NoopProximity(),
        evidence_provider="offline",
    )

    result = await executor.execute(program, cycle=1)

    assert result.coscientist_run_id
    assert Path(result.coscientist_db_path).exists()
    assert result.team_name == "Disease Biology"
    assert result.team_confidence == 0.82
    assert result.agent_trace
    assert sum(t["tasks"] for t in result.agent_trace) == 12
    assert any(t["agent_type"] == "generation" for t in result.agent_trace)
    assert {
        t["persona"] for t in result.agent_trace
    } == set(result.team_roster)
    assert all(t["status"] == "succeeded" for t in result.agent_trace)
    assert backend.calls > 1
    assert result.llm_calls == backend.calls
    assert result.tokens_spent > 0
    assert result.token_accounting == (
        "portable chars/4 estimate; not exact provider billing"
    )
    assert backend.max_active <= 2
    assert result.provider == "portable-test"
    assert result.models == {
        "fast": "portable-fast-v1",
        "strong": "portable-strong-v1",
    }
    assert result.workers == 2
    assert result.max_tasks == 12
    assert result.seed == 4
    assert result.max_backend_concurrency <= 2
    task_requests = [
        f"{system}\n{user}" for system, user in backend.requests
        if "You lead the" not in system and "Repair invalid" not in system
    ]
    assert any(
        "Assigned specialist role: disease biologist" in request
        for request in task_requests
    )
    assert any("Quantitative baseline" in request for request in task_requests)
    assert result.quantitative_baseline["experiments"]


@pytest.mark.asyncio
async def test_sync_stage_baselines_overlap_across_program_teams(
    tmp_path: Path, monkeypatch,
):
    import asyncio
    import threading

    import company.hierarchy as hierarchy

    real_run_stage = hierarchy.run_stage
    rendezvous = threading.Barrier(2)
    active = 0
    max_active = 0
    lock = threading.Lock()

    def overlapping_run_stage(*args, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            rendezvous.wait(timeout=2)
            return real_run_stage(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(hierarchy, "run_stage", overlapping_run_stage)
    pf = engine.new_company("Helix", "PAH", seed=8)
    programs = [
        engine.add_program(
            pf,
            name,
            "pulmonary arterial hypertension",
            seed=seed,
        )
        for name, seed in (("PAH-A", 8), ("PAH-B", 9))
    ]
    executor = CoScientistStageExecutor(
        _PortableFakeBackend(),
        n_workers=2,
        max_tasks=12,
        max_time_seconds=5,
        db_dir=str(tmp_path / "stage-runs"),
        proximity_agent=_NoopProximity(),
        evidence_provider="offline",
    )

    results = await asyncio.gather(
        *(executor.execute(program, cycle=1) for program in programs)
    )

    assert max_active == 2
    assert len(results) == 2


@pytest.mark.asyncio
async def test_whole_stage_timeout_includes_sync_quantitative_baseline(
    tmp_path: Path, monkeypatch,
):
    import time

    import company.hierarchy as hierarchy

    real_run_stage = hierarchy.run_stage

    def slow_run_stage(*args, **kwargs):
        time.sleep(0.05)
        return real_run_stage(*args, **kwargs)

    monkeypatch.setattr(hierarchy, "run_stage", slow_run_stage)
    pf = engine.new_company("Helix", "PAH", seed=10)
    program = engine.add_program(pf, "PAH-A", "PAH", seed=10)
    backend = _PortableFakeBackend()
    executor = CoScientistStageExecutor(
        backend,
        n_workers=1,
        max_tasks=4,
        max_time_seconds=0.01,
        db_dir=str(tmp_path / "stage-runs"),
        proximity_agent=_NoopProximity(),
        evidence_provider="offline",
    )

    with pytest.raises(
        StageExecutionError,
        match="whole-stage timeout",
    ) as failure:
        await executor.execute(program, cycle=1)

    assert backend.calls == 0
    assert program.lead_candidate is None
    assert failure.value.attempted_credits == 2.0


@pytest.mark.asyncio
async def test_provider_timeout_is_not_relabelled_as_whole_stage_timeout(
    tmp_path: Path, monkeypatch,
):
    pf = engine.new_company("Helix", "PAH", seed=10)
    program = engine.add_program(pf, "PAH-A", "PAH", seed=10)
    executor = CoScientistStageExecutor(
        _PortableFakeBackend(),
        n_workers=1,
        max_tasks=4,
        max_time_seconds=30,
        db_dir=str(tmp_path / "stage-runs"),
        proximity_agent=_NoopProximity(),
        evidence_provider="offline",
    )

    async def provider_timeout(*args, **kwargs):
        error = TimeoutError("provider deadline")
        error.attempted_tokens = 11
        error.attempted_credits = 2.5
        raise error

    monkeypatch.setattr(executor, "_execute", provider_timeout)

    with pytest.raises(TimeoutError, match="provider deadline") as failure:
        await executor.execute(program, cycle=1)

    assert not isinstance(failure.value, StageExecutionError)
    assert failure.value.attempted_tokens == 11
    assert failure.value.attempted_credits == 2.5


@pytest.mark.asyncio
async def test_whole_stage_timeout_includes_final_synthesis(
    tmp_path: Path, monkeypatch,
):
    import asyncio

    import bench.runner as runner
    import company.hierarchy as hierarchy
    from bench.runner import BenchHypothesis, BenchRun

    async def fake_run_system(*args, **kwargs):
        return BenchRun(
            goal_id="g",
            variant="company-stage",
            hypotheses=[
                BenchHypothesis(
                    id="h",
                    text="candidate",
                    summary="candidate",
                    elo_rating=1200,
                    created_at="now",
                )
            ],
            n_llm_calls=1,
            wall_clock_s=0,
            db_path=str(tmp_path / "run.db"),
            run_id="run",
            agent_task_counts={"generation": 1},
            agent_task_trace=[{
                "priority": 0,
                "agent_type": "generation",
                "persona": "disease biologist",
                "strategy": "debate",
                "status": "succeeded",
                "error": None,
            }],
            workers=1,
            backend="portable-test",
            models={"fast": "fast", "strong": "strong"},
        )

    async def stuck_synthesis(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(runner, "run_system", fake_run_system)
    monkeypatch.setattr(hierarchy, "_synthesize", stuck_synthesis)
    pf = engine.new_company("Helix", "PAH", seed=11)
    program = engine.add_program(pf, "PAH-A", "PAH", seed=11)
    executor = CoScientistStageExecutor(
        _PortableFakeBackend(),
        n_workers=1,
        max_tasks=4,
        max_time_seconds=0.05,
        db_dir=str(tmp_path / "stage-runs"),
        proximity_agent=_NoopProximity(),
        evidence_provider="offline",
    )

    with pytest.raises(StageExecutionError, match="whole-stage timeout"):
        await executor.execute(program, cycle=1)


@pytest.mark.asyncio
async def test_stage_executor_reports_quarantine_not_generic_failure(
    tmp_path: Path, monkeypatch,
):
    """CRITICAL 2 (same class as the CLI fix): 'produced no hypotheses' is
    false when hypotheses WERE produced and then quarantined by verification
    (a contradicted claim or an untraced citation). The raised message must
    say so, distinctly from a genuinely empty run."""
    import bench.runner as runner
    from bench.runner import BenchRun

    async def fake_run_system(*args, **kwargs):
        return BenchRun(
            goal_id="g", variant="company-stage", hypotheses=[],
            n_llm_calls=1, wall_clock_s=0, db_path=str(tmp_path / "run.db"),
            run_id="run", n_quarantined=3,
        )

    monkeypatch.setattr(runner, "run_system", fake_run_system)
    pf = engine.new_company("Helix", "PAH", seed=12)
    program = engine.add_program(pf, "PAH-A", "PAH", seed=12)
    executor = CoScientistStageExecutor(
        _PortableFakeBackend(),
        n_workers=1,
        max_tasks=4,
        max_time_seconds=30,
        db_dir=str(tmp_path / "stage-runs"),
        proximity_agent=_NoopProximity(),
        evidence_provider="offline",
    )

    with pytest.raises(StageExecutionError, match="quarantined") as failure:
        await executor.execute(program, cycle=1)

    assert "produced no hypotheses" not in str(failure.value)
    assert "3" in str(failure.value)


@pytest.mark.asyncio
async def test_stage_executor_reports_genuinely_empty_run(
    tmp_path: Path, monkeypatch,
):
    """The flip side of the quarantine message: a run with zero hypotheses AND
    zero quarantined is a real failure and must keep the original message."""
    import bench.runner as runner
    from bench.runner import BenchRun

    async def fake_run_system(*args, **kwargs):
        return BenchRun(
            goal_id="g", variant="company-stage", hypotheses=[],
            n_llm_calls=1, wall_clock_s=0, db_path=str(tmp_path / "run.db"),
            run_id="run",
        )

    monkeypatch.setattr(runner, "run_system", fake_run_system)
    pf = engine.new_company("Helix", "PAH", seed=13)
    program = engine.add_program(pf, "PAH-A", "PAH", seed=13)
    executor = CoScientistStageExecutor(
        _PortableFakeBackend(),
        n_workers=1,
        max_tasks=4,
        max_time_seconds=30,
        db_dir=str(tmp_path / "stage-runs"),
        proximity_agent=_NoopProximity(),
        evidence_provider="offline",
    )

    with pytest.raises(StageExecutionError, match="produced no hypotheses"):
        await executor.execute(program, cycle=1)


class _RankingBackend(_PortableFakeBackend):
    async def call(self, system, user, **kwargs):
        import asyncio
        self.calls += 1
        self.requests.append((system, user))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        if "You lead the" in system or "Repair invalid" in system:
            return (
                '{"confidence": 0.76, '
                '"red_flags": ["Confirm tadalafil exposure in the target population."], '
                '"summary": "Tadalafil has the strongest decision-ready evidence.", '
                '"ranked_candidates": [{"drug": "tadalafil", '
                '"indication": "pulmonary arterial hypertension", '
                '"rationale": "Supported by the quantitative candidate panel.", '
                '"evidence_refs": ["quant-2"]}, {"drug": "sildenafil", '
                '"indication": "pulmonary arterial hypertension", '
                '"rationale": "Strong alternative supported by the panel.", '
                '"evidence_refs": ["quant-1"]}]}'
            )
        return (
            "HYPOTHESIS\n"
            "Hypothesis: PDE5 inhibition with tadalafil can restore cGMP signaling "
            "in pulmonary arterial hypertension.\n"
            "Summary: Compare tadalafil and sildenafil exposure-response.\n"
            "Category: Repurposing\n"
            "Verdict: PASSED\n"
            "Observation: measurable cGMP rescue\n"
            "better hypothesis: 1"
        )


@pytest.mark.asyncio
async def test_team_ranking_controls_lead_and_survives_completed_package(
    tmp_path: Path, monkeypatch,
):
    class AlwaysSurvive:
        @staticmethod
        def random():
            return 0.0

    monkeypatch.setattr(engine, "_gate_rng", lambda program, cycle: AlwaysSurvive())
    pf = engine.new_company("Helix", "PAH", credit_budget=500, seed=4)
    program = engine.add_program(
        pf, "PAH-1", "pulmonary arterial hypertension", seed=4,
    )
    program.stage = Stage.HYPOTHESES
    executor = CoScientistStageExecutor(
        _RankingBackend(),
        n_workers=2,
        max_tasks=12,
        max_time_seconds=30,
        db_dir=str(tmp_path / "stage-runs"),
        proximity_agent=_NoopProximity(),
        evidence_provider="offline",
    )

    result, _ = await engine.run_program_stage_async(
        pf, program, executor=executor,
    )

    assert [candidate.drug for candidate in result.top_candidates] == [
        "tadalafil", "sildenafil",
    ]
    assert program.lead_drug == "tadalafil"
    assert result.team_candidate_ranking[0]["evidence_refs"] == ["quant-2"]
    assert result.red_flag_details == [
        "Confirm tadalafil exposure in the target population.",
    ]
    assert result.token_accounting.startswith(
        "hard portable estimated/requested-token envelope"
    )
    assert pf.company.token_spent == result.tokens_spent

    engine.resolve_gate(pf, program, GateDecision.ADVANCE)
    program.stage = Stage.TRANSLATIONAL
    engine.run_program_stage(pf, program)
    engine.resolve_gate(pf, program, GateDecision.ADVANCE)
    assert program.status is ProgramStatus.COMPLETED

    state_path = tmp_path / "company.json"
    store.save(pf, str(state_path))
    reloaded = store.load(str(state_path))
    package = engine.build_clinical_hypothesis_package(
        reloaded, reloaded.programs[0],
    )
    hierarchy = package["hierarchy"]
    assert package["drug"] == "tadalafil"
    assert package["red_flags"] == result.red_flag_details
    assert hierarchy["team_name"] == "Repurposing Discovery"
    assert hierarchy["team_roster"]
    assert hierarchy["coscientist_run_id"] == result.coscientist_run_id
    assert hierarchy["candidate_ranking"][0]["drug"] == "tadalafil"
    assert hierarchy["quantitative_baseline"]["candidates"][0]["drug"] == "sildenafil"
    assert hierarchy["llm_calls"] == result.llm_calls
    assert hierarchy["provider"] == result.provider
    assert hierarchy["models"] == result.models
    assert hierarchy["workers"] == result.workers
    assert hierarchy["max_tasks"] == result.max_tasks
    assert hierarchy["seed"] == result.seed


class _InvalidSynthesisBackend(_PortableFakeBackend):
    async def call(self, system, user, **kwargs):
        if "You lead the" in system or "Repair invalid" in system:
            self.calls += 1
            self.requests.append((system, user))
            return "not valid JSON"
        return await super().call(system, user, **kwargs)


@pytest.mark.asyncio
async def test_failed_synthesis_rolls_back_existing_program_lead(tmp_path: Path):
    pf = engine.new_company("Helix", "PAH", seed=5)
    program = engine.add_program(
        pf, "PAH-1", "pulmonary arterial hypertension", seed=5,
    )
    program.stage = Stage.HYPOTHESES
    program.lead_candidate = "existing drug -> existing indication"
    executor = CoScientistStageExecutor(
        _InvalidSynthesisBackend(),
        n_workers=2,
        max_tasks=12,
        max_time_seconds=30,
        db_dir=str(tmp_path / "stage-runs"),
        proximity_agent=_NoopProximity(),
        evidence_provider="offline",
    )

    with pytest.raises(StageExecutionError, match="output contract twice"):
        await executor.execute(program, cycle=1)

    assert program.lead_candidate == "existing drug -> existing indication"
    assert program.lead_drug == "existing drug"
    assert program.lead_indication == "existing indication"


@pytest.mark.asyncio
async def test_hierarchy_executor_forwards_evidence_provider_to_run_system(
    tmp_path: Path, monkeypatch,
):
    """Regression pin for the CRITICAL company/hierarchy.py regression: a real
    hierarchical run injects its backend into run_system() with no `provider=`
    at all (run_system() cannot infer a provider from an already-constructed
    backend object). CoScientistStageExecutor must forward its own
    `evidence_provider` through explicitly -- company/cli.py sets it from the
    provider it already resolved -- or a real, possibly paid run risks
    grounding (and then "verifying") its hypotheses against whatever
    run_system() falls back to, with nothing recording that it happened."""
    import bench.runner as runner

    class _Captured(Exception):
        pass

    captured = {}

    async def spy_run_system(*args, **kwargs):
        captured.update(kwargs)
        raise _Captured("stop before any real Supervisor/backend work")

    monkeypatch.setattr(runner, "run_system", spy_run_system)
    pf = engine.new_company("Helix", "PAH", seed=12)
    program = engine.add_program(pf, "PAH-A", "PAH", seed=12)
    executor = CoScientistStageExecutor(
        _PortableFakeBackend(),
        n_workers=1,
        max_tasks=4,
        max_time_seconds=30,
        db_dir=str(tmp_path / "stage-runs"),
        evidence_provider="anthropic",
    )

    with pytest.raises(_Captured):
        await executor.execute(program, cycle=1)

    assert captured.get("evidence_provider") == "anthropic"
