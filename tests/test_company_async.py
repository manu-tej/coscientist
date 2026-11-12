import asyncio
import copy

import pytest

from company import engine
from company.hierarchy import PortableTokenBudget, SeededStageExecutor
from company.ledger import InsufficientCredits, InsufficientTokens
from company.models import ProgramStatus, Stage


class _ConcurrencySpyExecutor(SeededStageExecutor):
    def __init__(self):
        self.active = 0
        self.max_active = 0

    async def execute(self, *args, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        try:
            return await super().execute(*args, **kwargs)
        finally:
            self.active -= 1


class _MixedFailureExecutor(SeededStageExecutor):
    """Let one program finish (and mutate its lead) before its sibling fails."""

    def __init__(self):
        self.succeeded = []

    async def execute(self, program, *args, **kwargs):
        if program.name == "FAIL":
            await asyncio.sleep(0.02)
            raise RuntimeError("stage team failed")
        result = await super().execute(program, *args, **kwargs)
        self.succeeded.append(program.id)
        return result


class _CostOverrunExecutor(SeededStageExecutor):
    async def execute(self, *args, **kwargs):
        result = await super().execute(*args, **kwargs)
        result.credits_spent = 15.0
        return result


class _TokenOverrunExecutor(SeededStageExecutor):
    async def execute(self, *args, **kwargs):
        result = await super().execute(*args, **kwargs)
        result.tokens_spent = 60
        return result


class _SharedBudgetSpyExecutor(SeededStageExecutor):
    def __init__(self):
        self.created_limits = []
        self.seen_budgets = []

    def create_token_budget(self, limit):
        self.created_limits.append(limit)
        return object()

    async def execute(self, *args, token_budget=None, **kwargs):
        self.seen_budgets.append(token_budget)
        await asyncio.sleep(0)
        return await super().execute(
            *args, token_budget=token_budget, **kwargs,
        )


class _FailedTokenAttemptExecutor:
    def create_token_budget(self, limit):
        return PortableTokenBudget(limit)

    async def execute(self, *args, token_budget=None, **kwargs):
        await token_budget.reserve(10)
        await token_budget.reconcile(10, 10)
        raise RuntimeError("failed after provider call")


class _CancellationSpendExecutor:
    def __init__(self):
        self.spent = asyncio.Event()

    def create_token_budget(self, limit):
        return PortableTokenBudget(limit)

    async def execute(self, *args, token_budget=None, **kwargs):
        await token_budget.reserve(10)
        await token_budget.reconcile(10, 10)
        self.spent.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            exc.attempted_credits = 2.0
            raise


class _InvalidGateResultExecutor(SeededStageExecutor):
    async def execute(self, *args, **kwargs):
        result = await super().execute(*args, **kwargs)
        result.red_flags = "invalid"
        return result


class _SelfCancelledSpendExecutor:
    def create_token_budget(self, limit):
        return PortableTokenBudget(limit)

    async def execute(self, *args, token_budget=None, **kwargs):
        await token_budget.reserve(7)
        await token_budget.reconcile(7, 7)
        error = asyncio.CancelledError()
        error.attempted_credits = 1.5
        raise error


class _SlowCancellationSpendExecutor:
    def __init__(self):
        self.spent = asyncio.Event()
        self.cleanup_started = asyncio.Event()
        self.release_cleanup = asyncio.Event()

    def create_token_budget(self, limit):
        return PortableTokenBudget(limit)

    async def execute(self, *args, token_budget=None, **kwargs):
        await token_budget.reserve(10)
        await token_budget.reconcile(10, 10)
        self.spent.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            exc.attempted_credits = 2.0
            self.cleanup_started.set()
            await self.release_cleanup.wait()
            raise


def _assert_scientific_state_rolled_back(actual, before):
    comparison = copy.deepcopy(actual)
    comparison.company.failed_credit_spent = before.company.failed_credit_spent
    comparison.company.failed_token_spent = before.company.failed_token_spent
    assert comparison == before


@pytest.mark.asyncio
async def test_run_quarter_obeys_approved_cso_allocation():
    pf = engine.new_company("Helix", "PAH", credit_budget=20)
    strong = engine.add_program(pf, "STRONG", "PAH", estimated_value=900)
    weak = engine.add_program(pf, "WEAK", "PAH", estimated_value=50)

    ran = await engine.run_quarter_async(
        pf,
        executor=SeededStageExecutor(),
        approved_allocation={strong.id: 2.0, weak.id: 0.0},
        n_program_workers=2,
    )

    assert [p.id for p, _, _ in ran] == [strong.id]
    assert weak.status is ProgramStatus.HELD
    assert "approved CSO allocation" in weak.history[-1]


@pytest.mark.asyncio
async def test_cso_allocation_can_resume_held_program_and_skips_pending_share():
    from company.cso import propose_allocation

    pf = engine.new_company("Helix", "PAH", credit_budget=100)
    held = engine.add_program(pf, "HELD", "PAH", estimated_value=900)
    pending = engine.add_program(pf, "PENDING", "PAH", estimated_value=800)
    active = engine.add_program(pf, "ACTIVE", "PAH", estimated_value=100)
    held.status = ProgramStatus.HELD
    pending_result, _ = engine.run_program_stage(pf, pending)
    assert pf.pending[pending.id] is pending_result

    allocation = propose_allocation(
        pf.programs,
        engine.Ledger(pf.company).credits_remaining,
        pending_ids=set(pf.pending),
    )
    ran = await engine.run_quarter_async(
        pf,
        executor=SeededStageExecutor(),
        approved_allocation=allocation,
        n_program_workers=2,
    )

    assert held.id in {program.id for program, _, _ in ran}
    assert active.id in {program.id for program, _, _ in ran}
    assert pending.id not in allocation
    assert pending.id not in {program.id for program, _, _ in ran}
    assert held.status is ProgramStatus.ACTIVE


@pytest.mark.asyncio
async def test_program_stage_runs_respect_worker_limit_concurrently():
    pf = engine.new_company("Helix", "PAH", credit_budget=20)
    programs = [
        engine.add_program(pf, name, "PAH")
        for name in ("A", "B", "C")
    ]
    executor = _ConcurrencySpyExecutor()

    ran = await engine.run_quarter_async(
        pf, executor=executor, n_program_workers=2,
    )

    assert len(ran) == 3
    assert executor.max_active == 2
    assert [program for program, _, _ in ran] == programs
    assert all(actual is expected for (actual, _, _), expected in zip(ran, programs))
    assert pf.company.cycle == 1
    assert pf.company.credit_spent == 6.0
    assert set(pf.pending) == {program.id for program in programs}
    assert len(pf.stage_runs) == 3


@pytest.mark.asyncio
async def test_async_quarter_creates_one_shared_token_envelope_before_dispatch():
    pf = engine.new_company(
        "Helix", "PAH", credit_budget=20, token_budget=200_000,
    )
    for name in ("A", "B"):
        engine.add_program(pf, name, "PAH")
    executor = _SharedBudgetSpyExecutor()

    await engine.run_quarter_async(
        pf, executor=executor, n_program_workers=2,
    )

    assert executor.created_limits == [200_000]
    assert len(executor.seen_budgets) == 2
    assert executor.seen_budgets[0] is executor.seen_budgets[1]


@pytest.mark.asyncio
async def test_async_quarter_is_atomic_when_one_of_two_executors_fails():
    pf = engine.new_company("Helix", "PAH", credit_budget=20)
    succeeds = engine.add_program(pf, "SUCCESS", "pulmonary arterial hypertension")
    fails = engine.add_program(pf, "FAIL", "pulmonary arterial hypertension")
    # The hypotheses executor mutates lead_candidate before returning, which makes
    # this test cover executor-side program mutation as well as ledger/archive state.
    succeeds.stage = Stage.HYPOTHESES
    fails.stage = Stage.HYPOTHESES
    before = copy.deepcopy(pf)
    executor = _MixedFailureExecutor()

    with pytest.raises(RuntimeError, match="stage team failed"):
        await engine.run_quarter_async(
            pf, executor=executor, n_program_workers=2,
        )

    assert executor.succeeded == [succeeds.id]
    _assert_scientific_state_rolled_back(pf, before)
    assert pf.programs[0] is succeeds
    assert pf.programs[1] is fails
    assert succeeds.lead_candidate is None
    assert pf.company.cycle == 0
    assert pf.company.credit_spent == 0.0
    assert pf.company.token_spent == 0
    assert pf.company.failed_credit_spent > 0
    assert pf.company.failed_token_spent > 0
    assert not pf.pending
    assert not pf.stage_runs
    assert not pf.experiments


@pytest.mark.asyncio
async def test_async_quarter_rejects_combined_exact_credit_overrun_atomically():
    pf = engine.new_company("Helix", "PAH", credit_budget=20)
    programs = [
        engine.add_program(pf, name, "PAH")
        for name in ("A", "B")
    ]
    before = copy.deepcopy(pf)

    with pytest.raises(InsufficientCredits, match="quarter results need 30.0"):
        await engine.run_quarter_async(
            pf, executor=_CostOverrunExecutor(), n_program_workers=2,
        )

    _assert_scientific_state_rolled_back(pf, before)
    assert pf.company.failed_credit_spent == 30.0
    assert pf.company.failed_token_spent > 0
    assert all(actual is expected for actual, expected in zip(pf.programs, programs))


@pytest.mark.asyncio
async def test_async_quarter_rejects_combined_token_overrun_atomically():
    pf = engine.new_company(
        "Helix", "PAH", credit_budget=20, token_budget=100,
    )
    programs = [
        engine.add_program(pf, name, "PAH")
        for name in ("A", "B")
    ]
    before = copy.deepcopy(pf)

    with pytest.raises(InsufficientTokens, match="120 tokens"):
        await engine.run_quarter_async(
            pf, executor=_TokenOverrunExecutor(), n_program_workers=2,
        )

    _assert_scientific_state_rolled_back(pf, before)
    assert pf.company.failed_credit_spent > 0
    assert pf.company.failed_token_spent == 120
    assert all(actual is expected for actual, expected in zip(pf.programs, programs))


@pytest.mark.asyncio
async def test_async_quarter_enforces_exact_approved_envelope():
    pf = engine.new_company("Helix", "PAH", credit_budget=50)
    program = engine.add_program(pf, "A", "PAH")
    before = copy.deepcopy(pf)

    with pytest.raises(InsufficientCredits, match="approved CSO envelope"):
        await engine.run_quarter_async(
            pf,
            executor=_CostOverrunExecutor(),
            approved_allocation={program.id: 10.0},
        )

    _assert_scientific_state_rolled_back(pf, before)
    assert pf.company.failed_credit_spent == 15.0
    assert pf.company.failed_token_spent > 0


@pytest.mark.asyncio
async def test_failed_provider_attempt_reduces_lifetime_token_budget():
    pf = engine.new_company(
        "Helix",
        "PAH",
        credit_budget=20,
        token_budget=100,
    )
    engine.add_program(pf, "A", "PAH")
    before = copy.deepcopy(pf)

    with pytest.raises(RuntimeError, match="failed after provider call"):
        await engine.run_quarter_async(
            pf,
            executor=_FailedTokenAttemptExecutor(),
            n_program_workers=1,
        )

    _assert_scientific_state_rolled_back(pf, before)
    assert pf.company.failed_token_spent == 10
    assert engine.Ledger(pf.company).tokens_remaining == 90


@pytest.mark.asyncio
async def test_retry_cannot_reset_failed_attempt_token_usage():
    pf = engine.new_company(
        "Helix",
        "PAH",
        credit_budget=20,
        token_budget=15,
    )
    engine.add_program(pf, "A", "PAH")
    executor = _FailedTokenAttemptExecutor()

    with pytest.raises(RuntimeError, match="failed after provider call"):
        await engine.run_quarter_async(
            pf,
            executor=executor,
            n_program_workers=1,
        )
    assert pf.company.failed_token_spent == 10

    with pytest.raises(InsufficientTokens):
        await engine.run_quarter_async(
            pf,
            executor=executor,
            n_program_workers=1,
        )

    assert pf.company.failed_token_spent == 10
    assert engine.Ledger(pf.company).tokens_remaining == 5


@pytest.mark.asyncio
async def test_failed_single_stage_attempt_is_also_charged():
    pf = engine.new_company(
        "Helix",
        "PAH",
        credit_budget=20,
        token_budget=100,
    )
    program = engine.add_program(pf, "A", "PAH")

    with pytest.raises(RuntimeError, match="failed after provider call"):
        await engine.run_program_stage_async(
            pf,
            program,
            executor=_FailedTokenAttemptExecutor(),
        )

    assert pf.company.failed_token_spent == 10
    assert engine.Ledger(pf.company).tokens_remaining == 90
    assert not pf.pending
    assert not pf.stage_runs


@pytest.mark.asyncio
async def test_cancelled_quarter_drains_workers_and_charges_measured_usage():
    pf = engine.new_company(
        "Helix",
        "PAH",
        credit_budget=20,
        token_budget=100,
    )
    engine.add_program(pf, "A", "PAH")
    before = copy.deepcopy(pf)
    executor = _CancellationSpendExecutor()
    task = asyncio.create_task(engine.run_quarter_async(
        pf,
        executor=executor,
        n_program_workers=1,
    ))
    await executor.spent.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    _assert_scientific_state_rolled_back(pf, before)
    assert pf.company.failed_credit_spent == 2.0
    assert pf.company.failed_token_spent == 10
    assert engine.Ledger(pf.company).tokens_remaining == 90


@pytest.mark.asyncio
async def test_worker_cancellation_metadata_survives_gather():
    pf = engine.new_company(
        "Helix",
        "PAH",
        credit_budget=20,
        token_budget=100,
    )
    engine.add_program(pf, "A", "PAH")
    before = copy.deepcopy(pf)

    with pytest.raises(asyncio.CancelledError):
        await engine.run_quarter_async(
            pf,
            executor=_SelfCancelledSpendExecutor(),
            n_program_workers=1,
        )

    _assert_scientific_state_rolled_back(pf, before)
    assert pf.company.failed_credit_spent == 1.5
    assert pf.company.failed_token_spent == 7


@pytest.mark.asyncio
async def test_repeated_caller_cancellation_cannot_erase_attempt_usage():
    pf = engine.new_company(
        "Helix",
        "PAH",
        credit_budget=20,
        token_budget=100,
    )
    engine.add_program(pf, "A", "PAH")
    before = copy.deepcopy(pf)
    executor = _SlowCancellationSpendExecutor()
    task = asyncio.create_task(engine.run_quarter_async(
        pf,
        executor=executor,
        n_program_workers=1,
    ))
    await executor.spent.wait()

    task.cancel()
    await executor.cleanup_started.wait()
    task.cancel()
    executor.release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    _assert_scientific_state_rolled_back(pf, before)
    assert pf.company.failed_credit_spent == 2.0
    assert pf.company.failed_token_spent == 10


@pytest.mark.asyncio
async def test_failed_direct_commit_rolls_back_all_scientific_state():
    pf = engine.new_company("Helix", "PAH", credit_budget=20)
    program = engine.add_program(pf, "A", "PAH")
    before = copy.deepcopy(pf)

    with pytest.raises(ValueError, match="red_flags"):
        await engine.run_program_stage_async(
            pf,
            program,
            executor=_InvalidGateResultExecutor(),
        )

    _assert_scientific_state_rolled_back(pf, before)
    assert pf.programs[0] is program
    assert pf.company.credit_spent == 0.0
    assert pf.company.token_spent == 0
    assert pf.company.failed_credit_spent == 2.0
    assert pf.company.failed_token_spent > 0
    assert not pf.pending
    assert not pf.stage_runs
    assert not pf.experiments


@pytest.mark.asyncio
async def test_async_quarter_rejects_zero_program_workers():
    pf = engine.new_company("Helix", "PAH")
    with pytest.raises(ValueError, match="n_program_workers"):
        await engine.run_quarter_async(
            pf, executor=SeededStageExecutor(), n_program_workers=0,
        )
