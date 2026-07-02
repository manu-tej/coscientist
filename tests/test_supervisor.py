# tests/test_supervisor.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from core.supervisor import Supervisor, SupervisorSettings
from core.stats import WeightThresholds
from core.models import ResearchPlanConfig, AgentTask, AgentType


@pytest.fixture
def config():
    return ResearchPlanConfig(
        run_id="run1", goal="ALS", preferences="Novel",
        attributes=["Novelty"], constraints="testable", safety_approved=True,
    )


def make_settings(max_tasks=5):
    return SupervisorSettings(
        n_workers=2,
        max_tasks=max_tasks,
        max_time_seconds=10,
        checkpoint_interval=2,
        thresholds=WeightThresholds(),
        seed=42,
    )


@pytest.fixture
def runner():
    r = MagicMock()
    r.run_task = AsyncMock(return_value=None)
    return r


async def test_supervisor_dispatches_tasks(config, runner):
    settings = make_settings(max_tasks=5)
    supervisor = Supervisor(config=config, runner=runner, settings=settings)
    supervisor._compute_stats = AsyncMock(return_value=MagicMock(
        n_hypotheses=0, n_pending_review=0, elo_variance=6000.0, last_meta_review_age=0,
    ))
    await supervisor.run()
    assert runner.run_task.call_count == 5


async def test_supervisor_stops_at_max_tasks(config, runner):
    settings = make_settings(max_tasks=3)
    supervisor = Supervisor(config=config, runner=runner, settings=settings)
    supervisor._compute_stats = AsyncMock(return_value=MagicMock(
        n_hypotheses=0, n_pending_review=0, elo_variance=6000.0, last_meta_review_age=0,
    ))
    await supervisor.run()
    assert runner.run_task.call_count == 3


async def test_supervisor_handles_task_errors_gracefully(config):
    settings = make_settings(max_tasks=3)
    runner = MagicMock()
    runner.run_task = AsyncMock(side_effect=[ValueError("boom"), None, None])
    supervisor = Supervisor(config=config, runner=runner, settings=settings)
    supervisor._compute_stats = AsyncMock(return_value=MagicMock(
        n_hypotheses=0, n_pending_review=0, elo_variance=6000.0, last_meta_review_age=0,
    ))
    await supervisor.run()
    assert runner.run_task.call_count == 3


def test_supervisor_rotates_generation_strategies(config, runner):
    settings = make_settings(max_tasks=4)
    supervisor = Supervisor(config=config, runner=runner, settings=settings)
    t1 = supervisor._build_task(AgentType.GENERATION)
    t2 = supervisor._build_task(AgentType.GENERATION)
    t3 = supervisor._build_task(AgentType.GENERATION)
    assert t1.strategy == "debate"
    assert t2.strategy == "literature"
    assert t3.strategy == "assumptions"


def test_supervisor_rotates_evolution_strategies(config, runner):
    settings = make_settings(max_tasks=4)
    supervisor = Supervisor(config=config, runner=runner, settings=settings)
    e1 = supervisor._build_task(AgentType.EVOLUTION)
    e2 = supervisor._build_task(AgentType.EVOLUTION)
    assert e1.strategy == "grounding"
    assert e2.strategy == "coherence"


# --- Bug 9: unbounded dispatch queue defeats adaptive weighting ---

def _fresh_stats():
    return MagicMock(
        n_hypotheses=0, n_pending_review=0, elo_variance=6000.0, last_meta_review_age=0,
    )


def test_dispatch_queue_bounded_to_worker_count(config, runner):
    settings = make_settings(max_tasks=5)
    supervisor = Supervisor(config=config, runner=runner, settings=settings)
    assert supervisor.task_queue.maxsize != 0  # 0 means unbounded
    assert supervisor.task_queue.maxsize == supervisor.settings.n_workers


async def test_dispatch_backpressures_on_slow_workers(config):
    """Dispatch must stall on slow workers instead of queueing every task upfront,
    so SystemStats can update between dispatches and weights can adapt."""
    settings = make_settings(max_tasks=10)
    settings.max_time_seconds = 60
    release = asyncio.Event()

    async def slow_task(task):
        await release.wait()

    runner = MagicMock()
    runner.run_task = AsyncMock(side_effect=slow_task)
    supervisor = Supervisor(config=config, runner=runner, settings=settings)
    supervisor._compute_stats = AsyncMock(return_value=_fresh_stats())
    run = asyncio.create_task(supervisor.run())
    # Deterministic settle: yield the loop repeatedly; no worker can finish
    # until `release` is set, so dispatch progress is bounded, not timed.
    for _ in range(50):
        await asyncio.sleep(0)
    # At most n_workers in-flight + n_workers (maxsize) waiting in the queue.
    assert supervisor._tasks_dispatched <= settings.n_workers * 2
    release.set()
    await asyncio.wait_for(run, timeout=5)
    assert runner.run_task.await_count == 10


async def test_drain_respects_time_budget(config, monkeypatch):
    """Once the time budget is exhausted, run() must not wait unboundedly on the
    backlog: it should time the drain out and cancel workers."""
    from types import SimpleNamespace
    from core import supervisor as supervisor_mod

    fake = SimpleNamespace(t=0.0)
    monkeypatch.setattr(
        supervisor_mod, "time", SimpleNamespace(monotonic=lambda: fake.t)
    )
    settings = make_settings(max_tasks=10)
    settings.max_time_seconds = 25

    async def never_finish(task):
        await asyncio.Event().wait()

    runner = MagicMock()
    runner.run_task = AsyncMock(side_effect=never_finish)
    supervisor = Supervisor(config=config, runner=runner, settings=settings)

    def advance_and_stats():
        fake.t += 10.0  # budget (25) trips after the third dispatch
        return _fresh_stats()

    supervisor._compute_stats = AsyncMock(side_effect=advance_and_stats)
    # Without the drain timeout this hangs on task_queue.join() forever.
    await asyncio.wait_for(supervisor.run(), timeout=5)
    assert supervisor._tasks_dispatched == 3
    assert runner.run_task.await_count <= 3
