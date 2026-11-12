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


@pytest.mark.parametrize("workers", [0, -1])
def test_supervisor_rejects_nonpositive_worker_count(workers):
    with pytest.raises(ValueError, match="n_workers"):
        SupervisorSettings(n_workers=workers)


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
    assert len(supervisor.task_failures) == 1
    assert sum(supervisor.task_counts.values()) == 3


async def test_supervisor_bounds_queued_work_for_backpressure(config):
    """A slow provider must not let the scheduler dispatch the entire run from stale stats."""
    release = asyncio.Event()
    started = 0

    class SlowRunner:
        async def run_task(self, task):
            nonlocal started
            started += 1
            await release.wait()

    settings = SupervisorSettings(
        n_workers=2, max_tasks=30, max_time_seconds=10,
        checkpoint_interval=2, max_queue_size=2,
        thresholds=WeightThresholds(), seed=42,
    )
    supervisor = Supervisor(config=config, runner=SlowRunner(), settings=settings)
    supervisor._compute_stats = AsyncMock(return_value=MagicMock(
        n_hypotheses=0, n_pending_review=0, elo_variance=6000.0,
        last_meta_review_age=0,
    ))
    run = asyncio.create_task(supervisor.run())
    for _ in range(100):
        if started == 2 and supervisor.task_queue.full():
            break
        await asyncio.sleep(0)
    # Two running + two queued. The fifth put may be waiting, but is not dispatched.
    assert supervisor._tasks_dispatched <= 4
    release.set()
    await run
    assert supervisor._tasks_completed == 30


async def test_supervisor_wall_clock_limit_cancels_stuck_work(config):
    never = asyncio.Event()

    class StuckRunner:
        async def run_task(self, task):
            await never.wait()

    settings = SupervisorSettings(
        n_workers=2, max_tasks=30, max_time_seconds=0.02,
        checkpoint_interval=2, max_queue_size=2,
        thresholds=WeightThresholds(), seed=42,
    )
    supervisor = Supervisor(config=config, runner=StuckRunner(), settings=settings)
    supervisor._compute_stats = AsyncMock(return_value=MagicMock(
        n_hypotheses=0, n_pending_review=0, elo_variance=6000.0,
        last_meta_review_age=0,
    ))

    with pytest.raises(TimeoutError, match="Supervisor exceeded"):
        await supervisor.run()


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


async def test_supervisor_rotates_personas_and_records_task_status(config, runner):
    settings = SupervisorSettings(
        n_workers=2,
        max_tasks=4,
        max_time_seconds=10,
        checkpoint_interval=2,
        thresholds=WeightThresholds(),
        seed=42,
        personas=("biologist", "chemist"),
    )
    supervisor = Supervisor(config=config, runner=runner, settings=settings)
    supervisor._compute_stats = AsyncMock(return_value=MagicMock(
        n_hypotheses=0,
        n_pending_review=0,
        elo_variance=6000.0,
        last_meta_review_age=0,
    ))

    await supervisor.run()

    assert [trace["persona"] for trace in supervisor.agent_task_trace] == [
        "biologist", "chemist", "biologist", "chemist",
    ]
    assert [trace["priority"] for trace in supervisor.agent_task_trace] == [
        0, 1, 2, 3,
    ]
    assert all(
        trace["status"] == "succeeded" and trace["error"] is None
        for trace in supervisor.agent_task_trace
    )


def test_supervisor_rejects_invalid_persona_roster():
    with pytest.raises(ValueError, match="personas"):
        SupervisorSettings(personas=("biologist", ""))
    with pytest.raises(ValueError, match="personas"):
        SupervisorSettings(personas="biologist")
