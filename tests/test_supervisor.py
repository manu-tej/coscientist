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
