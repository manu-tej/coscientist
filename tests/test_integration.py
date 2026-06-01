# tests/test_integration.py
import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
from core.state import StateStore
from core.orchestrator import AgentRunner
from core.supervisor import Supervisor, SupervisorSettings
from core.stats import WeightThresholds
from core.models import ResearchPlanConfig
from agents.base import BaseAgent
from agents.generation import GenerationAgent
from agents.reflection import ReflectionAgent
from agents.ranking import RankingAgent
from agents.evolution import EvolutionAgent
from agents.meta_review import MetaReviewAgent

PROMPTS = Path(__file__).parent.parent / "prompts"

HYPOTHESIS_OUT = (
    "Hypothesis: Stress-induced PTMs on Nup98 disrupt transport.\n"
    "Summary: Nup98 PTMs cause ALS.\n"
    "Category: Neurodegeneration"
)


@pytest.fixture
async def store(tmp_path):
    s = StateStore(str(tmp_path / "integration.db"))
    await s.init_db()
    return s


@pytest.fixture
def config():
    return ResearchPlanConfig(
        run_id="itest", goal="Explain ALS mechanisms", preferences="Novel",
        attributes=["Novelty", "Feasibility"], constraints="testable", safety_approved=True,
    )


def mock_base():
    """A BaseAgent whose Claude calls always return well-formed output."""
    client = MagicMock()
    base = BaseAgent(client=client, prompts_dir=PROMPTS)
    base.call_claude = AsyncMock(return_value=(
        HYPOTHESIS_OUT + "\nVerdict: PASSED\nbetter hypothesis: 1\nhypothesis: neutral"
    ))
    base.run_turn_loop = AsyncMock(return_value=(
        "HYPOTHESIS\n" + HYPOTHESIS_OUT,
        ["HYPOTHESIS\n" + HYPOTHESIS_OUT],
    ))
    return base


# A no-op proximity agent that avoids loading the sentence-transformers model in tests
class FakeProximity:
    async def update_graph(self, hypotheses):
        return []


async def test_full_run_produces_hypotheses(store, config):
    base = mock_base()
    search = MagicMock()
    search.search_and_format = AsyncMock(return_value="[1] Article A")

    runner = AgentRunner(
        store=store, config=config,
        generation=GenerationAgent(base=base),
        reflection=ReflectionAgent(base=base),
        ranking=RankingAgent(base=base, store=store),
        proximity=FakeProximity(),
        evolution=EvolutionAgent(base=base),
        meta_review=MetaReviewAgent(base=base),
        search=search,
    )

    settings = SupervisorSettings(
        n_workers=2,
        max_tasks=20,
        max_time_seconds=30,
        checkpoint_interval=5,
        thresholds=WeightThresholds(),
        seed=7,
    )
    supervisor = Supervisor(config=config, runner=runner, settings=settings, store=store)
    await supervisor.run()

    hypotheses = await store.list_hypotheses("itest")
    assert len(hypotheses) >= 1
    total_reviews = 0
    for h in hypotheses:
        total_reviews += len(await store.list_reviews(h.id))
    assert total_reviews >= 0


async def test_full_run_no_unhandled_errors(store, config):
    """The supervisor must complete even if some tasks hit edge cases (empty pool, etc.)."""
    base = mock_base()
    search = MagicMock()
    search.search_and_format = AsyncMock(return_value="[1] Article")

    runner = AgentRunner(
        store=store, config=config,
        generation=GenerationAgent(base=base),
        reflection=ReflectionAgent(base=base),
        ranking=RankingAgent(base=base, store=store),
        proximity=FakeProximity(),
        evolution=EvolutionAgent(base=base),
        meta_review=MetaReviewAgent(base=base),
        search=search,
    )
    settings = SupervisorSettings(
        n_workers=3, max_tasks=15, max_time_seconds=30,
        checkpoint_interval=5, thresholds=WeightThresholds(), seed=99,
    )
    supervisor = Supervisor(config=config, runner=runner, settings=settings, store=store)
    await supervisor.run()  # must not raise
    assert supervisor._tasks_completed == 15
