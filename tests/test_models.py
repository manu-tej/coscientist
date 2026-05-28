import pytest
from core.models import Hypothesis, ResearchPlanConfig, AgentTask, AgentType, ReviewVerdict


def test_hypothesis_defaults():
    h = Hypothesis(
        id="h1",
        run_id="run1",
        text="Test hypothesis text",
        summary="A test hypothesis",
        generation_method="literature",
        source="system",
    )
    assert h.elo_rating == 1200.0
    assert h.status == "active"
    assert h.evolved_from is None
    assert h.annotations == []


def test_research_plan_config_idea_attributes():
    config = ResearchPlanConfig(
        run_id="run1",
        goal="Find a cure for ALS",
        preferences="Focus on novel mechanisms",
        attributes=["Novelty", "Feasibility"],
        constraints="Must be testable in vitro",
        safety_approved=True,
    )
    assert config.idea_attributes == "Novelty and Feasibility"


def test_agent_task_ordering():
    t1 = AgentTask(priority=1, agent_type=AgentType.GENERATION, run_id="run1")
    t2 = AgentTask(priority=2, agent_type=AgentType.REFLECTION, run_id="run1")
    assert t1 < t2
