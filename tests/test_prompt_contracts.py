"""Every generation prompt must ask for the fields its parser demands.

`debate` and `assumptions` shipped asking only for a free-form "HYPOTHESIS"
exposition while `_parse_hypothesis_output` required `Hypothesis:` and `Summary:`
labels. The mismatch is invisible offline — fixture backends emit whatever the
parser wants — and cost 5/5 generation tasks on the first working live run
(2026-08-04). These tests bind prompt text to parser contract.
"""
from pathlib import Path

import pytest

PROMPTS = Path(__file__).parent.parent / "prompts" / "generation"

# Strategies whose output is fed to agents.generation._parse_hypothesis_output.
PARSED_STRATEGIES = ["debate", "assumptions", "literature", "expansion"]


@pytest.mark.parametrize("strategy", PARSED_STRATEGIES)
def test_prompt_requests_every_required_field(strategy):
    text = (PROMPTS / f"{strategy}.txt").read_text()
    assert "Hypothesis:" in text, (
        f"{strategy}.txt never asks for `Hypothesis:`, but the parser requires it"
    )
    assert "Summary" in text, (
        f"{strategy}.txt never asks for a summary, but the parser requires `Summary:`"
    )


@pytest.mark.parametrize("strategy", ["debate", "assumptions"])
def test_turn_loop_prompts_state_the_termination_signal(strategy):
    """These two run through run_turn_loop, which terminates on a literal HYPOTHESIS."""
    text = (PROMPTS / f"{strategy}.txt").read_text()
    assert "HYPOTHESIS" in text


def test_parser_accepts_the_shape_the_prompts_request():
    """The documented shape must actually parse — not just be mentioned."""
    from agents.generation import _parse_hypothesis_output
    from core.models import ResearchPlanConfig

    documented = (
        "HYPOTHESIS\n\n"
        "Hypothesis: TGF-beta driven PDE5A upregulation sustains stellate cell "
        "activation.\n\n"
        "Rationale and Specificity: SMAD3 occupies the PDE5A promoter.\n\n"
        "Experimental Design: SMAD3 ChIP in TGF-beta stimulated primary human HSCs.\n\n"
        "Summary (one sentence for non-experts): PDE5 inhibition may be antifibrotic.\n\n"
        "Category: Hepatic fibrosis\n"
    )
    config = ResearchPlanConfig(
        run_id="r", goal="g", preferences="p", attributes=["Novelty"],
        constraints="c", safety_approved=True,
    )
    h = _parse_hypothesis_output(documented, config, "debate")
    assert h.summary.startswith("PDE5 inhibition")
    assert h.category == "Hepatic fibrosis"


EVOLUTION_PROMPTS = [
    "coherence", "combination", "grounding",
    "inspiration", "out_of_box", "simplification",
]


@pytest.mark.parametrize("strategy", EVOLUTION_PROMPTS)
def test_evolution_prompt_requests_required_fields(strategy):
    """Evolution feeds _parse_hypothesis_output too, and is single-shot: a prompt
    that never asks for the labels fails the task outright. coherence and
    out_of_box shipped without them (live run, 2026-08-09)."""
    text = (Path(__file__).parent.parent / "prompts" / "evolution"
            / f"{strategy}.txt").read_text()
    assert "Hypothesis:" in text, f"evolution/{strategy}.txt never asks for `Hypothesis:`"
    assert "Summary" in text, f"evolution/{strategy}.txt never asks for a summary"


async def test_single_shot_generation_retries_before_failing():
    """literature/expansion have no turn loop; a single malformed response must
    cost one retry, not the whole task."""
    from unittest.mock import AsyncMock, MagicMock
    from agents.base import BaseAgent
    from agents.generation import GenerationAgent
    from core.models import ResearchPlanConfig

    base = BaseAgent(client=MagicMock(), prompts_dir=Path(__file__).parent.parent / "prompts")
    base.call_claude = AsyncMock(side_effect=[
        "Here is a lovely essay with no labels at all.",
        "Hypothesis: a real mechanism\nSummary: it works\nCategory: bio",
    ])
    config = ResearchPlanConfig(
        run_id="r", goal="g", preferences="p", attributes=["Novelty"],
        constraints="c", safety_approved=True,
    )
    agent = GenerationAgent(base=base)
    h = await agent.run_literature(config, articles_with_reasoning="[1] A")
    assert h.summary == "it works"
    assert base.call_claude.await_count == 2
