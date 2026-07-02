from __future__ import annotations

import re
import uuid

from agents.base import BaseAgent
from core.models import Hypothesis, ResearchPlanConfig
from tools.llm import LLMBackendError


def _extract_field(text: str, field: str, default: str = "") -> str:
    """Extract a labeled field from structured text output."""
    pattern = rf"^{re.escape(field)}:\s*(.+?)(?=\n[A-Z][a-zA-Z]|\Z)"
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else default


def _parse_hypothesis_output(
    text: str, config: ResearchPlanConfig, method: str
) -> Hypothesis:
    """Parse raw LLM output into a Hypothesis dataclass."""
    # Trim to everything from the HYPOTHESIS marker onwards if present
    if "HYPOTHESIS" in text:
        text = text[text.index("HYPOTHESIS"):]

    summary = _extract_field(text, "Summary") or text[:120].strip()
    category = _extract_field(text, "Category")

    return Hypothesis(
        id=str(uuid.uuid4()),
        run_id=config.run_id,
        text=text,
        summary=summary,
        category=category or None,
        generation_method=method,
        source="system",
    )


class GenerationAgent:
    """Agent responsible for generating novel scientific hypotheses."""

    def __init__(self, base: BaseAgent):
        self.base = base

    # ------------------------------------------------------------------
    # Literature strategy
    # ------------------------------------------------------------------

    async def run_literature(
        self,
        config: ResearchPlanConfig,
        articles_with_reasoning: str,
        source_hypothesis: str = "",
        instructions: str = "",
    ) -> Hypothesis:
        """Generate a hypothesis grounded in a literature review."""
        prompt = self.base.render_prompt(
            "generation/literature",
            goal=config.goal,
            preferences=config.preferences,
            source_hypothesis=source_hypothesis,
            instructions=instructions,
            articles_with_reasoning=articles_with_reasoning,
        )
        response = await self.base.call_claude(
            "You are an expert scientific researcher.", prompt
        )
        return _parse_hypothesis_output(response, config, "literature")

    # ------------------------------------------------------------------
    # Debate strategy
    # ------------------------------------------------------------------

    async def run_debate(
        self,
        config: ResearchPlanConfig,
        reviews_overview: str = "",
        instructions: str = "",
    ) -> Hypothesis:
        """Generate a hypothesis via multi-turn expert debate simulation."""
        _, transcript = await self.base.run_turn_loop(
            template_name="generation/debate",
            variables={
                "goal": config.goal,
                "preferences": config.preferences,
                "idea_attributes": config.idea_attributes,
                "instructions": instructions,
                "reviews_overview": reviews_overview,
            },
            transcript_key="transcript",
            termination_signal="HYPOTHESIS",
            max_turns=10,
            system_prompt="You are an expert scientific researcher.",
        )
        final_text = transcript[-1] if transcript else ""
        if "HYPOTHESIS" not in final_text:
            raise LLMBackendError(
                "debate turn loop exhausted without reaching the HYPOTHESIS "
                "termination signal; refusing to store an unfinished hypothesis."
            )
        return _parse_hypothesis_output(final_text, config, "debate")

    # ------------------------------------------------------------------
    # Assumptions strategy
    # ------------------------------------------------------------------

    async def run_assumptions(
        self,
        config: ResearchPlanConfig,
        instructions: str = "",
    ) -> Hypothesis:
        """Generate a hypothesis by chaining testable assumptions."""
        _, transcript = await self.base.run_turn_loop(
            template_name="generation/assumptions",
            variables={
                "goal": config.goal,
                "preferences": config.preferences,
                "instructions": instructions,
            },
            transcript_key="transcript",
            termination_signal="HYPOTHESIS",
            max_turns=10,
            system_prompt="You are an expert scientific researcher.",
        )
        final_text = transcript[-1] if transcript else ""
        if "HYPOTHESIS" not in final_text:
            raise LLMBackendError(
                "assumptions turn loop exhausted without reaching the HYPOTHESIS "
                "termination signal; refusing to store an unfinished hypothesis."
            )
        return _parse_hypothesis_output(final_text, config, "assumptions")

    # ------------------------------------------------------------------
    # Expansion strategy
    # ------------------------------------------------------------------

    async def run_expansion(
        self,
        config: ResearchPlanConfig,
        research_overview: str,
        existing_hypotheses_summary: str,
        instructions: str = "",
    ) -> Hypothesis:
        """Generate a hypothesis that fills gaps in the existing hypothesis pool."""
        prompt = self.base.render_prompt(
            "generation/expansion",
            goal=config.goal,
            preferences=config.preferences,
            instructions=instructions,
            research_overview=research_overview,
            existing_hypotheses_summary=existing_hypotheses_summary,
        )
        response = await self.base.call_claude(
            "You are an expert scientific researcher.", prompt
        )
        return _parse_hypothesis_output(response, config, "expansion")
