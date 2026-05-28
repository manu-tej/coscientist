import uuid
import re
from typing import Optional
from agents.base import BaseAgent
from core.models import Hypothesis, ResearchPlanConfig


def _parse_evolved_hypothesis(
    text: str,
    config: ResearchPlanConfig,
    method: str,
    evolved_from: Optional[str] = None,
) -> Hypothesis:
    def extract(field):
        match = re.search(
            rf"^{re.escape(field)}:\s*(.+?)(?=\n[A-Z][a-zA-Z]|\Z)",
            text, re.MULTILINE | re.DOTALL
        )
        return match.group(1).strip() if match else ""

    summary = extract("Summary") or text[:120].strip()
    category = extract("Category") or None
    return Hypothesis(
        id=str(uuid.uuid4()),
        run_id=config.run_id,
        text=text,
        summary=summary,
        category=category,
        generation_method=method,
        evolved_from=evolved_from,
        source="system",
    )


class EvolutionAgent:
    def __init__(self, base: BaseAgent):
        self.base = base

    async def run_grounding(
        self,
        hypothesis: Hypothesis,
        config: ResearchPlanConfig,
        weaknesses: str,
        articles: str,
    ) -> Hypothesis:
        prompt = self.base.render_prompt(
            "evolution/grounding",
            goal=config.goal,
            preferences=config.preferences,
            hypothesis=hypothesis.text,
            weaknesses=weaknesses,
            articles_with_reasoning=articles,
        )
        response = await self.base.call_claude("You are an expert scientific researcher.", prompt)
        return _parse_evolved_hypothesis(response, config, "grounding", hypothesis.id)

    async def run_coherence(self, hypothesis: Hypothesis, config: ResearchPlanConfig) -> Hypothesis:
        prompt = self.base.render_prompt(
            "evolution/coherence",
            goal=config.goal,
            preferences=config.preferences,
            hypothesis=hypothesis.text,
        )
        response = await self.base.call_claude("You are an expert scientific researcher.", prompt)
        return _parse_evolved_hypothesis(response, config, "coherence", hypothesis.id)

    async def run_inspiration(self, hypothesis: Hypothesis, config: ResearchPlanConfig) -> Hypothesis:
        prompt = self.base.render_prompt(
            "evolution/inspiration",
            goal=config.goal,
            preferences=config.preferences,
            hypothesis=hypothesis.text,
        )
        response = await self.base.call_claude("You are an expert scientific researcher.", prompt)
        return _parse_evolved_hypothesis(response, config, "inspiration", hypothesis.id)

    async def run_combination(
        self, hypotheses: list[Hypothesis], config: ResearchPlanConfig
    ) -> Hypothesis:
        hypotheses_text = "\n\n---\n\n".join(
            f"Hypothesis {i+1} (Elo: {h.elo_rating:.0f}):\n{h.text}"
            for i, h in enumerate(hypotheses)
        )
        prompt = self.base.render_prompt(
            "evolution/combination",
            goal=config.goal,
            preferences=config.preferences,
            hypotheses=hypotheses_text,
        )
        response = await self.base.call_claude("You are an expert scientific researcher.", prompt)
        return _parse_evolved_hypothesis(response, config, "combination")

    async def run_simplification(self, hypothesis: Hypothesis, config: ResearchPlanConfig) -> Hypothesis:
        prompt = self.base.render_prompt(
            "evolution/simplification",
            goal=config.goal,
            preferences=config.preferences,
            hypothesis=hypothesis.text,
        )
        response = await self.base.call_claude("You are an expert scientific researcher.", prompt)
        return _parse_evolved_hypothesis(response, config, "simplification", hypothesis.id)

    async def run_out_of_box(
        self, hypotheses: list[Hypothesis], config: ResearchPlanConfig
    ) -> Hypothesis:
        hypotheses_text = "\n\n---\n\n".join(
            f"Concept {i+1}: {h.summary}" for i, h in enumerate(hypotheses)
        )
        prompt = self.base.render_prompt(
            "evolution/out_of_box",
            goal=config.goal,
            preferences=config.preferences,
            hypotheses=hypotheses_text,
        )
        response = await self.base.call_claude("You are an expert scientific researcher.", prompt)
        return _parse_evolved_hypothesis(response, config, "out_of_box")
