from typing import Optional
from agents.base import BaseAgent
from agents.generation import _parse_hypothesis_output, call_until_contract
from core.models import Hypothesis, ResearchPlanConfig


def _parse_evolved_hypothesis(
    text: str,
    config: ResearchPlanConfig,
    method: str,
    evolved_from: Optional[str] = None,
) -> Hypothesis:
    parsed = _parse_hypothesis_output(text, config, method)
    parsed.evolved_from = evolved_from
    return parsed


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
        response = await call_until_contract(self.base, "You are an expert scientific researcher.", prompt)
        return _parse_evolved_hypothesis(response, config, "grounding", hypothesis.id)

    async def run_coherence(self, hypothesis: Hypothesis, config: ResearchPlanConfig) -> Hypothesis:
        prompt = self.base.render_prompt(
            "evolution/coherence",
            goal=config.goal,
            preferences=config.preferences,
            hypothesis=hypothesis.text,
        )
        response = await call_until_contract(self.base, "You are an expert scientific researcher.", prompt)
        return _parse_evolved_hypothesis(response, config, "coherence", hypothesis.id)

    async def run_inspiration(self, hypothesis: Hypothesis, config: ResearchPlanConfig) -> Hypothesis:
        prompt = self.base.render_prompt(
            "evolution/inspiration",
            goal=config.goal,
            preferences=config.preferences,
            hypothesis=hypothesis.text,
        )
        response = await call_until_contract(self.base, "You are an expert scientific researcher.", prompt)
        return _parse_evolved_hypothesis(response, config, "inspiration", hypothesis.id)

    async def run_combination(
        self,
        hypotheses: list[Hypothesis],
        config: ResearchPlanConfig,
        *,
        ranking_scores: dict[str, float] | None = None,
        ranking_label: str = "Elo",
    ) -> Hypothesis:
        scores = (
            ranking_scores
            if ranking_scores is not None
            else {
                hypothesis.id: hypothesis.elo_rating
                for hypothesis in hypotheses
            }
        )
        hypotheses_text = "\n\n---\n\n".join(
            (
                f"Hypothesis {i+1} ({ranking_label}: "
                f"{scores[h.id]:.2f}):\n{h.text}"
                if h.id in scores
                else f"Hypothesis {i+1} ({ranking_label}: unscored):\n{h.text}"
            )
            for i, h in enumerate(hypotheses)
        )
        prompt = self.base.render_prompt(
            "evolution/combination",
            goal=config.goal,
            preferences=config.preferences,
            hypotheses=hypotheses_text,
        )
        response = await call_until_contract(self.base, "You are an expert scientific researcher.", prompt)
        return _parse_evolved_hypothesis(response, config, "combination")

    async def run_simplification(self, hypothesis: Hypothesis, config: ResearchPlanConfig) -> Hypothesis:
        prompt = self.base.render_prompt(
            "evolution/simplification",
            goal=config.goal,
            preferences=config.preferences,
            hypothesis=hypothesis.text,
        )
        response = await call_until_contract(self.base, "You are an expert scientific researcher.", prompt)
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
        response = await call_until_contract(self.base, "You are an expert scientific researcher.", prompt)
        return _parse_evolved_hypothesis(response, config, "out_of_box")
