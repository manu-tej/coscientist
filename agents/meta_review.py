from agents.base import BaseAgent
from core.models import Hypothesis, ResearchPlanConfig
from core.ranking import native_ranking_key


class MetaReviewAgent:
    def __init__(self, base: BaseAgent):
        self.base = base

    async def run_meta_critique(
        self,
        config: ResearchPlanConfig,
        reviews_text: str,
        instructions: str = "",
    ) -> str:
        prompt = self.base.render_prompt(
            "meta_review/meta_critique",
            goal=config.goal,
            preferences=config.preferences,
            instructions=instructions,
            reviews=reviews_text,
        )
        return await self.base.call_claude(
            "You are an expert in scientific research and meta-analysis.", prompt
        )

    async def run_research_overview(
        self,
        config: ResearchPlanConfig,
        hypotheses: list[Hypothesis],
        *,
        ranking_scores: dict[str, float] | None = None,
        ranking_label: str = "Elo",
    ) -> str:
        scores = (
            ranking_scores
            if ranking_scores is not None
            else {
                hypothesis.id: hypothesis.elo_rating
                for hypothesis in hypotheses
            }
        )
        ordered = sorted(
            hypotheses,
            key=lambda hypothesis: native_ranking_key(
                hypothesis, scores.get(hypothesis.id)
            ),
        )
        top_hypotheses = "\n\n---\n\n".join(
            (
                f"Hypothesis ({ranking_label}: {scores[h.id]:.2f}):\n"
                f"{h.summary}\n\n{h.text}"
                if h.id in scores
                else f"Hypothesis ({ranking_label}: unscored):\n"
                f"{h.summary}\n\n{h.text}"
            )
            for h in ordered[:10]
        )
        prompt = self.base.render_prompt(
            "meta_review/research_overview",
            goal=config.goal,
            top_hypotheses=top_hypotheses,
        )
        return await self.base.call_claude(
            "You are an expert scientific researcher.", prompt
        )

    async def run_research_contacts(
        self,
        config: ResearchPlanConfig,
        cited_literature: str,
    ) -> str:
        prompt = self.base.render_prompt(
            "meta_review/research_contacts",
            goal=config.goal,
            cited_literature=cited_literature,
        )
        return await self.base.call_claude(
            "You are an expert scientific researcher.", prompt
        )
