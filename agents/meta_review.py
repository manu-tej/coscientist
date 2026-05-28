from agents.base import BaseAgent
from core.models import Hypothesis, ResearchPlanConfig


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
    ) -> str:
        top_hypotheses = "\n\n---\n\n".join(
            f"Hypothesis (Elo: {h.elo_rating:.0f}):\n{h.summary}\n\n{h.text}"
            for h in sorted(hypotheses, key=lambda x: x.elo_rating, reverse=True)[:10]
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
