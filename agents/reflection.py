from __future__ import annotations

import json
import re
import uuid
from typing import Optional

from agents.base import LABEL_DECO, AgentOutputError, BaseAgent
from core.models import Hypothesis, ResearchPlanConfig, Review


def _parse_verdict(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise AgentOutputError("reflection returned no verdict")
    for match in re.finditer(r"\{", text):
        try:
            value, _ = json.JSONDecoder().raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "verdict" in value:
            verdict = str(value["verdict"]).strip().lower()
            if verdict in {"passed", "rejected", "flagged"}:
                return verdict
            raise AgentOutputError(f"unknown reflection verdict {verdict!r}")
    match = re.search(
        rf"^\s*{LABEL_DECO}verdict{LABEL_DECO}:{LABEL_DECO}(passed|rejected|flagged)\b",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if match:
        return match.group(1).lower()
    raise AgentOutputError(
        "reflection output must contain an explicit PASSED/REJECTED/FLAGGED verdict"
    )


def _extract_observation(text: str) -> Optional[str]:
    """Return the line containing 'missing piece' if present, else None."""
    if "missing piece" in text.lower():
        lines = text.strip().split("\n")
        for line in lines:
            if "missing piece" in line.lower():
                return line.strip()
    return None


class ReflectionAgent:
    """Multi-tier scientific hypothesis reviewer."""

    def __init__(self, base: BaseAgent):
        self.base = base

    async def run_initial_review(
        self, hypothesis: Hypothesis, config: ResearchPlanConfig
    ) -> Review:
        """Tier 1 — fast initial screening (no external tools)."""
        prompt = self.base.render_prompt(
            "reflection/initial",
            goal=config.goal,
            preferences=config.preferences,
            hypothesis=hypothesis.text,
        )
        response = await self.base.call_claude(
            "You are a scientific peer reviewer.", prompt
        )
        return Review(
            id=str(uuid.uuid4()),
            hypothesis_id=hypothesis.id,
            tier=1,
            critique=response,
            verdict=_parse_verdict(response),
        )

    async def run_full_review(
        self,
        hypothesis: Hypothesis,
        config: ResearchPlanConfig,
        articles_with_reasoning: str,
    ) -> Review:
        """Tier 2 — full literature-grounded review."""
        prompt = self.base.render_prompt(
            "reflection/full",
            goal=config.goal,
            preferences=config.preferences,
            hypothesis=hypothesis.text,
            articles_with_reasoning=articles_with_reasoning,
        )
        response = await self.base.call_claude(
            "You are a scientific peer reviewer.", prompt, use_strong=True
        )
        return Review(
            id=str(uuid.uuid4()),
            hypothesis_id=hypothesis.id,
            tier=2,
            critique=response,
            verdict=_parse_verdict(response),
        )

    async def run_deep_verification(
        self,
        hypothesis: Hypothesis,
        config: ResearchPlanConfig,
        prior_review: str,
    ) -> Review:
        """Tier 3 — deep assumption-level verification."""
        prompt = self.base.render_prompt(
            "reflection/deep_verification",
            goal=config.goal,
            hypothesis=hypothesis.text,
            prior_review=prior_review,
        )
        response = await self.base.call_claude(
            "You are a scientific peer reviewer.", prompt, use_strong=True
        )
        return Review(
            id=str(uuid.uuid4()),
            hypothesis_id=hypothesis.id,
            tier=3,
            critique=response,
            verdict=_parse_verdict(response),
        )

    async def run_observation_review(
        self,
        hypothesis: Hypothesis,
        config: ResearchPlanConfig,
        article: str,
    ) -> tuple[Review, Optional[str]]:
        """Tier 4 — observation-based causal analysis against a single article.

        Returns (Review, observation_line) where observation_line is non-None
        when the article contains a 'missing piece' for the hypothesis.
        """
        prompt = self.base.render_prompt(
            "reflection/observation",
            hypothesis=hypothesis.text,
            article=article,
        )
        response = await self.base.call_claude(
            "You are an expert in scientific hypothesis evaluation.", prompt
        )
        observation = _extract_observation(response)
        verdict = "rejected" if "disproved" in response.lower() else "passed"
        review = Review(
            id=str(uuid.uuid4()),
            hypothesis_id=hypothesis.id,
            tier=4,
            critique=response,
            verdict=verdict,
        )
        return review, observation

    async def run_simulation_review(
        self, hypothesis: Hypothesis, config: ResearchPlanConfig
    ) -> Review:
        """Tier 5 — step-by-step mechanism simulation."""
        prompt = self.base.render_prompt(
            "reflection/simulation",
            goal=config.goal,
            hypothesis=hypothesis.text,
        )
        response = await self.base.call_claude(
            "You are an expert scientific researcher.", prompt, use_strong=True
        )
        return Review(
            id=str(uuid.uuid4()),
            hypothesis_id=hypothesis.id,
            tier=5,
            critique=response,
            verdict=_parse_verdict(response),
        )

    async def run_tournament_review(
        self,
        hypothesis: Hypothesis,
        config: ResearchPlanConfig,
        prior_reviews: str,
        meta_critique: str,
        tournament_history: str,
    ) -> Review:
        """Tier 6 — recurrent review informed by tournament history."""
        prompt = self.base.render_prompt(
            "reflection/tournament",
            goal=config.goal,
            preferences=config.preferences,
            hypothesis=hypothesis.text,
            prior_reviews=prior_reviews,
            meta_critique=meta_critique,
            tournament_history=tournament_history,
        )
        response = await self.base.call_claude(
            "You are a scientific peer reviewer.", prompt
        )
        return Review(
            id=str(uuid.uuid4()),
            hypothesis_id=hypothesis.id,
            tier=6,
            critique=response,
            verdict=_parse_verdict(response),
        )
