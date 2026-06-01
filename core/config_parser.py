import re
from pathlib import Path
from typing import Optional
from core.models import ResearchPlanConfig
from tools.claude import ClaudeClient


def _extract(label: str, text: str) -> str:
    pattern = rf"^{re.escape(label)}:\s*(.+?)(?=\n[A-Z][A-Z]+:|\Z)"
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


class ConfigParser:
    def __init__(self, client: ClaudeClient, prompts_dir: Path):
        self.client = client
        self.prompts_dir = Path(prompts_dir)

    def _render(self, template_name: str, **variables) -> str:
        path = self.prompts_dir / f"{template_name}.txt"
        return path.read_text().format(**variables)

    async def parse(self, run_id: str, goal: str) -> ResearchPlanConfig:
        prompt = self._render("config/parse", goal=goal)
        response = await self.client.call(
            "You are an expert research assistant.", prompt
        )
        preferences = _extract("PREFERENCES", response) or "Focus on novel, testable hypotheses"
        attributes_raw = _extract("ATTRIBUTES", response) or "Novelty, Feasibility"
        constraints = _extract("CONSTRAINTS", response) or "Must be testable"
        attributes = [a.strip() for a in attributes_raw.split(",") if a.strip()]
        return ResearchPlanConfig(
            run_id=run_id,
            goal=goal,
            preferences=preferences,
            attributes=attributes,
            constraints=constraints,
            safety_approved=False,
        )

    async def safety_review(self, goal: str) -> tuple[bool, str]:
        prompt = self._render("config/safety", goal=goal)
        response = await self.client.call(
            "You are a research safety reviewer.", prompt
        )
        decision = _extract("DECISION", response).upper()
        reason = _extract("REASON", response)
        approved = "APPROVED" in decision and "REJECTED" not in decision
        return approved, reason

    async def parse_and_review(self, run_id: str, goal: str) -> Optional[ResearchPlanConfig]:
        approved, reason = await self.safety_review(goal)
        if not approved:
            return None
        config = await self.parse(run_id, goal)
        config.safety_approved = True
        return config
