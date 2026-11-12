import json
import re
from pathlib import Path
from typing import Optional
from agents.base import AgentOutputError
from core.models import ResearchPlanConfig
from tools.llm import LLMBackend
from tools.resources import resource_path


# Optional markdown decoration around a field label: `**LABEL:**`, `## LABEL:`,
# `__LABEL__:`. Models emit these even when told to use exact labels; tolerating
# the decoration is not fail-open — a missing field still fails the parse.
_DECO = r"[ \t]*[#*_]*[ \t]*"


def _extract(label: str, text: str) -> str:
    pattern = (
        rf"^{_DECO}{re.escape(label)}{_DECO}:{_DECO}(.+?)"
        rf"(?=\n{_DECO}[A-Z][A-Z]+{_DECO}:|\Z)"
    )
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    return match.group(1).strip().strip("*_").strip() if match else ""


class ConfigParser:
    def __init__(self, client: LLMBackend, prompts_dir: Path):
        self.client = client
        self.prompts_dir = resource_path(prompts_dir)

    def _render(self, template_name: str, **variables) -> str:
        path = self.prompts_dir / f"{template_name}.txt"
        return path.read_text().format(**variables)

    async def parse(self, run_id: str, goal: str) -> ResearchPlanConfig:
        prompt = self._render("config/parse", goal=goal)
        response = await self.client.call(
            "You are an expert research assistant.", prompt
        )
        if not isinstance(response, str) or not response.strip():
            raise AgentOutputError("configuration parser returned an empty response")
        preferences = _extract("PREFERENCES", response)
        attributes_raw = _extract("ATTRIBUTES", response)
        constraints = _extract("CONSTRAINTS", response)
        if not all((preferences, attributes_raw, constraints)):
            # Some model families follow the semantic contract but prefer JSON.
            try:
                start = response.index("{")
                value, _ = json.JSONDecoder().raw_decode(response[start:])
            except (ValueError, json.JSONDecodeError):
                value = {}
            preferences = preferences or str(value.get("preferences", "")).strip()
            raw_json_attributes = value.get("attributes", "")
            if not attributes_raw:
                attributes_raw = (
                    ", ".join(str(x) for x in raw_json_attributes)
                    if isinstance(raw_json_attributes, list)
                    else str(raw_json_attributes).strip()
                )
            constraints = constraints or str(value.get("constraints", "")).strip()
        if not all((preferences, attributes_raw, constraints)):
            raise AgentOutputError(
                "configuration output must provide preferences, attributes, and constraints"
            )
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
        approved = decision.strip() == "APPROVED"
        return approved, reason

    async def parse_and_review(self, run_id: str, goal: str) -> Optional[ResearchPlanConfig]:
        approved, reason = await self.safety_review(goal)
        if not approved:
            return None
        config = await self.parse(run_id, goal)
        config.safety_approved = True
        return config
