from __future__ import annotations

import json
import re
import uuid

from agents.base import LABEL_DECO, AgentOutputError, BaseAgent
from core.models import Hypothesis, ResearchPlanConfig
from tools.llm import LLMBackendError


def _extract_field(text: str, field: str, default: str = "") -> str:
    """Extract a labeled field from structured text output.

    Tolerates markdown decoration around the label and an optional parenthetical
    qualifier after it — the shipped prompts themselves ask for
    `Summary (one sentence for non-experts):`.
    """
    pattern = (
        rf"^{LABEL_DECO}{re.escape(field)}(?:\s*\([^)\n]*\))?{LABEL_DECO}:"
        rf"{LABEL_DECO}(.+?)(?=\n{LABEL_DECO}[A-Z][a-zA-Z]|\Z)"
    )
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    return match.group(1).strip().strip("*_").strip() if match else default


def _satisfies_output_contract(text: str) -> bool:
    """True when a turn carries the fields _parse_hypothesis_output requires.

    Used as the turn-loop completion test so a model that merely mentions the
    termination signal keeps deliberating instead of ending on a turn that will be
    discarded.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    if re.search(r"\{", text):  # a JSON payload is validated by the parser itself
        return True
    return bool(_extract_field(text, "Hypothesis") and _extract_field(text, "Summary"))


_CONTRACT_REMINDER = (
    "\n\nYour previous response could not be used because it omitted the required "
    "field labels. Reply again with the same scientific content, but structure it "
    "with these labels, each at the start of its own line:\n"
    "Hypothesis: ...\nSummary (one sentence for non-experts): ...\nCategory: ..."
)


async def call_until_contract(
    base: BaseAgent,
    system_prompt: str,
    prompt: str,
    *,
    attempts: int = 2,
    use_strong: bool = False,
) -> str:
    """Call the model until the response carries the required fields.

    The single-shot strategies (literature, expansion, every evolution variant)
    have no turn loop, so one unlabelled response destroyed the whole task — the
    dominant live failure mode even after the prompts were fixed. A retry with an
    explicit corrective costs one extra call instead of the task.

    Returns the last response either way; the caller's parser raises the specific
    error if the retry also fails.
    """
    response = await base.call_claude(system_prompt, prompt, use_strong=use_strong)
    for _ in range(max(0, attempts - 1)):
        if _satisfies_output_contract(response):
            return response
        response = await base.call_claude(
            system_prompt, prompt + _CONTRACT_REMINDER, use_strong=use_strong
        )
    return response


def _parse_hypothesis_output(
    text: str, config: ResearchPlanConfig, method: str
) -> Hypothesis:
    """Parse raw LLM output into a Hypothesis dataclass."""
    if not isinstance(text, str) or not text.strip():
        raise AgentOutputError(f"{method} generation returned no hypothesis")

    parsed = None
    for match in re.finditer(r"\{", text):
        try:
            value, _ = json.JSONDecoder().raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed = {str(k).lower(): v for k, v in value.items()}
            break
    if parsed is not None:
        hypothesis_text = parsed.get("hypothesis")
        summary = parsed.get("summary")
        category = parsed.get("category")
        if not isinstance(hypothesis_text, str) or not hypothesis_text.strip():
            raise AgentOutputError("JSON generation output is missing `hypothesis`")
        if not isinstance(summary, str) or not summary.strip():
            raise AgentOutputError("JSON generation output is missing `summary`")
        text = hypothesis_text.strip()
        category = category.strip() if isinstance(category, str) else ""
    else:
        hypothesis_text = _extract_field(text, "Hypothesis")
        summary = _extract_field(text, "Summary")
        category = _extract_field(text, "Category")
        if not hypothesis_text:
            raise AgentOutputError(
                "generation output must contain `Hypothesis:` or JSON `hypothesis`"
            )
        if not summary:
            raise AgentOutputError(
                "generation output must contain `Summary:` or JSON `summary`"
            )

    # Trim to everything from the HYPOTHESIS marker onwards if present
    if "HYPOTHESIS" in text:
        text = text[text.index("HYPOTHESIS"):]

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
        response = await call_until_contract(
            self.base, "You are an expert scientific researcher.", prompt
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
            is_complete=_satisfies_output_contract,
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
            is_complete=_satisfies_output_contract,
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
        response = await call_until_contract(
            self.base, "You are an expert scientific researcher.", prompt
        )
        return _parse_hypothesis_output(response, config, "expansion")
