from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Optional

from agents.base import LABEL_DECO, AgentOutputError, BaseAgent
from core.models import Hypothesis, ResearchPlanConfig, TournamentMatch
from core.state import StateStore
from core.tournament import compute_elo_update

logger = logging.getLogger(__name__)


def _parse_winner(text: str, h1_id: str, h2_id: str) -> str:
    """Parse a model-portable explicit winner; ambiguity fails instead of biasing h1."""
    if not isinstance(text, str) or not text.strip():
        raise AgentOutputError("ranking returned no winner")
    for match in re.finditer(r"\{", text):
        try:
            value, _ = json.JSONDecoder().raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "winner" in value:
            winner = str(value["winner"]).strip().lower()
            if winner in {"1", "h1", "hypothesis 1", h1_id.lower()}:
                return h1_id
            if winner in {"2", "h2", "hypothesis 2", h2_id.lower()}:
                return h2_id
            raise AgentOutputError(f"unknown ranking winner {value['winner']!r}")
    text_lower = text.lower()
    for pattern in [
        rf"better (?:hypothesis|idea){LABEL_DECO}:{LABEL_DECO}1\b",
        r"hypothesis\s*1\s+is\s+(?:better|stronger|preferred|the winner)\b",
        r"(?:choose|prefer)\s+hypothesis\s*1\b",
    ]:
        if re.search(pattern, text_lower):
            return h1_id
    for pattern in [
        rf"better (?:hypothesis|idea){LABEL_DECO}:{LABEL_DECO}2\b",
        r"hypothesis\s*2\s+is\s+(?:better|stronger|preferred|the winner)\b",
        r"(?:choose|prefer)\s+hypothesis\s*2\b",
    ]:
        if re.search(pattern, text_lower):
            return h2_id
    logger.warning(
        "Could not parse winner from ranking response. Response snippet: %s",
        text[-200:],
    )
    raise AgentOutputError(
        "ranking output must explicitly select hypothesis 1 or 2"
    )


class RankingAgent:
    """Ranks hypotheses via single-turn comparison or multi-turn debate."""

    def __init__(
        self,
        base: BaseAgent,
        elo_k: float = 32.0,
        multi_turn_threshold: float = 1350.0,
        store: Optional[StateStore] = None,
    ):
        self.base = base
        self.elo_k = elo_k
        self.multi_turn_threshold = multi_turn_threshold
        self.store = store

    async def run_single_turn_match(
        self,
        h1: Hypothesis,
        h2: Hypothesis,
        config: ResearchPlanConfig,
        review_1: str,
        review_2: str,
        notes: str = "",
    ) -> TournamentMatch:
        """Compare two hypotheses in a single Claude call."""
        prompt = self.base.render_prompt(
            "ranking/single_turn",
            goal=config.goal,
            preferences=config.preferences,
            idea_attributes=config.idea_attributes,
            notes=notes,
            hypothesis_1=h1.text,
            hypothesis_2=h2.text,
            review_1=review_1,
            review_2=review_2,
        )
        response = await self.base.call_claude(
            "You are an expert evaluator.", prompt, use_strong=True,
        )
        winner_id = _parse_winner(response, h1.id, h2.id)
        winner = "a" if winner_id == h1.id else "b"
        old_r1, old_r2 = h1.elo_rating, h2.elo_rating
        new_r1, new_r2 = compute_elo_update(old_r1, old_r2, winner, self.elo_k)
        h1.elo_rating = new_r1
        h2.elo_rating = new_r2
        match = TournamentMatch(
            id=str(uuid.uuid4()),
            run_id=config.run_id,
            h1_id=h1.id,
            h2_id=h2.id,
            winner_id=winner_id,
            match_type="single_turn",
            elo_before_h1=old_r1,
            elo_before_h2=old_r2,
            elo_after_h1=new_r1,
            elo_after_h2=new_r2,
        )
        if self.store:
            await self.store.save_match_and_elos(match)
        return match

    async def run_multi_turn_match(
        self,
        h1: Hypothesis,
        h2: Hypothesis,
        config: ResearchPlanConfig,
        review_1: str,
        review_2: str,
        notes: str = "",
    ) -> TournamentMatch:
        """Compare two hypotheses via a structured multi-turn debate."""
        final_text, transcript = await self.base.run_turn_loop(
            template_name="ranking/multi_turn_debate",
            variables={
                "goal": config.goal,
                "preferences": config.preferences,
                "hypothesis_1": h1.text,
                "hypothesis_2": h2.text,
                "review_1": review_1,
                "review_2": review_2,
                "notes": notes,
            },
            transcript_key="transcript",
            termination_signal="better idea:",
            max_turns=10,
            system_prompt="You are an expert in comparative analysis.",
            use_strong=True,
        )
        winner_id = _parse_winner(final_text, h1.id, h2.id)
        winner = "a" if winner_id == h1.id else "b"
        old_r1, old_r2 = h1.elo_rating, h2.elo_rating
        new_r1, new_r2 = compute_elo_update(old_r1, old_r2, winner, self.elo_k)
        h1.elo_rating = new_r1
        h2.elo_rating = new_r2
        match = TournamentMatch(
            id=str(uuid.uuid4()),
            run_id=config.run_id,
            h1_id=h1.id,
            h2_id=h2.id,
            winner_id=winner_id,
            match_type="multi_turn",
            debate_transcript="\n---\n".join(transcript),
            elo_before_h1=old_r1,
            elo_before_h2=old_r2,
            elo_after_h1=new_r1,
            elo_after_h2=new_r2,
        )
        if self.store:
            await self.store.save_match_and_elos(match)
        return match

    async def run_match(
        self,
        h1: Hypothesis,
        h2: Hypothesis,
        config: ResearchPlanConfig,
        review_1: str,
        review_2: str,
        notes: str = "",
    ) -> TournamentMatch:
        """Route to single-turn or multi-turn based on ELO thresholds."""
        use_multi = (
            h1.elo_rating >= self.multi_turn_threshold
            and h2.elo_rating >= self.multi_turn_threshold
        )
        if use_multi:
            return await self.run_multi_turn_match(h1, h2, config, review_1, review_2, notes)
        return await self.run_single_turn_match(h1, h2, config, review_1, review_2, notes)
