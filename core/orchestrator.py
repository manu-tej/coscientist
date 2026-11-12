import asyncio
import uuid
from dataclasses import replace
from core.models import AgentTask, AgentType, ResearchPlanConfig
from core.ranking import native_ranking_key
from core.state import StateStore
from core.tournament import select_match_pairs
from core.reflection_pipeline import run_reflection_pipeline


def _format_literature_articles(records) -> str:
    """Format pre-fetched evidence records the same way
    LiteratureTool.search_and_format formats a freshly-fetched outcome.

    Duplicated (not delegated) on purpose: the records must be fetched via
    search_outcome() directly here so the exact same records used to build the
    prompt are the ones set_grounding_refs() persists as "shown". A second,
    independent search_and_format() call could return a different result set
    (source flakiness, a paginated/rotating index) and silently break the
    citation-attribution guarantee this is for.
    """
    if not records:
        return "No relevant articles found."
    lines = []
    for i, r in enumerate(records, 1):
        lines.append(f"[{i}] {r.title} ({r.source.upper()}:{r.ref})")
        lines.append(f"URL: {r.url}")
        lines.append(f"Summary: {r.snippet[:1200]}")
        lines.append("")
    return "\n".join(lines)


class AgentRunner:
    """Dispatches an AgentTask to the correct agent and persists results."""

    def __init__(
        self,
        store: StateStore,
        config: ResearchPlanConfig,
        generation,
        reflection,
        ranking,
        proximity,
        evolution,
        meta_review,
        search,
        ranking_mode: str = "elo",
        verifier=None,
    ):
        if ranking_mode not in {"elo", "absolute"}:
            raise ValueError("ranking_mode must be 'elo' or 'absolute'")
        self.store = store
        self.config = config
        self.generation = generation
        self.reflection = reflection
        self.ranking = ranking
        self.proximity = proximity
        self.evolution = evolution
        self.meta_review = meta_review
        self.search = search
        self.ranking_mode = ranking_mode
        self.verifier = verifier
        # Elo updates are read/compute/write operations. Different ranking tasks
        # may select overlapping pairs, so serialize tournaments within one run
        # to prevent a later stale rating from overwriting an earlier update.
        self._ranking_lock = asyncio.Lock()

    def _config_for_task(self, task: AgentTask) -> ResearchPlanConfig:
        """Return an immutable-per-task view that activates its assigned role."""
        persona = task.extra.get("persona", "")
        if not isinstance(persona, str) or not persona.strip():
            return self.config
        role = persona.strip()
        return replace(
            self.config,
            goal=(
                f"{self.config.goal}\n\n"
                f"Assigned specialist role: {role}. Work from this specialist "
                "perspective, state role-specific assumptions, and challenge claims "
                "that are outside your expertise."
            ),
        )

    async def run_task(self, task: AgentTask) -> None:
        if task.agent_type == AgentType.GENERATION:
            await self._run_generation(task)
        elif task.agent_type == AgentType.REFLECTION:
            await self._run_reflection(task)
        elif task.agent_type == AgentType.RANKING:
            await self._run_ranking(task)
        elif task.agent_type == AgentType.PROXIMITY:
            await self._run_proximity(task)
        elif task.agent_type == AgentType.EVOLUTION:
            await self._run_evolution(task)
        elif task.agent_type == AgentType.META_REVIEW:
            await self._run_meta_review(task)

    async def _current_meta_critique(self) -> str:
        latest = await self.store.get_latest_meta_review(self.config.run_id)
        return latest["meta_critique"] if latest else ""

    async def _order_hypotheses(self, hypotheses):
        if self.ranking_mode == "absolute":
            return await self.ranking.order_hypotheses(hypotheses)
        return sorted(
            hypotheses,
            key=lambda hypothesis: native_ranking_key(
                hypothesis, hypothesis.elo_rating
            ),
        )

    async def _ranking_context(self, hypotheses):
        if self.ranking_mode == "absolute":
            persisted = await self.store.list_absolute_scores(self.config.run_id)
            return (
                "Absolute rubric score",
                {
                    hypothesis.id: persisted[hypothesis.id].total
                    for hypothesis in hypotheses
                    if hypothesis.id in persisted
                },
            )
        return (
            "Elo",
            {hypothesis.id: hypothesis.elo_rating for hypothesis in hypotheses},
        )

    async def _run_generation(self, task: AgentTask) -> None:
        config = self._config_for_task(task)
        strategy = task.strategy or "debate"
        instructions = task.extra.get("instructions", "")
        shown_refs = None
        if strategy == "literature":
            # LiteratureTool (the production search collaborator wired in
            # coscientist.py) exposes search_outcome() with typed records, which
            # lets the exact ref set shown to the model be recorded for the
            # citation-attribution check in the verification block. Older/legacy
            # search collaborators (e.g. bench's plain Tavily-backed SearchTool)
            # only implement search_and_format() -- shown_refs stays None for
            # those, which get_grounding_refs correctly reports as "never
            # recorded" rather than an accusable empty set.
            search_outcome = getattr(self.search, "search_outcome", None)
            if search_outcome is None:
                articles = await self.search.search_and_format(config.goal)
            else:
                outcome = await search_outcome(config.goal)
                shown_refs = {r.ref for r in outcome.records}
                articles = _format_literature_articles(outcome.records)
            h = await self.generation.run_literature(
                config, articles_with_reasoning=articles, instructions=instructions
            )
        elif strategy == "assumptions":
            h = await self.generation.run_assumptions(config, instructions=instructions)
        elif strategy == "expansion":
            latest = await self.store.get_latest_meta_review(self.config.run_id)
            overview = latest["research_overview"] if latest and latest.get("research_overview") else ""
            existing = await self.store.list_hypotheses(self.config.run_id)
            existing = await self._order_hypotheses(existing)
            existing_summary = "\n".join(f"- {x.summary}" for x in existing[:20])
            h = await self.generation.run_expansion(
                config, research_overview=overview,
                existing_hypotheses_summary=existing_summary, instructions=instructions,
            )
        else:  # debate (default)
            reviews_overview = await self._current_meta_critique()
            h = await self.generation.run_debate(
                config, reviews_overview=reviews_overview, instructions=instructions
            )
        await self.store.save_hypothesis(h)
        if shown_refs is not None:
            await self.store.set_grounding_refs(h.id, shown_refs)

    async def _run_reflection(self, task: AgentTask) -> None:
        config = self._config_for_task(task)
        hypothesis = None
        if task.hypothesis_id:
            hypothesis = await self.store.get_hypothesis(task.hypothesis_id)
        if hypothesis is None:
            for h in await self.store.list_hypotheses(self.config.run_id):
                reviews = await self.store.list_reviews(h.id)
                if not any(r.tier >= 1 for r in reviews):
                    hypothesis = h
                    break
        if hypothesis is None:
            return
        # Atomically claim this hypothesis so a concurrent worker doesn't
        # run the reflection pipeline on it at the same time.
        claimed = await self.store.try_claim_review(hypothesis.id)
        if not claimed:
            return
        try:
            meta_critique = await self._current_meta_critique()
            await run_reflection_pipeline(
                hypothesis, config, self.reflection, self.search, self.store,
                meta_critique=meta_critique, verifier=self.verifier,
            )
        except BaseException:
            # A permanent claim after a provider error would make this hypothesis
            # impossible to review for the rest of the run.
            await self.store.release_review_claim(hypothesis.id)
            raise

    async def _review_text(self, hypothesis_id: str) -> str:
        reviews = await self.store.list_reviews(hypothesis_id)
        return "\n\n".join(f"Tier {r.tier}: {r.critique}" for r in reviews) or "No reviews yet."

    async def _run_ranking(self, task: AgentTask) -> None:
        async with self._ranking_lock:
            await self._run_ranking_serial(task)

    async def _run_ranking_serial(self, task: AgentTask) -> None:
        config = self._config_for_task(task)
        hypotheses = await self.store.list_hypotheses(self.config.run_id)
        if self.ranking_mode == "absolute":
            if hypotheses:
                await self.ranking.score_hypotheses(hypotheses, config)
            return
        if len(hypotheses) < 2:
            return
        similar = await self.store.get_similar_pairs(self.config.run_id, threshold=0.0)
        # Run several matches per ranking task so the tournament accumulates the
        # match volume needed to spread Elo (single-turn matches are cheap). The
        # pairs are non-overlapping, so in-place Elo updates don't collide.
        pairs = select_match_pairs(hypotheses, similar, n_pairs=task.extra.get("n_pairs", 5))
        for h1, h2 in pairs:
            review_1 = await self._review_text(h1.id)
            review_2 = await self._review_text(h2.id)
            match = await self.ranking.run_match(h1, h2, config, review_1, review_2)
            # Production RankingAgent persists atomically when it owns this store.
            # Injected/mocked ranking implementations return a match for us to save.
            if getattr(self.ranking, "store", None) is not self.store:
                await self.store.save_match_and_elos(match)

    async def finalize_ranking(self) -> None:
        """Score hypotheses created after the final scheduled ranking task."""
        if self.ranking_mode != "absolute":
            return
        async with self._ranking_lock:
            hypotheses = await self.store.list_hypotheses(self.config.run_id)
            if hypotheses:
                await self.ranking.score_hypotheses(hypotheses, self.config)

    async def _run_proximity(self, task: AgentTask) -> None:
        hypotheses = await self.store.list_hypotheses(self.config.run_id)
        if len(hypotheses) < 2:
            return
        near_dupes = await self.proximity.update_graph(hypotheses)
        for dup_id in near_dupes:
            await self.store.set_hypothesis_status(dup_id, "rejected")

    async def _run_evolution(self, task: AgentTask) -> None:
        config = self._config_for_task(task)
        hypotheses = await self.store.list_hypotheses(self.config.run_id)
        if not hypotheses:
            return
        top = await self._order_hypotheses(hypotheses)
        strategy = task.strategy or "grounding"
        if strategy == "combination":
            ranking_label, ranking_scores = await self._ranking_context(top[:3])
            new_h = await self.evolution.run_combination(
                top[:3],
                config,
                ranking_scores=ranking_scores,
                ranking_label=ranking_label,
            )
        elif strategy == "out_of_box":
            new_h = await self.evolution.run_out_of_box(top[:5], config)
        elif strategy == "coherence":
            new_h = await self.evolution.run_coherence(top[0], config)
        elif strategy == "inspiration":
            new_h = await self.evolution.run_inspiration(top[0], config)
        elif strategy == "simplification":
            new_h = await self.evolution.run_simplification(top[0], config)
        else:  # grounding (default)
            reviews = await self._review_text(top[0].id)
            articles = await self.search.search_and_format(
                f"{config.goal} {top[0].summary}"
            )
            new_h = await self.evolution.run_grounding(
                top[0], config, weaknesses=reviews, articles=articles
            )
        await self.store.save_hypothesis(new_h)

    async def _run_meta_review(self, task: AgentTask) -> None:
        config = self._config_for_task(task)
        hypotheses = await self.store.list_hypotheses(self.config.run_id)
        hypotheses = await self._order_hypotheses(hypotheses)
        all_reviews = []
        cited = []
        for h in hypotheses:
            for r in await self.store.list_reviews(h.id):
                all_reviews.append(f"Tier {r.tier} ({h.summary[:40]}): {r.critique}")
                for c in r.web_citations:
                    cited.append(f"{c.get('title', '')}: {c.get('url', '')}")
        reviews_text = "\n\n".join(all_reviews) or "No reviews yet."
        critique = await self.meta_review.run_meta_critique(config, reviews_text)
        ranking_label, ranking_scores = await self._ranking_context(hypotheses)
        overview = await self.meta_review.run_research_overview(
            config,
            hypotheses,
            ranking_scores=ranking_scores,
            ranking_label=ranking_label,
        )
        contacts = await self.meta_review.run_research_contacts(
            config, "\n".join(cited) or "No citations yet."
        )
        tick = task.extra.get("tick", 0)
        await self.store.save_meta_review(
            id=str(uuid.uuid4()),
            run_id=self.config.run_id,
            meta_critique=critique,
            research_overview=overview,
            research_contacts=contacts,
            tick=tick,
        )
