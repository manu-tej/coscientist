import uuid
from core.models import AgentTask, AgentType, ResearchPlanConfig
from core.state import StateStore
from core.tournament import select_match_pairs
from core.reflection_pipeline import run_reflection_pipeline


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
    ):
        self.store = store
        self.config = config
        self.generation = generation
        self.reflection = reflection
        self.ranking = ranking
        self.proximity = proximity
        self.evolution = evolution
        self.meta_review = meta_review
        self.search = search

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

    async def _run_generation(self, task: AgentTask) -> None:
        strategy = task.strategy or "debate"
        instructions = task.extra.get("instructions", "")
        if strategy == "literature":
            articles = await self.search.search_and_format(self.config.goal)
            h = await self.generation.run_literature(
                self.config, articles_with_reasoning=articles, instructions=instructions
            )
        elif strategy == "assumptions":
            h = await self.generation.run_assumptions(self.config, instructions=instructions)
        elif strategy == "expansion":
            latest = await self.store.get_latest_meta_review(self.config.run_id)
            overview = latest["research_overview"] if latest and latest.get("research_overview") else ""
            existing = await self.store.list_hypotheses(self.config.run_id)
            existing_summary = "\n".join(f"- {x.summary}" for x in existing[:20])
            h = await self.generation.run_expansion(
                self.config, research_overview=overview,
                existing_hypotheses_summary=existing_summary, instructions=instructions,
            )
        else:  # debate (default)
            reviews_overview = await self._current_meta_critique()
            h = await self.generation.run_debate(
                self.config, reviews_overview=reviews_overview, instructions=instructions
            )
        await self.store.save_hypothesis(h)

    async def _run_reflection(self, task: AgentTask) -> None:
        candidates = []
        if task.hypothesis_id:
            h = await self.store.get_hypothesis(task.hypothesis_id)
            if h is not None:
                candidates = [h]
        if not candidates:
            for h in await self.store.list_hypotheses(self.config.run_id):
                reviews = await self.store.list_reviews(h.id)
                if not any(r.tier >= 1 for r in reviews):
                    candidates.append(h)
        for hypothesis in candidates:
            # Atomically claim this hypothesis so a concurrent worker doesn't
            # run the reflection pipeline on it at the same time. If another
            # in-flight worker holds the claim, try the next candidate
            # instead of bailing (otherwise one held claim starves all
            # reflection).
            claimed = await self.store.try_claim_review(hypothesis.id)
            if not claimed:
                continue
            # Hold the claim during the review to block double-review, and
            # always release it afterwards: on success a tier>=1 review now
            # exists so the scan won't re-pick this hypothesis; on a crash
            # the claim is freed for retry instead of being held forever.
            try:
                meta_critique = await self._current_meta_critique()
                await run_reflection_pipeline(
                    hypothesis, self.config, self.reflection, self.search, self.store,
                    meta_critique=meta_critique,
                )
            finally:
                await self.store.release_claim(hypothesis.id)
            return

    async def _review_text(self, hypothesis_id: str) -> str:
        reviews = await self.store.list_reviews(hypothesis_id)
        return "\n\n".join(f"Tier {r.tier}: {r.critique}" for r in reviews) or "No reviews yet."

    async def _run_ranking(self, task: AgentTask) -> None:
        hypotheses = await self.store.list_hypotheses(self.config.run_id)
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
            match = await self.ranking.run_match(h1, h2, self.config, review_1, review_2)
            await self.store.save_match_and_elos(match)

    async def _run_proximity(self, task: AgentTask) -> None:
        hypotheses = await self.store.list_hypotheses(self.config.run_id)
        if len(hypotheses) < 2:
            return
        near_dupes = await self.proximity.update_graph(hypotheses)
        for dup_id in near_dupes:
            await self.store.set_hypothesis_status(dup_id, "rejected")

    async def _run_evolution(self, task: AgentTask) -> None:
        hypotheses = await self.store.list_hypotheses(self.config.run_id)
        if not hypotheses:
            return
        top = sorted(hypotheses, key=lambda x: x.elo_rating, reverse=True)
        strategy = task.strategy or "grounding"
        if strategy == "combination":
            new_h = await self.evolution.run_combination(top[:3], self.config)
        elif strategy == "out_of_box":
            new_h = await self.evolution.run_out_of_box(top[:5], self.config)
        elif strategy == "coherence":
            new_h = await self.evolution.run_coherence(top[0], self.config)
        elif strategy == "inspiration":
            new_h = await self.evolution.run_inspiration(top[0], self.config)
        elif strategy == "simplification":
            new_h = await self.evolution.run_simplification(top[0], self.config)
        else:  # grounding (default)
            reviews = await self._review_text(top[0].id)
            articles = await self.search.search_and_format(
                f"{self.config.goal} {top[0].summary}"
            )
            new_h = await self.evolution.run_grounding(
                top[0], self.config, weaknesses=reviews, articles=articles
            )
        await self.store.save_hypothesis(new_h)

    async def _run_meta_review(self, task: AgentTask) -> None:
        hypotheses = await self.store.list_hypotheses(self.config.run_id)
        all_reviews = []
        cited = []
        for h in hypotheses:
            for r in await self.store.list_reviews(h.id):
                all_reviews.append(f"Tier {r.tier} ({h.summary[:40]}): {r.critique}")
                for c in r.web_citations:
                    cited.append(f"{c.get('title', '')}: {c.get('url', '')}")
        reviews_text = "\n\n".join(all_reviews) or "No reviews yet."
        critique = await self.meta_review.run_meta_critique(self.config, reviews_text)
        overview = await self.meta_review.run_research_overview(self.config, hypotheses)
        contacts = await self.meta_review.run_research_contacts(
            self.config, "\n".join(cited) or "No citations yet."
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
