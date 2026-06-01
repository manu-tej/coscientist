from core.models import Hypothesis, ResearchPlanConfig
from core.state import StateStore


def _stamp(review, hypothesis_id: str):
    """Ensure the review's hypothesis_id matches the actual hypothesis being reviewed."""
    review.hypothesis_id = hypothesis_id
    return review


async def run_reflection_pipeline(
    hypothesis: Hypothesis,
    config: ResearchPlanConfig,
    reflection_agent,
    search_tool,
    store: StateStore,
    meta_critique: str = "",
) -> str:
    """Run the 6 reflection tiers sequentially with early exit on rejection.

    Returns the final verdict string ("passed" | "rejected" | "flagged").
    Persists every review and appends positive observations to annotations.
    """
    hid = hypothesis.id

    # Tier 1 - initial review (no web search)
    r1 = _stamp(await reflection_agent.run_initial_review(hypothesis, config), hid)
    await store.save_review(r1)
    if r1.verdict == "rejected":
        await store.set_hypothesis_status(hid, "rejected")
        return "rejected"

    # Tier 2 - full review with literature
    articles = await search_tool.search_and_format(
        f"{config.goal} {hypothesis.summary}", context=hypothesis.text
    )
    r2 = _stamp(await reflection_agent.run_full_review(hypothesis, config, articles), hid)
    await store.save_review(r2)
    if r2.verdict == "rejected":
        await store.set_hypothesis_status(hid, "rejected")
        return "rejected"

    # Tier 3 - deep verification (non-fundamental errors do NOT discard)
    r3 = _stamp(await reflection_agent.run_deep_verification(hypothesis, config, r2.critique), hid)
    await store.save_review(r3)
    if r3.verdict == "rejected":
        await store.set_hypothesis_status(hid, "rejected")
        return "rejected"

    # Tier 4 - observation review; append positive observations
    r4_raw, observation = await reflection_agent.run_observation_review(hypothesis, config, articles)
    r4 = _stamp(r4_raw, hid)
    await store.save_review(r4)
    if observation:
        await store.append_annotation(hid, observation)
    if r4.verdict == "rejected":
        await store.set_hypothesis_status(hid, "rejected")
        return "rejected"

    # Tier 5 - simulation review (flags but does not reject)
    r5 = _stamp(await reflection_agent.run_simulation_review(hypothesis, config), hid)
    await store.save_review(r5)

    # Tier 6 - tournament/recurrent review: only if hypothesis has tournament history
    matches = await store.list_matches(config.run_id)
    in_tournament = any(
        m.h1_id == hid or m.h2_id == hid for m in matches
    )
    if in_tournament:
        prior = await store.list_reviews(hid)
        prior_text = "\n\n".join(f"Tier {r.tier}: {r.critique}" for r in prior)
        tournament_history = "\n".join(
            f"{m.match_type}: winner={m.winner_id}"
            for m in matches
            if m.h1_id == hid or m.h2_id == hid
        )
        r6 = _stamp(
            await reflection_agent.run_tournament_review(
                hypothesis, config, prior_text, meta_critique, tournament_history
            ),
            hid,
        )
        await store.save_review(r6)

    return "passed"
