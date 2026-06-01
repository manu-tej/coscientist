import uuid
from core.state import StateStore
from core.models import Hypothesis, Review


async def get_ranked_hypotheses(store: StateStore, run_id: str) -> list[dict]:
    hypotheses = await store.list_hypotheses(run_id, status="active")
    rows = []
    for h in hypotheses:  # already ordered by elo DESC from the store
        reviews = await store.list_reviews(h.id)
        rows.append({
            "id": h.id,
            "elo": round(h.elo_rating, 1),
            "summary": h.summary,
            "category": h.category or "",
            "method": h.generation_method,
            "source": h.source,
            "n_reviews": len(reviews),
            "text": h.text,
        })
    return rows


async def get_research_overview(store: StateStore, run_id: str) -> str:
    latest = await store.get_latest_meta_review(run_id)
    if latest is None or not latest.get("research_overview"):
        return "No research overview yet — waiting for the first meta-review."
    return latest["research_overview"]


async def inject_expert_hypothesis(store: StateStore, run_id: str, text: str) -> str:
    h = Hypothesis(
        id=str(uuid.uuid4()),
        run_id=run_id,
        text=text,
        summary=text[:120],
        generation_method="expert",
        source="expert",
    )
    await store.save_hypothesis(h)
    return h.id


async def submit_expert_review(store: StateStore, hypothesis_id: str, critique: str) -> None:
    review = Review(
        id=str(uuid.uuid4()),
        hypothesis_id=hypothesis_id,
        tier=0,  # expert review
        critique=critique,
        verdict="flagged",
    )
    await store.save_review(review)
