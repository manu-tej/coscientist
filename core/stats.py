from __future__ import annotations
import random
from dataclasses import dataclass
from core.models import SystemStats, AgentType
from core.state import StateStore


@dataclass
class WeightThresholds:
    min_hypothesis_count: int = 8
    elo_variance_threshold: float = 5000.0
    meta_review_interval: int = 20


def compute_weights(stats: SystemStats, thresholds: WeightThresholds) -> dict[AgentType, float]:
    weights: dict[AgentType, float] = {
        AgentType.GENERATION: 1.0,
        AgentType.REFLECTION: 1.0,
        AgentType.RANKING: 1.0,
        AgentType.EVOLUTION: 1.0,
        AgentType.PROXIMITY: 0.5,
        AgentType.META_REVIEW: 0.3,
    }
    # Boost Reflection if review backlog is high
    if stats.n_hypotheses > 0 and stats.n_pending_review > stats.n_hypotheses * 0.3:
        weights[AgentType.REFLECTION] *= 2.0
    # Boost Evolution if Elo variance is low (pool converging, need diversity)
    if stats.n_hypotheses >= 2 and stats.elo_variance < thresholds.elo_variance_threshold:
        weights[AgentType.EVOLUTION] *= 1.8
    # Boost Generation if pool is small
    if stats.n_hypotheses < thresholds.min_hypothesis_count:
        weights[AgentType.GENERATION] *= 2.5
    # Boost Meta-review if stale
    if stats.last_meta_review_age > thresholds.meta_review_interval:
        weights[AgentType.META_REVIEW] = 2.0
    return weights


def sample_agent_type(weights: dict[AgentType, float], seed: int) -> AgentType:
    rng = random.Random(seed)
    types = list(weights.keys())
    w = [max(0.0, weights[t]) for t in types]
    total = sum(w)
    if total <= 0:
        return types[0]
    return rng.choices(types, weights=w, k=1)[0]


# Generation-style methods vs evolution-style methods (for effectiveness stats)
_GENERATION_METHODS = {"literature", "debate", "assumptions", "expansion"}
_EVOLUTION_METHODS = {"grounding", "coherence", "inspiration", "combination", "simplification", "out_of_box"}


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


async def compute_stats(
    store: StateStore,
    run_id: str,
    last_meta_review_tick: int,
    current_tick: int,
) -> SystemStats:
    hypotheses = await store.list_hypotheses(run_id, status="active")
    n_hypotheses = len(hypotheses)

    elos = [h.elo_rating for h in hypotheses]
    elo_variance = _variance(elos)

    # Count which hypotheses have at least one tier>=1 review
    n_reviewed = 0
    for h in hypotheses:
        reviews = await store.list_reviews(h.id)
        if any(r.tier >= 1 for r in reviews):
            n_reviewed += 1
    n_pending_review = n_hypotheses - n_reviewed

    gen_elos = [h.elo_rating for h in hypotheses if h.generation_method in _GENERATION_METHODS]
    evo_elos = [h.elo_rating for h in hypotheses if h.generation_method in _EVOLUTION_METHODS]
    generation_effectiveness = sum(gen_elos) / len(gen_elos) if gen_elos else 1200.0
    evolution_effectiveness = sum(evo_elos) / len(evo_elos) if evo_elos else 1200.0

    matches = await store.list_matches(run_id)
    tournament_progress = float(len(matches))

    similar = await store.get_similar_pairs(run_id, threshold=0.0)
    avg_proximity = (sum(s for _, _, s in similar) / len(similar)) if similar else 0.0

    return SystemStats(
        n_hypotheses=n_hypotheses,
        n_pending_review=n_pending_review,
        n_reviewed=n_reviewed,
        tournament_progress=tournament_progress,
        elo_variance=elo_variance,
        avg_proximity=avg_proximity,
        generation_effectiveness=generation_effectiveness,
        evolution_effectiveness=evolution_effectiveness,
        last_meta_review_age=current_tick - last_meta_review_tick,
    )
