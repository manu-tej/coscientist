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
    absolute_variance_threshold: float = 0.25


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
    low_ranking_variance = (
        stats.ranking_variance < thresholds.absolute_variance_threshold
        if stats.ranking_mode == "absolute"
        else stats.elo_variance < thresholds.elo_variance_threshold
    )
    if stats.n_hypotheses >= 2 and low_ranking_variance:
        weights[AgentType.EVOLUTION] *= 1.8
    # Boost Generation if pool is small
    if stats.n_hypotheses < thresholds.min_hypothesis_count:
        weights[AgentType.GENERATION] *= 2.5
    # As soon as there are a few hypotheses to compare, heavily favor Ranking so
    # the tournament plays many matches and Elo spreads (concordance needs a wide
    # Elo range, not just a full pool). Gated low (>=4) because proximity dedup
    # keeps the active pool near the generation threshold; without an early, strong
    # boost the tournament stays starved (~2-4 matches) and Elo barely leaves 1200.
    # Single-turn matches are cheap (1 call each), so this is low-cost.
    if stats.ranking_mode == "absolute" and stats.n_unranked == 0:
        weights[AgentType.RANKING] = 0.0
    elif stats.n_hypotheses >= 4:
        weights[AgentType.RANKING] *= 4.0
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
    ranking_mode: str = "elo",
) -> SystemStats:
    if ranking_mode not in {"elo", "absolute"}:
        raise ValueError("ranking_mode must be 'elo' or 'absolute'")
    hypotheses = await store.list_hypotheses(run_id, status="active")
    n_hypotheses = len(hypotheses)
    active_ids = {hypothesis.id for hypothesis in hypotheses}

    elos = [h.elo_rating for h in hypotheses]
    elo_variance = _variance(elos)
    absolute_scores = (
        await store.list_absolute_scores(run_id)
        if ranking_mode == "absolute"
        else {}
    )
    if ranking_mode == "absolute":
        absolute_scores = {
            hypothesis_id: score
            for hypothesis_id, score in absolute_scores.items()
            if hypothesis_id in active_ids
        }
    ranking_values = (
        [score.total for score in absolute_scores.values()]
        if ranking_mode == "absolute"
        else elos
    )
    ranking_variance = _variance(ranking_values)
    n_unranked = (
        sum(h.id not in absolute_scores for h in hypotheses)
        if ranking_mode == "absolute"
        else 0
    )

    # Count which hypotheses have at least one tier>=1 review
    n_reviewed = 0
    for h in hypotheses:
        reviews = await store.list_reviews(h.id)
        if any(r.tier >= 1 for r in reviews):
            n_reviewed += 1
    n_pending_review = n_hypotheses - n_reviewed

    def native_score(h):
        if ranking_mode == "absolute":
            score = absolute_scores.get(h.id)
            return score.total if score is not None else None
        return h.elo_rating

    gen_scores = []
    evo_scores = []
    for hypothesis in hypotheses:
        score = native_score(hypothesis)
        if score is None:
            continue
        if hypothesis.generation_method in _GENERATION_METHODS:
            gen_scores.append(score)
        if hypothesis.generation_method in _EVOLUTION_METHODS:
            evo_scores.append(score)
    native_default = 3.0 if ranking_mode == "absolute" else 1200.0
    generation_effectiveness = (
        sum(gen_scores) / len(gen_scores) if gen_scores else native_default
    )
    evolution_effectiveness = (
        sum(evo_scores) / len(evo_scores) if evo_scores else native_default
    )

    matches = await store.list_matches(run_id)
    tournament_progress = float(
        len(absolute_scores) if ranking_mode == "absolute" else len(matches)
    )

    similar = await store.get_similar_pairs(run_id, threshold=0.0)
    similar = [
        (hypothesis_1, hypothesis_2, score)
        for hypothesis_1, hypothesis_2, score in similar
        if hypothesis_1 in active_ids and hypothesis_2 in active_ids
    ]
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
        ranking_mode=ranking_mode,
        ranking_variance=ranking_variance,
        n_unranked=n_unranked,
    )
