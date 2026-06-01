from __future__ import annotations
import random
from dataclasses import dataclass
from core.models import SystemStats, AgentType


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
