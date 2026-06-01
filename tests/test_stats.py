import pytest
from core.stats import compute_weights, sample_agent_type, WeightThresholds
from core.models import SystemStats, AgentType


def default_thresholds():
    return WeightThresholds(
        min_hypothesis_count=8,
        elo_variance_threshold=5000.0,
        meta_review_interval=20,
    )


def test_weights_boost_generation_when_pool_small():
    stats = SystemStats(n_hypotheses=2, n_pending_review=0, elo_variance=6000.0, last_meta_review_age=0)
    weights = compute_weights(stats, default_thresholds())
    assert weights[AgentType.GENERATION] > weights[AgentType.RANKING]


def test_weights_boost_reflection_when_backlog_high():
    stats = SystemStats(n_hypotheses=10, n_pending_review=5, elo_variance=6000.0, last_meta_review_age=0)
    weights = compute_weights(stats, default_thresholds())
    # 5 pending > 10 * 0.3 = 3 -> reflection boosted
    assert weights[AgentType.REFLECTION] >= 2.0


def test_weights_boost_evolution_when_variance_low():
    stats = SystemStats(n_hypotheses=10, n_pending_review=0, elo_variance=100.0, last_meta_review_age=0)
    weights = compute_weights(stats, default_thresholds())
    assert weights[AgentType.EVOLUTION] > 1.0


def test_weights_boost_meta_review_when_stale():
    stats = SystemStats(n_hypotheses=10, n_pending_review=0, elo_variance=6000.0, last_meta_review_age=25)
    weights = compute_weights(stats, default_thresholds())
    assert weights[AgentType.META_REVIEW] >= 2.0


def test_sample_agent_type_is_deterministic_with_seed():
    weights = {
        AgentType.GENERATION: 1.0,
        AgentType.REFLECTION: 1.0,
        AgentType.RANKING: 1.0,
        AgentType.EVOLUTION: 1.0,
        AgentType.PROXIMITY: 0.5,
        AgentType.META_REVIEW: 0.3,
    }
    a = sample_agent_type(weights, seed=42)
    b = sample_agent_type(weights, seed=42)
    assert a == b


def test_sample_agent_type_returns_valid_type():
    weights = {AgentType.GENERATION: 1.0, AgentType.REFLECTION: 0.0}
    # With reflection at weight 0, only generation can be picked
    result = sample_agent_type(weights, seed=1)
    assert result == AgentType.GENERATION
