from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AgentType(Enum):
    GENERATION = "generation"
    REFLECTION = "reflection"
    RANKING = "ranking"
    PROXIMITY = "proximity"
    EVOLUTION = "evolution"
    META_REVIEW = "meta_review"


class ReviewVerdict(Enum):
    PASSED = "passed"
    REJECTED = "rejected"
    FLAGGED = "flagged"


class GenerationStrategy(Enum):
    LITERATURE = "literature"
    DEBATE = "debate"
    ASSUMPTIONS = "assumptions"
    EXPANSION = "expansion"


class EvolutionStrategy(Enum):
    GROUNDING = "grounding"
    COHERENCE = "coherence"
    INSPIRATION = "inspiration"
    COMBINATION = "combination"
    SIMPLIFICATION = "simplification"
    OUT_OF_BOX = "out_of_box"


@dataclass
class Hypothesis:
    id: str
    run_id: str
    text: str
    summary: str
    generation_method: str
    source: str                          # "system" | "expert"
    category: Optional[str] = None
    evolved_from: Optional[str] = None   # parent hypothesis id
    elo_rating: float = 1200.0
    annotations: list[str] = field(default_factory=list)  # positive observations
    status: str = "active"               # "active" | "rejected"


@dataclass
class Review:
    id: str
    hypothesis_id: str
    tier: int                            # 0=expert, 1-6=system tiers
    critique: str
    verdict: Optional[str] = None        # "passed" | "rejected" | "flagged"
    web_citations: list[dict] = field(default_factory=list)


@dataclass
class TournamentMatch:
    id: str
    run_id: str
    h1_id: str
    h2_id: str
    winner_id: str
    match_type: str                      # "single_turn" | "multi_turn"
    debate_transcript: Optional[str] = None
    elo_before_h1: float = 0.0
    elo_before_h2: float = 0.0
    elo_after_h1: float = 0.0
    elo_after_h2: float = 0.0


@dataclass
class ResearchPlanConfig:
    run_id: str
    goal: str
    preferences: str
    attributes: list[str]
    constraints: str
    safety_approved: bool

    @property
    def idea_attributes(self) -> str:
        return " and ".join(self.attributes)


@dataclass
class SystemStats:
    n_hypotheses: int = 0
    n_pending_review: int = 0
    n_reviewed: int = 0
    tournament_progress: float = 0.0
    elo_variance: float = 0.0
    avg_proximity: float = 0.0
    generation_effectiveness: float = 1200.0
    evolution_effectiveness: float = 1200.0
    last_meta_review_age: int = 0


@dataclass
class AgentTask:
    priority: int
    agent_type: AgentType
    run_id: str
    hypothesis_id: Optional[str] = None
    strategy: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def __lt__(self, other: AgentTask) -> bool:
        return self.priority < other.priority
