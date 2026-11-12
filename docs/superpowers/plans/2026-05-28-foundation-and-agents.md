# AI Co-Scientist: Foundation + Agents — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and test all core infrastructure and all 6 specialized agents of the AI co-scientist system (arXiv:2502.18864), with mocked Claude calls so every component is independently verifiable.

**Architecture:** Prompt-template-driven agents where each agent fills `.txt` templates with runtime variables and calls Claude. All hypothesis state persists to SQLite. A `BaseAgent` provides shared prompt rendering and the multi-turn self-play loop used by debate-style agents. This plan does NOT include the Supervisor or Gradio UI (those are Plan 2).

**Tech Stack:** Python 3.12, `anthropic` SDK, `aiosqlite`, `sentence-transformers`, `tavily-python`, `pytest`, `pytest-asyncio`

---

## File Map

```
coscientist/
├── pyproject.toml
├── config.yaml
├── core/
│   ├── __init__.py
│   ├── models.py          # Hypothesis, ResearchPlanConfig, SystemStats, AgentTask dataclasses
│   └── state.py           # SQLite schema + all async CRUD via aiosqlite
├── agents/
│   ├── __init__.py
│   ├── base.py            # BaseAgent: render_prompt + call_claude + run_turn_loop
│   ├── generation.py      # 4 strategies: literature, debate, assumptions, expansion
│   ├── reflection.py      # 6 tiers: initial, full, deep_verification, observation, simulation, tournament
│   ├── ranking.py         # Elo tournament orchestration + match execution
│   ├── proximity.py       # Similarity graph via sentence-transformers
│   ├── evolution.py       # 6 strategies: grounding, coherence, inspiration, combination, simplification, out_of_box
│   └── meta_review.py     # meta_critique + research_overview + research_contacts
├── tools/
│   ├── __init__.py
│   ├── claude.py          # Anthropic SDK wrapper with prompt caching
│   └── search.py          # Tavily web search wrapper
├── prompts/
│   ├── generation/
│   │   ├── literature.txt
│   │   ├── debate.txt
│   │   ├── assumptions.txt
│   │   └── expansion.txt
│   ├── reflection/
│   │   ├── initial.txt
│   │   ├── full.txt
│   │   ├── deep_verification.txt
│   │   ├── observation.txt
│   │   ├── simulation.txt
│   │   └── tournament.txt
│   ├── ranking/
│   │   ├── single_turn.txt
│   │   └── multi_turn_debate.txt
│   ├── evolution/
│   │   ├── grounding.txt
│   │   ├── coherence.txt
│   │   ├── inspiration.txt
│   │   ├── combination.txt
│   │   ├── simplification.txt
│   │   └── out_of_box.txt
│   └── meta_review/
│       ├── meta_critique.txt
│       ├── research_overview.txt
│       └── research_contacts.txt
└── tests/
    ├── conftest.py
    ├── test_models.py
    ├── test_state.py
    ├── test_claude.py
    ├── test_base_agent.py
    ├── test_generation.py
    ├── test_reflection.py
    ├── test_ranking.py
    ├── test_proximity.py
    ├── test_evolution.py
    └── test_meta_review.py
```

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `config.yaml`
- Create: `core/__init__.py`, `agents/__init__.py`, `tools/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "coscientist"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.40.0",
    "aiosqlite>=0.20.0",
    "sentence-transformers>=3.0.0",
    "tavily-python>=0.3.0",
    "gradio>=5.0.0",
    "pyyaml>=6.0",
    "numpy>=1.26.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-mock>=3.12.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Create `config.yaml`**

```yaml
anthropic:
  model_strong: claude-opus-4-7
  model_fast: claude-sonnet-4-6

supervisor:
  n_workers: 4
  tick_interval_seconds: 10
  checkpoint_interval: 5
  max_time_minutes: 30
  min_hypothesis_count: 8
  elo_variance_threshold: 5000
  meta_review_interval: 20

tournament:
  elo_initial: 1200.0
  elo_k_factor: 32.0
  multi_turn_threshold: 1350.0
  max_debate_turns: 10

proximity:
  model: all-MiniLM-L6-v2
  similarity_threshold: 0.5
  duplicate_threshold: 0.92

tools:
  search_provider: tavily
  max_search_results: 5

db_path: coscientist.db
log_path: coscientist.log
```

- [ ] **Step 3: Create empty `__init__.py` files and `tests/` directory**

```bash
mkdir -p core agents tools prompts/generation prompts/reflection prompts/ranking prompts/evolution prompts/meta_review tests
touch core/__init__.py agents/__init__.py tools/__init__.py tests/__init__.py
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -e ".[dev]"
```

Expected: No errors. `python -c "import anthropic, aiosqlite, gradio"` succeeds.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml config.yaml core/__init__.py agents/__init__.py tools/__init__.py
git commit -m "feat: project setup with dependencies"
```

---

## Task 2: Core Data Models

**Files:**
- Create: `core/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_models.py
import pytest
from core.models import Hypothesis, ResearchPlanConfig, AgentTask, AgentType, ReviewVerdict

def test_hypothesis_defaults():
    h = Hypothesis(
        id="h1",
        run_id="run1",
        text="Test hypothesis text",
        summary="A test hypothesis",
        generation_method="literature",
        source="system",
    )
    assert h.elo_rating == 1200.0
    assert h.status == "active"
    assert h.evolved_from is None
    assert h.annotations == []

def test_research_plan_config_idea_attributes():
    config = ResearchPlanConfig(
        run_id="run1",
        goal="Find a cure for ALS",
        preferences="Focus on novel mechanisms",
        attributes=["Novelty", "Feasibility"],
        constraints="Must be testable in vitro",
        safety_approved=True,
    )
    assert config.idea_attributes == "Novelty and Feasibility"

def test_agent_task_ordering():
    t1 = AgentTask(priority=1, agent_type=AgentType.GENERATION, run_id="run1")
    t2 = AgentTask(priority=2, agent_type=AgentType.REFLECTION, run_id="run1")
    assert t1 < t2
```

- [ ] **Step 2: Run tests, confirm failure**

```bash
pytest tests/test_models.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.models'`

- [ ] **Step 3: Write `core/models.py`**

```python
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
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_models.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add core/models.py tests/test_models.py
git commit -m "feat: core data models"
```

---

## Task 3: SQLite State Layer

**Files:**
- Create: `core/state.py`
- Create: `tests/test_state.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_state.py
import pytest
import aiosqlite
from core.state import StateStore
from core.models import Hypothesis, ResearchPlanConfig, Review, TournamentMatch


@pytest.fixture
async def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = StateStore(db_path)
    await store.init_db()
    return store


async def test_save_and_get_hypothesis(store):
    h = Hypothesis(
        id="h1", run_id="run1", text="full text", summary="short",
        generation_method="literature", source="system",
    )
    await store.save_hypothesis(h)
    result = await store.get_hypothesis("h1")
    assert result is not None
    assert result.id == "h1"
    assert result.elo_rating == 1200.0
    assert result.annotations == []


async def test_update_elo(store):
    h = Hypothesis(
        id="h2", run_id="run1", text="text", summary="s",
        generation_method="debate", source="system",
    )
    await store.save_hypothesis(h)
    await store.update_elo("h2", 1350.5)
    result = await store.get_hypothesis("h2")
    assert result.elo_rating == 1350.5


async def test_list_active_hypotheses(store):
    for i in range(3):
        h = Hypothesis(
            id=f"h{i}", run_id="run1", text="t", summary="s",
            generation_method="literature", source="system",
        )
        await store.save_hypothesis(h)
    # Reject one
    await store.set_hypothesis_status("h0", "rejected")
    active = await store.list_hypotheses("run1", status="active")
    assert len(active) == 2


async def test_save_and_list_reviews(store):
    h = Hypothesis(
        id="h1", run_id="run1", text="t", summary="s",
        generation_method="literature", source="system",
    )
    await store.save_hypothesis(h)
    r = Review(id="r1", hypothesis_id="h1", tier=1, critique="looks ok", verdict="passed")
    await store.save_review(r)
    reviews = await store.list_reviews("h1")
    assert len(reviews) == 1
    assert reviews[0].verdict == "passed"


async def test_save_and_list_matches(store):
    match = TournamentMatch(
        id="m1", run_id="run1", h1_id="h1", h2_id="h2",
        winner_id="h1", match_type="single_turn",
        elo_before_h1=1200.0, elo_before_h2=1200.0,
        elo_after_h1=1216.0, elo_after_h2=1184.0,
    )
    await store.save_match(match)
    matches = await store.list_matches("run1")
    assert len(matches) == 1
    assert matches[0].winner_id == "h1"
```

- [ ] **Step 2: Run tests, confirm failure**

```bash
pytest tests/test_state.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.state'`

- [ ] **Step 3: Write `core/state.py`**

```python
import json
import aiosqlite
from typing import Optional
from core.models import Hypothesis, Review, TournamentMatch, ResearchPlanConfig


CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS configs (
    run_id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    preferences TEXT NOT NULL,
    attributes TEXT NOT NULL,
    constraints TEXT NOT NULL,
    safety_approved BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    text TEXT NOT NULL,
    summary TEXT NOT NULL,
    category TEXT,
    generation_method TEXT NOT NULL,
    evolved_from TEXT,
    source TEXT NOT NULL,
    elo_rating REAL DEFAULT 1200.0,
    annotations TEXT DEFAULT '[]',
    status TEXT DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES configs(run_id)
);

CREATE TABLE IF NOT EXISTS reviews (
    id TEXT PRIMARY KEY,
    hypothesis_id TEXT NOT NULL,
    tier INTEGER NOT NULL,
    verdict TEXT,
    critique TEXT NOT NULL,
    web_citations TEXT DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
);

CREATE TABLE IF NOT EXISTS tournament_matches (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    h1_id TEXT NOT NULL,
    h2_id TEXT NOT NULL,
    winner_id TEXT NOT NULL,
    match_type TEXT NOT NULL,
    debate_transcript TEXT,
    elo_before_h1 REAL,
    elo_before_h2 REAL,
    elo_after_h1 REAL,
    elo_after_h2 REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS proximity_graph (
    h1_id TEXT NOT NULL,
    h2_id TEXT NOT NULL,
    similarity_score REAL NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (h1_id, h2_id)
);

CREATE TABLE IF NOT EXISTS meta_reviews (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    meta_critique TEXT NOT NULL,
    research_overview TEXT,
    research_contacts TEXT,
    tick INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class StateStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def init_db(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(CREATE_TABLES)
            await db.commit()

    async def save_hypothesis(self, h: Hypothesis) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO hypotheses
                   (id, run_id, text, summary, category, generation_method,
                    evolved_from, source, elo_rating, annotations, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (h.id, h.run_id, h.text, h.summary, h.category,
                 h.generation_method, h.evolved_from, h.source,
                 h.elo_rating, json.dumps(h.annotations), h.status),
            )
            await db.commit()

    async def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,)
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return Hypothesis(
            id=row["id"], run_id=row["run_id"], text=row["text"],
            summary=row["summary"], category=row["category"],
            generation_method=row["generation_method"],
            evolved_from=row["evolved_from"], source=row["source"],
            elo_rating=row["elo_rating"],
            annotations=json.loads(row["annotations"]),
            status=row["status"],
        )

    async def list_hypotheses(self, run_id: str, status: str = "active") -> list[Hypothesis]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM hypotheses WHERE run_id = ? AND status = ? ORDER BY elo_rating DESC",
                (run_id, status),
            ) as cursor:
                rows = await cursor.fetchall()
        return [
            Hypothesis(
                id=r["id"], run_id=r["run_id"], text=r["text"],
                summary=r["summary"], category=r["category"],
                generation_method=r["generation_method"],
                evolved_from=r["evolved_from"], source=r["source"],
                elo_rating=r["elo_rating"],
                annotations=json.loads(r["annotations"]),
                status=r["status"],
            )
            for r in rows
        ]

    async def update_elo(self, hypothesis_id: str, new_rating: float) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE hypotheses SET elo_rating = ? WHERE id = ?",
                (new_rating, hypothesis_id),
            )
            await db.commit()

    async def append_annotation(self, hypothesis_id: str, observation: str) -> None:
        h = await self.get_hypothesis(hypothesis_id)
        if h is None:
            return
        h.annotations.append(observation)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE hypotheses SET annotations = ? WHERE id = ?",
                (json.dumps(h.annotations), hypothesis_id),
            )
            await db.commit()

    async def set_hypothesis_status(self, hypothesis_id: str, status: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE hypotheses SET status = ? WHERE id = ?",
                (status, hypothesis_id),
            )
            await db.commit()

    async def save_review(self, review: Review) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO reviews
                   (id, hypothesis_id, tier, verdict, critique, web_citations)
                   VALUES (?,?,?,?,?,?)""",
                (review.id, review.hypothesis_id, review.tier,
                 review.verdict, review.critique,
                 json.dumps(review.web_citations)),
            )
            await db.commit()

    async def list_reviews(self, hypothesis_id: str) -> list[Review]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM reviews WHERE hypothesis_id = ? ORDER BY tier",
                (hypothesis_id,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [
            Review(
                id=r["id"], hypothesis_id=r["hypothesis_id"], tier=r["tier"],
                verdict=r["verdict"], critique=r["critique"],
                web_citations=json.loads(r["web_citations"]),
            )
            for r in rows
        ]

    async def save_match(self, match: TournamentMatch) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO tournament_matches
                   (id, run_id, h1_id, h2_id, winner_id, match_type,
                    debate_transcript, elo_before_h1, elo_before_h2,
                    elo_after_h1, elo_after_h2)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (match.id, match.run_id, match.h1_id, match.h2_id,
                 match.winner_id, match.match_type, match.debate_transcript,
                 match.elo_before_h1, match.elo_before_h2,
                 match.elo_after_h1, match.elo_after_h2),
            )
            await db.commit()

    async def list_matches(self, run_id: str) -> list[TournamentMatch]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM tournament_matches WHERE run_id = ? ORDER BY created_at",
                (run_id,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [
            TournamentMatch(
                id=r["id"], run_id=r["run_id"], h1_id=r["h1_id"],
                h2_id=r["h2_id"], winner_id=r["winner_id"],
                match_type=r["match_type"],
                debate_transcript=r["debate_transcript"],
                elo_before_h1=r["elo_before_h1"],
                elo_before_h2=r["elo_before_h2"],
                elo_after_h1=r["elo_after_h1"],
                elo_after_h2=r["elo_after_h2"],
            )
            for r in rows
        ]

    async def save_proximity(self, h1_id: str, h2_id: str, score: float) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO proximity_graph (h1_id, h2_id, similarity_score)
                   VALUES (?,?,?)""",
                (h1_id, h2_id, score),
            )
            await db.commit()

    async def get_similar_pairs(
        self, run_id: str, threshold: float = 0.5
    ) -> list[tuple[str, str, float]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT p.h1_id, p.h2_id, p.similarity_score
                   FROM proximity_graph p
                   JOIN hypotheses h1 ON p.h1_id = h1.id
                   JOIN hypotheses h2 ON p.h2_id = h2.id
                   WHERE h1.run_id = ? AND h2.run_id = ?
                     AND p.similarity_score >= ?
                   ORDER BY p.similarity_score DESC""",
                (run_id, run_id, threshold),
            ) as cursor:
                rows = await cursor.fetchall()
        return [(r["h1_id"], r["h2_id"], r["similarity_score"]) for r in rows]

    async def save_meta_review(
        self,
        id: str,
        run_id: str,
        meta_critique: str,
        research_overview: Optional[str],
        research_contacts: Optional[str],
        tick: int,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO meta_reviews
                   (id, run_id, meta_critique, research_overview, research_contacts, tick)
                   VALUES (?,?,?,?,?,?)""",
                (id, run_id, meta_critique, research_overview, research_contacts, tick),
            )
            await db.commit()

    async def get_latest_meta_review(self, run_id: str) -> Optional[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM meta_reviews WHERE run_id = ?
                   ORDER BY tick DESC LIMIT 1""",
                (run_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_state.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add core/state.py tests/test_state.py
git commit -m "feat: SQLite state layer with full CRUD"
```

---

## Task 4: Claude Wrapper with Prompt Caching

**Files:**
- Create: `tools/claude.py`
- Create: `tests/test_claude.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_claude.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from tools.claude import ClaudeClient


@pytest.fixture
def client():
    return ClaudeClient(model_strong="claude-opus-4-7", model_fast="claude-sonnet-4-6")


async def test_call_returns_text(client):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Test response")]

    with patch.object(client._client.messages, "create", return_value=mock_response):
        result = await client.call("You are helpful.", "What is ALS?", use_strong=False)

    assert result == "Test response"


async def test_strong_model_used_for_deep_tasks(client):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Deep analysis")]

    with patch.object(client._client.messages, "create", return_value=mock_response) as mock_create:
        await client.call("system", "prompt", use_strong=True)

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "claude-opus-4-7"


async def test_fast_model_used_by_default(client):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Fast response")]

    with patch.object(client._client.messages, "create", return_value=mock_response) as mock_create:
        await client.call("system", "prompt", use_strong=False)

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-6"
```

- [ ] **Step 2: Run tests, confirm failure**

```bash
pytest tests/test_claude.py -v
```

Expected: `ModuleNotFoundError: No module named 'tools.claude'`

- [ ] **Step 3: Write `tools/claude.py`**

```python
import anthropic
from typing import Optional


class ClaudeClient:
    def __init__(self, model_strong: str, model_fast: str):
        self.model_strong = model_strong
        self.model_fast = model_fast
        self._client = anthropic.Anthropic()

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        use_strong: bool = False,
        max_tokens: int = 8192,
    ) -> str:
        model = self.model_strong if use_strong else self.model_fast
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_claude.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add tools/claude.py tests/test_claude.py
git commit -m "feat: Claude wrapper with prompt caching"
```

---

## Task 5: Tavily Search Wrapper

**Files:**
- Create: `tools/search.py`
- Create: `tests/test_search.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_search.py
import pytest
from unittest.mock import patch, MagicMock
from tools.search import SearchTool


@pytest.fixture
def tool():
    return SearchTool(api_key="test-key", max_results=3)


async def test_search_returns_articles(tool):
    mock_results = {
        "results": [
            {"title": "ALS Study", "url": "http://example.com/1", "content": "ALS content"},
            {"title": "Motor Neuron", "url": "http://example.com/2", "content": "Motor content"},
        ]
    }
    with patch.object(tool._client, "search", return_value=mock_results):
        articles = await tool.search("ALS mechanisms")

    assert len(articles) == 2
    assert articles[0]["title"] == "ALS Study"
    assert "url" in articles[0]
    assert "content" in articles[0]


async def test_search_formats_for_prompt(tool):
    mock_results = {
        "results": [
            {"title": "Study A", "url": "http://a.com", "content": "Content A"},
        ]
    }
    with patch.object(tool._client, "search", return_value=mock_results):
        formatted = await tool.search_and_format("ALS", "Article on ALS mechanisms")

    assert "Study A" in formatted
    assert "Content A" in formatted
```

- [ ] **Step 2: Run tests, confirm failure**

```bash
pytest tests/test_search.py -v
```

Expected: `ModuleNotFoundError: No module named 'tools.search'`

- [ ] **Step 3: Write `tools/search.py`**

```python
from tavily import TavilyClient


class SearchTool:
    def __init__(self, api_key: str, max_results: int = 5):
        self._client = TavilyClient(api_key=api_key)
        self.max_results = max_results

    async def search(self, query: str) -> list[dict]:
        response = self._client.search(
            query=query,
            max_results=self.max_results,
            search_depth="advanced",
        )
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in response.get("results", [])
        ]

    async def search_and_format(self, query: str, context: str = "") -> str:
        articles = await self.search(query)
        if not articles:
            return "No relevant articles found."
        lines = []
        for i, a in enumerate(articles, 1):
            lines.append(f"[{i}] {a['title']}")
            lines.append(f"URL: {a['url']}")
            lines.append(f"Summary: {a['content'][:500]}")
            lines.append("")
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_search.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add tools/search.py tests/test_search.py
git commit -m "feat: Tavily search wrapper"
```

---

## Task 6: BaseAgent — Prompt Rendering and Turn Loop

**Files:**
- Create: `agents/base.py`
- Create: `tests/test_base_agent.py`
- Create: `prompts/generation/debate.txt` (used in turn loop test)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_base_agent.py
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from agents.base import BaseAgent
from tools.claude import ClaudeClient


@pytest.fixture
def client(mocker):
    c = mocker.MagicMock(spec=ClaudeClient)
    c.call = AsyncMock(return_value="test response")
    return c


@pytest.fixture
def agent(tmp_path, client):
    # Create a test prompt file
    prompt_dir = tmp_path / "prompts" / "test"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "hello.txt").write_text("Hello {name}, goal is {goal}.")
    return BaseAgent(client=client, prompts_dir=tmp_path / "prompts")


def test_render_prompt(agent):
    result = agent.render_prompt("test/hello", name="Alice", goal="cure ALS")
    assert result == "Hello Alice, goal is cure ALS."


async def test_call_claude_delegates(agent, client):
    result = await agent.call_claude("sys", "user")
    client.call.assert_called_once_with("sys", "user", use_strong=False, max_tokens=8192)
    assert result == "test response"


async def test_turn_loop_terminates_on_signal(agent, client):
    # Simulate: first turn returns text, second turn contains termination signal
    client.call = AsyncMock(side_effect=[
        "I propose three ideas...",
        "After debate... HYPOTHESIS\nFinal hypothesis text here.",
    ])
    final, transcript = await agent.run_turn_loop(
        template_name="test/hello",
        variables={"name": "Alice", "goal": "test"},
        transcript_key="transcript",
        termination_signal="HYPOTHESIS",
        max_turns=10,
    )
    assert "HYPOTHESIS" in final
    assert len(transcript) == 2


async def test_turn_loop_respects_max_turns(agent, client):
    client.call = AsyncMock(return_value="No termination signal here.")
    final, transcript = await agent.run_turn_loop(
        template_name="test/hello",
        variables={"name": "Alice", "goal": "test"},
        transcript_key="transcript",
        termination_signal="HYPOTHESIS",
        max_turns=3,
    )
    assert len(transcript) == 3
```

- [ ] **Step 2: Run tests, confirm failure**

```bash
pytest tests/test_base_agent.py -v
```

Expected: `ModuleNotFoundError: No module named 'agents.base'`

- [ ] **Step 3: Write `agents/base.py`**

```python
from __future__ import annotations
from pathlib import Path
from typing import Optional
from tools.claude import ClaudeClient


class BaseAgent:
    def __init__(self, client: ClaudeClient, prompts_dir: Path):
        self.client = client
        self.prompts_dir = Path(prompts_dir)

    def render_prompt(self, template_name: str, **variables) -> str:
        path = self.prompts_dir / f"{template_name}.txt"
        template = path.read_text()
        return template.format(**variables)

    async def call_claude(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        use_strong: bool = False,
        max_tokens: int = 8192,
    ) -> str:
        return await self.client.call(
            system_prompt, user_prompt, use_strong=use_strong, max_tokens=max_tokens
        )

    async def run_turn_loop(
        self,
        template_name: str,
        variables: dict,
        transcript_key: str,
        termination_signal: str,
        max_turns: int = 10,
        system_prompt: str = "You are an expert scientific researcher.",
        use_strong: bool = False,
    ) -> tuple[str, list[str]]:
        transcript: list[str] = []
        last_response = ""
        for _ in range(max_turns):
            vars_with_transcript = {
                **variables,
                transcript_key: "\n".join(transcript),
            }
            prompt = self.render_prompt(template_name, **vars_with_transcript)
            response = await self.call_claude(
                system_prompt, prompt, use_strong=use_strong
            )
            transcript.append(response)
            last_response = response
            if termination_signal in response:
                break
        return last_response, transcript
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_base_agent.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add agents/base.py tests/test_base_agent.py
git commit -m "feat: BaseAgent with prompt rendering and turn loop"
```

---

## Task 7: All Prompt Files

**Files:**
- Create: all 17 `.txt` files in `prompts/`

Each prompt is transcribed faithfully from the paper's appendix (Figures A.1–A.8) or inferred from the paper's descriptions where no appendix prompt exists. Template variables use `{variable_name}` syntax.

- [ ] **Step 1: Create `prompts/generation/literature.txt`** (from Figure A.1)

```
You are an expert tasked with formulating a novel and robust hypothesis to address the following objective.

Describe the proposed hypothesis in detail, including specific entities, mechanisms, and anticipated outcomes.

This description is intended for an audience of domain experts.

You have conducted a thorough review of relevant literature and developed a logical framework for addressing the objective. The articles consulted, along with your analytical reasoning, are provided below.

Goal: {goal}

Criteria for a strong hypothesis:
{preferences}

Existing hypothesis (if applicable):
{source_hypothesis}

{instructions}

Literature review and analytical rationale (chronologically ordered, beginning with the most recent analysis):

{articles_with_reasoning}

Proposed hypothesis (detailed description for domain experts):

Introduction:

Recent Findings:

Hypothesis:

Rationale and Specificity:

Experimental Design:

Summary (one sentence for non-experts):

Category:
```

- [ ] **Step 2: Create `prompts/generation/debate.txt`** (from Figure A.2)

```
You are an expert participating in a collaborative discourse concerning the generation of a {idea_attributes} hypothesis. You will engage in a simulated discussion with other experts. The overarching objective of this discourse is to collaboratively develop a novel and robust {idea_attributes} hypothesis.

Goal: {goal}

Criteria for a high-quality hypothesis:
{preferences}

Instructions:
{instructions}

Review Overview:
{reviews_overview}

Procedure:

Initial contribution (if initiating the discussion):
    Propose three distinct {idea_attributes} hypotheses.

Subsequent contributions (continuing the discussion):
    * Pose clarifying questions if ambiguities or uncertainties arise.
    * Critically evaluate the hypotheses proposed thus far, addressing the following aspects:
        - Adherence to {idea_attributes} criteria.
        - Utility and practicality.
        - Level of detail and specificity.
    * Identify any weaknesses or potential limitations.
    * Propose concrete improvements and refinements to address identified weaknesses.
    * Conclude your response with a refined iteration of the hypothesis.

General guidelines:
    * Exhibit boldness and creativity in your contributions.
    * Maintain a helpful and collaborative approach.
    * Prioritize the generation of a high-quality {idea_attributes} hypothesis.

Termination condition:
    When sufficient discussion has transpired (typically 3-5 conversational turns,
    with a maximum of 10 turns) and all relevant questions and points have been
    thoroughly addressed and clarified, conclude the process by writing "HYPOTHESIS"
    (in all capital letters) followed by a concise and self-contained exposition of the finalized idea.

#BEGIN TRANSCRIPT#
{transcript}
#END TRANSCRIPT#

Your Turn:
```

- [ ] **Step 3: Create `prompts/generation/assumptions.txt`** (inferred from paper §3.3.1)

```
You are an expert scientific researcher tasked with generating a novel research hypothesis through systematic assumption identification.

Goal: {goal}

Criteria for a strong hypothesis:
{preferences}

{instructions}

Your task is to iteratively identify testable intermediate assumptions that, if proven true, could lead to novel scientific discovery.

Procedure:
1. Identify a plausible high-level assumption relevant to the research goal.
2. For each assumption, identify 2-3 testable sub-assumptions via conditional reasoning:
   "If [assumption] is true, then [sub-assumption] could follow because..."
3. Evaluate whether each sub-assumption is independently testable.
4. Aggregate the most promising assumption chain into a complete, testable hypothesis.

Work through at least 3 assumption chains before selecting the strongest one.

After exploring the assumption chains, write "HYPOTHESIS" (in all capital letters) followed by the complete hypothesis built from the strongest assumption chain.

#BEGIN TRANSCRIPT#
{transcript}
#END TRANSCRIPT#

Your Turn:
```

- [ ] **Step 4: Create `prompts/generation/expansion.txt`** (inferred from paper §3.3.1)

```
You are an expert scientific researcher. Your task is to generate a novel research hypothesis that explores areas not yet covered by existing work on this research goal.

Goal: {goal}

Criteria for a strong hypothesis:
{preferences}

{instructions}

Current research overview (areas already explored):
{research_overview}

Summary of existing hypotheses in the pool:
{existing_hypotheses_summary}

Your task:
1. Identify 2-3 areas of the hypothesis space that are underexplored or entirely absent from the current pool.
2. Select the most promising unexplored area.
3. Generate a novel hypothesis that fills this gap, distinct from all existing hypotheses.

Proposed hypothesis (detailed description for domain experts):

Introduction:

Recent Findings:

Hypothesis:

Rationale and Specificity:

Experimental Design:

Summary (one sentence for non-experts):

Category:
```

- [ ] **Step 5: Create `prompts/reflection/initial.txt`**

```
You are a scientific peer reviewer performing an initial screening of a research hypothesis.

Goal: {goal}

Evaluation criteria:
{preferences}

Hypothesis to review:
{hypothesis}

Perform an initial review assessing:
1. Correctness: Is the hypothesis free of obvious scientific errors?
2. Quality: Is it well-reasoned and specific enough to be meaningful?
3. Novelty: Does it go beyond synthesizing existing information?
4. Safety: Does it avoid enabling unsafe, unethical, or harmful research?

This initial review does not use external tools. Its purpose is to quickly discard flawed, non-novel, or otherwise unsuitable hypotheses.

Provide your assessment as:

Correctness: [HIGH/MEDIUM/LOW] - [brief reason]
Quality: [HIGH/MEDIUM/LOW] - [brief reason]
Novelty: [HIGH/MEDIUM/LOW] - [brief reason]
Safety: [SAFE/UNSAFE] - [brief reason]

Critique:
[bulleted list of specific issues]

Verdict: [PASSED/REJECTED]
Reason for verdict: [one sentence]
```

- [ ] **Step 6: Create `prompts/reflection/full.txt`** (from Figure A.13 output structure)

```
You are a scientific peer reviewer performing a thorough review of a research hypothesis.

Goal: {goal}

Evaluation criteria:
{preferences}

Hypothesis to review:
{hypothesis}

You have access to the following relevant literature:

{articles_with_reasoning}

Perform a full review covering:

Related Articles:
[List numbered references from the provided literature that are relevant]

Assumptions of the Idea:
[List each key assumption the hypothesis makes]

Reasoning about assumptions:
[For each assumption: evaluate plausibility given current literature. State if it needs experimental verification.]

Novelty:
Aspects already explored: [list what is known]
Novel aspects: [list what is genuinely new about this hypothesis]

Overall critique:
[Bulleted list of specific strengths and weaknesses]

Verdict: [PASSED/REJECTED/FLAGGED]
```

- [ ] **Step 7: Create `prompts/reflection/deep_verification.txt`** (from Figure A.14 output structure)

```
You are a scientific expert performing deep verification of a research hypothesis.

Goal: {goal}

Hypothesis:
{hypothesis}

Prior review:
{prior_review}

Your task is to decompose this hypothesis into its constituent assumptions and evaluate each independently.

Step 1 — List all assumptions:
[Number each assumption the hypothesis relies on]

Step 2 — For each assumption, break it into fundamental sub-assumptions and evaluate:
[For each sub-assumption:]
- Decontextualize: state it independently of the hypothesis
- Evaluate: is this sub-assumption supported by established science?
- Evidence: cite any known supporting or contradicting evidence
- Verdict: [SUPPORTED/PLAUSIBLE/UNSUPPORTED/CONTRADICTED]

Step 3 — Probing question:
Identify the single most critical assumption. Generate a hard probing question that challenges it:
Question: [probing question]
Answer: [detailed reasoning]
Impact: [does this answer fundamentally invalidate the hypothesis, or is it a non-fundamental error that evolution could address?]

Step 4 — Overall deep verification verdict:
Fundamental errors found: [YES/NO]
If YES: [which assumption is fundamental and why it invalidates the hypothesis]
If NO: [non-fundamental errors noted for evolution to address]

Verdict: [PASSED/REJECTED]
```

- [ ] **Step 8: Create `prompts/reflection/observation.txt`** (from Figure A.3)

```
You are an expert in scientific hypothesis evaluation. Your task is to analyze the relationship between a provided hypothesis and observations from a scientific article. Specifically, determine if the hypothesis provides a novel causal explanation for the observations, or if they contradict it.

Instructions:

1. Observation extraction: list relevant observations from the article.
2. Causal analysis (individual): for each observation:
   a. State if its cause is already established.
   b. Assess if the hypothesis could be a causal factor (hypothesis => observation).
   c. Start with: "would we see this observation if the hypothesis was true?":
   d. Explain if it's a novel explanation. If not, or if a better explanation exists, state: "not a missing piece."
3. Causal analysis (summary): determine if the hypothesis offers a novel explanation for a subset of observations. Include reasoning. Start with: "would we see some of the observations if the hypothesis was true?".
4. Disproof analysis: determine if any observations contradict the hypothesis. Start with: "does some observations disprove the hypothesis?".
5. Conclusion: state: "hypothesis: <already explained, other explanations more likely, missing piece, neutral, or disproved>".

Scoring:
    * Already explained: hypothesis consistent, but causes are known. No novel explanation.
    * Other explanations more likely: hypothesis *could* explain, but better explanations exist.
    * Missing piece: hypothesis offers a novel, plausible explanation.
    * Neutral: hypothesis neither explains nor is contradicted.
    * Disproved: observations contradict the hypothesis.

Important: if observations are expected regardless of the hypothesis, and don't disprove it, it's neutral.

Article:
{article}

Hypothesis:
{hypothesis}

Response (provide reasoning. end with: "hypothesis: <already explained, other explanations more likely, missing piece, neutral, or disproved>".):
```

- [ ] **Step 9: Create `prompts/reflection/simulation.txt`**

```
You are an expert scientific researcher tasked with simulating a proposed mechanism step-by-step to identify potential failure scenarios.

Goal: {goal}

Hypothesis:
{hypothesis}

Your task is to simulate the proposed mechanism as if running it in a thought experiment. Work through each step of the proposed mechanism:

Step-by-step simulation:
[For each step in the proposed mechanism:]
Step N: [describe what happens at this step]
Plausibility: [HIGH/MEDIUM/LOW]
Potential failure: [describe what could go wrong at this step]

Identified failure scenarios:
[List the most critical ways the mechanism could fail]

Simulation verdict:
Overall plausibility: [HIGH/MEDIUM/LOW]
Critical bottlenecks: [list the 1-3 steps most likely to fail]

Verdict: [PASSED/FLAGGED]
Note for evolution: [what specific issues should be addressed to make this hypothesis more robust]
```

- [ ] **Step 10: Create `prompts/reflection/tournament.txt`**

```
You are an expert scientific peer reviewer. You are reviewing a hypothesis that has already participated in tournament matches. Use the tournament history and the meta-review critique to perform a targeted, adapted review.

Goal: {goal}

Evaluation criteria:
{preferences}

Hypothesis:
{hypothesis}

Prior reviews:
{prior_reviews}

Meta-review critique (common issues across all hypotheses in this research area):
{meta_critique}

Tournament history for this hypothesis:
{tournament_history}

Perform a recurrent review that:
1. Addresses the specific recurring issues identified in the meta-review critique
2. Evaluates whether prior critiques have been adequately addressed
3. Identifies any new issues surfaced by tournament comparisons

Recurrent review:

Meta-critique compliance:
[For each issue in the meta-critique: does this hypothesis address it? YES/PARTIAL/NO]

Improvements since initial review:
[What has improved or worsened based on tournament feedback]

Updated critique:
[Refined bulleted critique incorporating tournament learnings]

Updated verdict: [PASSED/REJECTED/FLAGGED]
```

- [ ] **Step 11: Create `prompts/ranking/single_turn.txt`** (from Figure A.4)

```
You are an expert evaluator tasked with comparing two hypotheses.

Evaluate the two provided hypotheses (hypothesis 1 and hypothesis 2) and determine which one is superior based on the specified {idea_attributes}.
Provide a concise rationale for your selection, concluding with the phrase "better hypothesis: <1 or 2>".

Goal: {goal}

Evaluation criteria:
{preferences}

Considerations:
{notes}
Each hypothesis includes an independent review. These reviews may contain numerical scores.
Disregard these scores in your comparative analysis, as they may not be directly comparable across reviews.

Hypothesis 1:
{hypothesis_1}

Hypothesis 2:
{hypothesis_2}

Review of hypothesis 1:
{review_1}

Review of hypothesis 2:
{review_2}

Reasoning and conclusion (end with "better hypothesis: <1 or 2>"):
```

- [ ] **Step 12: Create `prompts/ranking/multi_turn_debate.txt`** (from Figure A.5)

```
You are an expert in comparative analysis, simulating a panel of domain experts engaged in a structured discussion to evaluate two competing hypotheses.
The objective is to rigorously determine which hypothesis is superior based on a predefined set of attributes and criteria.
The experts possess no pre-existing biases toward either hypothesis and are solely focused on identifying the optimal choice, given that only one can be implemented.

Goal: {goal}

Criteria for hypothesis superiority:
{preferences}

Hypothesis 1:
{hypothesis_1}

Hypothesis 2:
{hypothesis_2}

Initial review of hypothesis 1:
{review_1}

Initial review of hypothesis 2:
{review_2}

Additional notes:
{notes}

Debate procedure:

The discussion will unfold in a series of turns, typically ranging from 3 to 5, with a maximum of 10.

Turn 1: begin with a concise summary of both hypotheses and their respective initial reviews.

Subsequent turns:

    * Pose clarifying questions to address any ambiguities or uncertainties.
    * Critically evaluate each hypothesis in relation to the stated Goal and Criteria.
    This evaluation should consider aspects such as:
        - Potential for correctness/validity.
        - Utility and practical applicability.
        - Sufficiency of detail and specificity.
        - Novelty and originality.
        - Desirability for implementation.
    * Identify and articulate any weaknesses, limitations, or potential flaws in either hypothesis.

Termination and judgment:

Once the discussion has reached a point of sufficient depth (typically 3-5 turns, up to 10 turns)
and all relevant questions and concerns have been thoroughly addressed, provide a conclusive judgment.
This judgment should succinctly state the rationale for the selection.
Then, indicate the superior hypothesis by writing "better idea: ",
followed by "1" (for hypothesis 1) or "2" (for hypothesis 2).

#BEGIN TRANSCRIPT#
{transcript}
#END TRANSCRIPT#

Your Turn:
```

- [ ] **Step 13: Create all 6 Evolution prompts**

`prompts/evolution/grounding.txt`:
```
You are an expert scientific researcher tasked with improving a research hypothesis by grounding it in literature.

Goal: {goal}

Evaluation criteria:
{preferences}

Original hypothesis:
{hypothesis}

Weaknesses identified in prior reviews:
{weaknesses}

Relevant literature found:
{articles_with_reasoning}

Your task:
1. Identify specific reasoning gaps or unsupported claims in the original hypothesis.
2. Use the provided literature to fill these gaps with evidence-based elaboration.
3. Generate a NEW, improved hypothesis that addresses the identified weaknesses.
4. Do not simply restate the original — produce a meaningfully improved version.

New hypothesis (detailed description for domain experts):

Introduction:

Recent Findings:

Hypothesis:

Rationale and Specificity:

Experimental Design:

Summary (one sentence for non-experts):

Category:
```

`prompts/evolution/coherence.txt` (from Figure A.6):
```
You are an expert in scientific research and technological feasibility analysis.
Your task is to refine the provided conceptual idea, enhancing its practical implementability by leveraging contemporary technological capabilities. Ensure the revised concept retains its novelty, logical coherence, and specific articulation.

Goal: {goal}

Guidelines:
1. Begin with an introductory overview of the relevant scientific domain.
2. Provide a concise synopsis of recent pertinent research findings and related investigations, highlighting successful methodologies and established precedents.
3. Articulate a reasoned argument for how current technological advancements can facilitate the realization of the proposed concept.
4. CORE CONTRIBUTION: Develop a detailed, innovative, and technologically viable alternative to achieve the objective, emphasizing simplicity and practicality.

Evaluation Criteria:
{preferences}

Original Conceptualization:
{hypothesis}

Response:
```

`prompts/evolution/inspiration.txt`:
```
You are an expert scientific researcher. Your task is to generate a novel research hypothesis inspired by an existing top-ranked hypothesis.

Goal: {goal}

Criteria for a strong hypothesis:
{preferences}

Source hypothesis for inspiration (build upon its core insight, do not replicate it):
{hypothesis}

Generate a new hypothesis that:
1. Takes the core insight of the source hypothesis as a starting point
2. Extends, reframes, or applies it in a new direction
3. Is distinct enough to represent a genuinely different scientific proposal
4. Is independently testable

New hypothesis (detailed description for domain experts):

Introduction:

Recent Findings:

Hypothesis:

Rationale and Specificity:

Experimental Design:

Summary (one sentence for non-experts):

Category:
```

`prompts/evolution/combination.txt`:
```
You are an expert scientific researcher. Your task is to synthesize the best aspects of multiple top-ranked hypotheses into a single, superior hypothesis.

Goal: {goal}

Criteria for a strong hypothesis:
{preferences}

Top-ranked hypotheses to synthesize:
{hypotheses}

Your task:
1. Identify the strongest element of each hypothesis
2. Determine how these elements can be unified into a coherent single mechanism or proposal
3. Generate a NEW synthesized hypothesis that is more comprehensive than any individual input
4. The result should not be a list — it should be a unified, coherent hypothesis

Synthesized hypothesis (detailed description for domain experts):

Introduction:

Recent Findings:

Hypothesis:

Rationale and Specificity:

Experimental Design:

Summary (one sentence for non-experts):

Category:
```

`prompts/evolution/simplification.txt`:
```
You are an expert scientific researcher. Your task is to simplify a complex hypothesis to its minimal testable core.

Goal: {goal}

Criteria for a strong hypothesis:
{preferences}

Original hypothesis:
{hypothesis}

Your task:
1. Identify the single most important, testable claim in the original hypothesis
2. Strip away secondary claims, caveats, and elaborations
3. Produce a simplified hypothesis that focuses on this one testable core claim
4. The simplified hypothesis should be easier to validate experimentally while retaining the key scientific insight

Simplified hypothesis (detailed description for domain experts):

Introduction:

Recent Findings:

Hypothesis:

Rationale and Specificity:

Experimental Design:

Summary (one sentence for non-experts):

Category:
```

`prompts/evolution/out_of_box.txt` (from Figure A.7):
```
You are an expert researcher tasked with generating a novel, singular hypothesis inspired by analogous elements from provided concepts.

Goal: {goal}

Instructions:
1. Provide a concise introduction to the relevant scientific domain.
2. Summarize recent findings and pertinent research, highlighting successful approaches.
3. Identify promising avenues for exploration that may yield innovative hypotheses.
4. CORE HYPOTHESIS: Develop a detailed, original, and specific single hypothesis for achieving the stated goal, leveraging analogous principles from the provided ideas. This should not be a mere aggregation of existing methods or entities. Think out-of-the-box.

Criteria for a robust hypothesis:
{preferences}

Inspiration may be drawn from the following concepts (utilize analogy and inspiration, not direct replication):
{hypotheses}

Response:
```

- [ ] **Step 14: Create all 3 Meta-review prompts**

`prompts/meta_review/meta_critique.txt` (from Figure A.8):
```
You are an expert in scientific research and meta-analysis.
Synthesize a comprehensive meta-review of provided reviews pertaining to the following research goal:

Goal: {goal}

Preferences:
{preferences}

Additional instructions:
{instructions}

Provided reviews for meta-analysis:
{reviews}

Instructions:
    * Generate a structured meta-analysis report of the provided reviews.
    * Focus on identifying recurring critique points and common issues raised by reviewers.
    * The generated meta-analysis should provide actionable insights for researchers developing future proposals.
    * Refrain from evaluating individual proposals or reviews; focus on producing a synthesized meta-analysis.

Response:
```

`prompts/meta_review/research_overview.txt`:
```
You are an expert scientific researcher. Your task is to synthesize the top-ranked hypotheses in a research area into a comprehensive research overview that maps the current hypothesis landscape and identifies future directions.

Goal: {goal}

Top-ranked hypotheses (ordered by Elo rating):
{top_hypotheses}

Generate a research overview with the following structure:

[Main Research Directions]
For each major direction identified across the hypotheses:

Direction: [name and brief description]
Rationale: [why this direction is important and promising]
Recent Findings: [what the current hypotheses have established in this area]
Areas of Research:
  Sub-area: [name]
    Why Research? [scientific motivation]
    What to Research? [specific open questions]
    Example Idea: [a concrete hypothesis or experiment]

The overview should highlight both well-covered areas and notable gaps.
```

`prompts/meta_review/research_contacts.txt`:
```
You are an expert scientific researcher. Based on the literature cited in hypothesis reviews for this research goal, identify qualified domain experts who would be valuable collaborators or reviewers.

Goal: {goal}

Literature cited in reviews:
{cited_literature}

For each relevant research direction, identify 1-2 potential research contacts:

Research Direction: [title of the research direction]
[Researcher name(s)]: [explanation of their relevance — what they study, why their expertise is valuable, and what specific experiments or analyses they could contribute to]

Focus on researchers whose published work directly addresses key aspects of the hypotheses under review.
```

- [ ] **Step 15: Run verify all prompt files exist**

```bash
find prompts -name "*.txt" | sort
```

Expected output (17 files):
```
prompts/evolution/coherence.txt
prompts/evolution/combination.txt
prompts/evolution/grounding.txt
prompts/evolution/inspiration.txt
prompts/evolution/out_of_box.txt
prompts/evolution/simplification.txt
prompts/generation/assumptions.txt
prompts/generation/debate.txt
prompts/generation/expansion.txt
prompts/generation/literature.txt
prompts/meta_review/meta_critique.txt
prompts/meta_review/research_contacts.txt
prompts/meta_review/research_overview.txt
prompts/reflection/deep_verification.txt
prompts/reflection/full.txt
prompts/reflection/initial.txt
prompts/reflection/observation.txt
prompts/reflection/simulation.txt
prompts/reflection/tournament.txt
prompts/ranking/multi_turn_debate.txt
prompts/ranking/single_turn.txt
```

- [ ] **Step 16: Commit**

```bash
git add prompts/
git commit -m "feat: all 17 agent prompt templates"
```

---

## Task 8: Generation Agent

**Files:**
- Create: `agents/generation.py`
- Create: `tests/test_generation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_generation.py
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from agents.generation import GenerationAgent
from agents.base import BaseAgent
from core.models import ResearchPlanConfig, GenerationStrategy


@pytest.fixture
def config():
    return ResearchPlanConfig(
        run_id="run1",
        goal="Develop a novel hypothesis for ALS mechanisms",
        preferences="Focus on novel, testable hypotheses",
        attributes=["Novelty", "Feasibility"],
        constraints="Must be testable in vitro",
        safety_approved=True,
    )


@pytest.fixture
def mock_base(tmp_path):
    prompts_dir = Path(__file__).parent.parent / "prompts"
    client = MagicMock()
    base = BaseAgent(client=client, prompts_dir=prompts_dir)
    base.call_claude = AsyncMock(return_value=(
        "HYPOTHESIS\n"
        "Introduction: ALS context\n"
        "Recent Findings: TDP-43\n"
        "Hypothesis: Stress induces PTMs on Nup98\n"
        "Rationale and Specificity: Novel because...\n"
        "Experimental Design: Use iPSC cells\n"
        "Summary: Stress-induced PTMs on nucleoporins cause ALS\n"
        "Category: Neurodegeneration"
    ))
    base.run_turn_loop = AsyncMock(return_value=(
        "HYPOTHESIS\nHypothesis: Stress induces PTMs on Nup98\nSummary: PTM hypothesis\nCategory: Neurodegeneration",
        ["turn1", "HYPOTHESIS\nHypothesis: ..."]
    ))
    return base


@pytest.fixture
def agent(mock_base):
    return GenerationAgent(base=mock_base)


async def test_literature_strategy_returns_hypothesis(agent, config):
    articles = "Article 1: TDP-43 study..."
    h = await agent.run_literature(config, articles_with_reasoning=articles)
    assert h.text != ""
    assert h.summary != ""
    assert h.generation_method == "literature"
    assert h.source == "system"
    assert h.run_id == "run1"


async def test_debate_strategy_returns_hypothesis(agent, config):
    h = await agent.run_debate(config, reviews_overview="No prior meta-review")
    assert h.generation_method == "debate"
    assert "HYPOTHESIS" in agent.base.run_turn_loop.call_args[1].get("termination_signal", "") or \
           "HYPOTHESIS" in str(agent.base.run_turn_loop.call_args)


async def test_hypothesis_has_uuid(agent, config):
    h1 = await agent.run_literature(config, articles_with_reasoning="articles")
    h2 = await agent.run_literature(config, articles_with_reasoning="articles")
    assert h1.id != h2.id
```

- [ ] **Step 2: Run tests, confirm failure**

```bash
pytest tests/test_generation.py -v
```

Expected: `ModuleNotFoundError: No module named 'agents.generation'`

- [ ] **Step 3: Write `agents/generation.py`**

```python
import uuid
import re
from agents.base import BaseAgent
from core.models import Hypothesis, ResearchPlanConfig


def _extract_field(text: str, field: str, default: str = "") -> str:
    pattern = rf"^{re.escape(field)}:\s*(.+?)(?=\n[A-Z][a-z]|\Z)"
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else default


def _parse_hypothesis_output(text: str, config: ResearchPlanConfig, method: str) -> Hypothesis:
    # Extract after HYPOTHESIS marker if present
    if "HYPOTHESIS" in text:
        text = text[text.index("HYPOTHESIS"):]

    summary = _extract_field(text, "Summary") or text[:120].strip()
    category = _extract_field(text, "Category")

    return Hypothesis(
        id=str(uuid.uuid4()),
        run_id=config.run_id,
        text=text,
        summary=summary,
        category=category or None,
        generation_method=method,
        source="system",
    )


class GenerationAgent:
    def __init__(self, base: BaseAgent):
        self.base = base

    async def run_literature(
        self,
        config: ResearchPlanConfig,
        articles_with_reasoning: str,
        source_hypothesis: str = "",
        instructions: str = "",
    ) -> Hypothesis:
        prompt = self.base.render_prompt(
            "generation/literature",
            goal=config.goal,
            preferences=config.preferences,
            source_hypothesis=source_hypothesis,
            instructions=instructions,
            articles_with_reasoning=articles_with_reasoning,
        )
        response = await self.base.call_claude(
            "You are an expert scientific researcher.", prompt
        )
        return _parse_hypothesis_output(response, config, "literature")

    async def run_debate(
        self,
        config: ResearchPlanConfig,
        reviews_overview: str = "",
        instructions: str = "",
    ) -> Hypothesis:
        _, transcript = await self.base.run_turn_loop(
            template_name="generation/debate",
            variables={
                "goal": config.goal,
                "preferences": config.preferences,
                "idea_attributes": config.idea_attributes,
                "instructions": instructions,
                "reviews_overview": reviews_overview,
            },
            transcript_key="transcript",
            termination_signal="HYPOTHESIS",
            max_turns=10,
            system_prompt="You are an expert scientific researcher.",
        )
        final_text = transcript[-1] if transcript else ""
        return _parse_hypothesis_output(final_text, config, "debate")

    async def run_assumptions(
        self,
        config: ResearchPlanConfig,
        instructions: str = "",
    ) -> Hypothesis:
        _, transcript = await self.base.run_turn_loop(
            template_name="generation/assumptions",
            variables={
                "goal": config.goal,
                "preferences": config.preferences,
                "instructions": instructions,
            },
            transcript_key="transcript",
            termination_signal="HYPOTHESIS",
            max_turns=10,
            system_prompt="You are an expert scientific researcher.",
        )
        final_text = transcript[-1] if transcript else ""
        return _parse_hypothesis_output(final_text, config, "assumptions")

    async def run_expansion(
        self,
        config: ResearchPlanConfig,
        research_overview: str,
        existing_hypotheses_summary: str,
        instructions: str = "",
    ) -> Hypothesis:
        prompt = self.base.render_prompt(
            "generation/expansion",
            goal=config.goal,
            preferences=config.preferences,
            instructions=instructions,
            research_overview=research_overview,
            existing_hypotheses_summary=existing_hypotheses_summary,
        )
        response = await self.base.call_claude(
            "You are an expert scientific researcher.", prompt
        )
        return _parse_hypothesis_output(response, config, "expansion")
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_generation.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add agents/generation.py tests/test_generation.py
git commit -m "feat: Generation agent with 4 strategies"
```

---

## Task 9: Reflection Agent

**Files:**
- Create: `agents/reflection.py`
- Create: `tests/test_reflection.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_reflection.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
from agents.reflection import ReflectionAgent
from agents.base import BaseAgent
from core.models import Hypothesis, ResearchPlanConfig, Review


@pytest.fixture
def config():
    return ResearchPlanConfig(
        run_id="run1",
        goal="ALS mechanisms",
        preferences="Novel and feasible",
        attributes=["Novelty", "Feasibility"],
        constraints="testable",
        safety_approved=True,
    )


@pytest.fixture
def hypothesis():
    return Hypothesis(
        id="h1", run_id="run1", text="full text of hypothesis",
        summary="A hypothesis about PTMs", generation_method="debate", source="system",
    )


@pytest.fixture
def mock_base():
    client = MagicMock()
    base = BaseAgent(client=client, prompts_dir=Path(__file__).parent.parent / "prompts")
    base.call_claude = AsyncMock(return_value="Correctness: HIGH\nQuality: HIGH\nNovelty: HIGH\nSafety: SAFE\nVerdict: PASSED\nReason for verdict: Well-reasoned hypothesis.")
    return base


@pytest.fixture
def agent(mock_base):
    return ReflectionAgent(base=mock_base)


async def test_initial_review_passed(agent, hypothesis, config):
    review = await agent.run_initial_review(hypothesis, config)
    assert review.tier == 1
    assert review.hypothesis_id == "h1"
    assert review.verdict == "passed"


async def test_initial_review_rejected_on_keyword(agent, hypothesis, config, mock_base):
    mock_base.call_claude = AsyncMock(return_value="Verdict: REJECTED\nReason for verdict: Not novel.")
    review = await agent.run_initial_review(hypothesis, config)
    assert review.verdict == "rejected"


async def test_observation_review_extracts_missing_piece(agent, hypothesis, config, mock_base):
    mock_base.call_claude = AsyncMock(return_value=(
        "1. Observations: widespread TDP-43 in ALS neurons\n"
        "2. Analysis: would we see this if hypothesis true? Yes, because PTMs alter NPC function\n"
        "hypothesis: missing piece"
    ))
    review, observation = await agent.run_observation_review(
        hypothesis, config, article="Article content here"
    )
    assert review.tier == 4
    assert observation is not None
    assert len(observation) > 0
```

- [ ] **Step 2: Run tests, confirm failure**

```bash
pytest tests/test_reflection.py -v
```

Expected: `ModuleNotFoundError: No module named 'agents.reflection'`

- [ ] **Step 3: Write `agents/reflection.py`**

```python
import uuid
import re
from typing import Optional
from agents.base import BaseAgent
from core.models import Hypothesis, ResearchPlanConfig, Review


def _parse_verdict(text: str) -> str:
    text_lower = text.lower()
    if "verdict: rejected" in text_lower:
        return "rejected"
    if "verdict: flagged" in text_lower:
        return "flagged"
    if "verdict: passed" in text_lower:
        return "passed"
    return "passed"


def _extract_observation(text: str) -> Optional[str]:
    if "missing piece" in text.lower():
        lines = text.strip().split("\n")
        for line in lines:
            if "missing piece" in line.lower():
                return line.strip()
    return None


class ReflectionAgent:
    def __init__(self, base: BaseAgent):
        self.base = base

    async def run_initial_review(
        self, hypothesis: Hypothesis, config: ResearchPlanConfig
    ) -> Review:
        prompt = self.base.render_prompt(
            "reflection/initial",
            goal=config.goal,
            preferences=config.preferences,
            hypothesis=hypothesis.text,
        )
        response = await self.base.call_claude(
            "You are a scientific peer reviewer.", prompt
        )
        return Review(
            id=str(uuid.uuid4()),
            hypothesis_id=hypothesis.id,
            tier=1,
            critique=response,
            verdict=_parse_verdict(response),
        )

    async def run_full_review(
        self,
        hypothesis: Hypothesis,
        config: ResearchPlanConfig,
        articles_with_reasoning: str,
    ) -> Review:
        prompt = self.base.render_prompt(
            "reflection/full",
            goal=config.goal,
            preferences=config.preferences,
            hypothesis=hypothesis.text,
            articles_with_reasoning=articles_with_reasoning,
        )
        response = await self.base.call_claude(
            "You are a scientific peer reviewer.", prompt, use_strong=True
        )
        return Review(
            id=str(uuid.uuid4()),
            hypothesis_id=hypothesis.id,
            tier=2,
            critique=response,
            verdict=_parse_verdict(response),
        )

    async def run_deep_verification(
        self,
        hypothesis: Hypothesis,
        config: ResearchPlanConfig,
        prior_review: str,
    ) -> Review:
        prompt = self.base.render_prompt(
            "reflection/deep_verification",
            goal=config.goal,
            hypothesis=hypothesis.text,
            prior_review=prior_review,
        )
        response = await self.base.call_claude(
            "You are a scientific peer reviewer.", prompt, use_strong=True
        )
        return Review(
            id=str(uuid.uuid4()),
            hypothesis_id=hypothesis.id,
            tier=3,
            critique=response,
            verdict=_parse_verdict(response),
        )

    async def run_observation_review(
        self,
        hypothesis: Hypothesis,
        config: ResearchPlanConfig,
        article: str,
    ) -> tuple[Review, Optional[str]]:
        prompt = self.base.render_prompt(
            "reflection/observation",
            hypothesis=hypothesis.text,
            article=article,
        )
        response = await self.base.call_claude(
            "You are an expert in scientific hypothesis evaluation.", prompt
        )
        observation = _extract_observation(response)
        verdict = "rejected" if "disproved" in response.lower() else "passed"
        review = Review(
            id=str(uuid.uuid4()),
            hypothesis_id=hypothesis.id,
            tier=4,
            critique=response,
            verdict=verdict,
        )
        return review, observation

    async def run_simulation_review(
        self, hypothesis: Hypothesis, config: ResearchPlanConfig
    ) -> Review:
        prompt = self.base.render_prompt(
            "reflection/simulation",
            goal=config.goal,
            hypothesis=hypothesis.text,
        )
        response = await self.base.call_claude(
            "You are an expert scientific researcher.", prompt, use_strong=True
        )
        return Review(
            id=str(uuid.uuid4()),
            hypothesis_id=hypothesis.id,
            tier=5,
            critique=response,
            verdict=_parse_verdict(response),
        )

    async def run_tournament_review(
        self,
        hypothesis: Hypothesis,
        config: ResearchPlanConfig,
        prior_reviews: str,
        meta_critique: str,
        tournament_history: str,
    ) -> Review:
        prompt = self.base.render_prompt(
            "reflection/tournament",
            goal=config.goal,
            preferences=config.preferences,
            hypothesis=hypothesis.text,
            prior_reviews=prior_reviews,
            meta_critique=meta_critique,
            tournament_history=tournament_history,
        )
        response = await self.base.call_claude(
            "You are a scientific peer reviewer.", prompt
        )
        return Review(
            id=str(uuid.uuid4()),
            hypothesis_id=hypothesis.id,
            tier=6,
            critique=response,
            verdict=_parse_verdict(response),
        )
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_reflection.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add agents/reflection.py tests/test_reflection.py
git commit -m "feat: Reflection agent with 6 review tiers"
```

---

## Task 10: Elo Math + Proximity Agent

**Files:**
- Create: `core/tournament.py`
- Create: `agents/proximity.py`
- Create: `tests/test_ranking_math.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ranking_math.py
import pytest
import numpy as np
from core.tournament import compute_elo_update, select_match_pairs
from core.models import Hypothesis


def make_h(id_, elo):
    return Hypothesis(
        id=id_, run_id="r1", text="t", summary="s",
        generation_method="debate", source="system", elo_rating=elo,
    )


def test_elo_update_winner_gains():
    r_a, r_b = compute_elo_update(1200.0, 1200.0, winner="a", k=32.0)
    assert r_a > 1200.0
    assert r_b < 1200.0


def test_elo_update_symmetric():
    r_a1, r_b1 = compute_elo_update(1200.0, 1200.0, winner="a", k=32.0)
    r_b2, r_a2 = compute_elo_update(1200.0, 1200.0, winner="a", k=32.0)
    assert abs(r_a1 - r_a2) < 0.01


def test_elo_upset_bigger_swing():
    # Low-rated beats high-rated → bigger Elo gain
    r_low, r_high = compute_elo_update(1000.0, 1500.0, winner="a", k=32.0)
    r_even_w, _ = compute_elo_update(1200.0, 1200.0, winner="a", k=32.0)
    assert r_low - 1000.0 > r_even_w - 1200.0


def test_select_pairs_prefers_similar(tmp_path):
    hypotheses = [make_h(f"h{i}", 1200.0) for i in range(4)]
    similarity_pairs = [("h0", "h1", 0.9), ("h0", "h2", 0.3), ("h1", "h3", 0.8)]
    pairs = select_match_pairs(hypotheses, similarity_pairs, n_pairs=2)
    pair_ids = [(a.id, b.id) for a, b in pairs]
    assert ("h0", "h1") in pair_ids or ("h1", "h0") in pair_ids
```

- [ ] **Step 2: Run tests, confirm failure**

```bash
pytest tests/test_ranking_math.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.tournament'`

- [ ] **Step 3: Write `core/tournament.py`**

```python
from __future__ import annotations
from core.models import Hypothesis


def compute_elo_update(
    rating_a: float, rating_b: float, winner: str, k: float = 32.0
) -> tuple[float, float]:
    expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    expected_b = 1.0 - expected_a
    score_a = 1.0 if winner == "a" else 0.0
    score_b = 1.0 - score_a
    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * (score_b - expected_b)
    return new_a, new_b


def select_match_pairs(
    hypotheses: list[Hypothesis],
    similarity_pairs: list[tuple[str, str, float]],
    n_pairs: int,
    multi_turn_threshold: float = 1350.0,
) -> list[tuple[Hypothesis, Hypothesis]]:
    h_map = {h.id: h for h in hypotheses}
    # Sort similarity pairs by score descending (prefer similar hypotheses)
    sorted_pairs = sorted(similarity_pairs, key=lambda x: x[2], reverse=True)
    selected: list[tuple[Hypothesis, Hypothesis]] = []
    used: set[str] = set()
    for h1_id, h2_id, _ in sorted_pairs:
        if len(selected) >= n_pairs:
            break
        if h1_id in used or h2_id in used:
            continue
        if h1_id not in h_map or h2_id not in h_map:
            continue
        selected.append((h_map[h1_id], h_map[h2_id]))
        used.add(h1_id)
        used.add(h2_id)
    # Fill remaining with round-robin if needed
    if len(selected) < n_pairs:
        for i, h1 in enumerate(hypotheses):
            if len(selected) >= n_pairs:
                break
            for h2 in hypotheses[i + 1:]:
                if len(selected) >= n_pairs:
                    break
                if h1.id not in used and h2.id not in used:
                    selected.append((h1, h2))
                    used.add(h1.id)
                    used.add(h2.id)
    return selected
```

- [ ] **Step 4: Write `agents/proximity.py`**

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from core.models import Hypothesis
from core.state import StateStore


class ProximityAgent:
    def __init__(
        self,
        store: StateStore,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        duplicate_threshold: float = 0.92,
    ):
        self.store = store
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.duplicate_threshold = duplicate_threshold

    async def update_graph(self, hypotheses: list[Hypothesis]) -> list[str]:
        if len(hypotheses) < 2:
            return []
        texts = [h.summary for h in hypotheses]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        similarity_matrix = np.dot(embeddings, embeddings.T)
        near_duplicates: list[str] = []
        for i in range(len(hypotheses)):
            for j in range(i + 1, len(hypotheses)):
                score = float(similarity_matrix[i][j])
                if score >= self.similarity_threshold:
                    await self.store.save_proximity(
                        hypotheses[i].id, hypotheses[j].id, score
                    )
                if score >= self.duplicate_threshold:
                    # Flag the lower-Elo one as near-duplicate
                    lower = (
                        hypotheses[i].id
                        if hypotheses[i].elo_rating <= hypotheses[j].elo_rating
                        else hypotheses[j].id
                    )
                    near_duplicates.append(lower)
        return near_duplicates
```

- [ ] **Step 5: Run tests, confirm pass**

```bash
pytest tests/test_ranking_math.py -v
```

Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add core/tournament.py agents/proximity.py tests/test_ranking_math.py
git commit -m "feat: Elo tournament math and proximity agent"
```

---

## Task 11: Ranking Agent

**Files:**
- Create: `agents/ranking.py`
- Create: `tests/test_ranking.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ranking.py
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from agents.ranking import RankingAgent
from agents.base import BaseAgent
from core.models import Hypothesis, ResearchPlanConfig


@pytest.fixture
def config():
    return ResearchPlanConfig(
        run_id="run1", goal="ALS", preferences="Novel",
        attributes=["Novelty"], constraints="testable", safety_approved=True,
    )


def make_h(id_, elo):
    return Hypothesis(
        id=id_, run_id="run1", text=f"text {id_}", summary=f"summary {id_}",
        generation_method="debate", source="system", elo_rating=elo,
    )


@pytest.fixture
def mock_base():
    client = MagicMock()
    base = BaseAgent(client=client, prompts_dir=Path(__file__).parent.parent / "prompts")
    base.call_claude = AsyncMock(return_value="better hypothesis: 1")
    base.run_turn_loop = AsyncMock(return_value=(
        "The debate concludes. better idea: 1",
        ["turn1", "better idea: 1"],
    ))
    return base


@pytest.fixture
def agent(mock_base):
    return RankingAgent(base=mock_base, elo_k=32.0, multi_turn_threshold=1350.0)


async def test_single_turn_match_h1_wins(agent, config):
    h1 = make_h("h1", 1200.0)
    h2 = make_h("h2", 1200.0)
    match = await agent.run_single_turn_match(h1, h2, config, review_1="ok", review_2="ok")
    assert match.winner_id == "h1"
    assert match.elo_after_h1 > 1200.0
    assert match.elo_after_h2 < 1200.0
    assert match.match_type == "single_turn"


async def test_multi_turn_match_h1_wins(agent, config):
    h1 = make_h("h1", 1400.0)
    h2 = make_h("h2", 1400.0)
    match = await agent.run_multi_turn_match(h1, h2, config, review_1="ok", review_2="ok")
    assert match.winner_id == "h1"
    assert match.match_type == "multi_turn"
    assert match.debate_transcript is not None


async def test_selects_multi_turn_for_high_elo(agent, config):
    h1 = make_h("h1", 1400.0)
    h2 = make_h("h2", 1400.0)
    match = await agent.run_match(h1, h2, config, review_1="r1", review_2="r2")
    assert match.match_type == "multi_turn"


async def test_selects_single_turn_for_low_elo(agent, config):
    h1 = make_h("h1", 1200.0)
    h2 = make_h("h2", 1200.0)
    match = await agent.run_match(h1, h2, config, review_1="r1", review_2="r2")
    assert match.match_type == "single_turn"
```

- [ ] **Step 2: Run tests, confirm failure**

```bash
pytest tests/test_ranking.py -v
```

Expected: `ModuleNotFoundError: No module named 'agents.ranking'`

- [ ] **Step 3: Write `agents/ranking.py`**

```python
import uuid
import re
from agents.base import BaseAgent
from core.models import Hypothesis, ResearchPlanConfig, TournamentMatch
from core.tournament import compute_elo_update


def _parse_winner(text: str, h1_id: str, h2_id: str) -> str:
    text_lower = text.lower()
    for pattern in [r"better (?:hypothesis|idea):\s*1", r"better idea:\s*1"]:
        if re.search(pattern, text_lower):
            return h1_id
    for pattern in [r"better (?:hypothesis|idea):\s*2", r"better idea:\s*2"]:
        if re.search(pattern, text_lower):
            return h2_id
    return h1_id  # default to h1 if parsing fails


class RankingAgent:
    def __init__(self, base: BaseAgent, elo_k: float = 32.0, multi_turn_threshold: float = 1350.0):
        self.base = base
        self.elo_k = elo_k
        self.multi_turn_threshold = multi_turn_threshold

    async def run_single_turn_match(
        self,
        h1: Hypothesis,
        h2: Hypothesis,
        config: ResearchPlanConfig,
        review_1: str,
        review_2: str,
        notes: str = "",
    ) -> TournamentMatch:
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
            "You are an expert evaluator.", prompt
        )
        winner_id = _parse_winner(response, h1.id, h2.id)
        winner = "a" if winner_id == h1.id else "b"
        new_r1, new_r2 = compute_elo_update(h1.elo_rating, h2.elo_rating, winner, self.elo_k)
        return TournamentMatch(
            id=str(uuid.uuid4()),
            run_id=config.run_id,
            h1_id=h1.id,
            h2_id=h2.id,
            winner_id=winner_id,
            match_type="single_turn",
            elo_before_h1=h1.elo_rating,
            elo_before_h2=h2.elo_rating,
            elo_after_h1=new_r1,
            elo_after_h2=new_r2,
        )

    async def run_multi_turn_match(
        self,
        h1: Hypothesis,
        h2: Hypothesis,
        config: ResearchPlanConfig,
        review_1: str,
        review_2: str,
        notes: str = "",
    ) -> TournamentMatch:
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
        new_r1, new_r2 = compute_elo_update(h1.elo_rating, h2.elo_rating, winner, self.elo_k)
        return TournamentMatch(
            id=str(uuid.uuid4()),
            run_id=config.run_id,
            h1_id=h1.id,
            h2_id=h2.id,
            winner_id=winner_id,
            match_type="multi_turn",
            debate_transcript="\n---\n".join(transcript),
            elo_before_h1=h1.elo_rating,
            elo_before_h2=h2.elo_rating,
            elo_after_h1=new_r1,
            elo_after_h2=new_r2,
        )

    async def run_match(
        self,
        h1: Hypothesis,
        h2: Hypothesis,
        config: ResearchPlanConfig,
        review_1: str,
        review_2: str,
        notes: str = "",
    ) -> TournamentMatch:
        use_multi = (
            h1.elo_rating >= self.multi_turn_threshold
            and h2.elo_rating >= self.multi_turn_threshold
        )
        if use_multi:
            return await self.run_multi_turn_match(h1, h2, config, review_1, review_2, notes)
        return await self.run_single_turn_match(h1, h2, config, review_1, review_2, notes)
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_ranking.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add agents/ranking.py tests/test_ranking.py
git commit -m "feat: Ranking agent with single-turn and multi-turn debate"
```

---

## Task 12: Evolution Agent

**Files:**
- Create: `agents/evolution.py`
- Create: `tests/test_evolution.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_evolution.py
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from agents.evolution import EvolutionAgent
from agents.base import BaseAgent
from core.models import Hypothesis, ResearchPlanConfig


HYPOTHESIS_OUTPUT = (
    "Introduction: ALS context\n"
    "Hypothesis: New PTM mechanism\n"
    "Summary: Improved PTM hypothesis\n"
    "Category: Neurodegeneration"
)


@pytest.fixture
def config():
    return ResearchPlanConfig(
        run_id="run1", goal="ALS mechanisms", preferences="Novel",
        attributes=["Novelty", "Feasibility"], constraints="testable", safety_approved=True,
    )


@pytest.fixture
def source_hypothesis():
    return Hypothesis(
        id="h1", run_id="run1", text="Original hypothesis text",
        summary="Original", generation_method="debate", source="system",
    )


@pytest.fixture
def mock_base():
    client = MagicMock()
    base = BaseAgent(client=client, prompts_dir=Path(__file__).parent.parent / "prompts")
    base.call_claude = AsyncMock(return_value=HYPOTHESIS_OUTPUT)
    return base


@pytest.fixture
def agent(mock_base):
    return EvolutionAgent(base=mock_base)


async def test_grounding_creates_new_hypothesis(agent, source_hypothesis, config):
    h = await agent.run_grounding(source_hypothesis, config, weaknesses="lacks specificity", articles="article content")
    assert h.id != source_hypothesis.id
    assert h.evolved_from == source_hypothesis.id
    assert h.generation_method == "grounding"


async def test_out_of_box_uses_multiple_hypotheses(agent, config, mock_base):
    hypotheses = [
        Hypothesis(id=f"h{i}", run_id="run1", text=f"text {i}", summary=f"s{i}",
                   generation_method="debate", source="system")
        for i in range(3)
    ]
    h = await agent.run_out_of_box(hypotheses, config)
    assert h.generation_method == "out_of_box"
    # Verify the prompt received multiple hypotheses
    call_args = mock_base.call_claude.call_args
    assert "s0" in call_args[0][1] or "s1" in call_args[0][1]


async def test_combination_uses_multiple(agent, config):
    hypotheses = [
        Hypothesis(id=f"h{i}", run_id="run1", text=f"text {i}", summary=f"s{i}",
                   generation_method="debate", source="system")
        for i in range(2)
    ]
    h = await agent.run_combination(hypotheses, config)
    assert h.generation_method == "combination"
    assert h.evolved_from is None  # combination has no single parent
```

- [ ] **Step 2: Run tests, confirm failure**

```bash
pytest tests/test_evolution.py -v
```

Expected: `ModuleNotFoundError: No module named 'agents.evolution'`

- [ ] **Step 3: Write `agents/evolution.py`**

```python
import uuid
import re
from typing import Optional
from agents.base import BaseAgent
from core.models import Hypothesis, ResearchPlanConfig


def _parse_evolved_hypothesis(
    text: str,
    config: ResearchPlanConfig,
    method: str,
    evolved_from: Optional[str] = None,
) -> Hypothesis:
    def extract(field):
        match = re.search(rf"^{re.escape(field)}:\s*(.+?)(?=\n[A-Z]|\Z)", text, re.MULTILINE | re.DOTALL)
        return match.group(1).strip() if match else ""

    summary = extract("Summary") or text[:120].strip()
    category = extract("Category") or None
    return Hypothesis(
        id=str(uuid.uuid4()),
        run_id=config.run_id,
        text=text,
        summary=summary,
        category=category,
        generation_method=method,
        evolved_from=evolved_from,
        source="system",
    )


class EvolutionAgent:
    def __init__(self, base: BaseAgent):
        self.base = base

    async def run_grounding(
        self,
        hypothesis: Hypothesis,
        config: ResearchPlanConfig,
        weaknesses: str,
        articles: str,
    ) -> Hypothesis:
        prompt = self.base.render_prompt(
            "evolution/grounding",
            goal=config.goal,
            preferences=config.preferences,
            hypothesis=hypothesis.text,
            weaknesses=weaknesses,
            articles_with_reasoning=articles,
        )
        response = await self.base.call_claude("You are an expert scientific researcher.", prompt)
        return _parse_evolved_hypothesis(response, config, "grounding", hypothesis.id)

    async def run_coherence(self, hypothesis: Hypothesis, config: ResearchPlanConfig) -> Hypothesis:
        prompt = self.base.render_prompt(
            "evolution/coherence",
            goal=config.goal,
            preferences=config.preferences,
            hypothesis=hypothesis.text,
        )
        response = await self.base.call_claude("You are an expert scientific researcher.", prompt)
        return _parse_evolved_hypothesis(response, config, "coherence", hypothesis.id)

    async def run_inspiration(self, hypothesis: Hypothesis, config: ResearchPlanConfig) -> Hypothesis:
        prompt = self.base.render_prompt(
            "evolution/inspiration",
            goal=config.goal,
            preferences=config.preferences,
            hypothesis=hypothesis.text,
        )
        response = await self.base.call_claude("You are an expert scientific researcher.", prompt)
        return _parse_evolved_hypothesis(response, config, "inspiration", hypothesis.id)

    async def run_combination(
        self, hypotheses: list[Hypothesis], config: ResearchPlanConfig
    ) -> Hypothesis:
        hypotheses_text = "\n\n---\n\n".join(
            f"Hypothesis {i+1} (Elo: {h.elo_rating:.0f}):\n{h.text}"
            for i, h in enumerate(hypotheses)
        )
        prompt = self.base.render_prompt(
            "evolution/combination",
            goal=config.goal,
            preferences=config.preferences,
            hypotheses=hypotheses_text,
        )
        response = await self.base.call_claude("You are an expert scientific researcher.", prompt)
        return _parse_evolved_hypothesis(response, config, "combination")

    async def run_simplification(self, hypothesis: Hypothesis, config: ResearchPlanConfig) -> Hypothesis:
        prompt = self.base.render_prompt(
            "evolution/simplification",
            goal=config.goal,
            preferences=config.preferences,
            hypothesis=hypothesis.text,
        )
        response = await self.base.call_claude("You are an expert scientific researcher.", prompt)
        return _parse_evolved_hypothesis(response, config, "simplification", hypothesis.id)

    async def run_out_of_box(
        self, hypotheses: list[Hypothesis], config: ResearchPlanConfig
    ) -> Hypothesis:
        hypotheses_text = "\n\n---\n\n".join(
            f"Concept {i+1}: {h.summary}" for i, h in enumerate(hypotheses)
        )
        prompt = self.base.render_prompt(
            "evolution/out_of_box",
            goal=config.goal,
            preferences=config.preferences,
            hypotheses=hypotheses_text,
        )
        response = await self.base.call_claude("You are an expert scientific researcher.", prompt)
        return _parse_evolved_hypothesis(response, config, "out_of_box")
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_evolution.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add agents/evolution.py tests/test_evolution.py
git commit -m "feat: Evolution agent with 6 strategies"
```

---

## Task 13: Meta-review Agent

**Files:**
- Create: `agents/meta_review.py`
- Create: `tests/test_meta_review.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_meta_review.py
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from agents.meta_review import MetaReviewAgent
from agents.base import BaseAgent
from core.models import Hypothesis, ResearchPlanConfig


@pytest.fixture
def config():
    return ResearchPlanConfig(
        run_id="run1", goal="ALS mechanisms", preferences="Novel",
        attributes=["Novelty"], constraints="testable", safety_approved=True,
    )


@pytest.fixture
def hypotheses():
    return [
        Hypothesis(id=f"h{i}", run_id="run1", text=f"text {i}", summary=f"summary {i}",
                   generation_method="debate", source="system", elo_rating=1200.0 + i * 50)
        for i in range(3)
    ]


@pytest.fixture
def mock_base():
    client = MagicMock()
    base = BaseAgent(client=client, prompts_dir=Path(__file__).parent.parent / "prompts")
    base.call_claude = AsyncMock(return_value="I. Core Hypothesis: recurring issue is lack of specificity.\nII. Experimental: model limitations noted.")
    return base


@pytest.fixture
def agent(mock_base):
    return MetaReviewAgent(base=mock_base)


async def test_run_meta_critique(agent, config):
    reviews_text = "Review 1: lacks specificity\nReview 2: needs better controls"
    critique = await agent.run_meta_critique(config, reviews_text)
    assert len(critique) > 0
    assert isinstance(critique, str)


async def test_run_research_overview(agent, config, hypotheses, mock_base):
    mock_base.call_claude = AsyncMock(return_value="[Main Research Directions]\nDirection: PTM mechanisms\nRationale: Novel")
    overview = await agent.run_research_overview(config, hypotheses)
    assert "[Main Research Directions]" in overview or "PTM" in overview or len(overview) > 0


async def test_run_research_contacts(agent, config, mock_base):
    mock_base.call_claude = AsyncMock(return_value="Research Direction: ALS\n[Smith Lab]: Studies TDP-43, expertise in NPC")
    cited = "TDP-43 study: http://example.com"
    contacts = await agent.run_research_contacts(config, cited)
    assert len(contacts) > 0
```

- [ ] **Step 2: Run tests, confirm failure**

```bash
pytest tests/test_meta_review.py -v
```

Expected: `ModuleNotFoundError: No module named 'agents.meta_review'`

- [ ] **Step 3: Write `agents/meta_review.py`**

```python
from agents.base import BaseAgent
from core.models import Hypothesis, ResearchPlanConfig


class MetaReviewAgent:
    def __init__(self, base: BaseAgent):
        self.base = base

    async def run_meta_critique(
        self,
        config: ResearchPlanConfig,
        reviews_text: str,
        instructions: str = "",
    ) -> str:
        prompt = self.base.render_prompt(
            "meta_review/meta_critique",
            goal=config.goal,
            preferences=config.preferences,
            instructions=instructions,
            reviews=reviews_text,
        )
        return await self.base.call_claude(
            "You are an expert in scientific research and meta-analysis.", prompt
        )

    async def run_research_overview(
        self,
        config: ResearchPlanConfig,
        hypotheses: list[Hypothesis],
    ) -> str:
        top_hypotheses = "\n\n---\n\n".join(
            f"Hypothesis (Elo: {h.elo_rating:.0f}):\n{h.text}"
            for h in sorted(hypotheses, key=lambda x: x.elo_rating, reverse=True)[:10]
        )
        prompt = self.base.render_prompt(
            "meta_review/research_overview",
            goal=config.goal,
            top_hypotheses=top_hypotheses,
        )
        return await self.base.call_claude(
            "You are an expert scientific researcher.", prompt
        )

    async def run_research_contacts(
        self,
        config: ResearchPlanConfig,
        cited_literature: str,
    ) -> str:
        prompt = self.base.render_prompt(
            "meta_review/research_contacts",
            goal=config.goal,
            cited_literature=cited_literature,
        )
        return await self.base.call_claude(
            "You are an expert scientific researcher.", prompt
        )
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest tests/test_meta_review.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add agents/meta_review.py tests/test_meta_review.py
git commit -m "feat: Meta-review agent with critique, overview, contacts"
```

---

## Task 14: Full Test Suite Pass

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Create shared conftest**

```python
# tests/conftest.py
import pytest
import asyncio

# pytest-asyncio auto mode is set in pyproject.toml
```

- [ ] **Step 2: Run the complete test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: All tests pass. Output should include:
```
tests/test_models.py::... PASSED
tests/test_state.py::... PASSED
tests/test_claude.py::... PASSED
tests/test_base_agent.py::... PASSED
tests/test_generation.py::... PASSED
tests/test_reflection.py::... PASSED
tests/test_ranking_math.py::... PASSED
tests/test_ranking.py::... PASSED
tests/test_evolution.py::... PASSED
tests/test_meta_review.py::... PASSED
```

If any test fails, fix before proceeding.

- [ ] **Step 3: Final commit for this plan**

```bash
git add tests/conftest.py
git commit -m "feat: complete Foundation + Agents implementation

All 6 agents implemented and tested:
- Generation (literature, debate, assumptions, expansion)
- Reflection (6 tiers with sequential pipeline)
- Ranking (single-turn + multi-turn debate, Elo math)
- Proximity (sentence-transformers similarity graph)
- Evolution (6 strategies, creates new hypotheses)
- Meta-review (critique, overview, contacts)

17 prompt templates faithful to paper appendix (arXiv:2502.18864).
SQLite state layer with full restart capability."
```

---

## What's Next

**Plan 2: Supervisor + Gradio UI**
- Async task queue (`asyncio.PriorityQueue`) with N worker coroutines
- Dynamic agent weighting from `SystemStats`
- Config parser (LLM-parses research goal → `ResearchPlanConfig`)
- Supervisor control loop with SQLite checkpointing
- Gradio 5 UI with live feed, hypothesis explorer, expert input panel
- Expert-in-the-loop: hypothesis injection, manual review, chat
