# AI Co-Scientist — Design Specification

**Date:** 2026-05-28
**Paper:** "Towards an AI co-scientist" (arXiv:2502.18864)
**Stack:** Python 3.12, asyncio, Anthropic SDK (Claude), SQLite/aiosqlite, Gradio 5, Tavily web search, sentence-transformers

---

## 1. Goals & Scope

Implement a faithful open-source reproduction of the Google AI co-scientist system using Claude as the LLM backend. The system accepts a natural-language research goal and iteratively generates, reviews, ranks, and evolves research hypotheses through a multi-agent self-improving loop.

**Differentiators vs. existing open-source implementations (llnl, jataware, conradry):**
- True async Supervisor with dynamic agent weighting (no existing repo has this)
- Proximity agent integrated into Ranking matchmaking (no existing repo does this)
- Expert-in-the-loop mid-run: hypothesis injection, manual reviews, chat (no existing repo)
- All 6 Reflection tiers faithfully implemented including deep verification and observation review
- All 6 Evolution strategies as specified in the paper
- Prompt-template-driven architecture matching the paper's exact prompt structures
- Claude-native with Anthropic SDK prompt caching on long hypothesis contexts
- SQLite persistence with full restart capability

**Out of scope (v1):**
- AlphaFold / specialized AI model integration
- Private publication repository indexing
- NIH Specific Aims Page constrained decoding
- Multi-user / hosted service

---

## 2. Project Layout

```
coscientist/
├── core/
│   ├── supervisor.py        # Async task queue + dynamic agent weighting
│   ├── state.py             # SQLite-backed persistent state (aiosqlite)
│   ├── hypothesis.py        # Hypothesis dataclass + Elo model
│   ├── tournament.py        # Elo math + proximity-biased matchmaking
│   └── config.py            # ResearchPlanConfig dataclass
├── agents/
│   ├── base.py              # BaseAgent: prompt template rendering + Claude call
│   ├── generation.py        # 4 strategies: literature, debate, assumptions, expansion
│   ├── reflection.py        # 6 tiers: initial, full, deep_verification, observation, simulation, tournament
│   ├── ranking.py           # Elo tournament orchestration (single-turn + multi-turn loops)
│   ├── proximity.py         # Similarity graph + matchmaking bias scores
│   ├── evolution.py         # 6 strategies: grounding, coherence, inspiration, combination, simplification, out_of_box
│   └── meta_review.py       # Meta-critique + research overview + research contacts
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
├── tools/
│   ├── search.py            # Tavily web search wrapper
│   ├── pubmed.py            # PubMed API (biomedical demo)
│   └── claude.py            # Anthropic SDK wrapper with prompt caching
├── ui/
│   └── app.py               # Gradio 5 interface
├── config.yaml              # Default system settings
└── pyproject.toml
```

**Key design principle:** Every agent call fills a fixed set of template variables from current system state. Prompt files in `prompts/` are the canonical source of truth for agent behavior. No ad-hoc f-string formatting in agent code.

---

## 3. Research Plan Configuration

The first step before the Supervisor starts. The system LLM-parses the raw research goal into a structured `ResearchPlanConfig`:

```python
@dataclass
class ResearchPlanConfig:
    goal: str                    # Original research goal (verbatim)
    preferences: str             # What to focus on (e.g. "novel hypothesis with detailed mechanism")
    attributes: list[str]        # Evaluation dimensions (e.g. ["Novelty", "Feasibility"])
    constraints: str             # Hard requirements (e.g. "should be correct, should be novel")
    idea_attributes: str         # Joined attributes string for prompts (e.g. "novel and feasible")
    safety_approved: bool        # Result of initial safety review
```

The config fields map to prompt template variables `{preferences}`, `{idea_attributes}`, `{constraints}` used in every agent prompt.

**The `{instructions}` variable:** Several prompts (Figure A.1, A.2, A.8) include an `{instructions}` field that the paper never explicitly defines. Implementation interpretation: the Supervisor populates this per-task with context-specific guidance assembled from: (a) any direct scientist chat instructions received since the last task of this type, (b) a note if the research expansion strategy is active ("prioritize underexplored areas"), (c) empty string by default. This is the mechanism by which the chat expert-in-the-loop feature influences generation behavior.

**Safety gate:** The research goal undergoes automated safety review before any agent work begins. Goals deemed potentially unsafe are rejected with an explanation.

---

## 4. Supervisor: Async Task Queue with Dynamic Weighting

The Supervisor is the only component that is not a specialized agent — it is the orchestrator.

### 4.1 Architecture

```python
class Supervisor:
    task_queue: asyncio.PriorityQueue  # (priority, AgentTask)
    worker_pool: list[asyncio.Task]    # N concurrent workers pulling from queue
    state: SystemState                 # Current snapshot from SQLite
    config: ResearchPlanConfig
```

Workers pull tasks from the priority queue and execute them. The Supervisor runs a separate control loop that:
1. Periodically computes summary statistics
2. Updates agent weights based on statistics
3. Enqueues the next batch of tasks with correct priorities
4. Writes a checkpoint to SQLite
5. Checks terminal conditions

### 4.2 Summary Statistics (computed each tick)

```python
@dataclass
class SystemStats:
    n_hypotheses: int
    n_pending_review: int          # Hypotheses not yet through Reflection tier 1
    n_reviewed: int
    tournament_progress: float     # Matches completed / matches planned
    elo_variance: float            # Spread of Elo ratings (diversity proxy)
    avg_proximity: float           # Mean pairwise similarity (cluster density proxy)
    generation_effectiveness: float  # Avg Elo of Generation-produced vs Evolution-produced
    evolution_effectiveness: float
    last_meta_review_age: int      # Ticks since last meta-review run
```

### 4.3 Dynamic Agent Weights

```python
def compute_weights(stats: SystemStats) -> dict[AgentType, float]:
    weights = {
        AgentType.GENERATION: 1.0,
        AgentType.REFLECTION: 1.0,
        AgentType.RANKING: 1.0,
        AgentType.EVOLUTION: 1.0,
        AgentType.PROXIMITY: 0.5,   # runs async, lower base weight
        AgentType.META_REVIEW: 0.3, # periodic, not every tick
    }
    # Boost Reflection if review backlog is high
    if stats.n_pending_review > stats.n_hypotheses * 0.3:
        weights[AgentType.REFLECTION] *= 2.0
    # Boost Evolution if Elo variance is low (pool is converging, need diversity)
    if stats.elo_variance < ELO_VARIANCE_THRESHOLD:
        weights[AgentType.EVOLUTION] *= 1.8
    # Boost Generation if pool is small
    if stats.n_hypotheses < MIN_HYPOTHESIS_COUNT:
        weights[AgentType.GENERATION] *= 2.5
    # Boost Meta-review if it's been many ticks since last run
    if stats.last_meta_review_age > META_REVIEW_INTERVAL:
        weights[AgentType.META_REVIEW] = 2.0
    return weights
```

The Supervisor samples the next agent type from this distribution and enqueues the appropriate task.

### 4.4 Terminal Conditions

The run ends when any of:
- Wall-clock time budget exhausted (configurable, default 30 min)
- Elo variance below convergence threshold for N consecutive ticks
- User explicitly stops via UI

### 4.5 Persistence / Restart

Every `CHECKPOINT_INTERVAL` ticks (default: every 5 tasks completed), the Supervisor writes:
- Full hypothesis pool with Elo ratings
- Tournament match history
- Meta-review history
- Current agent weights and stats

On restart, the Supervisor reads the latest checkpoint and resumes the queue from where it left off.

---

## 5. Specialized Agents

All agents inherit from `BaseAgent`:

```python
class BaseAgent:
    def render_prompt(self, template_name: str, **variables) -> str:
        # Load from prompts/ directory, fill template variables
    
    async def call_claude(self, prompt: str, *, use_caching: bool = True) -> str:
        # Anthropic SDK call with prompt caching on system prompt
    
    async def run_turn_loop(
        self,
        template_name: str,
        variables: dict,
        transcript: list[str],
        termination_signal: str,
        max_turns: int = 10,
    ) -> tuple[str, list[str]]:
        # Multi-turn self-play loop: appends responses to transcript,
        # re-renders prompt with updated {transcript}, calls Claude,
        # stops when termination_signal found in response.
        # Returns (final_output, full_transcript).
```

### 5.1 Generation Agent

Four strategies selected by the Supervisor (weighted by effectiveness stats):

**Literature exploration** — `prompts/generation/literature.txt`
Template variables: `{goal}`, `{preferences}`, `{instructions}`, `{source_hypothesis}` (optional), `{articles_with_reasoning}`

Procedure:
1. Tavily search for 3–5 queries derived from the research goal
2. Retrieve and summarize relevant articles (chronologically ordered, most recent first)
3. Fill `{articles_with_reasoning}` with summaries + reasoning
4. Single LLM call → structured hypothesis document

**Simulated scientific debate** — `prompts/generation/debate.txt`
Template variables: `{goal}`, `{preferences}`, `{idea_attributes}`, `{instructions}`, `{reviews_overview}`, `{transcript}`

Procedure:
1. Initial turn: model proposes 3 distinct hypotheses
2. Subsequent turns: model critiques, poses clarifying questions, proposes improvements, refines
3. Termination: model outputs `HYPOTHESIS` (all caps) followed by finalized hypothesis
4. `{reviews_overview}` is populated with the current meta-critique (selective propagation of Meta-review feedback)

**Iterative assumptions** — `prompts/generation/assumptions.txt`
Template variables: `{goal}`, `{preferences}`, `{instructions}`
*Note: No prompt shown in paper appendix — template is inferred from the paper's description.*

Procedure: Model iteratively identifies testable intermediate assumptions via conditional reasoning hops ("if A is true, then B could follow..."), identifies sub-assumptions, aggregates into a complete hypothesis. Implemented as a multi-turn loop (like debate) where each turn deepens the assumption chain.

**Research expansion** — `prompts/generation/expansion.txt`
Template variables: `{goal}`, `{preferences}`, `{instructions}`, `{research_overview}`, `{existing_hypotheses_summary}`
*Note: No prompt shown in paper appendix — template is inferred from the paper's description.*

Procedure: Reads the research overview from Meta-review agent + summary of existing hypotheses to identify underexplored areas, generates hypotheses in those gaps. Only viable once at least one meta-review has run.

**Output format** (all strategies produce this structure):
```
Introduction: [domain context]
Recent Findings: [relevant literature]
Hypothesis: [the core claim]
Rationale and Specificity: [why this is novel and specific]
Experimental Design: [how to test it]
Summary: [one-sentence layman description]
Category: [domain category tag]
```

The `Summary` and `Category` fields allow scientists to quickly grasp core ideas (explicitly noted in paper §3.3.1).

### 5.2 Reflection Agent

The paper lists six review types. Tiers 1→2 are explicitly gated ("if a hypothesis passes the initial review, the Reflection agent performs a full review"). Tiers 3–6 are listed as review types the agent *employs* — the paper does not state they all run sequentially on every hypothesis. **Implementation choice:** we run them sequentially with early exit, as this is the most faithful interpretation that ensures comprehensive review of surviving hypotheses. Tiers 3–6 may be selectively skipped in future optimizations.

**Tier 1 — Initial review** (`prompts/reflection/initial.txt`)
- No web search
- Assesses: correctness, quality, novelty, preliminary safety
- Fast discard: hypothesis is marked `REJECTED` if clearly flawed, non-novel, or unsafe
- Surviving hypotheses proceed to Tier 2

**Tier 2 — Full review** (`prompts/reflection/full.txt`)
- Tavily search for relevant literature
- Template: `{goal}`, `{preferences}`, `{hypothesis}`, `{articles_with_reasoning}`
- Output structure:
  - Related Articles: [numbered references]
  - Assumptions of the Idea: [list each assumption]
  - Reasoning about assumptions: [evaluate plausibility of each]
  - Novelty: [aspects already explored / novel aspects]
  - Overall critique: [bulleted list]

**Tier 3 — Deep verification** (`prompts/reflection/deep_verification.txt`)
- Decomposes hypothesis into constituent assumptions
- Each assumption broken into fundamental sub-assumptions, decontextualized, evaluated independently
- Also includes probing questions variant: agent generates a hard probing question about a fundamental assumption, answers it, reasons about whether the answer invalidates the hypothesis
- Critical: a non-fundamental error does NOT discard the hypothesis — the agent assesses whether the incorrect assumption is fundamental and notes this in the review

**Tier 4 — Observation review** (`prompts/reflection/observation.txt`)
Template variables: `{goal}`, `{hypothesis}`, `{article}`

Article selection: reuse the highest-relevance article retrieved during Tier 2's web search (stored in `reviews.web_citations`). If no Tier 2 citations exist, run a targeted Tavily search first.

Five-step protocol (from Figure A.3 of paper):
1. Observation extraction: list relevant observations from retrieved article
2. Per-observation causal analysis: "would we see this observation if the hypothesis were true?"
   - State if cause is already established
   - Assess if hypothesis could be a causal factor
   - Explain if novel explanation; if not, state "not a missing piece"
3. Summary causal analysis: "would we see some observations if hypothesis were true?"
4. Disproof analysis: "does some observation disprove the hypothesis?"
5. Conclusion — one of five classifications: `already explained` | `other explanations more likely` | `missing piece` | `neutral` | `disproved`

Positive observations (classified as `missing piece`) are appended directly to the hypothesis's `annotations` field.

**Tier 5 — Simulation review** (`prompts/reflection/simulation.txt`)
- Step-wise simulation of the proposed mechanism
- Agent works through the mechanism step by step, identifies potential failure points and failure scenarios
- Relies on the LLM's internal world model to simulate scientific phenomena

**Tier 6 — Tournament/recurrent review** (`prompts/reflection/tournament.txt`)
- Reads the current meta-critique and tournament history
- Adapts review to known recurring issues identified by Meta-review
- Only runs on hypotheses that have already been in at least one tournament match

### 5.3 Ranking Agent + Elo Tournament

**Elo mechanics:**
- All hypotheses (including expert-submitted) start at rating 1200
- K-factor: 32 (standard)
- Expected score formula: `E_A = 1 / (1 + 10^((R_B - R_A) / 400))`

**Match type selection** (implementation choice — paper specifies the split but not the threshold):
- Hypotheses with Elo ≥ 1350: multi-turn debate match
- Hypotheses with Elo < 1350: single-turn comparison
- The 1350 cutoff is a configurable default, not a paper-specified value

**Matchmaking** (first open-source implementation to do this faithfully):
1. Proximity agent provides a similarity matrix over all hypothesis pairs
2. Ranking agent preferentially selects pairs with high similarity scores (similar hypotheses compete)
3. Newer hypotheses and top-ranked hypotheses get priority slots in the queue
4. Round-robin is used as fallback when similarity-based pairing exhausts candidates

**Single-turn comparison** — `prompts/ranking/single_turn.txt` (Figure A.4)
Template: `{goal}`, `{preferences}`, `{idea_attributes}`, `{notes}`, `{hypothesis 1}`, `{hypothesis 2}`, `{review 1}`, `{review 2}`
Output: concludes with `"better hypothesis: <1 or 2>"`
Note: prompt explicitly instructs to disregard numerical scores across reviews.

**Multi-turn debate** — `prompts/ranking/multi_turn_debate.txt` (Figure A.5)
Template: same as single-turn plus `{transcript}`
Turn 1: summary of both hypotheses + initial reviews
Subsequent turns: clarifying questions, critique on correctness/validity/utility/specificity/novelty/desirability, identify weaknesses
Termination: `"better idea: "` followed by `1` or `2` (typically 3–5 turns, max 10)

### 5.4 Proximity Agent

Runs asynchronously alongside the main loop.

**Similarity computation:**
- Embeds each hypothesis's `Summary` field using `sentence-transformers` (`all-MiniLM-L6-v2`)
- Computes pairwise cosine similarity matrix
- Builds adjacency graph: edge exists if similarity > 0.5, weighted by similarity score
- Graph persists to `proximity_graph` SQLite table

**Two consumers:**
1. **Ranking agent** reads top-N similar pairs per hypothesis for matchmaking bias
2. **Supervisor** reads aggregate metrics (mean similarity, cluster count) for diversity stats

**Near-duplicate flagging:** pairs with similarity > 0.92 are flagged to the Supervisor for potential pruning from the pool.

### 5.5 Evolution Agent

Six strategies. All create a **new** hypothesis — they never modify existing ones. Each new hypothesis enters the tournament pool at Elo 1200 and must compete alongside the originals.

**Grounding** — `prompts/evolution/grounding.txt`
Identifies weaknesses in a top-ranked hypothesis, runs Tavily searches targeting those weaknesses, suggests improvements, produces new hypothesis with gaps filled.

**Coherence/Feasibility** — `prompts/evolution/coherence.txt` (Figure A.6)
Procedure:
1. Introductory overview of scientific domain
2. Synopsis of recent pertinent research
3. Reasoned argument for how current tech enables realization
4. Core contribution: technologically viable alternative, emphasizing simplicity and practicality

**Inspiration** — `prompts/evolution/inspiration.txt`
Inspired by a single top-ranked hypothesis as seed → divergent generation building on the same core insight.

**Combination** — `prompts/evolution/combination.txt`
Combines best aspects of 2–3 top-ranked hypotheses into a new synthesized proposal.

**Simplification** — `prompts/evolution/simplification.txt`
Strips a top-ranked hypothesis to its minimal testable core — removes secondary claims, focuses on the key verifiable claim.

**Out-of-box** — `prompts/evolution/out_of_box.txt` (Figure A.7)
Template: `{goal}`, `{preferences}`, `{hypotheses}` (plural — multiple hypotheses as analogical inspiration)
Instructions: "Inspiration may be drawn from the following concepts (utilize analogy and inspiration, NOT direct replication)"
Core contribution: develop a hypothesis leveraging analogous principles — explicitly not a combination or aggregation.

### 5.6 Meta-review Agent

Runs periodically (triggered by Supervisor when `last_meta_review_age > META_REVIEW_INTERVAL`).

Produces three outputs per run:

**Meta-critique** — `prompts/meta_review/meta_critique.txt` (Figure A.8)
Input: all reviews accumulated since the last meta-review
Instructions:
- Generate a structured meta-analysis (not evaluation of individual proposals)
- Focus on recurring critique points and common issues
- Provide actionable insights for future proposals
Output structure (per Figures A.18–A.19):
```
I. Core Hypothesis and Mechanism
   - Primary Driver vs. Consequence
   - Specificity
   - [domain-specific recurring issues]
II. Experimental Design and Feasibility
   - Model System Limitations
   - Technical Challenges
   - Quantitative Rigor
III. Novelty and Impact
IV. Assumptions and Validation
V. General Advice Based on Common Critiques
```
This output is stored as `current_meta_critique` in system state. It is passed as `{reviews_overview}` to the Generation debate strategy — this is the feedback-without-fine-tuning mechanism. It is NOT passed to literature exploration (selective propagation).

**Research overview** — `prompts/meta_review/research_overview.txt`
Synthesizes top-ranked hypotheses into a structured research roadmap:
```
[Main Research Directions]
  Direction: [name + description]
  Rationale: [why this area matters]
  Recent Findings: [current state of knowledge]
  Areas of Research:
    Sub-area:
      Why Research? [motivation]
      What to Research? [specific questions]
      Example Idea: [concrete hypothesis]
```
This overview is also passed as `{research_overview}` to the Generation expansion strategy.

**Research contacts** — `prompts/meta_review/research_contacts.txt`
Uses prior literature (from Full reviews) to identify qualified domain experts:
```
Research Direction: [title]
[Researcher name(s)]: [why relevant, what experiments they can contribute to]
```

---

## 6. SQLite Schema

Six tables. All writes are idempotent (upsert pattern).

```sql
-- Parsed research plan configuration
CREATE TABLE configs (
    run_id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    preferences TEXT NOT NULL,
    attributes TEXT NOT NULL,       -- JSON array
    constraints TEXT NOT NULL,
    safety_approved BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Hypothesis pool
CREATE TABLE hypotheses (
    id TEXT PRIMARY KEY,            -- UUID
    run_id TEXT NOT NULL,
    text TEXT NOT NULL,             -- Full structured document
    summary TEXT NOT NULL,          -- One-sentence layman description
    category TEXT,
    generation_method TEXT NOT NULL, -- literature|debate|assumptions|expansion|grounding|...
    evolved_from TEXT,              -- parent hypothesis id (NULL if original)
    source TEXT NOT NULL,           -- system|expert
    elo_rating REAL DEFAULT 1200.0,
    annotations TEXT,               -- JSON: positive observations appended here
    status TEXT DEFAULT 'active',   -- active|rejected
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES configs(run_id)
);

-- Reviews (one row per tier per hypothesis)
CREATE TABLE reviews (
    id TEXT PRIMARY KEY,
    hypothesis_id TEXT NOT NULL,
    tier INTEGER NOT NULL,          -- 1-6
    verdict TEXT,                   -- passed|rejected|flagged
    critique TEXT NOT NULL,         -- Full review text
    web_citations TEXT,             -- JSON array of {title, url, summary}
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
);

-- Tournament match history
CREATE TABLE tournament_matches (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    h1_id TEXT NOT NULL,
    h2_id TEXT NOT NULL,
    winner_id TEXT NOT NULL,
    match_type TEXT NOT NULL,       -- single_turn|multi_turn
    debate_transcript TEXT,         -- Full turn-by-turn transcript
    elo_before_h1 REAL,
    elo_before_h2 REAL,
    elo_after_h1 REAL,
    elo_after_h2 REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Proximity graph
CREATE TABLE proximity_graph (
    h1_id TEXT NOT NULL,
    h2_id TEXT NOT NULL,
    similarity_score REAL NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (h1_id, h2_id)
);

-- Meta-review history
CREATE TABLE meta_reviews (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    meta_critique TEXT NOT NULL,
    research_overview TEXT,
    research_contacts TEXT,
    tick INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 7. Expert-in-the-Loop

Scientists interact via the Gradio UI in four ways:

1. **Refine research goal**: submit updated goal text mid-run → Supervisor re-parses config and adjusts weights (new preferences propagate to next meta-review tick)
2. **Submit manual review**: attach a free-text review to any hypothesis → stored in `reviews` table as `tier=0` (expert review), used by Reflection tier 6 and Ranking
3. **Inject hypothesis**: submit own hypothesis text → enters pool at Elo 1200 with `source=expert`, participates in tournament, can be selected by Evolution for Combination/Inspiration
4. **Chat / direct instruction**: text box sends natural-language instructions to the Supervisor which adjusts the task queue (e.g. "focus on mechanism X", "explore more out-of-box ideas")

---

## 8. Gradio UI

Single-page app with five panels:

```
┌─────────────────────────────────────────────────────────┐
│ Research Goal  [text input]   [Start] [Stop] [Export]   │
│ Settings: max_time, n_workers, model selection          │
├──────────────┬──────────────────────────────────────────┤
│ Live Feed    │ Hypothesis Explorer                       │
│              │                                           │
│ Real-time    │ Sorted by Elo ▼                           │
│ task log:    │ ┌──────────────────────────────────────┐  │
│              │ │ #1  Elo:1487  [Hypothesis summary]   │  │
│ Gen debate   │ │     [Reviews] [Tournament history]   │  │
│ Reflection   │ ├──────────────────────────────────────┤  │
│ Match: H3>H7 │ │ #2  Elo:1421  ...                    │  │
│ Evolution    │ └──────────────────────────────────────┘  │
├──────────────┼──────────────────────────────────────────┤
│ Research Overview (from Meta-review)                     │
├──────────────┬──────────────────────────────────────────┤
│ Expert Input │ Chat                                      │
│ [Submit      │ [type instruction to Supervisor]          │
│  hypothesis] │                                           │
│ [Submit      │                                           │
│  review]     │                                           │
└──────────────┴──────────────────────────────────────────┘
```

Updates stream via `gr.update()` polling every 2 seconds from the SQLite state — no WebSocket complexity.

---

## 9. Tool Use

**Tavily search** (`tools/search.py`): primary web search for Generation (literature), Reflection (full review, grounding), Evolution (grounding strategy). Returns `{title, url, content_summary}` objects formatted as `{articles_with_reasoning}` template variable.

**PubMed** (`tools/pubmed.py`): biomedical demo — used in place of or alongside Tavily for the biomedical research goal examples. Returns structured article metadata.

**Claude wrapper** (`tools/claude.py`): Anthropic SDK calls with:
- Prompt caching on the system prompt (large hypothesis contexts reused across tiers)
- `claude-opus-4-7` for deep verification and multi-turn debates (highest reasoning requirement)
- `claude-sonnet-4-6` for initial review, single-turn comparison, proximity embedding (cost efficiency)
- Configurable per agent strategy

---

## 10. Configuration

`config.yaml`:
```yaml
anthropic:
  model_strong: claude-opus-4-7       # Used for deep verification, multi-turn debates
  model_fast: claude-sonnet-4-6       # Used for initial review, single-turn comparison

supervisor:
  n_workers: 4                        # Concurrent worker coroutines
  tick_interval_seconds: 10
  checkpoint_interval: 5              # Checkpoint every N completed tasks
  max_time_minutes: 30
  min_hypothesis_count: 8             # Below this, boost Generation weight
  elo_variance_threshold: 5000        # Below this, boost Evolution weight
  meta_review_interval: 20            # Ticks between meta-review runs

tournament:
  elo_initial: 1200
  elo_k_factor: 32
  multi_turn_threshold: 1350          # Elo >= this → multi-turn debate
  max_debate_turns: 10

proximity:
  model: all-MiniLM-L6-v2
  similarity_threshold: 0.5           # Min similarity for graph edge
  duplicate_threshold: 0.92           # Above this → flag as near-duplicate

tools:
  search_provider: tavily             # tavily | pubmed | both
  max_search_results: 5
```

---

## 11. Safety

On startup:
1. Research goal undergoes automated safety review (LLM judge) before any agents run
2. Each generated hypothesis undergoes safety check in Reflection Tier 1 — unsafe hypotheses are excluded from the pool and not shown to the user
3. All system activities logged to `coscientist.log` for audit

---

## 12. Testing Strategy

- **Unit**: each agent strategy in isolation with a mock Claude client
- **Integration**: full 3-minute run on a simple research goal, verify hypothesis pool grows, Elo diverges, meta-review produces output
- **Prompt tests**: snapshot tests for each prompt template with known inputs → assert key phrases in output
- **Restart test**: run for N tasks, kill process, restart, verify pool state matches pre-kill checkpoint
