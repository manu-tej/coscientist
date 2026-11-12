# Evidence spine: grounding and verification for the co-scientist core

Design approved 2026-08-09. Sub-project 1 of 3.

**Revised 2026-08-09** after an adversarial review of the v1 plan (68 agents, 52
verified findings — `docs/superpowers/reviews/2026-08-09-evidence-spine-adversarial-review.md`).
Changes: the cost model was wrong at its baseline (6 tiers, not 1); the error
table conflated source failure with empty results; `contradicted` now quarantines
rather than rejects, and requires abstract-backed evidence. The component
architecture was reviewed as sound and is unchanged.

## Problem

The core agents have no working literature access. `coscientist.py:131` builds a
`SearchTool` (Tavily), `TAVILY_API_KEY` is absent from `.env`, and
`SearchTool.search_and_format` returns the string `"No relevant articles found."`
when unkeyed. The run proceeds as though that were a real search result.

Consequently every hypothesis the system has produced — including the live runs
of 2026-08-09, which named specific phosphorylation sites and promoter-binding
events — was pure model recall, checked against nothing.

Meanwhile `tools/pubmed.py` is a complete, tested PubMed E-utilities client with
the same `search_and_format` interface. Nothing imports it. It is orphaned.

The reflection agent, whose entire job is judging correctness, has no tool access
at all and judges from model priors.

## Scope

Three sub-projects. This spec covers only the first.

| # | Sub-project | Delivers |
|---|---|---|
| **1** | **Evidence spine** (this spec) | real citations; reviewers can check facts |
| 2 | Structured entity backends (ChEMBL, Open Targets, ClinicalTrials) | drug/gene/trial ground truth |
| 3 | Evidence-derived confidence and calibration | quantitative self-assessment |

Sub-projects 2 and 3 are consumers of what this spec builds. Do not start them
until this one ships.

## Constraint that shapes everything: the sources are free

PubMed E-utilities, ChEMBL, Open Targets, and ClinicalTrials.gov all expose
public APIs with no authentication. PubMed rate-limits unkeyed clients to 3
requests/second. Therefore the entire rigor upgrade requires no API keys, no
signup, and no spend, and remains reproducible in CI.

## Approach

Ground during generation; verify during reflection. Each capability goes to the
pipeline stage that already exists for it, rather than adding stages.

Rejected alternatives:

- *Retrieval only* — injects real abstracts but never checks the result, so a
  model can still miscite a paper it was handed.
- *Post-hoc verification only* — catches miscitation but leaves generation
  ungrounded, so there is more to catch.

## Components

### 1. `tools/evidence.py` — one interface for every source

```python
class EvidenceSource(Protocol):
    name: str
    async def search_outcome(self, query: str, *, limit: int = 5) -> SearchOutcome: ...

@dataclass(frozen=True)
class EvidenceRecord:
    source: str        # "pubmed" | "web" | "fixture"
    ref: str           # PMID / URL — the citable identifier
    title: str
    snippet: str       # the abstract when one was retrieved
    url: str
    year: int | None
    has_abstract: bool # gates whether `contradicted` may be asserted

@dataclass(frozen=True)
class SearchOutcome:
    records: list[EvidenceRecord]
    failed_sources: list[str]   # so an outage is never read as absence of literature
```

`SearchOutcome` rather than a bare list is the correction the adversarial review
forced: v1's sources caught every transport error and returned `[]`, which made
"NCBI is down" indistinguishable from "no such literature" and turned outages
into recorded scientific findings.

This is the seam that makes sub-project 2 additive: new databases are new
implementations and nothing else changes. `tools/pubmed.py` is adapted to it as
a thin shim over existing tested code.

### 2. `tools/verify.py` — the verification primitive

```python
@dataclass(frozen=True)
class ClaimVerdict:
    claim: str
    status: str              # supported | unsupported | contradicted | unchecked
    records: list[EvidenceRecord]
    rationale: str

async def extract_claims(text, backend, *, max_claims=4) -> list[str]
async def verify_claim(claim, sources, backend) -> ClaimVerdict
```

Two functions rather than a class: both are pure given their inputs and test
without fixtures or mocking ceremony.

`unchecked` is a first-class status. "We could not check" and "we checked and
found nothing" are different facts, and conflating them is how a system starts
overstating itself.

### 3. Persistence — provenance as a queryable row

New `claim_verdicts` table in `core/state.py`, keyed by `hypothesis_id`, holding
the claim, status, rationale, and the record refs consulted. Provenance must be
queryable rather than buried in prose, so the run explorer and the bench harness
can both consume it. This mirrors `company/evidence.py`, which already treats
provenance as a product for the portfolio layer.

### 4. `LiteratureTool` — generation grounding

A composite over N sources that formats retrieved records (with PMIDs) into
generation prompts. Replaces `SearchTool` at `coscientist.py:131`; web search is
demoted to one source behind the interface.

It must log a warning when every source returns empty. The current silent
`"No relevant articles found."` is the specific bug that let ungrounded runs look
successful.

### 5. Reflection gains the verifier

`ReflectionAgent` takes an optional verifier. Before emitting a verdict it
verifies extracted claims. **A hypothesis with any `contradicted` claim is
quarantined** — enforced in code, not in prompt text, so it cannot be talked
around.

Quarantine, not rejection: the judge's false-contradiction rate is unmeasured
until sub-project 3 calibrates it, and `contradicted` additionally requires
abstract-backed evidence (metadata-only evidence caps at `unchecked`). A
quarantined hypothesis leaves the active pool but stays distinguishable from a
peer-review rejection, so it can be re-scored once calibration exists.
Promoting quarantine to rejection requires that measurement.

### Deliberately omitted (YAGNI)

- No caching layer. Measure real call volume first.
- No new agent. Verification belongs in the stage that already judges
  correctness.

## Data flow

Grounding, per literature-strategy generation task:

```text
goal ─▶ LiteratureTool.search()
          ├─ PubMedSource   (free, keyless, 3 req/s)
          └─ WebSource      (only when TAVILY_API_KEY is set)
        ─▶ [EvidenceRecord] ─▶ formatted into prompt with PMIDs
        ─▶ hypothesis cites refs it was actually shown
```

Verification, inside the reflection task, before the verdict:

```text
hypothesis.text ─▶ extract_claims()           1 model call → ≤4 claims
                     └─ per claim (concurrent):
                          sources.search_outcome(claim)  0 model calls
                          judge evidence vs claim     1 model call
                        ─▶ ClaimVerdict
                   ─▶ persist to claim_verdicts
                   ─▶ any contradicted ⇒ quarantined, verdict FLAGGED
                   ─▶ else model emits PASSED/FLAGGED/REJECTED as today
```

### Cost

A reflection task goes from 6 model calls (six tiers) to `6 + 1 + n_claims` — 11
at the default `max_claims=4`, so reflection costs roughly 1.8× more, not 6×.
The v1 spec's "1 model call" baseline was wrong: `core/reflection_pipeline.py`
runs six tiers, each with its own call.

The claim-judge calls use the **strong** model (they can veto the strong model's
own review), so the added cost is weighted toward the expensive tier.

Mitigations: `max_claims` is configurable; `evidence.concurrency` bounds
concurrent lookups per hypothesis. Note that concurrency is **not** a cost
mitigation — it was one in v1, and it was the mechanism that produced
self-inflicted 429s. It is bounded here for correctness, not for speed.

PubMed itself is free, and adds no per-call cost at any setting.

## Error handling

Governing principle, learned from the 2026-08-09 live-run failures: **an optional
capability must never fail a run the user paid for.**

| Failure | Behavior |
|---|---|
| PubMed unreachable or rate-limited | throttle + `Retry-After` backoff, then `unchecked`; never raises |
| All sources failed | `unchecked` (reported via `SearchOutcome.failed_sources`), and log a warning |
| All sources searched, no hits | `unsupported` — a finding, not an error |
| Evidence is metadata-only (no abstract) | `contradicted` is downgraded to `unchecked` |
| Claim extraction returns junk | zero claims; reflection proceeds as today |
| Verifier model call fails | `unchecked`; reflection still emits its verdict |
| No network (`--provider offline`) | fixture source only; zero outbound connections |

There is no path where evidence trouble raises into the supervisor.

## Testing

1. **Unit** — `extract_claims` / `verify_claim` against canned records. Every
   status reachable, including `contradicted` forcing REJECTED.
2. **Network-blocked** — the whole evidence layer under a socket guard (the
   pattern `company/offline_smoke.py` uses), asserting zero outbound connections
   and no raised exceptions.
3. **Contract tests against the real APIs** — marked `@pytest.mark.network`,
   excluded from the default run, opt-in via `pytest -m network`. These catch
   NCBI changing its response shape.

   `[tool.pytest.ini_options]` currently declares no markers, so this needs two
   additions or the marker is silently ignored and the tests run by default:

   ```toml
   markers = ["network: hits a real external API; excluded by default"]
   addopts = "-m 'not network'"
   ```

Layer 3 exists because of a specific lesson: 620 offline tests passed while every
live generation task failed, since fixture backends agree with the parsers by
construction. Offline tests cannot detect a contract mismatch with the outside
world.

`--provider offline` additionally gains a `FixtureEvidenceSource` so the free
path exercises grounding and verification end to end.

## Success criteria

1. A live run produces hypotheses whose citations are real PMIDs that resolve.
2. `claim_verdicts` is populated for every reviewed hypothesis.
3. A hypothesis with an abstract-backed contradicted claim is quarantined.
4. `--provider offline` runs the full flow with zero network and no new failures.
5. The default `pytest` run stays hermetic, free, and green.
6. A run with no network and no keys still completes — degraded to `unchecked`,
   never failed.
