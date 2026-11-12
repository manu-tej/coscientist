# CoScientist

Experimental reimplementation of an AI co-scientist-style system on Claude.

**Status:** paused research prototype. A 2026-09-25 audit found that concordance between tournament rankings and ground truth is not established. See [Status](#status).

**About this repository.** This is a single-commit snapshot, taken in November 2026, of a private development repository. Commit hashes and commit counts cited in the docs refer to that history and do not resolve here.

This repo explores multi-agent scientific hypothesis generation: worker agents propose and refine hypotheses, a supervisor coordinates the process, and an evaluation harness tests whether tournament-style rankings track external ground truth. On top of that core sits a model-portable, hierarchical layer for biotech research and drug-repurposing portfolio work.

It is inspired by the Google AI co-scientist paper (`arXiv:2502.18864`) but is not affiliated with Google. Eight of this repo's prompt templates reproduce or adapt the prompts published in that paper, and one plan document in `docs/superpowers/` quotes them; [`NOTICE`](NOTICE) lists these files and their licence. See [`docs/REFERENCES.md`](docs/REFERENCES.md).

## What I built

- A multi-agent hypothesis-generation scaffold with supervisor coordination.
- Provider adapters for Claude subscription/API-style workflows, plus Codex and any OpenAI-compatible endpoint.
- A benchmark/evaluation harness aimed at checking whether agent rankings correspond to real scientific correctness, not just plausible debate output.
- A hierarchical portfolio layer that chains co-scientist runs through gated stages with a dual token/compute-credit ledger.
- Tests for core agent, bench, report, and provider behavior.

## Results

The evaluation harness asks one falsifiable question: do the tournament's Elo
rankings actually track ground-truth correctness? On GPQA-bio, the answer so far
is **not established, and the retained runs cannot establish it either way.**

Every run on a question is its own tournament (every hypothesis starts at Elo
1200 and moves only against siblings from the same run), so the verdict rests
on a conditional logistic regression of correctness on Elo stratified by
tournament (`bench/concordance.py`). That fit uses only tournaments where both
correctness and Elo vary. An offline audit of the 17 retained GPQA-bio run
databases (`bench/tournament_audit.py`, 2026-09-25, no model calls) found that
one of them meets that bar honestly.

The run databases are not in this repository (`bench_runs/` is gitignored), so
none of the figures in this section can be recomputed from a clone: on a clone,
`python -m bench.tournament_audit` stops with "no run databases match". The
questions are the biology items of the `hendrydong/gpqa_diamond_mc` mirror on
Hugging Face (`bench/datasets/gpqa.py`). The match-level counts below (which
side won, text length) came from queries against the databases that are not
committed. Correctness was scored by the answer parser as it stood before its
2026-11-11 fix, which read the article "a" after "answer is" as option A.
Rescoring all 171 hypothesis rows in the 17 databases with the corrected parser
changes none of them, so the figures stand as computed.

The audit sorted the 17 tournaments as follows:

| Tournaments | What the audit found |
|---|---|
| 8 | Ceiling: every generated hypothesis, rejected ones included, chose the gold letter. |
| 1 | Ceiling by selection: 7 of 9 correct, but review rejected exactly the two wrong ones and the benchmark reads active rows only. |
| 1 | Elo tied: the only match was between two hypotheses that review later rejected. Two of the eight ceilings are also tied this way. |
| 1 | Every hypothesis was persisted with empty text, a backend/parser defect fixed in July and August 2026. |
| 2 | Pass the fit's test only because empty-text rows score as wrong, and their Elo spread came from empty-vs-empty matches the old judge defaulted to slot one. |
| 1 | Passes only through two parser misses (`Option (B)`, `**(B)**`, `\boxed{B}`); read correctly it is 8 of 8, a ceiling. |
| 2 | Pass on a single match each, played between two hypotheses of the same correctness. |
| 1 | Clean: 13 matches, real text, mixed correctness. The correct side won 1 of its 7 correct-vs-wrong matches. |

The slopes reported here before 2026-09-25 (−1.85 per SD, p = 0.076 on the six
seed-2 tournaments; −0.80, p = 0.083 on all 17) are withdrawn. One of the two
strata behind the first and five of the six behind the second are in the
contaminated rows above. Dropping empty-text rows gives −0.73, p = 0.119;
also dropping single-match tournaments leaves one stratum and no fit. An
Elo-free check says the same: across all 89 matches ever played, 8 pitted a
correct answer against a wrong one, and the correct side won once. Elo also
tracks length: in the latest runs the longer text won 60 of the 79 matches
between two non-empty texts (49 of 64 where both answers were correct). The
difficulty-corrected comparison against base sampling was never
computed on these runs (no reference red line is stored), so that check is
still open.
The earlier headline (Spearman ρ = −0.56, p = 0.32) correlated five Elo-bucket
means, not the responses, and is now reported only as a descriptive.

This is not a negative result. The post-hoc audit, not the headline fit, showed
that the retained runs cannot answer the question either way. **Caveats:** the
sample is small, and because GPQA is widely reposted by 2026, a correct answer
is not evidence of uncontaminated capability.

## Quickstart

All commands run under [`uv`](https://docs.astral.sh/uv/). Python 3.12 or newer is required.

```bash
# Install (dev extras include the test + benchmark deps)
uv sync --extra dev
```

Everything in this first group is **free**: no credentials, no provider spend, and
none of these commands makes a network request. (The explorer page that step 3
opens in your browser loads its web font from Google Fonts and an iframe-resizer
script from cdnjs.cloudflare.com.) Start here — it exercises the real
orchestrator, tournament, and persistence.

```bash
# 1. Run the full pipeline against the deterministic offline provider
uv run python coscientist.py "Explain mechanisms of pulmonary arterial hypertension" \
    --provider offline --max-tasks 12

# 2. Deterministic structural smokes (network-guarded, seeded fixtures)
uv run python -m company.offline_smoke
uv run python -m bench.offline_smoke

# 3. Explore any run produced above — pass the .db path printed by step 1
uv run python -m ui.launch <path-printed-by-step-1>
```

The `offline` provider is a fixture, not a model: its hypotheses are templated and
its rankings are synthetic. It proves the machinery runs, and nothing about
scientific quality.

For real model calls, configure a provider first. The default, `claude-code`,
uses an installed and authenticated `claude` CLI and needs no token in `.env`.

```bash
cp .env.example .env   # add keys only for token-based providers; never commit real tokens
```

```bash
# 4. A co-scientist session on a real provider   (BILLED / uses your subscription)
uv run python coscientist.py "Your research goal here"

# 5. GPQA-bio concordance harness (Elo <-> ground truth)
#    (BILLED; prints a call-range estimate and asks before spending)
uv run python -m bench concordance --dataset gpqa-bio --limit 6 --out artifacts/concordance
```

Do not commit `.env`; it is ignored by Git.

## Architecture

Six specialist agents — generation, reflection, ranking, proximity, evolution,
and meta-review (`agents/`) — are dispatched by a **dynamic-weighting
supervisor** (`core/supervisor.py`) that reweights which agent runs next from
live system stats, so the mix adapts as a run progresses. Generated hypotheses
compete in an **Elo tournament** (`core/tournament.py`). The **concordance loop**
(`bench/`) then closes the evaluation: it checks whether those Elo rankings track
ground-truth correctness on **GPQA-bio**, and reports the result honestly whether
or not concordance holds.

The portfolio layer wraps that core in a second hierarchy:

```text
CSO portfolio
  └─ programs (bounded concurrent stage teams)
       └─ stage Supervisor
            └─ generation, reflection, ranking, proximity,
               evolution, and meta-review tasks
                 └─ evidence-cited, schema-validated team synthesis
                      └─ budgeted CEO gate
```

The hierarchy is executable, not just descriptive. Each live stage has its own
run ID, SQLite evidence trail, specialist roster, task trace, ranked candidates,
token estimate, and gate packet. Concurrent quarters are failure-atomic in the
scientific portfolio state: one failed team cannot partially commit another
team's result. External/computational usage from every attempted team remains
charged to the lifetime credit and token limits, including when science state
rolls back.

## Evidence and verification

Hypothesis generation is grounded on PubMed, and reviewed hypotheses have their
factual claims checked against retrieved abstracts. PubMed is free and needs no
API key; set `NCBI_API_KEY` (also free) to raise the rate ceiling from 3 to 10
requests/second.

Each checked claim is stored in `claim_verdicts` with its status, the records
consulted, and a one-line rationale, so provenance is queryable rather than
buried in prose.

- `supported` / `contradicted` — the retrieved abstracts addressed the claim.
- `unsupported` — the search ran and returned nothing relevant.
- `unchecked` — the check could not be made: a source was unavailable, the judge
  was unparseable, or the evidence was metadata-only. **Never** read `unchecked`
  as evidence of absence.

`contradicted` has two independent producers, and they are governed
differently on purpose:

1. **Literature judgement** — the fact-checking model reviews the retrieved
   abstracts and rules the claim contradicted. This is a model *opinion*, so
   it is gated: it requires abstract-backed evidence and that the judge cite
   the record it relied on (`[N]`); metadata-only evidence, or a contradiction
   that doesn't cite a backed record, caps at `unchecked` instead, since a
   title and a journal name cannot justify an irreversible veto.
2. **Citation attribution** — a PMID the hypothesis cites was never among the
   records shown to it at generation time. This is a deterministic textual
   fact about what the model wrote versus what it was given, not a model
   judgement, so it deliberately skips the literature-judgement gate above —
   there is no "record it relied on" to cite, because the whole point is that
   no such record was ever retrieved for this hypothesis.

The two are distinguishable in `claim_verdicts`: a citation-attribution
`contradicted` row always has `records=[]` (and a `claim` of the form
`"citation PMID <ref>"`), while a literature-judgement `contradicted` row
carries the record(s) the judge cited. `records=[]` is the machine-readable
discriminator between them.

A hypothesis with a contradicted claim (from either producer) is quarantined
(removed from the active pool, distinguishable from a peer-review rejection)
rather than deleted or rejected. It is quarantine and not rejection because the
judge's false-contradiction rate is not yet measured; calibrating the judge is
deferred work.

Verification costs about one strong-model call per claim, plus one to extract
them. Tune it:

```yaml
evidence:
  max_claims: 4    # 2 roughly halves the added cost
  concurrency: 2   # concurrent lookups per hypothesis
```

Under `--provider offline` a deterministic fixture source is used, so the free
path exercises the whole flow with no network. Fixture citations are
non-numeric `OFFLINE-#####` refs so they can never be mistaken for a real PMID.

## Model portability and scaling

The system depends on one small asynchronous backend contract:

```python
await backend.call(system_prompt, user_prompt, use_strong=False, max_tokens=8192)
```

Built-in routes are:

- `claude-code`: authenticated Claude Code CLI (the `config.yaml` default)
- `anthropic`: metered Anthropic API
- `codex`: authenticated Codex CLI
- `openai-compatible`: any local or hosted `/v1/chat/completions` endpoint

An arbitrary model works when one of those routes can serve it, or when a new
adapter implements the contract. Model capability still matters: more workers
increase in-flight throughput, but they do not make a weak model follow schemas
or reason like a stronger one.

There are three independent scale controls:

- `--workers`: concurrent in-flight tasks inside each program's stage team.
- `--program-workers`: program stage teams allowed to run concurrently.
- `--max-tasks` / `--max-agent-tasks`: total task/search budget and therefore
  breadth inside one run/team.

Concurrency uses bounded queues/semaphores. Raising worker counts without raising
the task budget completes the same maximum amount of work with more overlap; it
does not create additional task roles. Provider rate limits and the machine's
SQLite/file capacity remain practical ceilings.

Model and provider overrides are explicit and recorded in benchmark reports:

```bash
uv run python coscientist.py \
  "Identify testable mechanisms for pulmonary arterial hypertension" \
  --provider openai-compatible \
  --model-fast llama3.1:8b \
  --model-strong llama3.1:70b \
  --workers 8 \
  --max-tasks 40
```

Configure the OpenAI-compatible base URL and API-key environment variable in
`config.yaml`. A local endpoint can set `api_key_env: null`. See
[`docs/HOSTED_MODELS.md`](docs/HOSTED_MODELS.md) for a hosted or remote endpoint.

## Reproducible offline hierarchy smoke

Run this before the commands in the next section. It uses the real Supervisor,
six agent behaviors, two concurrent program teams, evidence-cited ranking,
ledger reconciliation, persistent SQLite provenance, and an intentional
atomic-rollback probe. Its backend responses and science fixtures are
deterministic and it makes no network request.

```bash
uv run python -m company.offline_smoke
```

The command writes `artifacts/hierarchy_offline_smoke.json`. This is a structural
integration check, **not** evidence of scientific quality or model capability.

## Exploratory: simulated drug-repurposing portfolio

This layer is a simulation and has not been evaluated. Most stage experiment
readouts are seeded random draws labelled T0 (`company/science.py`); only some
scores come from real methods (knowledge-graph proximity on the CPU by default,
PubMed and Modal GPU providers when requested). A program the CEO advances
survives its gate only if a seeded random roll falls under a probability of
success built from illustrative baseline values (`company/models.py`,
`company/engine.py`). With `--hierarchical`, the stage teams make real model
calls on the selected provider.

Create a portfolio and programs:

```bash
uv run python company.py --state demo_company.json new \
  --name Helix \
  --disease "pulmonary arterial hypertension" \
  --budget 500 \
  --token-budget 5000000

uv run python company.py --state demo_company.json program-new \
  --name PAH-A --value 900

uv run python company.py --state demo_company.json program-new \
  --name PAH-B --value 700
```

Run one live quarter with two program teams and four workers per team:

```bash
uv run python company.py --state demo_company.json quarter \
  --hierarchical \
  --workers 4 \
  --program-workers 2 \
  --max-agent-tasks 24 \
  --provider claude-code
```

`--token-budget` is enforced before each hierarchical model call through one
shared concurrent envelope. Its units are the portable `chars/4` prompt estimate
plus requested output tokens; they are a model-independent admission control,
not exact provider billing tokens. The common backend contract does not expose
vendor tokenizer usage, and subscription CLIs may not enforce `max_tokens`.
Failed or cancelled quarters roll back candidates, experiments, gates, and
committed program spend, but retain measured attempt usage so retries cannot
reset the company's hard resource walls.

`--max-agent-time` is the async deadline for a stage-team result. Python cannot
terminate an already-running synchronous adapter thread; if that boundary is
crossed, its result is discarded and the conservative preflight credit
reservation remains charged. Network/GPU adapters should therefore also enforce
their own provider-side deadlines.

Omit `--auto` to leave gate decisions pending for CEO review:

```bash
uv run python company.py --state demo_company.json status
uv run python company.py --state demo_company.json gate PAH-A
uv run python company.py --state demo_company.json advance PAH-A
```

`--literature` requests live PubMed evidence. `--structure` and `--affinity`
request billed Modal GPU providers. Install and deploy the optional GPU surface
first:

```bash
uv sync --extra dev --extra gpu   # `uv sync` is exact: dropping --extra dev uninstalls pytest
uv run modal deploy modal_app.py
```

The registry records the provider and fidelity tier actually used. If Modal is
missing, unauthenticated, undeployed, or a provider abstains, it falls back to
the T0 method; the CLI describes these flags as requests rather than promising
that a T2 call succeeded. The credit ledger reserves the conservative maximum
cost before launch.

Run the separate live PubMed report card with:

```bash
uv run python company.py validate
```

This command is strict: it emits a complete report only when all 54 curated
gold/decoy pairs return valid scores. Missing/invalid observations produce no
scientific verdict and exit with status 2. A complete report prints observed
metrics but does not invent a pass/fail threshold.

## Benchmarks

Every paid benchmark prints a call-range estimate and asks for confirmation
unless `--yes` is supplied. A goal that has agent failures or produces no
hypotheses fails the benchmark by default.

Completed system runs are reused through `bench/manifest.sqlite`. The cache
identity includes the research prompt, exact provider/model pair, worker and
task budgets, seed, execution configuration, personas/ablations, and
run-producing source. Captured SQLite runs are integrity-checked before reuse;
partial, failed, missing, or modified runs are never accepted. Gold labels,
answer parsing, and analysis settings such as `--min-support` are intentionally
excluded so the same paid run can be reanalyzed. The ranking mode is part of the
identity, so an absolute-ranking run is a separate paid run. Concurrent
processes use expiring, owner-guarded leases: one process
runs a missing cell while the others wait for and reuse its verified result;
the owner timeout requests cancellation, and ownership is retained until the
backend actually stops so a non-cooperative adapter cannot trigger duplicate
paid work. The built-in HTTP/CLI adapters honor cancellation; custom adapters
must do the same for prompt takeover. Calls made before a provider failure
remain visible in failure and aggregate usage accounting.
Use `--manifest PATH` to share another manifest or `--no-cache` to force fresh
execution. Legacy pre-v3 manifest rows are ignored. Base-model
`--reference-samples` are still fresh calls and are reported separately.

Gold-entity rediscovery (the default goals are three BiomniBench-DA tasks; see
[`docs/REFERENCES.md`](docs/REFERENCES.md) for the source and licence):

```bash
uv run python -m bench rediscover \
  --limit 3 \
  --workers 4 \
  --max-tasks 30 \
  --provider claude-code \
  --out artifacts/rediscovery
```

The tournament ablation is runnable on the same provider-neutral backend
contract. It scores each hypothesis exactly once on the 1–5 rubric, persists
those scores separately from Elo, and requires a distinct judge model:

```bash
uv run python -m bench rediscover \
  --ranking-mode absolute \
  --judge-provider openai-compatible \
  --judge-model gpt-5 \
  --limit 3 \
  --workers 4 \
  --max-tasks 30 \
  --provider claude-code \
  --out artifacts/rediscovery_no_tournament
```

Absolute-score calls share the same attempted-call/token accounting as system
calls. The judge provider/model, rubric prompt hash/version, axes, weights,
panel size, and decoding tier are hashed into the cache identity and persisted
inside the captured run database. A judge model matching either system tier is
rejected rather than silently self-judging.

GPQA concordance:

```bash
uv run python -m bench concordance \
  --dataset gpqa-bio \
  --limit 10 \
  --workers 4 \
  --reference-samples 4 \
  --provider claude-code \
  --out artifacts/concordance
```

The GPQA loader fails closed if its source cannot be reached. Use
`--allow-fixture-fallback` only for an explicitly labeled local smoke run. The
fallback fixture (`bench/datasets/fixtures/gpqa_sample.jsonl`) holds three
hand-written textbook questions, not GPQA items. Real questions are fetched at
run time from the `hendrydong/gpqa_diamond_mc` mirror on Hugging Face and are
not stored in this repository.

Model × worker × seed sweep:

```bash
uv run python -m bench sweep \
  --benchmark rediscover \
  --limit 2 \
  --max-tasks 20 \
  --worker-counts 1,2,4 \
  --models 'openai-compatible=llama3.1:8b,llama3.1:70b' \
  --models 'codex=gpt-5.1-codex' \
  --seeds 0,1 \
  --out artifacts/model_worker_sweep
```

The preferred model-spec syntax is
`provider=fast-model[,strong-model]`; it preserves colons in local model IDs.
Duplicate cells are removed before execution. JSON reports include the exact
provider/models, worker count, seed, task trace, failures, call count, observed
backend concurrency, estimated tokens, run ID, database path, dataset
provenance, and source version.

Benchmark results are measurements of the selected model/data/run settings, not
proof that generated biomedical claims are clinically valid. Candidate packages
retain evidence provenance and red flags so they can be reviewed rather than
treated as autonomous decisions.

## Also included

`qc/` is a QC scientist for biomedical tabular/omics data whose uncertainty is
computed, not model-estimated. Every flag carries an effect size. The
family-wise detectors — covariate shift and target leakage — report
Benjamini-Hochberg FDR-corrected q-values across their test family. The
descriptive detectors (missingness, outlier fraction) report a rate and
currently emit a placeholder `p = q = 0.0`, which renders as "confidence 1.00";
treat those two as descriptive counts, not inference. An optional agent layer
sits on top whose added value is measured, not assumed. It sits outside the
benchmarked path and is exploratory.

## Where coding agents helped

Coding agents co-authored most of the commits: in the private development history, 151 of the 213 commits made from 2026-05-28 to 2026-11-09 carry a `Co-Authored-By: Claude` trailer. I own the architecture choices, evaluation framing, and any public claims made from the repo.

## Status

Paused research prototype. This is an experimental system for studying agentic
scientific reasoning and evaluation, not a production tool. Hypotheses and rankings
it produces need human review.

The offline audit of 2026-09-25 found that the retained GPQA-bio runs do not
establish concordance between tournament Elo rankings and ground-truth correctness.
They cannot establish it either way. The audit table is in [Results](#results).

Next step if the project resumes: a powered comparison over a frozen candidate set.
[`docs/ROADMAP.md`](docs/ROADMAP.md) describes the planned experiment.
[`docs/superpowers/`](docs/superpowers/README.md) holds the design record: specs,
agent-facing implementation plans, and one adversarial review.

[`PUBLICATION.md`](PUBLICATION.md) records when the repository was public and private.

## Verification

Latest local verification (2026-11-11, Python 3.12):

```bash
uv run --extra dev pytest -q
# 774 passed, 1 skipped, 4 deselected
```

The four deselected tests are marked `network` and call external APIs. One more
test skips when the Codex CLI is not installed (773 passed, 2 skipped).

Verified against a fresh `git archive` checkout, not just a working copy — the
suite previously passed only where a gitignored `bench_runs/` already existed.

## License

MIT. See [`LICENSE`](LICENSE). The MIT licence does not cover the files listed
in [`NOTICE`](NOTICE): eight prompt templates and one plan document that
reproduce or quote the co-scientist paper's prompts, used under CC BY 4.0.
