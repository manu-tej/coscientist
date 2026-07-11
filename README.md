# coscientist

Experimental reimplementation of an AI co-scientist-style system on Claude.

This repo explores multi-agent scientific hypothesis generation: worker agents propose and refine hypotheses, a supervisor coordinates the process, and an evaluation harness tests whether tournament-style rankings track external ground truth.

It is inspired by the Google AI co-scientist paper (`arXiv:2502.18864`) but is not affiliated with Google. See [`docs/REFERENCES.md`](docs/REFERENCES.md).

## What I built

- A multi-agent hypothesis-generation scaffold with supervisor coordination.
- Provider adapters for Claude subscription/API-style workflows.
- A benchmark/evaluation harness aimed at checking whether agent rankings correspond to real scientific correctness, not just plausible debate output.
- Tests for core agent, bench, report, and provider behavior.

## Results

The evaluation harness asks one falsifiable question: do the tournament's Elo
rankings actually track ground-truth correctness? On GPQA-bio, the answer so far
is **no — concordance was not established.** The most-populated run scored
**Spearman ρ = −0.56** between Elo bucket and answer accuracy (**n = 41 graded
responses across 6 GPQA-bio questions, 5 Elo buckets; p = 0.32, not
statistically significant**), and the system did not beat difficulty-corrected
base sampling.

Reporting the negative result is the point: the harness exists to catch
tournaments that produce plausible debate without tracking correctness, and here
it did. **Caveats:** sample sizes are small, so figures are directional rather
than proven; and because GPQA is widely reposted by 2026, top-1 accuracy
(~0.6) is not evidence of uncontaminated capability.

## Quickstart

All commands run under [`uv`](https://docs.astral.sh/uv/).

```bash
# Install (dev extras include the test + benchmark deps)
uv sync --extra dev
cp .env.example .env   # add provider auth; never commit real tokens

# 1. Run a co-scientist session on a research goal
uv run python coscientist.py "Your research goal here"

# 2. Run the GPQA-bio concordance harness (Elo <-> ground truth)
uv run python -m bench concordance --dataset gpqa-bio --limit 6

# 3. Launch the run explorer UI (hypotheses, tournament matches, eval reports)
uv run python -m ui.launch bench_runs
```

## Architecture

Six specialist agents — generation, reflection, ranking, proximity, evolution,
and meta-review (`agents/`) — are dispatched by a **dynamic-weighting
supervisor** (`core/supervisor.py`) that reweights which agent runs next from
live system stats, so the mix adapts as a run progresses. Generated hypotheses
compete in an **Elo tournament** (`core/tournament.py`). The **concordance loop**
(`bench/`) then closes the evaluation: it checks whether those Elo rankings track
ground-truth correctness on **GPQA-bio**, and reports the result honestly whether
or not concordance holds.

## Also included

Beyond the core co-scientist, the repo carries two extended experiments that sit
outside the benchmarked path. `company/` simulates an AI-native drug-repurposing
techbio: a portfolio loop that chains co-scientist runs through gated stages with
a dual token/compute-credit budget ledger and a CSO/CEO control surface — the
per-stage science is a seeded, reproducible stub, so the org loop runs end-to-end
today and swapping in live runs is the single integration seam. `qc/` is a QC
scientist for biomedical tabular/omics data whose uncertainty is real statistics,
not model-estimated: every flag carries a multiplicity-corrected p-value and
effect size, with an optional agent layer on top whose added value is measured,
not assumed. Both are exploratory.

## Where coding agents helped

Coding agents helped implement and iterate on the scaffold, tests, benchmark runners, and documentation. I own the architecture choices, evaluation framing, and any public claims made from the repo.

## Status

Public research prototype — an experimental system for studying agentic scientific
reasoning and evaluation, not a production tool. Hypotheses and rankings it produces
need human review.

## Setup

```bash
uv sync --extra dev
cp .env.example .env
```

The default configuration can use Claude subscription/OAuth-style auth. Do not commit `.env` or real provider tokens.

## Verification

Latest local verification:

```bash
uv run pytest -q
# 283 passed, 1 skipped, 4 warnings
```

## License

MIT. See [`LICENSE`](LICENSE).
