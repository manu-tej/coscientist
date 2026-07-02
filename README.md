# coscientist

Experimental reimplementation of an AI co-scientist-style system on Claude.

This repo explores multi-agent scientific hypothesis generation: worker agents propose and refine hypotheses, a supervisor coordinates the process, and an evaluation harness tests whether tournament-style rankings track external ground truth.

It is inspired by the Google AI co-scientist paper (`arXiv:2502.18864`) but is not affiliated with Google. See [`docs/REFERENCES.md`](docs/REFERENCES.md).

## What I Built

- A multi-agent hypothesis-generation scaffold with supervisor coordination.
- Provider adapters for Claude subscription/API-style workflows.
- A benchmark/evaluation harness aimed at checking whether agent rankings correspond to real scientific correctness, not just plausible debate output.
- Tests for core agent, bench, report, and provider behavior.

## Where Coding Agents Helped

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
