# Publication Readiness

Status: private GitHub repo, not public-ready yet. Do not make public until the user explicitly does so after review.

## What I Did

- Built the experimental architecture for a Claude-based AI co-scientist reimplementation.
- Focused the repo on scientific hypothesis generation and evaluation, not production deployment.
- Added a benchmark/evaluation direction to test whether ranking mechanisms track truth.

## Coding-Agent Help

- Coding agents assisted with implementation, tests, refactors, and intermediate docs.
- Agent-generated summaries are not publication claims until reviewed against source files and test outputs.

## Current Gates

- `poc/` is ignored by default. Promote only reviewed files from it if they are part of the public scope.
- Check for generated logs, local DBs, and provider tokens before making the repo public.
- Keep the local removal of `google_co_scientist_2502.18864.pdf`; cite the paper through `docs/REFERENCES.md` unless redistribution rights are confirmed.

## Verification

- 2026-07-02: `uv run pytest -q` -> 283 passed, 1 skipped, 4 warnings.
- 2026-07-02: `git diff --check` -> passed.

## Safe Public Framing

Use: "experimental reimplementation inspired by the AI co-scientist paper."

Avoid: implying affiliation, equivalence to the original system, or validated scientific discovery performance unless backed by current benchmark results.
