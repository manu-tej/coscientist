# Publication Readiness

Status: **paused research prototype.**

Visibility history: public from 2026-07-02; private from 2026-08-01; republished in November 2026 as a single-commit snapshot of the private development repository.

The framing and hygiene rules below apply to every public update.

## What I built

- Built the experimental architecture for a Claude-based AI co-scientist reimplementation.
- Focused the repo on scientific hypothesis generation and evaluation, not production deployment.
- Added a benchmark/evaluation direction to test whether ranking mechanisms track truth.

## Where coding agents helped

- Coding agents co-authored most of the commits: in the private development history, 151 of the 213 commits made from 2026-05-28 to 2026-11-09 carry a `Co-Authored-By: Claude` trailer.
- Agent-generated summaries are not publication claims until reviewed against source files and test outputs.

## Release hygiene

- Run databases, logs, and `.env` are gitignored. Before each public update, check that none of them, and no provider token, is staged.
- The co-scientist paper is cited through `docs/REFERENCES.md`; its PDF is not in the current tree. Eight prompt templates come from the paper, and one plan document quotes them; `NOTICE` lists these files.

## Verification

- 2026-07-02: `uv run pytest -q` -> 283 passed, 1 skipped, 4 warnings.
- 2026-08-04: `uv run pytest -q` -> 610 passed, 1 skipped (post branch consolidation).
- 2026-11-09: `uv run --extra dev pytest -q` -> 771 passed, 1 skipped, 4 deselected (fresh `git archive` export).
- 2026-11-11: `uv run --extra dev pytest -q` -> 774 passed, 1 skipped, 4 deselected (fresh `git archive` export, Python 3.12.8; 773 passed, 2 skipped without the Codex CLI).
- 2026-07-02: `git diff --check` -> passed.

## Safe Public Framing

Use: "experimental reimplementation inspired by the AI co-scientist paper."

Avoid: implying affiliation, equivalence to the original system, or validated scientific discovery performance unless backed by current benchmark results.
