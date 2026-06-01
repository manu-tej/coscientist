# Evaluation Harness (`bench/`) — Design Specification

**Date:** 2026-06-01
**Goal:** Build a rigorous, automatable evaluation harness for the AI co-scientist that (1) validates the core premise that the tournament Elo tracks real quality, (2) shows the self-improvement loop works, (3) proves the multi-agent architecture beats trivial baselines and identifies which agents contribute — with **computational biology as the flagship configured domain** while staying faithful to the paper's methodology (arXiv:2502.18864, §4.1–4.3).

---

## 1. Goals & Scope

**The central problem this solves:** the system's internal quality signal (Elo) is auto-evaluated by the same model family that generates hypotheses. Nothing guarantees high Elo means good science rather than "good at winning LLM debates." This harness grounds Elo against ground truth (the paper's key construct-validity check) and against external anchors, so the self-improvement loop is provably optimizing quality, not noise.

**In scope (v1 — full bundle):**
- **Concordance** (Tier 1): Elo↔ground-truth-accuracy with proper statistics (Spearman, Kendall, logistic) and the paper's reference-baseline control.
- **Scaling** (Tier 1b): temporal-bucket Elo curves (best, top-10) showing monotonic improvement.
- **LLM-judge** (Tier 2): configurable, bias-controlled rubric scoring (novelty/feasibility/correctness/impact) as the run-independent quality anchor.
- **Cross-tournament**: a shared Elo tournament that makes baseline/ablation variants comparable (Elo is otherwise per-run-internal).
- **Single-shot baseline** (Tier 3a): single-shot + best-of-n(32) controls.
- **Ablations** (Tier 3b): leave-one-out agent ablations, budget-matched.
- **Comp-bio specialization**: GPQA-diamond Biology subset + ResearchBench biology splits + a curated comp-bio gold-entity goal set; judge rubric parameterized to `field="computational biology"`.

**Out of scope (v1):**
- Wet-lab validation (out of reach without a laboratory).
- Large-scale human expert panels (the harness records artifacts that a human study *could* later score, but recruiting experts is not part of this build).
- Live leaderboard / hosted dashboard (reports are local markdown/JSON).

**Decoupling note:** this harness does **not** depend on the parked "pluggable LLM backends" work. It runs the system-under-test through the existing wiring and uses a self-contained, separately-configured judge client. If the backend work lands later, `bench/runner.py` picks it up transparently via the same factory the entrypoint uses.

---

## 2. Module Layout

```
bench/
├── __init__.py
├── goalset.py            # BenchGoal model + goal-set loaders (jsonl)
├── datasets/
│   ├── __init__.py
│   ├── gpqa.py           # load GPQA-diamond (hendrydong/gpqa_diamond_mc), filter subject=Biology; MCQ scoring
│   ├── researchbench.py  # load ResearchBench biology splits (ankilok/Researchbench): ground-truth hypotheses
│   └── comp_bio_goldset.jsonl   # curated comp-bio goals with gold entity sets (user-extendable)
├── goldset.py            # entity gold-set recall scorer (token-subsequence match, Kaimen pattern)
├── runner.py             # run full system OR single-shot baseline on a goal → BenchRun (pool + Elo trajectory)
├── concordance.py        # Tier 1: bucketing, per-bucket accuracy, Spearman/Kendall/logistic, reference baseline
├── scaling.py            # Tier 1b: temporal-bucket Elo curves (best, top-10)
├── judge.py              # Tier 2: pluggable bias-controlled LLM-judge rubric
├── cross_tournament.py   # shared Elo across candidates/variants (reuses core/tournament)
├── baseline.py           # Tier 3a: single-shot + best-of-n(32)
├── ablation.py           # Tier 3b: leave-one-out agent ablations, budget-matched
├── report.py             # aggregate results → markdown + JSON report
└── cli.py                # `python -m bench <command> [opts]`
tests/
├── test_bench_goalset.py
├── test_bench_goldset.py
├── test_bench_concordance.py
├── test_bench_scaling.py
├── test_bench_judge.py
├── test_bench_cross_tournament.py
├── test_bench_baseline.py
└── test_bench_ablation.py
```

**Reuse from existing code:** `core/tournament.py::compute_elo_update` (init 1200, K=32) powers the cross-tournament; `core/state.py` match history (`elo_before/after`, `created_at`) enables temporal-bucket replay; `core/supervisor.py` + `core/orchestrator.py` run the system-under-test; `core/config_parser.py` parses bench goals into `ResearchPlanConfig`.

---

## 3. Data Model

`bench/goalset.py`:

```python
@dataclass
class BenchGoal:
    id: str
    goal: str                              # the research-goal text fed to the system
    domain: str = "computational biology"  # used to parameterize the judge rubric
    gold_answer: Optional[str] = None      # MCQ letter, for concordance (GPQA)
    gold_hypothesis: Optional[str] = None  # reference hypothesis, for discovery scoring (ResearchBench)
    gold_entities: list[str] = field(default_factory=list)  # known entities, for gold-set recall
    choices: Optional[list[str]] = None    # MCQ options (GPQA)
    metadata: dict = field(default_factory=dict)
```

A goal carries whichever ground-truth flavor its source provides: `gold_answer` (MCQ → concordance), `gold_hypothesis` (reference finding → discovery/judge), or `gold_entities` (entity recall). Goals load from `.jsonl` files (one `BenchGoal` per line) or dataset loaders.

`bench/runner.py` produces:

```python
@dataclass
class BenchHypothesis:
    id: str
    text: str
    summary: str
    elo_rating: float
    created_at: str          # for temporal bucketing
    elo_trajectory: list[tuple[str, float]]  # (timestamp, elo) from match history

@dataclass
class BenchRun:
    goal_id: str
    variant: str             # "full" | "no_evolution" | "single_shot" | "best_of_32" | ...
    hypotheses: list[BenchHypothesis]
    n_llm_calls: int
    wall_clock_s: float
    db_path: str             # the run's SQLite, for replay
```

---

## 4. Datasets (comp-bio flavored)

### 4.1 GPQA-diamond (`datasets/gpqa.py`) — paper-faithful concordance
- Load `hendrydong/gpqa_diamond_mc` (ungated mirror; 198 expert MCQs across bio/chem/physics).
- **Filter to the Biology subset** by default (`subject` field) for comp-bio focus; expose `--all-subjects` to run the full set.
- Each question → `BenchGoal(gold_answer=<letter>, choices=[...], domain="biology")`.
- Answer scoring: parse the chosen option letter from a hypothesis's text (regex for `answer: (A-D)` / `option (A-D)` / leading letter), compare to `gold_answer` → binary correct.
- **Contamination caveat** logged in the report: GPQA is widely reposted by 2026; high scores are not proof of uncontaminated capability.

### 4.2 ResearchBench biology splits (`datasets/researchbench.py`) — comp-bio held-out discovery
- Load `ankilok/Researchbench` (Parquet), filter to `Biology` / `Cell_Biology` rows (2024-only papers → contamination-resistant).
- Each row → `BenchGoal(goal=<research question + background>, gold_hypothesis=<published hypothesis>, domain="computational biology")`.
- Scored two ways: (a) gold-set entity recall (§5) if the gold hypothesis yields entities, (b) LLM-judge similarity of the system's top hypothesis to `gold_hypothesis` (§7) — "did it rediscover the real finding?"

### 4.3 Curated comp-bio gold-entity set (`datasets/comp_bio_goldset.jsonl`)
- A small, hand-curated, version-controlled set of comp-bio research goals (e.g. "identify regulatory mechanisms of X", "propose pathways linking Y and Z") each with `gold_entities` (genes, pathways, methods known to be correct answers).
- User-extendable: add lines to score the system on domains you care about. Ships with ~10 seed goals derived from well-established comp-bio findings.

---

## 5. Gold-Set Entity Recall (`bench/goldset.py`)

Adapted from Kaimen's robust matcher. A `GoldSet` is a list of canonical entities with aliases; recall = which gold entities the system surfaced anywhere across its hypotheses.

- **Match semantics:** Unicode-normalize → lowercase → split into alphanumeric tokens → check the entity's tokens appear as a contiguous subsequence in any searched field (title/summary/text). Robust to `PI3K-Akt`, `TGF-β`, `dimethyl-fumarate` where naive `\b` regex fails. Aliases express spelling variants (`TGF-beta` vs `TGF-β`).
- **Metric:** `recall@pool` = (gold entities hit) / (total gold entities) across all of the run's active hypotheses; also `recall@k` for top-k by Elo.

```python
def score_recall(hypotheses: list[BenchHypothesis], gold_entities: list[str], k: int | None = None) -> float: ...
```

---

## 6. Concordance (`bench/concordance.py`) — Tier 1, the crown jewel

Proves Elo is monotonically predictive of correctness — the validity check no open implementation has built.

**Procedure (reconstructed from §4.1):**
1. Each GPQA question → research goal (template below). Run the system; every generated hypothesis carries an Elo and a parsed answer.
2. Score each hypothesis's answer against `gold_answer` → binary correct.
3. **Bucket by Elo** in 50-point bins (`[1001-1050], [1051-1100], …`), pooled across all questions. Min support 5 hypotheses/bucket.
4. **Per-bucket accuracy** = fraction correct in the bucket (the "blue line").
5. **Reference baseline (the key control):** generate 32 **base-model** samples per question, where the base model is the system's underlying generator model called **directly** (a single `backend.call` with the GPQA→goal prompt, no agent harness — exactly the `single_shot` mechanism from §9, reused). Per-question reference accuracy = fraction correct of those 32. Per-bucket reference = mean reference accuracy over questions contributing to that bucket (the "red line"). This corrects for question-difficulty non-uniformity — without it, Elo↔accuracy could merely reflect "Elo tracks difficulty." The 32-sample reference is cached per question (keyed by question id + model) so it isn't regenerated across runs.
6. **Top-1 accuracy** = accuracy of the single highest-Elo hypothesis per question.

**Statistics (the value-add the paper omitted — it showed concordance only visually):**
- **Spearman ρ** (bucket-midpoint Elo vs per-bucket accuracy) — primary monotonicity. Pass: ρ ≥ 0.7, p < 0.05.
- **Kendall τ-b** — robust monotonicity with ties.
- **Response-level logistic regression** `correct ~ elo` — report coefficient (log-odds/Elo-point) + likelihood-ratio test; avoids binning artifacts. Pass: positive, significant.
- **Mean (blue − red) spread** with paired bootstrap CI — pass: CI excludes 0 (system beats difficulty-corrected base sampling).

**GPQA→goal template** (flagged uncertain in paper; this is our explicit choice):
```
Research goal: Determine the correct answer to the following graduate-level
{domain} question and justify it rigorously.

Question: {question}

Options:
(A) ... (B) ... (C) ... (D) ...

Produce a hypothesis stating which option is correct and why.
```

`scipy` (`spearmanr`, `kendalltau`) and `statsmodels` (logistic GLM) added as deps.

---

## 7. LLM-Judge (`bench/judge.py`) — Tier 2, bias-controlled

The run-independent quality anchor (Elo can't compare across runs; the rubric can). Also scores ResearchBench discovery similarity.

**Backend (configurable, default cross-family):**
- A `JudgeClient` separate from the system-under-test. Selection: if a cross-family key is present (`OPENAI_API_KEY` / `GEMINI_API_KEY`), default the judge to that family (strongest bias control); else fall back to a *different Claude model* than the generator with a **logged warning** that same-family judging weakens bias control. Configurable via `bench.judge_provider` / `bench.judge_model`.
- **Hard rule enforced in code:** judge model ID ≠ generator model ID. If equal, raise `BenchError`.

**Rubric (4 axes, 1–5, explicit anchors from §4.3 extended):** novelty, feasibility, correctness/soundness, impact. Rubric table parameterized by `field` (default "computational biology"). Output parsed as JSON from the judge's text response (no cross-provider tool-calling dependency).

**Bias controls (the paper under-specified these):**
- **Judge ≠ generator** (enforced).
- **Position-swap** for pairwise comparisons: run (A,B) and (B,A); a "win" counts only if consistent across orders; inconsistent → tie.
- **Panel ≥3 judges**, aggregate absolute scores by **median**, pairwise by majority.
- **Blind to source:** strip system-identifying text before judging.
- **Inter-judge agreement** reported: **Krippendorff's α (ordinal)** per axis (primary); quadratic-weighted Cohen's κ for pairs. α ≥ 0.67 tentative, ≥ 0.80 good. Adds `krippendorff` dep.

**Judge prompt** (absolute): the `{field}`-parameterized template producing per-axis `{score, justification}` JSON.

---

## 8. Cross-Tournament (`bench/cross_tournament.py`)

Makes baseline/ablation variants comparable. Elo is only meaningful within one tournament, so to compare "full vs −evolution vs single-shot" we pool their top-N hypotheses into one shared tournament and let them play.

- Reuses `core/tournament.compute_elo_update` and the existing `RankingAgent` pairwise judging (with position-swap added per §7).
- Input: `{variant: [BenchHypothesis]}`. Output: a common-scale Elo per hypothesis → per-variant Elo distribution (mean/median/best).
- Uses the **judge** client for match adjudication (not the generator), so the cross-tournament is itself bias-controlled.

---

## 9. Single-Shot Baseline (`bench/baseline.py`) — Tier 3a

The floor control: does the multi-agent apparatus beat one LLM call?

- **`single_shot`**: one call to the generator model — "propose your single best novel testable hypothesis with rationale + validation plan" for the goal.
- **`best_of_32`**: sample 32 single-shot hypotheses, keep the best by judge score. Controls for "the gain is just more sampling" — the real bar is beating naive parallel sampling at matched budget.
- Both run through `bench/runner.py` as `variant="single_shot"` / `"best_of_32"`, producing `BenchHypothesis` lists fed into the cross-tournament (§8) and judge (§7).
- **Pass criterion:** the full system significantly beats `best_of_32` (beating only `single_shot` n=1 would mean we built an expensive sampler).

---

## 10. Ablations (`bench/ablation.py`) — Tier 3b

Quantify each agent's marginal contribution by leave-one-out.

- **Variants:** `full`, `no_evolution`, `no_meta_review`, `no_reflection` (clean weight-zeros), plus `no_tournament` (special — see below).
- **Implementation (weight-zero ablations):** `no_evolution`, `no_meta_review`, `no_reflection` each wrap `core/stats.compute_weights` and force the dropped agent's weight to `0.0` before `sample_agent_type`, so that agent is never dispatched. No agent code changes.
- **`no_tournament` (special case):** zeroing the RANKING weight would leave every hypothesis at the initial Elo 1200 (no updates), making the pool un-rankable and breaking the cross-tournament comparison. Instead, this variant runs generation/reflection/evolution normally but **replaces Elo ranking with a single-pass absolute judge-score sort**: the bench's judge (§7) scores each hypothesis once on the rubric, and `top-N` is taken by weighted rubric total instead of Elo. This isolates the contribution of the *tournament* mechanism specifically (iterative pairwise refinement vs. one-shot absolute scoring). The runner exposes a `ranking_mode: "elo" | "absolute"` flag for this.
- **Budget control:** hold total LLM-call budget fixed across variants (so "evolution helps" isn't confounded by "evolution just spent more compute"). Report both budget-matched and budget-free if time permits.
- **Comparison:** pool all variants' top-N into the cross-tournament (§8) for common-scale ΔElo, **and** score with the judge rubric (§7) for a run-independent Δ. Paired **Wilcoxon signed-rank** across goals; report effect size per agent.
- **Expected** (from the paper's framing): removing Evolution and Tournament should hurt most.

---

## 11. Scaling (`bench/scaling.py`) — Tier 1b

Shows quality rises with test-time compute.

- Partition each run's hypotheses into **10 temporal buckets** by `created_at` (equal-time slices by default; `--equal-count` flag for equal-count). Bucket 1 = earliest 10%, bucket 10 = latest.
- Per bucket: **best Elo** (max) and **top-10-avg Elo** — using each hypothesis's Elo **as of its bucket boundary** (replay from match history), not its final Elo, to avoid leaking future tournament info backward.
- Average across goals. Report Spearman ρ(bucket, Elo) per metric (expect ≈ +1) and OLS slope; "no saturation" = buckets 8–10 slope still positive.

---

## 12. Report (`bench/report.py`) + CLI (`bench/cli.py`)

`python -m bench <command>`:
- `concordance --dataset gpqa-bio --limit 25` → run + concordance stats + plot data.
- `scaling --goals comp_bio` → scaling curves.
- `judge --run <db>` → rubric scores + α.
- `baseline --goals comp_bio --limit 10` → full vs single-shot vs best-of-32.
- `ablation --goals comp_bio --limit 10` → leave-one-out ΔElo + Δrubric.
- `all` → run the full suite, emit one report.

`report.py` aggregates to:
- **JSON** (machine-readable: every statistic, per-bucket data, per-variant Elo/rubric, α).
- **Markdown** (human report: concordance verdict, scaling verdict, baseline table, ablation table, contamination/bias caveats, cost summary).

**Cost guardrails:** every command defaults to a small `--limit` (e.g. 25 questions / 10 goals) and the fast model; the report prints `n_llm_calls` and estimated cost. A full concordance run on all 198 GPQA-diamond questions is a real paid experiment and is gated behind an explicit `--full` flag with a printed cost warning.

---

## 13. Error Handling & Honesty

- **Contamination** (GPQA 2026) → logged caveat in every concordance report; never claimed as uncontaminated.
- **Self-preference bias** → judge≠generator enforced in code (raises on violation); same-family fallback prints a warning.
- **Elo non-comparability across runs** → the harness never compares raw Elo across separate tournaments; all cross-variant comparison goes through the cross-tournament or the judge anchor.
- **Small-N noise** → reports print N and CIs; the markdown explicitly labels small-sample results "directional, not significant."
- **Reference-baseline failure mode** → if blue ≈ red (Elo tracks difficulty but adds no quality), the report flags it as a *failure of the system*, not hidden.

---

## 14. Testing Strategy

All logic is unit-testable with fixtures/mocks — only real runs cost tokens (mirroring Kaimen's `fixtures/*.jsonl`).

- `test_bench_goldset.py` — token-subsequence matching: `PI3K-Akt`/`TGF-β`/alias hits and non-hits; recall@k math.
- `test_bench_concordance.py` — synthetic (elo, correct) rows with a known monotonic relationship → assert Spearman/Kendall/logistic come out as expected; reference-baseline buckets computed correctly; min-support filtering.
- `test_bench_scaling.py` — synthetic temporal hypotheses → assert 10-bucket partition (time and count modes), as-of-boundary Elo replay, monotonic curve detection.
- `test_bench_judge.py` — mock judge client returning canned JSON → assert rubric parsing, weighted total, judge≠generator enforcement (raises), position-swap consistency logic, Krippendorff α on a known matrix.
- `test_bench_cross_tournament.py` — fixed pairwise verdicts → assert common-scale Elo updates and per-variant aggregation.
- `test_bench_baseline.py` — mock runner → assert single_shot and best_of_32 produce the right variant records and feed the tournament.
- `test_bench_ablation.py` — assert each leave-one-out zeroes the right agent weight (via `compute_weights`), budget accounting, Wilcoxon on synthetic paired data.
- `test_bench_goalset.py` — jsonl loading, GPQA/ResearchBench loaders with a tiny fixture (no network: a local sample file).

Dataset loaders are tested against small local fixture files, not live HF downloads; a separate opt-in integration test (skipped by default) exercises the real HF pull.

---

## 15. Dependencies

Add to `pyproject.toml`: `scipy` (spearman/kendall), `statsmodels` (logistic GLM), `krippendorff` (inter-judge α), `datasets` (HF dataset loading), `pandas` (Parquet/ResearchBench). `sentence-transformers` already present (novelty embedding, if added later). The judge's cross-family path uses `openai` (optional, only if `OPENAI_API_KEY` set).

---

## 16. Verification (post-implementation)

- Unit suite green (all statistics/matching/aggregation logic).
- A small real run gated on the user's keys/subscription: `python -m bench concordance --dataset gpqa-bio --limit 10` → produces a report with a Spearman ρ and the blue/red curves; observe whether ρ is positive (the validity signal). This is the manual capstone — the first real evidence that *this* implementation's Elo tracks truth.
