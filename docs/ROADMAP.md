# Roadmap: from working prototype to a real result

Status as of 2026-09-25; the project is now paused (see the README). The
system runs end to end, free and offline (`--provider offline`) and live
(`--provider claude-code`); the README's Verification section gives the current
test count. What the project does **not** yet have is an answer to its own
central question.

## The question this repo exists to answer

> Do tournament Elo rankings track ground-truth scientific correctness?

The README reports the current answer honestly: **concordance was not
established.** Until 2026-09-25 that rested on a Spearman ρ = −0.56, p = 0.32
that correlated **5 bucket midpoints**, not 41 responses: on n = 5 nothing
short of near-perfect monotonicity is significant, and it said nothing about
the responses themselves.

The headline is a conditional logistic regression `correct ~ Elo`
stratified by tournament (`bench/concordance.py`,
`_conditional_logistic_correct_on_elo`): every run on a question is its own
tournament, so only the within-tournament effect is the claim and question
difficulty is conditioned out. The fit uses a stratum only when both correctness
and Elo vary within it.

An offline audit of the 17 retained run databases in the gitignored
`bench_runs/` (`python -m bench.tournament_audit`, 2026-09-25, no model calls)
showed that the strata behind the numbers reported before that date (−1.85 per
SD, p = 0.076 on six tournaments; −0.80, p = 0.083 on all 17) were not what
the fit assumed. Eight tournaments are a true ceiling: every generated
hypothesis, rejected ones included, chose the gold letter. One more is a ceiling by selection: review rejected exactly its two
wrong answers and the benchmark reads active rows only. One more has every
active Elo tied because its only match was between hypotheses later rejected
(two of the eight ceilings are tied the same way). Three
databases persisted hypotheses with empty text, a defect of the 2026-06-04 code
(the backends had just started returning "" on an empty stop, the parser did not
check, the judge defaulted unparseable matches to slot one); those rows score as
wrong and manufactured two "informative" strata. One stratum passes the test
only through two parser misses. Two pass on a single match between hypotheses
of the same correctness. That leaves one clean tournament, and a fit needs two.
The numbers are withdrawn; the README carries the audit table. The databases
are not in this repository, so none of the figures in this document can be
recomputed from a clone (Stage 2, item 1). Rescoring all 171 hypothesis rows
with the answer parser as corrected on 2026-11-11 changes none of them, so the
figures stand as computed.

So the honest reading is still: *we do not know*, and the reason is now
precise. The June runs cannot answer the question under any exclusion rule, and
four measurement defects would carry into a new run:

1. `bench/datasets/gpqa.py` `parse_mcq_answer` ignores `\boxed{X}`,
   markdown around the letter, and "answer is Option (X)": its answer
   patterns do not match that phrasing, so the text falls through to the
   generic option pattern, where the last mention wins. The two June misses
   are exactly these. One hypothesis states B as "The correct answer is
   Option (B)" and as `\boxed{B}`, and scores as D from a later "Option D is
   excluded"; the other states B as "Answer **(B)**", "Answer: **B**" and
   `\boxed{B}`, and scores as C from a later "ruling out option (C)". The
   article bug fixed on 2026-11-11 (reading "the answer is a ..." as
   option A) played no part in them.
2. `bench/runner.py` `read_run` scores active hypotheses only, although rejected
   ones were ranked before rejection. Including them moves the (withdrawn) fit
   from p = 0.083 to 0.055, still anti-concordant; the policy must be chosen
   and pre-registered, not discovered.
3. Binary correctness on a question the model answers correctly every time
   carries zero information. Question selection must be gated on the reference
   accuracy from `bench/reference.py` (well inside 0 and 1) before a tournament
   is paid for. This is also the missing input to the power calculation.
4. The judge prefers longer text (60 of 79 non-empty matches went to the longer
   side). The report should print the within-tournament Elo-vs-length
   correlation next to the verdict so verbosity cannot pass as concordance.

Power is a measurement problem before it is a sample-size problem.

## Stage 0 — Reliability (open; the project is paused)

Live running surfaced defects invisible to the offline suite, all now fixed:
markdown-decorated field labels (config and agent parsers), a prompt/parser
contract mismatch in `debate`/`assumptions`, a turn loop that quit on the *word*
rather than the *work*, and a timeout path that discarded results already paid
for.

**Exit criterion:** two consecutive live runs with zero agent task failures.

Why it gates everything downstream: `bench` fails a goal by default if any agent
task errors. Benchmarking while generation tasks fail burns budget on failed
goals and produces nothing.

## Stage 1 — Free continuous verification

Put `--provider offline` in CI. Every push runs the full suite plus both
deterministic smokes with no credentials and no spend.

This is cheap and high-leverage: the offline provider exists precisely so the
plumbing can be verified continuously, leaving paid runs for questions only real
models can answer.

**Caveat to keep visible:** offline runs prove the machinery works and prove
nothing about scientific quality. Stage 0's defects are the proof: the offline
suite passed while live generation tasks failed. CI must never be cited as
evidence the science works.

## Stage 2 — Design the experiment before spending on it

Before any large paid run, settle the methodology. Item 1 costs nothing.

1. **Done 2026-09-25.** The headline is the conditional logistic regression
   described above; `bench/report.py` keys the verdict on it (positive slope,
   LRT p < 0.05) and demotes the bucketed Spearman to a descriptive.
   `bench/tournament_audit.py` classifies every run database on both axes of
   the fit's informativeness test and refits under the exclusions the June
   defects call for; run it on any new batch before reading the verdict. The
   original report (`bench_runs/reading2.json`) and the 17 run databases are
   kept locally by the author and gitignored, so a fresh clone cannot reproduce the
   README table. The RAW section of `calibrate_concordance.py` rebuilds the
   rows from all 17 databases and prints the all-17 row; the 41-row reading is
   the six `*_2` databases. After the RAW section the script goes straight on to
   paid reference sampling (16 fast-model samples per question on the
   configured provider) without asking for confirmation.
2. **Do a power calculation.** For a plausible true effect at 80% power, derive
   the required number of graded *responses*, then the question count implied
   by the observed responses-per-question rate. Do this before choosing
   `--limit`.
3. **Pre-register the analysis.** Write the statistical plan into
   `PUBLICATION.md` before the run. The repo already reports negative results
   honestly; pre-registration is what makes that credible rather than lucky.
4. **Keep the difficulty-corrected baseline.** `bench/reference.py` already
   samples the base model per question. A tournament that cannot beat base
   sampling has not earned its cost.

## Stage 3 — The decisive experiment

The ablation is already built (`--ranking-mode absolute`, landed in `de1e708`):
score each hypothesis once on a rubric with a *distinct* judge model, no pairwise
tournament. This isolates the actual claim.

| Arm | What it tests |
|---|---|
| Tournament Elo | the repo's core mechanism |
| Absolute rubric scoring | does pairwise debate add anything over direct scoring? |
| Difficulty-corrected base sampling | does the whole scaffold beat the raw model? |

The three arms cannot yet share one set of paid runs. Absolute rubric scoring
runs inside the system run, and the ranking mode and judge settings are part of
the run-cache identity (`BenchmarkExecution.as_dict` in `bench/runner.py`,
hashed by `bench/manifest.py`), so an absolute-ranking run is a separate paid
run, and Elo concordance refuses absolute-ranking runs (`bench/orchestrate.py`).
Comparing the arms on one frozen candidate set needs harness work first:
generate the candidates once, then rank that same set both ways. Base sampling
(`bench/reference.py`) makes fresh calls in any case.

**Every outcome is publishable.** "The tournament adds nothing over rubric
scoring" is a genuinely useful finding, and the harness was built to catch
exactly that.

## Stage 4 — Decide what this is

Fork in the road, to be taken on evidence rather than sunk cost:

- **Concordance holds** → the ranking mechanism works; write it up, and the
  `company/` portfolio layer becomes the natural application. The external
  benchmark for that layer is now fixed: the Virtual Biotech (Zou lab, *Science*,
  2026-09-17; see `REFERENCES.md`) has a similar CSO-over-specialists design and
  was used to annotate and analyze outcomes of ~56k clinical trials. A
  comparable trial-outcome analysis on this architecture, with the
  pre-registered plan from Stage 2, is the target — not a larger agent count.
- **Concordance fails, robustly and with adequate power** → that is a real
  negative result about LLM tournament ranking, worth reporting. Publishing it
  is more valuable than another scaffold.
- **Underpowered again** → stop running benchmarks and fix the measurement
  first.

## What is deliberately not on this list

- Scaling workers or adding agents. Throughput is not the bottleneck; knowing
  whether the output is any good is.
- New generation strategies. Four exist and two were structurally broken until 2026-08-09.
- The GPU/Modal surface (`--structure`, `--affinity`). Optional, billed, and
  irrelevant until Stage 3 answers the ranking question.
