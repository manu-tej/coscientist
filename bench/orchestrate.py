from __future__ import annotations

from bench.goalset import BenchGoal
from bench.runner import BenchRun
from bench.datasets.gpqa import score_answer
from bench.concordance import (
    ScoredHypothesis, concordance_stats, blue_minus_red_spread,
)


def concordance_from_runs(
    goals: list[BenchGoal], runs: list[BenchRun],
    reference_accuracy: dict[str, float],
    bin_width: int = 50, min_support: int = 5,
) -> dict:
    """Score every hypothesis's parsed answer against its goal's gold_answer,
    then compute the full concordance statistics + blue−red spread."""
    gold = {g.id: g.gold_answer for g in goals if g.gold_answer}
    rows: list[ScoredHypothesis] = []
    for run in runs:
        ga = gold.get(run.goal_id)
        if ga is None:
            continue
        for h in run.hypotheses:
            rows.append(ScoredHypothesis(
                elo=h.elo_rating, correct=score_answer(h.text, ga),
                question_id=run.goal_id))
    stats = concordance_stats(rows, bin_width, min_support)
    stats["blue_minus_red"] = blue_minus_red_spread(
        rows, reference_accuracy, bin_width, min_support, n_boot=10000, seed=0)
    return stats


async def run_command(args) -> int:
    """Top-level CLI dispatch. Imports the model stack lazily; for the heavy
    path this would: load goals → manifest check → run_system (or reuse) →
    compute tier metrics → write report. v1 wires concordance end-to-end; other
    commands assemble from the same captured runs (run-reuse, §17.2)."""
    from bench.cost import estimate_cost
    from bench.cli import _confirm

    # Cost gate (all token-spending commands)
    limit = args.limit if not getattr(args, "full", False) else 198
    est = estimate_cost(C=limit, a=0, v=1, calls_per_run=100)
    if not _confirm(est, backend=args.backend, assume_yes=args.yes):
        print("Aborted before spending tokens.")
        return 1

    # NOTE: the heavy run loop (runner.run_system per goal, manifest reuse) is the
    # one part requiring API keys; it is exercised by the manual capstone in §16,
    # not in CI. The pure-analysis helpers above are fully unit-tested.
    print("Run loop requires configured backend + keys; see spec §16 for the "
          "manual capstone command.")
    return 0
