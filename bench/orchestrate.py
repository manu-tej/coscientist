from __future__ import annotations

from bench.goalset import BenchGoal
from bench.runner import BenchRun
from bench.datasets.gpqa import score_answer
from bench.concordance import (
    ScoredHypothesis, concordance_stats, blue_minus_red_spread,
)
from bench.report import build_report, render_markdown, to_json


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


def _load_concordance_goals(dataset: str, limit: int) -> list[BenchGoal]:
    """Load the goal set for a concordance run. 'gpqa-bio' pulls the GPQA-diamond
    biology subset from HuggingFace; a tiny local fixture is the offline fallback."""
    from bench.datasets.gpqa import load_gpqa_hf, load_gpqa_fixture
    if dataset in ("gpqa-bio", "gpqa"):
        try:
            return load_gpqa_hf(all_subjects=(dataset == "gpqa"), limit=limit)
        except Exception as exc:  # network / HF unavailable → fixture
            print(f"  (HF pull failed: {exc}; using local fixture)")
            return load_gpqa_fixture("bench/datasets/fixtures/gpqa_sample.jsonl")[:limit]
    raise ValueError(f"Unknown concordance dataset: {dataset!r}")


async def run_concordance(args) -> dict:
    """Run the system on each goal, score answers vs gold, compute concordance.
    Returns the assembled report dict (also written to disk by the caller)."""
    import time
    from bench.runner import run_system
    from bench.manifest import system_version

    limit = 198 if getattr(args, "full", False) else args.limit
    max_tasks = getattr(args, "max_tasks", 30)
    min_support = getattr(args, "min_support", 2)
    goals = _load_concordance_goals(args.dataset, limit)
    print(f"Loaded {len(goals)} goals from '{args.dataset}'. "
          f"Running each with max_tasks={max_tasks} (seed={args.seed})...")

    runs = []
    for i, goal in enumerate(goals, 1):
        t0 = time.monotonic()
        run = await run_system(
            goal, variant="full", seed=args.seed,
            db_path=f"bench_runs/concordance_{goal.id}_{args.seed}.db",
            max_tasks=max_tasks, max_time_seconds=getattr(args, "max_time", 900),
        )
        elos = [h.elo_rating for h in run.hypotheses]
        spread = (max(elos) - min(elos)) if elos else 0.0
        print(f"  [{i}/{len(goals)}] {goal.id}: {len(run.hypotheses)} hyps, "
              f"Elo spread {spread:.0f}, {time.monotonic() - t0:.0f}s")
        runs.append(run)

    stats = concordance_from_runs(goals, runs, reference_accuracy={},
                                  min_support=min_support)
    return build_report({
        "concordance": stats,
        "meta": {"dataset": args.dataset, "n_questions": len(goals),
                 "system_version": system_version(), "max_tasks": max_tasks},
    })


async def run_command(args) -> int:
    """Top-level CLI dispatch. Cost-gates, then runs the requested command.
    v1 wires the 'concordance' command end-to-end; other tiers reuse the same
    captured runs (run-reuse, §17.2) and are added incrementally."""
    from pathlib import Path
    from bench.cost import estimate_cost
    from bench.cli import _confirm

    limit = 198 if getattr(args, "full", False) else args.limit
    est = estimate_cost(C=limit, a=0, v=1,
                        calls_per_run=getattr(args, "max_tasks", 30))
    if not _confirm(est, backend=args.backend, assume_yes=args.yes):
        print("Aborted before spending tokens.")
        return 1

    if args.command != "concordance":
        print(f"Command '{args.command}' not yet wired in v1; only 'concordance' "
              f"runs end-to-end. The other tiers reuse the captured run DBs.")
        return 0

    report = await run_concordance(args)
    out = getattr(args, "out", "bench_report")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}.json").write_text(to_json(report))
    Path(f"{out}.md").write_text(render_markdown(report))

    c = report["concordance"]
    v = report["verdicts"]
    print(f"\n=== Concordance ===")
    print(f"  rows={c['n_rows']}  buckets={c['n_buckets']}  "
          f"Spearman rho={c['spearman_rho']:.3f} (p={c['spearman_p']:.4f})  "
          f"top1_acc={c['top1_accuracy']:.3f}")
    print(f"  verdict: {'PASS — Elo tracks correctness' if v['concordance_pass'] else 'not established (small N or weak signal)'}")
    print(f"  report → {out}.json / {out}.md")
    return 0
