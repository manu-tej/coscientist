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


def rediscovery_from_runs(goals: list[BenchGoal], runs: list[BenchRun]) -> dict:
    """Score whether the system's hypotheses surface the known gold entities
    (gold-set recall), and whether higher-Elo hypotheses contain MORE of them
    (the properly-aligned concordance: Elo vs hypothesis quality, not MCQ answer)."""
    from bench.goldset import score_recall, entity_in_text
    from scipy.stats import spearmanr

    gold_map = {g.id: g.gold_entities for g in goals if g.gold_entities}
    per_goal: list[dict] = []
    elo_recall_rows: list[tuple[float, int]] = []  # (elo, #gold entities in that hypothesis)
    for run in runs:
        gold = gold_map.get(run.goal_id)
        if not gold:
            continue
        hyps = run.hypotheses
        top = sorted(hyps, key=lambda h: h.elo_rating, reverse=True)
        per_goal.append({
            "goal_id": run.goal_id,
            "n_gold": len(gold),
            "n_hyps": len(hyps),
            "pool_recall": score_recall(hyps, gold),       # surfaced anywhere
            "top1_recall": score_recall(top[:1], gold),    # in the single best hypothesis
            "top3_recall": score_recall(hyps, gold, k=3),  # in the top 3 by Elo
        })
        for h in hyps:
            blob = f"{h.summary} {h.text}"
            n_hit = sum(1 for e in gold if entity_in_text(e, blob))
            elo_recall_rows.append((h.elo_rating, n_hit))

    n = len(per_goal)
    elos = [r[0] for r in elo_recall_rows]
    if len(elo_recall_rows) >= 3 and len(set(elos)) > 1:
        rho, p = spearmanr(elos, [r[1] for r in elo_recall_rows])
    else:
        rho, p = float("nan"), float("nan")
    return {
        "per_goal": per_goal,
        "n_goals": n,
        "mean_pool_recall": sum(g["pool_recall"] for g in per_goal) / n if n else 0.0,
        "mean_top1_recall": sum(g["top1_recall"] for g in per_goal) / n if n else 0.0,
        "mean_top3_recall": sum(g["top3_recall"] for g in per_goal) / n if n else 0.0,
        "elo_recall_spearman": float(rho),   # >0 ⇒ higher-Elo hyps hold more gold biology
        "elo_recall_p": float(p),
        "n_hyps_scored": len(elo_recall_rows),
    }


async def run_rediscovery(args) -> dict:
    """Run the system on each curated goal, score gold-entity recall + Elo↔recall."""
    import time
    from bench.goalset import load_goalset
    from bench.runner import run_system
    from bench.manifest import system_version

    goals = load_goalset(args.goals)
    if not getattr(args, "full", False):
        goals = goals[:args.limit]
    max_tasks = getattr(args, "max_tasks", 30)
    print(f"Loaded {len(goals)} curated goals from '{args.goals}'. "
          f"Running each with max_tasks={max_tasks}...")
    runs = []
    for i, goal in enumerate(goals, 1):
        t0 = time.monotonic()
        try:
            run = await run_system(
                goal, variant="full", seed=args.seed,
                db_path=f"bench_runs/rediscover_{goal.id}_{args.seed}.db",
                max_tasks=max_tasks, max_time_seconds=getattr(args, "max_time", 900),
            )
        except Exception as exc:
            print(f"  [{i}/{len(goals)}] {goal.id}: FAILED ({type(exc).__name__}: {exc}); skipping")
            continue
        rec = None
        from bench.goldset import score_recall
        rec = score_recall(run.hypotheses, goal.gold_entities)
        print(f"  [{i}/{len(goals)}] {goal.id}: {len(run.hypotheses)} hyps, "
              f"pool recall {rec:.2f}, {time.monotonic() - t0:.0f}s")
        runs.append(run)

    stats = rediscovery_from_runs(goals, runs)
    return build_report({
        "rediscovery": stats,
        "meta": {"dataset": args.goals, "n_goals": len(goals),
                 "system_version": system_version(), "max_tasks": max_tasks},
    })


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
        try:
            run = await run_system(
                goal, variant="full", seed=args.seed,
                db_path=f"bench_runs/concordance_{goal.id}_{args.seed}.db",
                max_tasks=max_tasks, max_time_seconds=getattr(args, "max_time", 900),
            )
        except Exception as exc:  # one bad goal must not sink the whole reading
            print(f"  [{i}/{len(goals)}] {goal.id}: FAILED ({type(exc).__name__}: {exc}); skipping")
            continue
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

    out = getattr(args, "out", "bench_report")
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    if args.command == "concordance":
        report = await run_concordance(args)
        Path(f"{out}.json").write_text(to_json(report))
        c = report["concordance"]; v = report["verdicts"]
        print(f"\n=== Concordance ===")
        print(f"  rows={c['n_rows']}  buckets={c['n_buckets']}  "
              f"Spearman rho={c['spearman_rho']:.3f} (p={c['spearman_p']:.4f})  "
              f"top1_acc={c['top1_accuracy']:.3f}")
        print(f"  verdict: {'PASS — Elo tracks correctness' if v['concordance_pass'] else 'not established (small N or weak signal)'}")
        print(f"  report → {out}.json")
        return 0

    if args.command == "rediscover":
        report = await run_rediscovery(args)
        Path(f"{out}.json").write_text(to_json(report))
        r = report["rediscovery"]
        print(f"\n=== Rediscovery (gold-entity recall) ===")
        for g in r["per_goal"]:
            print(f"  {g['goal_id']}: pool {g['pool_recall']:.2f}  top1 {g['top1_recall']:.2f}  "
                  f"top3 {g['top3_recall']:.2f}  ({g['n_hyps']} hyps, {g['n_gold']} gold)")
        print(f"  MEAN pool recall={r['mean_pool_recall']:.2f}  top1={r['mean_top1_recall']:.2f}  top3={r['mean_top3_recall']:.2f}")
        print(f"  Elo↔recall Spearman={r['elo_recall_spearman']:.3f} (p={r['elo_recall_p']:.3f}, n={r['n_hyps_scored']}) "
              f"— >0 means higher-Elo hypotheses hold more gold biology")
        print(f"  report → {out}.json")
        return 0

    print(f"Command '{args.command}' not yet wired in v1; only 'concordance' and "
          f"'rediscover' run end-to-end.")
    return 0
