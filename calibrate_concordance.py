"""Calibration: does the per-question difficulty correction change the GPQA
concordance picture? Reuses already-captured concordance runs (free); only pays
for reference sampling (16 fast-model samples per distinct question).
"""
import asyncio
import glob
import os
import re
import sqlite3

from dotenv import load_dotenv
load_dotenv()  # pick up COSCIENTIST_PROVIDER=claude-oauth + token from .env

from bench.runner import read_run, _load_yaml_config
from bench.datasets.gpqa import load_gpqa_hf, score_answer
from bench.concordance import per_bucket_accuracy, reference_per_bucket, ScoredHypothesis
from bench.orchestrate import concordance_from_runs
from bench.reference import reference_accuracy_for_goals, backend_generate
from tools.llm import make_backend

BIN = 100
MIN_SUPPORT = 2


async def main():
    gold_goals = {g.id: g for g in load_gpqa_hf(all_subjects=True)}

    runs = []
    for db in sorted(glob.glob("bench_runs/concordance_gpqa-biology-*.db")):
        tag = os.path.basename(db)[:-3]
        m = re.match(r"concordance_(gpqa-biology-\d+)_\d+", tag)
        if not m:
            continue
        qid = m.group(1)
        conn = sqlite3.connect(db); conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT run_id FROM configs ORDER BY created_at DESC LIMIT 1").fetchone()
        conn.close()
        if not row:
            continue
        run = await read_run(db, row["run_id"], qid, "full", 0, 0.0)
        if run.hypotheses:
            runs.append(run)

    qids = sorted({r.goal_id for r in runs})
    goals = [gold_goals[q] for q in qids if q in gold_goals]
    n_hyps = sum(len(r.hypotheses) for r in runs)
    print(f"Loaded {len(runs)} captured runs · {len(goals)} distinct questions · "
          f"{n_hyps} hypotheses (bin_width={BIN}, min_support={MIN_SUPPORT})\n")

    raw = concordance_from_runs(goals, runs, reference_accuracy={},
                                min_support=MIN_SUPPORT, bin_width=BIN)
    print("=== RAW concordance (no difficulty correction) ===")
    print(f"  Spearman rho = {raw['spearman_rho']:.3f}  (p={raw['spearman_p']:.3f})")
    print(f"  Kendall tau  = {raw['kendall_tau']:.3f}")
    print(f"  top-1 acc    = {raw['top1_accuracy']:.3f}")
    print(f"  buckets      = {raw['n_buckets']}  ·  rows = {raw['n_rows']}\n")

    print("=== Sampling reference (16 fast-model samples / question) ===")
    backend = make_backend(_load_yaml_config("config.yaml"))
    ref = await reference_accuracy_for_goals(
        goals, backend_generate(backend, use_strong=False), n_samples=16, concurrency=6)
    print("  per-question base accuracy:",
          {k: round(v, 2) for k, v in sorted(ref.items())}, "\n")

    full = concordance_from_runs(goals, runs, reference_accuracy=ref,
                                 min_support=MIN_SUPPORT, bin_width=BIN)
    bmr = full["blue_minus_red"]
    print("=== Difficulty-corrected (blue - red) ===")
    print(f"  mean spread = {bmr['mean_spread']:.3f}  "
          f"95% CI [{bmr['ci_low']:.3f}, {bmr['ci_high']:.3f}]  buckets={bmr['n_buckets']}")
    verdict = ("POSITIVE — system beats base difficulty" if bmr["ci_low"] > 0 else
               "not established (CI includes 0)")
    print(f"  verdict: {verdict}\n")

    rows = []
    goldmap = {g.id: g.gold_answer for g in goals}
    for run in runs:
        ga = goldmap.get(run.goal_id)
        if not ga:
            continue
        for h in run.hypotheses:
            rows.append(ScoredHypothesis(
                elo=h.elo_rating, correct=score_answer(h.text, ga), question_id=run.goal_id))
    blue = per_bucket_accuracy(rows, BIN, MIN_SUPPORT)
    red = reference_per_bucket(rows, ref, BIN, MIN_SUPPORT)
    print("=== Per-Elo-bucket: blue (system) vs red (base reference) ===")
    print(f"  {'Elo>=':>6} | {'blue':>5} | {'red':>5} | {'blue-red':>8}")
    for f in sorted(set(blue) | set(red)):
        b, r = blue.get(f), red.get(f)
        bd = None if (b is None or r is None) else round(b - r, 2)
        print(f"  {f:>6} | {('-' if b is None else round(b,2)):>5} | "
              f"{('-' if r is None else round(r,2)):>5} | {str(bd):>8}")


asyncio.run(main())
