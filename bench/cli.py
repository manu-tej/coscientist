from __future__ import annotations

import argparse
import asyncio

from bench.cost import format_estimate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench", description="AI co-scientist evaluation harness")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--limit", type=int, default=25, help="max goals/questions (cost guard)")
        sp.add_argument("--full", action="store_true", help="ignore --limit; run the full set (paid)")
        sp.add_argument("--yes", action="store_true", help="skip the cost-estimate confirmation")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--backend", choices=["api", "subscription"], default="api")
        sp.add_argument("--out", default="bench_report", help="output path prefix")
        sp.add_argument("--max-tasks", dest="max_tasks", type=int, default=30,
                        help="Supervisor tasks per goal (higher = more Elo spread, more cost)")
        sp.add_argument("--min-support", dest="min_support", type=int, default=2,
                        help="min hypotheses per Elo bucket")
        sp.add_argument("--max-time", dest="max_time", type=int, default=900,
                        help="per-goal wall-clock cap (seconds)")

    c = sub.add_parser("concordance"); common(c)
    c.add_argument("--dataset", default="gpqa-bio")

    rd = sub.add_parser("rediscover"); common(rd)
    rd.add_argument("--goals", default="bench/datasets/biomnibench_goldset.jsonl",
                    help="jsonl of curated goals with gold_entities")

    s = sub.add_parser("scaling"); common(s)
    s.add_argument("--goals", default="comp_bio")

    j = sub.add_parser("judge"); common(j)
    j.add_argument("--run", required=True, help="path to a captured run SQLite")

    b = sub.add_parser("baseline"); common(b)
    b.add_argument("--goals", default="comp_bio")

    a = sub.add_parser("ablation"); common(a)
    a.add_argument("--goals", default="comp_bio")

    al = sub.add_parser("all"); common(al)
    return p


def _confirm(est: dict, backend: str, assume_yes: bool) -> bool:
    print(format_estimate(est, backend=backend))
    if assume_yes:
        return True
    try:
        return input("Proceed? [y/N] ").strip().lower() == "y"
    except EOFError:
        return False


async def _run(args) -> int:
    from bench.orchestrate import run_command   # Task E4 supplies this
    return await run_command(args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
