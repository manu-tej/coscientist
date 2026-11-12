from __future__ import annotations

import argparse
import asyncio

from bench.cost import format_estimate


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be an integer >= 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be an integer >= 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench", description="AI co-scientist evaluation harness")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--limit", type=_positive_int, default=25,
                        help="max goals/questions (cost guard)")
        sp.add_argument("--full", action="store_true", help="ignore --limit; run the full set (paid)")
        sp.add_argument("--yes", action="store_true", help="skip the cost-estimate confirmation")
        sp.add_argument("--seed", type=_nonnegative_int, default=0)
        sp.add_argument(
            "--backend",
            choices=["auto", "api", "subscription", "custom"],
            default="auto",
            help="billing/cost label only; auto derives it from the effective provider",
        )
        sp.add_argument("--out", default="bench_report", help="output path prefix")
        sp.add_argument("--max-tasks", dest="max_tasks", type=_positive_int, default=30,
                        help="Supervisor tasks per goal (higher = more search, more cost)")
        sp.add_argument("--min-support", dest="min_support", type=_positive_int, default=2,
                        help="min hypotheses per ranking bucket")
        sp.add_argument("--max-time", dest="max_time", type=_positive_int, default=900,
                        help="per-goal wall-clock cap (seconds)")
        sp.add_argument("--workers", type=_positive_int, default=None,
                        help="concurrent Supervisor agents per goal (default: config.yaml)")
        sp.add_argument("--provider", default=None,
                        help="LLM provider override (e.g. claude-code, anthropic, codex, "
                             "openai-compatible)")
        sp.add_argument("--model-fast", dest="model_fast", default=None,
                        help="fast-tier model override for the selected provider")
        sp.add_argument("--model-strong", dest="model_strong", default=None,
                        help="strong-tier model override for the selected provider")
        sp.add_argument("--judge-provider", dest="judge_provider", default=None,
                        help="separate LLM provider for absolute-ranking benchmarks")
        sp.add_argument("--judge-model", dest="judge_model", default=None,
                        help="separate model ID for absolute-ranking benchmarks")
        sp.add_argument("--allow-partial", action="store_true",
                        help="emit a report and exit zero even when some goals fail")
        sp.add_argument(
            "--manifest",
            default="bench/manifest.sqlite",
            help="completed-run cache manifest (shared across compatible commands/cells)",
        )
        sp.add_argument(
            "--no-cache",
            action="store_true",
            help="run every goal fresh and do not read or write the manifest",
        )

    c = sub.add_parser("concordance"); common(c)
    c.add_argument("--dataset", default="gpqa-bio")
    c.add_argument("--reference-samples", dest="reference_samples",
                   type=_nonnegative_int, default=0,
                   help="base-model samples/question for difficulty correction "
                        "(paper used 32; 0 = off). Adds ~N×#questions extra calls.")
    c.add_argument("--reference-strong", dest="reference_strong", action="store_true",
                   help="sample the reference from the strong model (default: fast, "
                        "matching the Generation agent's tier).")
    c.add_argument(
        "--allow-fixture-fallback", action="store_true",
        help="if the live GPQA pull fails, explicitly permit the tiny local fixture",
    )

    rd = sub.add_parser("rediscover"); common(rd)
    rd.add_argument("--goals", default="bench/datasets/biomnibench_goldset.jsonl",
                    help="jsonl of curated goals with gold_entities")
    rd.add_argument(
        "--ranking-mode",
        choices=["elo", "absolute"],
        default="elo",
        help="internal ranking mechanism; absolute is the no-tournament ablation",
    )

    sw = sub.add_parser("sweep", help="benchmark a model × worker-count × seed grid")
    common(sw)
    sw.add_argument("--benchmark", choices=["rediscover", "concordance"],
                    default="rediscover")
    sw.add_argument("--worker-counts", default="1,2,4",
                    help="comma-separated Supervisor worker counts")
    sw.add_argument("--models", action="append", default=[],
                    metavar="PROVIDER=FAST[,STRONG]",
                    help="model cell; repeat for multiple providers/models; '=' keeps "
                         "colons inside local model IDs (legacy provider:fast:strong works)")
    sw.add_argument("--seeds", default="0",
                    help="comma-separated paired seeds")
    sw.add_argument("--goals", default="bench/datasets/biomnibench_goldset.jsonl")
    sw.add_argument("--dataset", default="gpqa-bio")
    sw.add_argument("--allow-fixture-fallback", action="store_true")
    sw.add_argument("--reference-samples", dest="reference_samples",
                    type=_nonnegative_int, default=0)
    sw.add_argument(
        "--ranking-mode",
        choices=["elo", "absolute"],
        default="elo",
        help="ranking mechanism for rediscovery cells; concordance requires Elo",
    )
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
    from bench.orchestrate import run_command
    return await run_command(args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
