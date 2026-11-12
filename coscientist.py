"""Top-level entrypoint: wires config, agents, supervisor, and UI.

Usage:
    python coscientist.py "Your research goal here" --workers 8
"""
import argparse
import asyncio
import uuid
import logging
import os
import re
import sys
from pathlib import Path

from tools.llm import PROVIDERS, LLMBackendError
from core.state import StateStore
from core.config_parser import ConfigParser
from core.orchestrator import AgentRunner
from core.supervisor import Supervisor, SupervisorSettings
from core.stats import WeightThresholds
from agents.base import AgentOutputError, BaseAgent
from agents.generation import GenerationAgent
from agents.reflection import ReflectionAgent
from agents.ranking import RankingAgent
from agents.evolution import EvolutionAgent
from agents.meta_review import MetaReviewAgent
from tools.verify import ClaimVerifier


def load_config(path: str = "config.yaml") -> dict:
    from tools.resources import load_yaml_config
    return load_yaml_config(path)


def _first_prose_line(text: str, *, joined: bool = False) -> str:
    """Return the first line of real prose, skipping structural markers.

    Model output is wrapped in section markers (`HYPOTHESIS`) and `Label:` prefixes;
    printing line one verbatim usually just shows a marker.
    """
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    prose = [
        ln for ln in lines
        if " " in ln and not ln.isupper() and not ln.endswith(":")
    ]
    if not prose:
        return ""
    if joined:
        return " ".join(" ".join(prose).split())
    # Drop a leading `Label: ` so the sentence itself is what shows.
    first = prose[0]
    return first.split(": ", 1)[1] if ": " in first[:24] else first


_QUARANTINE_EXPLANATION = (
    "a claim was contradicted or a citation could not be traced to a retrieved "
    "record. See the claim_verdicts table."
)


async def _print_results(store, config, db_path: str, top_n: int = 5) -> None:
    """Surface what the run actually produced.

    Without this a successful run prints only a task count, leaving the user to open
    SQLite by hand to see whether anything was generated. `list_hypotheses` defaults
    to status='active', so a naive read here silently hides every quarantined
    hypothesis: CRITICAL 2 found a real offline run that generated 4 hypotheses,
    quarantined 3, and printed "Top 1 hypotheses by Elo" with no indication that 3
    were withheld or why -- and, when every hypothesis is quarantined, printed the
    flatly false "No hypotheses were produced." Both branches below must name the
    quarantined count and cause instead of staying silent about it.
    """
    hypotheses = await store.list_hypotheses(config.run_id)
    quarantined = await store.list_hypotheses(config.run_id, status="quarantined")
    if not hypotheses:
        if quarantined:
            print(
                f"\nNo active hypotheses — all {len(quarantined)} hypothesis(es) "
                f"generated were quarantined — {_QUARANTINE_EXPLANATION}"
            )
        else:
            print("\nNo hypotheses were produced.")
        return
    ranked = sorted(hypotheses, key=lambda h: h.elo_rating, reverse=True)[:top_n]
    print(f"\nTop {len(ranked)} hypotheses by Elo:")
    for i, h in enumerate(ranked, 1):
        print(f"  {i}. [{h.elo_rating:7.1f}] {h.summary}")
        detail = _first_prose_line(h.text)
        if detail:
            print(f"     {detail[:160]}")
    if quarantined:
        print(f"\n{len(quarantined)} hypothesis(es) quarantined — {_QUARANTINE_EXPLANATION}")
    meta = await store.get_latest_meta_review(config.run_id)
    if meta and meta.get("research_overview"):
        raw = str(meta["research_overview"])
        labelled = re.search(r"^research overview\s*:\s*(.+)$", raw,
                             re.IGNORECASE | re.MULTILINE)
        overview = (labelled.group(1).strip() if labelled
                    else _first_prose_line(raw, joined=True))
        if overview:
            print(f"\nResearch overview: {overview[:400]}")
    print(f"\nView full run: uv run python -m ui.launch {db_path}")


async def main(
    goal: str,
    *,
    workers: int | None = None,
    max_tasks: int = 100,
    max_time_seconds: int | None = None,
    provider: str | None = None,
    model_fast: str | None = None,
    model_strong: str | None = None,
    seed: int = 0,
    db_path: str | None = None,
    quiet: bool = False,
) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    from tools.llm import configure_backend, load_backend_environment
    load_backend_environment()
    cfg = load_config()
    selected_provider, _ = configure_backend(
        cfg,
        provider=provider,
        model_fast=model_fast,
        model_strong=model_strong,
    )
    if db_path is not None:
        cfg["db_path"] = db_path
    run_id = str(uuid.uuid4())[:8]

    store = StateStore(cfg["db_path"])
    await store.init_db()

    from tools.llm import close_backend, make_backend
    client = make_backend(cfg, provider_override=selected_provider)
    try:
        prompts_dir = Path("prompts")
        base = BaseAgent(client=client, prompts_dir=prompts_dir)

        parser = ConfigParser(client=client, prompts_dir=prompts_dir)
        config = await parser.parse_and_review(run_id=run_id, goal=goal)
        if config is None:
            print("Research goal rejected by safety review. Aborting.")
            return 2
        await store.save_config(config)
        print(f"Run {run_id} — parsed config: attributes={config.attributes}")

        from tools.evidence import build_sources, LiteratureTool

        search = LiteratureTool(
            sources=build_sources(
                selected_provider,
                cfg["tools"]["max_search_results"],
                os.environ.get("TAVILY_API_KEY", ""),
            ),
            max_results=cfg["tools"]["max_search_results"],
        )
        # Sentence-transformers is an optional heavyweight runtime dependency.
        # Keep it off the CLI startup path (especially ``--help``) and let the
        # proximity agent load/cache the model on its first real task.
        from agents.proximity import ProximityAgent

        runner = AgentRunner(
            store=store, config=config,
            generation=GenerationAgent(base=base),
            reflection=ReflectionAgent(base=base),
            ranking=RankingAgent(
                base=base, store=store,
                elo_k=cfg["tournament"]["elo_k_factor"],
                multi_turn_threshold=cfg["tournament"]["multi_turn_threshold"],
            ),
            proximity=ProximityAgent(
                store=store,
                model_name=cfg["proximity"]["model"],
                similarity_threshold=cfg["proximity"]["similarity_threshold"],
                duplicate_threshold=cfg["proximity"]["duplicate_threshold"],
            ),
            evolution=EvolutionAgent(base=base),
            meta_review=MetaReviewAgent(base=base),
            search=search,
            verifier=ClaimVerifier(
                max_claims=cfg.get("evidence", {}).get("max_claims", 4),
                concurrency=cfg.get("evidence", {}).get("concurrency", 2),
            ),
        )

        settings = SupervisorSettings(
            n_workers=workers if workers is not None else cfg["supervisor"]["n_workers"],
            max_tasks=max_tasks,
            max_time_seconds=(
                max_time_seconds
                if max_time_seconds is not None
                else cfg["supervisor"]["max_time_minutes"] * 60
            ),
            checkpoint_interval=cfg["supervisor"]["checkpoint_interval"],
            thresholds=WeightThresholds(
                min_hypothesis_count=cfg["supervisor"]["min_hypothesis_count"],
                elo_variance_threshold=cfg["supervisor"]["elo_variance_threshold"],
                meta_review_interval=cfg["supervisor"]["meta_review_interval"],
            ),
            seed=seed,
        )
        supervisor = Supervisor(
            config=config, runner=runner, settings=settings, store=store,
        )
        timed_out = None
        try:
            await supervisor.run()
        except TimeoutError as exc:
            # Hypotheses already persisted were paid for; surface them rather than
            # discarding the run because the wall clock expired during shutdown.
            timed_out = exc

        status = 0
        if timed_out is not None:
            print(f"Run {run_id} TIMED OUT — {timed_out}. DB: {cfg['db_path']}")
            status = 3
        elif supervisor.task_failures:
            print(
                f"Run {run_id} FAILED with {len(supervisor.task_failures)} "
                f"agent task error(s). DB: {cfg['db_path']}"
            )
            status = 2
        else:
            print(
                f"Run {run_id} complete — {sum(supervisor.task_counts.values())} tasks "
                f"across {settings.n_workers} workers. DB: {cfg['db_path']}"
            )
        if supervisor.task_failures and timed_out is None:
            return status
        if not quiet:
            await _print_results(store, config, cfg["db_path"])
        return status
    finally:
        await close_backend(client)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be an integer >= 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the model-portable hierarchical co-scientist core",
    )
    parser.add_argument("goal")
    parser.add_argument("--workers", type=_positive_int, default=None)
    parser.add_argument("--max-tasks", type=_positive_int, default=100)
    parser.add_argument("--max-time", type=_positive_int, default=None,
                        help="run wall-clock cap in seconds")
    parser.add_argument(
        "--provider", default=None, choices=sorted(PROVIDERS),
        help="model backend; 'offline' is a free deterministic fixture (no network, "
             "no credentials, no scientific validity)",
    )
    parser.add_argument("--model-fast", default=None)
    parser.add_argument("--model-strong", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--db", dest="db_path", default=None)
    parser.add_argument(
        "--quiet", action="store_true",
        help="suppress the end-of-run results summary",
    )
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(main(
            args.goal,
            workers=args.workers,
            max_tasks=args.max_tasks,
            max_time_seconds=args.max_time,
            provider=args.provider,
            model_fast=args.model_fast,
            model_strong=args.model_strong,
            seed=args.seed,
            db_path=args.db_path,
            quiet=args.quiet,
        ))
    except (LLMBackendError, AgentOutputError) as exc:
        # The message is already actionable; a traceback around it just reads as a
        # crash. company/cli.py already uses this convention.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except TimeoutError as exc:
        print(
            f"error: {exc}\nPartial results (if any) are in the run DB; raise "
            f"--max-time or lower --max-tasks.", file=sys.stderr,
        )
        return 3
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(cli())
