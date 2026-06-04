"""Top-level entrypoint: wires config, agents, supervisor, and UI.

Usage:
    python coscientist.py "Your research goal here"
"""
import asyncio
import sys
import uuid
import logging
import os
from pathlib import Path

import yaml

from core.state import StateStore
from core.config_parser import ConfigParser
from core.orchestrator import AgentRunner
from core.supervisor import Supervisor, SupervisorSettings
from core.stats import WeightThresholds
from tools.search import SearchTool
from agents.base import BaseAgent
from agents.generation import GenerationAgent
from agents.reflection import ReflectionAgent
from agents.ranking import RankingAgent
from agents.proximity import ProximityAgent
from agents.evolution import EvolutionAgent
from agents.meta_review import MetaReviewAgent


def load_config(path: str = "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


async def main(goal: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    try:
        from dotenv import load_dotenv
        load_dotenv()  # pick up .env (provider + OAuth token) before building the backend
    except ImportError:
        pass
    cfg = load_config()
    run_id = str(uuid.uuid4())[:8]

    store = StateStore(cfg["db_path"])
    await store.init_db()

    from tools.llm import make_backend
    client = make_backend(cfg)
    prompts_dir = Path("prompts")
    base = BaseAgent(client=client, prompts_dir=prompts_dir)

    parser = ConfigParser(client=client, prompts_dir=prompts_dir)
    config = await parser.parse_and_review(run_id=run_id, goal=goal)
    if config is None:
        print("Research goal rejected by safety review. Aborting.")
        return
    await store.save_config(config)
    print(f"Run {run_id} — parsed config: attributes={config.attributes}")

    search = SearchTool(
        api_key=os.environ.get("TAVILY_API_KEY", ""),
        max_results=cfg["tools"]["max_search_results"],
    )

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
    )

    settings = SupervisorSettings(
        n_workers=cfg["supervisor"]["n_workers"],
        max_tasks=100,
        max_time_seconds=cfg["supervisor"]["max_time_minutes"] * 60,
        checkpoint_interval=cfg["supervisor"]["checkpoint_interval"],
        thresholds=WeightThresholds(
            min_hypothesis_count=cfg["supervisor"]["min_hypothesis_count"],
            elo_variance_threshold=cfg["supervisor"]["elo_variance_threshold"],
            meta_review_interval=cfg["supervisor"]["meta_review_interval"],
        ),
        seed=0,
    )
    supervisor = Supervisor(config=config, runner=runner, settings=settings, store=store)
    await supervisor.run()
    print(f"Run {run_id} complete. DB: {cfg['db_path']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python coscientist.py "Your research goal"')
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
