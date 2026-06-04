from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml

from core.state import StateStore


@dataclass
class BenchHypothesis:
    id: str
    text: str
    summary: str
    elo_rating: float
    created_at: str
    generation_method: str = ""
    elo_trajectory: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class BenchRun:
    goal_id: str
    variant: str
    hypotheses: list[BenchHypothesis]
    n_llm_calls: int
    wall_clock_s: float
    db_path: str


def trajectory_from_matches(
    hyp_id: str,
    ordered_match_rows: list[tuple[str, str, str, float, float]],
) -> list[tuple[str, float]]:
    """Replay a hypothesis's Elo over time from deterministically-ordered match rows.
    Each row is (created_at, h1_id, h2_id, elo_after_h1, elo_after_h2)."""
    traj: list[tuple[str, float]] = []
    for created_at, h1_id, h2_id, elo_after_h1, elo_after_h2 in ordered_match_rows:
        if h1_id == hyp_id:
            traj.append((created_at, elo_after_h1))
        elif h2_id == hyp_id:
            traj.append((created_at, elo_after_h2))
    return traj


async def _ordered_match_rows(db_path: str, run_id: str) -> list[tuple[str, str, str, float, float]]:
    """Fetch (created_at, h1_id, h2_id, elo_after_h1, elo_after_h2) for all matches,
    ordered deterministically by (created_at, id) so trajectory replay is stable even
    when several matches share a 1-second CURRENT_TIMESTAMP."""
    import aiosqlite
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT created_at, h1_id, h2_id, elo_after_h1, elo_after_h2 "
            "FROM tournament_matches WHERE run_id=? ORDER BY created_at, id",
            (run_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [(str(r["created_at"]), r["h1_id"], r["h2_id"],
             r["elo_after_h1"], r["elo_after_h2"]) for r in rows]


async def _hypothesis_created_ats(db_path: str, run_id: str) -> dict[str, str]:
    import aiosqlite
    out: dict[str, str] = {}
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, created_at FROM hypotheses WHERE run_id=?", (run_id,),
        ) as cur:
            for r in await cur.fetchall():
                out[r["id"]] = str(r["created_at"])
    return out


async def read_run(
    db_path: str, run_id: str, goal_id: str, variant: str,
    n_llm_calls: int, wall_clock_s: float,
) -> BenchRun:
    """Read a completed run's SQLite into a BenchRun (pure read, no tokens)."""
    store = StateStore(db_path)
    hyps = await store.list_hypotheses(run_id, status="active")
    match_rows = await _ordered_match_rows(db_path, run_id)
    hyp_ts = await _hypothesis_created_ats(db_path, run_id)
    bench_hyps = [
        BenchHypothesis(
            id=h.id, text=h.text, summary=h.summary, elo_rating=h.elo_rating,
            created_at=hyp_ts.get(h.id, ""), generation_method=h.generation_method,
            elo_trajectory=trajectory_from_matches(h.id, match_rows),
        )
        for h in hyps
    ]
    return BenchRun(goal_id=goal_id, variant=variant, hypotheses=bench_hyps,
                    n_llm_calls=n_llm_calls, wall_clock_s=wall_clock_s, db_path=db_path)


def _load_yaml_config(path: str = "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


async def run_system(
    goal: "BenchGoal",  # noqa: F821 (bench.goalset.BenchGoal)
    *,
    variant: str = "full",
    seed: int = 0,
    db_path: Optional[str] = None,
    max_tasks: int = 100,
    max_time_seconds: Optional[int] = None,
    weight_overrides: Optional[dict] = None,
    ranking_mode: str = "elo",
    client_factory: Optional[Callable[[str, str], object]] = None,
    config_yaml: str = "config.yaml",
) -> BenchRun:
    """Run the full multi-agent system on one goal and capture a BenchRun.

    - `variant`: label stored on the run ("full", "no_evolution", ...).
    - `weight_overrides`: {AgentType: 0.0} forced into compute_weights (ablations).
    - `ranking_mode`: "elo" (normal) | "absolute" (no_tournament — Task D2 reads this).
    - `client_factory(model_strong, model_fast)`: injects a fake client in tests.
    """
    from core.orchestrator import AgentRunner
    from core.supervisor import Supervisor, SupervisorSettings
    from core.stats import WeightThresholds
    from core.models import ResearchPlanConfig
    from tools.search import SearchTool
    from agents.base import BaseAgent
    from agents.generation import GenerationAgent
    from agents.reflection import ReflectionAgent
    from agents.ranking import RankingAgent
    from agents.proximity import ProximityAgent
    from agents.evolution import EvolutionAgent
    from agents.meta_review import MetaReviewAgent

    try:
        from dotenv import load_dotenv
        load_dotenv()  # pick up .env (provider + OAuth token) at run time
    except ImportError:
        pass
    cfg = _load_yaml_config(config_yaml)
    if ranking_mode not in ("elo", "absolute"):
        raise ValueError(f"Unknown ranking_mode: {ranking_mode!r} (expected 'elo' or 'absolute')")
    if ranking_mode == "absolute":
        raise NotImplementedError(
            "ranking_mode='absolute' (the no_tournament ablation's absolute judge-score "
            "sort) is deferred to the v1 capstone run loop (see plan §16/E4). It is NOT "
            "yet wired into run_system; refusing to run rather than silently behaving like "
            "the 'full' variant."
        )
    run_id = uuid.uuid4().hex[:8]
    db_path = db_path or f"bench_runs/{goal.id}_{variant}_{seed}_{run_id}.db"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    store = StateStore(db_path)
    await store.init_db()

    if client_factory is not None:
        client = client_factory(cfg["anthropic"]["model_strong"],
                                cfg["anthropic"]["model_fast"])
    else:
        # Use the configured backend (default 'claude-code' subscription), the
        # same factory the entrypoint uses — so bench runs go through whatever
        # provider config.yaml / COSCIENTIST_PROVIDER selects, not a hardcoded API client.
        from tools.llm import make_backend
        client = make_backend(cfg)

    prompts_dir = Path("prompts")
    base = BaseAgent(client=client, prompts_dir=prompts_dir)

    config = ResearchPlanConfig(
        run_id=run_id, goal=goal.goal,
        preferences="Focus on novel, testable hypotheses",
        attributes=["Novelty", "Feasibility"], constraints="Must be testable",
        safety_approved=True,
    )
    await store.save_config(config)

    search = SearchTool(api_key=os.environ.get("TAVILY_API_KEY", ""),
                        max_results=cfg["tools"]["max_search_results"])
    runner = AgentRunner(
        store=store, config=config,
        generation=GenerationAgent(base=base),
        reflection=ReflectionAgent(base=base),
        ranking=RankingAgent(base=base, store=store,
                             elo_k=cfg["tournament"]["elo_k_factor"],
                             multi_turn_threshold=cfg["tournament"]["multi_turn_threshold"]),
        proximity=ProximityAgent(store=store, model_name=cfg["proximity"]["model"],
                                 similarity_threshold=cfg["proximity"]["similarity_threshold"],
                                 duplicate_threshold=cfg["proximity"]["duplicate_threshold"]),
        evolution=EvolutionAgent(base=base),
        meta_review=MetaReviewAgent(base=base),
        search=search,
    )

    settings = SupervisorSettings(
        n_workers=cfg["supervisor"]["n_workers"],
        max_tasks=max_tasks,
        max_time_seconds=max_time_seconds or cfg["supervisor"]["max_time_minutes"] * 60,
        checkpoint_interval=cfg["supervisor"]["checkpoint_interval"],
        thresholds=WeightThresholds(
            min_hypothesis_count=cfg["supervisor"]["min_hypothesis_count"],
            elo_variance_threshold=cfg["supervisor"]["elo_variance_threshold"],
            meta_review_interval=cfg["supervisor"]["meta_review_interval"]),
        seed=seed,
    )
    supervisor = _make_supervisor(config, runner, settings, store, weight_overrides)

    start = time.monotonic()
    await supervisor.run()
    wall = time.monotonic() - start

    # NOTE: Supervisor has no call counter yet, so n_llm_calls is a placeholder (0).
    # Cost/efficiency reporting uses bench.cost estimates, not this field (see plan §17.3).
    n_calls = supervisor.runner_call_count if hasattr(supervisor, "runner_call_count") else 0
    return await read_run(db_path, run_id=run_id, goal_id=goal.id, variant=variant,
                          n_llm_calls=n_calls, wall_clock_s=wall)


def _make_supervisor(config, runner, settings, store, weight_overrides):
    """Build a Supervisor; if weight_overrides given, subclass to force weights
    to zero before sampling (the ablation mechanism, §10)."""
    from core.supervisor import Supervisor
    from core.stats import compute_weights, sample_agent_type

    if not weight_overrides:
        return Supervisor(config=config, runner=runner, settings=settings, store=store)

    class _AblatedSupervisor(Supervisor):
        async def _next_task(self):
            stats = await self._compute_stats()
            weights = compute_weights(stats, self.settings.thresholds)
            for atype, w in weight_overrides.items():
                weights[atype] = w
            agent_type = sample_agent_type(weights, seed=self.settings.seed + self._tick)
            self._tick += 1
            return self._build_task(agent_type)

    return _AblatedSupervisor(config=config, runner=runner, settings=settings, store=store)
