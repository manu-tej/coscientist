from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass
from core.models import AgentTask, AgentType, ResearchPlanConfig, SystemStats
from core.stats import compute_weights, sample_agent_type, WeightThresholds, compute_stats
from core.state import StateStore

logger = logging.getLogger("coscientist.supervisor")


@dataclass
class SupervisorSettings:
    n_workers: int = 4
    max_tasks: int = 100
    max_time_seconds: int = 1800
    checkpoint_interval: int = 5
    thresholds: WeightThresholds = None
    seed: int = 0

    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = WeightThresholds()


# Strategy rotation per agent type. The Supervisor cycles through these so the
# system explores all strategies rather than fixating on one.
_GENERATION_STRATEGIES = ["debate", "literature", "assumptions", "expansion"]
_EVOLUTION_STRATEGIES = ["grounding", "coherence", "inspiration", "combination", "simplification", "out_of_box"]


class Supervisor:
    def __init__(
        self,
        config: ResearchPlanConfig,
        runner,
        settings: SupervisorSettings,
        store: StateStore = None,
    ):
        self.config = config
        self.runner = runner
        self.settings = settings
        self.store = store
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self._tasks_dispatched = 0
        self._tasks_completed = 0
        self._tick = 0
        self._last_meta_review_tick = 0
        self._gen_idx = 0
        self._evo_idx = 0
        self._start_time = 0.0

    async def _compute_stats(self) -> SystemStats:
        if self.store is None:
            return SystemStats()
        return await compute_stats(
            self.store, self.config.run_id,
            last_meta_review_tick=self._last_meta_review_tick,
            current_tick=self._tick,
        )

    def _build_task(self, agent_type: AgentType) -> AgentTask:
        strategy = None
        extra: dict = {}
        if agent_type == AgentType.GENERATION:
            strategy = _GENERATION_STRATEGIES[self._gen_idx % len(_GENERATION_STRATEGIES)]
            self._gen_idx += 1
        elif agent_type == AgentType.EVOLUTION:
            strategy = _EVOLUTION_STRATEGIES[self._evo_idx % len(_EVOLUTION_STRATEGIES)]
            self._evo_idx += 1
        elif agent_type == AgentType.META_REVIEW:
            extra["tick"] = self._tick
            self._last_meta_review_tick = self._tick
        return AgentTask(
            priority=self._tasks_dispatched,
            agent_type=agent_type,
            run_id=self.config.run_id,
            strategy=strategy,
            extra=extra,
        )

    async def _next_task(self) -> AgentTask:
        stats = await self._compute_stats()
        weights = compute_weights(stats, self.settings.thresholds)
        agent_type = sample_agent_type(weights, seed=self.settings.seed + self._tick)
        self._tick += 1
        return self._build_task(agent_type)

    def _terminal(self) -> bool:
        if self._tasks_dispatched >= self.settings.max_tasks:
            return True
        if time.monotonic() - self._start_time > self.settings.max_time_seconds:
            return True
        return False

    async def _worker(self) -> None:
        while True:
            task = await self.task_queue.get()
            if task is None:
                self.task_queue.task_done()
                break
            try:
                await self.runner.run_task(task)
            except Exception as exc:
                logger.exception("Task %s failed: %s", task.agent_type, exc)
            finally:
                self._tasks_completed += 1
                self.task_queue.task_done()

    async def run(self) -> None:
        self._start_time = time.monotonic()
        workers = [
            asyncio.create_task(self._worker())
            for _ in range(self.settings.n_workers)
        ]
        while not self._terminal():
            task = await self._next_task()
            await self.task_queue.put(task)
            self._tasks_dispatched += 1
            if self._tasks_dispatched % self.settings.checkpoint_interval == 0:
                logger.info(
                    "Checkpoint: %d dispatched, %d completed",
                    self._tasks_dispatched, self._tasks_completed,
                )
        await self.task_queue.join()
        for _ in workers:
            await self.task_queue.put(None)
        await asyncio.gather(*workers)
