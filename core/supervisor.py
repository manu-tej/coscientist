from __future__ import annotations
import asyncio
from collections import Counter
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
    max_queue_size: int | None = None
    thresholds: WeightThresholds = None
    seed: int = 0
    personas: tuple[str, ...] = ()

    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = WeightThresholds()
        if self.n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        if self.max_tasks < 0:
            raise ValueError("max_tasks must be >= 0")
        if self.max_time_seconds <= 0:
            raise ValueError("max_time_seconds must be > 0")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be >= 1")
        if self.max_queue_size is not None and self.max_queue_size < 1:
            raise ValueError("max_queue_size must be >= 1 when provided")
        if isinstance(self.personas, str):
            raise ValueError("personas must be a tuple of non-empty role names")
        self.personas = tuple(self.personas)
        if any(not isinstance(persona, str) or not persona.strip()
               for persona in self.personas):
            raise ValueError("personas must contain only non-empty role names")


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
        # Bound work-in-flight so scheduling stays coupled to completed science.
        # The previous unbounded queue could dispatch 100 tasks while only one had
        # completed, making dynamic weights stale and amplifying provider rate limits.
        queue_size = settings.max_queue_size or max(1, settings.n_workers * 2)
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self._tasks_dispatched = 0
        self._tasks_completed = 0
        self._tick = 0
        self._last_meta_review_tick = 0
        self._gen_idx = 0
        self._evo_idx = 0
        self._persona_idx = 0
        self._start_time = 0.0
        self.task_counts: Counter[str] = Counter()
        self.task_failures: list[dict[str, str]] = []
        self.agent_task_trace: list[dict[str, object]] = []
        self._trace_by_task: dict[int, dict[str, object]] = {}

    async def _compute_stats(self) -> SystemStats:
        if self.store is None:
            return SystemStats()
        return await compute_stats(
            self.store, self.config.run_id,
            last_meta_review_tick=self._last_meta_review_tick,
            current_tick=self._tick,
            ranking_mode=getattr(self.runner, "ranking_mode", "elo"),
        )

    def _build_task(self, agent_type: AgentType) -> AgentTask:
        strategy = None
        extra: dict = {}
        persona = ""
        if self.settings.personas:
            persona = self.settings.personas[
                self._persona_idx % len(self.settings.personas)
            ]
            self._persona_idx += 1
            extra["persona"] = persona
        if agent_type == AgentType.GENERATION:
            strategy = _GENERATION_STRATEGIES[self._gen_idx % len(_GENERATION_STRATEGIES)]
            self._gen_idx += 1
        elif agent_type == AgentType.EVOLUTION:
            strategy = _EVOLUTION_STRATEGIES[self._evo_idx % len(_EVOLUTION_STRATEGIES)]
            self._evo_idx += 1
        elif agent_type == AgentType.META_REVIEW:
            extra["tick"] = self._tick
            self._last_meta_review_tick = self._tick
        task = AgentTask(
            priority=self._tasks_dispatched,
            agent_type=agent_type,
            run_id=self.config.run_id,
            strategy=strategy,
            extra=extra,
        )
        trace = {
            "priority": task.priority,
            "agent_type": task.agent_type.value,
            "persona": persona,
            "strategy": strategy or "",
            "status": "queued",
            "error": None,
        }
        self.agent_task_trace.append(trace)
        self._trace_by_task[id(task)] = trace
        return task

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
            trace = self._trace_by_task.get(id(task))
            if trace is not None:
                trace["status"] = "running"
            try:
                await self.runner.run_task(task)
            except asyncio.CancelledError as exc:
                if trace is not None:
                    trace["status"] = "cancelled"
                    trace["error"] = f"{type(exc).__name__}: task cancelled"
                raise
            except Exception as exc:
                logger.exception("Task %s failed: %s", task.agent_type, exc)
                failure = {
                    "agent_type": task.agent_type.value,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                self.task_failures.append(failure)
                if trace is not None:
                    trace["status"] = "failed"
                    trace["error"] = failure["error"]
            else:
                if trace is not None:
                    trace["status"] = "succeeded"
            finally:
                self.task_counts[task.agent_type.value] += 1
                self._tasks_completed += 1
                self.task_queue.task_done()

    async def run(self) -> None:
        self._start_time = time.monotonic()
        workers = [
            asyncio.create_task(self._worker())
            for _ in range(self.settings.n_workers)
        ]
        try:
            async with asyncio.timeout(self.settings.max_time_seconds):
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
        except TimeoutError as exc:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise TimeoutError(
                f"Supervisor exceeded {self.settings.max_time_seconds}s "
                f"after completing {self._tasks_completed}/"
                f"{self._tasks_dispatched} dispatched tasks"
            ) from exc
        except BaseException:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise
