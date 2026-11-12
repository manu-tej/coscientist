from __future__ import annotations

import asyncio
import contextlib
import math
import time
import uuid
from dataclasses import dataclass

from bench.goalset import BenchGoal
from bench.runner import BenchRun
from bench.datasets.gpqa import score_answer
from bench.concordance import (
    ScoredHypothesis, concordance_stats, blue_minus_red_spread,
)
from bench.report import build_report, render_markdown, to_json


@dataclass
class _ReturnedRun:
    run: BenchRun
    accepted_for_scoring: bool
    rejection_reason: str | None
    cache_state: str
    cache_key: str
    cache_stale_reason: str | None
    execution_source: str
    cache_write_error: str | None
    lease_waited: bool = False
    lease_wait_seconds: float = 0.0
    lease_takeover: bool = False
    lease_renewals: int = 0


@dataclass
class _AcquiredRun:
    run: BenchRun
    completion_problem: str | None
    cache_state: str
    cache_key: str
    cache_stale_reason: str | None
    execution_source: str
    cache_write_error: str | None
    lease_waited: bool = False
    lease_wait_seconds: float = 0.0
    lease_takeover: bool = False
    lease_renewals: int = 0


class _FreshRunError(Exception):
    """Carry cache provenance when the model-backed run itself raises."""

    def __init__(
        self,
        cause: Exception,
        *,
        cache_state: str,
        cache_key: str,
        cache_stale_reason: str | None,
        db_path: str,
        lease_waited: bool = False,
        lease_wait_seconds: float = 0.0,
        lease_takeover: bool = False,
        n_llm_calls: int = 0,
        estimated_tokens: int = 0,
    ):
        super().__init__(str(cause))
        self.cause = cause
        self.cache_state = cache_state
        self.cache_key = cache_key
        self.cache_stale_reason = cache_stale_reason
        self.db_path = db_path
        self.lease_waited = lease_waited
        self.lease_wait_seconds = lease_wait_seconds
        self.lease_takeover = lease_takeover
        self.n_llm_calls = n_llm_calls
        self.estimated_tokens = estimated_tokens


class _LeaseLostError(RuntimeError):
    """The owner must stop before another process is allowed to take over."""


@dataclass(frozen=True)
class _LeaseTiming:
    lease_seconds: float
    poll_seconds: float
    hard_timeout_seconds: float
    wait_timeout_seconds: float


def _positive_finite(value, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _lease_timing(args, execution) -> _LeaseTiming:
    """Internal defaults keep leases responsive without shortening paid runs."""
    max_time = _positive_finite(
        execution.max_time_seconds,
        name="benchmark max time",
    )
    # Keep enough post-run headroom to validate and atomically record the DB
    # before release. A heartbeat still makes arbitrarily long live runs safe.
    default_lease = 60.0
    grace = max(30.0, min(300.0, max_time / 10.0))
    lease_seconds = _positive_finite(
        getattr(args, "_lease_seconds", default_lease),
        name="lease duration",
    )
    hard_timeout = _positive_finite(
        getattr(args, "_lease_hard_timeout_seconds", max_time + grace),
        name="lease owner hard timeout",
    )
    poll_seconds = _positive_finite(
        getattr(args, "_lease_poll_seconds", min(1.0, lease_seconds / 4.0)),
        name="lease poll interval",
    )
    wait_timeout = _positive_finite(
        getattr(
            args,
            "_lease_wait_timeout_seconds",
            hard_timeout + lease_seconds + max(5.0, poll_seconds),
        ),
        name="lease wait timeout",
    )
    return _LeaseTiming(
        lease_seconds=lease_seconds,
        poll_seconds=poll_seconds,
        hard_timeout_seconds=hard_timeout,
        wait_timeout_seconds=wait_timeout,
    )


def _new_cache_counts() -> dict[str, int | float]:
    return {
        "hit": 0,
        "miss": 0,
        "stale": 0,
        "disabled": 0,
        "fresh_attempts": 0,
        "fresh_runs_returned": 0,
        "reused_runs": 0,
        "uncacheable_runs": 0,
        "write_failures": 0,
        "lease_claim_attempts": 0,
        "lease_owners": 0,
        "lease_waits": 0,
        "lease_wait_seconds": 0.0,
        "lease_takeovers": 0,
        "lease_renewals": 0,
        "lease_renewal_failures": 0,
        "lease_timeouts": 0,
        "lease_releases": 0,
        "lease_release_failures": 0,
        "coalesced_runs": 0,
    }


async def _run_with_lease_heartbeat(
    run_coro,
    *,
    manifest,
    key,
    owner_token: str,
    timing: _LeaseTiming,
    cache_counts: dict[str, int | float],
    ready_is_terminal: bool = False,
):
    """Run one owner task while renewing its short, crash-recoverable lease."""
    run_task = asyncio.create_task(run_coro)
    lease_lost = asyncio.Event()
    owner_timed_out = asyncio.Event()
    lease_error: list[BaseException] = []
    interval = max(0.05, timing.lease_seconds / 3.0)

    async def heartbeat() -> None:
        while not run_task.done():
            await asyncio.sleep(interval)
            if run_task.done():
                return
            try:
                renewed = await manifest.renew(
                    key,
                    owner_token=owner_token,
                    lease_seconds=timing.lease_seconds,
                )
            except Exception as exc:  # losing the lease must stop paid work
                lease_error.append(exc)
                renewed = False
            if not renewed:
                try:
                    recovered = await manifest.claim(
                        key,
                        lease_seconds=timing.lease_seconds,
                        owner_token=owner_token,
                    )
                except Exception as exc:
                    lease_error.append(exc)
                    recovered = None
                if (
                    recovered is not None
                    and recovered.state == "acquired"
                    and recovered.owner_token == owner_token
                ):
                    cache_counts["lease_renewals"] += 1
                    continue
                if (
                    ready_is_terminal
                    and recovered is not None
                    and recovered.state == "ready"
                ):
                    return
                cache_counts["lease_renewal_failures"] += 1
                lease_lost.set()
                run_task.cancel()
                return
            cache_counts["lease_renewals"] += 1

    heartbeat_task = asyncio.create_task(heartbeat())

    async def cancel_on_owner_timeout() -> None:
        await asyncio.sleep(timing.hard_timeout_seconds)
        if not run_task.done():
            owner_timed_out.set()
            cache_counts["lease_timeouts"] += 1
            run_task.cancel()

    timeout_task = asyncio.create_task(cancel_on_owner_timeout())

    def timeout_error() -> TimeoutError:
        return TimeoutError(
            "benchmark execution exceeded the cooperative owner timeout "
            f"of {timing.hard_timeout_seconds:.1f}s"
        )

    try:
        try:
            result = await run_task
        except asyncio.CancelledError:
            if lease_lost.is_set():
                detail = (
                    f": {type(lease_error[0]).__name__}: {lease_error[0]}"
                    if lease_error
                    else ""
                )
                raise _LeaseLostError(
                    f"benchmark execution lease was lost{detail}"
                ) from (lease_error[0] if lease_error else None)
            if owner_timed_out.is_set():
                raise timeout_error()
            raise
        if lease_lost.is_set():
            raise _LeaseLostError("benchmark execution lease was lost")
        if owner_timed_out.is_set():
            raise timeout_error()
        return result
    finally:
        heartbeat_task.cancel()
        timeout_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task
        with contextlib.suppress(asyncio.CancelledError):
            await timeout_task


async def _release_lease(
    manifest,
    key,
    owner_token: str,
    cache_counts: dict[str, int | float],
) -> None:
    """Best-effort guarded release; expiration remains the crash fallback."""
    try:
        released = await manifest.release(key, owner_token=owner_token)
    except Exception:
        cache_counts["lease_release_failures"] += 1
        return
    if released:
        cache_counts["lease_releases"] += 1
    # False is benign when a waiter observed the newly completed row and
    # atomically cleared the now-obsolete lease before this owner finalized.


async def _manifest_for_args(args):
    """Open one manifest per command (and one shared instance per sweep)."""
    if getattr(args, "no_cache", False):
        return None
    existing = getattr(args, "_manifest_instance", None)
    if existing is not None:
        return existing
    path = getattr(args, "manifest", None)
    if not path:
        # Programmatic callers written before the cache surface remain opt-in;
        # CLI callers always receive the parser's durable default path.
        return None

    from bench.manifest import Manifest

    manifest = Manifest(path)
    await manifest.init()
    setattr(args, "_manifest_instance", manifest)
    return manifest


async def _execution_context(args, *, max_tasks: int):
    """Resolve the exact execution once, before deriving any cache identity."""
    from bench.manifest import system_version
    from bench.runner import resolve_benchmark_execution
    from tools.llm import load_backend_environment

    # Direct Python callers need the same .env/config/explicit precedence as
    # the top-level CLI dispatcher.
    load_backend_environment()
    execution = resolve_benchmark_execution(
        provider=getattr(args, "provider", None),
        model_strong=getattr(args, "model_strong", None),
        model_fast=getattr(args, "model_fast", None),
        n_workers=getattr(args, "workers", None),
        max_tasks=max_tasks,
        max_time_seconds=getattr(args, "max_time", 900),
        ranking_mode=getattr(args, "ranking_mode", "elo"),
        weight_overrides=getattr(args, "weight_overrides", None),
        personas=tuple(getattr(args, "personas", ()) or ()),
        judge_provider=getattr(args, "judge_provider", None),
        judge_model=getattr(args, "judge_model", None),
        judge_weights=getattr(args, "judge_weights", None),
    )
    return execution, system_version(), await _manifest_for_args(args)


async def _acquire_run(
    args,
    goal: BenchGoal,
    *,
    label: str,
    variant: str,
    max_tasks: int,
    execution,
    version: str,
    manifest,
    cache_counts: dict[str, int | float],
) -> _AcquiredRun:
    """Reuse one complete run or single-flight one fresh paid execution."""
    from bench.manifest import make_run_cache_key, run_completion_problem
    from bench.runner import (
        BenchmarkUsageSink,
        benchmark_db_path,
        run_system,
    )

    key = make_run_cache_key(
        goal,
        variant=variant,
        system_version_value=version,
        seed=args.seed,
        execution=execution,
    )
    cache_state = "disabled"
    stale_reason = None
    lease_waited = False
    lease_wait_seconds = 0.0
    lease_takeover = False
    lease_owner_token: str | None = None
    timing = _lease_timing(args, execution) if manifest is not None else None
    if manifest is not None:
        lookup = await manifest.load_complete(
            key,
            expected_goal=goal.goal,
            expected_tasks=max_tasks,
        )
        cache_state = lookup.state
        stale_reason = lookup.reason
        cache_counts[cache_state] += 1
        if lookup.state == "hit":
            if lookup.run is None:
                raise RuntimeError(
                    f"manifest returned a cache hit without a run for {goal.id!r}"
                )
            cache_counts["reused_runs"] += 1
            problem = run_completion_problem(
                lookup.run,
                key,
                expected_tasks=max_tasks,
            )
            return _AcquiredRun(
                run=lookup.run,
                completion_problem=problem,
                cache_state=cache_state,
                cache_key=lookup.cache_key or key.id,
                cache_stale_reason=None,
                execution_source="reused",
                cache_write_error=None,
            )
        wait_started: float | None = None
        while lease_owner_token is None:
            cache_counts["lease_claim_attempts"] += 1
            claim = await manifest.claim(
                key,
                lease_seconds=timing.lease_seconds,
            )
            if claim.state == "ready":
                ready = await manifest.load_complete(
                    key,
                    expected_goal=goal.goal,
                    expected_tasks=max_tasks,
                )
                if ready.state == "hit":
                    if ready.run is None:
                        raise RuntimeError(
                            "manifest returned a ready lease state without a run"
                        )
                    cache_counts["hit"] += 1
                    cache_counts["reused_runs"] += 1
                    if lease_waited:
                        lease_wait_seconds = time.monotonic() - wait_started
                        cache_counts["lease_wait_seconds"] += lease_wait_seconds
                        cache_counts["coalesced_runs"] += 1
                    problem = run_completion_problem(
                        ready.run,
                        key,
                        expected_tasks=max_tasks,
                    )
                    return _AcquiredRun(
                        run=ready.run,
                        completion_problem=problem,
                        cache_state="hit",
                        cache_key=ready.cache_key or key.id,
                        cache_stale_reason=None,
                        execution_source="reused",
                        cache_write_error=None,
                        lease_waited=lease_waited,
                        lease_wait_seconds=lease_wait_seconds,
                    )
                # A row can become corrupt between claim's transactional ready
                # check and validation. load_complete marks it stale; the next
                # claim can then elect exactly one replacement owner.
                cache_counts[ready.state] += 1
                stale_reason = ready.reason
                continue
            if claim.state == "acquired":
                if not claim.owner_token:
                    raise RuntimeError("acquired execution lease has no owner token")
                lease_owner_token = claim.owner_token
                cache_counts["lease_owners"] += 1
                if lease_waited:
                    lease_wait_seconds = time.monotonic() - wait_started
                    cache_counts["lease_wait_seconds"] += lease_wait_seconds
                    cache_counts["lease_takeovers"] += 1
                    lease_takeover = True
                break
            if claim.state != "waiting":
                raise RuntimeError(f"unknown execution lease state: {claim.state!r}")
            if not lease_waited:
                lease_waited = True
                wait_started = time.monotonic()
                cache_counts["lease_waits"] += 1
            elapsed = time.monotonic() - wait_started
            if elapsed >= timing.wait_timeout_seconds:
                cache_counts["lease_timeouts"] += 1
                cache_counts["lease_wait_seconds"] += elapsed
                raise _FreshRunError(
                    TimeoutError(
                        "timed out waiting for another benchmark process "
                        f"after {elapsed:.1f}s"
                    ),
                    cache_state=cache_state,
                    cache_key=key.id,
                    cache_stale_reason=stale_reason,
                    db_path="",
                    lease_waited=True,
                    lease_wait_seconds=elapsed,
                )
            delay = min(timing.poll_seconds, timing.wait_timeout_seconds - elapsed)
            if claim.expires_at is not None:
                delay = min(delay, max(0.01, claim.expires_at - time.time()))
            await asyncio.sleep(max(0.01, delay))
    else:
        cache_counts["disabled"] += 1

    cache_counts["fresh_attempts"] += 1
    lease_renewals_before = int(cache_counts["lease_renewals"])
    # Never let two model/worker cells, or a replacement for stale/corrupt
    # evidence, write the same SQLite file.
    fresh_id = f"{key.id[:12]}-{uuid.uuid4().hex[:8]}"
    db_path = benchmark_db_path(
        goal.id,
        label=label,
        seed=args.seed,
        run_id=fresh_id,
    )
    usage = BenchmarkUsageSink()
    try:
        try:
            run_coro = run_system(
                goal,
                variant=variant,
                seed=args.seed,
                db_path=db_path,
                max_tasks=execution.max_tasks,
                max_time_seconds=execution.max_time_seconds,
                n_workers=execution.workers,
                provider=execution.provider,
                model_fast=execution.model_fast,
                model_strong=execution.model_strong,
                ranking_mode=execution.ranking_mode,
                weight_overrides=getattr(args, "weight_overrides", None),
                personas=execution.personas,
                resolved_execution=execution,
                usage_sink=usage,
            )
            if manifest is not None:
                run = await _run_with_lease_heartbeat(
                    run_coro,
                    manifest=manifest,
                    key=key,
                    owner_token=lease_owner_token,
                    timing=timing,
                    cache_counts=cache_counts,
                )
            else:
                run = await run_coro
        except Exception as exc:
            raise _FreshRunError(
                exc,
                cache_state=cache_state,
                cache_key=key.id,
                cache_stale_reason=stale_reason,
                db_path=db_path,
                lease_waited=lease_waited,
                lease_wait_seconds=lease_wait_seconds,
                lease_takeover=lease_takeover,
                n_llm_calls=usage.n_llm_calls,
                estimated_tokens=usage.estimated_tokens,
            ) from exc

        problem = run_completion_problem(run, key, expected_tasks=max_tasks)
        write_error = None
        if manifest is not None:
            try:
                still_owner = await manifest.renew(
                    key,
                    owner_token=lease_owner_token,
                    lease_seconds=timing.lease_seconds,
                )
            except Exception as exc:
                still_owner = False
                ownership_error = exc
            else:
                ownership_error = None
            if not still_owner:
                cache_counts["lease_renewal_failures"] += 1
                lost = _LeaseLostError(
                    "benchmark execution lease was lost before cache commit"
                )
                raise _FreshRunError(
                    lost,
                    cache_state=cache_state,
                    cache_key=key.id,
                    cache_stale_reason=stale_reason,
                    db_path=db_path,
                    lease_waited=lease_waited,
                    lease_wait_seconds=lease_wait_seconds,
                    lease_takeover=lease_takeover,
                    n_llm_calls=usage.n_llm_calls,
                    estimated_tokens=usage.estimated_tokens,
                ) from ownership_error
            cache_counts["lease_renewals"] += 1
        if manifest is not None and problem is None:
            try:
                await _run_with_lease_heartbeat(
                    manifest.record_complete(
                        key,
                        run,
                        expected_goal=goal.goal,
                        expected_tasks=max_tasks,
                        owner_token=lease_owner_token,
                    ),
                    manifest=manifest,
                    key=key,
                    owner_token=lease_owner_token,
                    timing=timing,
                    cache_counts=cache_counts,
                    ready_is_terminal=True,
                )
            except Exception as exc:
                # The scientific run remains usable and its spend remains
                # accounted even if the cache write itself fails.
                write_error = f"{type(exc).__name__}: {exc}"
                cache_counts["write_failures"] += 1
        elif manifest is not None:
            cache_counts["uncacheable_runs"] += 1

        cache_counts["fresh_runs_returned"] += 1
        return _AcquiredRun(
            run=run,
            completion_problem=problem,
            cache_state=cache_state,
            cache_key=key.id,
            cache_stale_reason=stale_reason,
            execution_source="fresh",
            cache_write_error=write_error,
            lease_waited=lease_waited,
            lease_wait_seconds=lease_wait_seconds,
            lease_takeover=lease_takeover,
            lease_renewals=(
                int(cache_counts["lease_renewals"]) - lease_renewals_before
            ),
        )
    finally:
        if manifest is not None and lease_owner_token is not None:
            await _release_lease(
                manifest,
                key,
                lease_owner_token,
                cache_counts,
            )


def _run_metadata(
    run: BenchRun,
    *,
    accepted_for_scoring: bool = True,
    rejection_reason: str | None = None,
    cache_state: str = "disabled",
    cache_key: str = "",
    cache_stale_reason: str | None = None,
    execution_source: str = "fresh",
    cache_write_error: str | None = None,
    lease_waited: bool = False,
    lease_wait_seconds: float = 0.0,
    lease_takeover: bool = False,
    lease_renewals: int = 0,
) -> dict:
    """Reproducibility and topology/accounting record for one benchmark cell."""
    reused = execution_source == "reused"
    return {
        "goal_id": run.goal_id,
        "run_id": run.run_id,
        "variant": run.variant,
        "accepted_for_scoring": accepted_for_scoring,
        "rejection_reason": rejection_reason,
        "db_path": run.db_path,
        "n_hypotheses": len(run.hypotheses),
        # IMPORTANT 6: `n_hypotheses` above only counts the active pool --
        # this discloses how many verification removed, so a report can never
        # silently improve concordance/recall via undisclosed quarantining.
        "n_quarantined": run.n_quarantined,
        "n_llm_calls": run.n_llm_calls,
        "estimated_tokens": run.estimated_tokens,
        "wall_clock_s": run.wall_clock_s,
        "workers": run.workers,
        "backend": run.backend,
        "models": run.models,
        "max_backend_concurrency": run.max_backend_concurrency,
        "agent_task_counts": run.agent_task_counts,
        "agent_failures": run.agent_failures,
        "agent_task_trace": run.agent_task_trace,
        "cache_state": cache_state,
        "cache_key": cache_key,
        "cache_stale_reason": cache_stale_reason,
        "cache_write_error": cache_write_error,
        "execution_source": execution_source,
        "lease_waited": lease_waited,
        "lease_wait_seconds": lease_wait_seconds,
        "lease_takeover": lease_takeover,
        "lease_renewals": lease_renewals,
        "fresh_llm_calls": 0 if reused else run.n_llm_calls,
        "reused_llm_calls": run.n_llm_calls if reused else 0,
        "fresh_estimated_tokens": 0 if reused else run.estimated_tokens,
        "reused_estimated_tokens": run.estimated_tokens if reused else 0,
    }


def _accounting_metadata(
    returned_runs: list[_ReturnedRun],
    *,
    manifest,
    cache_counts: dict[str, int | float],
    failed_attempts: tuple[_FreshRunError, ...] | list[_FreshRunError] = (),
    reference_calls: int = 0,
    reference_estimated_tokens: int = 0,
) -> dict:
    returned_fresh_calls = sum(
        item.run.n_llm_calls
        for item in returned_runs
        if item.execution_source == "fresh"
    )
    reused_calls = sum(
        item.run.n_llm_calls
        for item in returned_runs
        if item.execution_source == "reused"
    )
    returned_fresh_tokens = sum(
        item.run.estimated_tokens
        for item in returned_runs
        if item.execution_source == "fresh"
    )
    reused_tokens = sum(
        item.run.estimated_tokens
        for item in returned_runs
        if item.execution_source == "reused"
    )
    failed_calls = sum(item.n_llm_calls for item in failed_attempts)
    failed_tokens = sum(item.estimated_tokens for item in failed_attempts)
    fresh_system_calls = returned_fresh_calls + failed_calls
    fresh_system_tokens = returned_fresh_tokens + failed_tokens
    fresh_calls = fresh_system_calls + reference_calls
    fresh_tokens = fresh_system_tokens + reference_estimated_tokens
    manifest_path = getattr(manifest, "path", None)
    cache = {
        "enabled": manifest is not None,
        "manifest": manifest_path,
        "hits": cache_counts["hit"],
        "misses": cache_counts["miss"],
        "stale": cache_counts["stale"],
        "disabled": cache_counts["disabled"],
        "write_failures": cache_counts["write_failures"],
        "uncacheable_runs": cache_counts["uncacheable_runs"],
        "fresh_attempts": cache_counts["fresh_attempts"],
        "fresh_runs_returned": cache_counts["fresh_runs_returned"],
        "reused_runs": cache_counts["reused_runs"],
        "lease_claim_attempts": cache_counts["lease_claim_attempts"],
        "lease_owners": cache_counts["lease_owners"],
        "lease_waits": cache_counts["lease_waits"],
        "lease_wait_seconds": cache_counts["lease_wait_seconds"],
        "lease_takeovers": cache_counts["lease_takeovers"],
        "lease_renewals": cache_counts["lease_renewals"],
        "lease_renewal_failures": cache_counts["lease_renewal_failures"],
        "lease_timeouts": cache_counts["lease_timeouts"],
        "lease_releases": cache_counts["lease_releases"],
        "lease_release_failures": cache_counts["lease_release_failures"],
        "coalesced_runs": cache_counts["coalesced_runs"],
    }
    return {
        "cache": cache,
        "n_cache_hits": cache_counts["hit"],
        "n_cache_misses": cache_counts["miss"],
        "n_cache_stale": cache_counts["stale"],
        "n_fresh_runs": cache_counts["fresh_runs_returned"],
        "n_reused_runs": cache_counts["reused_runs"],
        "fresh_system_attempts": cache_counts["fresh_attempts"],
        "fresh_system_llm_calls": fresh_system_calls,
        "failed_fresh_llm_calls": failed_calls,
        "reused_llm_calls": reused_calls,
        "fresh_llm_calls": fresh_calls,
        "invocation_llm_calls": fresh_calls,
        "n_llm_calls": fresh_calls + reused_calls,
        "fresh_system_estimated_tokens": fresh_system_tokens,
        "failed_fresh_estimated_tokens": failed_tokens,
        "reused_estimated_tokens": reused_tokens,
        "fresh_estimated_tokens": fresh_tokens,
        "invocation_estimated_tokens": fresh_tokens,
        "estimated_tokens": fresh_tokens + reused_tokens,
    }


def concordance_from_runs(
    goals: list[BenchGoal], runs: list[BenchRun],
    reference_accuracy: dict[str, float],
    bin_width: int = 50, min_support: int = 5,
) -> dict:
    """Score every hypothesis's parsed answer against its goal's gold_answer,
    then compute the full concordance statistics + blue−red spread."""
    non_elo = [run.goal_id for run in runs if run.ranking_mode != "elo"]
    if non_elo:
        from bench.errors import BenchError

        raise BenchError(
            "Elo concordance cannot consume absolute-ranking runs: "
            + ", ".join(non_elo)
        )
    gold = {g.id: g.gold_answer for g in goals if g.gold_answer}
    rows: list[ScoredHypothesis] = []
    for run in runs:
        ga = gold.get(run.goal_id)
        if ga is None:
            continue
        for h in run.hypotheses:
            rows.append(ScoredHypothesis(
                elo=h.elo_rating, correct=score_answer(h.text, ga),
                question_id=run.goal_id, stratum=run.run_id or None))
    stats = concordance_stats(rows, bin_width, min_support)
    stats["blue_minus_red"] = blue_minus_red_spread(
        rows, reference_accuracy, bin_width, min_support, n_boot=10000, seed=0)
    return stats


def rediscovery_from_runs(goals: list[BenchGoal], runs: list[BenchRun]) -> dict:
    """Score whether the system's hypotheses surface the known gold entities
    and whether higher-ranked hypotheses contain more of them."""
    from bench.goldset import entity_in_text, rank_hypotheses, score_recall
    from scipy.stats import spearmanr

    gold_map = {g.id: g.gold_entities for g in goals if g.gold_entities}
    modes = {run.ranking_mode for run in runs}
    if len(modes) > 1:
        from bench.errors import BenchError

        raise BenchError("rediscovery cannot mix Elo and absolute ranking scales")
    ranking_mode = next(iter(modes), "elo")
    if ranking_mode not in {"elo", "absolute"}:
        raise ValueError(f"unknown ranking mode {ranking_mode!r}")
    per_goal: list[dict] = []
    ranking_recall_rows: list[tuple[float, int]] = []
    for run in runs:
        gold = gold_map.get(run.goal_id)
        if not gold:
            continue
        hyps = run.hypotheses
        top = rank_hypotheses(hyps, ranking_mode=ranking_mode)
        per_goal.append({
            "goal_id": run.goal_id,
            "n_gold": len(gold),
            "n_hyps": len(hyps),
            "pool_recall": score_recall(hyps, gold),       # surfaced anywhere
            "top1_recall": score_recall(top[:1], gold),    # in the single best hypothesis
            "top3_recall": score_recall(
                hyps, gold, k=3, ranking_mode=ranking_mode
            ),
        })
        for h in hyps:
            blob = f"{h.summary} {h.text}"
            n_hit = sum(1 for e in gold if entity_in_text(e, blob))
            score = h.elo_rating if ranking_mode == "elo" else h.absolute_score
            if score is None:
                raise ValueError(
                    f"absolute hypothesis {h.id!r} is missing its rubric score"
                )
            ranking_recall_rows.append((float(score), n_hit))

    n = len(per_goal)
    ranking_scores = [r[0] for r in ranking_recall_rows]
    if len(ranking_recall_rows) >= 3 and len(set(ranking_scores)) > 1:
        rho, p = spearmanr(
            ranking_scores, [r[1] for r in ranking_recall_rows]
        )
    else:
        rho, p = float("nan"), float("nan")
    result = {
        "per_goal": per_goal,
        "ranking_mode": ranking_mode,
        "n_goals": n,
        "mean_pool_recall": sum(g["pool_recall"] for g in per_goal) / n if n else 0.0,
        "mean_top1_recall": sum(g["top1_recall"] for g in per_goal) / n if n else 0.0,
        "mean_top3_recall": sum(g["top3_recall"] for g in per_goal) / n if n else 0.0,
        "ranking_recall_spearman": float(rho),
        "ranking_recall_p": float(p),
        "n_hyps_scored": len(ranking_recall_rows),
    }
    if ranking_mode == "elo":
        result["elo_recall_spearman"] = float(rho)
        result["elo_recall_p"] = float(p)
    else:
        result["absolute_recall_spearman"] = float(rho)
        result["absolute_recall_p"] = float(p)
    return result


async def run_rediscovery(args) -> dict:
    """Run the system on each curated goal, score gold-entity recall + Elo↔recall."""
    import time
    from bench.goalset import load_goalset, validate_rediscovery_goals

    goals = load_goalset(args.goals)
    validate_rediscovery_goals(goals, source=args.goals)
    if not getattr(args, "full", False):
        goals = goals[:args.limit]
    max_tasks = getattr(args, "max_tasks", 30)
    print(f"Loaded {len(goals)} curated goals from '{args.goals}'. "
          f"Running each with max_tasks={max_tasks}...")
    runs: list[BenchRun] = []
    returned_runs: list[_ReturnedRun] = []
    failures: list[dict[str, object]] = []
    failed_attempts: list[_FreshRunError] = []
    cache_counts = _new_cache_counts()
    execution, version, manifest = await _execution_context(
        args, max_tasks=max_tasks,
    )
    run_variant = (
        "no_tournament" if execution.ranking_mode == "absolute" else "full"
    )
    for i, goal in enumerate(goals, 1):
        t0 = time.monotonic()
        try:
            acquired = await _acquire_run(
                args,
                goal,
                label=(
                    "rediscover-absolute"
                    if execution.ranking_mode == "absolute"
                    else "rediscover"
                ),
                variant=run_variant,
                max_tasks=max_tasks,
                execution=execution,
                version=version,
                manifest=manifest,
                cache_counts=cache_counts,
            )
        except _FreshRunError as failed:
            failed_attempts.append(failed)
            exc = failed.cause
            print(f"  [{i}/{len(goals)}] {goal.id}: FAILED ({type(exc).__name__}: {exc}); skipping")
            failures.append({"goal_id": goal.id,
                             "error": f"{type(exc).__name__}: {exc}",
                             "run_returned": False,
                             "cache_state": failed.cache_state,
                             "cache_key": failed.cache_key,
                             "cache_stale_reason":
                                 failed.cache_stale_reason,
                             "db_path": failed.db_path,
                             "n_llm_calls": failed.n_llm_calls,
                             "estimated_tokens": failed.estimated_tokens,
                             "lease_waited": failed.lease_waited,
                             "lease_wait_seconds": failed.lease_wait_seconds,
                             "lease_takeover": failed.lease_takeover})
            continue
        run = acquired.run
        why = acquired.completion_problem
        if why is not None:
            print(
                f"  [{i}/{len(goals)}] {goal.id}: FAILED ({why}); "
                "excluded from scoring"
            )
            returned_runs.append(_ReturnedRun(
                run=run,
                accepted_for_scoring=False,
                rejection_reason=why,
                cache_state=acquired.cache_state,
                cache_key=acquired.cache_key,
                cache_stale_reason=acquired.cache_stale_reason,
                execution_source=acquired.execution_source,
                cache_write_error=acquired.cache_write_error,
                lease_waited=acquired.lease_waited,
                lease_wait_seconds=acquired.lease_wait_seconds,
                lease_takeover=acquired.lease_takeover,
                lease_renewals=acquired.lease_renewals,
            ))
            failures.append({
                "goal_id": goal.id,
                "run_id": run.run_id,
                "error": why,
                "run_returned": True,
                "cache_state": acquired.cache_state,
                "cache_key": acquired.cache_key,
                "cache_stale_reason": acquired.cache_stale_reason,
            })
            continue
        returned_runs.append(_ReturnedRun(
            run=run,
            accepted_for_scoring=True,
            rejection_reason=None,
            cache_state=acquired.cache_state,
            cache_key=acquired.cache_key,
            cache_stale_reason=acquired.cache_stale_reason,
            execution_source=acquired.execution_source,
            cache_write_error=acquired.cache_write_error,
            lease_waited=acquired.lease_waited,
            lease_wait_seconds=acquired.lease_wait_seconds,
            lease_takeover=acquired.lease_takeover,
            lease_renewals=acquired.lease_renewals,
        ))
        rec = None
        from bench.goldset import score_recall
        rec = score_recall(run.hypotheses, goal.gold_entities)
        cache_note = (
            "cache hit"
            if acquired.execution_source == "reused"
            else (
                f"{acquired.cache_state}; fresh"
                if manifest is not None
                else "fresh"
            )
        )
        print(f"  [{i}/{len(goals)}] {goal.id}: {len(run.hypotheses)} hyps, "
              f"pool recall {rec:.2f}, {time.monotonic() - t0:.0f}s "
              f"({cache_note})")
        runs.append(run)

    stats = rediscovery_from_runs(goals, runs)
    accounting = _accounting_metadata(
        returned_runs,
        manifest=manifest,
        cache_counts=cache_counts,
        failed_attempts=failed_attempts,
    )
    return build_report({
        "rediscovery": stats,
        "meta": {"dataset": args.goals, "n_goals": len(goals),
                 "n_attempted": len(goals),
                 "n_runs_returned": len(returned_runs),
                 "n_accepted": len(runs),
                 "n_completed": len(runs), "n_failed": len(failures),
                 "n_scorable": stats["n_goals"],
                 "n_unaccounted_attempts": len(goals) - len(returned_runs),
                 "failures": failures, "system_version": version,
                 "max_tasks": max_tasks, "workers": execution.workers,
                 "provider": execution.provider,
                 "model_fast": execution.model_fast,
                 "model_strong": execution.model_strong,
                 "execution_identity": {
                     "system_version": version,
                     "seed": args.seed,
                     **execution.as_dict(),
                 },
                 **accounting,
                 "runs": [
                     _run_metadata(
                         item.run,
                         accepted_for_scoring=item.accepted_for_scoring,
                         rejection_reason=item.rejection_reason,
                         cache_state=item.cache_state,
                         cache_key=item.cache_key,
                         cache_stale_reason=item.cache_stale_reason,
                         execution_source=item.execution_source,
                         cache_write_error=item.cache_write_error,
                         lease_waited=item.lease_waited,
                         lease_wait_seconds=item.lease_wait_seconds,
                         lease_takeover=item.lease_takeover,
                         lease_renewals=item.lease_renewals,
                     )
                     for item in returned_runs
                 ]},
    })


def _load_concordance_goals(
    dataset: str, limit: int, *, allow_fixture_fallback: bool = False,
) -> tuple[list[BenchGoal], dict]:
    """Load the goal set for a concordance run. 'gpqa-bio' pulls the GPQA-diamond
    biology subset from HuggingFace; a tiny local fixture is the offline fallback."""
    from bench.datasets.gpqa import load_gpqa_hf, load_gpqa_fixture
    if dataset in ("gpqa-bio", "gpqa"):
        try:
            goals = load_gpqa_hf(all_subjects=(dataset == "gpqa"), limit=limit)
            return goals, {
                "requested": dataset,
                "source": "huggingface:hendrydong/gpqa_diamond_mc",
                "fixture_fallback": False,
            }
        except Exception as exc:  # network / HF unavailable → fixture
            if not allow_fixture_fallback:
                from bench.errors import BenchError
                raise BenchError(
                    f"GPQA pull failed ({type(exc).__name__}: {exc}). "
                    "Re-run with --allow-fixture-fallback to use the tiny local "
                    "smoke-test fixture explicitly."
                ) from exc
            print(f"  (HF pull failed: {exc}; using explicitly allowed local fixture)")
            goals = load_gpqa_fixture(
                "bench/datasets/fixtures/gpqa_sample.jsonl"
            )[:limit]
            return goals, {
                "requested": dataset,
                "source": "bench/datasets/fixtures/gpqa_sample.jsonl",
                "fixture_fallback": True,
                "fallback_error": f"{type(exc).__name__}: {exc}",
            }
    raise ValueError(f"Unknown concordance dataset: {dataset!r}")


async def run_concordance(args) -> dict:
    """Run the system on each goal, score answers vs gold, compute concordance.
    Returns the assembled report dict (also written to disk by the caller)."""
    import time

    limit = 198 if getattr(args, "full", False) else args.limit
    max_tasks = getattr(args, "max_tasks", 30)
    min_support = getattr(args, "min_support", 2)
    goals, dataset_provenance = _load_concordance_goals(
        args.dataset,
        limit,
        allow_fixture_fallback=getattr(args, "allow_fixture_fallback", False),
    )
    print(f"Loaded {len(goals)} goals from '{args.dataset}'. "
          f"Running each with max_tasks={max_tasks} (seed={args.seed})...")

    runs: list[BenchRun] = []
    returned_runs: list[_ReturnedRun] = []
    failures: list[dict[str, object]] = []
    failed_attempts: list[_FreshRunError] = []
    cache_counts = _new_cache_counts()
    execution, version, manifest = await _execution_context(
        args, max_tasks=max_tasks,
    )
    for i, goal in enumerate(goals, 1):
        t0 = time.monotonic()
        try:
            acquired = await _acquire_run(
                args,
                goal,
                label="concordance",
                variant="full",
                max_tasks=max_tasks,
                execution=execution,
                version=version,
                manifest=manifest,
                cache_counts=cache_counts,
            )
        except _FreshRunError as failed:
            # One bad model-backed goal must not sink the whole reading.
            failed_attempts.append(failed)
            exc = failed.cause
            print(f"  [{i}/{len(goals)}] {goal.id}: FAILED ({type(exc).__name__}: {exc}); skipping")
            failures.append({"goal_id": goal.id,
                             "error": f"{type(exc).__name__}: {exc}",
                             "run_returned": False,
                             "cache_state": failed.cache_state,
                             "cache_key": failed.cache_key,
                             "cache_stale_reason":
                                 failed.cache_stale_reason,
                             "db_path": failed.db_path,
                             "n_llm_calls": failed.n_llm_calls,
                             "estimated_tokens": failed.estimated_tokens,
                             "lease_waited": failed.lease_waited,
                             "lease_wait_seconds": failed.lease_wait_seconds,
                             "lease_takeover": failed.lease_takeover})
            continue
        run = acquired.run
        why = acquired.completion_problem
        if why is not None:
            print(
                f"  [{i}/{len(goals)}] {goal.id}: FAILED ({why}); "
                "excluded from scoring"
            )
            returned_runs.append(_ReturnedRun(
                run=run,
                accepted_for_scoring=False,
                rejection_reason=why,
                cache_state=acquired.cache_state,
                cache_key=acquired.cache_key,
                cache_stale_reason=acquired.cache_stale_reason,
                execution_source=acquired.execution_source,
                cache_write_error=acquired.cache_write_error,
                lease_waited=acquired.lease_waited,
                lease_wait_seconds=acquired.lease_wait_seconds,
                lease_takeover=acquired.lease_takeover,
                lease_renewals=acquired.lease_renewals,
            ))
            failures.append({
                "goal_id": goal.id,
                "run_id": run.run_id,
                "error": why,
                "run_returned": True,
                "cache_state": acquired.cache_state,
                "cache_key": acquired.cache_key,
                "cache_stale_reason": acquired.cache_stale_reason,
            })
            continue
        returned_runs.append(_ReturnedRun(
            run=run,
            accepted_for_scoring=True,
            rejection_reason=None,
            cache_state=acquired.cache_state,
            cache_key=acquired.cache_key,
            cache_stale_reason=acquired.cache_stale_reason,
            execution_source=acquired.execution_source,
            cache_write_error=acquired.cache_write_error,
            lease_waited=acquired.lease_waited,
            lease_wait_seconds=acquired.lease_wait_seconds,
            lease_takeover=acquired.lease_takeover,
            lease_renewals=acquired.lease_renewals,
        ))
        elos = [h.elo_rating for h in run.hypotheses]
        spread = (max(elos) - min(elos)) if elos else 0.0
        cache_note = (
            "cache hit"
            if acquired.execution_source == "reused"
            else (
                f"{acquired.cache_state}; fresh"
                if manifest is not None
                else "fresh"
            )
        )
        print(f"  [{i}/{len(goals)}] {goal.id}: {len(run.hypotheses)} hyps, "
              f"Elo spread {spread:.0f}, {time.monotonic() - t0:.0f}s "
              f"({cache_note})")
        runs.append(run)

    # Difficulty correction (the paper's red line): sample N base-model responses
    # per question so blue_minus_red_spread can subtract per-question difficulty
    # from the Elo↔correctness signal. Off by default (cost); paper used 32.
    ref_n = getattr(args, "reference_samples", 0)
    reference_accuracy: dict[str, float] = {}
    if ref_n > 0:
        from bench.reference import reference_accuracy_for_goals, backend_generate
        from bench.runner import _CountingBackend, _load_yaml_config, configure_backend
        from tools.llm import close_backend, make_backend
        n_mcq = sum(1 for g in goals if g.gold_answer)
        print(f"Sampling {ref_n} base-model reference responses/question for "
              f"difficulty correction ({n_mcq} MCQ goals, ~{ref_n * n_mcq} extra calls)...")
        reference_cfg = _load_yaml_config("config.yaml")
        selected_provider, _ = configure_backend(
            reference_cfg,
            provider=getattr(args, "provider", None),
            model_fast=getattr(args, "model_fast", None),
            model_strong=getattr(args, "model_strong", None),
        )
        reference_backend = make_backend(
            reference_cfg, provider_override=selected_provider,
        )
        backend = _CountingBackend(reference_backend)
        try:
            use_strong = getattr(args, "reference_strong", False)
            reference_accuracy = await reference_accuracy_for_goals(
                goals,
                backend_generate(backend, use_strong=use_strong),
                n_samples=ref_n,
            )
            reference_calls = backend.call_count
            reference_estimated_tokens = backend.estimated_tokens
        finally:
            await close_backend(reference_backend)
    else:
        reference_calls = 0
        reference_estimated_tokens = 0

    stats = concordance_from_runs(goals, runs, reference_accuracy=reference_accuracy,
                                  min_support=min_support)
    accounting = _accounting_metadata(
        returned_runs,
        manifest=manifest,
        cache_counts=cache_counts,
        failed_attempts=failed_attempts,
        reference_calls=reference_calls,
        reference_estimated_tokens=reference_estimated_tokens,
    )
    return build_report({
        "concordance": stats,
        "meta": {"dataset": args.dataset,
                 "dataset_provenance": dataset_provenance,
                 "n_questions": len(goals),
                 "n_attempted": len(goals),
                 "n_runs_returned": len(returned_runs),
                 "n_accepted": len(runs),
                 "n_completed": len(runs), "n_failed": len(failures),
                 "n_scorable": len(
                     {
                         run.goal_id
                         for run in runs
                         if run.hypotheses
                         and any(
                             goal.id == run.goal_id and goal.gold_answer
                             for goal in goals
                         )
                     }
                 ),
                 "n_unaccounted_attempts": len(goals) - len(returned_runs),
                 "failures": failures, "system_version": version,
                 "max_tasks": max_tasks, "workers": execution.workers,
                 "provider": execution.provider,
                 "model_fast": execution.model_fast,
                 "model_strong": execution.model_strong,
                 "execution_identity": {
                     "system_version": version,
                     "seed": args.seed,
                     **execution.as_dict(),
                 },
                 "reference_samples": ref_n,
                 "reference_llm_calls": reference_calls,
                 "reference_estimated_tokens": reference_estimated_tokens,
                 **accounting,
                 "runs": [
                     _run_metadata(
                         item.run,
                         accepted_for_scoring=item.accepted_for_scoring,
                         rejection_reason=item.rejection_reason,
                         cache_state=item.cache_state,
                         cache_key=item.cache_key,
                         cache_stale_reason=item.cache_stale_reason,
                         execution_source=item.execution_source,
                         cache_write_error=item.cache_write_error,
                         lease_waited=item.lease_waited,
                         lease_wait_seconds=item.lease_wait_seconds,
                         lease_takeover=item.lease_takeover,
                         lease_renewals=item.lease_renewals,
                     )
                     for item in returned_runs
                 ]},
    })


async def run_sweep(args) -> dict:
    """Run an explicit provider/model × worker-count × seed benchmark grid."""
    import copy
    from pathlib import Path

    from bench.sweep import build_cells

    ranking_mode = getattr(args, "ranking_mode", "elo")
    if args.benchmark == "concordance" and ranking_mode != "elo":
        from bench.errors import BenchError

        raise BenchError("concordance sweep cells require ranking_mode='elo'")

    cells = build_cells(args.worker_counts, args.models, args.seeds)
    # All cells address the same configured manifest. Their goal fingerprint,
    # exact execution settings, seed, and system version remain in each key.
    shared_manifest = await _manifest_for_args(args)
    rows: list[dict] = []
    out_prefix = getattr(args, "out", "bench_sweep")
    for i, cell in enumerate(cells, 1):
        print(f"\n=== Sweep cell {i}/{len(cells)}: {cell.id} ===")
        cell_args = copy.copy(args)
        cell_args.command = args.benchmark
        cell_args.provider = cell.provider
        cell_args.model_fast = cell.model_fast
        cell_args.model_strong = cell.model_strong
        cell_args.workers = cell.workers
        cell_args.seed = cell.seed
        cell_args.out = f"{out_prefix}.{cell.id}"
        cell_args._manifest_instance = shared_manifest
        try:
            if args.benchmark == "rediscover":
                report = await run_rediscovery(cell_args)
                rediscovery = report["rediscovery"]
                quality = {
                    "ranking_mode": rediscovery.get(
                        "ranking_mode", ranking_mode
                    ),
                    "mean_pool_recall": rediscovery["mean_pool_recall"],
                    "mean_top1_recall": rediscovery["mean_top1_recall"],
                    "ranking_recall_spearman": rediscovery.get(
                        "ranking_recall_spearman",
                        rediscovery.get("elo_recall_spearman"),
                    ),
                }
            else:
                report = await run_concordance(cell_args)
                concordance = report["concordance"]
                quality = {
                    # Headline: within-question logistic; Spearman is descriptive.
                    "logistic_coef": concordance["logistic_coef"],
                    "logistic_p": concordance["logistic_p"],
                    "logistic_n_rows": concordance.get("logistic_n_rows"),
                    "logistic_n_questions": concordance.get("logistic_n_questions"),
                    "concordance_pass": report.get("verdicts", {}).get("concordance_pass"),
                    "spearman_rho": concordance["spearman_rho"],
                    "top1_accuracy": concordance["top1_accuracy"],
                }
            report_path = f"{cell_args.out}.json"
            Path(report_path).write_text(to_json(report))
            meta = report.get("meta", {})
            failed = int(meta.get("n_failed", 0))
            completed = int(meta.get("n_completed", 0))
            scorable = int(meta.get("n_scorable", completed))
            partial_allowed = bool(getattr(args, "allow_partial", False))
            status = (
                "complete"
                if (
                    completed > 0
                    and scorable > 0
                    and (failed == 0 or partial_allowed)
                )
                else "failed"
            )
            rows.append({
                "cell_id": cell.id, "status": status,
                "provider": cell.provider, "model_fast": cell.model_fast,
                "model_strong": cell.model_strong, "workers": cell.workers,
                "seed": cell.seed, "quality": quality,
                "n_attempted": meta.get("n_attempted", completed + failed),
                "n_accepted": meta.get("n_accepted", completed),
                "n_completed": completed, "n_failed": failed,
                "n_scorable": scorable,
                "n_llm_calls": meta.get("n_llm_calls", 0),
                "estimated_tokens": meta.get("estimated_tokens", 0),
                "fresh_llm_calls": meta.get(
                    "fresh_llm_calls", meta.get("n_llm_calls", 0)
                ),
                "reused_llm_calls": meta.get("reused_llm_calls", 0),
                "fresh_estimated_tokens": meta.get(
                    "fresh_estimated_tokens",
                    meta.get("estimated_tokens", 0),
                ),
                "reused_estimated_tokens": meta.get(
                    "reused_estimated_tokens", 0,
                ),
                "n_fresh_runs": meta.get("n_fresh_runs", completed),
                "n_reused_runs": meta.get("n_reused_runs", 0),
                "execution_identity": meta.get("execution_identity", {}),
                "cache": meta.get("cache", {
                    "enabled": shared_manifest is not None,
                    "manifest": getattr(shared_manifest, "path", None),
                    "hits": 0,
                    "misses": 0,
                    "stale": 0,
                }),
                "report_path": report_path,
            })
        except Exception as exc:
            rows.append({
                "cell_id": cell.id, "status": "failed",
                "provider": cell.provider, "model_fast": cell.model_fast,
                "model_strong": cell.model_strong, "workers": cell.workers,
                "seed": cell.seed,
                "error": f"{type(exc).__name__}: {exc}",
            })
    n_complete = sum(r["status"] == "complete" for r in rows)
    cache_hits = sum(
        int(row.get("cache", {}).get("hits", 0)) for row in rows
    )
    cache_misses = sum(
        int(row.get("cache", {}).get("misses", 0)) for row in rows
    )
    cache_stale = sum(
        int(row.get("cache", {}).get("stale", 0)) for row in rows
    )
    return {
        "sweep": {
            "benchmark": args.benchmark,
            "n_cells": len(rows),
            "n_completed": n_complete,
            "n_failed": len(rows) - n_complete,
            "n_llm_calls": sum(
                int(row.get("n_llm_calls", 0)) for row in rows
            ),
            "estimated_tokens": sum(
                int(row.get("estimated_tokens", 0)) for row in rows
            ),
            "fresh_llm_calls": sum(
                int(row.get("fresh_llm_calls", 0)) for row in rows
            ),
            "reused_llm_calls": sum(
                int(row.get("reused_llm_calls", 0)) for row in rows
            ),
            "fresh_estimated_tokens": sum(
                int(row.get("fresh_estimated_tokens", 0)) for row in rows
            ),
            "reused_estimated_tokens": sum(
                int(row.get("reused_estimated_tokens", 0)) for row in rows
            ),
            "cache": {
                "enabled": shared_manifest is not None,
                "manifest": getattr(shared_manifest, "path", None),
                "hits": cache_hits,
                "misses": cache_misses,
                "stale": cache_stale,
            },
            "cells": rows,
        }
    }


_SUBSCRIPTION_PROVIDERS = {"claude-code", "codex"}
_API_PROVIDERS = {"anthropic"}
_CUSTOM_PROVIDERS = {"openai-compatible", "openai_compatible"}


def _provider_billing_route(provider: str) -> str:
    """Classify only what the harness can know from the selected adapter."""
    if provider in _SUBSCRIPTION_PROVIDERS:
        return "subscription"
    if provider in _API_PROVIDERS:
        return "api"
    if provider in _CUSTOM_PROVIDERS:
        return "custom"
    return "custom"


def _auto_billing_route(providers: set[str]) -> str:
    routes = {_provider_billing_route(provider) for provider in providers}
    return next(iter(routes)) if len(routes) == 1 else "mixed"


async def run_command(args) -> int:
    """Cost-gate and dispatch one of the benchmark commands exposed by the CLI."""
    from pathlib import Path
    from bench.cost import estimate_cost
    from bench.cli import _confirm
    from bench.runner import _load_yaml_config
    from tools.llm import load_backend_environment, resolve_provider

    # Resolve the same dotenv/config/CLI precedence before showing the cost gate
    # that run_system will use after approval. Pin non-sweep commands to that
    # result so the report and execution cannot drift if the environment changes.
    load_backend_environment()

    runtime_config = _load_yaml_config("config.yaml")
    limit = 198 if getattr(args, "full", False) else args.limit
    sweep_cells = None
    if args.command == "sweep":
        from bench.sweep import build_cells
        sweep_cells = build_cells(args.worker_counts, args.models, args.seeds)
        effective_providers = {cell.provider for cell in sweep_cells}
    else:
        effective_provider = resolve_provider(
            runtime_config,
            getattr(args, "provider", None),
        )
        args.provider = effective_provider
        effective_providers = {effective_provider}
    judge_provider = getattr(args, "judge_provider", None)
    if (
        getattr(args, "ranking_mode", "elo") == "absolute"
        and judge_provider is not None
    ):
        effective_providers.add(
            resolve_provider(runtime_config, judge_provider)
        )
    est = estimate_cost(C=limit, a=0, v=1,
                        calls_per_run=getattr(args, "max_tasks", 30),
                        ref_samples_per_goal=getattr(args, "reference_samples", 0))
    if sweep_cells:
        # Each cell repeats the same paired goal set; expose the real multiplier
        # before a single model call is made.
        est = estimate_cost(C=limit * len(sweep_cells), a=0, v=1,
                            calls_per_run=getattr(args, "max_tasks", 30),
                            ref_samples_per_goal=getattr(args, "reference_samples", 0))
    inferred_billing_backend = _auto_billing_route(effective_providers)
    billing_backend = args.backend
    if billing_backend == "auto":
        billing_backend = inferred_billing_backend
    elif billing_backend != inferred_billing_backend:
        print(
            f"WARNING: billing label {billing_backend!r} overrides the "
            f"{inferred_billing_backend!r} route inferred from the selected "
            "provider adapter(s)."
        )
    if not _confirm(est, backend=billing_backend, assume_yes=args.yes):
        print("Aborted before spending tokens.")
        return 1

    out = getattr(args, "out", "bench_report")
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    if args.command == "sweep":
        report = await run_sweep(args)
        Path(f"{out}.json").write_text(to_json(report))
        s = report["sweep"]
        print(f"\n=== Sweep ===")
        print(f"  benchmark={s['benchmark']}  cells={s['n_cells']}  "
              f"completed={s['n_completed']}  failed={s['n_failed']}")
        print(f"  report → {out}.json")
        return 0 if s["n_failed"] == 0 else 2

    if args.command == "concordance":
        report = await run_concordance(args)
        Path(f"{out}.json").write_text(to_json(report))
        c = report["concordance"]; v = report["verdicts"]
        print(f"\n=== Concordance ===")
        print(f"  within-question logistic coef={c['logistic_coef']:+.3f}/SD "
              f"(LRT p={c['logistic_p']:.4f})  fitted on {c.get('logistic_n_rows', '?')} rows "
              f"across {c.get('logistic_n_questions', '?')} informative tournaments"
              + ("  [SEPARATED: magnitude not meaningful]" if c.get("logistic_separated") else ""))
        print(f"  rows={c['n_rows']}  buckets={c['n_buckets']}  "
              f"descriptive Spearman rho={c['spearman_rho']:.3f} (p={c['spearman_p']:.4f})  "
              f"top1_acc={c['top1_accuracy']:.3f}")
        print(f"  verdict: {'PASS — Elo tracks correctness' if v['concordance_pass'] else 'not established (small N or weak signal)'}")
        print(f"  report → {out}.json")
        meta = report.get("meta", {})
        failed = meta.get("n_failed", 0)
        completed = meta.get("n_completed", 0)
        scorable = meta.get("n_scorable", 0)
        if (
            completed == 0
            or scorable == 0
            or c.get("n_rows", 0) == 0
            or (failed and not getattr(args, "allow_partial", False))
        ):
            print(
                f"  BENCHMARK FAILED: completed={completed}, failed={failed}, "
                f"scorable={scorable}"
            )
            return 2
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
        ranking_label = (
            "Elo" if r.get("ranking_mode", "elo") == "elo"
            else "Absolute score"
        )
        ranking_rho = r.get(
            "ranking_recall_spearman", r.get("elo_recall_spearman")
        )
        ranking_p = r.get("ranking_recall_p", r.get("elo_recall_p"))
        print(
            f"  {ranking_label}↔recall Spearman="
            f"{ranking_rho:.3f} "
            f"(p={ranking_p:.3f}, n={r['n_hyps_scored']}) "
            f"— >0 means higher-ranked hypotheses hold more gold biology"
        )
        print(f"  report → {out}.json")
        meta = report.get("meta", {})
        failed = meta.get("n_failed", 0)
        completed = meta.get("n_completed", 0)
        scorable = meta.get("n_scorable", r.get("n_goals", 0))
        if (
            completed == 0
            or scorable == 0
            or r.get("n_goals", 0) == 0
            or r.get("n_hyps_scored", 0) == 0
            or (failed and not getattr(args, "allow_partial", False))
        ):
            print(
                f"  BENCHMARK FAILED: completed={completed}, failed={failed}, "
                f"scorable={scorable}"
            )
            return 2
        return 0

    print(f"Unsupported benchmark command: {args.command!r}.")
    return 2
