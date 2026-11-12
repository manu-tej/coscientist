from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import aiosqlite

if TYPE_CHECKING:
    from bench.goalset import BenchGoal
    from bench.runner import BenchmarkExecution, BenchRun


_CREATE = """
CREATE TABLE IF NOT EXISTS manifest_v3 (
    goal_id TEXT NOT NULL,
    goal_fingerprint TEXT NOT NULL,
    variant TEXT NOT NULL,
    system_version TEXT NOT NULL,
    seed INTEGER NOT NULL,
    execution_key TEXT NOT NULL,
    execution_json TEXT NOT NULL,
    run_id TEXT NOT NULL,
    db_path TEXT NOT NULL,
    -- IMPORTANT 6 (deliberately kept, not relaxed): a manifest row is a cache
    -- entry promising reusable, scorable results. A fully-quarantined run has
    -- zero active hypotheses to reuse regardless of why the pool is empty, so
    -- it stays uncacheable and this CHECK stays > 0. What changed is that
    -- run_completion_problem() below no longer reports that case with the
    -- same generic "no active hypotheses" text used for a real failure --
    -- see its `n_quarantined` branch. BenchRun.n_quarantined (bench/runner.py)
    -- and the run report dict (bench/orchestrate.py::_run_metadata) carry the
    -- disclosure that this table cannot.
    n_hypotheses INTEGER NOT NULL CHECK (n_hypotheses > 0),
    n_llm_calls INTEGER NOT NULL CHECK (n_llm_calls >= 0),
    estimated_tokens INTEGER NOT NULL CHECK (estimated_tokens >= 0),
    wall_clock_s REAL NOT NULL CHECK (wall_clock_s >= 0),
    run_fingerprint TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('complete', 'stale')),
    stale_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (
        goal_id, goal_fingerprint, variant, system_version, seed, execution_key
    )
);

CREATE TABLE IF NOT EXISTS manifest_leases_v1 (
    cache_key TEXT PRIMARY KEY,
    owner_token_digest TEXT NOT NULL,
    expires_at REAL NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""

_SOURCE_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_PATHS = (
    "agents",
    "bench/absolute.py",
    "bench/judge.py",
    "bench/runner.py",
    "core",
    "tools",
    "prompts",
    "config.yaml",
    "pyproject.toml",
    "uv.lock",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _package_version() -> str:
    try:
        version = importlib.metadata.version("coscientist")
    except importlib.metadata.PackageNotFoundError:
        version = "source"
    return f"package-{version}"


def _source_tree_digest(root: Path) -> str:
    """Digest executable package data without depending on the caller's Git tree."""
    digest = hashlib.sha256()
    found = False
    for relative in _SOURCE_PATHS:
        candidate = root / relative
        paths = [candidate] if candidate.is_file() else (
            sorted(path for path in candidate.rglob("*") if path.is_file())
            if candidate.is_dir()
            else []
        )
        for path in paths:
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            found = True
            name = str(path.relative_to(root))
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()[:12] if found else "unknown"


def system_version() -> str:
    """Identify run-producing source, including uncommitted changes.

    A repository commit is too broad for a paid-run cache: changing only a
    scorer or report renderer must not invalidate model output. Hash the
    execution modules/assets directly, which also works from an installed
    wheel and is independent of the caller's current Git checkout.
    """
    digest = _source_tree_digest(_SOURCE_ROOT)
    if digest == "unknown":
        return _package_version()
    return f"execution-{digest}"


def goal_fingerprint(
    goal: "BenchGoal", *, include_domain: bool = False,
) -> str:
    """Hash only input sent to the model-backed system.

    Gold answers/entities and dataset metadata are analysis inputs. Keeping
    them out of the execution key allows one paid captured run to support
    multiple scoring passes without letting a changed research prompt reuse
    stale hypotheses.
    """
    prompt_inputs = {"goal": goal.goal}
    if include_domain:
        prompt_inputs["domain"] = goal.domain
    return _sha256_text(_canonical_json(prompt_inputs))


@dataclass(frozen=True)
class RunCacheKey:
    goal_id: str
    goal_fingerprint: str
    variant: str
    system_version: str
    seed: int
    execution_key: str
    execution_json: str

    @property
    def id(self) -> str:
        return _sha256_text(_canonical_json({
            "goal_id": self.goal_id,
            "goal_fingerprint": self.goal_fingerprint,
            "variant": self.variant,
            "system_version": self.system_version,
            "seed": self.seed,
            "execution_key": self.execution_key,
        }))


def make_run_cache_key(
    goal: "BenchGoal",
    *,
    variant: str,
    system_version_value: str,
    seed: int,
    execution: "BenchmarkExecution",
) -> RunCacheKey:
    execution_json = _canonical_json(execution.as_dict())
    return RunCacheKey(
        goal_id=goal.id,
        goal_fingerprint=goal_fingerprint(
            goal, include_domain=execution.ranking_mode == "absolute"
        ),
        variant=variant,
        system_version=system_version_value,
        seed=seed,
        execution_key=_sha256_text(execution_json),
        execution_json=execution_json,
    )


@dataclass
class CacheLookup:
    state: Literal["hit", "miss", "stale"]
    run: Optional["BenchRun"] = None
    reason: Optional[str] = None
    cache_key: str = ""


@dataclass(frozen=True)
class LeaseClaim:
    """Result of atomically checking cache readiness and claiming work.

    ``owner_token`` is a short-lived capability returned only to the process
    that owns a lease. Its raw value is deliberately excluded from reprs and
    is never persisted to the manifest database.
    """

    state: Literal["ready", "acquired", "waiting"]
    owner_token: Optional[str] = field(default=None, repr=False)
    expires_at: Optional[float] = None


def _metadata_for_run(run: "BenchRun") -> dict[str, Any]:
    metadata = {
        "agent_task_counts": run.agent_task_counts,
        "agent_failures": run.agent_failures,
        "agent_task_trace": run.agent_task_trace,
        "workers": run.workers,
        "backend": run.backend,
        "models": run.models,
        "max_backend_concurrency": run.max_backend_concurrency,
    }
    # Preserve v3 Elo fingerprints/cache rows. Absolute execution did not exist
    # when those rows were written, so only the new mode carries extra metadata.
    if run.ranking_mode == "absolute":
        metadata["ranking_mode"] = "absolute"
        metadata["absolute_scorer"] = run.absolute_scorer
    return metadata


def _run_fingerprint(run: "BenchRun") -> str:
    hypotheses = []
    for hypothesis in run.hypotheses:
        captured = {
            "id": hypothesis.id,
            "text": hypothesis.text,
            "summary": hypothesis.summary,
            "elo_rating": hypothesis.elo_rating,
            "created_at": hypothesis.created_at,
            "generation_method": hypothesis.generation_method,
            "elo_trajectory": hypothesis.elo_trajectory,
        }
        if run.ranking_mode == "absolute":
            captured.update({
                "absolute_score": hypothesis.absolute_score,
                "absolute_axis_scores": hypothesis.absolute_axis_scores,
                "absolute_scorer_id": hypothesis.absolute_scorer_id,
            })
        hypotheses.append(captured)
    payload = {
        "goal_id": run.goal_id,
        "variant": run.variant,
        "run_id": run.run_id,
        "n_llm_calls": run.n_llm_calls,
        "estimated_tokens": run.estimated_tokens,
        "wall_clock_s": run.wall_clock_s,
        "metadata": _metadata_for_run(run),
        "hypotheses": hypotheses,
    }
    return _sha256_text(_canonical_json(payload))


def run_completion_problem(
    run: "BenchRun",
    key: RunCacheKey,
    *,
    expected_tasks: int,
) -> Optional[str]:
    try:
        execution = json.loads(key.execution_json)
    except (json.JSONDecodeError, TypeError) as exc:
        return f"invalid execution metadata: {exc}"
    if _sha256_text(key.execution_json) != key.execution_key:
        return "execution metadata hash mismatch"
    if execution.get("max_tasks") != expected_tasks:
        return (
            "expected task budget does not match execution identity "
            f"({expected_tasks!r} != {execution.get('max_tasks')!r})"
        )
    if run.goal_id != key.goal_id:
        return f"goal mismatch ({run.goal_id!r} != {key.goal_id!r})"
    if run.variant != key.variant:
        return f"variant mismatch ({run.variant!r} != {key.variant!r})"
    expected_ranking_mode = execution.get("ranking_mode", "elo")
    if run.ranking_mode != expected_ranking_mode:
        return (
            f"ranking mode mismatch ({run.ranking_mode!r} != "
            f"{expected_ranking_mode!r})"
        )
    if expected_ranking_mode == "absolute":
        from core.models import ABSOLUTE_SCORE_AXES

        expected_scorer = execution.get("absolute_scorer")
        if not isinstance(expected_scorer, dict):
            return "absolute execution is missing scorer metadata"
        if run.absolute_scorer != expected_scorer:
            return "absolute scorer metadata mismatch"
        scorer_id = expected_scorer.get("scorer_id")
        if not isinstance(scorer_id, str) or not scorer_id:
            return "absolute scorer metadata has no scorer_id"
        for hypothesis in (
            run.hypotheses if isinstance(run.hypotheses, list) else []
        ):
            if (
                hypothesis.absolute_score is None
                or isinstance(hypothesis.absolute_score, bool)
                or not isinstance(hypothesis.absolute_score, (int, float))
                or not math.isfinite(float(hypothesis.absolute_score))
                or not 1.0 <= float(hypothesis.absolute_score) <= 5.0
            ):
                return f"hypothesis {hypothesis.id!r} has no valid absolute score"
            if hypothesis.absolute_scorer_id != scorer_id:
                return f"hypothesis {hypothesis.id!r} has a mismatched scorer"
            if set(hypothesis.absolute_axis_scores) != set(ABSOLUTE_SCORE_AXES):
                return f"hypothesis {hypothesis.id!r} has invalid absolute axes"
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                or value > 5
                for value in hypothesis.absolute_axis_scores.values()
            ):
                return f"hypothesis {hypothesis.id!r} has invalid absolute values"
            try:
                from bench.judge import weighted_total

                expected_total = weighted_total(
                    hypothesis.absolute_axis_scores,
                    expected_scorer.get("weights"),
                )
            except Exception as exc:
                return f"invalid absolute scorer weights: {exc}"
            if not math.isclose(
                float(hypothesis.absolute_score),
                expected_total,
                abs_tol=1e-9,
            ):
                return f"hypothesis {hypothesis.id!r} has an inconsistent total"
            if hypothesis.elo_trajectory:
                return f"absolute hypothesis {hypothesis.id!r} has an Elo trajectory"
    elif expected_ranking_mode != "elo":
        return f"unknown ranking mode {expected_ranking_mode!r}"
    if not isinstance(run.run_id, str) or not run.run_id:
        return "missing run_id"
    if not isinstance(run.agent_failures, list) or any(
        not isinstance(failure, dict) for failure in run.agent_failures
    ):
        return "invalid agent failures"
    if run.agent_failures:
        return f"{len(run.agent_failures)} agent task failure(s)"
    if not isinstance(run.hypotheses, list) or not run.hypotheses:
        # IMPORTANT 6: a fully-quarantined run completed cleanly -- verification
        # caught a real problem in every hypothesis, which is a legitimate
        # scientific outcome, not the same failure as generating nothing at
        # all (a crash, a misconfigured goal, ...). Both remain uncacheable
        # here (there is nothing scorable to reuse either way -- see the
        # accompanying report for why the CHECK(n_hypotheses > 0) schema
        # constraint is left as-is rather than relaxed), but the *reason*
        # surfaced to report consumers must not conflate the two.
        n_quarantined = getattr(run, "n_quarantined", 0)
        if n_quarantined:
            return (
                f"all {n_quarantined} generated hypothesis(es) were "
                "quarantined by verification (no run failure -- nothing "
                "scorable to cache)"
            )
        return "no active hypotheses"
    if (
        isinstance(run.n_llm_calls, bool)
        or not isinstance(run.n_llm_calls, int)
        or run.n_llm_calls < 0
    ):
        return "invalid LLM call count"
    if (
        isinstance(run.estimated_tokens, bool)
        or not isinstance(run.estimated_tokens, int)
        or run.estimated_tokens < 0
    ):
        return "invalid token estimate"
    if (
        isinstance(run.wall_clock_s, bool)
        or not isinstance(run.wall_clock_s, (int, float))
        or not math.isfinite(float(run.wall_clock_s))
        or run.wall_clock_s < 0
    ):
        return "invalid wall-clock duration"
    if not isinstance(run.agent_task_counts, dict) or any(
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for count in run.agent_task_counts.values()
    ):
        return "invalid agent task counts"
    completed_tasks = sum(run.agent_task_counts.values())
    if completed_tasks != expected_tasks:
        return (
            f"incomplete task budget ({completed_tasks}/{expected_tasks} tasks)"
        )
    if not isinstance(run.agent_task_trace, list) or any(
        not isinstance(trace, dict) for trace in run.agent_task_trace
    ):
        return "invalid agent task trace"
    if len(run.agent_task_trace) != expected_tasks:
        return (
            f"incomplete task trace ({len(run.agent_task_trace)}/"
            f"{expected_tasks} tasks)"
        )
    bad_statuses = [
        trace.get("status")
        for trace in run.agent_task_trace
        if trace.get("status") != "succeeded"
    ]
    if bad_statuses:
        return f"non-success task trace status(es): {bad_statuses}"
    if (
        isinstance(run.workers, bool)
        or not isinstance(run.workers, int)
        or run.workers < 1
    ):
        return "invalid worker count"
    if run.workers != execution.get("workers"):
        return f"worker mismatch ({run.workers!r} != {execution.get('workers')!r})"
    if not isinstance(run.backend, str) or not run.backend:
        return "invalid provider"
    if run.backend != execution.get("provider"):
        return (
            f"provider mismatch ({run.backend!r} != "
            f"{execution.get('provider')!r})"
        )
    expected_models = {
        "fast": execution.get("model_fast", ""),
        "strong": execution.get("model_strong", ""),
    }
    if not isinstance(run.models, dict) or any(
        not isinstance(name, str) or not isinstance(model, str)
        for name, model in run.models.items()
    ):
        return "invalid model metadata"
    if run.models != expected_models:
        return f"model mismatch ({run.models!r} != {expected_models!r})"
    return None


class Manifest:
    def __init__(self, path: str):
        self.path = str(Path(path).resolve())

    async def init(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path, timeout=30.0) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA busy_timeout=30000")
            await db.executescript(_CREATE)
            await db.commit()

    @staticmethod
    def _where(key: RunCacheKey) -> tuple[Any, ...]:
        return (
            key.goal_id,
            key.goal_fingerprint,
            key.variant,
            key.system_version,
            key.seed,
            key.execution_key,
        )

    @staticmethod
    def _lease_deadline(lease_seconds: float, *, now: float) -> float:
        if (
            isinstance(lease_seconds, bool)
            or not isinstance(lease_seconds, (int, float))
            or not math.isfinite(float(lease_seconds))
            or lease_seconds <= 0
        ):
            raise ValueError("lease_seconds must be a finite positive number")
        return now + float(lease_seconds)

    @staticmethod
    def _owner_token_digest(owner_token: str) -> str:
        if not isinstance(owner_token, str) or not owner_token:
            raise ValueError("owner_token must be a non-empty string")
        return _sha256_text(owner_token)

    async def claim(
        self,
        key: RunCacheKey,
        *,
        lease_seconds: float,
        owner_token: Optional[str] = None,
    ) -> LeaseClaim:
        """Atomically return cached readiness or acquire one execution lease.

        A write transaction makes the completed-row check and lease mutation a
        single ordering point across processes. A caller may supply a token to
        make a retried claim idempotent. Other callers receive ``waiting`` until
        the lease expires; exactly one of them can then steal it.
        """
        # Validate before taking a database write lock.
        self._lease_deadline(lease_seconds, now=0.0)
        if owner_token is not None:
            self._owner_token_digest(owner_token)

        async with aiosqlite.connect(self.path, timeout=30.0) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA busy_timeout=30000")
            await db.execute("BEGIN IMMEDIATE")
            try:
                async with db.execute(
                    """SELECT 1 FROM manifest_v3
                       WHERE goal_id=? AND goal_fingerprint=? AND variant=?
                         AND system_version=? AND seed=? AND execution_key=?
                         AND status='complete'""",
                    self._where(key),
                ) as cursor:
                    complete = await cursor.fetchone()
                if complete is not None:
                    # A finished run supersedes any outstanding worker lease.
                    await db.execute(
                        "DELETE FROM manifest_leases_v1 WHERE cache_key=?",
                        (key.id,),
                    )
                    await db.commit()
                    return LeaseClaim(state="ready")

                token = owner_token or secrets.token_urlsafe(32)
                token_digest = self._owner_token_digest(token)
                now = time.time()
                expires_at = self._lease_deadline(lease_seconds, now=now)
                async with db.execute(
                    """SELECT owner_token_digest, expires_at
                       FROM manifest_leases_v1 WHERE cache_key=?""",
                    (key.id,),
                ) as cursor:
                    row = await cursor.fetchone()

                if row is not None and row["expires_at"] > now:
                    if row["owner_token_digest"] == token_digest:
                        cursor = await db.execute(
                            """UPDATE manifest_leases_v1
                               SET expires_at=?, updated_at=?
                               WHERE cache_key=? AND owner_token_digest=?
                                 AND expires_at=?""",
                            (
                                expires_at,
                                now,
                                key.id,
                                token_digest,
                                row["expires_at"],
                            ),
                        )
                        if cursor.rowcount != 1:
                            raise RuntimeError(
                                "idempotent lease renewal lost its compare-and-set"
                            )
                        await db.commit()
                        return LeaseClaim(
                            state="acquired",
                            owner_token=token,
                            expires_at=expires_at,
                        )
                    await db.commit()
                    return LeaseClaim(
                        state="waiting",
                        expires_at=row["expires_at"],
                    )

                if row is None:
                    cursor = await db.execute(
                        """INSERT INTO manifest_leases_v1 (
                               cache_key, owner_token_digest, expires_at,
                               created_at, updated_at
                           ) VALUES (?, ?, ?, ?, ?)""",
                        (key.id, token_digest, expires_at, now, now),
                    )
                else:
                    # Include the observed owner and expiry in the predicate so
                    # a future refactor cannot turn lease stealing into a
                    # blind overwrite.
                    cursor = await db.execute(
                        """UPDATE manifest_leases_v1
                           SET owner_token_digest=?, expires_at=?, updated_at=?
                           WHERE cache_key=? AND owner_token_digest=?
                             AND expires_at=? AND expires_at<=?""",
                        (
                            token_digest,
                            expires_at,
                            now,
                            key.id,
                            row["owner_token_digest"],
                            row["expires_at"],
                            now,
                        ),
                    )
                if cursor.rowcount != 1:
                    raise RuntimeError("lease claim lost its compare-and-set")
                await db.commit()
                return LeaseClaim(
                    state="acquired",
                    owner_token=token,
                    expires_at=expires_at,
                )
            except BaseException:
                await db.rollback()
                raise

    async def renew(
        self,
        key: RunCacheKey,
        *,
        owner_token: str,
        lease_seconds: float,
    ) -> bool:
        """Extend an unexpired lease only when the owner capability matches."""
        token_digest = self._owner_token_digest(owner_token)
        self._lease_deadline(lease_seconds, now=0.0)
        async with aiosqlite.connect(self.path, timeout=30.0) as db:
            await db.execute("PRAGMA busy_timeout=30000")
            await db.execute("BEGIN IMMEDIATE")
            try:
                now = time.time()
                expires_at = self._lease_deadline(lease_seconds, now=now)
                cursor = await db.execute(
                    """UPDATE manifest_leases_v1
                       SET expires_at=?, updated_at=?
                       WHERE cache_key=? AND owner_token_digest=?
                         AND expires_at>?""",
                    (expires_at, now, key.id, token_digest, now),
                )
                await db.commit()
                return cursor.rowcount == 1
            except BaseException:
                await db.rollback()
                raise

    async def release(
        self,
        key: RunCacheKey,
        *,
        owner_token: str,
    ) -> bool:
        """Release a lease only when the owner capability matches."""
        token_digest = self._owner_token_digest(owner_token)
        async with aiosqlite.connect(self.path, timeout=30.0) as db:
            await db.execute("PRAGMA busy_timeout=30000")
            await db.execute("BEGIN IMMEDIATE")
            try:
                cursor = await db.execute(
                    """DELETE FROM manifest_leases_v1
                       WHERE cache_key=? AND owner_token_digest=?""",
                    (key.id, token_digest),
                )
                await db.commit()
                return cursor.rowcount == 1
            except BaseException:
                await db.rollback()
                raise

    async def _mark_stale(
        self,
        key: RunCacheKey,
        reason: str,
        *,
        observed_run_id: str,
        observed_run_fingerprint: str,
    ) -> bool:
        """Mark only the row this reader actually validated.

        Another process may repair the same cache key while validation is in
        flight. The compare-and-set guards prevent an older reader from
        invalidating that newly recorded run.
        """
        async with aiosqlite.connect(self.path, timeout=30.0) as db:
            cursor = await db.execute(
                """UPDATE manifest_v3
                   SET status='stale', stale_reason=?, updated_at=CURRENT_TIMESTAMP
                   WHERE goal_id=? AND goal_fingerprint=? AND variant=?
                     AND system_version=? AND seed=? AND execution_key=?
                     AND status='complete' AND run_id=?
                     AND run_fingerprint=?""",
                (
                    reason,
                    *self._where(key),
                    observed_run_id,
                    observed_run_fingerprint,
                ),
            )
            await db.commit()
            return cursor.rowcount == 1

    async def load_complete(
        self,
        key: RunCacheKey,
        *,
        expected_goal: str,
        expected_tasks: int,
        _retry_after_concurrent_update: bool = True,
    ) -> CacheLookup:
        from bench.runner import read_run

        async with aiosqlite.connect(self.path, timeout=30.0) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM manifest_v3
                   WHERE goal_id=? AND goal_fingerprint=? AND variant=?
                     AND system_version=? AND seed=? AND execution_key=?""",
                self._where(key),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return CacheLookup(state="miss", cache_key=key.id)
        if row["status"] == "stale":
            return CacheLookup(
                state="stale",
                reason=row["stale_reason"] or "previously marked stale",
                cache_key=key.id,
            )

        try:
            if row["execution_json"] != key.execution_json:
                raise ValueError("execution metadata differs for the same key")
            execution = json.loads(key.execution_json)
            if not isinstance(execution, dict):
                raise ValueError("execution metadata is not a JSON object")
            metadata = json.loads(row["metadata_json"])
            if not isinstance(metadata, dict):
                raise ValueError("run metadata is not a JSON object")
            db_path = Path(row["db_path"])
            if not db_path.is_file():
                raise FileNotFoundError(f"run database does not exist: {db_path}")
            run = await read_run(
                str(db_path),
                run_id=row["run_id"],
                goal_id=key.goal_id,
                variant=key.variant,
                n_llm_calls=row["n_llm_calls"],
                wall_clock_s=row["wall_clock_s"],
                agent_task_counts=metadata.get("agent_task_counts"),
                agent_failures=metadata.get("agent_failures"),
                agent_task_trace=metadata.get("agent_task_trace"),
                workers=metadata.get("workers", 0),
                backend=metadata.get("backend", ""),
                models=metadata.get("models"),
                max_backend_concurrency=metadata.get(
                    "max_backend_concurrency", 0
                ),
                estimated_tokens=row["estimated_tokens"],
                expected_goal=expected_goal,
                expected_hypotheses=row["n_hypotheses"],
                ranking_mode=execution.get("ranking_mode", "elo"),
                absolute_scorer=execution.get("absolute_scorer"),
            )
            problem = run_completion_problem(
                run, key, expected_tasks=expected_tasks
            )
            if problem:
                raise ValueError(problem)
            if _run_fingerprint(run) != row["run_fingerprint"]:
                raise ValueError("captured run fingerprint mismatch")
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            OSError,
            sqlite3.Error,
            TypeError,
            ValueError,
        ) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            marked = await self._mark_stale(
                key,
                reason,
                observed_run_id=row["run_id"],
                observed_run_fingerprint=row["run_fingerprint"],
            )
            if not marked and _retry_after_concurrent_update:
                return await self.load_complete(
                    key,
                    expected_goal=expected_goal,
                    expected_tasks=expected_tasks,
                    _retry_after_concurrent_update=False,
                )
            return CacheLookup(
                state="stale",
                reason=reason,
                cache_key=key.id,
            )
        return CacheLookup(state="hit", run=run, cache_key=key.id)

    async def record_complete(
        self,
        key: RunCacheKey,
        run: "BenchRun",
        *,
        expected_goal: str,
        expected_tasks: int,
        owner_token: Optional[str] = None,
    ) -> None:
        from bench.runner import read_run

        problem = run_completion_problem(
            run, key, expected_tasks=expected_tasks
        )
        if problem:
            raise ValueError(f"run is not cacheable: {problem}")
        db_path = Path(run.db_path).resolve()
        if not db_path.is_file():
            raise ValueError(f"run database does not exist: {db_path}")

        # Re-open through the read-only provenance path before making the
        # manifest row authoritative.
        captured = await read_run(
            str(db_path),
            run_id=run.run_id,
            goal_id=run.goal_id,
            variant=run.variant,
            n_llm_calls=run.n_llm_calls,
            wall_clock_s=run.wall_clock_s,
            agent_task_counts=run.agent_task_counts,
            agent_failures=run.agent_failures,
            agent_task_trace=run.agent_task_trace,
            workers=run.workers,
            backend=run.backend,
            models=run.models,
            max_backend_concurrency=run.max_backend_concurrency,
            estimated_tokens=run.estimated_tokens,
            expected_goal=expected_goal,
            expected_hypotheses=len(run.hypotheses),
            ranking_mode=run.ranking_mode,
            absolute_scorer=run.absolute_scorer,
        )
        if _run_fingerprint(captured) != _run_fingerprint(run):
            raise ValueError(
                "run database does not reproduce the returned benchmark run"
            )

        metadata_json = _canonical_json(_metadata_for_run(run))
        run_fingerprint = _run_fingerprint(run)
        owner_token_digest = (
            self._owner_token_digest(owner_token)
            if owner_token is not None
            else None
        )
        async with aiosqlite.connect(self.path, timeout=30.0) as db:
            await db.execute("PRAGMA busy_timeout=30000")
            await db.execute("BEGIN IMMEDIATE")
            try:
                if owner_token_digest is not None:
                    async with db.execute(
                        """SELECT 1 FROM manifest_leases_v1
                           WHERE cache_key=? AND owner_token_digest=?
                             AND expires_at>?""",
                        (key.id, owner_token_digest, time.time()),
                    ) as cursor:
                        owns_lease = await cursor.fetchone()
                    if owns_lease is None:
                        raise ValueError(
                            "cannot publish completed run without the active "
                            "execution lease"
                        )
                else:
                    async with db.execute(
                        "SELECT 1 FROM manifest_leases_v1 WHERE cache_key=?",
                        (key.id,),
                    ) as cursor:
                        existing_lease = await cursor.fetchone()
                    if existing_lease is not None:
                        raise ValueError(
                            "owner_token is required while an execution lease "
                            "exists for this cache key"
                        )
                await db.execute(
                    """INSERT INTO manifest_v3 (
                           goal_id, goal_fingerprint, variant, system_version, seed,
                           execution_key, execution_json, run_id, db_path,
                           n_hypotheses, n_llm_calls, estimated_tokens, wall_clock_s,
                           run_fingerprint, metadata_json, status, stale_reason
                       ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'complete',NULL)
                       ON CONFLICT (
                           goal_id, goal_fingerprint, variant, system_version, seed,
                           execution_key
                       ) DO UPDATE SET
                           execution_json=excluded.execution_json,
                           run_id=excluded.run_id,
                           db_path=excluded.db_path,
                           n_hypotheses=excluded.n_hypotheses,
                           n_llm_calls=excluded.n_llm_calls,
                           estimated_tokens=excluded.estimated_tokens,
                           wall_clock_s=excluded.wall_clock_s,
                           run_fingerprint=excluded.run_fingerprint,
                           metadata_json=excluded.metadata_json,
                           status='complete',
                           stale_reason=NULL,
                           updated_at=CURRENT_TIMESTAMP""",
                    (
                        *self._where(key),
                        key.execution_json,
                        run.run_id,
                        str(db_path),
                        len(run.hypotheses),
                        run.n_llm_calls,
                        run.estimated_tokens,
                        run.wall_clock_s,
                        run_fingerprint,
                        metadata_json,
                    ),
                )
                if owner_token_digest is not None:
                    deleted = await db.execute(
                        """DELETE FROM manifest_leases_v1
                           WHERE cache_key=? AND owner_token_digest=?""",
                        (key.id, owner_token_digest),
                    )
                    if deleted.rowcount != 1:
                        raise RuntimeError(
                            "execution lease changed during cache publication"
                        )
                await db.commit()
            except BaseException:
                await db.rollback()
                raise
