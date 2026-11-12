from __future__ import annotations

import asyncio
import hashlib
import json
import os
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from core.orchestrator import AgentRunner
from core.state import StateStore

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class BenchHypothesis:
    id: str
    text: str
    summary: str
    elo_rating: float
    created_at: str
    generation_method: str = ""
    elo_trajectory: list[tuple[str, float]] = field(default_factory=list)
    absolute_score: Optional[float] = None
    absolute_axis_scores: dict[str, int] = field(default_factory=dict)
    absolute_scorer_id: str = ""

@dataclass
class BenchRun:
    goal_id: str
    variant: str
    hypotheses: list[BenchHypothesis]
    n_llm_calls: int
    wall_clock_s: float
    db_path: str
    run_id: str = ""
    # IMPORTANT 6: `hypotheses` above is already filtered to status='active'
    # (see read_run's WHERE status='active'), so its length silently drops the
    # denominator every time verification quarantines a hypothesis --
    # concordance and rediscovery recall improve via undisclosed sample
    # selection unless this is surfaced next to it on every artifact.
    n_quarantined: int = 0
    agent_task_counts: dict[str, int] = field(default_factory=dict)
    agent_failures: list[dict[str, str]] = field(default_factory=list)
    agent_task_trace: list[dict[str, object]] = field(default_factory=list)
    workers: int = 1
    backend: str = ""
    models: dict[str, str] = field(default_factory=dict)
    max_backend_concurrency: int = 0
    estimated_tokens: int = 0
    ranking_mode: str = "elo"
    absolute_scorer: dict[str, object] = field(default_factory=dict)


@dataclass
class BenchmarkUsageSink:
    """Mutable usage totals that remain available when a benchmark run raises."""

    n_llm_calls: int = 0
    estimated_tokens: int = 0
    active_calls: int = 0
    max_active_calls: int = 0


@dataclass(frozen=True)
class BenchmarkExecution:
    """Exact run-producing settings used to key completed-run reuse."""

    provider: str
    model_fast: str
    model_strong: str
    workers: int
    max_tasks: int
    max_time_seconds: int
    config_sha256: str
    # Renamed from `search_enabled`: that name became a misnomer once PubMed
    # grounding (keyless, always-on for non-offline providers) shipped -- a
    # run with no Tavily key is no longer an ungrounded run. This field now
    # names exactly what it means: whether the paid/keyed web search source
    # is in the mix. `literature_grounded` below is the honest "was this run
    # grounded on real literature at all" signal.
    web_search_enabled: bool
    evidence_max_claims: int
    literature_grounded: bool
    ranking_mode: str = "elo"
    weight_overrides: tuple[tuple[str, float], ...] = ()
    personas: tuple[str, ...] = ()
    absolute_scorer_json: str = ""
    # These fields freeze the runtime inputs used to resolve the public
    # identity above. They are deliberately excluded from ``as_dict`` (and
    # therefore from the manifest): configuration is represented by its hash,
    # and credentials must never be persisted.
    runtime_config_json: str = field(default="", repr=False, compare=False)
    search_api_key: str = field(default="", repr=False, compare=False)

    def as_dict(self) -> dict[str, object]:
        identity = {
            "provider": self.provider,
            "model_fast": self.model_fast,
            "model_strong": self.model_strong,
            "workers": self.workers,
            "max_tasks": self.max_tasks,
            "max_time_seconds": self.max_time_seconds,
            "config_sha256": self.config_sha256,
            "web_search_enabled": self.web_search_enabled,
            "evidence_max_claims": self.evidence_max_claims,
            "literature_grounded": self.literature_grounded,
            "ranking_mode": self.ranking_mode,
            "weight_overrides": [list(item) for item in self.weight_overrides],
            "personas": list(self.personas),
        }
        if self.ranking_mode == "absolute":
            if not self.absolute_scorer_json:
                raise ValueError(
                    "absolute execution is missing its scorer identity"
                )
            scorer = json.loads(self.absolute_scorer_json)
            identity["absolute_scorer"] = _validated_absolute_scorer(scorer)
        return identity


class _CountingBackend:
    """Backend-transparent call/concurrency accounting for benchmarks and stage runs."""

    def __init__(
        self,
        delegate,
        usage_sink: Optional[BenchmarkUsageSink] = None,
    ):
        self.delegate = delegate
        self.usage_sink = usage_sink
        self.call_count = 0
        self.active = 0
        self.max_active = 0
        self.estimated_tokens = 0

    async def call(self, *args, **kwargs) -> str:
        self.call_count += 1
        if self.usage_sink is not None:
            self.usage_sink.n_llm_calls += 1
            self.usage_sink.active_calls += 1
            self.usage_sink.max_active_calls = max(
                self.usage_sink.max_active_calls,
                self.usage_sink.active_calls,
            )
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        system_prompt = args[0] if len(args) > 0 else kwargs.get("system_prompt", "")
        user_prompt = args[1] if len(args) > 1 else kwargs.get("user_prompt", "")
        # Backends expose different tokenizer/usage objects. Character/4 is a
        # deliberately labeled portable estimate used for cross-model budgeting.
        estimated_input_tokens = math.ceil(
            (len(str(system_prompt)) + len(str(user_prompt))) / 4
        )
        self.estimated_tokens += estimated_input_tokens
        if self.usage_sink is not None:
            self.usage_sink.estimated_tokens += estimated_input_tokens
        try:
            response = await self.delegate.call(*args, **kwargs)
            estimated_output_tokens = math.ceil(len(str(response)) / 4)
            self.estimated_tokens += estimated_output_tokens
            if self.usage_sink is not None:
                self.usage_sink.estimated_tokens += estimated_output_tokens
            return response
        finally:
            self.active -= 1
            if self.usage_sink is not None:
                self.usage_sink.active_calls -= 1


class _JudgeBackendAdapter:
    """Expose the provider-neutral judge protocol over any LLM backend."""

    def __init__(self, delegate):
        self.delegate = delegate

    async def score_text(self, system: str, user: str) -> str:
        return await self.delegate.call(
            system, user, use_strong=True, max_tokens=2048
        )


class _CountingAbsoluteJudge:
    """Account for an injected semantic scorer that bypasses an LLM backend."""

    def __init__(self, delegate, usage_sink: BenchmarkUsageSink):
        self.delegate = delegate
        self.usage_sink = usage_sink
        self.weights = getattr(delegate, "weights", None)

    async def score(self, hypothesis: str, *, field: str) -> dict:
        from bench.judge import build_judge_prompt

        system = "You are a rigorous scientific evaluator."
        prompt = build_judge_prompt(hypothesis, field=field)
        self.usage_sink.n_llm_calls += 1
        self.usage_sink.active_calls += 1
        self.usage_sink.max_active_calls = max(
            self.usage_sink.max_active_calls,
            self.usage_sink.active_calls,
        )
        self.usage_sink.estimated_tokens += math.ceil(
            (len(system) + len(prompt)) / 4
        )
        try:
            result = await self.delegate.score(hypothesis, field=field)
            self.usage_sink.estimated_tokens += math.ceil(
                len(json.dumps(result, sort_keys=True, default=str)) / 4
            )
            return result
        finally:
            self.usage_sink.active_calls -= 1


def _validated_absolute_scorer(
    value: object,
) -> dict[str, object]:
    """Recompute the canonical scorer hash instead of trusting metadata."""
    from bench.judge import absolute_scorer_identity

    if not isinstance(value, dict):
        raise ValueError("absolute scorer identity must be a mapping")
    try:
        panel_size = value.get("panel_size", 1)
        if isinstance(panel_size, bool) or not isinstance(panel_size, int):
            raise TypeError("panel_size must be an integer")
        expected = absolute_scorer_identity(
            provider=str(value["provider"]),
            model=str(value["model"]),
            weights=value.get("weights"),
            panel_size=panel_size,
            decoding_tier=str(value.get("decoding_tier", "strong")),
        )
    except Exception as exc:
        raise ValueError("absolute scorer identity is incomplete") from exc
    if value != expected:
        raise ValueError("absolute scorer identity is not canonical")
    if expected["panel_size"] != 1:
        raise ValueError(
            "absolute scorer panel_size must be 1 until judge panels are implemented"
        )
    if expected["decoding_tier"] != "strong":
        raise ValueError("absolute scorer decoding_tier must be 'strong'")
    return expected


def _validate_absolute_scorer_runtime(
    scorer: dict[str, object],
    *,
    system_models: dict[str, str],
    judge: object | None = None,
) -> None:
    """Ensure persisted judge provenance describes the actual one-call path."""
    fast = str(system_models.get("fast", "")).strip()
    strong = str(system_models.get("strong", "")).strip()
    if not fast or not strong:
        raise ValueError(
            "absolute ranking requires identifiable fast and strong system models"
        )
    scorer_model = str(scorer["model"])
    if scorer_model in {fast, strong}:
        raise ValueError(
            "absolute judge model must differ from both system model tiers"
        )
    if judge is None:
        return
    actual_provider = str(getattr(judge, "provider", "")).strip()
    actual_model = str(
        getattr(judge, "judge_model", getattr(judge, "model", ""))
    ).strip()
    if not actual_provider or not actual_model:
        raise ValueError(
            "an injected absolute judge must expose provider and model identity"
        )
    if actual_provider != scorer["provider"] or actual_model != scorer_model:
        raise ValueError(
            "injected absolute judge identity differs from absolute_scorer"
        )


def _unwrap_backend(backend: object) -> object:
    """Return the configured backend beneath local accounting/budget wrappers."""
    current = backend
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        delegate = getattr(current, "__dict__", {}).get("delegate")
        if delegate is None:
            break
        current = delegate
    return current


def _backend_label(backend: object, configured_provider: str = "") -> str:
    """Report a stable provider when known, with a useful fake-backend fallback."""
    if configured_provider:
        return configured_provider
    actual = _unwrap_backend(backend)
    declared = getattr(actual, "provider", "")
    if isinstance(declared, str) and declared:
        return declared
    known = {
        "ClaudeClient": "anthropic",
        "ClaudeCodeBackend": "claude-code",
        "CodexBackend": "codex",
        "OpenAICompatibleBackend": "openai-compatible",
    }
    return known.get(type(actual).__name__, type(actual).__name__)


def benchmark_db_path(
    goal_id: str,
    *,
    label: str,
    seed: int,
    run_id: str | None = None,
) -> str:
    """Return a traversal-safe filename while retaining the original ID in metadata."""
    goal_text = str(goal_id)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", goal_text).strip("-_")[:32] or "goal"
    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "-", str(label)).strip("-_") or "run"
    digest = hashlib.sha256(goal_text.encode("utf-8")).hexdigest()[:12]
    suffix = f"_{run_id}" if run_id else ""
    filename = f"{safe_label}_{slug}-{digest}_{int(seed)}{suffix}.db"
    # bench_runs/ is gitignored, so it is absent from every fresh clone. Create it here
    # rather than only in run_system: callers that build a path without going through
    # run_system (including tests) otherwise hit "unable to open database file".
    out = Path("bench_runs") / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    return str(out)


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


async def read_run(
    db_path: str, run_id: str, goal_id: str, variant: str,
    n_llm_calls: int, wall_clock_s: float,
    *, agent_task_counts: Optional[dict[str, int]] = None,
    agent_failures: Optional[list[dict[str, str]]] = None,
    agent_task_trace: Optional[list[dict[str, object]]] = None,
    workers: int = 1, backend: str = "", models: Optional[dict[str, str]] = None,
    max_backend_concurrency: int = 0,
    estimated_tokens: int = 0,
    expected_goal: Optional[str] = None,
    expected_hypotheses: Optional[int] = None,
    ranking_mode: str = "elo",
    absolute_scorer: Optional[dict[str, object]] = None,
) -> BenchRun:
    """Read a completed run through SQLite read-only mode.

    Cache reads must never create a missing database. The integrity/config/count
    checks also prevent a manifest row from blessing a truncated or mismatched
    run database.
    """
    import aiosqlite

    if ranking_mode not in {"elo", "absolute"}:
        raise ValueError("ranking_mode must be 'elo' or 'absolute'")
    scorer = dict(absolute_scorer or {})
    scorer_id = str(scorer.get("scorer_id", ""))

    resolved = Path(db_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"run database does not exist: {resolved}")
    uri = f"{resolved.as_uri()}?mode=ro"
    async with aiosqlite.connect(uri, uri=True) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("PRAGMA quick_check") as cursor:
            check_rows = await cursor.fetchall()
        if not check_rows or any(str(row[0]).lower() != "ok" for row in check_rows):
            raise ValueError(f"SQLite integrity check failed for {resolved}")
        async with db.execute(
            "SELECT goal FROM configs WHERE run_id=?", (run_id,),
        ) as cursor:
            config_row = await cursor.fetchone()
        if config_row is None:
            raise ValueError(f"run_id {run_id!r} has no persisted config")
        if expected_goal is not None and config_row["goal"] != expected_goal:
            raise ValueError(f"run_id {run_id!r} has a mismatched persisted goal")
        if ranking_mode == "absolute":
            async with db.execute(
                "SELECT scorer_json FROM absolute_scorers WHERE run_id=?",
                (run_id,),
            ) as cursor:
                scorer_row = await cursor.fetchone()
            if scorer_row is None:
                raise ValueError("absolute run has no persisted scorer identity")
            try:
                persisted_scorer = json.loads(scorer_row["scorer_json"])
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(
                    "absolute run has invalid persisted scorer identity"
                ) from exc
            persisted_scorer = _validated_absolute_scorer(persisted_scorer)
            if scorer and scorer != persisted_scorer:
                raise ValueError(
                    "absolute run scorer differs from the expected identity"
                )
            scorer = persisted_scorer
            scorer_id = str(scorer["scorer_id"])
        hypothesis_query = (
            """SELECT h.id, h.text, h.summary, h.elo_rating, h.created_at,
                      h.generation_method, s.total AS absolute_score,
                      s.axis_scores AS absolute_axis_scores,
                      s.scorer_id AS absolute_scorer_id
               FROM hypotheses h
               LEFT JOIN absolute_scores s ON s.hypothesis_id=h.id
               WHERE h.run_id=? AND h.status='active'
               ORDER BY h.id ASC"""
            if ranking_mode == "absolute"
            else
            """SELECT id, text, summary, elo_rating, created_at,
                      generation_method
               FROM hypotheses
               WHERE run_id=? AND status='active'
               ORDER BY id ASC"""
        )
        async with db.execute(hypothesis_query, (run_id,)) as cursor:
            hyp_rows = await cursor.fetchall()
        if (
            expected_hypotheses is not None
            and len(hyp_rows) != expected_hypotheses
        ):
            raise ValueError(
                f"run_id {run_id!r} has {len(hyp_rows)} active hypotheses; "
                f"expected {expected_hypotheses}"
            )
        # IMPORTANT 6: disclose how many hypotheses verification removed from
        # the active pool -- without this, a quarantined hypothesis simply
        # vanishes from every artifact that counts `len(run.hypotheses)`.
        async with db.execute(
            "SELECT COUNT(*) AS n FROM hypotheses WHERE run_id=? AND status='quarantined'",
            (run_id,),
        ) as cursor:
            quarantined_row = await cursor.fetchone()
        n_quarantined = int(quarantined_row["n"]) if quarantined_row else 0
        async with db.execute(
            """SELECT created_at, h1_id, h2_id, elo_after_h1, elo_after_h2
               FROM tournament_matches
               WHERE run_id=? ORDER BY created_at, id""",
            (run_id,),
        ) as cursor:
            rows = await cursor.fetchall()
    if ranking_mode == "absolute" and rows:
        raise ValueError("absolute run unexpectedly contains tournament matches")
    match_rows = [
        (
            str(row["created_at"]),
            row["h1_id"],
            row["h2_id"],
            row["elo_after_h1"],
            row["elo_after_h2"],
        )
        for row in rows
    ]
    bench_hyps = []
    for row in hyp_rows:
        absolute_score = None
        absolute_axes: dict[str, int] = {}
        absolute_row_scorer = ""
        if ranking_mode == "absolute":
            if row["absolute_score"] is None:
                raise ValueError(
                    f"absolute run hypothesis {row['id']!r} has no rubric score"
                )
            absolute_score = float(row["absolute_score"])
            absolute_row_scorer = str(row["absolute_scorer_id"] or "")
            if absolute_row_scorer != scorer_id:
                raise ValueError(
                    f"absolute run hypothesis {row['id']!r} has a mismatched scorer"
                )
            try:
                absolute_axes = json.loads(row["absolute_axis_scores"])
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(
                    f"absolute run hypothesis {row['id']!r} has invalid axis scores"
                ) from exc
            from core.models import AbsoluteScore

            # Reuse the canonical value validation at the read boundary.
            AbsoluteScore(
                run_id=run_id,
                hypothesis_id=row["id"],
                scorer_id=absolute_row_scorer,
                axis_scores=absolute_axes,
                total=absolute_score,
            )
            from bench.judge import weighted_total

            recomputed_total = weighted_total(
                absolute_axes, scorer.get("weights")
            )
            if not math.isclose(
                absolute_score, recomputed_total, abs_tol=1e-9
            ):
                raise ValueError(
                    f"absolute run hypothesis {row['id']!r} has an inconsistent total"
                )
        bench_hyps.append(BenchHypothesis(
            id=row["id"],
            text=row["text"],
            summary=row["summary"],
            elo_rating=row["elo_rating"],
            created_at=str(row["created_at"]),
            generation_method=row["generation_method"],
            elo_trajectory=trajectory_from_matches(row["id"], match_rows),
            absolute_score=absolute_score,
            absolute_axis_scores=absolute_axes,
            absolute_scorer_id=absolute_row_scorer,
        ))
    from core.ranking import native_ranking_key

    bench_hyps.sort(
        key=lambda hypothesis: native_ranking_key(
            hypothesis,
            hypothesis.absolute_score
            if ranking_mode == "absolute"
            else hypothesis.elo_rating,
        )
    )
    return BenchRun(goal_id=goal_id, variant=variant, hypotheses=bench_hyps,
                    n_llm_calls=n_llm_calls, wall_clock_s=wall_clock_s, db_path=db_path,
                    run_id=run_id, n_quarantined=n_quarantined,
                    agent_task_counts=agent_task_counts or {},
                    agent_failures=agent_failures or [], workers=workers,
                    agent_task_trace=agent_task_trace or [],
                    backend=backend, models=models or {},
                    max_backend_concurrency=max_backend_concurrency,
                    estimated_tokens=estimated_tokens,
                    ranking_mode=ranking_mode,
                    absolute_scorer=scorer)


def _load_yaml_config(path: str = "config.yaml") -> dict:
    from tools.resources import load_yaml_config
    return load_yaml_config(path)


def resolve_benchmark_execution(
    *,
    provider: Optional[str],
    model_strong: Optional[str],
    model_fast: Optional[str],
    n_workers: Optional[int],
    max_tasks: int,
    max_time_seconds: int,
    ranking_mode: str = "elo",
    weight_overrides: Optional[dict] = None,
    personas: tuple[str, ...] = (),
    judge_provider: Optional[str] = None,
    judge_model: Optional[str] = None,
    judge_weights: Optional[dict[str, float]] = None,
    config_yaml: str = "config.yaml",
) -> BenchmarkExecution:
    """Resolve config/defaults once so cache identity and execution cannot drift."""
    if ranking_mode not in {"elo", "absolute"}:
        raise ValueError("ranking_mode must be 'elo' or 'absolute'")
    if ranking_mode != "absolute" and any(
        value is not None for value in (judge_provider, judge_model, judge_weights)
    ):
        raise ValueError("judge settings are only valid for ranking_mode='absolute'")
    cfg = _load_yaml_config(config_yaml)
    selected_provider, model_cfg = configure_backend(
        cfg,
        provider=provider,
        model_strong=model_strong,
        model_fast=model_fast,
    )
    workers = (
        n_workers
        if n_workers is not None
        else int(cfg["supervisor"]["n_workers"])
    )
    if workers < 1:
        raise ValueError("n_workers must be >= 1")
    if max_tasks < 0:
        raise ValueError("max_tasks must be >= 0")
    if max_time_seconds <= 0:
        raise ValueError("max_time_seconds must be > 0")
    legacy_model = str(model_cfg.get("model", ""))
    resolved_fast = str(model_cfg.get("model_fast", legacy_model))
    resolved_strong = str(model_cfg.get("model_strong", legacy_model))
    absolute_scorer_json = ""
    if ranking_mode == "absolute":
        from bench.judge import absolute_scorer_identity

        judge_cfg = json.loads(json.dumps(cfg))
        selected_judge_provider, judge_model_cfg = configure_backend(
            judge_cfg,
            provider=judge_provider or selected_provider,
            model_strong=judge_model,
            model_fast=judge_model,
        )
        judge_legacy = str(judge_model_cfg.get("model", ""))
        resolved_judge_model = str(
            judge_model_cfg.get("model_strong", judge_legacy)
        )
        if not resolved_judge_model:
            raise ValueError("absolute judge model could not be resolved")
        if resolved_judge_model in {resolved_fast, resolved_strong}:
            raise ValueError(
                "absolute judge model must differ from both system model tiers; "
                "pass judge_provider/judge_model explicitly"
            )
        scorer = absolute_scorer_identity(
            provider=selected_judge_provider,
            model=resolved_judge_model,
            weights=judge_weights,
        )
        absolute_scorer_json = json.dumps(
            scorer,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    normalized_overrides = tuple(sorted(
        (
            str(getattr(agent_type, "value", agent_type)),
            float(value),
        )
        for agent_type, value in (weight_overrides or {}).items()
    ))
    config_json = json.dumps(
        cfg,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return BenchmarkExecution(
        provider=selected_provider,
        model_fast=resolved_fast,
        model_strong=resolved_strong,
        workers=workers,
        max_tasks=max_tasks,
        max_time_seconds=max_time_seconds,
        config_sha256=hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        web_search_enabled=bool(os.environ.get("TAVILY_API_KEY")),
        # .get(): some callers (e.g. tests stubbing _load_yaml_config) pass a
        # config without an "evidence" section; 4 matches config.yaml's and
        # coscientist.py's own default so a missing section doesn't silently
        # change behavior, only its provenance.
        evidence_max_claims=int(cfg.get("evidence", {}).get("max_claims", 4)),
        # "offline" is the only provider that swaps real PubMed lookups for
        # the deterministic FixtureEvidenceSource (see build_sources); every
        # other provider is literature-grounded regardless of Tavily key.
        literature_grounded=(selected_provider != "offline"),
        ranking_mode=ranking_mode,
        weight_overrides=normalized_overrides,
        personas=tuple(personas),
        absolute_scorer_json=absolute_scorer_json,
        runtime_config_json=config_json,
        search_api_key=os.environ.get("TAVILY_API_KEY", ""),
    )


def configure_backend(
    cfg: dict,
    *,
    provider: str | None = None,
    model_strong: str | None = None,
    model_fast: str | None = None,
) -> tuple[str, dict]:
    """Apply an exact provider/model selection and return its effective config.

    The explicit provider is resolved before model overrides, including for
    ``COSCIENTIST_PROVIDER``. Codex supports one or two model IDs via the same
    strong/fast contract as the other backends.
    """
    from tools.llm import configure_backend as configure

    return configure(
        cfg,
        provider=provider,
        model_strong=model_strong,
        model_fast=model_fast,
    )


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
    backend: Optional[object] = None,
    n_workers: Optional[int] = None,
    provider: Optional[str] = None,
    evidence_provider: Optional[str] = None,
    model_strong: Optional[str] = None,
    model_fast: Optional[str] = None,
    proximity_agent: Optional[object] = None,
    personas: tuple[str, ...] = (),
    absolute_judge: Optional[object] = None,
    absolute_scorer: Optional[dict[str, object]] = None,
    judge_provider: Optional[str] = None,
    judge_model: Optional[str] = None,
    judge_weights: Optional[dict[str, float]] = None,
    config_yaml: str = "config.yaml",
    resolved_execution: Optional[BenchmarkExecution] = None,
    usage_sink: Optional[BenchmarkUsageSink] = None,
) -> BenchRun:
    """Run the full multi-agent system on one goal and capture a BenchRun.

    - `variant`: label stored on the run ("full", "no_evolution", ...).
    - `weight_overrides`: {AgentType: 0.0} forced into compute_weights (ablations).
    - `ranking_mode`: "elo" (normal) | "absolute" (no_tournament — Task D2 reads this).
    - `client_factory(model_strong, model_fast)`: constructs a run-owned client
      (commonly a fake in tests); it is closed when the run ends.
    - `usage_sink`: optional mutable attempted-call/token totals which remain
      available to the caller if the run raises before it can return a BenchRun.
    - `absolute_judge`: injectable provider-neutral Judge-compatible scorer for
      offline tests/custom deployments; `absolute_scorer` must identify it.
    - `evidence_provider`: which literature source set to ground on (see
      `tools.evidence.build_sources`), independent of `provider`/`backend`.
      Resolution order: (1) this, if given; (2) the run's normally-resolved
      `provider`, when one exists (the bench CLI path); (3) otherwise -- an
      injected `backend=` with no stated evidence intent -- the real, keyless
      PubMed source. This must stay independent of the `backend`/`provider`
      mutual-exclusivity check below: that check exists so an ambient env var
      cannot relabel an already-injected backend, not to decide where
      evidence comes from. A caller that wants the deterministic offline
      fixture with an injected backend (e.g. a zero-network smoke test) must
      say so explicitly via `evidence_provider="offline"` -- silently
      defaulting an injected backend to the fixture would let a real,
      possibly paid run be grounded and "verified" against synthetic
      OFFLINE-##### records without anything recording that it happened.
    """
    from core.supervisor import Supervisor, SupervisorSettings
    from core.stats import WeightThresholds
    from core.models import ResearchPlanConfig
    from tools.evidence import build_sources, LiteratureTool
    from tools.verify import ClaimVerifier
    from agents.base import BaseAgent
    from agents.generation import GenerationAgent
    from agents.reflection import ReflectionAgent
    from agents.ranking import RankingAgent
    from agents.proximity import ProximityAgent
    from agents.evolution import EvolutionAgent
    from agents.meta_review import MetaReviewAgent

    if (
        backend is None
        and client_factory is None
        and resolved_execution is None
    ):
        from tools.llm import load_backend_environment
        load_backend_environment()
    if resolved_execution is None:
        cfg = _load_yaml_config(config_yaml)
        search_api_key = os.environ.get("TAVILY_API_KEY", "")
    else:
        if not resolved_execution.runtime_config_json:
            raise ValueError(
                "resolved_execution is missing its frozen runtime configuration"
            )
        try:
            cfg = json.loads(resolved_execution.runtime_config_json)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                "resolved_execution has invalid runtime configuration"
            ) from exc
        if not isinstance(cfg, dict):
            raise ValueError(
                "resolved_execution runtime configuration must be a mapping"
            )
        frozen_hash = hashlib.sha256(
            json.dumps(
                cfg,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if frozen_hash != resolved_execution.config_sha256:
            raise ValueError(
                "resolved_execution runtime configuration hash mismatch"
            )
        if bool(resolved_execution.search_api_key) != (
            resolved_execution.web_search_enabled
        ):
            raise ValueError(
                "resolved_execution search credential state does not match "
                "its execution identity"
            )
        # Symmetric to the web_search_enabled check above: `evidence_provider`
        # is normally left unset and defaults to `resolved_execution.provider`
        # (see the docstring's resolution-order note), which is exactly what
        # `resolved_execution.literature_grounded` was computed from. An
        # explicit `evidence_provider` that disagrees would ground this run on
        # a different literature source than the frozen identity claims --
        # the manifest's `literature_grounded` flag would then misdescribe how
        # the run was actually produced, silently.
        if (
            evidence_provider is not None
            and evidence_provider != resolved_execution.provider
        ):
            raise ValueError(
                "resolved_execution conflicts with explicit argument(s): evidence_provider"
            )

        supplied_overrides = tuple(sorted(
            (
                str(getattr(agent_type, "value", agent_type)),
                float(value),
            )
            for agent_type, value in (weight_overrides or {}).items()
        ))
        frozen_absolute_scorer = (
            _validated_absolute_scorer(json.loads(
                resolved_execution.absolute_scorer_json
            ))
            if resolved_execution.ranking_mode == "absolute"
            else {}
        )
        from bench.judge import normalize_rubric_weights

        conflicts = {
            "provider": (
                provider is not None
                and provider != resolved_execution.provider
            ),
            "model_strong": (
                model_strong is not None
                and model_strong != resolved_execution.model_strong
            ),
            "model_fast": (
                model_fast is not None
                and model_fast != resolved_execution.model_fast
            ),
            "n_workers": (
                n_workers is not None
                and n_workers != resolved_execution.workers
            ),
            "max_tasks": max_tasks != resolved_execution.max_tasks,
            "max_time_seconds": (
                max_time_seconds is not None
                and max_time_seconds != resolved_execution.max_time_seconds
            ),
            "ranking_mode": ranking_mode != resolved_execution.ranking_mode,
            "weight_overrides": (
                supplied_overrides != resolved_execution.weight_overrides
            ),
            "personas": tuple(personas) != resolved_execution.personas,
            "absolute_scorer": (
                absolute_scorer is not None
                and json.dumps(
                    _validated_absolute_scorer(absolute_scorer),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ) != resolved_execution.absolute_scorer_json
            ),
            "judge_provider": (
                judge_provider is not None
                and judge_provider != frozen_absolute_scorer.get("provider")
            ),
            "judge_model": (
                judge_model is not None
                and judge_model != frozen_absolute_scorer.get("model")
            ),
            "judge_weights": (
                judge_weights is not None
                and normalize_rubric_weights(judge_weights)
                != frozen_absolute_scorer.get("weights")
            ),
        }
        conflicting = [name for name, differs in conflicts.items() if differs]
        if conflicting:
            raise ValueError(
                "resolved_execution conflicts with explicit argument(s): "
                + ", ".join(conflicting)
            )

        provider = resolved_execution.provider
        model_strong = resolved_execution.model_strong
        model_fast = resolved_execution.model_fast
        n_workers = resolved_execution.workers
        max_tasks = resolved_execution.max_tasks
        max_time_seconds = resolved_execution.max_time_seconds
        ranking_mode = resolved_execution.ranking_mode
        personas = resolved_execution.personas
        if weight_overrides is None and resolved_execution.weight_overrides:
            from core.models import AgentType

            weight_overrides = {
                AgentType(agent_type): value
                for agent_type, value in resolved_execution.weight_overrides
            }
        search_api_key = resolved_execution.search_api_key
    if ranking_mode not in ("elo", "absolute"):
        raise ValueError(f"Unknown ranking_mode: {ranking_mode!r} (expected 'elo' or 'absolute')")
    if ranking_mode != "absolute" and any((
        absolute_judge,
        absolute_scorer,
        judge_provider,
        judge_model,
        judge_weights,
    )):
        raise ValueError("absolute judge settings require ranking_mode='absolute'")
    resolved_absolute_scorer: dict[str, object] = {}
    if ranking_mode == "absolute":
        if resolved_execution is not None:
            try:
                frozen_scorer = json.loads(
                    resolved_execution.absolute_scorer_json
                )
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(
                    "resolved_execution has invalid absolute scorer metadata"
                ) from exc
            resolved_absolute_scorer = _validated_absolute_scorer(frozen_scorer)
        elif absolute_scorer is not None:
            resolved_absolute_scorer = _validated_absolute_scorer(absolute_scorer)
        elif absolute_judge is not None:
            raise ValueError(
                "an injected absolute_judge requires absolute_scorer identity"
            )
        if resolved_absolute_scorer:
            if (
                judge_provider is not None
                and judge_provider != resolved_absolute_scorer.get("provider")
            ):
                raise ValueError("judge_provider conflicts with absolute_scorer")
            if (
                judge_model is not None
                and judge_model != resolved_absolute_scorer.get("model")
            ):
                raise ValueError("judge_model conflicts with absolute_scorer")
    run_id = uuid.uuid4().hex[:8]
    db_path = db_path or benchmark_db_path(
        goal.id,
        label=variant,
        seed=seed,
        run_id=run_id,
    )
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    store = StateStore(db_path)
    await store.init_db()

    if backend is not None and client_factory is not None:
        raise ValueError("pass either backend or client_factory, not both")
    if backend is not None and any((provider, model_strong, model_fast)):
        raise ValueError(
            "provider/model overrides cannot be combined with an injected backend"
        )
    if backend is not None:
        # An injected object is already an exact backend selection. Do not let a
        # process-level provider environment variable relabel or reconfigure it.
        selected_provider = ""
        model_cfg = {}
        raw_client = backend
    else:
        selected_provider, model_cfg = configure_backend(
            cfg,
            provider=provider,
            model_strong=model_strong,
            model_fast=model_fast,
        )
        if client_factory is not None:
            raw_client = client_factory(
                model_cfg.get("model_strong", ""),
                model_cfg.get("model_fast", ""),
            )
        else:
            # Use the configured backend (default 'claude-code' subscription),
            # with the exact resolved provider pinned for reproducible sweeps.
            from tools.llm import make_backend
            raw_client = make_backend(cfg, provider_override=selected_provider)
    # Explicitly injected backends remain caller-owned. A factory result is
    # created for this run, so this function owns and closes it.
    owns_backend = backend is None
    raw_judge_client = None
    owns_judge_backend = False
    run_usage = usage_sink or BenchmarkUsageSink()
    try:
        client = _CountingBackend(raw_client, usage_sink=run_usage)

        prompts_dir = _PACKAGE_ROOT / "prompts"
        base = BaseAgent(client=client, prompts_dir=prompts_dir)

        configured_backend = _unwrap_backend(raw_client)
        fallback_model = str(getattr(configured_backend, "model", ""))
        models = {
            "strong": str(getattr(
                configured_backend, "model_strong",
                model_cfg.get("model_strong", fallback_model),
            )),
            "fast": str(getattr(
                configured_backend, "model_fast",
                model_cfg.get("model_fast", fallback_model),
            )),
        }
        effective_workers = (
            n_workers
            if n_workers is not None
            else cfg["supervisor"]["n_workers"]
        )

        config = ResearchPlanConfig(
            run_id=run_id, goal=goal.goal,
            preferences="Focus on novel, testable hypotheses",
            attributes=["Novelty", "Feasibility"], constraints="Must be testable",
            safety_approved=True,
        )
        await store.save_config(config)

        # Resolution order (see the `evidence_provider` docstring entry above
        # for the full justification): an explicit `evidence_provider` wins;
        # otherwise the run's normally-resolved `selected_provider` (the bench
        # CLI path); otherwise `selected_provider` is "" (an injected raw
        # `backend=` with no stated evidence intent -- configure_backend
        # never ran for it) and build_sources() falls through to real,
        # keyless PubMed, since only the literal string "offline" selects the
        # synthetic fixture. This must NEVER silently become "offline": that
        # previously made a real company/hierarchy.py run -- which injects a
        # backend and, before this fix, passed no provider at all -- verify
        # its hypotheses against synthetic OFFLINE-##### records while
        # recording nothing about it.
        literature_provider = (
            evidence_provider if evidence_provider is not None else selected_provider
        )
        search = LiteratureTool(
            sources=build_sources(
                literature_provider,
                cfg["tools"]["max_search_results"],
                search_api_key,
            ),
            max_results=cfg["tools"]["max_search_results"],
        )
        if ranking_mode == "absolute":
            from bench.absolute import AbsoluteRankingAgent
            from bench.judge import (
                Judge,
                absolute_scorer_identity,
                normalize_rubric_weights,
            )

            if absolute_judge is None:
                if not resolved_absolute_scorer:
                    if not selected_provider:
                        raise ValueError(
                            "an injected system backend requires an injected "
                            "absolute_judge and canonical absolute_scorer"
                        )
                    judge_cfg_for_identity = json.loads(json.dumps(cfg))
                    selected_judge_provider, selected_judge_models = configure_backend(
                        judge_cfg_for_identity,
                        provider=judge_provider or selected_provider,
                        model_strong=judge_model,
                        model_fast=judge_model,
                    )
                    judge_legacy = str(selected_judge_models.get("model", ""))
                    selected_judge_model = str(
                        selected_judge_models.get("model_strong", judge_legacy)
                    )
                    resolved_absolute_scorer = absolute_scorer_identity(
                        provider=selected_judge_provider,
                        model=selected_judge_model,
                        weights=judge_weights,
                    )
                selected_judge_provider = str(
                    resolved_absolute_scorer["provider"]
                )
                selected_judge_model = str(resolved_absolute_scorer["model"])
                _validate_absolute_scorer_runtime(
                    resolved_absolute_scorer, system_models=models
                )
                judge_cfg_for_run = json.loads(json.dumps(cfg))
                configure_backend(
                    judge_cfg_for_run,
                    provider=selected_judge_provider,
                    model_strong=selected_judge_model,
                    model_fast=selected_judge_model,
                )
                from tools.llm import make_backend

                raw_judge_client = make_backend(
                    judge_cfg_for_run,
                    provider_override=selected_judge_provider,
                )
                owns_judge_backend = True
                counted_judge = _CountingBackend(
                    raw_judge_client, usage_sink=run_usage
                )
                judge = Judge(
                    backend=_JudgeBackendAdapter(counted_judge),
                    judge_model=selected_judge_model,
                    generator_model=models["fast"],
                    weights=resolved_absolute_scorer["weights"],
                )
            else:
                _validate_absolute_scorer_runtime(
                    resolved_absolute_scorer,
                    system_models=models,
                    judge=absolute_judge,
                )
                actual_weights = normalize_rubric_weights(
                    getattr(absolute_judge, "weights", None)
                )
                if actual_weights != resolved_absolute_scorer.get("weights"):
                    raise ValueError(
                        "injected absolute judge weights differ from scorer identity"
                    )
                judge = _CountingAbsoluteJudge(absolute_judge, run_usage)
            await store.save_absolute_scorer(
                run_id, resolved_absolute_scorer
            )
            ranking_agent = AbsoluteRankingAgent(
                judge=judge,
                store=store,
                scorer_id=str(resolved_absolute_scorer["scorer_id"]),
                field=goal.domain,
                max_concurrency=effective_workers,
            )
        else:
            ranking_agent = RankingAgent(
                base=base, store=store,
                elo_k=cfg["tournament"]["elo_k_factor"],
                multi_turn_threshold=cfg["tournament"]["multi_turn_threshold"],
            )
        runner = AgentRunner(
            store=store, config=config,
            generation=GenerationAgent(base=base),
            reflection=ReflectionAgent(base=base),
            ranking=ranking_agent,
            proximity=proximity_agent or ProximityAgent(
                store=store, model_name=cfg["proximity"]["model"],
                similarity_threshold=cfg["proximity"]["similarity_threshold"],
                duplicate_threshold=cfg["proximity"]["duplicate_threshold"],
                ranking_mode=ranking_mode,
            ),
            evolution=EvolutionAgent(base=base),
            meta_review=MetaReviewAgent(base=base),
            search=search,
            ranking_mode=ranking_mode,
            verifier=ClaimVerifier(
                # .get() chain: a config without an "evidence" section (e.g. an
                # older config.yaml, or a test double) must not KeyError here --
                # defaults match the sibling reads at resolve_benchmark_execution
                # above and coscientist.py's own default.
                max_claims=cfg.get("evidence", {}).get("max_claims", 4),
                concurrency=cfg.get("evidence", {}).get("concurrency", 2),
            ),
        )

        settings = SupervisorSettings(
            n_workers=effective_workers,
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
            personas=personas,
        )
        supervisor = _make_supervisor(
            config, runner, settings, store, weight_overrides,
        )

        start = time.monotonic()
        async with asyncio.timeout(settings.max_time_seconds):
            await supervisor.run()
            await runner.finalize_ranking()
        wall = time.monotonic() - start

        return await read_run(
            db_path, run_id=run_id, goal_id=goal.id, variant=variant,
            n_llm_calls=run_usage.n_llm_calls, wall_clock_s=wall,
            agent_task_counts=dict(supervisor.task_counts),
            agent_failures=list(supervisor.task_failures),
            agent_task_trace=list(supervisor.agent_task_trace),
            workers=settings.n_workers,
            backend=_backend_label(raw_client, selected_provider), models=models,
            max_backend_concurrency=run_usage.max_active_calls,
            estimated_tokens=run_usage.estimated_tokens,
            ranking_mode=ranking_mode,
            absolute_scorer=resolved_absolute_scorer,
        )
    finally:
        try:
            if owns_judge_backend and raw_judge_client is not None:
                from tools.llm import close_backend
                await close_backend(raw_judge_client)
        finally:
            if owns_backend:
                from tools.llm import close_backend
                await close_backend(raw_client)


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
