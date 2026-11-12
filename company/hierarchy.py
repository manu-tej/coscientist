"""Executable hierarchy connecting biotech programs to co-scientist agent teams.

The company layer owns programs and gates.  A :class:`StageExecutor` owns one
stage run.  The live executor below creates a dedicated co-scientist Supervisor
for that stage, whose worker pool dispatches the six specialist agent behaviors,
then performs a schema-validated team synthesis before returning to the company
gate engine.

This makes the runtime hierarchy explicit and inspectable:

    CSO / portfolio -> program -> stage team Supervisor -> agent tasks -> synthesis

The seeded executor remains available for deterministic, zero-LLM offline tests;
it is named explicitly so a simulated run cannot be mistaken for a live team run.
"""
from __future__ import annotations

import asyncio
import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bench.goalset import BenchGoal
from company.ledger import InsufficientTokens
from company.models import Candidate, Program, Stage, StageResult
from company.science import (
    estimate_stage_credits,
    integrate_candidate,
    regulatory_record,
    run_stage,
)


class StageExecutionError(RuntimeError):
    """A stage team did not produce a complete, contract-valid result."""

    def __init__(
        self,
        message: str,
        *,
        attempted_tokens: int = 0,
        attempted_credits: float = 0.0,
    ):
        super().__init__(message)
        self.attempted_tokens = attempted_tokens
        self.attempted_credits = attempted_credits


def _attempt_token_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _attempt_credit_count(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        return 0.0
    return float(value)


def _portable_token_estimate(*parts: object) -> int:
    """Backend-neutral accounting estimate used throughout the company layer."""
    return math.ceil(sum(len(str(part)) for part in parts) / 4)


class PortableTokenBudget:
    """Concurrent hard envelope for portable estimated/requested token units.

    Before a model call starts, the caller reserves the chars/4 prompt estimate
    plus the requested ``max_tokens`` output allowance. Successful responses
    reconcile that reservation to the prompt estimate plus the smaller of the
    response estimate and requested output allowance. Failed calls conservatively
    consume their full reservation because provider billing is unknowable.

    This is deliberately not presented as exact provider billing: the common
    backend contract returns text but no tokenizer-specific usage record, and
    subscription CLI backends may not enforce their ``max_tokens`` argument.
    """

    def __init__(self, limit: int):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("portable token limit must be an integer >= 0")
        self.limit = limit
        self.spent = 0
        self.reserved = 0
        self._condition = asyncio.Condition()

    @property
    def remaining(self) -> int:
        return self.limit - self.spent - self.reserved

    async def reserve(self, amount: int) -> None:
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError("portable token reservation must be an integer >= 0")
        async with self._condition:
            while amount > self.remaining:
                # A sibling request may merely be in flight. Wait for its unused
                # output reservation to reconcile instead of turning temporary
                # concurrency pressure into an agent failure.
                if amount > self.limit - self.spent:
                    raise InsufficientTokens(
                        f"LLM request needs {amount} portable token units; "
                        f"only {self.remaining} remain unreserved"
                    )
                await self._condition.wait()
            self.reserved += amount

    async def reconcile(self, reservation: int, consumed: int) -> int:
        """Release one reservation and charge its successful or failed usage."""
        if not 0 <= consumed <= reservation:
            raise ValueError("consumed token units must fit within the reservation")
        async with self._condition:
            if reservation > self.reserved:
                raise RuntimeError("token reservation was already reconciled")
            self.reserved -= reservation
            self.spent += consumed
            self._condition.notify_all()
        return consumed


class BudgetedLLMBackend:
    """Reserve a shared portable-token envelope before delegating each LLM call."""

    def __init__(self, delegate, budget: PortableTokenBudget):
        self.delegate = delegate
        self.budget = budget
        self.accounted_tokens = 0

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        use_strong: bool = False,
        max_tokens: int = 8192,
    ) -> str:
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 0
        ):
            raise ValueError("max_tokens must be an integer >= 0")
        prompt_tokens = _portable_token_estimate(system_prompt, user_prompt)
        reservation = prompt_tokens + max_tokens
        # This await completes before delegate.call: refusal therefore cannot
        # start a provider request.
        await self.budget.reserve(reservation)
        try:
            response = await self.delegate.call(
                system_prompt,
                user_prompt,
                use_strong=use_strong,
                max_tokens=max_tokens,
            )
        except BaseException:
            # Reconciliation must survive cancellation; otherwise a cancelled
            # worker can strand an in-flight reservation and deadlock siblings.
            charged = await asyncio.shield(
                self.budget.reconcile(reservation, reservation),
            )
            self.accounted_tokens += charged
            raise

        response_tokens = min(_portable_token_estimate(response), max_tokens)
        consumed = prompt_tokens + response_tokens
        charged = await self.budget.reconcile(reservation, consumed)
        self.accounted_tokens += charged
        return response


@dataclass(frozen=True)
class TeamSpec:
    name: str
    stage: Stage
    personas: tuple[str, ...]


TEAM_SPECS: dict[Stage, TeamSpec] = {
    Stage.DISEASE_DOSSIER: TeamSpec(
        "Disease Biology",
        Stage.DISEASE_DOSSIER,
        ("disease biologist", "target geneticist", "pathway analyst", "critic"),
    ),
    Stage.HYPOTHESES: TeamSpec(
        "Repurposing Discovery",
        Stage.HYPOTHESES,
        ("network scientist", "cheminformatician", "literature scientist", "critic"),
    ),
    Stage.MECHANISM: TeamSpec(
        "Mechanism & Pharmacology",
        Stage.MECHANISM,
        ("structural biologist", "pharmacologist", "systems biologist", "critic"),
    ),
    Stage.TRANSLATIONAL: TeamSpec(
        "Translational & Safety",
        Stage.TRANSLATIONAL,
        ("translational scientist", "RWE epidemiologist", "safety scientist", "critic"),
    ),
}


class StageExecutor(Protocol):
    async def execute(
        self,
        program: Program,
        cycle: int,
        *,
        registry=None,
        allow_expensive: bool = True,
        token_budget: PortableTokenBudget | None = None,
    ) -> StageResult:
        ...


class SeededStageExecutor:
    """Explicit zero-LLM executor used for deterministic offline simulation."""

    async def execute(
        self,
        program,
        cycle,
        *,
        registry=None,
        allow_expensive=True,
        token_budget=None,
    ):
        result = run_stage(
            program, cycle, registry=registry, allow_expensive=allow_expensive,
        )
        result.quantitative_baseline = _quantitative_baseline(result)
        result.red_flag_details = list(
            result.quantitative_baseline["red_flags"],
        )
        spec = TEAM_SPECS[program.stage]
        _attach_team_metadata(
            result,
            team=spec,
            run_id="",
            db_path="",
            task_counts={},
            summary="seeded offline simulation; no co-scientist agents invoked",
            team_confidence=None,
            failures=[],
        )
        return result


def _quantitative_baseline(result: StageResult) -> dict:
    """Return the compact, JSON-safe computed evidence supplied to the team."""
    candidates = []
    for index, candidate in enumerate(result.top_candidates, 1):
        evidence = candidate.evidence
        candidates.append({
            "evidence_ref": f"quant-{index}",
            "drug": candidate.drug,
            "indication": candidate.indication,
            "mean_score": round(candidate.mean_score, 4),
            "method_scores": dict(candidate.method_scores),
            "rationale": candidate.rationale,
            "evidence_chain": evidence.explain() if evidence is not None else [],
        })
    quantitative_flags = [
        (
            f"{experiment.assay} returned {experiment.readout:.3f} "
            f"({experiment.interpretation}; {experiment.fidelity_tier})."
        )
        for experiment in result.experiments
        if experiment.readout < 0.4
    ][:result.red_flags]
    if (
        result.stage == Stage.HYPOTHESES.value
        and result.red_flags
        and not result.top_candidates
    ):
        quantitative_flags.append(
            "No quantitative candidate received a non-zero supported method score."
        )
    while len(quantitative_flags) < result.red_flags:
        quantitative_flags.append(
            "The quantitative stage baseline reported an unresolved risk flag."
        )
    return {
        "confidence": result.confidence,
        "method_agreement": result.method_agreement,
        "red_flag_count": result.red_flags,
        "red_flags": quantitative_flags,
        "method_provenance": dict(result.method_provenance),
        "candidates": candidates,
        "experiments": [
            {
                "assay": experiment.assay,
                "method": experiment.method,
                "fidelity_tier": experiment.fidelity_tier,
                "readout": experiment.readout,
                "interpretation": experiment.interpretation,
            }
            for experiment in result.experiments
        ],
    }


def _stage_goal(
    program: Program,
    *,
    team: TeamSpec,
    baseline: dict,
) -> str:
    lead = program.lead_candidate or "no lead selected yet"
    prompts = {
        Stage.DISEASE_DOSSIER: (
            f"Build a decision-grade disease dossier for {program.disease}. "
            "Identify causal biology, tractable targets, disease modules, model-system "
            "limitations, and disconfirming evidence for drug repurposing."
        ),
        Stage.HYPOTHESES: (
            f"Identify and rank approved drugs that could be repurposed for "
            f"{program.disease}. Require independent mechanism, network, structure, "
            "and literature evidence; include negative controls and falsification tests."
        ),
        Stage.MECHANISM: (
            f"Mechanistically validate {lead} for {program.disease}. Evaluate target "
            "engagement, binding, pathway rescue, exposure feasibility, alternative "
            "explanations, and decisive in-silico experiments."
        ),
        Stage.TRANSLATIONAL: (
            f"Build a translational clinical-hypothesis package for {lead} in "
            f"{program.disease}. Assess prior trials, RWE, dose/exposure, known safety, "
            "patient stratification, feasibility, and a falsifiable validation plan."
        ),
    }
    try:
        task = prompts[program.stage]
    except KeyError as exc:
        raise StageExecutionError(f"no team configured for stage {program.stage}") from exc
    personas = ", ".join(team.personas)
    return (
        f"{task}\n\n"
        f"You are the {team.name} stage team. Explicitly integrate these specialist "
        f"perspectives: {personas}. Distinguish computed evidence from agent inference. "
        "Candidate recommendations must cite the supplied evidence references and must "
        "not invent a drug, indication, assay, or score.\n\n"
        "Quantitative baseline (computed before deliberation):\n"
        f"{json.dumps(baseline, sort_keys=True)}"
    )


def _json_object(text: str) -> dict:
    """Extract the first JSON object from model text; prose/fences are tolerated."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text or ""):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("response contained no JSON object")


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _mentioned(value: str, evidence: str) -> bool:
    """Require a complete phrase rather than accepting an arbitrary substring."""
    needle = _normalized(value)
    haystack = _normalized(evidence)
    if not needle:
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


def _validate_synthesis(
    text: str,
    *,
    quantitative_candidates: dict[str, Candidate] | None = None,
    agent_evidence: dict[str, str] | None = None,
    program_disease: str = "",
    allow_candidate_ranking: bool = False,
) -> dict:
    """Validate both shape and evidence grounding of the team decision.

    A ranked candidate is accepted only when one of its cited references either
    identifies the exact quantitative candidate or contains the drug name in a
    supplied agent hypothesis. Agent-nominated indications must be the program
    disease or occur in that same cited evidence.
    """
    value = _json_object(text)
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be in [0,1]")
    summary = value.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("summary must be a non-empty string")
    flags = value.get("red_flags", [])
    if not isinstance(flags, list) or not all(isinstance(x, str) for x in flags):
        raise ValueError("red_flags must be a list of strings")
    ranked = value.get("ranked_candidates")
    if not isinstance(ranked, list):
        raise ValueError("ranked_candidates must be a list")
    if ranked and not allow_candidate_ranking:
        raise ValueError("ranked_candidates must be empty outside the hypotheses stage")

    quantitative_candidates = quantitative_candidates or {}
    agent_evidence = agent_evidence or {}
    valid_refs = set(quantitative_candidates) | set(agent_evidence)
    normalized_disease = _normalized(program_disease)
    seen: set[tuple[str, str]] = set()
    validated_ranking: list[dict] = []
    for entry in ranked:
        if not isinstance(entry, dict):
            raise ValueError("each ranked candidate must be an object")
        drug = entry.get("drug")
        indication = entry.get("indication")
        rationale = entry.get("rationale")
        refs = entry.get("evidence_refs")
        if not all(isinstance(x, str) and x.strip()
                   for x in (drug, indication, rationale)):
            raise ValueError(
                "ranked candidate drug, indication, and rationale must be non-empty strings"
            )
        if not isinstance(refs, list) or not refs or not all(
            isinstance(ref, str) and ref in valid_refs for ref in refs
        ):
            raise ValueError(
                "ranked candidate evidence_refs must cite supplied evidence references"
            )

        pair = (_normalized(drug), _normalized(indication))
        if pair in seen:
            raise ValueError(f"duplicate ranked candidate: {drug} / {indication}")
        seen.add(pair)

        grounded = False
        for ref in refs:
            quantitative = quantitative_candidates.get(ref)
            if quantitative is not None and pair == (
                _normalized(quantitative.drug),
                _normalized(quantitative.indication),
            ):
                grounded = True
                break
            evidence = agent_evidence.get(ref)
            if evidence is not None and _mentioned(drug, evidence):
                indication_grounded = (
                    _normalized(indication) == normalized_disease
                    or _mentioned(indication, evidence)
                )
                if indication_grounded:
                    grounded = True
                    break
        if not grounded:
            raise ValueError(
                f"candidate {drug!r} / {indication!r} is not grounded in its cited evidence"
            )
        regulatory = regulatory_record(drug)
        validated_ranking.append({
            "drug": drug.strip(),
            "indication": indication.strip(),
            "rationale": rationale.strip(),
            "evidence_refs": list(dict.fromkeys(refs)),
            "regulatory_status": regulatory["status"],
            "regulatory_source": regulatory["source"],
            "regulatory_live_verified": regulatory["live_verified"],
        })

    if allow_candidate_ranking and not validated_ranking:
        raise ValueError("hypotheses-stage synthesis must select at least one candidate")

    clean_flags = list(dict.fromkeys(x.strip() for x in flags if x.strip()))
    return {
        "confidence": float(confidence),
        "summary": summary.strip(),
        "red_flags": clean_flags,
        "ranked_candidates": validated_ranking,
    }


async def _synthesize(
    backend,
    program: Program,
    result: StageResult,
    hypotheses,
    *,
    team: TeamSpec,
) -> dict:
    top = sorted(hypotheses, key=lambda h: h.elo_rating, reverse=True)[:5]
    agent_evidence = {
        h.id: f"{h.text}\nSummary: {h.summary}"[:6000]
        for h in top
    }
    quantitative_candidates = {
        item["evidence_ref"]: candidate
        for item, candidate in zip(
            result.quantitative_baseline.get("candidates", []),
            result.top_candidates,
        )
    }
    agent_block = "\n\n".join(
        f"Evidence ref {h.id} (Elo {h.elo_rating:.1f}):\n"
        f"{agent_evidence[h.id]}"
        for h in top
    )
    allow_ranking = program.stage is Stage.HYPOTHESES
    example_ref = (
        next(iter(quantitative_candidates), None)
        or next(iter(agent_evidence), "SUPPLIED_EVIDENCE_REF")
    )
    ranked_example = (
        '[{"drug": "exact evidence-grounded name", '
        '"indication": "exact indication", '
        '"rationale": "why this evidence supports its rank", '
        f'"evidence_refs": ["{example_ref}"]}}]'
        if allow_ranking else "[]"
    )
    schema = (
        '{"confidence": 0.0, "red_flags": ["specific concern"], '
        '"summary": "decision-grade synthesis grounded in the supplied evidence", '
        f'"ranked_candidates": {ranked_example}}}'
    )
    personas = ", ".join(team.personas)
    system = (
        f"You lead the {team.name} team. Reconcile these specialist perspectives: "
        f"{personas}. Use only the supplied quantitative and agent evidence. "
        "Every ranked candidate must cite evidence_refs. Return one JSON object "
        "and no unsupported facts."
    )
    user = (
        f"Program: {program.name}\nDisease: {program.disease}\n"
        f"Stage: {program.stage.value}\nTeam: {team.name}\n\n"
        "Quantitative evidence (including quant-* evidence refs):\n"
        f"{json.dumps(result.quantitative_baseline, sort_keys=True)}\n\n"
        f"Agent evidence:\n{agent_block or '(none)'}\n\n"
        f"Return exactly this schema: {schema}"
    )
    first = await backend.call(system, user, use_strong=True, max_tokens=1200)
    try:
        return _validate_synthesis(
            first,
            quantitative_candidates=quantitative_candidates,
            agent_evidence=agent_evidence,
            program_disease=program.disease,
            allow_candidate_ranking=allow_ranking,
        )
    except ValueError as exc:
        repair = await backend.call(
            "Repair invalid structured output. Do not add new scientific claims.",
            f"Validation error: {exc}\nInvalid output:\n{first[:6000]}\n"
            f"Return exactly one JSON object matching: {schema}",
            use_strong=False,
            max_tokens=800,
        )
        try:
            return _validate_synthesis(
                repair,
                quantitative_candidates=quantitative_candidates,
                agent_evidence=agent_evidence,
                program_disease=program.disease,
                allow_candidate_ranking=allow_ranking,
            )
        except ValueError as repair_exc:
            raise StageExecutionError(
                f"team synthesis violated its output contract twice: {repair_exc}"
            ) from repair_exc


def _materialize_candidate_ranking(
    result: StageResult,
    ranking: list[dict],
) -> list[Candidate]:
    """Turn validated synthesis entries into the authoritative candidate list."""
    quantitative = {
        (_normalized(candidate.drug), _normalized(candidate.indication)): candidate
        for candidate in result.top_candidates
    }
    selected: list[Candidate] = []
    for entry in ranking:
        pair = (_normalized(entry["drug"]), _normalized(entry["indication"]))
        candidate = quantitative.get(pair)
        if candidate is None:
            candidate = Candidate(
                drug=entry["drug"],
                indication=entry["indication"],
                rationale=entry["rationale"],
                is_fixture=False,
            )
        selected.append(candidate)
    return selected


def _attach_team_metadata(
    result: StageResult,
    *,
    team: TeamSpec,
    run_id: str,
    db_path: str,
    task_counts: dict[str, int],
    summary: str,
    team_confidence: float | None,
    failures: list[dict[str, str]],
    task_trace: list[dict[str, object]] | None = None,
    provider: str = "",
    models: dict[str, str] | None = None,
    workers: int = 0,
    max_tasks: int = 0,
    seed: int = 0,
    max_backend_concurrency: int = 0,
) -> None:
    """Set fields introduced on StageResult while remaining tolerant of old snapshots."""
    result.team_name = team.name
    result.team_roster = list(team.personas)
    result.coscientist_run_id = run_id
    result.coscientist_db_path = db_path
    if task_trace:
        result.agent_trace = [
            {
                "parent": f"stage:{run_id or 'offline'}",
                "depth": 3,
                "team": team.name,
                "priority": int(trace.get("priority", index)),
                "agent_type": str(trace.get("agent_type", "")),
                "persona": str(trace.get("persona", "")),
                "strategy": str(trace.get("strategy", "")),
                "tasks": 1,
                "status": str(trace.get("status", "unknown")),
                "error": trace.get("error"),
            }
            for index, trace in enumerate(task_trace)
        ]
    else:
        result.agent_trace = [
            {
                "parent": f"stage:{run_id or 'offline'}",
                "depth": 3,
                "team": team.name,
                "agent_type": agent_type,
                "persona": "",
                "strategy": "",
                "tasks": count,
                "status": "failed" if any(
                    f.get("agent_type") == agent_type for f in failures
                ) else "succeeded",
                "error": None,
            }
            for agent_type, count in sorted(task_counts.items())
        ]
    result.team_summary = summary
    result.team_confidence = team_confidence
    result.provider = provider
    result.models = dict(models or {})
    result.workers = workers
    result.max_tasks = max_tasks
    result.seed = seed
    result.max_backend_concurrency = max_backend_concurrency


class CoScientistStageExecutor:
    """Run one real co-scientist team per biotech stage and aggregate its output."""

    def __init__(
        self,
        backend,
        *,
        n_workers: int = 4,
        max_tasks: int = 24,
        max_time_seconds: int = 900,
        db_dir: str = "company_runs",
        strict_task_failures: bool = True,
        proximity_agent=None,
        evidence_provider: str | None = None,
    ):
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        min_roster_size = max(len(spec.personas) for spec in TEAM_SPECS.values())
        if max_tasks < min_roster_size:
            raise ValueError(
                f"max_tasks must be >= {min_roster_size} so every stage-team "
                "persona receives at least one task"
            )
        if max_time_seconds <= 0:
            raise ValueError("max_time_seconds must be > 0")
        self.backend = backend
        self.n_workers = n_workers
        self.max_tasks = max_tasks
        self.max_time_seconds = max_time_seconds
        self.db_dir = Path(db_dir)
        self.strict_task_failures = strict_task_failures
        self.proximity_agent = proximity_agent
        # `backend` is an already-constructed client with no attached provider
        # identity, so run_system() cannot infer where to ground literature
        # from it. Callers that resolved a real provider (company/cli.py) must
        # pass it here explicitly; offline/deterministic callers must pass
        # "offline" explicitly. Leaving this None is only safe when run_system
        # itself is mocked out (as in several tests) -- a live run with this
        # left None would fall back to real, keyless PubMed rather than ever
        # silently substituting synthetic fixture data.
        self.evidence_provider = evidence_provider

    @staticmethod
    def create_token_budget(limit: int) -> PortableTokenBudget:
        """Create the shared quarter envelope before any stage team is dispatched."""
        return PortableTokenBudget(limit)

    async def execute(
        self,
        program,
        cycle,
        *,
        registry=None,
        allow_expensive=True,
        token_budget: PortableTokenBudget | None = None,
    ) -> StageResult:
        """Run the complete baseline→team→synthesis path under one wall-clock cap."""
        deadline = asyncio.timeout(self.max_time_seconds)
        try:
            async with deadline:
                return await self._execute(
                    program,
                    cycle,
                    registry=registry,
                    allow_expensive=allow_expensive,
                    token_budget=token_budget,
                )
        except TimeoutError as exc:
            # A provider may itself raise TimeoutError before our deadline. Do
            # not relabel that distinct failure as the whole-stage wall clock.
            if not deadline.expired():
                raise
            cause = exc.__cause__ or exc.__context__
            metered_tokens = _attempt_token_count(
                getattr(token_budget, "spent", 0)
            )
            cause_tokens = _attempt_token_count(
                getattr(cause, "attempted_tokens", 0)
            )
            cause_credits = _attempt_credit_count(
                getattr(cause, "attempted_credits", 0.0)
            )
            # A sync adapter running in asyncio.to_thread cannot be killed by
            # task cancellation. Charge the conservative stage reservation so
            # such a timed-out call cannot escape the lifetime credit wall.
            reserved_credits = estimate_stage_credits(
                program,
                registry=registry,
                allow_expensive=allow_expensive,
            )
            raise StageExecutionError(
                f"{program.name} {program.stage.value} stage exceeded the "
                f"{self.max_time_seconds}s whole-stage timeout",
                attempted_tokens=max(metered_tokens, cause_tokens),
                attempted_credits=max(cause_credits, reserved_credits),
            ) from exc

    async def _execute(
        self,
        program,
        cycle,
        *,
        registry=None,
        allow_expensive=True,
        token_budget: PortableTokenBudget | None = None,
    ) -> StageResult:
        from bench.runner import _CountingBackend, run_system

        spec = TEAM_SPECS[program.stage]
        budgeted_backend = (
            BudgetedLLMBackend(self.backend, token_budget)
            if token_budget is not None else None
        )
        stage_backend = _CountingBackend(budgeted_backend or self.backend)
        lead_snapshot = (
            program.lead_candidate,
            program.lead_drug,
            program.lead_indication,
        )
        baseline_started = False
        try:
            # Science adapters run against a copy: a failed agent run or invalid
            # synthesis must not leak a newly selected lead into the live program.
            baseline_program = copy.deepcopy(program)
            baseline_started = True
            result = await asyncio.to_thread(
                run_stage,
                baseline_program,
                cycle,
                registry=registry,
                allow_expensive=allow_expensive,
            )
            result.quantitative_baseline = _quantitative_baseline(result)

            goal_id = f"{program.id}-{program.stage.value}-q{cycle}"
            db_path = self.db_dir / f"{goal_id}.db"
            goal = BenchGoal(
                id=goal_id,
                goal=_stage_goal(program, team=spec, baseline=result.quantitative_baseline),
                domain="drug repurposing",
                metadata={
                    "program_id": program.id,
                    "stage": program.stage.value,
                    "team": spec.name,
                    "personas": list(spec.personas),
                    "quantitative_baseline": result.quantitative_baseline,
                },
            )
            run_seed = program.seed + cycle
            run = await run_system(
                goal,
                variant="company-stage",
                seed=run_seed,
                db_path=str(db_path),
                max_tasks=self.max_tasks,
                max_time_seconds=self.max_time_seconds,
                backend=stage_backend,
                evidence_provider=self.evidence_provider,
                n_workers=self.n_workers,
                proximity_agent=self.proximity_agent,
                personas=spec.personas,
            )
            if not run.hypotheses:
                # CRITICAL 2 (same class): "produced no hypotheses" is false
                # when hypotheses were produced and then quarantined by
                # verification (a contradicted claim or an untraced citation)
                # -- that is a real, informative outcome, not the empty run
                # this message otherwise correctly describes.
                n_quarantined = getattr(run, "n_quarantined", 0)
                if n_quarantined:
                    raise StageExecutionError(
                        f"{spec.name} generated {n_quarantined} hypothesis(es) for "
                        f"{program.name}, but all were quarantined by verification "
                        "(a claim was contradicted or a citation could not be traced "
                        "to a retrieved record) -- none are usable"
                    )
                raise StageExecutionError(
                    f"{spec.name} produced no hypotheses for {program.name}"
                )
            if run.agent_failures and self.strict_task_failures:
                raise StageExecutionError(
                    f"{spec.name} had {len(run.agent_failures)} failed agent tasks"
                )

            synthesis = await _synthesize(
                stage_backend,
                program,
                result,
                run.hypotheses,
                team=spec,
            )
            selected = _materialize_candidate_ranking(
                result, synthesis["ranked_candidates"],
            )
            if program.stage is Stage.HYPOTHESES:
                result.top_candidates = selected
                # The quantitative component must follow the team's selected lead,
                # rather than retaining the score of the pre-synthesis top choice.
                quantitative_confidence, agreement = integrate_candidate(selected[0])
                result.confidence = quantitative_confidence
                result.method_agreement = agreement

            result.confidence = round(
                0.7 * result.confidence + 0.3 * synthesis["confidence"], 3
            )
            result.red_flag_details = list(
                result.quantitative_baseline["red_flags"],
            )
            normalized_existing_flags = {
                _normalized(flag) for flag in result.red_flag_details
            }
            regulatory_flags = [
                (
                    f"Regulatory approval and safety/PK status for "
                    f"{entry['drug']} are not established by this run."
                )
                for entry in synthesis["ranked_candidates"]
                if entry["regulatory_status"] == "unknown"
            ]
            for flag in [*synthesis["red_flags"], *regulatory_flags]:
                if _normalized(flag) not in normalized_existing_flags:
                    result.red_flag_details.append(flag)
                    normalized_existing_flags.add(_normalized(flag))
                    result.red_flags += 1
            result.team_candidate_ranking = list(synthesis["ranked_candidates"])
            result.llm_calls = stage_backend.call_count
            result.tokens_spent = (
                budgeted_backend.accounted_tokens
                if budgeted_backend is not None
                else stage_backend.estimated_tokens
            )
            result.token_accounting = (
                "hard portable estimated/requested-token envelope; "
                "not exact provider billing"
                if budgeted_backend is not None
                else "portable chars/4 estimate; not exact provider billing"
            )
            _attach_team_metadata(
                result,
                team=spec,
                run_id=run.run_id,
                db_path=run.db_path,
                task_counts=run.agent_task_counts,
                summary=synthesis["summary"],
                team_confidence=synthesis["confidence"],
                failures=run.agent_failures,
                task_trace=run.agent_task_trace,
                provider=run.backend,
                models=run.models,
                workers=run.workers,
                max_tasks=self.max_tasks,
                seed=run_seed,
                max_backend_concurrency=run.max_backend_concurrency,
            )

            # Commit the lead only after every model output and evidence reference
            # has passed validation.
            if program.stage is Stage.HYPOTHESES:
                lead = result.top_candidates[0]
                program.lead_drug = lead.drug
                program.lead_indication = lead.indication
            return result
        except BaseException as exc:
            attempted_tokens = (
                budgeted_backend.accounted_tokens
                if budgeted_backend is not None
                else stage_backend.estimated_tokens
            )
            if "result" in locals():
                attempted_credits = float(result.credits_spent)
            elif baseline_started:
                # Cancellation cannot stop work already running in to_thread.
                # Conservatively retain the pre-dispatch reservation.
                attempted_credits = estimate_stage_credits(
                    program,
                    registry=registry,
                    allow_expensive=allow_expensive,
                )
            else:
                attempted_credits = 0.0
            if not hasattr(exc, "attempted_tokens"):
                setattr(exc, "attempted_tokens", attempted_tokens)
            if not hasattr(exc, "attempted_credits"):
                setattr(exc, "attempted_credits", attempted_credits)
            # Assign directly to avoid the compatibility setters constructing an
            # intermediate lead while the prior structured state is restored.
            object.__setattr__(program, "lead_candidate", lead_snapshot[0])
            object.__setattr__(program, "lead_drug", lead_snapshot[1])
            object.__setattr__(program, "lead_indication", lead_snapshot[2])
            raise
