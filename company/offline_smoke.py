"""Deterministic, zero-network structural smoke for the company hierarchy.

This is intentionally *not* a scientific-quality benchmark. It verifies that
two program teams can run concurrently through the real Supervisor/agent
orchestration, produce evidence-cited candidate rankings, persist hierarchy
provenance, debit the ledger, and roll back scientific state on a worker failure
while retaining the operational usage consumed by the failed quarter.

Run from the repository root:

    python -m company.offline_smoke

The default JSON artifact is ``artifacts/hierarchy_offline_smoke.json``.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from company import engine
from company.hierarchy import CoScientistStageExecutor, SeededStageExecutor
from company.models import (
    STAGE_CREDIT_COST,
    STAGE_ORDER,
    GateDecision,
    GateStatus,
    ProgramStatus,
    Stage,
)
from company import store


SMOKE_LABEL = "STRUCTURAL_OFFLINE_HIERARCHY_SMOKE"
SCHEMA_VERSION = 3


class StructuralSmokeError(RuntimeError):
    """A required structural invariant did not hold."""


class _DeterministicBackend:
    """Portable backend contract with no SDK, credential, or network dependency."""

    def __init__(self, delay_seconds: float = 0.005):
        self.delay_seconds = delay_seconds
        self.call_count = 0
        self.active = 0
        self.max_active = 0

    async def call(self, system: str, user: str, **kwargs) -> str:
        self.call_count += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            # A short real yield makes configured worker concurrency observable.
            await asyncio.sleep(self.delay_seconds)
            if "You lead the" in system:
                stage = ""
                for line in user.splitlines():
                    if line.startswith("Stage: "):
                        stage = line.removeprefix("Stage: ").strip()
                        break
                if stage != Stage.HYPOTHESES.value:
                    summaries = {
                        Stage.DISEASE_DOSSIER.value: (
                            "The disease-biology team completed a structured "
                            "dossier and retained its computed uncertainty."
                        ),
                        Stage.MECHANISM.value: (
                            "The mechanism team integrated the archived lead "
                            "with computed pharmacology readouts."
                        ),
                        Stage.TRANSLATIONAL.value: (
                            "The translational team integrated feasibility, "
                            "safety, and validation-plan evidence."
                        ),
                    }
                    return json.dumps({
                        "confidence": 0.82,
                        "red_flags": [],
                        "summary": summaries.get(
                            stage,
                            "The stage team completed its evidence synthesis.",
                        ),
                        "ranked_candidates": [],
                    })
                if "Program: OFFLINE-ALPHA" in user:
                    return (
                        '{"confidence": 0.79, '
                        '"red_flags": ["Verify tadalafil exposure in the target cohort."], '
                        '"summary": "Tadalafil leads the computed candidate panel.", '
                        '"ranked_candidates": [{"drug": "tadalafil", '
                        '"indication": "pulmonary arterial hypertension", '
                        '"rationale": "The quantitative panel supports a PDE5 mechanism.", '
                        '"evidence_refs": ["quant-2"]}, {"drug": "sildenafil", '
                        '"indication": "pulmonary arterial hypertension", '
                        '"rationale": "Independent quantitative support makes it the backup.", '
                        '"evidence_refs": ["quant-1"]}]}'
                    )
                return (
                    '{"confidence": 0.81, '
                    '"red_flags": ["Confirm sildenafil dose-response prospectively."], '
                    '"summary": "Sildenafil leads the computed candidate panel.", '
                    '"ranked_candidates": [{"drug": "sildenafil", '
                    '"indication": "pulmonary arterial hypertension", '
                    '"rationale": "The quantitative panel supports a PDE5 mechanism.", '
                    '"evidence_refs": ["quant-1"]}, {"drug": "tadalafil", '
                    '"indication": "pulmonary arterial hypertension", '
                    '"rationale": "Independent quantitative support makes it the backup.", '
                    '"evidence_refs": ["quant-2"]}]}'
                )
            if "Repair invalid structured output" in system:
                # This branch should remain unused; retaining a valid response makes
                # the smoke robust to harmless parser/prompt wording changes.
                return (
                    '{"confidence": 0.75, "red_flags": [], '
                    '"summary": "Repaired evidence-grounded ranking.", '
                    '"ranked_candidates": [{"drug": "sildenafil", '
                    '"indication": "pulmonary arterial hypertension", '
                    '"rationale": "Supported by the quantitative panel.", '
                    '"evidence_refs": ["quant-1"]}]}'
                )
            return (
                "HYPOTHESIS\n"
                "Hypothesis: Sildenafil and tadalafil can restore cGMP signaling "
                "in pulmonary arterial hypertension.\n"
                "Summary: Compare the two PDE5 inhibitors with exposure-response controls.\n"
                "Category: Repurposing\n"
                "Verdict: PASSED\n"
                "Observation: measurable cGMP rescue\n"
                "better hypothesis: 1"
            )
        finally:
            self.active -= 1


class _NoopProximity:
    """Avoid loading an embedding model during a structural/offline run."""

    async def update_graph(self, hypotheses):
        return []


class _TeamConcurrencyProbe:
    def __init__(self, delegate):
        self.delegate = delegate
        self.active = 0
        self.max_active = 0

    def create_token_budget(self, limit: int):
        """Preserve the live executor's shared admission-control capability."""
        return self.delegate.create_token_budget(limit)

    async def execute(self, *args, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            return await self.delegate.execute(*args, **kwargs)
        finally:
            self.active -= 1


class _FailOneExecutor:
    """Make the historically partial-commit ordering deterministic."""

    def __init__(self):
        self.seeded = SeededStageExecutor()

    async def execute(self, program, cycle, **kwargs):
        if program.id == "atomic-fail":
            await asyncio.sleep(0.02)
            raise RuntimeError("intentional offline atomicity probe")
        await asyncio.sleep(0)
        return await self.seeded.execute(program, cycle, **kwargs)


@contextmanager
def _zero_network_environment():
    """Disable optional clients and fail immediately on any socket attempt."""
    sentinel = object()
    previous: object | str = os.environ.get("TAVILY_API_KEY", sentinel)
    os.environ["TAVILY_API_KEY"] = ""

    def blocked(*args, **kwargs):
        raise StructuralSmokeError(
            "network access attempted during zero-network structural smoke"
        )

    try:
        with (
            patch.object(socket.socket, "connect", blocked),
            patch.object(socket.socket, "connect_ex", blocked),
            patch.object(socket, "create_connection", blocked),
            patch.object(socket, "getaddrinfo", blocked),
        ):
            yield
    finally:
        if previous is sentinel:
            os.environ.pop("TAVILY_API_KEY", None)
        else:
            os.environ["TAVILY_API_KEY"] = str(previous)


def _fixed_portfolio():
    pf = engine.new_company(
        "Offline Structural Company",
        "pulmonary arterial hypertension",
        credit_budget=50.0,
        seed=101,
    )
    pf.company.id = "offline-company"
    alpha = engine.add_program(
        pf,
        "OFFLINE-ALPHA",
        "pulmonary arterial hypertension",
        estimated_value=900.0,
        seed=4,
    )
    beta = engine.add_program(
        pf,
        "OFFLINE-BETA",
        "pulmonary arterial hypertension",
        estimated_value=800.0,
        seed=9,
    )
    alpha.id = "offline-alpha"
    beta.id = "offline-beta"
    alpha.company_id = pf.company.id
    beta.company_id = pf.company.id
    alpha.stage = Stage.HYPOTHESES
    beta.stage = Stage.HYPOTHESES
    return pf


def _fixed_atomicity_portfolio():
    pf = engine.new_company(
        "Offline Atomicity Probe",
        "pulmonary arterial hypertension",
        credit_budget=20.0,
        seed=202,
    )
    pf.company.id = "atomic-company"
    succeeds = engine.add_program(
        pf, "ATOMIC-SUCCEEDS", "pulmonary arterial hypertension",
        estimated_value=900.0, seed=1,
    )
    fails = engine.add_program(
        pf, "ATOMIC-FAILS", "pulmonary arterial hypertension",
        estimated_value=100.0, seed=2,
    )
    succeeds.id = "atomic-succeeds"
    fails.id = "atomic-fail"
    succeeds.company_id = pf.company.id
    fails.company_id = pf.company.id
    return pf


def _fixed_end_to_end_portfolio():
    pf = engine.new_company(
        "Offline End-to-End Company",
        "pulmonary arterial hypertension",
        credit_budget=100.0,
        seed=303,
    )
    pf.company.id = "offline-e2e-company"
    program = engine.add_program(
        pf,
        "OFFLINE-E2E",
        "pulmonary arterial hypertension",
        estimated_value=900.0,
        # This stable seed survives all four real seeded gate rolls for the
        # deterministic stage outputs. No gate RNG is mocked.
        seed=10,
    )
    program.id = "offline-e2e"
    program.company_id = pf.company.id
    return pf


def _fresh_db_paths(work_dir: Path) -> dict[str, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "offline-alpha": work_dir / "offline-alpha-hypotheses-q1.db",
        "offline-beta": work_dir / "offline-beta-hypotheses-q1.db",
    }
    # These exact files are owned by this smoke command. Starting clean prevents
    # prior smoke rows from affecting reproducibility without deleting a directory.
    for path in paths.values():
        path.unlink(missing_ok=True)
    return paths


def _fresh_end_to_end_paths(work_dir: Path) -> tuple[Path, dict[Stage, Path]]:
    run_dir = work_dir / "end_to_end"
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "portfolio.json"
    state_path.unlink(missing_ok=True)
    paths = {
        stage: run_dir / f"offline-e2e-{stage.value}-q{cycle}.db"
        for cycle, stage in enumerate(STAGE_ORDER, 1)
    }
    for path in paths.values():
        path.unlink(missing_ok=True)
    return state_path, paths


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StructuralSmokeError(message)


async def _run_atomicity_probe(work_dir: Path) -> dict[str, Any]:
    pf = _fixed_atomicity_portfolio()
    before = copy.deepcopy(pf)
    state_path = work_dir / "atomicity_portfolio.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.unlink(missing_ok=True)
    store.save(before, str(state_path))
    error = ""
    try:
        await engine.run_quarter_async(
            pf,
            executor=_FailOneExecutor(),
            n_program_workers=2,
        )
    except RuntimeError as exc:
        error = str(exc)
        # Exercise the same delta-only durability primitive used by the CEO CLI.
        # This proves a process restart cannot reset a failed attempt's hard wall.
        from company.cli import _persist_failed_attempt_delta
        _persist_failed_attempt_delta(
            str(state_path),
            pf,
            token_before=before.company.failed_token_spent,
            credit_before=before.company.failed_credit_spent,
        )

    # Failed-attempt usage is operational accounting, not scientific state. Put
    # those two counters back to their pre-attempt values before comparing every
    # other field so this check cannot silently overlook a partial stage commit,
    # cycle increment, pending gate, archive entry, or program mutation.
    scientific_after = copy.deepcopy(pf)
    scientific_after.company.failed_token_spent = (
        before.company.failed_token_spent
    )
    scientific_after.company.failed_credit_spent = (
        before.company.failed_credit_spent
    )
    scientific_state_rolled_back = scientific_after == before
    expected_attempt_credits = STAGE_CREDIT_COST[Stage.DISEASE_DOSSIER]
    failed_attempt_usage_charged = (
        pf.company.failed_credit_spent
        == before.company.failed_credit_spent + expected_attempt_credits
        and pf.company.failed_token_spent > before.company.failed_token_spent
    )
    reloaded = store.load(str(state_path))
    durable_science = copy.deepcopy(reloaded)
    durable_science.company.failed_token_spent = (
        before.company.failed_token_spent
    )
    durable_science.company.failed_credit_spent = (
        before.company.failed_credit_spent
    )
    failed_attempt_usage_survived_reload = (
        durable_science == before
        and reloaded.company.failed_credit_spent
        == pf.company.failed_credit_spent
        and reloaded.company.failed_token_spent
        == pf.company.failed_token_spent
    )
    _require(error == "intentional offline atomicity probe",
             "atomicity probe did not surface the intentional worker failure")
    _require(
        scientific_state_rolled_back,
        "failed concurrent quarter partially committed scientific state",
    )
    _require(
        failed_attempt_usage_charged,
        "failed concurrent quarter did not retain its operational attempt usage",
    )
    _require(
        failed_attempt_usage_survived_reload,
        "failed concurrent quarter usage did not survive snapshot reload",
    )
    return {
        "passed": True,
        "error_type": "RuntimeError",
        "scientific_state_rolled_back": scientific_state_rolled_back,
        "failed_attempt_usage_charged": failed_attempt_usage_charged,
        "failed_attempt_usage_survived_reload": (
            failed_attempt_usage_survived_reload
        ),
        "state_reloaded": True,
        "cycle": pf.company.cycle,
        "committed_credits_spent": pf.company.credit_spent,
        "committed_tokens_spent": pf.company.token_spent,
        "failed_attempt_credits_spent": pf.company.failed_credit_spent,
        "failed_attempt_tokens_spent": pf.company.failed_token_spent,
        "pending_results": len(pf.pending),
        "archived_stage_runs": len(pf.stage_runs),
    }


async def _run_end_to_end_probe(
    *,
    work_dir: Path,
    agent_workers: int,
    max_tasks: int,
) -> dict[str, Any]:
    """Run, gate, persist, and reload one program across all four teams."""
    pf = _fixed_end_to_end_portfolio()
    program = pf.programs[0]
    state_path, expected_db_paths = _fresh_end_to_end_paths(work_dir)
    backend = _DeterministicBackend()
    executor = CoScientistStageExecutor(
        backend,
        n_workers=agent_workers,
        max_tasks=max_tasks,
        max_time_seconds=60,
        db_dir=str(state_path.parent),
        proximity_agent=_NoopProximity(),
        evidence_provider="offline",
    )

    archived: list[dict[str, Any]] = []
    for expected_stage in STAGE_ORDER:
        _require(
            program.stage is expected_stage,
            f"end-to-end program reached {program.stage.value}; "
            f"expected {expected_stage.value}",
        )
        ran = await engine.run_quarter_async(
            pf,
            executor=executor,
            n_program_workers=1,
        )
        _require(
            len(ran) == 1,
            f"{expected_stage.value} did not complete exactly one stage team",
        )
        _, result, _ = ran[0]
        stage_run = pf.latest_stage_run(program, expected_stage)
        expected_db = expected_db_paths[expected_stage]
        _require(
            result.stage == expected_stage.value,
            f"{expected_stage.value} returned a mismatched stage result",
        )
        _require(
            stage_run is not None and stage_run.result is result,
            f"{expected_stage.value} is missing from the durable stage archive",
        )
        _require(
            Path(result.coscientist_db_path) == expected_db
            and expected_db.is_file(),
            f"{expected_stage.value} SQLite provenance is missing",
        )
        _require(
            sum(item["tasks"] for item in result.agent_trace) == max_tasks
            and all(item["status"] == "succeeded" for item in result.agent_trace),
            f"{expected_stage.value} agent trace is incomplete",
        )
        active_personas = {
            item.get("persona")
            for item in result.agent_trace
            if item.get("persona")
        }
        _require(
            active_personas == set(result.team_roster),
            f"{expected_stage.value} did not assign every team persona",
        )

        gate = engine.resolve_gate(pf, program, GateDecision.ADVANCE)
        _require(
            gate.ceo_decision == GateDecision.ADVANCE.value,
            f"{expected_stage.value} is missing an explicit CEO advance decision",
        )
        _require(
            gate.survived_roll is True,
            f"{expected_stage.value} did not survive its real seeded PoS roll",
        )
        expected_status = (
            GateStatus.COMPLETED
            if expected_stage is Stage.TRANSLATIONAL
            else GateStatus.ADVANCED
        )
        _require(
            stage_run.gate_status is expected_status,
            f"{expected_stage.value} gate status was not durably resolved",
        )
        archived.append({
            "stage": expected_stage.value,
            "team_name": result.team_name,
            "team_roster": list(result.team_roster),
            "all_roster_personas_assigned": (
                active_personas == set(result.team_roster)
            ),
            "all_trace_tasks_succeeded": all(
                item["status"] == "succeeded" for item in result.agent_trace
            ),
            "trace_tasks": sum(
                item["tasks"] for item in result.agent_trace
            ),
            "run_id_present": bool(result.coscientist_run_id),
            "database_file": expected_db.name,
            "database_existed_during_validation": expected_db.is_file(),
            "team_summary": result.team_summary,
            "confidence": result.confidence,
            "experiment_count": len(result.experiments),
            "gate_status": stage_run.gate_status.value,
            "ceo_decision": gate.ceo_decision,
            "survived_roll": gate.survived_roll,
            "transition": gate.transition,
        })

    _require(
        program.stage is Stage.COMPLETED
        and program.status is ProgramStatus.COMPLETED,
        "end-to-end program did not clear its final gate",
    )
    _require(not pf.pending, "completed end-to-end program retained a pending gate")
    _require(
        len(pf.stage_runs) == len(STAGE_ORDER)
        and len(pf.gate_records) == len(STAGE_ORDER),
        "end-to-end archive does not contain four linked stage runs and gates",
    )
    expected_credits = sum(STAGE_CREDIT_COST[stage] for stage in STAGE_ORDER)
    _require(
        pf.company.credit_spent == expected_credits
        and program.credits_spent == expected_credits,
        "end-to-end stage credits do not reconcile",
    )
    _require(
        pf.company.token_spent
        == sum(run.result.tokens_spent for run in pf.stage_runs),
        "end-to-end stage tokens do not reconcile",
    )

    store.save(pf, str(state_path))
    reloaded = store.load(str(state_path))
    reloaded_program = reloaded.program("OFFLINE-E2E")
    package = engine.build_clinical_hypothesis_package(
        reloaded,
        reloaded_program,
    )
    pipeline = package.get("pipeline", {})
    pipeline_stage_runs = pipeline.get("stage_runs", [])
    pipeline_gate_decisions = pipeline.get("gate_decisions", [])
    _require(
        reloaded_program.stage is Stage.COMPLETED
        and reloaded_program.status is ProgramStatus.COMPLETED,
        "completed program status did not survive save/reload",
    )
    _require(
        [run.result.stage for run in reloaded.stage_runs] == [
            stage.value for stage in STAGE_ORDER
        ],
        "stage archive order did not survive save/reload",
    )
    _require(
        len(reloaded.gate_records) == len(STAGE_ORDER)
        and all(
            record.ceo_decision == GateDecision.ADVANCE.value
            and record.survived_roll is True
            for record in reloaded.gate_records
        ),
        "linked CEO gate decisions did not survive save/reload",
    )
    _require(
        pipeline.get("stage_order") == [
            stage.value for stage in STAGE_ORDER
        ],
        "final package does not declare the canonical four-stage pipeline",
    )
    _require(
        [stage.get("stage") for stage in pipeline_stage_runs]
        == [stage.value for stage in STAGE_ORDER],
        "final package discarded or reordered archived stage science",
    )
    _require(
        len(pipeline_gate_decisions) == len(STAGE_ORDER)
        and all(
            gate.get("ceo_decision") == GateDecision.ADVANCE.value
            and gate.get("survived_roll") is True
            for gate in pipeline_gate_decisions
        ),
        "final package does not include all linked CEO gate outcomes",
    )
    _require(
        all(stage.get("team", {}).get("name") for stage in pipeline_stage_runs),
        "final package is missing a stage-team identity",
    )
    _require(
        all(stage.get("gate_decisions") for stage in pipeline_stage_runs),
        "final package stage science is not linked to its gate decision",
    )
    later_stage_sections = [
        stage
        for stage in pipeline_stage_runs
        if stage.get("stage") in {
            Stage.MECHANISM.value,
            Stage.TRANSLATIONAL.value,
        }
    ]
    _require(
        len(later_stage_sections) == 2
        and all(stage.get("experiments") for stage in later_stage_sections),
        "final package discarded mechanism or translational experiments",
    )

    return {
        "passed": True,
        "program_id": reloaded_program.id,
        "program_status": reloaded_program.status.value,
        "final_stage": reloaded_program.stage.value,
        "state_reloaded": True,
        "stages_in_order": [
            run.result.stage for run in reloaded.stage_runs
        ],
        "stage_runs": archived,
        "gate_decisions": [
            {
                "transition": record.transition,
                "ceo_decision": record.ceo_decision,
                "survived_roll": record.survived_roll,
                "run_id_present": bool(record.run_id),
            }
            for record in reloaded.gate_records
        ],
        "ledger": {
            "expected_credits_spent": expected_credits,
            "credits_spent": reloaded.company.credit_spent,
            # Adaptive concurrent scheduling can change the mix of successful
            # agent calls while preserving the configured task budget. Report
            # stable accounting invariants, not a timing-sensitive token total.
            "tokens_spent_positive": reloaded.company.token_spent > 0,
            "tokens_reconciled": (
                reloaded.company.token_spent
                == reloaded_program.tokens_spent
            ),
            "reconciled": (
                reloaded.company.credit_spent
                == reloaded_program.credits_spent
                == expected_credits
                and reloaded.company.token_spent
                == reloaded_program.tokens_spent
            ),
        },
        "package": {
            "drug": package["drug"],
            "indication": package["indication"],
            "mechanism": package["mechanism"],
            "confidence": package["confidence"],
            "program": package["program"],
            "stage_order": list(pipeline["stage_order"]),
            "stage_run_count": len(pipeline_stage_runs),
            "gate_decision_count": len(pipeline_gate_decisions),
            "stage_sections": [
                {
                    "stage": stage["stage"],
                    "source": stage["source"],
                    "gate_status": stage["gate_status"],
                    "team_name": stage["team"]["name"],
                    "experiment_count": len(stage["experiments"]),
                    "linked_gate_count": len(stage["gate_decisions"]),
                }
                for stage in pipeline_stage_runs
            ],
            "includes_mechanism_and_translational_evidence": (
                len(later_stage_sections) == 2
                and all(
                    stage.get("experiments")
                    for stage in later_stage_sections
                )
            ),
        },
    }


async def run_structural_smoke(
    *,
    output_path: str | Path = "artifacts/hierarchy_offline_smoke.json",
    work_dir: str | Path = "artifacts/hierarchy_offline_smoke_runs",
    agent_workers: int = 2,
    program_workers: int = 2,
    max_tasks: int = 12,
) -> dict[str, Any]:
    """Run the offline structural smoke and write its machine-readable artifact."""
    if agent_workers < 1:
        raise ValueError("agent_workers must be >= 1")
    if program_workers < 2:
        raise ValueError("program_workers must be >= 2 for the concurrency smoke")
    if max_tasks < 1:
        raise ValueError("max_tasks must be >= 1")

    output_path = Path(output_path)
    work_dir = Path(work_dir)
    expected_db_paths = _fresh_db_paths(work_dir)
    pf = _fixed_portfolio()
    backend = _DeterministicBackend()
    executor = _TeamConcurrencyProbe(CoScientistStageExecutor(
        backend,
        n_workers=agent_workers,
        max_tasks=max_tasks,
        max_time_seconds=60,
        db_dir=str(work_dir),
        proximity_agent=_NoopProximity(),
        evidence_provider="offline",
    ))

    with _zero_network_environment():
        ran = await engine.run_quarter_async(
            pf,
            executor=executor,
            n_program_workers=program_workers,
        )
        end_to_end = await _run_end_to_end_probe(
            work_dir=work_dir,
            agent_workers=agent_workers,
            max_tasks=max_tasks,
        )

    _require(len(ran) == 2, "expected exactly two completed program stage teams")
    _require(executor.max_active == 2, "the two program stage teams did not overlap")
    _require(backend.max_active >= 2, "agent backend calls did not overlap")

    expected_leads = {
        "offline-alpha": "tadalafil",
        "offline-beta": "sildenafil",
    }
    programs: list[dict[str, Any]] = []
    trace_tasks = 0
    for program, result, _ in sorted(ran, key=lambda item: item[0].id):
        stage_run = pf.latest_stage_run(program, Stage.HYPOTHESES)
        expected_db = expected_db_paths[program.id]
        _require(result.team_name == "Repurposing Discovery",
                 f"{program.id} is missing its stage-team identity")
        _require(len(result.team_roster) == 4,
                 f"{program.id} is missing its specialist roster")
        _require(bool(result.coscientist_run_id),
                 f"{program.id} is missing its co-scientist run id")
        _require(Path(result.coscientist_db_path) == expected_db,
                 f"{program.id} recorded an unexpected database path")
        _require(expected_db.exists(), f"{program.id} SQLite provenance is missing")
        _require(stage_run is not None and stage_run.result is result,
                 f"{program.id} result is missing from the durable stage archive")
        _require(sum(item["tasks"] for item in result.agent_trace) == max_tasks,
                 f"{program.id} trace does not cover all configured agent tasks")
        _require(all(item["depth"] == 3 for item in result.agent_trace),
                 f"{program.id} trace has an invalid hierarchy depth")
        _require(all(item["status"] == "succeeded" for item in result.agent_trace),
                 f"{program.id} trace contains a failed task type")
        active_personas = {
            item.get("persona") for item in result.agent_trace
            if item.get("persona")
        }
        _require(active_personas == set(result.team_roster),
                 f"{program.id} did not assign every specialist persona")
        _require(result.top_candidates[0].drug == expected_leads[program.id],
                 f"{program.id} did not apply the authoritative team ranking")
        _require(bool(result.team_candidate_ranking[0]["evidence_refs"]),
                 f"{program.id} ranking is missing evidence references")
        _require(bool(result.quantitative_baseline["candidates"]),
                 f"{program.id} is missing its quantitative baseline")

        task_count = sum(item["tasks"] for item in result.agent_trace)
        trace_tasks += task_count
        programs.append({
            "program_id": program.id,
            "selected_drug": result.top_candidates[0].drug,
            "selected_indication": result.top_candidates[0].indication,
            "confidence": result.confidence,
            "credits_spent": program.credits_spent,
            "red_flags": list(result.red_flag_details),
            "token_accounting": result.token_accounting,
            "hierarchy": {
                "team_name": result.team_name,
                "team_roster": list(result.team_roster),
                "run_id_present": bool(result.coscientist_run_id),
                "database_file": expected_db.name,
                "database_existed_during_validation": expected_db.exists(),
                "trace_tasks": task_count,
                # Adaptive scheduling can redistribute the same fixed task budget
                # among agent types under concurrency. Record stable provenance
                # invariants rather than presenting that timing-sensitive split as
                # a reproducible benchmark result.
                "all_trace_tasks_succeeded": all(
                    item["status"] == "succeeded" for item in result.agent_trace
                ),
                "all_roster_personas_assigned": active_personas
                == set(result.team_roster),
                "assigned_personas": sorted(active_personas),
                "trace_depth": 3,
                "team_summary": result.team_summary,
                "candidate_ranking": list(result.team_candidate_ranking),
            },
        })

    expected_credits = 2 * STAGE_CREDIT_COST[Stage.HYPOTHESES]
    result_tokens = sum(result.tokens_spent for _, result, _ in ran)
    _require(pf.company.credit_spent == expected_credits,
             "company ledger does not equal the two stage debits")
    _require(sum(program.credits_spent for program in pf.programs) == expected_credits,
             "program ledgers do not reconcile with the company ledger")
    _require(pf.company.token_spent == result_tokens,
             "company token ledger does not reconcile with stage results")
    _require(len(pf.pending) == 2 and len(pf.stage_runs) == 2,
             "completed stage teams were not archived as pending gate packets")

    atomicity = await _run_atomicity_probe(work_dir)
    checks = {
        "two_programs_completed": len(ran) == 2,
        "stage_teams_ran_concurrently": executor.max_active == 2,
        "agent_calls_ran_concurrently": backend.max_active >= 2,
        "hierarchy_provenance_complete": all(
            program["hierarchy"]["run_id_present"]
            and program["hierarchy"]["database_existed_during_validation"]
            and program["hierarchy"]["trace_tasks"] == max_tasks
            and program["hierarchy"]["all_roster_personas_assigned"]
            for program in programs
        ),
        "rankings_are_evidence_cited": all(
            program["hierarchy"]["candidate_ranking"][0]["evidence_refs"]
            for program in programs
        ),
        "ledger_reconciled": pf.company.credit_spent == expected_credits,
        "token_ledger_reconciled": pf.company.token_spent == result_tokens,
        "shared_token_admission_enabled": all(
            program["token_accounting"].startswith(
                "hard portable estimated/requested-token envelope"
            )
            for program in programs
        ),
        "failed_quarter_scientific_state_rolled_back": atomicity[
            "scientific_state_rolled_back"
        ],
        "failed_quarter_attempt_usage_charged": atomicity[
            "failed_attempt_usage_charged"
        ],
        "failed_quarter_attempt_usage_survived_reload": atomicity[
            "failed_attempt_usage_survived_reload"
        ],
        "four_stage_program_completed": (
            end_to_end["program_status"] == ProgramStatus.COMPLETED.value
            and end_to_end["final_stage"] == Stage.COMPLETED.value
        ),
        "four_stage_order_preserved": end_to_end["stages_in_order"] == [
            stage.value for stage in STAGE_ORDER
        ],
        "four_ceo_gates_persisted": (
            len(end_to_end["gate_decisions"]) == len(STAGE_ORDER)
            and all(
                gate["ceo_decision"] == GateDecision.ADVANCE.value
                and gate["survived_roll"] is True
                for gate in end_to_end["gate_decisions"]
            )
        ),
        "completed_portfolio_reloaded": end_to_end["state_reloaded"],
        "final_package_covers_all_stages": (
            end_to_end["package"]["stage_order"]
            == [stage.value for stage in STAGE_ORDER]
            and end_to_end["package"]["stage_run_count"] == len(STAGE_ORDER)
            and end_to_end["package"][
                "includes_mechanism_and_translational_evidence"
            ]
        ),
        "four_stage_ledger_reconciled": end_to_end["ledger"]["reconciled"],
        "network_guard_was_enabled": True,
    }
    _require(all(checks.values()), "one or more structural smoke checks failed")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "label": SMOKE_LABEL,
        "status": "passed",
        "scientific_quality_benchmark": False,
        "disclaimer": (
            "Offline structural orchestration smoke only. Deterministic model text "
            "and seeded science fixtures are not evidence of scientific quality."
        ),
        "configuration": {
            "programs": 2,
            "end_to_end_programs": 1,
            "program_workers": program_workers,
            "agent_workers_per_team": agent_workers,
            "max_agent_tasks_per_team": max_tasks,
            "network_enabled": False,
            "socket_network_guard": True,
            "backend": "deterministic-in-process",
        },
        "checks": checks,
        "execution": {
            "max_concurrent_stage_teams": executor.max_active,
            "backend_calls_overlapped": backend.max_active >= 2,
            "agent_trace_tasks": trace_tasks,
        },
        "ledger": {
            "credit_budget": pf.company.credit_budget,
            "expected_credits_spent": expected_credits,
            "credits_spent": pf.company.credit_spent,
            "token_ledger_reconciled": pf.company.token_spent == result_tokens,
        },
        "programs": programs,
        "end_to_end": end_to_end,
        "atomicity_probe": atomicity,
    }
    _write_json(output_path, report)
    return report


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m company.offline_smoke",
        description=(
            "Run the deterministic, zero-network hierarchical structural smoke "
            "(not a scientific-quality benchmark)."
        ),
    )
    parser.add_argument(
        "--output",
        default="artifacts/hierarchy_offline_smoke.json",
        help="machine-readable JSON artifact path",
    )
    parser.add_argument(
        "--work-dir",
        default="artifacts/hierarchy_offline_smoke_runs",
        help="directory for the two co-scientist SQLite runs",
    )
    parser.add_argument("--agent-workers", type=int, default=2)
    parser.add_argument("--program-workers", type=int, default=2)
    parser.add_argument("--max-tasks", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = asyncio.run(run_structural_smoke(
            output_path=args.output,
            work_dir=args.work_dir,
            agent_workers=args.agent_workers,
            program_workers=args.program_workers,
            max_tasks=args.max_tasks,
        ))
    except Exception as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "label": SMOKE_LABEL,
            "status": "failed",
            "scientific_quality_benchmark": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(Path(args.output), report)
        print(json.dumps(report, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
