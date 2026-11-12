"""JSON persistence for a Portfolio.

v1 uses a single JSON snapshot (the org state is small). The detailed per-stage
co-scientist runs already persist to their own SQLite via `core/state.py`; when the
live stage engine is wired in, this store keeps the org-level index and the
SQLite run holds the science (spec §5, §12).
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from enum import Enum
from pathlib import Path

from company.engine import Portfolio
from company.evidence import EvidenceGraph
from company.models import (
    Candidate,
    Company,
    Experiment,
    GateRecord,
    GateStatus,
    Program,
    ProgramStatus,
    Stage,
    StageResult,
    StageRun,
)

DEFAULT_PATH = "company_state.json"


def _encode(obj):
    if isinstance(obj, Enum):
        return obj.value
    if dataclasses.is_dataclass(obj):
        return {f.name: _encode(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    return obj


def _candidate(d: dict) -> Candidate:
    d = dict(d)
    ev = d.get("evidence")
    d["evidence"] = EvidenceGraph.from_dict(ev) if isinstance(ev, dict) else None
    return Candidate(**d)


def _experiment(d: dict) -> Experiment:
    return Experiment(**d)


def _stage_result(d: dict) -> StageResult:
    d = dict(d)
    d["top_candidates"] = [_candidate(c) for c in d.get("top_candidates", [])]
    d["experiments"] = [_experiment(e) for e in d.get("experiments", [])]
    return StageResult(**d)


def _stage_run(d: dict) -> StageRun:
    d = dict(d)
    d["result"] = _stage_result(d["result"])
    d["gate_status"] = GateStatus(d.get("gate_status", GateStatus.PENDING.value))
    return StageRun(**d)


def _program(d: dict) -> Program:
    d = dict(d)
    d["stage"] = Stage(d["stage"])
    d["status"] = ProgramStatus(d["status"])
    # v1 stored only the display string. Parse it through the compatibility
    # property after constructing the new structured model.
    legacy_lead = d.pop("lead_candidate", None)
    program = Program(**d)
    if program.lead_drug is None and legacy_lead is not None:
        program.lead_candidate = legacy_lead
    return program


def _encode_program(program: Program) -> dict:
    """Include the legacy lead string alongside the structured source of truth."""
    payload = _encode(program)
    payload["lead_candidate"] = program.lead_candidate
    return payload


def save(pf: Portfolio, path: str = DEFAULT_PATH) -> None:
    payload = {
        "company": _encode(pf.company),
        "programs": [_encode_program(p) for p in pf.programs],
        "pending": {k: _encode(v) for k, v in pf.pending.items()},
        "gate_records": [_encode(g) for g in pf.gate_records],
        "experiments": [_encode(e) for e in pf.experiments],
        "stage_runs": [_encode(r) for r in pf.stage_runs],
    }
    target = Path(path)
    serialized = json.dumps(payload, indent=2)
    temporary: Path | None = None
    try:
        # A temporary in the destination directory keeps ``os.replace`` atomic:
        # it cannot accidentally cross filesystems.  Sync the complete snapshot
        # before exposing it at the stable path.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        # On serialization/write/sync/replace failure the old target was never
        # touched.  Do not leave a partial snapshot beside it.
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load(path: str = DEFAULT_PATH) -> Portfolio:
    payload = json.loads(Path(path).read_text())
    company = Company(**payload["company"])
    pending = {k: _stage_result(v) for k, v in payload.get("pending", {}).items()}
    stage_runs = [_stage_run(r) for r in payload.get("stage_runs", [])]

    # Old snapshots have pending gate packets but no StageRun archive. Give each
    # packet a stable migration identity so it can be resolved and retained.
    pending_run_programs = {
        r.program_id for r in stage_runs if r.gate_status is GateStatus.PENDING
    }
    for program_id, result in pending.items():
        if program_id not in pending_run_programs:
            stage_runs.append(StageRun(
                run_id=f"legacy-{program_id}-{company.cycle}-{result.stage}",
                program_id=program_id,
                cycle=company.cycle,
                result=result,
            ))

    return Portfolio(
        company=company,
        programs=[_program(p) for p in payload["programs"]],
        pending=pending,
        gate_records=[GateRecord(**g) for g in payload.get("gate_records", [])],
        experiments=[_experiment(e) for e in payload.get("experiments", [])],
        stage_runs=stage_runs,
    )


def exists(path: str = DEFAULT_PATH) -> bool:
    return Path(path).exists()
