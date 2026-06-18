"""JSON persistence for a Portfolio.

v1 uses a single JSON snapshot (the org state is small). The detailed per-stage
co-scientist runs already persist to their own SQLite via `core/state.py`; when the
live stage engine is wired in, this store keeps the org-level index and the
SQLite run holds the science (spec §5, §12).
"""
from __future__ import annotations

import dataclasses
import json
from enum import Enum
from pathlib import Path

from company.engine import Portfolio
from company.models import (
    Candidate,
    Company,
    Experiment,
    GateRecord,
    Program,
    ProgramStatus,
    Stage,
    StageResult,
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
    return Candidate(**d)


def _experiment(d: dict) -> Experiment:
    return Experiment(**d)


def _stage_result(d: dict) -> StageResult:
    d = dict(d)
    d["top_candidates"] = [_candidate(c) for c in d.get("top_candidates", [])]
    d["experiments"] = [_experiment(e) for e in d.get("experiments", [])]
    return StageResult(**d)


def _program(d: dict) -> Program:
    d = dict(d)
    d["stage"] = Stage(d["stage"])
    d["status"] = ProgramStatus(d["status"])
    return Program(**d)


def save(pf: Portfolio, path: str = DEFAULT_PATH) -> None:
    payload = {
        "company": _encode(pf.company),
        "programs": [_encode(p) for p in pf.programs],
        "pending": {k: _encode(v) for k, v in pf.pending.items()},
        "gate_records": [_encode(g) for g in pf.gate_records],
        "experiments": [_encode(e) for e in pf.experiments],
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load(path: str = DEFAULT_PATH) -> Portfolio:
    payload = json.loads(Path(path).read_text())
    return Portfolio(
        company=Company(**payload["company"]),
        programs=[_program(p) for p in payload["programs"]],
        pending={k: _stage_result(v) for k, v in payload.get("pending", {}).items()},
        gate_records=[GateRecord(**g) for g in payload.get("gate_records", [])],
        experiments=[_experiment(e) for e in payload.get("experiments", [])],
    )


def exists(path: str = DEFAULT_PATH) -> bool:
    return Path(path).exists()
