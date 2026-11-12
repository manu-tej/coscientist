"""Fair, explicit benchmark grids across model, worker-count, and seed.

The grid is deliberately a thin orchestration layer over the existing benchmark
commands.  It does not reinterpret worker coroutines as scientific roles: each
cell records both the requested worker count and the observed per-role task trace.
"""
from __future__ import annotations

import itertools
import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model_fast: str
    model_strong: str

    @classmethod
    def parse(cls, value: str) -> "ModelSpec":
        """Parse an unambiguous model cell.

        Preferred syntax is ``provider=model_fast[,model_strong]``; model IDs may
        contain colons (for example ``llama3.1:8b``). The original
        ``provider:fast[:strong]`` form remains supported when unambiguous.
        """
        if "=" in value:
            provider, models = (p.strip() for p in value.split("=", 1))
            model_parts = [p.strip() for p in models.split(",", 1)]
            if not provider or not all(model_parts) or len(model_parts) not in (1, 2):
                raise ValueError(
                    f"invalid model spec {value!r}; expected "
                    "provider=model_fast[,model_strong]"
                )
            fast = model_parts[0]
            strong = model_parts[1] if len(model_parts) == 2 else fast
            return cls(provider, fast, strong)

        parts = [p.strip() for p in value.split(":")]
        if len(parts) not in (2, 3) or not all(parts):
            raise ValueError(
                f"invalid model spec {value!r}; expected "
                "provider=model_fast[,model_strong]"
            )
        provider, fast = parts[:2]
        strong = parts[2] if len(parts) == 3 else fast
        return cls(provider, fast, strong)


@dataclass(frozen=True)
class SweepCell:
    provider: str
    model_fast: str
    model_strong: str
    workers: int
    seed: int

    @property
    def id(self) -> str:
        raw = (f"{self.provider}-{self.model_fast}-{self.model_strong}-"
               f"w{self.workers}-s{self.seed}")
        readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-")
        digest = hashlib.sha256(raw.encode()).hexdigest()[:10]
        return f"{readable[:120]}-{digest}"


def parse_int_list(value: str, *, minimum: int = 0) -> list[int]:
    try:
        values = [int(v.strip()) for v in value.split(",") if v.strip()]
    except ValueError as exc:
        raise ValueError(f"expected comma-separated integers, got {value!r}") from exc
    if not values or any(v < minimum for v in values):
        raise ValueError(f"values must be integers >= {minimum}: {value!r}")
    # Stable de-duplication prevents accidentally paying for a duplicate cell.
    return list(dict.fromkeys(values))


def build_cells(worker_counts: str, models: list[str], seeds: str) -> list[SweepCell]:
    workers = parse_int_list(worker_counts, minimum=1)
    seed_values = parse_int_list(seeds, minimum=0)
    model_values = list(dict.fromkeys(ModelSpec.parse(m) for m in models))
    if not model_values:
        raise ValueError("at least one --models provider:model specification is required")
    cells = [
        SweepCell(m.provider, m.model_fast, m.model_strong, w, s)
        for m, w, s in itertools.product(model_values, workers, seed_values)
    ]
    ids = [cell.id for cell in cells]
    if len(ids) != len(set(ids)):
        raise ValueError("model sweep cell IDs collided; use distinct model specifications")
    return cells
