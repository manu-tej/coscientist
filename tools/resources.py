"""Resolve repository assets both from source and from an installed wheel."""
from __future__ import annotations

from pathlib import Path

import yaml


# Hatch places the forced-included config and prompts beside the top-level
# packages. This is also the repository root in a source checkout.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def resource_path(path: str | Path) -> Path:
    """Prefer a caller/CWD path, then fall back to the packaged project asset."""
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    packaged = PACKAGE_ROOT / candidate
    return packaged if packaged.exists() else candidate


def load_yaml_config(path: str | Path = "config.yaml") -> dict:
    """Read a YAML config with local override and installed-asset fallback."""
    return yaml.safe_load(resource_path(path).read_text(encoding="utf-8"))
