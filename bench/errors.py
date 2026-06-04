from __future__ import annotations


class BenchError(Exception):
    """Raised on harness misconfiguration or invariant violation
    (e.g. judge model equals generator model)."""
