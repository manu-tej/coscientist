"""The QC *scientist* — the LLM orchestration layer on top of the statistical core.

The LLM's job is NOT to estimate uncertainty (that would be the theater we rejected).
It does what statistics can't: read a free-text description of a dataset and route
the *right* checks under a budget, then interpret the resulting flags into a report.
Whether that routing actually beats naive baselines is measured, not assumed
(see value_experiment.py). The agent degrades to a fixed default battery with no LLM.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

# Check catalogue shown to the LLM. Keep names in sync with detectors.py.
CHECK_CATALOGUE = {
    "missingness": "absent/empty values in one or more features",
    "univariate_outlier": "extreme values within a single feature",
    "covariate_shift": "feature distributions differ between a reference set and new/deployment data",
    "batch_effect": "samples are separable by a technical batch / site / instrument",
    "leakage": "a feature encodes or near-perfectly predicts the label (target leakage)",
}
DEFAULT_BATTERY = ["covariate_shift", "univariate_outlier"]  # the static fallback


@dataclass
class Selection:
    checks: list[str]
    reasoning: str
    source: str  # "llm" | "fallback"


def _parse_checks(text: str, valid: list[str], budget: int) -> list[str]:
    """Pull check names out of an LLM reply, tolerant of prose or JSON. Order-preserving,
    de-duplicated, clipped to budget, validated against the catalogue."""
    picked: list[str] = []
    # try JSON array first
    m = re.search(r"\[[^\]]*\]", text, re.DOTALL)
    if m:
        try:
            for item in json.loads(m.group(0)):
                name = str(item).strip().lower()
                if name in valid and name not in picked:
                    picked.append(name)
        except (json.JSONDecodeError, TypeError):
            pass
    if not picked:  # fall back to substring scan in mention order
        for name in valid:
            pos = text.lower().find(name)
            if pos >= 0:
                picked.append((pos, name))
        picked = [n for _, n in sorted(picked)]
    return picked[:budget]


class QCScientist:
    """LLM-orchestrated QC. Pass a backend with an async .call(system, user, ...) — e.g.
    tools.llm.make_backend(cfg). Pass backend=None to force the fixed-battery fallback."""

    def __init__(self, backend=None, budget: int = 2):
        self.backend = backend
        self.budget = budget

    def _fallback(self) -> Selection:
        return Selection(DEFAULT_BATTERY[: self.budget],
                         "no LLM backend — fixed default battery", "fallback")

    async def select_checks(self, description: str, *, available: list[str] | None = None) -> Selection:
        available = available or list(CHECK_CATALOGUE)
        if self.backend is None:
            return self._fallback()
        catalogue = "\n".join(f"- {c}: {CHECK_CATALOGUE[c]}" for c in available)
        system = ("You are a meticulous QC scientist for biomedical datasets. You choose "
                  "which data-quality checks to run given limited budget. You reason from "
                  "the dataset description; you do NOT estimate statistics yourself.")
        user = (f"Available checks:\n{catalogue}\n\n"
                f"Dataset description:\n\"{description}\"\n\n"
                f"You may run only {self.budget} checks. Pick the {self.budget} most likely "
                f"to catch a real quality problem in THIS dataset. Reply with a JSON array of "
                f"exactly {self.budget} check names from the list above, then one sentence of reasoning.")
        try:
            reply = await self.backend.call(system, user, max_tokens=400)
        except Exception as exc:  # rate limit, backend error — degrade, don't crash QC
            return Selection(DEFAULT_BATTERY[: self.budget], f"LLM error ({exc}); fixed battery", "fallback")
        checks = _parse_checks(reply, available, self.budget)
        if not checks:
            return Selection(DEFAULT_BATTERY[: self.budget], "unparseable LLM reply; fixed battery", "fallback")
        return Selection(checks, reply.strip()[:300], "llm")

    async def rank_checks(self, description: str, *, available: list[str] | None = None) -> Selection:
        """Rank ALL available checks by priority for this dataset. Lets a caller read off
        recall@K for any budget K from one call. Missing names are padded in catalogue
        order so the result is always a full permutation."""
        available = available or list(CHECK_CATALOGUE)

        def _pad(order: list[str]) -> list[str]:
            return order + [c for c in available if c not in order]

        if self.backend is None:
            return Selection(_pad(DEFAULT_BATTERY), "no LLM backend — default order", "fallback")
        catalogue = "\n".join(f"- {c}: {CHECK_CATALOGUE[c]}" for c in available)
        system = ("You are a meticulous QC scientist for biomedical datasets. You reason from "
                  "the dataset description to prioritise checks; you do NOT compute statistics.")
        user = (f"Available checks:\n{catalogue}\n\nDataset description:\n\"{description}\"\n\n"
                f"Rank ALL {len(available)} checks from most to least likely to catch a real "
                f"quality problem in THIS dataset. Reply with a JSON array of all check names "
                f"in priority order.")
        try:
            reply = await self.backend.call(system, user, max_tokens=400)
        except Exception as exc:
            return Selection(_pad(DEFAULT_BATTERY), f"LLM error ({exc}); default order", "fallback")
        ranked = _parse_checks(reply, available, budget=len(available))
        if not ranked:
            return Selection(_pad(DEFAULT_BATTERY), "unparseable; default order", "fallback")
        return Selection(_pad(ranked), reply.strip()[:300], "llm")

    async def write_report(self, description: str, flags: list) -> str:
        """Interpret flags into a short report. Falls back to a plain listing without an LLM."""
        if not flags:
            base = "No quality issues detected above threshold."
        else:
            base = "\n".join(f"- [{f.severity}] {f.check} on {f.target}: {f.detail} "
                             f"(confidence {f.confidence:.2f})" for f in flags)
        if self.backend is None:
            return base
        try:
            reply = await self.backend.call(
                "You are a QC scientist writing a crisp data-quality memo for a biomedical team.",
                f"Dataset: \"{description}\"\n\nDetected flags:\n{base}\n\n"
                "Write 3-5 sentences: what's wrong, likely cause, and whether the data is fit to use.",
                max_tokens=500)
            return reply.strip()
        except Exception:
            return base
