"""Deterministic in-process backend — the free, network-free way to run the system.

This is a real provider (``--provider offline``), not a test double. It exists so the
pipeline has an execution path that costs nothing and needs no credentials: a fresh
clone can run ``coscientist.py`` end to end, and CI can exercise the orchestrator,
tournament, and persistence layers without a model.

What it is NOT: evidence of scientific quality. Its text is templated and seeded, so
hypotheses, review verdicts, and tournament winners are synthetic. Rankings produced
under this provider measure plumbing, not reasoning. Every artifact it produces is
tagged with the ``offline-deterministic-v1`` model name so a run can never be mistaken
for a real one.

Determinism: output depends only on (system_prompt, user_prompt, seed), so the same
run configuration replays byte-identically.
"""

from __future__ import annotations

import hashlib
import re

_MODEL_NAME = "offline-deterministic-v1"

# Deliberately mundane, clearly-synthetic mechanisms. They must not read as real
# scientific claims, because they are not.
_MECHANISMS = (
    ("upregulated transport activity", "membrane transport"),
    ("impaired proteostasis under stress", "protein homeostasis"),
    ("altered mitochondrial calcium buffering", "bioenergetics"),
    ("dysregulated inflammatory signalling", "immune signalling"),
    ("disrupted circadian gene expression", "chronobiology"),
    ("compensatory receptor desensitisation", "receptor pharmacology"),
    ("aberrant RNA splicing of a regulatory exon", "RNA biology"),
    ("shifted redox balance in the cytosol", "redox biology"),
)


def _goal_fragment(text: str) -> str:
    """Pull a short, readable fragment of the research goal out of a prompt.

    Operates line-wise on purpose: prompts put the goal on its own line followed by
    instructions, so collapsing whitespace first would splice the boilerplate in.
    """
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    if not lines:
        return "the stated research goal"
    candidate = ""
    for line in lines:
        match = re.match(r"(?:research\s+)?goal\s*[:\-]\s*(.+)", line, re.IGNORECASE)
        if match:
            candidate = match.group(1)
            break
    candidate = (candidate or lines[0]).strip().strip('"').strip()
    # Keep one sentence at most.
    candidate = re.split(r"(?<=[.?!])\s", candidate)[0].strip()
    if not candidate:
        return "the stated research goal"
    return (candidate[:100].rstrip() + "…") if len(candidate) > 100 else candidate


class OfflineBackend:
    """A deterministic backend satisfying the portable ``call`` contract.

    One composite response satisfies every agent parser in the repo: the generation
    agent reads ``Hypothesis:``/``Summary:``, reflection reads ``Verdict:``, and
    ranking reads ``better hypothesis:``. Emitting all of them in plain text (never
    JSON, which the parsers would each try to claim) keeps a single code path.
    """

    model_fast = _MODEL_NAME
    model_strong = _MODEL_NAME

    def __init__(self, seed: int = 0, model_strong: str | None = None,
                 model_fast: str | None = None):
        self.seed = int(seed)
        # Overrides are accepted for interface parity but intentionally ignored:
        # a run must never claim to have used a model it did not use.
        self.model_strong = _MODEL_NAME
        self.model_fast = _MODEL_NAME
        self.call_count = 0

    def _digest(self, system_prompt: str, user_prompt: str) -> int:
        payload = f"{self.seed}\x00{system_prompt}\x00{user_prompt}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        use_strong: bool = False,
        max_tokens: int = 8192,
    ) -> str:
        self.call_count += 1
        digest = self._digest(system_prompt, user_prompt)
        mechanism, category = _MECHANISMS[digest % len(_MECHANISMS)]
        subject = _goal_fragment(user_prompt)
        # Alternate the tournament winner so Elo actually moves instead of degenerating
        # to a single ordering.
        winner = 1 if (digest >> 8) % 2 == 0 else 2
        verdict = "PASSED" if (digest >> 16) % 4 else "FLAGGED"
        # tools.verify.verify_claim needs a VERDICT: line (SUPPORTED / UNSUPPORTED /
        # CONTRADICTED) so the free `--provider offline` path actually exercises
        # claim verification instead of extract_claims always returning [].
        judgement = ("CONTRADICTED" if (digest >> 24) % 8 == 0
                     else "UNSUPPORTED" if (digest >> 24) % 3 == 0
                     else "SUPPORTED")

        # One composite response satisfies every parser in the repo. The ALL-CAPS
        # labels are consumed by core.config_parser._extract, which terminates each
        # field at the next `\n[A-Z][A-Z]+:` — hence the BODY: sentinel before the
        # mixed-case agent markers below it.
        return (
            "DECISION: APPROVED\n"
            # tools.verify._REASON_RE.search() (like core.config_parser._extract)
            # returns the FIRST "REASON:" line in the whole response -- including
            # this safety-review one -- not the one written next to VERDICT: further
            # down. A CONTRADICTED verdict is only honoured if verify_claim can find
            # a `[N]` citation to a record in the judge's rationale (it must point at
            # an abstract-backed record, or the contradiction is downgraded to
            # unchecked rather than trusted on a title alone), so the citation has to
            # live in *this* line, or the offline path could never reach
            # `contradicted` no matter what VERDICT: says.
            "REASON: Deterministic offline fixture — no substantive safety review was "
            "performed. Do not treat this approval as a safety judgement. Offline "
            "claim verification cites fixture record [1] for its rationale.\n"
            "PREFERENCES: Prefer mechanistic, experimentally testable hypotheses.\n"
            "ATTRIBUTES: Novelty, Feasibility\n"
            "CONSTRAINTS: Must be testable with standard laboratory methods.\n"
            "BODY:\n"
            # Literal uppercase HYPOTHESIS is the debate/assumptions turn-loop
            # termination signal; "better idea:" terminates the ranking loop. Both
            # must appear verbatim or the loops run to max_turns and then raise.
            "HYPOTHESIS\n"
            f"Hypothesis: {subject} may involve {mechanism}, which predicts a "
            "measurable change under the stated constraints.\n"
            f"Summary: Synthetic offline hypothesis — {mechanism}.\n"
            f"Category: {category}\n"
            f"Verdict: {verdict}\n"
            "Rationale: Generated by the deterministic offline provider; this is a "
            "plumbing fixture and carries no scientific evidence.\n"
            f"better hypothesis: {winner}\n"
            f"better idea: {winner}\n"
            "Research overview: offline deterministic run — the hypotheses above are "
            "templated fixtures, not model reasoning.\n"
            # Numbered claims + an ALL-CAPS VERDICT line so tools.verify's
            # extract_claims/_parse_numbered and _VERDICT_RE have something to
            # parse offline. Placed at the very end (well after BODY:) so
            # core.config_parser's `\n[A-Z][A-Z]+:` field terminator, already
            # closed by BODY:, is unaffected, and distinct from the lowercase
            # "Verdict: PASSED" line above that agents/reflection.py parses.
            f"1. {subject} involves {mechanism}\n"
            f"2. {subject} is measurable in {category}\n"
            f"VERDICT: {judgement}\n"
            "REASON: deterministic offline fixture citing record [1]; carries no "
            "scientific meaning.\n"
        )
