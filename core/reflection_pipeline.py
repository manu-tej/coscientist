import logging
import re

from core.models import Hypothesis, ResearchPlanConfig
from core.state import StateStore
from tools.evidence import ClaimVerdict

logger = logging.getLogger(__name__)

_PMID_RE = re.compile(r"(?:pmid|pubmed)\s*[:# ]\s*(\d{4,9})", re.IGNORECASE)


def fabricated_citations(text: str, shown: set[str] | None) -> list[str]:
    """PMIDs cited by a hypothesis that were never retrieved for it.

    `shown=None` means the grounding set was not recorded, so nothing can be
    called fabricated — absence of a record is not evidence of invention.
    """
    if shown is None:
        return []
    cited = {m.group(1) for m in _PMID_RE.finditer(text or "")}
    return sorted(cited - shown)


def _stamp(review, hypothesis_id: str):
    """Ensure the review's hypothesis_id matches the actual hypothesis being reviewed."""
    review.hypothesis_id = hypothesis_id
    return review


async def run_reflection_pipeline(
    hypothesis: Hypothesis,
    config: ResearchPlanConfig,
    reflection_agent,
    search_tool,
    store: StateStore,
    meta_critique: str = "",
    verifier=None,
) -> str:
    """Run the 6 reflection tiers sequentially with early exit on rejection.

    Returns the final verdict string ("passed" | "rejected" | "flagged").
    Persists every review and appends positive observations to annotations.
    """
    hid = hypothesis.id

    # Tier 1 - initial review (no web search)
    r1 = _stamp(await reflection_agent.run_initial_review(hypothesis, config), hid)
    await store.save_review(r1)
    if r1.verdict == "rejected":
        await store.set_hypothesis_status(hid, "rejected")
        return "rejected"

    # Tier 2 - full review with literature
    articles = await search_tool.search_and_format(
        f"{config.goal} {hypothesis.summary}", context=hypothesis.text
    )
    r2 = _stamp(await reflection_agent.run_full_review(hypothesis, config, articles), hid)
    await store.save_review(r2)
    if r2.verdict == "rejected":
        await store.set_hypothesis_status(hid, "rejected")
        return "rejected"

    # Claim verification. Optional: a `verifier.verify()` failure records an
    # explicit `unchecked` row rather than silently writing nothing, so zero
    # rows is never ambiguous. (`Exception` here never includes CancelledError,
    # which is BaseException since Python 3.8, so cancellation still propagates.)
    if verifier is not None:
        reason = ""
        try:
            verdicts = await verifier.verify(
                hypothesis.text, search_tool, reflection_agent.base.client,
            )
            if not verdicts:
                reason = "no checkable claims extracted"
        except Exception as exc:  # noqa: BLE001
            logger.warning("claim verification failed for %s (%s)", hid, exc)
            verdicts, reason = [], f"verifier raised: {exc}"

        # Citation attribution: a PMID cited by the hypothesis but never among the
        # records shown to it at generation time is a deterministic, non-model-judged
        # fact -- it gets folded into `contradicted` the same as a judge's
        # contradiction verdict, so it earns the same quarantine response (see the
        # quarantine comment below). Isolated in its own try/except so a store
        # failure here (e.g. get_grounding_refs) cannot raise into the supervisor,
        # and so it cannot block the unrelated verdict persistence below.
        fabricated: list[str] = []
        try:
            fabricated = fabricated_citations(
                hypothesis.text, await store.get_grounding_refs(hid))
            for ref in fabricated:
                await store.save_claim_verdict(hid, ClaimVerdict(
                    claim=f"citation PMID {ref}", status="contradicted",
                    rationale="cited a record never retrieved for this hypothesis",
                ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("citation attribution check failed for %s (%s)", hid, exc)
            fabricated = []

        contradicted = bool(fabricated) or any(v.status == "contradicted" for v in verdicts)
        try:
            for verdict in verdicts:
                await store.save_claim_verdict(hid, verdict)
            if reason:
                await store.save_claim_verdict(hid, ClaimVerdict(
                    claim="", status="unchecked", rationale=reason,
                ))
            if contradicted:
                # Quarantine, not terminal rejection: the judge's false-contradiction
                # rate is unmeasured until sub-project 3 calibrates it. The hypothesis
                # leaves the active pool but stays distinguishable from a peer-review
                # rejection, so it can be re-scored once calibration exists.
                await store.set_hypothesis_status(hid, "quarantined")
        except Exception as exc:  # noqa: BLE001
            # A store failure here must degrade, not raise: an exception into the
            # supervisor is exactly the failure mode this block exists to avoid.
            # Falling through to tier 3 is the honest outcome -- returning "flagged"
            # would claim the hypothesis was quarantined when the status write may
            # never have landed, letting a recorded contradiction sit undetected
            # in the active pool.
            logger.warning("persisting claim verdicts failed for %s (%s); continuing", hid, exc)
        else:
            if contradicted:
                return "flagged"

    # Tier 3 - deep verification (non-fundamental errors do NOT discard)
    r3 = _stamp(await reflection_agent.run_deep_verification(hypothesis, config, r2.critique), hid)
    await store.save_review(r3)
    if r3.verdict == "rejected":
        await store.set_hypothesis_status(hid, "rejected")
        return "rejected"

    # Tier 4 - observation review; append positive observations
    r4_raw, observation = await reflection_agent.run_observation_review(hypothesis, config, articles)
    r4 = _stamp(r4_raw, hid)
    await store.save_review(r4)
    if observation:
        await store.append_annotation(hid, observation)
    if r4.verdict == "rejected":
        await store.set_hypothesis_status(hid, "rejected")
        return "rejected"

    # Tier 5 - simulation review (flags but does not reject)
    r5 = _stamp(await reflection_agent.run_simulation_review(hypothesis, config), hid)
    await store.save_review(r5)

    # Tier 6 - tournament/recurrent review: only if hypothesis has tournament history
    matches = await store.list_matches(config.run_id)
    in_tournament = any(
        m.h1_id == hid or m.h2_id == hid for m in matches
    )
    if in_tournament:
        prior = await store.list_reviews(hid)
        prior_text = "\n\n".join(f"Tier {r.tier}: {r.critique}" for r in prior)
        tournament_history = "\n".join(
            f"{m.match_type}: winner={m.winner_id}"
            for m in matches
            if m.h1_id == hid or m.h2_id == hid
        )
        r6 = _stamp(
            await reflection_agent.run_tournament_review(
                hypothesis, config, prior_text, meta_critique, tournament_history
            ),
            hid,
        )
        await store.save_review(r6)

    return "passed"
