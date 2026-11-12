from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence

from bench.errors import BenchError
from bench.judge import RUBRIC_AXES, normalize_rubric_weights, weighted_total
from core.models import AbsoluteScore, Hypothesis, ResearchPlanConfig
from core.ranking import native_ranking_key
from core.state import StateStore


class AbsoluteRankingAgent:
    """Score hypotheses independently once, without creating pairwise matches.

    Claims live in SQLite rather than this object, so separate runner instances
    sharing a run database coalesce the paid judge call. Successful scores are
    immutable for a scorer identity and can be replayed from the run database.
    """

    def __init__(
        self,
        *,
        judge,
        store: StateStore,
        scorer_id: str,
        field: str = "computational biology",
        max_concurrency: int = 1,
        claim_lease_seconds: float = 300.0,
        claim_poll_seconds: float = 0.05,
    ):
        if not isinstance(scorer_id, str) or not scorer_id.strip():
            raise ValueError("scorer_id must be nonempty")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if not math.isfinite(claim_lease_seconds) or claim_lease_seconds <= 0:
            raise ValueError("claim_lease_seconds must be finite and > 0")
        if not math.isfinite(claim_poll_seconds) or claim_poll_seconds <= 0:
            raise ValueError("claim_poll_seconds must be finite and > 0")
        self.judge = judge
        self.store = store
        self.scorer_id = scorer_id.strip()
        self.field = field
        self.max_concurrency = max_concurrency
        self.claim_lease_seconds = claim_lease_seconds
        self.claim_poll_seconds = claim_poll_seconds
        self.weights = normalize_rubric_weights(getattr(judge, "weights", None))

    def _validated_score(
        self,
        result: object,
        *,
        hypothesis: Hypothesis,
        config: ResearchPlanConfig,
    ) -> AbsoluteScore:
        if not isinstance(result, Mapping):
            raise BenchError("absolute judge result must be a mapping")
        raw_scores = result.get("scores")
        if not isinstance(raw_scores, Mapping):
            raise BenchError("absolute judge result is missing rubric scores")
        if set(raw_scores) != set(RUBRIC_AXES):
            raise BenchError(
                "absolute judge result must contain exactly the rubric axes"
            )
        scores: dict[str, int] = {}
        for axis in RUBRIC_AXES:
            value = raw_scores.get(axis)
            if isinstance(value, bool) or not isinstance(value, int):
                raise BenchError(
                    f"absolute judge score for {axis!r} must be an integer"
                )
            if value < 1 or value > 5:
                raise BenchError(
                    f"absolute judge score for {axis!r} must be within [1, 5]"
                )
            scores[axis] = value
        total = weighted_total(scores, self.weights)
        supplied_total = result.get("total")
        if (
            isinstance(supplied_total, bool)
            or not isinstance(supplied_total, (int, float))
            or not math.isfinite(float(supplied_total))
            or not math.isclose(float(supplied_total), total, abs_tol=1e-9)
        ):
            raise BenchError(
                "absolute judge total is missing or inconsistent with rubric scores"
            )
        return AbsoluteScore(
            run_id=config.run_id,
            hypothesis_id=hypothesis.id,
            scorer_id=self.scorer_id,
            axis_scores=scores,
            total=total,
        )

    async def _heartbeat_claim(
        self,
        hypothesis_id: str,
        owner_token: str,
        stop: asyncio.Event,
        lost: asyncio.Event,
    ) -> None:
        interval = max(0.01, self.claim_lease_seconds / 3.0)
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
                return
            except TimeoutError:
                try:
                    renewed = await self.store.renew_absolute_score_claim(
                        hypothesis_id,
                        self.scorer_id,
                        owner_token,
                        lease_seconds=self.claim_lease_seconds,
                    )
                except Exception:
                    renewed = False
                if not renewed:
                    try:
                        recovery = await self.store.claim_absolute_score(
                            hypothesis_id,
                            self.scorer_id,
                            lease_seconds=self.claim_lease_seconds,
                            owner_token=owner_token,
                        )
                    except Exception:
                        renewed = False
                    else:
                        renewed = (
                            recovery.state == "acquired"
                            and recovery.owner_token == owner_token
                        )
                if not renewed:
                    lost.set()
                    return

    async def _score_one(
        self,
        hypothesis: Hypothesis,
        config: ResearchPlanConfig,
    ) -> AbsoluteScore:
        while True:
            existing = await self.store.get_absolute_score(hypothesis.id)
            if existing is not None:
                if existing.scorer_id != self.scorer_id:
                    raise BenchError(
                        "persisted absolute score uses a different scorer identity"
                    )
                return existing

            claim = await self.store.claim_absolute_score(
                hypothesis.id,
                self.scorer_id,
                lease_seconds=self.claim_lease_seconds,
            )
            if claim.state == "ready":
                continue
            if claim.state == "waiting":
                await asyncio.sleep(self.claim_poll_seconds)
                continue

            owner_token = claim.owner_token
            if owner_token is None:
                raise RuntimeError("acquired absolute-score claim has no owner token")
            stop = asyncio.Event()
            lost = asyncio.Event()
            heartbeat = asyncio.create_task(
                self._heartbeat_claim(hypothesis.id, owner_token, stop, lost)
            )
            published = False
            try:
                # Only hypothesis text and the public domain are sent. Source,
                # variant, Elo, and agent provenance remain blinded.
                result = await self._call_judge_while_claimed(
                    hypothesis.text, lost
                )
                score = self._validated_score(
                    result, hypothesis=hypothesis, config=config
                )
                if lost.is_set():
                    raise RuntimeError("lost absolute-score claim during judge call")
                await self.store.publish_absolute_score(
                    score, owner_token=owner_token
                )
                published = True
                return score
            finally:
                stop.set()
                try:
                    await heartbeat
                finally:
                    if not published:
                        await self.store.release_absolute_score_claim(
                            hypothesis.id, self.scorer_id, owner_token
                        )

    async def _call_judge_while_claimed(
        self,
        hypothesis_text: str,
        lost: asyncio.Event,
    ) -> object:
        """Cancel and fully drain a provider call if its claim is lost.

        Awaiting the cancelled task is deliberate: a custom backend may suppress
        cancellation. We do not return (or publish) until that backend actually
        stops, which prevents a failed run from mutating its DB in the background.
        """
        judge_task = asyncio.create_task(
            self.judge.score(hypothesis_text, field=self.field)
        )
        lost_task = asyncio.create_task(lost.wait())
        try:
            done, _ = await asyncio.wait(
                {judge_task, lost_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if lost_task in done and lost.is_set():
                judge_task.cancel()
                await asyncio.gather(judge_task, return_exceptions=True)
                raise RuntimeError("lost absolute-score claim during judge call")
            return await judge_task
        except BaseException:
            if not judge_task.done():
                judge_task.cancel()
            await asyncio.gather(judge_task, return_exceptions=True)
            raise
        finally:
            if not lost_task.done():
                lost_task.cancel()
            await asyncio.gather(lost_task, return_exceptions=True)

    async def score_hypotheses(
        self,
        hypotheses: Sequence[Hypothesis],
        config: ResearchPlanConfig,
    ) -> dict[str, AbsoluteScore]:
        """Persist scores for every active hypothesis, at bounded concurrency."""
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def guarded(hypothesis: Hypothesis) -> AbsoluteScore:
            async with semaphore:
                return await self._score_one(hypothesis, config)

        tasks = [asyncio.create_task(guarded(h)) for h in hypotheses]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Drain even cancellation-suppressing custom judges before exposing
            # the failure to run_system (which may then close backends/retry).
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return {score.hypothesis_id: score for score in results}

    async def order_hypotheses(
        self, hypotheses: Sequence[Hypothesis],
    ) -> list[Hypothesis]:
        """Order by persisted absolute total; leave not-yet-scored ideas last."""
        if not hypotheses:
            return []
        scores = await self.store.list_absolute_scores(hypotheses[0].run_id)
        for score in scores.values():
            if score.scorer_id != self.scorer_id:
                raise BenchError(
                    "persisted absolute scores use a different scorer identity"
                )
        return sorted(
            hypotheses,
            key=lambda hypothesis: native_ranking_key(
                hypothesis,
                scores[hypothesis.id].total
                if hypothesis.id in scores
                else None,
            ),
        )
