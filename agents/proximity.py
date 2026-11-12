import asyncio
import logging
import threading
from typing import Any

import numpy as np
from core.models import Hypothesis
from core.ranking import native_ranking_key
from core.state import StateStore


logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, Any] = {}
_INFERENCE_LOCKS: dict[str, threading.Lock] = {}
_MODEL_CACHE_LOCK = threading.Lock()
# Non-empty once the embedding model has failed to load, so a missing optional
# dependency is reported once per process rather than once per proximity task.
_EMBEDDING_UNAVAILABLE: list[str] = []


def _create_model(model_name: str):
    """Import and construct the optional heavyweight model only when needed."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _cached_model(model_name: str) -> tuple[Any, threading.Lock]:
    """Share one immutable model and serialize inference for thread safety."""
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(model_name)
        if model is None:
            model = _create_model(model_name)
            _MODEL_CACHE[model_name] = model
        inference_lock = _INFERENCE_LOCKS.setdefault(model_name, threading.Lock())
        return model, inference_lock


class ProximityAgent:
    def __init__(
        self,
        store: StateStore,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        duplicate_threshold: float = 0.92,
        ranking_mode: str = "elo",
    ):
        if ranking_mode not in {"elo", "absolute"}:
            raise ValueError("ranking_mode must be 'elo' or 'absolute'")
        self.store = store
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.duplicate_threshold = duplicate_threshold
        self.ranking_mode = ranking_mode

    async def update_graph(self, hypotheses: list[Hypothesis]) -> list[str]:
        if len(hypotheses) < 2:
            return []
        texts = [h.summary for h in hypotheses]
        # SentenceTransformer inference and the matrix multiply are CPU-bound.
        # Offload both so LLM/network workers continue to make progress.
        def compute_matrix():
            model, inference_lock = _cached_model(self.model_name)
            with inference_lock:
                embeddings = model.encode(texts, normalize_embeddings=True)
            return np.dot(embeddings, embeddings.T)

        try:
            similarity_matrix = await asyncio.to_thread(compute_matrix)
        except Exception as exc:  # noqa: BLE001 - any load/inference failure must degrade
            # Deduplication is an optimisation, not a result. The embedding model is an
            # optional ~90 MB download; when it is unavailable (offline, no HF cache,
            # transient failure) skipping dedup is strictly better than failing a run
            # the user may have paid for. Cached so it is not retried once per task.
            if not _EMBEDDING_UNAVAILABLE:
                _EMBEDDING_UNAVAILABLE.append(str(exc))
                logger.warning(
                    "embedding model %r unavailable (%s); continuing without "
                    "hypothesis deduplication", self.model_name, exc,
                )
            return []
        absolute_scores = (
            await self.store.list_absolute_scores(hypotheses[0].run_id)
            if self.ranking_mode == "absolute"
            else {}
        )

        def ranking_key(
            hypothesis: Hypothesis,
        ) -> tuple[bool, float, str, str, str]:
            if self.ranking_mode == "absolute":
                score = absolute_scores.get(hypothesis.id)
                return native_ranking_key(
                    hypothesis, score.total if score is not None else None
                )
            return native_ranking_key(hypothesis, hypothesis.elo_rating)

        near_duplicates: list[str] = []
        for i in range(len(hypotheses)):
            for j in range(i + 1, len(hypotheses)):
                score = float(similarity_matrix[i][j])
                if score >= self.similarity_threshold:
                    await self.store.save_proximity(
                        hypotheses[i].id, hypotheses[j].id, score
                    )
                if score >= self.duplicate_threshold:
                    # The native ranking key is best-first, so reject the
                    # worse member. This also matches the shared deterministic
                    # summary/text/id tie order used by ranking and reporting.
                    worse = max(
                        (hypotheses[i], hypotheses[j]),
                        key=ranking_key,
                    )
                    near_duplicates.append(worse.id)
        return near_duplicates
