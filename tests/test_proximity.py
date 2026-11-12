import asyncio
import threading
import time

import numpy as np
import pytest

from agents import proximity
from core.models import Hypothesis


class _Store:
    def __init__(self):
        self.saved = []

    async def save_proximity(self, hypothesis_1, hypothesis_2, score):
        self.saved.append((hypothesis_1, hypothesis_2, score))


@pytest.fixture(autouse=True)
def clear_model_cache():
    with proximity._MODEL_CACHE_LOCK:
        proximity._MODEL_CACHE.clear()
        proximity._INFERENCE_LOCKS.clear()
    yield
    with proximity._MODEL_CACHE_LOCK:
        proximity._MODEL_CACHE.clear()
        proximity._INFERENCE_LOCKS.clear()


@pytest.mark.asyncio
async def test_proximity_lazily_shares_model_and_serializes_inference(monkeypatch):
    constructions = 0
    active = 0
    max_active = 0
    counter_lock = threading.Lock()

    class FakeModel:
        def encode(self, texts, *, normalize_embeddings):
            nonlocal active, max_active
            assert normalize_embeddings is True
            with counter_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.01)
            with counter_lock:
                active -= 1
            return np.eye(len(texts))

    def create_model(model_name):
        nonlocal constructions
        constructions += 1
        assert model_name == "tiny-test-model"
        return FakeModel()

    monkeypatch.setattr(proximity, "_create_model", create_model)
    stores = [_Store(), _Store()]
    agents = [
        proximity.ProximityAgent(store, model_name="tiny-test-model")
        for store in stores
    ]
    assert constructions == 0
    hypotheses = [
        Hypothesis(
            id="h1", run_id="r", text="one", summary="one",
            generation_method="test", source="system", category="test",
        ),
        Hypothesis(
            id="h2", run_id="r", text="two", summary="two",
            generation_method="test", source="system", category="test",
        ),
    ]

    await asyncio.gather(*(agent.update_graph(hypotheses) for agent in agents))

    assert constructions == 1
    assert max_active == 1


@pytest.mark.asyncio
async def test_absolute_duplicate_ties_use_shared_canonical_order(
    monkeypatch, tmp_path,
):
    from bench.absolute import AbsoluteRankingAgent
    from bench.goldset import rank_hypotheses
    from core.models import AbsoluteScore
    from core.state import StateStore

    class SameEmbeddingModel:
        def encode(self, texts, *, normalize_embeddings):
            return np.ones((len(texts), 1))

    monkeypatch.setattr(
        proximity, "_create_model", lambda model_name: SameEmbeddingModel()
    )
    store = StateStore(str(tmp_path / "ties.db"))
    await store.init_db()
    hypotheses = [
        Hypothesis(
            id=identifier,
            run_id="r",
            text="same",
            summary="same",
            generation_method="test",
            source="system",
        )
        for identifier in ("a", "z")
    ]
    for hypothesis in hypotheses:
        await store.save_hypothesis(hypothesis)
        claim = await store.claim_absolute_score(
            hypothesis.id, "absolute-test"
        )
        await store.publish_absolute_score(AbsoluteScore(
            run_id="r",
            hypothesis_id=hypothesis.id,
            scorer_id="absolute-test",
            axis_scores={
                "novelty": 3,
                "feasibility": 3,
                "correctness": 3,
                "impact": 3,
            },
            total=3.0,
        ), owner_token=claim.owner_token)

    agent = proximity.ProximityAgent(
        store,
        model_name="tie-model",
        similarity_threshold=0.5,
        duplicate_threshold=0.9,
        ranking_mode="absolute",
    )

    assert await agent.update_graph(hypotheses) == ["z"]
    assert await agent.update_graph(list(reversed(hypotheses))) == ["z"]

    ranking = AbsoluteRankingAgent(
        judge=type("Judge", (), {"weights": None})(),
        store=store,
        scorer_id="absolute-test",
    )
    live_order = await ranking.order_hypotheses(list(reversed(hypotheses)))
    assert [hypothesis.id for hypothesis in live_order] == ["a", "z"]

    for hypothesis in hypotheses:
        hypothesis.absolute_score = 3.0
    goldset_order = rank_hypotheses(
        list(reversed(hypotheses)), ranking_mode="absolute"
    )
    assert [hypothesis.id for hypothesis in goldset_order] == ["a", "z"]
