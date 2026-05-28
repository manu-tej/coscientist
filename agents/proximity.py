import numpy as np
from sentence_transformers import SentenceTransformer
from core.models import Hypothesis
from core.state import StateStore


class ProximityAgent:
    def __init__(
        self,
        store: StateStore,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        duplicate_threshold: float = 0.92,
    ):
        self.store = store
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.duplicate_threshold = duplicate_threshold

    async def update_graph(self, hypotheses: list[Hypothesis]) -> list[str]:
        if len(hypotheses) < 2:
            return []
        texts = [h.summary for h in hypotheses]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        similarity_matrix = np.dot(embeddings, embeddings.T)
        near_duplicates: list[str] = []
        for i in range(len(hypotheses)):
            for j in range(i + 1, len(hypotheses)):
                score = float(similarity_matrix[i][j])
                if score >= self.similarity_threshold:
                    await self.store.save_proximity(
                        hypotheses[i].id, hypotheses[j].id, score
                    )
                if score >= self.duplicate_threshold:
                    # Flag the lower-Elo one as near-duplicate
                    lower = (
                        hypotheses[i].id
                        if hypotheses[i].elo_rating <= hypotheses[j].elo_rating
                        else hypotheses[j].id
                    )
                    near_duplicates.append(lower)
        return near_duplicates
