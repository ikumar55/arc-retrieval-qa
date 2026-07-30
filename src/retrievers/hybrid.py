"""Hybrid retrieval: BM25 produces a high-recall candidate pool, a dense
bi-encoder reranks it by cosine similarity.

Unlike DenseRetriever, this doesn't need precomputed embeddings for the whole
corpus -- it only embeds the (small) BM25 candidate pool per query, on the
fly, which is cheap even without a GPU.
"""
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DEFAULT_MODEL


@dataclass
class HybridRetriever:
    bm25: BM25Retriever
    model: SentenceTransformer

    @classmethod
    def build(cls, bm25: BM25Retriever, model_name: str = DEFAULT_MODEL, device: str | None = None) -> "HybridRetriever":
        model = SentenceTransformer(model_name, device=device)
        return cls(bm25=bm25, model=model)

    def retrieve(self, query: str, top_k: int = 10, candidate_pool_size: int = 100) -> list[tuple[str, float]]:
        candidates = self.bm25.retrieve(query, top_k=candidate_pool_size)
        if not candidates:
            return []
        candidate_texts = [text for text, _ in candidates]

        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        candidate_embs = self.model.encode(candidate_texts, normalize_embeddings=True)
        scores = candidate_embs @ query_emb

        order = np.argsort(-scores)[:top_k]
        return [(candidate_texts[i], float(scores[i])) for i in order]
