"""BM25 sparse/lexical retrieval over the ARC Corpus."""
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Retriever:
    passages: list[str]
    bm25: BM25Okapi

    @classmethod
    def build(cls, passages: list[str]) -> "BM25Retriever":
        tokenized = [tokenize(p) for p in passages]
        return cls(passages=passages, bm25=BM25Okapi(tokenized))

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top_indices = scores.argsort()[::-1][:top_k]
        return [(self.passages[i], float(scores[i])) for i in top_indices]
