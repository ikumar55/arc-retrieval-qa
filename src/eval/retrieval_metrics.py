"""Recall@10 and MRR over retrieved passages, split by Easy/Challenge/overall.

These operate on a labeled relevance sample (see data/relevance_labels/) --
for each question, which corpus passage(s) count as relevant ground truth.
ARC doesn't ship this; relevant passages are identified by exact text match
against the retriever's passage strings.
"""
from collections import defaultdict

from src.data.loader import ArcQuestion


def recall_at_k(retrieved: list[str], relevant: set[str], k: int = 10) -> float:
    """1.0 if any of the top-k retrieved passages is relevant, else 0.0."""
    return 1.0 if any(p in relevant for p in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant passage in `retrieved` (1-indexed), else 0.0."""
    for rank, passage in enumerate(retrieved, start=1):
        if passage in relevant:
            return 1.0 / rank
    return 0.0


def retrieval_report(
    questions: list[ArcQuestion],
    retrieved_per_question: list[list[str]],
    relevant_per_question: list[set[str]],
    k: int = 10,
) -> dict[str, dict[str, float]]:
    """Returns {'recall@10': {'easy':.., 'challenge':.., 'overall':..}, 'mrr': {...}}."""
    assert len(questions) == len(retrieved_per_question) == len(relevant_per_question)

    recall_by_split = defaultdict(list)
    mrr_by_split = defaultdict(list)
    for q, retrieved, relevant in zip(questions, retrieved_per_question, relevant_per_question):
        recall_by_split[q.split].append(recall_at_k(retrieved, relevant, k=k))
        mrr_by_split[q.split].append(reciprocal_rank(retrieved, relevant))

    def _report(by_split: dict[str, list[float]]) -> dict[str, float]:
        report = {split: sum(vals) / len(vals) for split, vals in by_split.items()}
        all_vals = [v for vals in by_split.values() for v in vals]
        report["overall"] = sum(all_vals) / len(all_vals) if all_vals else 0.0
        return report

    return {f"recall@{k}": _report(recall_by_split), "mrr": _report(mrr_by_split)}
