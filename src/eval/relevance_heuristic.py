"""Retriever-agnostic lexical-overlap heuristic for silver relevance labels.

Used only to build data/relevance_labels/ (see scripts/build_relevance_labels.py),
not at eval time -- Recall@10/MRR (src/eval/retrieval_metrics.py) just consume
the resulting labels.
"""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def content_words(text: str) -> set[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return {t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1}


def is_relevant(passage: str, query_text: str, threshold: float = 0.4) -> bool:
    """query_text is typically `question + " " + correct_answer_choice`.

    Relevant if at least `threshold` fraction of the query's distinct content
    words appear (verbatim) in the passage. Coverage relative to the query
    rather than full Jaccard, since a good supporting passage may legitimately
    contain extra context beyond just the query's terms.
    """
    query_words = content_words(query_text)
    if not query_words:
        return False
    passage_words = content_words(passage)
    overlap = len(query_words & passage_words) / len(query_words)
    return overlap >= threshold
