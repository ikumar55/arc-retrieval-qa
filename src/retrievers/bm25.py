"""BM25 sparse/lexical retrieval over the ARC Corpus.

Implemented directly on a scipy sparse term-document matrix (via sklearn's
CountVectorizer) rather than rank_bm25's pure-Python tokenized-corpus
approach, which stores every token as a Python string in memory and needs
~2GB per million passages -- too much for the ARC Corpus's 14.6M passages
under a 16GB memory budget. The sparse-matrix version only stores nonzero
(doc, term) counts as compact int32/float32 arrays.
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

TOKEN_PATTERN = r"[a-z0-9]+"
K1 = 1.5
B = 0.75


@dataclass
class BM25Retriever:
    passages: list[str]
    vectorizer: CountVectorizer
    doc_term_csc: sp.csc_matrix  # (n_docs, n_terms) raw term counts, column-sliceable
    idf: np.ndarray  # (n_terms,)
    doc_len: np.ndarray  # (n_docs,)
    avg_doc_len: float
    k1: float = K1
    b: float = B

    @classmethod
    def build(cls, passages: list[str], k1: float = K1, b: float = B) -> "BM25Retriever":
        vectorizer = CountVectorizer(lowercase=True, token_pattern=TOKEN_PATTERN, dtype=np.float32)
        doc_term = vectorizer.fit_transform(passages)
        doc_len = np.asarray(doc_term.sum(axis=1)).ravel()
        avg_doc_len = float(doc_len.mean())

        doc_term_csc = doc_term.tocsc()
        n_docs = doc_term.shape[0]
        df = np.diff(doc_term_csc.indptr)  # number of docs containing each term
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)

        return cls(
            passages=passages,
            vectorizer=vectorizer,
            doc_term_csc=doc_term_csc,
            idf=idf,
            doc_len=doc_len,
            avg_doc_len=avg_doc_len,
            k1=k1,
            b=b,
        )

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        analyzer = self.vectorizer.build_analyzer()
        vocab = self.vectorizer.vocabulary_
        term_ids = sorted({vocab[t] for t in analyzer(query) if t in vocab})
        if not term_ids:
            return []

        n_docs = len(self.passages)
        scores = np.zeros(n_docs, dtype=np.float32)
        length_norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avg_doc_len)

        indptr = self.doc_term_csc.indptr
        indices = self.doc_term_csc.indices
        data = self.doc_term_csc.data
        for tid in term_ids:
            start, end = indptr[tid], indptr[tid + 1]
            rows = indices[start:end]
            tf = data[start:end]
            denom = tf + length_norm[rows]
            scores[rows] += self.idf[tid] * (tf * (self.k1 + 1)) / denom

        k = min(top_k, n_docs)
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self.passages[i], float(scores[i])) for i in top_idx]
