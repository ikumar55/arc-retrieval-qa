"""Build BM25 + hybrid retrievers over a corpus subsample and sanity-check
top-k passages for a few real ARC questions, comparing BM25-only vs. hybrid.

Usage: python scripts/bench_hybrid.py [--max-passages N]
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import iter_corpus, load_questions
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.hybrid import HybridRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-passages", type=int, default=2_000_000)
    args = parser.parse_args()

    t0 = time.time()
    passages = list(itertools.islice(iter_corpus(), args.max_passages))
    bm25 = BM25Retriever.build(passages)
    print(f"built BM25 over {len(passages):,} passages in {time.time() - t0:.1f}s")

    hybrid = HybridRetriever.build(bm25)

    for q in load_questions("challenge", "Dev")[:3]:
        print(f"\n=== {q.question}")
        print("--- BM25 top-5 ---")
        for text, score in bm25.retrieve(q.question, top_k=5):
            print(f"  {score:6.2f}  {text[:110]}")
        print("--- Hybrid top-5 (BM25 top-100 -> dense rerank) ---")
        for text, score in hybrid.retrieve(q.question, top_k=5, candidate_pool_size=100):
            print(f"  {score:6.3f}  {text[:110]}")


if __name__ == "__main__":
    main()
