"""Build a BM25 index over the ARC Corpus and sanity-check it: timing, peak memory,
and top-k passages for a few real ARC questions (does BM25 return sane results?).

Usage: python scripts/bench_bm25.py [--max-passages N]
(omit --max-passages to index the full ~14.6M-sentence corpus)
"""
import argparse
import itertools
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import iter_corpus, load_questions
from src.retrievers.bm25 import BM25Retriever


def peak_rss_gb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, KB on Linux.
    return raw / 1024**3 if sys.platform == "darwin" else raw / 1024**2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-passages", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()
    corpus_iter = iter_corpus()
    passages = list(itertools.islice(corpus_iter, args.max_passages)) if args.max_passages else list(corpus_iter)
    t1 = time.time()
    print(f"loaded {len(passages):,} passages in {t1 - t0:.1f}s")

    retriever = BM25Retriever.build(passages)
    t2 = time.time()
    print(f"built BM25 index in {t2 - t1:.1f}s, peak RSS so far: {peak_rss_gb():.2f} GB")

    sample_questions = load_questions("challenge", "Dev")[:3]
    for q in sample_questions:
        print(f"\n=== {q.question}")
        for text, score in retriever.retrieve(q.question, top_k=5):
            print(f"  {score:6.2f}  {text[:120]}")


if __name__ == "__main__":
    main()
