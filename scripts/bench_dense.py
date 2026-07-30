"""Build dense passage embeddings over the ARC Corpus subset and sanity-check
retrieval: timing and top-k passages for a few real ARC questions.

Usage: python scripts/bench_dense.py [--max-passages N] [--embeddings-path PATH]
(omit --max-passages to use the shared CORPUS_SUBSET_SIZE from src/data/loader.py,
which BM25/dense/hybrid all default to so they're compared over the same corpus)
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import CORPUS_SUBSET_SIZE, iter_corpus, load_questions
from src.retrievers.dense import DenseRetriever, build_passage_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-passages", type=int, default=CORPUS_SUBSET_SIZE)
    parser.add_argument("--embeddings-path", default="/tmp/arc_data/dense_embeddings.npy")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    t0 = time.time()
    corpus_iter = iter_corpus()
    passages = list(itertools.islice(corpus_iter, args.max_passages)) if args.max_passages else list(corpus_iter)
    t1 = time.time()
    print(f"loaded {len(passages):,} passages in {t1 - t0:.1f}s")

    build_passage_embeddings(passages, args.embeddings_path, batch_size=args.batch_size, log_every_seconds=120)
    t2 = time.time()
    print(f"built embeddings in {t2 - t1:.1f}s ({len(passages) / (t2 - t1):.0f} passages/sec)")

    retriever = DenseRetriever.load(passages, args.embeddings_path)
    sample_questions = load_questions("challenge", "Dev")[:3]
    for q in sample_questions:
        print(f"\n=== {q.question}")
        for text, score in retriever.retrieve(q.question, top_k=5):
            print(f"  {score:6.3f}  {text[:120]}")


if __name__ == "__main__":
    main()
