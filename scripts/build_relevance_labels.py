"""Build a relevance-labeled sample for Recall@10/MRR.

Samples ~200 questions (100 Easy + 100 Challenge) from the Test sets, pools
top-20 candidates from BM25 + dense + hybrid for each (TREC-style pooling --
union across all three systems being compared, not just one, so the ground
truth isn't biased toward whichever retriever built it), and auto-labels
relevance via a retriever-agnostic lexical-overlap heuristic against
(question + correct answer). See PROJECT_BRIEF.md for the design rationale.
"""
import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import CORPUS_SUBSET_SIZE, ArcQuestion, iter_corpus, load_questions
from src.eval.relevance_heuristic import is_relevant
from src.reader.model import filter_to_four_choice
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DenseRetriever
from src.retrievers.hybrid import HybridRetriever

POOL_K = 20
N_PER_SPLIT = 100


def sample_questions(n_per_split: int, seed: int = 42) -> list[ArcQuestion]:
    easy = filter_to_four_choice(load_questions("easy", "Test"))
    challenge = filter_to_four_choice(load_questions("challenge", "Test"))
    rng = random.Random(seed)
    return rng.sample(easy, n_per_split) + rng.sample(challenge, n_per_split)


def correct_choice_text(q: ArcQuestion) -> str:
    return q.choices[q.choice_labels.index(q.answer_key)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-path", default="/tmp/arc_data/dense_embeddings.npy")
    parser.add_argument("--output", default="data/relevance_labels/sample.json")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--n-per-split", type=int, default=N_PER_SPLIT)
    parser.add_argument(
        "--max-passages", type=int, default=CORPUS_SUBSET_SIZE,
        help="must match whatever built --embeddings-path, or dense retrieval will misalign",
    )
    args = parser.parse_args()

    print("loading corpus + building retrievers...")
    t0 = time.time()
    corpus_iter = iter_corpus()
    passages = list(itertools.islice(corpus_iter, args.max_passages)) if args.max_passages else list(corpus_iter)
    bm25 = BM25Retriever.build(passages)
    dense = DenseRetriever.load(passages, args.embeddings_path)
    hybrid = HybridRetriever.build(bm25)
    print(f"retrievers ready over {len(passages):,} passages in {time.time() - t0:.1f}s")

    questions = sample_questions(args.n_per_split)
    print(
        f"sampled {len(questions)} questions "
        f"({sum(q.split == 'easy' for q in questions)} easy, "
        f"{sum(q.split == 'challenge' for q in questions)} challenge)"
    )

    labels = {}
    t0 = time.time()
    for i, q in enumerate(questions):
        query_text = f"{q.question} {correct_choice_text(q)}"
        pool = set()
        for text, _ in bm25.retrieve(q.question, top_k=POOL_K):
            pool.add(text)
        for text, _ in dense.retrieve(q.question, top_k=POOL_K):
            pool.add(text)
        for text, _ in hybrid.retrieve(q.question, top_k=POOL_K):
            pool.add(text)

        relevant = [p for p in pool if is_relevant(p, query_text, threshold=args.threshold)]
        labels[q.id] = {
            "split": q.split,
            "question": q.question,
            "pool_size": len(pool),
            "relevant_passages": relevant,
        }
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta_min = (elapsed / (i + 1)) * (len(questions) - i - 1) / 60
            print(f"[progress] {i + 1}/{len(questions)} labeled, elapsed {elapsed / 60:.1f}m eta {eta_min:.1f}m", flush=True)

    n_with_relevant = sum(1 for v in labels.values() if v["relevant_passages"])
    print(f"done: {n_with_relevant}/{len(labels)} questions have >=1 relevant passage in their pool")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"saved to {output_path}")


if __name__ == "__main__":
    main()
