"""Compute Recall@10 and MRR for BM25/dense/hybrid against the relevance-labeled sample."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_corpus_subset, load_questions
from src.eval.retrieval_metrics import retrieval_report
from src.reader.model import filter_to_four_choice
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DenseRetriever
from src.retrievers.hybrid import HybridRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/relevance_labels/sample.json")
    parser.add_argument("--embeddings-path", default="/tmp/arc_data/dense_embeddings.npy")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", default="results/retrieval_metrics.json")
    args = parser.parse_args()

    with open(args.labels) as f:
        labels = json.load(f)

    print("loading corpus subset + building retrievers...")
    passages = load_corpus_subset()
    bm25 = BM25Retriever.build(passages)
    dense = DenseRetriever.load(passages, args.embeddings_path)
    hybrid = HybridRetriever.build(bm25)

    all_test = filter_to_four_choice(load_questions("easy", "Test")) + filter_to_four_choice(load_questions("challenge", "Test"))
    by_id = {q.id: q for q in all_test}

    questions, relevant_sets = [], []
    for qid, entry in labels.items():
        if qid not in by_id:
            continue
        questions.append(by_id[qid])
        relevant_sets.append(set(entry["relevant_passages"]))
    print(f"scoring against {len(questions)} labeled questions")

    results = {}
    for name, retriever in [("bm25", bm25), ("dense", dense), ("hybrid", hybrid)]:
        retrieved = [[text for text, _ in retriever.retrieve(q.question, top_k=args.k)] for q in questions]
        results[name] = retrieval_report(questions, retrieved, relevant_sets, k=args.k)
        print(name, results[name])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved to {output_path}")


if __name__ == "__main__":
    main()
