"""End-to-end QA accuracy for BM25/dense/hybrid conditions, using the
already-trained closed-book reader in inference-only mode (no retraining --
the reader is fixed across all conditions per PROJECT_BRIEF.md).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from transformers import Trainer, TrainingArguments

from src.data.loader import load_corpus_subset, load_questions
from src.eval.qa_metrics import accuracy_report
from src.reader.model import (
    DataCollatorForMultipleChoice,
    build_dataset,
    filter_to_four_choice,
    load_model_and_tokenizer,
)
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DenseRetriever
from src.retrievers.hybrid import HybridRetriever

IDX_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/tmp/arc_results/final_model")
    parser.add_argument("--embeddings-path", default="/tmp/arc_data/dense_embeddings.npy")
    parser.add_argument("--num-passages", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    print("loading corpus subset + building retrievers...")
    passages_corpus = load_corpus_subset()
    bm25 = BM25Retriever.build(passages_corpus)
    dense = DenseRetriever.load(passages_corpus, args.embeddings_path)
    hybrid = HybridRetriever.build(bm25)

    print(f"loading trained reader from {args.model_path}...")
    tokenizer, model = load_model_and_tokenizer(args.model_path)
    collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/eval_scratch",
            per_device_eval_batch_size=args.batch_size,
            report_to=[],
        ),
        data_collator=collator,
    )

    test_questions = filter_to_four_choice(load_questions("easy", "Test")) + filter_to_four_choice(
        load_questions("challenge", "Test")
    )
    print(f"evaluating on {len(test_questions)} test questions")

    all_results = {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, retriever in [("bm25", bm25), ("dense", dense), ("hybrid", hybrid)]:
        print(f"retrieving top-{args.num_passages} passages for condition: {name}")
        passages_per_question = [
            [text for text, _ in retriever.retrieve(q.question, top_k=args.num_passages)] for q in test_questions
        ]
        ds = build_dataset(test_questions, tokenizer, max_length=args.max_length, passages=passages_per_question)

        out = trainer.predict(ds)
        preds = np.argmax(out.predictions, axis=1)
        pred_labels = [IDX_TO_LABEL[i] for i in preds]
        report = accuracy_report(test_questions, pred_labels)
        all_results[name] = report
        print(name, report)

        with open(output_dir / f"qa_accuracy_{name}.json", "w") as f:
            json.dump(report, f, indent=2)

    with open(output_dir / "qa_accuracy_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"saved all results to {output_dir}")


if __name__ == "__main__":
    main()
