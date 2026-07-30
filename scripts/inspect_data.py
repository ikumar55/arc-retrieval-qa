"""Quick sanity check: load ARC Easy/Challenge + Corpus and print summary stats."""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import iter_corpus, load_all_questions


def main():
    questions = load_all_questions()

    print("=== ARC Easy/Challenge question counts ===")
    for key, qs in questions.items():
        print(f"  {key:16s} {len(qs)}")

    print("\n=== Sample question (challenge_train[0]) ===")
    q = questions["challenge_train"][0]
    print(f"  id: {q.id}")
    print(f"  question: {q.question}")
    for label, choice in zip(q.choice_labels, q.choices):
        marker = "*" if label == q.answer_key else " "
        print(f"    [{marker}] {label}. {choice}")

    print("\n=== ARC Corpus (streaming first 5 lines + total line count) ===")
    corpus_iter = iter_corpus()
    for line in itertools.islice(corpus_iter, 5):
        print(" ", line[:100])

    print("\nCounting total corpus lines (this scans the full 14M-line file, may take ~30-60s)...")
    total = sum(1 for _ in iter_corpus())
    print(f"  total sentences: {total:,}")


if __name__ == "__main__":
    main()
