"""End-to-end QA accuracy, broken out by Easy/Challenge/overall."""
from collections import defaultdict

from src.data.loader import ArcQuestion


def accuracy_report(questions: list[ArcQuestion], predictions: list[str]) -> dict[str, float]:
    """predictions[i] is the predicted choice label (e.g. 'A') for questions[i].

    Returns accuracy keyed by question.split ('easy' / 'challenge') plus 'overall'.
    """
    assert len(questions) == len(predictions)
    correct_by_split = defaultdict(int)
    total_by_split = defaultdict(int)
    for q, pred in zip(questions, predictions):
        total_by_split[q.split] += 1
        if pred == q.answer_key:
            correct_by_split[q.split] += 1

    report = {split: correct_by_split[split] / total_by_split[split] for split in total_by_split}
    total = sum(total_by_split.values())
    correct = sum(correct_by_split.values())
    report["overall"] = correct / total if total else 0.0
    return report
