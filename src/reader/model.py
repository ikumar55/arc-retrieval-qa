"""Multiple-choice reader: [question (+ retrieved passages) + choice] -> compatibility score.

Built on transformers' AutoModelForMultipleChoice, which already implements exactly
the brief's scoring scheme (one score per choice, softmax over choices at loss time).
Closed-book and retrieval-augmented conditions share this same model/preprocessing;
retrieval-augmented runs just pass `passages` into `build_dataset`.
"""
from dataclasses import dataclass
from typing import Optional

import datasets as hf_datasets
import torch
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from src.data.loader import ArcQuestion

LABEL_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
NUM_CHOICES = 4


def filter_to_four_choice(questions: list[ArcQuestion]) -> list[ArcQuestion]:
    """Drop the small number of ARC questions that don't have exactly 4 choices
    with a recognized answer key. Returns the filtered list."""
    kept = [
        q
        for q in questions
        if len(q.choices) == NUM_CHOICES and q.answer_key in LABEL_TO_IDX
    ]
    dropped = len(questions) - len(kept)
    if dropped:
        print(f"filter_to_four_choice: dropped {dropped}/{len(questions)} questions")
    return kept


def load_model_and_tokenizer(model_name: str) -> tuple[PreTrainedTokenizerBase, AutoModelForMultipleChoice]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Force fp32: recent transformers versions default to whatever dtype is stored
    # in the checkpoint config, and deberta-v3-base's happens to be fp16, which
    # underflows to NaN gradients when trained without AMP/loss scaling.
    model = AutoModelForMultipleChoice.from_pretrained(model_name, dtype=torch.float32)
    return tokenizer, model


def build_dataset(
    questions: list[ArcQuestion],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 192,
    passages: Optional[list[list[str]]] = None,
) -> hf_datasets.Dataset:
    """passages[i], if given, is a list of retrieved passage strings for questions[i],
    concatenated after the question stem (closed-book if passages is None)."""
    if passages is not None:
        assert len(passages) == len(questions)

    first_sentences, second_sentences, labels = [], [], []
    for i, q in enumerate(questions):
        context = q.question
        if passages is not None and passages[i]:
            context = f"{context} {' '.join(passages[i])}"
        first_sentences.extend([context] * NUM_CHOICES)
        second_sentences.extend(q.choices)
        labels.append(LABEL_TO_IDX[q.answer_key])

    tokenized = tokenizer(first_sentences, second_sentences, truncation=True, max_length=max_length)

    # Regroup the flat (num_examples * NUM_CHOICES) lists back into
    # (num_examples, NUM_CHOICES) so each row is one multiple-choice example.
    regrouped = {
        key: [values[i : i + NUM_CHOICES] for i in range(0, len(values), NUM_CHOICES)]
        for key, values in tokenized.items()
    }
    regrouped["label"] = labels
    return hf_datasets.Dataset.from_dict(regrouped)


@dataclass
class DataCollatorForMultipleChoice:
    """Dynamically pads a batch of multiple-choice examples.

    Each example's fields are lists-of-lists shaped (NUM_CHOICES, seq_len); this
    flattens to (batch*NUM_CHOICES, seq_len) for the tokenizer's padding, then
    reshapes back so the model sees (batch, NUM_CHOICES, seq_len).
    """

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: list[dict]) -> dict:
        labels = [f.pop("label") for f in features]
        num_choices = len(features[0]["input_ids"])
        flattened = [
            {key: value[i] for key, value in f.items()} for f in features for i in range(num_choices)
        ]
        batch = self.tokenizer.pad(flattened, padding=True, return_tensors="pt")
        batch = {k: v.view(len(features), num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
