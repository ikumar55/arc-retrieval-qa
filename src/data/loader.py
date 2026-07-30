"""Loaders for the ARC Easy/Challenge question sets and the ARC Corpus.

Expects data/raw/ARC-V1-Feb2018-2/ to exist (see scripts/download_data.py).
"""
import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# ~Half the full 14.6M-sentence ARC Corpus. AI2's README states the corpus is
# already randomly shuffled, so a prefix is a valid random subsample -- this
# just trades some recall for a ~90min (vs ~180min) embedding build. All of
# BM25/dense/hybrid default to this same subset so they're compared over the
# same corpus (a fair comparison requires that -- see PROJECT_BRIEF.md).
CORPUS_SUBSET_SIZE = 7_300_000


@dataclass
class ArcQuestion:
    id: str
    question: str
    choices: list[str]
    choice_labels: list[str]
    answer_key: str
    split: str  # "easy" or "challenge"
    subset: str  # "Train", "Dev", or "Test"


def _find_release_dir(raw_dir: Path) -> Path:
    candidates = [p for p in raw_dir.glob("ARC-V1-Feb2018*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"No ARC-V1-Feb2018* directory found under {raw_dir}. "
            "Run scripts/download_data.py first."
        )
    return candidates[0]


def load_questions(split: str, subset: str, raw_dir: Path = DEFAULT_RAW_DIR) -> list[ArcQuestion]:
    """Load one subset (Train/Dev/Test) of one split (easy/challenge) from its .jsonl file."""
    assert split in ("easy", "challenge")
    assert subset in ("Train", "Dev", "Test")

    release_dir = _find_release_dir(raw_dir)
    folder_name = "ARC-Easy" if split == "easy" else "ARC-Challenge"
    jsonl_path = release_dir / folder_name / f"{folder_name}-{subset}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Expected {jsonl_path} — check the extracted folder layout.")

    questions = []
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            q = row["question"]
            questions.append(
                ArcQuestion(
                    id=row["id"],
                    question=q["stem"],
                    choices=[c["text"] for c in q["choices"]],
                    choice_labels=[c["label"] for c in q["choices"]],
                    answer_key=row["answerKey"],
                    split=split,
                    subset=subset,
                )
            )
    return questions


def load_all_questions(raw_dir: Path = DEFAULT_RAW_DIR) -> dict[str, list[ArcQuestion]]:
    """Load every split/subset combination into a dict keyed like 'easy_train'."""
    out = {}
    for split in ("easy", "challenge"):
        for subset in ("Train", "Dev", "Test"):
            out[f"{split}_{subset.lower()}"] = load_questions(split, subset, raw_dir)
    return out


def iter_corpus(raw_dir: Path = DEFAULT_RAW_DIR) -> Iterator[str]:
    """Yield sentences from the ARC Corpus one at a time (it's ~14M lines, don't load it all into memory)."""
    release_dir = _find_release_dir(raw_dir)
    corpus_path = release_dir / "ARC_Corpus.txt"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Expected {corpus_path} — check the extracted folder layout.")
    with open(corpus_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def count_corpus_lines(raw_dir: Path = DEFAULT_RAW_DIR) -> int:
    return sum(1 for _ in iter_corpus(raw_dir))


def load_corpus_subset(raw_dir: Path = DEFAULT_RAW_DIR, n: int = CORPUS_SUBSET_SIZE) -> list[str]:
    """The shared corpus subset used consistently by BM25/dense/hybrid. See
    CORPUS_SUBSET_SIZE above for why this isn't just the full corpus."""
    return list(itertools.islice(iter_corpus(raw_dir), n))
