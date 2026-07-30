"""Fine-tune the closed-book multiple-choice reader on combined ARC-Easy + ARC-Challenge train data.

This establishes the baseline reader (no retrieval) that will later be reused,
unchanged, for the BM25/dense/hybrid retrieval-augmented conditions.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformers import Trainer, TrainerCallback, TrainingArguments

from src.data.loader import load_questions
from src.eval.qa_metrics import accuracy_report
from src.reader.model import (
    DataCollatorForMultipleChoice,
    build_dataset,
    filter_to_four_choice,
    load_model_and_tokenizer,
)

IDX_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == labels).mean()}


class ProgressCallback(TrainerCallback):
    """Prints a clean, periodic status line (epoch, elapsed, ETA) instead of relying
    on tqdm bars, which are unreadable when a run is piped to a log file (e.g. on
    a remote/detached DataHub session)."""

    def __init__(self, log_every_steps: int = 25):
        self.log_every_steps = log_every_steps
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"[progress] training started: {state.max_steps} total steps over {args.num_train_epochs} epochs", flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.log_every_steps != 0:
            return
        elapsed = time.time() - self.start_time
        frac = state.global_step / state.max_steps
        eta_min = (elapsed / frac - elapsed) / 60 if frac > 0 else float("nan")
        print(
            f"[progress] step {state.global_step}/{state.max_steps} "
            f"epoch {state.epoch:.2f}/{args.num_train_epochs} "
            f"elapsed {elapsed / 60:.1f}m eta {eta_min:.1f}m",
            flush=True,
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        print(f"[progress] eval @ epoch {state.epoch:.2f}: {metrics}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reader_closed_book.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_questions = filter_to_four_choice(
        load_questions("easy", "Train") + load_questions("challenge", "Train")
    )
    val_questions = filter_to_four_choice(
        load_questions("easy", "Dev") + load_questions("challenge", "Dev")
    )
    easy_test = filter_to_four_choice(load_questions("easy", "Test"))
    challenge_test = filter_to_four_choice(load_questions("challenge", "Test"))
    test_questions = easy_test + challenge_test

    print(f"train={len(train_questions)} val={len(val_questions)} test={len(test_questions)}")

    tokenizer, model = load_model_and_tokenizer(cfg["model_name"])

    train_ds = build_dataset(train_questions, tokenizer, max_length=cfg["max_length"])
    val_ds = build_dataset(val_questions, tokenizer, max_length=cfg["max_length"])
    test_ds = build_dataset(test_questions, tokenizer, max_length=cfg["max_length"])

    collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["epochs"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=20,
        report_to=[],
        seed=cfg.get("seed", 42),
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[ProgressCallback(log_every_steps=cfg.get("log_every_steps", 25))],
    )

    trainer.train()

    print("Running final predictions on held-out test sets (Easy + Challenge)...")
    test_output = trainer.predict(test_ds)
    pred_indices = np.argmax(test_output.predictions, axis=1)
    pred_labels = [IDX_TO_LABEL[i] for i in pred_indices]

    report = accuracy_report(test_questions, pred_labels)
    print("Closed-book reader test accuracy:", report)

    results_path = Path(cfg["output_dir"]) / "closed_book_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved results to {results_path}")

    final_model_dir = Path(cfg["output_dir"]) / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    print(f"Saved model to {final_model_dir}")


if __name__ == "__main__":
    main()
