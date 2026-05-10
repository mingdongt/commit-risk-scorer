"""Smoke-test fine-tune: HuggingFace PEFT (LoRA) on DistilBERT for code-defect classification.

This is the FAST iteration path — runs on a CPU laptop in under an hour. It exists to
prove the data + LoRA + eval pipeline end-to-end without needing a CUDA GPU. Production
target is `train_nemo.py` (NVIDIA NeMo + Mistral-7B-v0.3 + LoRA).

Why DistilBERT?
    - Small (~67M parameters) → trains on CPU
    - Apache 2.0 license, no gating
    - Same tokenizer family as many code-encoder models, so tokenization patterns transfer

Usage:
    python -m src.models.finetune.train_smoke --output-dir data/models/smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data.load_devign import load_devign
from src.eval.metrics import hf_compute_metrics

BASE_MODEL = "distilbert/distilbert-base-uncased"
MAX_LENGTH = 256  # truncate aggressively for CPU speed


def tokenize_dataset(ds: DatasetDict, tokenizer) -> DatasetDict:
    """Tokenize the `func` column; keep `target` as the label column."""

    def _tok(examples):
        out = tokenizer(
            examples["func"],
            padding=False,  # the data collator handles dynamic padding per batch
            truncation=True,
            max_length=MAX_LENGTH,
        )
        # Devign's `target` is stored as bool; cast to int so PyTorch builds a
        # LongTensor (required for CrossEntropyLoss), not a float tensor.
        out["labels"] = [int(t) for t in examples["target"]]
        return out

    keep = {"input_ids", "attention_mask", "labels"}
    cols_to_remove = [c for c in ds["train"].column_names if c not in keep]
    return ds.map(_tok, batched=True, remove_columns=cols_to_remove)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/models/smoke")
    parser.add_argument(
        "--subsample",
        type=int,
        default=500,
        help="Examples per split (the full Devign train set is ~21k; a CPU run on the full set is impractical)",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading Devign (subsample={args.subsample} per split)")
    ds = load_devign(subsample=args.subsample, seed=args.seed)
    print(f"      train={len(ds['train'])}  val={len(ds['validation'])}  test={len(ds['test'])}")

    print(f"[2/5] Loading tokenizer + base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        # Force CrossEntropyLoss path; default auto-detection picks BCE when
        # labels are bool-like, which mismatches our (batch,) target shape.
        problem_type="single_label_classification",
    )

    print(f"[3/5] Wrapping with LoRA adapter (rank={args.lora_rank}, alpha={args.lora_alpha})")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_lin", "v_lin"],  # DistilBERT attention projections
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[4/5] Tokenizing dataset")
    tokenized = tokenize_dataset(ds, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=20,
        report_to="none",
        seed=args.seed,
        use_cpu=True,  # explicit CPU (replaces deprecated no_cuda kwarg)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,  # replaces deprecated `tokenizer` kwarg
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=hf_compute_metrics,
    )

    print("[5/5] Training")
    trainer.train()

    print("\nFinal evaluation on the test split:")
    test_results = trainer.evaluate(eval_dataset=tokenized["test"])
    for k, v in test_results.items():
        print(f"  {k}: {v}")

    # Save metrics JSON for the README "Initial Results" section.
    metrics_path = output_dir / "smoke_metrics.json"
    metrics_payload = {
        "base_model": BASE_MODEL,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "subsample_per_split": args.subsample,
        "epochs": args.epochs,
        "test_f1": test_results.get("eval_f1"),
        "test_precision": test_results.get("eval_precision"),
        "test_recall": test_results.get("eval_recall"),
        "test_accuracy": test_results.get("eval_accuracy"),
        "test_auc_roc": test_results.get("eval_auc_roc"),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"\nSaved metrics → {metrics_path}")

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    print(f"Saved LoRA adapter → {adapter_dir}")


if __name__ == "__main__":
    main()
