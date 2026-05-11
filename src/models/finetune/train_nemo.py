"""Production fine-tune: NVIDIA NeMo + LoRA on Mistral-7B-v0.3 for commit-risk classification.

This is the production-target script referenced in `docs/design-doc.md` §Architecture.
For local CPU iteration, use `train_smoke.py` instead (DistilBERT + HF PEFT).

Environment requirements:
    - CUDA-capable GPU (≥ 24 GB VRAM recommended for Mistral-7B with LoRA + bf16)
    - NVIDIA NeMo Toolkit (>= 2.0)
    - Recommended: run inside the official NeMo container
        docker pull nvcr.io/nvidia/nemo:24.07
        docker run --gpus all -it -v $PWD:/workspace nvcr.io/nvidia/nemo:24.07
    - HuggingFace token with access to mistralai/Mistral-7B-v0.3 (Apache 2.0 base model)

Status:
    The Mistral-7B base model first needs conversion to NeMo's checkpoint format. NeMo
    ships scripts for this in scripts/nlp_language_modeling/convert_hf_checkpoint_to_nemo.py.
    Run that once to produce mistral-7b-v0.3.nemo, then this script consumes it.

Usage (in NeMo container):
    python -m src.models.finetune.train_nemo \\
        --base-checkpoint mistral-7b-v0.3.nemo \\
        --output-dir data/models/production \\
        --epochs 3
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# NeMo imports are guarded so this file remains importable without NeMo installed.
# The actual training run requires NEMO_AVAILABLE=True, which is only true in a
# CUDA-enabled NeMo container.
try:
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import (
        MegatronGPTSFTModel,
    )
    from nemo.collections.nlp.parts.peft_config import LoraPEFTConfig
    from nemo.utils.exp_manager import exp_manager
    from omegaconf import OmegaConf
    from pytorch_lightning import Trainer

    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False


def _require_nemo() -> None:
    if not NEMO_AVAILABLE:
        raise RuntimeError(
            "NVIDIA NeMo is not installed in this Python environment. "
            "This script must run inside an NVIDIA NeMo container "
            "(recommended: nvcr.io/nvidia/nemo:24.07 or newer). "
            "For local CPU iteration, see src/models/finetune/train_smoke.py."
        )


def build_lora_config(rank: int = 16, alpha: int = 32, dropout: float = 0.05):
    """LoRA PEFT config for Mistral-7B's attention projections.

    Mistral's transformer blocks expose `attention_qkv` (the fused Q/K/V proj) and
    `attention_dense` (the output proj). These are the canonical LoRA targets in
    NeMo's PEFT recipes for Llama-class architectures.
    """
    _require_nemo()
    return LoraPEFTConfig(
        target_modules=["attention_qkv", "attention_dense"],
        adapter_dim=rank,
        adapter_dropout=dropout,
        alpha=alpha,
    )


def build_trainer_config(args: argparse.Namespace):
    """Construct the NeMo Trainer + experiment-manager config.

    Mirrors the structure of NeMo's example PEFT recipe at
    examples/nlp/language_modeling/megatron_gpt_finetuning.py.
    """
    _require_nemo()
    return OmegaConf.create(
        {
            "trainer": {
                "max_epochs": args.epochs,
                "accelerator": "gpu",
                "devices": args.devices,
                "precision": "bf16-mixed",
                "log_every_n_steps": 20,
                "val_check_interval": 1.0,
                "gradient_clip_val": 1.0,
            },
            "model": {
                "restore_from_path": args.base_checkpoint,
                "micro_batch_size": args.micro_batch_size,
                "global_batch_size": args.global_batch_size,
                "optim": {
                    "name": "fused_adam",
                    "lr": args.lr,
                    "weight_decay": 0.01,
                    "sched": {
                        "name": "CosineAnnealing",
                        "warmup_steps": 100,
                        "constant_steps": 0,
                        "min_lr": args.lr / 10,
                    },
                },
                "data": {
                    "train_ds": {
                        "file_names": [args.train_jsonl],
                        "max_seq_length": args.max_seq_length,
                        "shuffle": True,
                    },
                    "validation_ds": {
                        "file_names": [args.val_jsonl],
                        "max_seq_length": args.max_seq_length,
                    },
                },
            },
            "exp_manager": {
                "exp_dir": args.output_dir,
                "name": "commit_risk_lora",
                "create_checkpoint_callback": True,
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=str, default="checkpoints/mistral-7b-v0.3.nemo")
    parser.add_argument(
        "--train-jsonl",
        type=str,
        default="data/processed/devign_train.jsonl",
        help="JSONL with {input, output} fields — see scripts/format_for_nemo.py",
    )
    parser.add_argument("--val-jsonl", type=str, default="data/processed/devign_val.jsonl")
    parser.add_argument("--output-dir", type=str, default="data/models/production")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    _require_nemo()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_trainer_config(args)
    trainer = Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.exp_manager)

    print(f"[1/4] Restoring base model from {args.base_checkpoint}")
    model = MegatronGPTSFTModel.restore_from(restore_path=args.base_checkpoint, trainer=trainer)

    print(f"[2/4] Adding LoRA adapter (rank={args.lora_rank}, alpha={args.lora_alpha})")
    peft_cfg = build_lora_config(rank=args.lora_rank, alpha=args.lora_alpha)
    model.add_adapter(peft_cfg)

    print("[3/4] Training")
    trainer.fit(model)

    print("[4/4] Saving adapter")
    adapter_path = output_dir / "mistral7b_v03_commit_risk_lora.nemo"
    model.save_to(str(adapter_path))
    print(f"\nProduction NeMo + Mistral-7B-v0.3 LoRA fine-tune complete.")
    print(f"Adapter:  {adapter_path}")
    # Next step in the production path: compile the merged base+adapter weights
    # to a TensorRT-LLM engine for lower inference latency on Triton.
    #     trtllm-build --checkpoint_dir <merged_weights> \
    #                  --output_dir <trt_engine_dir> \
    #                  --gemm_plugin float16 \
    #                  --max_input_len 4096 --max_output_len 1024
    # The resulting engine is served via the Triton TRT-LLM backend; see
    # docs/design-doc.md §Architecture for the full path. — TODO(v0.2)
    print(f"Next:    compile to TensorRT-LLM engine (trtllm-build) -> serve via Triton TRT-LLM backend")


if __name__ == "__main__":
    main()
