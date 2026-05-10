"""Load CodeXGLUE Devign defect-detection dataset for commit-risk-scorer.

Devign (Zhou et al., 2019) is a function-level defect detection dataset distilled
from real-world C projects (FFmpeg, QEMU). Each example is a complete C function
labeled defective (1) or not (0).

We use it as the cleaner, pre-train signal in the hybrid predictive pipeline; the
noisier GitHub PR/CI scrapes (see scrape_github_prs.py — coming in v0.2) provide
the target-task signal.

Reference:
    https://huggingface.co/datasets/code_x_glue_cc_defect_detection
"""
from __future__ import annotations

from datasets import DatasetDict, load_dataset

DEVIGN_HF_ID = "code_x_glue_cc_defect_detection"


def load_devign(subsample: int | None = None, seed: int = 42) -> DatasetDict:
    """Load Devign with three splits: train / validation / test.

    Args:
        subsample: If set, take this many shuffled examples from each split. Useful
            for CPU smoke tests; the full dataset has ~21k train examples and a
            full epoch on CPU is impractical for our DistilBERT-based smoke runner.
        seed: Shuffle seed (only matters when subsampling).

    Returns:
        DatasetDict with splits {train, validation, test}, each having columns
        `func` (str — full C function source) and `target` (int — 0 or 1).
    """
    ds = load_dataset(DEVIGN_HF_ID)

    if subsample is not None:
        ds = DatasetDict(
            {
                split: ds[split].shuffle(seed=seed).select(range(min(subsample, len(ds[split]))))
                for split in ds.keys()
            }
        )

    return ds


def label_distribution(ds: DatasetDict) -> dict[str, dict[str, int]]:
    """Inspect class balance across splits."""
    out: dict[str, dict[str, int]] = {}
    for split in ds.keys():
        labels = ds[split]["target"]
        out[split] = {
            "total": len(labels),
            "defective": int(sum(labels)),
            "clean": int(len(labels) - sum(labels)),
        }
    return out


if __name__ == "__main__":
    print(f"Loading Devign from HuggingFace ({DEVIGN_HF_ID})...")
    ds = load_devign(subsample=100)
    print(ds)
    print()
    print("First train example (truncated):")
    sample = ds["train"][0]
    print(f"  target: {sample['target']}")
    print(f"  func[:200]: {sample['func'][:200]!r}...")
    print()
    print("Label distribution:")
    for split, dist in label_distribution(ds).items():
        print(f"  {split}: {dist}")
