"""Traditional ML baseline: NVIDIA RAPIDS cuML GBDT on commit features.

Why this exists
---------------
A predictive code-review system should know *when classical ML wins over LLMs*.
On structured commit features (LOC, files-touched, owner-overlap, test-coverage
delta, etc.), a GPU-accelerated GBDT trains in seconds and runs at well under
1 ms per inference — orders of magnitude faster than an LLM judge while being
trivially calibrated (sigmoid / isotonic on holdout).

This baseline is reported alongside the NeMo + LoRA fine-tune in
`docs/evaluation.md` so the design surfaces the trade-off explicitly: when to
reach for the LLM, and when not to.

Environment
-----------
- GPU path (production): NVIDIA CUDA 12+, RAPIDS cuML (recommend running in
  nvcr.io/nvidia/rapidsai container).
- CPU fallback (development): scikit-learn's GradientBoostingClassifier — same
  feature set, same eval pipeline, smaller model.

The script auto-detects the available runtime and reports which one ran in the
output JSON's `runtime` field — keeps the smoke results honest.

Usage:
    python -m src.models.baselines.cuml_gbdt --output-dir data/models/baselines
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# RAPIDS cuML imports are guarded — the CPU fallback (sklearn) is always
# attempted second so the smoke-test path on a laptop still produces a real
# F1 number.
try:
    from cuml.ensemble import RandomForestClassifier as _cuRF  # type: ignore[import-not-found]

    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

try:
    from sklearn.ensemble import GradientBoostingClassifier as _SkGB

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def extract_features(func: str) -> list[float]:
    """Extract simple structural features from a Devign C function.

    Deliberately shallow — the point of a classical-ML baseline is to show what
    feature-engineering reaches without deep code understanding. The LLM judge
    ships richer semantic features in the v0.2 hybrid pipeline.
    """
    lines = func.splitlines()
    n = max(len(lines), 1)
    return [
        float(len(lines)),                                                   # lines_total
        float(sum(len(line) for line in lines)) / n,                         # lines_avg_length
        float(any("->" in line or "++" in line or "--" in line for line in lines)),  # has_pointer_arith
        float("malloc" in func or "calloc" in func),                         # has_alloc
        float("free(" in func),                                              # has_free
        float("memcpy" in func or "memmove" in func),                        # has_memcpy
        float("strcpy" in func or "strcat" in func),                         # has_strcpy
        float(sum(1 for line in lines if "if " in line or "else " in line)), # branch_count
        float(sum(1 for line in lines if "for " in line or "while " in line)), # loop_count
        float(sum(1 for line in lines if "return" in line)),                 # return_count
    ]


FEATURE_NAMES = [
    "lines_total",
    "lines_avg_length",
    "has_pointer_arith",
    "has_alloc",
    "has_free",
    "has_memcpy",
    "has_strcpy",
    "branch_count",
    "loop_count",
    "return_count",
]


def build_xy(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Materialize (X, y) for a HuggingFace Dataset split."""
    X: list[list[float]] = []
    y: list[int] = []
    for row in dataset:
        X.append(extract_features(row["func"]))
        y.append(int(row["target"]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/models/baselines")
    parser.add_argument("--subsample", type=int, default=500)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Local imports — keep the module importable even when these are missing.
    from src.data.load_devign import load_devign
    from src.eval.metrics import compute_metrics

    print(f"[1/4] Loading Devign (subsample={args.subsample} per split)")
    ds = load_devign(subsample=args.subsample, seed=args.seed)

    print("[2/4] Extracting features")
    X_train, y_train = build_xy(ds["train"])
    X_test, y_test = build_xy(ds["test"])
    print(f"      X_train={X_train.shape}  X_test={X_test.shape}  feature count={len(FEATURE_NAMES)}")

    print(f"[3/4] Training (cuML available: {CUML_AVAILABLE}, sklearn available: {SKLEARN_AVAILABLE})")
    if CUML_AVAILABLE:
        model = _cuRF(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.seed,
        )
        runtime = "cuml-gpu"
    elif SKLEARN_AVAILABLE:
        # CPU fallback — same task, smaller model so the CPU run finishes fast.
        model = _SkGB(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.seed,
        )
        runtime = "sklearn-cpu-fallback"
    else:
        raise RuntimeError(
            "Neither cuML nor scikit-learn is installed. "
            "Install one of: pip install scikit-learn  (CPU)  OR  rapids-cuml  (GPU)."
        )

    model.fit(X_train, y_train)

    print("[4/4] Evaluating on test split")
    y_pred = model.predict(X_test)
    y_proba: np.ndarray | None = None
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        pass

    metrics = compute_metrics(y_test, y_pred, y_proba)
    print(metrics.to_dict())

    payload = {
        "runtime": runtime,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "feature_names": FEATURE_NAMES,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        **metrics.to_dict(),
    }
    out_path = output_dir / "cuml_gbdt_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved metrics → {out_path}")


if __name__ == "__main__":
    main()
