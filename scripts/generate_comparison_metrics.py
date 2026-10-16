"""Generate the per-dataset metrics JSON the Streamlit app reads.

Reuses the trained models in `models/` and the existing evaluation helpers
(`src.compare.evaluate_loaded_models`, `src.evaluate.compute_metrics`).

Output format (matches `app.py:load_metrics`):

    {
      "<dataset_name>": {
        "baseline":     {accuracy, precision, recall, f1,
                         false_positive_rate, auc, confusion_matrix, threshold},
        "distilbert":   {...},
        "hybrid":       {...},
        "personalized": {...},
      },
      ...
    }

Run once per dataset:

    python scripts/generate_comparison_metrics.py \
        --csv data/processed/spamassassin.csv \
        --dataset spamassassin \
        --out results/comparison_metrics.json

    python scripts/generate_comparison_metrics.py \
        --csv data/processed/enron_spam.csv \
        --dataset enron_spam \
        --out results/enron_validation_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.compare import evaluate_loaded_models
from src.config import MODELS_DIR, RESULTS_DIR
from src.data_loader import load_csv


def _stratified_subsample(emails: list, labels: np.ndarray,
                          per_class_max: int, seed: int = 0):
    """Per-class stratified cap, mirroring `src.compare.main`."""
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for c in np.unique(labels):
        cls_idx = np.where(labels == c)[0]
        take = min(len(cls_idx), per_class_max)
        keep.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
    keep = sorted(keep)
    return [emails[i] for i in keep], labels[keep]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, required=True,
                        help="Processed CSV with a 'label' column.")
    parser.add_argument("--dataset", required=True,
                        help="Top-level key for the output JSON "
                             "(e.g. 'spamassassin', 'enron_spam').")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output JSON file path.")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                        help="Directory containing baseline_nb/, distilbert/, "
                             "hybrid/, personalized/.")
    parser.add_argument("--max-eval", type=int, default=1500,
                        help="Per-class cap on evaluation emails (CPU bound).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    print(f"[{args.dataset}] loading {args.csv}…")
    emails, labels = load_csv(args.csv)
    labels = np.asarray(labels, dtype=np.int64)
    print(f"[{args.dataset}] {len(emails)} emails, "
          f"ham={int((labels == 0).sum())} spam={int((labels == 1).sum())}")

    if args.max_eval > 0 and len(emails) > args.max_eval:
        emails, labels = _stratified_subsample(
            emails, labels, per_class_max=args.max_eval // 2, seed=args.seed
        )
        print(f"[{args.dataset}] subsampled to {len(emails)} emails")

    print(f"[{args.dataset}] scoring all 4 models…")
    results = evaluate_loaded_models(
        args.models_dir, emails, labels, batch_size=args.batch_size,
    )

    if not results:
        raise SystemExit("No trained models found in models-dir. Run training first.")

    for name in ["baseline", "distilbert", "hybrid", "personalized"]:
        if name in results:
            m = results[name].to_dict()
            print(
                f"  {name:<13s} acc={m['accuracy']:.4f} "
                f"prec={m['precision']:.4f} rec={m['recall']:.4f} "
                f"f1={m['f1']:.4f} FPR={m['false_positive_rate']:.4f} "
                f"AUC={m['auc']:.4f}"
            )

    payload = {args.dataset: {k: v.to_dict() for k, v in results.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[{args.dataset}] wrote {args.out}")


if __name__ == "__main__":
    main()