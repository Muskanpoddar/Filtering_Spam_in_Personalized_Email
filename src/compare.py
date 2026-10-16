"""Model comparison and reporting utilities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import Metrics, format_metrics
from .features.structural import extract_batch
from .models.baseline_nb import TfidfNBClassifier
from .models.distilbert import DistilBertClassifier
from .models.hybrid import HybridClassifier
from .models.personalized import PersonalizedSpamDetector
from .personalization.user_profile import UserProfile


MODEL_ORDER = ["baseline", "distilbert", "hybrid", "personalized"]
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "false_positive_rate", "auc"]


def evaluate_loaded_models(models_dir: Path, test_emails, test_labels,
                            batch_size: int = 64) -> dict:
    results: dict[str, Metrics] = {}
    struct = extract_batch(test_emails)

    nb_path = models_dir / "baseline_nb"
    if (nb_path / "nb.joblib").exists():
        nb = TfidfNBClassifier.load(str(nb_path))
        probs = nb.predict_proba(test_emails)
        from .evaluate import compute_metrics
        results["baseline"] = compute_metrics(test_labels, (probs >= 0.5).astype(int), probs)

    db_path = models_dir / "distilbert"
    if (db_path / "config.json").exists():
        db = DistilBertClassifier.load(str(db_path))
        probs = db.predict_proba(test_emails, batch_size=batch_size)
        from .evaluate import compute_metrics
        results["distilbert"] = compute_metrics(test_labels, (probs >= 0.5).astype(int), probs)

    hy_path = models_dir / "hybrid"
    if (hy_path / "hybrid.pt").exists():
        hy = HybridClassifier.load(str(hy_path))
        probs = hy.predict_proba(test_emails, struct, batch_size=batch_size)
        from .evaluate import compute_metrics
        results["hybrid"] = compute_metrics(test_labels, (probs >= 0.5).astype(int), probs)

    p_path = models_dir / "personalized"
    if (p_path / "hybrid.pt").exists():
        det = PersonalizedSpamDetector.load(str(p_path))
        probs = det.predict_proba(test_emails, struct)
        from .evaluate import compute_metrics
        results["personalized"] = compute_metrics(
            test_labels, (probs >= det.threshold).astype(int), probs, threshold=det.threshold
        )
    return results


def plot_comparison(results: dict, out_path: Path) -> None:
    names = [n for n in MODEL_ORDER if n in results]
    if not names:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    metric_set = ["accuracy", "precision", "recall", "f1", "false_positive_rate"]
    width = 0.15
    x = np.arange(len(metric_set))
    for i, n in enumerate(names):
        vals = [results[n].to_dict()[m] for m in metric_set]
        axes[0].bar(x + (i - len(names) / 2) * width, vals, width, label=n)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_set, rotation=20)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Model comparison across metrics")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    fprs = [results[n].false_positive_rate for n in names]
    recs = [results[n].recall for n in names]
    axes[1].scatter(fprs, recs, s=80)
    for i, n in enumerate(names):
        axes[1].annotate(n, (fprs[i], recs[i]), xytext=(5, 5), textcoords="offset points")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("Recall (Spam)")
    axes[1].set_title("Recall vs FPR (lower-left is better)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_report(results: dict, out_path: Path = Path("results/comparison_report.md")) -> None:
    lines = ["# Model Comparison Report\n"]
    for name in MODEL_ORDER:
        if name not in results:
            continue
        m = results[name]
        lines.append(f"## {name}\n")
        lines.append(format_metrics(name, m) + "\n")
        cm = m.confusion_matrix
        lines.append(f"\nConfusion matrix (rows=true, cols=pred):\n")
        lines.append("```\n")
        lines.append(f"        pred_ham  pred_spam\n")
        lines.append(f"true_ham   {cm[0][0]:6d}    {cm[0][1]:6d}\n")
        lines.append(f"true_spam  {cm[1][0]:6d}    {cm[1][1]:6d}\n")
        lines.append("```\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(results: dict, out_path: Path = Path("results/comparison_metrics.json")) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v.to_dict() for k, v in results.items()}
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--csv", type=Path, required=True,
                        help="Processed CSV with 'label' column")
    parser.add_argument("--report", type=Path, default=Path("results/comparison_report.md"))
    parser.add_argument("--plot", type=Path, default=Path("results/comparison_plot.png"))
    parser.add_argument("--json", type=Path, default=Path("results/comparison_metrics.json"))
    parser.add_argument("--max-eval", type=int, default=2000,
                        help="Cap on the number of evaluation emails (CPU-bound).")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    from .data_loader import load_csv
    emails, labels = load_csv(args.csv)
    import numpy as np
    labels = np.asarray(labels, dtype=np.int64)
    if args.max_eval > 0 and len(emails) > args.max_eval:
        rng = np.random.default_rng(0)
        idx = []
        for c in np.unique(labels):
            ci = np.where(labels == c)[0]
            idx.extend(rng.choice(ci, size=min(len(ci), args.max_eval // 2),
                                   replace=False).tolist())
        idx = sorted(idx)
        emails = [emails[i] for i in idx]
        labels = labels[idx]
        print(f"Subsampled evaluation to {len(emails)} emails")
    results = evaluate_loaded_models(args.models_dir, emails, labels,
                                      batch_size=args.batch_size)
    if not results:
        raise SystemExit("No trained models found in models-dir. Run training first.")
    print("\n=== Comparison ===")
    for name in MODEL_ORDER:
        if name in results:
            print(format_metrics(name, results[name]))
    write_report(results, args.report)
    write_json(results, args.json)
    plot_comparison(results, args.plot)
    print(f"\nReport:    {args.report}")
    print(f"Plot:      {args.plot}")
    print(f"JSON:      {args.json}")


if __name__ == "__main__":
    main()