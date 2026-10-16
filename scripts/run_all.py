"""Run the full pipeline: download -> process -> train -> compare."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path = PROJECT_ROOT) -> None:
    print("\n>>>", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(cwd))
    if res.returncode != 0:
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--epochs-distilbert", type=int, default=1)
    parser.add_argument("--epochs-hybrid", type=int, default=1)
    parser.add_argument("--primary-csv", type=Path,
                        default=PROJECT_ROOT / "data" / "processed" / "spamassassin.csv")
    parser.add_argument("--validation-csv", type=Path,
                        default=PROJECT_ROOT / "data" / "processed" / "enron_spam.csv")
    args = parser.parse_args()

    py = sys.executable
    if not args.skip_download:
        run([py, "scripts/build_datasets.py", "--dataset", "all"])

    if not args.primary_csv.exists():
        raise SystemExit(f"Primary dataset missing: {args.primary_csv}")

    run([py, "-m", "src.train",
         "--csv", str(args.primary_csv),
         "--results", str(PROJECT_ROOT / "results" / "train_metrics.json"),
         "--epochs-distilbert", str(args.epochs_distilbert),
         "--epochs-hybrid", str(args.epochs_hybrid)])

    run([py, "-m", "src.compare",
         "--csv", str(args.primary_csv),
         "--models-dir", str(PROJECT_ROOT / "models"),
         "--report", str(PROJECT_ROOT / "results" / "comparison_report.md"),
         "--plot", str(PROJECT_ROOT / "results" / "comparison_plot.png"),
         "--json", str(PROJECT_ROOT / "results" / "comparison_metrics.json")])

    if args.validation_csv.exists():
        run([py, "-m", "src.compare",
             "--csv", str(args.validation_csv),
             "--models-dir", str(PROJECT_ROOT / "models"),
             "--report", str(PROJECT_ROOT / "results" / "enron_validation_report.md"),
             "--plot", str(PROJECT_ROOT / "results" / "enron_validation_plot.png"),
             "--json", str(PROJECT_ROOT / "results" / "enron_validation_metrics.json")])
    else:
        print(f"Validation dataset not found at {args.validation_csv}, skipping Enron validation.")

    print("\nAll steps complete.")


if __name__ == "__main__":
    main()