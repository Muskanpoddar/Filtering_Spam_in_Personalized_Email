"""End-to-end training pipeline for all four models."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .config import (
    ARTIFACTS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)
from .data_loader import load_csv
from .evaluate import compute_metrics
from .features.structural import extract_batch
from .models.baseline_nb import TfidfNBClassifier
from .models.distilbert import DistilBertClassifier
from .models.hybrid import HybridClassifier
from .models.personalized import PersonalizedSpamDetector
from .personalization.user_profile import UserProfile
from .preprocessor import ParsedEmail


def stratified_split(emails: list[ParsedEmail], labels: np.ndarray,
                     val_ratio: float = 0.1, test_ratio: float = 0.1,
                     seed: int = RANDOM_STATE):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(emails))
    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(labels):
        c_idx = idx[labels == c]
        rng.shuffle(c_idx)
        n = len(c_idx)
        n_test = max(1, int(round(test_ratio * n)))
        n_val = max(1, int(round(val_ratio * n)))
        test_idx.extend(c_idx[:n_test].tolist())
        val_idx.extend(c_idx[n_test:n_test + n_val].tolist())
        train_idx.extend(c_idx[n_test + n_val:].tolist())
    return (
        [emails[i] for i in train_idx], labels[train_idx],
        [emails[i] for i in val_idx], labels[val_idx],
        [emails[i] for i in test_idx], labels[test_idx],
    )


@dataclass
class TrainResult:
    name: str
    metrics: dict


def train_baseline(train_emails, train_labels, test_emails, test_labels):
    model = TfidfNBClassifier.build()
    model.fit(train_emails, train_labels)
    probs = model.predict_proba(test_emails)
    preds = (probs >= 0.5).astype(int)
    from .evaluate import compute_metrics
    m = compute_metrics(test_labels, preds, probs)
    model.save(str(MODELS_DIR / "baseline_nb"))
    return m


def _bootstrap_user_profile(emails: list[ParsedEmail], labels: np.ndarray,
                            sample_n: int = 200, seed: int = RANDOM_STATE) -> UserProfile:
    rng = np.random.default_rng(seed)
    profile = UserProfile()
    n = min(sample_n, len(emails))
    sel = rng.choice(len(emails), size=n, replace=False)
    for i in sel:
        e = emails[int(i)]
        profile.record(e.sender or "", int(labels[int(i)]), confidence=0.7)
    return profile


def run_training(csv_path: Path, num_epochs_distilbert: int = 1,
                 num_epochs_hybrid: int = 1, results_path: Optional[Path] = None,
                 max_train: int = 2000, batch_size: int = 16,
                 max_length: int = 256) -> dict:
    emails, labels = load_csv(csv_path)
    labels = np.asarray(labels, dtype=np.int64)
    train_e, train_y, val_e, val_y, test_e, test_y = stratified_split(emails, labels)
    if max_train > 0 and len(train_e) > max_train:
        rng = np.random.default_rng(RANDOM_STATE)
        keep_idx = []
        for c in np.unique(train_y):
            cls_idx = np.where(train_y == c)[0]
            take = min(len(cls_idx), max_train // 2)
            keep_idx.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
        keep_idx = np.asarray(sorted(keep_idx))
        train_e = [train_e[i] for i in keep_idx]
        train_y = train_y[keep_idx]
        print(f"Subsampled train to {len(train_e)} for CPU-bound models")
    print(f"Train: {len(train_e)} | Val: {len(val_e)} | Test: {len(test_e)}")
    results = {}
    print("\n[1/4] Training TF-IDF + Naive Bayes baseline...")
    results["baseline"] = train_baseline(train_e, train_y, test_e, test_y).to_dict()
    print("\n[2/4] Training DistilBERT...")
    from .models.distilbert import DistilBertClassifier
    db = DistilBertClassifier(num_epochs=num_epochs_distilbert,
                              batch_size=batch_size, max_length=max_length)
    db.fit(train_e, train_y, val_e, val_y)
    probs = db.predict_proba(test_e)
    preds = (probs >= 0.5).astype(int)
    from .evaluate import compute_metrics
    results["distilbert"] = compute_metrics(test_y, preds, probs).to_dict()
    db.save(str(MODELS_DIR / "distilbert"))
    print("\n[3/4] Training DistilBERT + structural features (hybrid)...")
    from .models.hybrid import HybridClassifier
    # Hybrid uses a smaller max_length to keep CPU training feasible.
    hy = HybridClassifier(num_epochs=num_epochs_hybrid,
                          batch_size=batch_size,
                          max_length=min(max_length, 192))
    train_struct = extract_batch(train_e)
    val_struct = extract_batch(val_e)
    test_struct = extract_batch(test_e)
    hy.fit(train_e, train_struct, train_y, val_e, val_struct, val_y)
    probs = hy.predict_proba(test_e, test_struct)
    preds = (probs >= 0.5).astype(int)
    results["hybrid"] = compute_metrics(test_y, preds, probs).to_dict()
    hy.save(str(MODELS_DIR / "hybrid"))
    print("\n[4/4] Evaluating personalized model on top of hybrid...")
    profile = _bootstrap_user_profile(train_e, train_y)
    detector = PersonalizedSpamDetector(hybrid=hy, profile=profile)
    detector.set_strength(0.55)
    probs = detector.predict_proba(test_e, test_struct)
    preds = (probs >= profile.threshold).astype(int)
    results["personalized"] = compute_metrics(test_y, preds, probs,
                                              threshold=profile.threshold).to_dict()
    detector.save(str(MODELS_DIR / "personalized"))
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=PROCESSED_DATA_DIR / "spamassassin.csv")
    parser.add_argument("--results", type=Path, default=Path("results/train_metrics.json"))
    parser.add_argument("--epochs-distilbert", type=int, default=1)
    parser.add_argument("--epochs-hybrid", type=int, default=1)
    parser.add_argument("--max-train", type=int, default=2000,
                        help="Cap on the number of training examples (deep models are CPU-bound).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    run_training(
        args.csv,
        num_epochs_distilbert=args.epochs_distilbert,
        num_epochs_hybrid=args.epochs_hybrid,
        results_path=args.results,
        max_train=args.max_train,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()