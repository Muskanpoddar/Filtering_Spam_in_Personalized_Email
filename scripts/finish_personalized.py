"""Resume training: only run the personalized step using the already-trained hybrid."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, RESULTS_DIR
from src.data_loader import load_csv
from src.evaluate import compute_metrics
from src.features.structural import extract_batch
from src.models.hybrid import HybridClassifier
from src.models.personalized import PersonalizedSpamDetector
from src.personalization.user_profile import UserProfile

# Import the train module via package path to get its helpers.
from src.train import stratified_split, _bootstrap_user_profile  # noqa: E402


def main():
    csv_path = PROCESSED_DATA_DIR / "spamassassin.csv"
    emails, labels = load_csv(csv_path)
    labels = np.asarray(labels, dtype=np.int64)
    train_e, train_y, val_e, val_y, test_e, test_y = stratified_split(emails, labels)
    rng = np.random.default_rng(RANDOM_STATE)
    keep_idx = []
    for c in np.unique(train_y):
        cls_idx = np.where(train_y == c)[0]
        take = min(len(cls_idx), 1500 // 2)
        keep_idx.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
    keep_idx = sorted(keep_idx)
    train_e = [train_e[i] for i in keep_idx]
    train_y = train_y[keep_idx]
    print(f"Train: {len(train_e)} | Val: {len(val_e)} | Test: {len(test_e)}")

    print("Loading hybrid model...")
    hy = HybridClassifier.load(str(MODELS_DIR / "hybrid"))
    test_struct = extract_batch(test_e)

    print("Bootstrapping user profile...")
    profile = _bootstrap_user_profile(train_e, train_y)
    detector = PersonalizedSpamDetector(hybrid=hy, profile=profile)
    detector.set_strength(0.55)
    print("Scoring personalized detector...")
    probs = detector.predict_proba(test_e, test_struct)
    preds = (probs >= profile.threshold).astype(int)
    metrics = compute_metrics(test_y, preds, probs, threshold=profile.threshold)
    print("Personalized metrics:", metrics.to_dict())

    print("Saving personalized detector...")
    detector.save(str(MODELS_DIR / "personalized"))
    print("Saved.")

    train_metrics_path = RESULTS_DIR / "train_metrics.json"
    if train_metrics_path.exists():
        with train_metrics_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    else:
        payload = {}
    payload["personalized"] = metrics.to_dict()
    train_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with train_metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Updated {train_metrics_path}")


if __name__ == "__main__":
    main()