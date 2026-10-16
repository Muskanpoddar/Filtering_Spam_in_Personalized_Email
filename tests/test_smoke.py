"""End-to-end smoke test using a tiny synthetic dataset (no internet needed)."""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR
from src.data_loader import write_csv
from src.evaluate import compute_metrics
from src.features.structural import extract_batch
from src.models.baseline_nb import TfidfNBClassifier
from src.models.hybrid import HybridClassifier
from src.models.personalized import PersonalizedSpamDetector
from src.personalization.user_profile import UserProfile
from src.preprocessor import ParsedEmail
import numpy as np


def make_synthetic():
    emails, labels = [], []
    samples = [
        ("WIN A FREE iPHONE!!!", "lottery@cheap.click",
         "Click http://cheap.click/win to claim your prize. FREE!!!", 1),
        ("URGENT verify your account", "security@bankk.tk",
         "Verify http://bankk.tk/login?id=1 immediately or be deactivated.", 1),
        ("Cheap meds online", "pharma@cheap.tk",
         "Viagra pills cheap free shipping. Visit http://cheap.tk/buy", 1),
        ("Lunch tomorrow?", "alex@friend.com",
         "Want to grab lunch tomorrow at the new Thai place?", 0),
        ("Project update", "manager@company.com",
         "Attached are the latest project slides for review.", 0),
        ("Re: invoice", "finance@partner.com",
         "Please find the invoice attached for your records.", 0),
    ]
    for i in range(40):
        subj, sender, body, lab = samples[i % len(samples)]
        e = ParsedEmail()
        e.subject = subj
        e.sender = sender
        e.body = body + f" reference={i}"
        e.raw_body = body + f" reference={i}"
        e.word_count = len(e.body.split())
        e.char_count = len(e.raw_body)
        e.urls = []
        emails.append(e)
        labels.append(lab)
    return emails, np.asarray(labels, dtype=np.int64)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="spam_smoke_"))
    csv_path = tmp / "tiny.csv"
    print(f"Working directory: {tmp}")
    emails, labels = make_synthetic()
    write_csv(list(zip(emails, labels)), csv_path)

    nb = TfidfNBClassifier.build()
    nb.fit(emails, labels)
    probs = nb.predict_proba(emails)
    m = compute_metrics(labels, (probs >= 0.5).astype(int), probs)
    print("TF-IDF+NB:", m.accuracy, m.f1, m.false_positive_rate)

    struct = extract_batch(emails)
    hy = HybridClassifier(num_epochs=1, batch_size=4)
    hy.fit(emails, struct, labels, verbose=False)
    probs = hy.predict_proba(emails, struct)
    m = compute_metrics(labels, (probs >= 0.5).astype(int), probs)
    print("Hybrid:", m.accuracy, m.f1, m.false_positive_rate)

    prof = UserProfile()
    prof.record("lottery@cheap.click", 1)
    prof.record("manager@company.com", 0)
    det = PersonalizedSpamDetector(hybrid=hy, profile=prof)
    probs = det.predict_proba(emails, struct)
    m = compute_metrics(labels, (probs >= prof.threshold).astype(int), probs)
    print("Personalized:", m.accuracy, m.f1, m.false_positive_rate)

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()