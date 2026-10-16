"""Build processed CSV datasets from SpamAssassin and Enron-Spam archives."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR  # noqa: E402
from src.data_loader import (  # noqa: E402
    extract_archive,
    load_enron_spam,
    load_spamassassin,
    write_csv,
)

# SpamAssassin corpus pieces (Apache mirrors / community archives).
SPAMASSASSIN_FILES = [
    ("https://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2",
     "20021010_spam.tar.bz2"),
    ("https://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2",
     "20021010_easy_ham.tar.bz2"),
    ("https://spamassassin.apache.org/old/publiccorpus/20021010_hard_ham.tar.bz2",
     "20021010_hard_ham.tar.bz2"),
    ("https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
     "20030228_spam.tar.bz2"),
    ("https://spamassassin.apache.org/old/publiccorpus/20030228_spam_2.tar.bz2",
     "20030228_spam_2.tar.bz2"),
    ("https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
     "20030228_easy_ham.tar.bz2"),
    ("https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
     "20030228_easy_ham_2.tar.bz2"),
    ("https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
     "20030228_hard_ham.tar.bz2"),
    ("https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2",
     "20050311_spam_2.tar.bz2"),
]

ENRON_SPAM_URL = "http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/enron_spam.tar.gz"


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with dest.open("wb") as fh, tqdm(total=total, unit="B", unit_scale=True,
                                          desc=dest.name) as pbar:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    fh.write(chunk)
                    pbar.update(len(chunk))
    return dest


def build_spamassassin(force_download: bool = False) -> Path:
    raw_dir = RAW_DATA_DIR / "spamassassin"
    extract_root = raw_dir / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)

    for url, name in SPAMASSASSIN_FILES:
        archive = raw_dir / name
        if force_download and archive.exists():
            archive.unlink()
        try:
            _download(url, archive)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! failed {url}: {exc}")
            continue
        target = extract_root / archive.stem
        if not target.exists():
            try:
                with tarfile.open(archive, "r:*") as tf:
                    tf.extractall(extract_root)
            except (tarfile.ReadError, EOFError) as exc:
                print(f"  ! could not extract {archive}: {exc}")

    csv_path = PROCESSED_DATA_DIR / "spamassassin.csv"
    rows = load_spamassassin(extract_root)
    if not rows:
        raise SystemExit("No SpamAssassin emails loaded. Check downloads.")
    write_csv(rows, csv_path)
    print(f"Wrote {len(rows)} SpamAssassin emails to {csv_path}")
    return csv_path


def build_enron_spam(force_download: bool = False) -> Path:
    raw_dir = RAW_DATA_DIR / "enron_spam"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_in = raw_dir / "enron_spam_data.csv"
    csv_url = "https://huggingface.co/datasets/SetFit/enron_spam/resolve/main/enron_spam_data.csv"

    if not csv_in.exists() or (force_download and csv_in.stat().st_size < 1_000_000):
        try:
            print(f"Downloading {csv_url} -> {csv_in}")
            with requests.get(csv_url, timeout=60, stream=True) as r:
                r.raise_for_status()
                with csv_in.open("wb") as fh:
                    for chunk in r.iter_content(64 * 1024):
                        if chunk:
                            fh.write(chunk)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! Enron-Spam CSV download failed: {exc}")
            raise

    from src.data_loader import load_enron_spam_csv  # local import to avoid cycles
    csv_path = PROCESSED_DATA_DIR / "enron_spam.csv"
    rows = load_enron_spam_csv(csv_in)
    if not rows:
        raise SystemExit("No Enron-Spam emails loaded.")
    write_csv(rows, csv_path)
    print(f"Wrote {len(rows)} Enron-Spam emails to {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["spamassassin", "enron", "all"],
                        default="all")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    if args.dataset in ("spamassassin", "all"):
        build_spamassassin(args.force_download)
    if args.dataset in ("enron", "all"):
        try:
            build_enron_spam(args.force_download)
        except Exception as exc:  # noqa: BLE001
            print(f"Enron-Spam step skipped: {exc}")


if __name__ == "__main__":
    main()