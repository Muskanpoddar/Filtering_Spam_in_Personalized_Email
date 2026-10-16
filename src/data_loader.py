"""Dataset loaders for SpamAssassin and Enron-Spam corpora."""
from __future__ import annotations

import csv
import os
import re
import tarfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

from .preprocessor import ParsedEmail, parse_email

LABEL_HAM = 0
LABEL_SPAM = 1


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def _label_from_path(path: Path) -> Optional[int]:
    parts = [p.lower() for p in path.parts]
    name = path.name.lower()
    if name == "spam" or name.startswith("spm") or "spam" in name:
        if name.endswith(".txt") and re.search(r"\d+", name):
            return LABEL_SPAM
        if name in {"spam", "spams"}:
            return LABEL_SPAM
    if name == "ham" or "ham" in name:
        if name.endswith(".txt") and re.search(r"\d+", name):
            return LABEL_HAM
        if name in {"ham", "hams", "easy_ham", "hard_ham"}:
            return LABEL_HAM
    for p in parts:
        if p in {"spam", "spams"} or p.startswith("spam_"):
            return LABEL_SPAM
        if p in {"ham", "hams", "easy_ham", "hard_ham"} or p.endswith("_ham"):
            return LABEL_HAM
    return None


def _looks_like_email(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in {"", ".txt", ".eml", ".msg", ".html"}:
        return path.stat().st_size > 0
    parent_name = path.parent.name.lower()
    if parent_name in {"spam", "spams"} or parent_name.startswith("spam_"):
        return path.stat().st_size > 0
    if parent_name in {"ham", "hams", "easy_ham", "hard_ham"} or parent_name.endswith("_ham"):
        return path.stat().st_size > 0
    return False


def load_spamassassin(root: Path) -> list[tuple[ParsedEmail, int]]:
    rows: list[tuple[ParsedEmail, int]] = []
    root = Path(root)
    if not root.exists():
        return rows
    for path in _iter_files(root):
        label = _label_from_path(path)
        if label is None or not _looks_like_email(path):
            continue
        try:
            parsed = parse_email_from_path(path)
        except Exception:
            continue
        rows.append((parsed, label))
    return rows


def parse_email_from_path(path: Path) -> ParsedEmail:
    raw = path.read_bytes()
    try:
        return parse_email(raw)
    except Exception:
        parsed = ParsedEmail()
        parsed.raw_body = raw.decode("utf-8", errors="ignore")
        parsed.body = parsed.raw_body
        parsed.char_count = len(parsed.raw_body)
        parsed.word_count = len(parsed.body.split())
        return parsed


def load_enron_spam(root: Path) -> list[tuple[ParsedEmail, int]]:
    """Enron-Spam folder layout: Enron/<folder>/{spam,ham}.txt per message."""
    rows: list[tuple[ParsedEmail, int]] = []
    root = Path(root)
    if not root.exists():
        return rows
    for path in _iter_files(root):
        name = path.name.lower()
        if name.endswith(".spam.txt"):
            label = LABEL_SPAM
            text_path = path
        elif name.endswith(".ham.txt"):
            label = LABEL_HAM
            text_path = path
        elif name in {"spam.txt", "ham.txt"}:
            label = LABEL_SPAM if name.startswith("spam") else LABEL_HAM
            text_path = path
        else:
            continue
        try:
            text = text_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        parsed = ParsedEmail()
        parsed.raw_body = text
        parsed.body = text
        parsed.char_count = len(text)
        parsed.word_count = len(text.split())
        rows.append((parsed, label))
    return rows


def load_enron_spam_csv(csv_path: Path) -> list[tuple[ParsedEmail, int]]:
    """Load the Enron-Spam HuggingFace CSV (`Message ID, Subject, Message, Spam/Ham, Date`).

    Falls back gracefully when fields overflow Python's csv default field limit.
    """
    import csv
    import sys as _sys
    csv.field_size_limit(2**31 - 1)
    rows: list[tuple[ParsedEmail, int]] = []
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return rows
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            subject = row.get("Subject", "") or ""
            body = row.get("Message", "") or ""
            label_str = (row.get("Spam/Ham", "") or "").strip().lower()
            label = LABEL_SPAM if label_str == "spam" else LABEL_HAM
            full = (subject + "\n" + body).strip()
            parsed = ParsedEmail()
            parsed.subject = subject
            parsed.raw_body = body
            parsed.body = body
            parsed.char_count = len(full)
            parsed.word_count = len(full.split())
            rows.append((parsed, label))
    return rows


def write_csv(rows: list[tuple[ParsedEmail, int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "subject", "sender", "sender_domain", "raw_body", "clean_body",
            "url_count", "has_html", "html_ratio", "has_attachment",
            "attachment_count", "char_count", "word_count",
            "num_exclamation", "num_question", "num_caps_words",
            "num_dollar_signs", "num_digits", "suspicious_subject", "label",
        ])
        for parsed, label in rows:
            d = asdict(parsed)
            writer.writerow([
                d["subject"], d["sender"], d["sender_domain"],
                d["raw_body"], d["body"],
                len(d["urls"]), int(d["has_html"]), d["html_ratio"],
                int(d["has_attachment"]), d["attachment_count"],
                d["char_count"], d["word_count"], d["num_exclamation"],
                d["num_question"], d["num_caps_words"], d["num_dollar_signs"],
                d["num_digits"], int(d["suspicious_subject"]), label,
            ])


def load_csv(in_path: Path) -> tuple[list[ParsedEmail], list[int]]:
    import sys as _sys
    csv.field_size_limit(2**31 - 1)
    emails: list[ParsedEmail] = []
    labels: list[int] = []
    with in_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed = ParsedEmail(
                subject=row.get("subject", "") or "",
                sender=row.get("sender", "") or "",
                sender_domain=row.get("sender_domain", "") or "",
                raw_body=row.get("raw_body", "") or "",
                body=row.get("clean_body", "") or "",
                urls=[],
                has_html=bool(int(row.get("has_html", 0) or 0)),
                html_ratio=float(row.get("html_ratio", 0) or 0.0),
                has_attachment=bool(int(row.get("has_attachment", 0) or 0)),
                attachment_count=int(row.get("attachment_count", 0) or 0),
                char_count=int(row.get("char_count", 0) or 0),
                word_count=int(row.get("word_count", 0) or 0),
                num_exclamation=int(row.get("num_exclamation", 0) or 0),
                num_question=int(row.get("num_question", 0) or 0),
                num_caps_words=int(row.get("num_caps_words", 0) or 0),
                num_dollar_signs=int(row.get("num_dollar_signs", 0) or 0),
                num_digits=int(row.get("num_digits", 0) or 0),
                suspicious_subject=bool(int(row.get("suspicious_subject", 0) or 0)),
            )
            parsed.urls = re.findall(r"https?://\S+", parsed.raw_body or "")
            emails.append(parsed)
            labels.append(int(row["label"]))
    return emails, labels


def extract_archive(archive: Path, dest: Path) -> None:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    suffix = archive.suffix.lower()
    if suffix in {".tar", ".tgz", ".gz"} or archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(dest)
    elif suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")