"""Structural (non-textual) email feature extractor."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..preprocessor import ParsedEmail

STRUCTURAL_FEATURE_NAMES = [
    "url_count",
    "unique_domain_count",
    "has_html",
    "html_ratio",
    "has_attachment",
    "attachment_count",
    "char_count_log",
    "word_count_log",
    "subject_len_log",
    "num_exclamation_log",
    "num_question_log",
    "num_caps_words_log",
    "num_dollar_signs_log",
    "num_digits_log",
    "caps_ratio",
    "suspicious_subject",
    "free_email_domain",
    "reply_to_mismatch",
    "recipient_count_log",
    "avg_url_len_log",
]


FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "yahoo.co.uk", "hotmail.com", "outlook.com",
    "aol.com", "icloud.com", "mail.com", "protonmail.com", "live.com",
    "msn.com", "yandex.com", "yandex.ru", "gmx.com",
}


def _safe_log1p(x: float) -> float:
    return math.log1p(max(0.0, x))


def extract_structural(parsed: ParsedEmail) -> np.ndarray:
    url_count = len(parsed.urls)
    domains = {u.split("/")[2].lower() for u in parsed.urls if "://" in u}
    unique_domain_count = len(domains)
    char_count = parsed.char_count
    word_count = parsed.word_count
    caps_ratio = (parsed.num_caps_words / word_count) if word_count else 0.0
    free_email = 1.0 if parsed.sender_domain in FREE_EMAIL_DOMAINS else 0.0
    rt_mismatch = 0.0
    if parsed.reply_to and parsed.sender:
        rt_dom = parsed.reply_to.split("@")[-1].strip("> ").lower()
        sd_dom = parsed.sender_domain
        if rt_dom and sd_dom and rt_dom != sd_dom:
            rt_mismatch = 1.0
    avg_url_len = (
        sum(len(u) for u in parsed.urls) / url_count if url_count else 0.0
    )
    feats = [
        url_count,
        unique_domain_count,
        float(parsed.has_html),
        parsed.html_ratio,
        float(parsed.has_attachment),
        float(parsed.attachment_count),
        _safe_log1p(char_count),
        _safe_log1p(word_count),
        _safe_log1p(len(parsed.subject)),
        _safe_log1p(parsed.num_exclamation),
        _safe_log1p(parsed.num_question),
        _safe_log1p(parsed.num_caps_words),
        _safe_log1p(parsed.num_dollar_signs),
        _safe_log1p(parsed.num_digits),
        caps_ratio,
        float(parsed.suspicious_subject),
        free_email,
        rt_mismatch,
        _safe_log1p(len(parsed.recipients)),
        _safe_log1p(avg_url_len),
    ]
    return np.asarray(feats, dtype=np.float32)


def extract_batch(emails: Iterable[ParsedEmail]) -> np.ndarray:
    return np.vstack([extract_structural(e) for e in emails])


@dataclass
class StructuralScaler:
    scaler: StandardScaler
    fitted: bool = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.scaler.fit(X)
        self.fitted = True
        return self.scaler.transform(X).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("StructuralScaler not fitted")
        return self.scaler.transform(X).astype(np.float32)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self.scaler, path)

    @classmethod
    def load(cls, path: str) -> "StructuralScaler":
        import joblib
        scaler = joblib.load(path)
        instance = cls(scaler=scaler, fitted=True)
        return instance