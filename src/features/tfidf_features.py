"""TF-IDF text feature extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer

from ..preprocessor import ParsedEmail


@dataclass
class TfidfFeaturizer:
    vectorizer: TfidfVectorizer
    fitted: bool = False

    @classmethod
    def default(cls) -> "TfidfFeaturizer":
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )
        return cls(vectorizer=vec, fitted=False)

    def _texts(self, emails: Iterable[ParsedEmail]) -> list[str]:
        texts: list[str] = []
        for e in emails:
            subject = e.subject or ""
            body = e.body or ""
            texts.append(f"{subject} \n {body}")
        return texts

    def fit_transform(self, emails: Iterable[ParsedEmail]):
        X = self.vectorizer.fit_transform(self._texts(emails))
        self.fitted = True
        return X

    def transform(self, emails: Iterable[ParsedEmail]):
        if not self.fitted:
            raise RuntimeError("TfidfFeaturizer not fitted")
        return self.vectorizer.transform(self._texts(emails))

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self.vectorizer, path)

    @classmethod
    def load(cls, path: str) -> "TfidfFeaturizer":
        import joblib
        vec = joblib.load(path)
        return cls(vectorizer=vec, fitted=True)