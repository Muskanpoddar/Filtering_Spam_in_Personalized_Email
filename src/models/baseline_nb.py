"""Baseline TF-IDF + Multinomial Naive Bayes classifier."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.naive_bayes import MultinomialNB

from ..features.tfidf_features import TfidfFeaturizer
from ..preprocessor import ParsedEmail


@dataclass
class TfidfNBClassifier:
    featurizer: TfidfFeaturizer
    model: MultinomialNB

    @classmethod
    def build(cls) -> "TfidfNBClassifier":
        return cls(featurizer=TfidfFeaturizer.default(),
                   model=MultinomialNB(alpha=0.1))

    def fit(self, emails: Iterable[ParsedEmail], labels: np.ndarray) -> "TfidfNBClassifier":
        X = self.featurizer.fit_transform(emails)
        self.model.fit(X, labels)
        return self

    def predict_proba(self, emails: Iterable[ParsedEmail]) -> np.ndarray:
        X = self.featurizer.transform(emails)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, emails: Iterable[ParsedEmail], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(emails) >= threshold).astype(int)

    def save(self, dir_path: str) -> None:
        import joblib
        import os
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(self.model, os.path.join(dir_path, "nb.joblib"))
        self.featurizer.save(os.path.join(dir_path, "tfidf.joblib"))

    @classmethod
    def load(cls, dir_path: str) -> "TfidfNBClassifier":
        import joblib
        import os
        nb = joblib.load(os.path.join(dir_path, "nb.joblib"))
        featurizer = TfidfFeaturizer.load(os.path.join(dir_path, "tfidf.joblib"))
        return cls(featurizer=featurizer, model=nb)