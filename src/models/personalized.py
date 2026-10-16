"""Personalized spam detector that wraps the hybrid classifier with a user-aware decision layer."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from ..features.structural import extract_batch
from ..models.hybrid import HybridClassifier
from ..personalization.user_profile import UserProfile
from ..preprocessor import ParsedEmail


@dataclass
class PersonalizedSpamDetector:
    hybrid: HybridClassifier
    profile: UserProfile

    @classmethod
    def build(cls, profile: Optional[UserProfile] = None) -> "PersonalizedSpamDetector":
        from ..models.hybrid import HybridClassifier
        return cls(hybrid=HybridClassifier(), profile=profile or UserProfile())

    @property
    def threshold(self) -> float:
        return self.profile.threshold

    def set_threshold(self, t: float) -> None:
        self.profile.set_threshold(t)

    def set_strength(self, s: float) -> None:
        self.profile.set_personalization_strength(s)

    def predict_proba_raw(self, emails: list[ParsedEmail], struct_feats: np.ndarray) -> np.ndarray:
        return self.hybrid.predict_proba(emails, struct_feats)

    def predict_proba(self, emails: list[ParsedEmail], struct_feats: Optional[np.ndarray] = None) -> np.ndarray:
        if struct_feats is None:
            struct_feats = extract_batch(emails)
        base = self.hybrid.predict_proba(emails, struct_feats)
        biases = np.asarray(
            [self.profile.personalization_bias(e.sender or "", e.body or "") for e in emails],
            dtype=np.float32,
        )
        alpha = self.profile.personalization_strength
        blended = (1 - alpha) * base + alpha * biases
        return blended

    def predict(self, emails: list[ParsedEmail], struct_feats: Optional[np.ndarray] = None,
                threshold: Optional[float] = None) -> np.ndarray:
        probs = self.predict_proba(emails, struct_feats)
        t = threshold if threshold is not None else self.profile.threshold
        return (probs >= t).astype(int)

    def provide_feedback(self, email: ParsedEmail, label: int, confidence: float = 1.0,
                         note: str = "") -> None:
        self.profile.record(email.sender or "", int(label), confidence=confidence, note=note)

    def save(self, dir_path: str) -> None:
        dir_path = str(dir_path).replace("\\", "/")
        os.makedirs(dir_path, exist_ok=True)
        self.hybrid.save(dir_path)
        self.profile.save(os.path.join(dir_path, "profile.json").replace("\\", "/"))

    @classmethod
    def load(cls, dir_path: str, profile_path: Optional[str] = None) -> "PersonalizedSpamDetector":
        from ..models.hybrid import HybridClassifier
        dir_path = str(dir_path).replace("\\", "/")
        hyb = HybridClassifier.load(dir_path)
        profile = UserProfile.load((profile_path or os.path.join(dir_path, "profile.json")).replace("\\", "/"))
        return cls(hybrid=hyb, profile=profile)