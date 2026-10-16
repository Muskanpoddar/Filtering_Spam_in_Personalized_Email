"""User profile + feedback tracking for personalized spam detection.

The user profile stores:
- trusted/blocked senders
- sender history (spam counts, ham counts, last seen)
- per-domain reputation
- keyword preferences (always-spam / never-spam tokens)
- feedback events with optional confidence/weighting
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class SenderStats:
    spam_count: int = 0
    ham_count: int = 0
    last_seen: str = ""
    reputation: float = 0.5

    def update(self, label: int) -> None:
        if int(label) == 1:
            self.spam_count += 1
        else:
            self.ham_count += 1
        total = self.spam_count + self.ham_count
        if total > 0:
            self.reputation = self.spam_count / total
        self.last_seen = datetime.utcnow().isoformat()


@dataclass
class FeedbackEvent:
    sender: str = ""
    domain: str = ""
    label: int = 0
    confidence: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    note: str = ""


@dataclass
class UserProfile:
    user_id: str = "default"
    trusted_senders: set[str] = field(default_factory=set)
    blocked_senders: set[str] = field(default_factory=set)
    trusted_domains: set[str] = field(default_factory=set)
    blocked_domains: set[str] = field(default_factory=set)
    spam_keywords: set[str] = field(default_factory=set)
    ham_keywords: set[str] = field(default_factory=set)
    sender_stats: dict[str, SenderStats] = field(default_factory=dict)
    domain_stats: dict[str, SenderStats] = field(default_factory=dict)
    feedback: list[FeedbackEvent] = field(default_factory=list)
    threshold: float = 0.5
    personalization_strength: float = 0.4

    def _key(self, address: str) -> str:
        return (address or "").strip().lower()

    def _domain_key(self, address: str) -> str:
        a = (address or "").strip().lower()
        if "@" in a:
            return a.rsplit("@", 1)[-1]
        return ""

    def trust(self, sender: str) -> None:
        self.trusted_senders.add(self._key(sender))

    def block(self, sender: str) -> None:
        self.blocked_senders.add(self._key(sender))

    def trust_domain(self, domain: str) -> None:
        self.trusted_domains.add((domain or "").strip().lower())

    def block_domain(self, domain: str) -> None:
        self.blocked_domains.add((domain or "").strip().lower())

    def add_spam_keyword(self, token: str) -> None:
        self.spam_keywords.add((token or "").strip().lower())

    def add_ham_keyword(self, token: str) -> None:
        self.ham_keywords.add((token or "").strip().lower())

    def set_threshold(self, t: float) -> None:
        self.threshold = float(max(0.05, min(0.95, t)))

    def set_personalization_strength(self, s: float) -> None:
        self.personalization_strength = float(max(0.0, min(1.0, s)))

    def record(self, sender: str, label: int, confidence: float = 1.0, note: str = "") -> None:
        sender_l = self._key(sender)
        dom = self._domain_key(sender)
        stats = self.sender_stats.setdefault(sender_l, SenderStats())
        stats.update(label)
        if dom:
            d = self.domain_stats.setdefault(dom, SenderStats())
            d.update(label)
        self.feedback.append(FeedbackEvent(
            sender=sender_l, domain=dom, label=int(label),
            confidence=float(confidence), note=note,
        ))

    def sender_reputation(self, sender: str) -> float:
        sender_l = self._key(sender)
        dom = self._domain_key(sender)
        if sender_l in self.trusted_senders:
            return 0.0
        if sender_l in self.blocked_senders:
            return 1.0
        if dom and dom in self.trusted_domains:
            return 0.0
        if dom and dom in self.blocked_domains:
            return 1.0
        if sender_l in self.sender_stats and self.sender_stats[sender_l].spam_count + self.sender_stats[sender_l].ham_count >= 2:
            return self.sender_stats[sender_l].reputation
        if dom and dom in self.domain_stats and self.domain_stats[dom].spam_count + self.domain_stats[dom].ham_count >= 3:
            return self.domain_stats[dom].reputation
        return 0.5

    def keyword_score(self, text: str) -> float:
        if not text:
            return 0.5
        tokens = set(t for t in text.lower().split() if t)
        spam_overlap = len(tokens & self.spam_keywords)
        ham_overlap = len(tokens & self.ham_keywords)
        if spam_overlap == 0 and ham_overlap == 0:
            return 0.5
        total = spam_overlap + ham_overlap
        return spam_overlap / total if total else 0.5

    def personalization_bias(self, sender: str, text: str) -> float:
        rep = self.sender_reputation(sender)
        kw = self.keyword_score(text)
        return 0.7 * rep + 0.3 * kw

    def to_dict(self) -> dict:
        d = asdict(self)
        d["trusted_senders"] = sorted(self.trusted_senders)
        d["blocked_senders"] = sorted(self.blocked_senders)
        d["trusted_domains"] = sorted(self.trusted_domains)
        d["blocked_domains"] = sorted(self.blocked_domains)
        d["spam_keywords"] = sorted(self.spam_keywords)
        d["ham_keywords"] = sorted(self.ham_keywords)
        return d

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "UserProfile":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        prof = cls(user_id=d.get("user_id", "default"))
        prof.trusted_senders = set(d.get("trusted_senders", []))
        prof.blocked_senders = set(d.get("blocked_senders", []))
        prof.trusted_domains = set(d.get("trusted_domains", []))
        prof.blocked_domains = set(d.get("blocked_domains", []))
        prof.spam_keywords = set(d.get("spam_keywords", []))
        prof.ham_keywords = set(d.get("ham_keywords", []))
        prof.sender_stats = {k: SenderStats(**v) for k, v in d.get("sender_stats", {}).items()}
        prof.domain_stats = {k: SenderStats(**v) for k, v in d.get("domain_stats", {}).items()}
        prof.feedback = [FeedbackEvent(**e) for e in d.get("feedback", [])]
        prof.threshold = float(d.get("threshold", 0.5))
        prof.personalization_strength = float(d.get("personalization_strength", 0.4))
        return prof

    def summary(self) -> dict:
        return {
            "user_id": self.user_id,
            "trusted_senders": len(self.trusted_senders),
            "blocked_senders": len(self.blocked_senders),
            "trusted_domains": len(self.trusted_domains),
            "blocked_domains": len(self.blocked_domains),
            "spam_keywords": len(self.spam_keywords),
            "ham_keywords": len(self.ham_keywords),
            "senders_tracked": len(self.sender_stats),
            "feedback_events": len(self.feedback),
            "threshold": self.threshold,
            "personalization_strength": self.personalization_strength,
        }