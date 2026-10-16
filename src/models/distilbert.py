"""DistilBERT-based contextual spam classifier."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from ..config import (
    BATCH_SIZE,
    DISTILBERT_MODEL_NAME,
    LEARNING_RATE,
    MAX_TOKEN_LENGTH,
    NUM_EPOCHS,
)
from ..preprocessor import ParsedEmail


class EmailDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def _build_text(parsed: ParsedEmail) -> str:
    subject = (parsed.subject or "").strip()
    body = (parsed.body or "").strip()
    sender = (parsed.sender or "").strip()
    head = f"[SUBJ] {subject} [FROM] {sender} [BODY]"
    text = f"{head} {body}"
    return text[:8000]


@dataclass
class DistilBertClassifier:
    model_name: str = DISTILBERT_MODEL_NAME
    max_length: int = MAX_TOKEN_LENGTH
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    num_epochs: int = NUM_EPOCHS
    tokenizer: Optional[object] = None
    model: Optional[object] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self):
        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2
            )
        self.model.to(self.device)

    def _encode(self, emails: Iterable[ParsedEmail], labels=None) -> EmailDataset:
        texts = [_build_text(e) for e in emails]
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        return EmailDataset(enc, labels)

    @torch.no_grad()
    def predict_proba(self, emails: list[ParsedEmail], batch_size: int = 32) -> np.ndarray:
        self._ensure_loaded()
        self.model.eval()
        ds = self._encode(emails)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        probs: list[float] = []
        for batch in dl:
            input_ids = batch["input_ids"].to(self.device)
            attn = batch["attention_mask"].to(self.device)
            with torch.amp.autocast(device_type=self.device.split(":")[0], enabled=self.device != "cpu"):
                out = self.model(input_ids=input_ids, attention_mask=attn)
            p = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
        return np.asarray(probs, dtype=np.float32)

    def predict(self, emails: list[ParsedEmail], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(emails) >= threshold).astype(int)

    def fit(self, train_emails: list[ParsedEmail], train_labels: np.ndarray,
            val_emails: Optional[list[ParsedEmail]] = None,
            val_labels: Optional[np.ndarray] = None,
            verbose: bool = True) -> "DistilBertClassifier":
        self._ensure_loaded()
        train_ds = self._encode(train_emails, train_labels)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = max(1, len(train_dl) * self.num_epochs)
        sched = get_linear_schedule_with_warmup(optim, int(0.1 * total_steps), total_steps)
        scaler = torch.amp.GradScaler(
            enabled=self.device != "cpu"
        )
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            for step, batch in enumerate(train_dl):
                optim.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attn = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                with torch.amp.autocast(device_type=self.device.split(":")[0], enabled=self.device != "cpu"):
                    out = self.model(input_ids=input_ids, attention_mask=attn, labels=labels)
                scaler.scale(out.loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                sched.step()
                total_loss += out.loss.item()
            if verbose:
                avg = total_loss / max(1, len(train_dl))
                msg = f"[DistilBERT] epoch {epoch + 1}/{self.num_epochs} train_loss={avg:.4f}"
                if val_emails is not None and val_labels is not None:
                    val_probs = self.predict_proba(val_emails)
                    val_pred = (val_probs >= 0.5).astype(int)
                    acc = float((val_pred == val_labels).mean())
                    msg += f" val_acc={acc:.4f}"
                if verbose:
                    print(msg)
        return self

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        self.model.save_pretrained(dir_path)
        self.tokenizer.save_pretrained(dir_path)

    @classmethod
    def load(cls, dir_path: str, model_name: Optional[str] = None) -> "DistilBertClassifier":
        inst = cls(model_name=model_name or DISTILBERT_MODEL_NAME)
        inst.tokenizer = AutoTokenizer.from_pretrained(dir_path)
        inst.model = AutoModelForSequenceClassification.from_pretrained(dir_path)
        inst._ensure_loaded()
        return inst