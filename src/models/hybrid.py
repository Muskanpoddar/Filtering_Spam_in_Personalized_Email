"""Hybrid DistilBERT + structural feature classifier."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from ..config import BATCH_SIZE, DISTILBERT_MODEL_NAME, LEARNING_RATE, MAX_TOKEN_LENGTH, NUM_EPOCHS
from ..features.structural import StructuralScaler, extract_batch
from ..models.distilbert import EmailDataset, _build_text
from ..preprocessor import ParsedEmail


class HybridModel(nn.Module):
    def __init__(self, model_name: str, structural_dim: int,
                 hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        enc_dim = self.encoder.config.hidden_size
        self.struct_proj = nn.Sequential(
            nn.Linear(structural_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(enc_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, input_ids, attention_mask, struct_feats):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        s = self.struct_proj(struct_feats)
        z = torch.cat([cls, s], dim=-1)
        return self.fusion(z)


class HybridDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, struct_feats, labels=None):
        self.encodings = encodings
        self.struct = struct_feats
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["struct"] = torch.tensor(self.struct[idx], dtype=torch.float32)
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


@dataclass
class HybridClassifier:
    model_name: str = DISTILBERT_MODEL_NAME
    max_length: int = MAX_TOKEN_LENGTH
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    num_epochs: int = NUM_EPOCHS
    scaler: Optional[StructuralScaler] = None
    tokenizer: Optional[object] = None
    model: Optional[HybridModel] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self, struct_dim: int):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.model is None:
            self.model = HybridModel(self.model_name, struct_dim)
        self.model.to(self.device)

    def _encode(self, emails, struct_feats, labels=None):
        texts = [_build_text(e) for e in emails]
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        return HybridDataset(enc, struct_feats, labels)

    def fit(self, train_emails: list[ParsedEmail], train_struct: np.ndarray,
            train_labels: np.ndarray,
            val_emails: Optional[list[ParsedEmail]] = None,
            val_struct: Optional[np.ndarray] = None,
            val_labels: Optional[np.ndarray] = None,
            verbose: bool = True) -> "HybridClassifier":
        self.scaler = StructuralScaler(scaler=StandardScaler())
        train_struct = self.scaler.fit_transform(train_struct)
        self._ensure_loaded(struct_dim=train_struct.shape[1])
        ds = self._encode(train_emails, train_struct, train_labels)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = max(1, len(dl) * self.num_epochs)
        sched = get_linear_schedule_with_warmup(optim, int(0.1 * total_steps), total_steps)
        use_amp = self.device != "cpu"
        scaler_amp = torch.amp.GradScaler(enabled=use_amp)
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            for batch in dl:
                optim.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attn = batch["attention_mask"].to(self.device)
                struct = batch["struct"].to(self.device)
                labels = batch["labels"].to(self.device)
                with torch.amp.autocast(device_type=self.device.split(":")[0], enabled=use_amp):
                    logits = self.model(input_ids, attn, struct)
                    loss = nn.functional.cross_entropy(logits, labels)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler_amp.step(optim)
                scaler_amp.update()
                sched.step()
                total_loss += loss.item()
            if verbose:
                avg = total_loss / max(1, len(dl))
                msg = f"[Hybrid] epoch {epoch + 1}/{self.num_epochs} train_loss={avg:.4f}"
                if val_emails is not None and val_labels is not None:
                    probs = self.predict_proba(val_emails, val_struct)
                    preds = (probs >= 0.5).astype(int)
                    acc = float((preds == val_labels).mean())
                    msg += f" val_acc={acc:.4f}"
                print(msg)
        return self

    @torch.no_grad()
    def predict_proba(self, emails: list[ParsedEmail], struct_feats: np.ndarray,
                      batch_size: int = 32) -> np.ndarray:
        if self.scaler is None or not self.scaler.fitted:
            raise RuntimeError("HybridClassifier not fitted")
        struct_feats = self.scaler.transform(struct_feats)
        struct_dim = struct_feats.shape[1]
        self._ensure_loaded(struct_dim=struct_dim)
        self.model.eval()
        ds = self._encode(emails, struct_feats)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        probs: list[float] = []
        for batch in dl:
            input_ids = batch["input_ids"].to(self.device)
            attn = batch["attention_mask"].to(self.device)
            struct = batch["struct"].to(self.device)
            with torch.amp.autocast(device_type=self.device.split(":")[0], enabled=self.device != "cpu"):
                logits = self.model(input_ids, attn, struct)
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
        return np.asarray(probs, dtype=np.float32)

    def predict(self, emails: list[ParsedEmail], struct_feats: np.ndarray,
                threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(emails, struct_feats) >= threshold).astype(int)

    def save(self, dir_path: str) -> None:
        dir_path = str(dir_path).replace("\\", "/")
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(dir_path, "hybrid.pt").replace("\\", "/"))
        self.tokenizer.save_pretrained(dir_path)
        self.scaler.save(os.path.join(dir_path, "scaler.joblib").replace("\\", "/"))

    @classmethod
    def load(cls, dir_path: str, model_name: Optional[str] = None,
             structural_dim: Optional[int] = None) -> "HybridClassifier":
        dir_path = str(dir_path).replace("\\", "/")
        inst = cls(model_name=model_name or DISTILBERT_MODEL_NAME)
        inst.tokenizer = AutoTokenizer.from_pretrained(dir_path)
        state = torch.load(os.path.join(dir_path, "hybrid.pt").replace("\\", "/"),
                           map_location=inst.device)
        if structural_dim is None:
            struct_proj_weight = state.get("struct_proj.0.weight")
            if struct_proj_weight is not None:
                structural_dim = int(struct_proj_weight.shape[1])
            else:
                structural_dim = 20
        inst.model = HybridModel(inst.model_name, structural_dim)
        inst.model.load_state_dict(state)
        inst.scaler = StructuralScaler.load(os.path.join(dir_path, "scaler.joblib").replace("\\", "/"))
        inst.model.to(inst.device)
        return inst