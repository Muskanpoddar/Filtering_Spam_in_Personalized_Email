"""Evaluation metrics utilities."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    auc: float
    confusion_matrix: list[list[int]]
    threshold: float = 0.5

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confusion_matrix"] = self.confusion_matrix
        return d


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_score: np.ndarray | None = None,
                    threshold: float = 0.5) -> Metrics:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    auc = float(roc_auc_score(y_true, y_score)) if y_score is not None and len(np.unique(y_true)) > 1 else float("nan")
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        false_positive_rate=float(fpr),
        auc=auc,
        confusion_matrix=cm.tolist(),
        threshold=float(threshold),
    )


def write_metrics(metrics: Metrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics.to_dict(), fh, indent=2)


def format_metrics(name: str, m: Metrics) -> str:
    return (f"{name:<30s} | acc={m.accuracy:.4f} | prec={m.precision:.4f} | "
            f"rec={m.recall:.4f} | f1={m.f1:.4f} | FPR={m.false_positive_rate:.4f} | "
            f"AUC={m.auc:.4f}")