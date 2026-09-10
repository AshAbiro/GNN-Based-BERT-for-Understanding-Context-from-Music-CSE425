from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import f1_score


def multilabel_f1(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    predictions = (np.asarray(probabilities) >= threshold).astype(np.int32)
    targets = np.asarray(targets).astype(np.int32)
    return {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
    }


def _recall_at_k(similarity: np.ndarray, ks: Iterable[int]) -> dict[str, float]:
    order = np.argsort(-similarity, axis=1)
    correct = np.arange(similarity.shape[0])[:, None]
    return {
        f"R@{k}": float(np.mean(np.any(order[:, : min(k, order.shape[1])] == correct, axis=1)))
        for k in ks
    }


def bidirectional_retrieval(
    audio_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, dict[str, float]]:
    audio = np.asarray(audio_embeddings, dtype=np.float64)
    text = np.asarray(text_embeddings, dtype=np.float64)
    if audio.shape != text.shape or audio.ndim != 2:
        raise ValueError("Audio and text embeddings must be equal-sized 2D arrays")
    audio /= np.maximum(np.linalg.norm(audio, axis=1, keepdims=True), 1e-12)
    text /= np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12)
    similarity = audio @ text.T
    return {
        "audio_to_caption": _recall_at_k(similarity, ks),
        "caption_to_audio": _recall_at_k(similarity.T, ks),
    }

