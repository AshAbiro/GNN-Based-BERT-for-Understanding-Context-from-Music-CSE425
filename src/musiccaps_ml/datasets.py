from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio import cached_graph_features, cached_mel_spectrogram


class TextDataset(Dataset):
    def __init__(self, frame, labels: np.ndarray):
        self.frame = frame.reset_index(drop=True)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        return {
            "id": str(row["ytid"]),
            "text": str(row["caption"]),
            "labels": torch.tensor(self.labels[index], dtype=torch.float32),
        }


def collate_text(batch, tokenizer, max_length: int = 128):
    encoded = tokenizer(
        [item["text"] for item in batch],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    encoded["labels"] = torch.stack([item["labels"] for item in batch])
    encoded["ids"] = [item["id"] for item in batch]
    encoded["texts"] = [item["text"] for item in batch]
    return encoded


def text_collator(tokenizer, max_length: int = 128):
    return partial(collate_text, tokenizer=tokenizer, max_length=max_length)


class GraphDataset(Dataset):
    def __init__(self, frame, labels: np.ndarray, cache_dir: str | Path):
        self.frame = frame.reset_index(drop=True)
        self.labels = labels
        self.cache_dir = Path(cache_dir)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        from torch_geometric.data import Data

        row = self.frame.iloc[index]
        x, edge_index = cached_graph_features(row["audio_path"], self.cache_dir)
        return Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index),
            y=torch.tensor(self.labels[index][None], dtype=torch.float32),
            sample_id=str(row["ytid"]),
        )


class MelDataset(Dataset):
    def __init__(self, frame, labels: np.ndarray, cache_dir: str | Path):
        self.frame = frame.reset_index(drop=True)
        self.labels = labels
        self.cache_dir = Path(cache_dir)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        mel = cached_mel_spectrogram(row["audio_path"], self.cache_dir)
        return (
            torch.from_numpy(mel),
            torch.tensor(self.labels[index], dtype=torch.float32),
            str(row["ytid"]),
        )


class MultimodalDataset(GraphDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        graph = super().__getitem__(index)
        row = self.frame.iloc[index]
        values = []
        for column in ("valence", "arousal"):
            try:
                value = float(row[column]) if column in self.frame.columns else np.nan
            except (TypeError, ValueError):
                value = np.nan
            values.append(value if np.isfinite(value) else np.nan)
        return {
            "graph": graph,
            "id": str(row["ytid"]),
            "text": str(row["caption"]),
            "labels": torch.tensor(self.labels[index], dtype=torch.float32),
            "affect": torch.tensor(values, dtype=torch.float32),
        }


def collate_multimodal(batch, tokenizer, max_length: int = 128):
    from torch_geometric.data import Batch

    encoded = tokenizer(
        [item["text"] for item in batch],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    return {
        "graph": Batch.from_data_list([item["graph"] for item in batch]),
        "tokens": encoded,
        "labels": torch.stack([item["labels"] for item in batch]),
        "affect": torch.stack([item["affect"] for item in batch]),
        "ids": [item["id"] for item in batch],
        "texts": [item["text"] for item in batch],
    }


def multimodal_collator(tokenizer, max_length: int = 128):
    return partial(collate_multimodal, tokenizer=tokenizer, max_length=max_length)
