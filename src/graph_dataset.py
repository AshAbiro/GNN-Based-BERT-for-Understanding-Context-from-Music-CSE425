"""
Ties captions_data.py (real captions/tags/paths) + audio_features.py +
graph_builder.py into torch_geometric Dataset objects for Tasks 2-4.

This glue path is tested on a deterministic waveform fixture and exercised
on downloaded MagnaTagATune audio in Tasks 2--4 pilot runs:

    python -c "from graph_dataset import TrackGraphDataset; \
               d = TrackGraphDataset(paths[:5], audio_root='data/raw/audio'); \
               print(d[0])"

Requires: torch, torch_geometric, librosa.
"""
import os
from typing import List, Optional

import numpy as np

from audio_features import load_and_extract, segment_windows, segment_features
from graph_builder import build_segment_graph, nx_to_edge_index


class TrackGraphDataset:
    """torch_geometric-style Dataset: index i -> Data(x, edge_index[, y, text])."""

    def __init__(self, paths: List[str], audio_root: str, Y: Optional[np.ndarray] = None,
                 texts: Optional[List[str]] = None, va_targets: Optional[np.ndarray] = None,
                 segment_seconds: float = 5.0,
                 similarity_threshold: float = 0.85, sample_rate: int = 22050,
                 n_mels: int = 128, n_mfcc: int = 20, feature_type: str = "mel",
                 cache_dir: Optional[str] = None):
        self.paths = paths
        self.audio_root = audio_root
        self.Y = Y
        self.texts = texts
        self.va_targets = va_targets
        self.segment_seconds = segment_seconds
        self.similarity_threshold = similarity_threshold
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.feature_type = feature_type
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.paths)

    def _build_graph(self, idx: int):
        import torch
        from torch_geometric.data import Data

        rel_path = self.paths[idx]
        cache_path = os.path.join(self.cache_dir, rel_path.replace("/", "__") + ".pt") if self.cache_dir else None
        if cache_path and os.path.exists(cache_path):
            return torch.load(cache_path, weights_only=False)

        full_path = os.path.join(self.audio_root, rel_path)
        feat = load_and_extract(full_path, sample_rate=self.sample_rate, n_mels=self.n_mels, n_mfcc=self.n_mfcc)
        windows = segment_windows(feat, segment_seconds=self.segment_seconds)
        seg_feats = segment_features(feat, windows, feature_type=self.feature_type)
        G = build_segment_graph(seg_feats, similarity_threshold=self.similarity_threshold)
        edge_index = nx_to_edge_index(G)

        data = Data(
            x=torch.tensor(seg_feats, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
        )
        if cache_path:
            torch.save(data, cache_path)
        return data

    def __getitem__(self, idx: int):
        data = self._build_graph(idx)
        if self.Y is not None:
            import torch

            data.y = torch.tensor(self.Y[idx], dtype=torch.float32)
        if self.texts is not None:
            data.text = self.texts[idx]
        if self.va_targets is not None:
            import torch

            data.va = torch.tensor(self.va_targets[idx], dtype=torch.float32).view(1, 2)
        return data


class MelSpectrogramDataset:
    """Fixed-size log-mel tensors for the Task-2 2D CNN baseline."""

    def __init__(self, paths: List[str], audio_root: str, Y: np.ndarray,
                 sample_rate: int = 22050, n_mels: int = 128,
                 duration_seconds: float = 30.0, cache_dir: Optional[str] = None):
        self.paths, self.audio_root, self.Y = paths, audio_root, Y
        self.sample_rate, self.n_mels = sample_rate, n_mels
        self.target_frames = int(np.ceil(duration_seconds * sample_rate / 512))
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        import torch

        key = self.paths[idx].replace("/", "__") + ".npy"
        cache_path = os.path.join(self.cache_dir, key) if self.cache_dir else None
        if cache_path and os.path.exists(cache_path):
            mel = np.load(cache_path)
        else:
            mel = load_and_extract(
                os.path.join(self.audio_root, self.paths[idx]),
                sample_rate=self.sample_rate,
                n_mels=self.n_mels,
            ).mel
            if cache_path:
                np.save(cache_path, mel)
        mel = mel[:, :self.target_frames]
        if mel.shape[1] < self.target_frames:
            mel = np.pad(mel, ((0, 0), (0, self.target_frames - mel.shape[1])), constant_values=mel.min())
        # Per-track standardization stabilizes the compact CNN baseline.
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return torch.tensor(mel[None], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)

def make_collate_fn(bert_tokenizer, max_length: int = 64, device=None):
    """Batches a list of Data objects (with .text set) into (graph_batch, tokenized_text, y)."""

    def collate(examples):
        import torch
        from torch_geometric.data import Batch

        texts = [e.text for e in examples]
        ys = torch.stack([e.y for e in examples]) if examples[0].y is not None else None
        for e in examples:
            e.text = None    # Batch.from_data_list chokes on non-tensor attributes with mismatched shapes
        graph_batch = Batch.from_data_list(examples)
        enc = bert_tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        if device is not None:
            graph_batch = graph_batch.to(device)
            enc = {k: v.to(device) for k, v in enc.items()}
            if ys is not None:
                ys = ys.to(device)
        return graph_batch, enc, ys

    return collate
