from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np


def _librosa():
    # Numba can otherwise select a read-only site-packages cache on managed Windows installs.
    cache_dir = Path(os.environ.get("MUSICCAPS_NUMBA_CACHE", "cache/numba")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
    import numba

    numba.config.CACHE_DIR = str(cache_dir)
    import librosa

    return librosa


def load_audio(path: str | Path, sample_rate: int = 22_050, seconds: float = 10.0) -> np.ndarray:
    """Load mono audio, resample, and deterministically pad/crop to a fixed duration."""
    librosa = _librosa()

    waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
    target = round(sample_rate * seconds)
    if len(waveform) < target:
        waveform = np.pad(waveform, (0, target - len(waveform)))
    return np.asarray(waveform[:target], dtype=np.float32)


def graph_features(
    path: str | Path,
    sample_rate: int = 22_050,
    seconds: float = 10.0,
    nodes: int = 20,
    n_mfcc: int = 20,
    recurrence_k: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Create temporal MFCC/chroma nodes with chain and recurrence edges."""
    librosa = _librosa()

    waveform = load_audio(path, sample_rate, seconds)
    hop_length = 512
    mfcc = librosa.feature.mfcc(
        y=waveform, sr=sample_rate, n_mfcc=n_mfcc, n_fft=2048, hop_length=hop_length
    )
    chroma = librosa.feature.chroma_stft(
        y=waveform, sr=sample_rate, n_fft=2048, hop_length=hop_length
    )
    features = np.concatenate([mfcc, chroma], axis=0)
    frame_groups = np.array_split(np.arange(features.shape[1]), nodes)
    node_features = []
    for group in frame_groups:
        values = features[:, group]
        node_features.append(np.concatenate([values.mean(axis=1), values.std(axis=1)]))
    x = np.asarray(node_features, dtype=np.float32)

    edges: set[tuple[int, int]] = set()
    for index in range(nodes - 1):
        edges.add((index, index + 1))
        edges.add((index + 1, index))
    normalized = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    for source in range(nodes):
        neighbours = np.argsort(-similarity[source])[:recurrence_k]
        for target in neighbours:
            edges.add((source, int(target)))
            edges.add((int(target), source))
    edge_index = np.asarray(sorted(edges), dtype=np.int64).T
    return x, edge_index


def cached_graph_features(
    path: str | Path,
    cache_dir: str | Path,
    sample_rate: int = 22_050,
    seconds: float = 10.0,
    nodes: int = 20,
    n_mfcc: int = 20,
    recurrence_k: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    source = Path(path)
    signature = f"{source.resolve()}|{source.stat().st_mtime_ns}|{sample_rate}|{seconds}|{nodes}|{n_mfcc}|{recurrence_k}"
    key = hashlib.sha1(signature.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{source.stem}-{key[:12]}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["x"], cached["edge_index"]
    x, edge_index = graph_features(
        source, sample_rate, seconds, nodes, n_mfcc, recurrence_k
    )
    np.savez_compressed(cache_path, x=x, edge_index=edge_index)
    return x, edge_index


def mel_spectrogram(
    path: str | Path,
    sample_rate: int = 22_050,
    seconds: float = 10.0,
    n_mels: int = 128,
) -> np.ndarray:
    librosa = _librosa()

    waveform = load_audio(path, sample_rate, seconds)
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / max(float(mel_db.std()), 1e-6)
    return mel_db[None].astype(np.float32)


def cached_mel_spectrogram(
    path: str | Path,
    cache_dir: str | Path,
    sample_rate: int = 22_050,
    seconds: float = 10.0,
    n_mels: int = 128,
) -> np.ndarray:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    source = Path(path)
    signature = f"{source.resolve()}|{source.stat().st_mtime_ns}|{sample_rate}|{seconds}|{n_mels}"
    key = hashlib.sha1(signature.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{source.stem}-{key[:12]}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    mel = mel_spectrogram(source, sample_rate, seconds, n_mels)
    np.save(cache_path, mel)
    return mel
