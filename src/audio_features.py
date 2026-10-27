"""
Audio -> segment feature extraction (Task 2/3/4 audio side).

Requires `librosa` and local MP3s (see scripts/download_audio.sh). The full
pipeline has been exercised on both the deterministic WAV fixture and real
MagnaTagATune pilot examples.
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class AudioFeatures:
    mel: np.ndarray      # (n_mels, T)
    chroma: np.ndarray   # (n_chroma, T)
    mfcc: np.ndarray     # (n_mfcc, T)
    sr: int
    hop_length: int


def load_and_extract(path: str, sample_rate: int = 22050, n_mels: int = 128, n_chroma: int = 12,
                     n_mfcc: int = 20) -> AudioFeatures:
    import librosa

    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    hop_length = 512
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max).astype("float32")
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length).astype("float32")
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=n_mfcc).astype("float32")
    return AudioFeatures(mel=mel_db, chroma=chroma, mfcc=mfcc, sr=sr, hop_length=hop_length)


def segment_windows(feat: AudioFeatures, segment_seconds: float = 5.0) -> List[Tuple[int, int]]:
    """Fixed-length windows over the frame axis. Returns (start_frame, end_frame) pairs."""
    frames_per_sec = feat.sr / feat.hop_length
    win = max(1, int(round(segment_seconds * frames_per_sec)))
    n_frames = feat.mel.shape[1]
    return [(s, min(s + win, n_frames)) for s in range(0, n_frames, win) if s < n_frames]


def segment_features(feat: AudioFeatures, windows: List[Tuple[int, int]], feature_type: str = "mel") -> np.ndarray:
    """Mean-pool segment features. ``mfcc_chroma`` is the Task-2 default."""
    if feature_type == "mel":
        source = feat.mel
    elif feature_type == "mfcc":
        source = feat.mfcc
    elif feature_type == "chroma":
        source = feat.chroma
    elif feature_type == "mfcc_chroma":
        source = np.concatenate([feat.mfcc, feat.chroma], axis=0)
    else:
        raise ValueError(f"unknown feature_type: {feature_type}")
    out = []
    for s, e in windows:
        if e <= s:
            continue
        out.append(source[:, s:e].mean(axis=1))
    return np.stack(out, axis=0).astype("float32") if out else np.zeros((0, source.shape[0]), dtype="float32")


def chroma_to_chord_labels(feat: AudioFeatures, windows: List[Tuple[int, int]]) -> List[int]:
    """
    Crude chord-root proxy: argmax pitch class per window (12 classes, C..B).
    A real chord estimator (e.g. a CRP/HMM chord recognizer) is a reasonable
    upgrade but out of scope for the baseline graph builder.
    """
    labels = []
    for s, e in windows:
        if e <= s:
            continue
        window_chroma = feat.chroma[:, s:e].mean(axis=1)
        labels.append(int(np.argmax(window_chroma)))
    return labels


def beat_synchronous_windows(path: str, sample_rate: int = 22050) -> List[Tuple[float, float]]:
    """Optional alternative to fixed windows: beat-synchronous segmentation via librosa's beat tracker.
    Returns (start_sec, end_sec) pairs between consecutive detected beats."""
    import librosa

    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    if len(beat_times) < 2:
        return [(0.0, librosa.get_duration(y=y, sr=sr))]
    return list(zip(beat_times[:-1], beat_times[1:]))
