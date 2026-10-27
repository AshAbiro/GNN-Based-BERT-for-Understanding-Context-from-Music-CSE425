"""
LP-MusicCaps-MTT loader (Doh et al., 2023, https://arxiv.org/abs/2307.16372).

This is the dataset that actually makes Task 1 / Task 3 / Task 4 match the
project brief as written: real LLM-generated captions over MagnaTagATune
audio, exactly the "MusicCaps caption -> tag proxy task" the brief names as
a valid Task 1 approach (Sec 4.1 deliverables), and MagnaTagATune is itself
in the brief's Table 1. Prefer this over the MTG-Jamendo pseudo-text
workaround in data.py wherever captions are needed.

Fields used (per the dataset card):
    track_id, title, artist_name, release, tag_top50, tag_top188,
    caption_writing, caption_summary, caption_paraphrase, path
Deliberately NOT used, per the dataset authors' own warning in README.md:
    caption_attribute_prediction, pseudo_attribute
    ("Be careful when using caption_attribute_prediction (We don't recommend to use)")

~4k / 25.9k rows have empty tag_top50/captions (tracks with no MagnaTagATune
tags); filtered out by default via `require_caption`.

Audio is NOT included -- only metadata/captions. `path` gives the relative
MagnaTagATune mp3 path (e.g. "0/artist-album-track-title-0-29.mp3") for
downloading the actual audio separately (see scripts/download_audio.sh).
"""
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

TEXT_FIELDS = ("caption_writing", "caption_summary", "caption_paraphrase")   # excludes caption_attribute_prediction, on purpose


@dataclass
class LPMusicCapsDataset:
    data_dir: str   # directory containing train.parquet / valid.parquet / test.parquet

    def __post_init__(self):
        self.frames = {
            "train": pd.read_parquet(os.path.join(self.data_dir, "train.parquet")),
            "validation": pd.read_parquet(os.path.join(self.data_dir, "valid.parquet")),
            "test": pd.read_parquet(os.path.join(self.data_dir, "test.parquet")),
        }

    def vocab(self, tag_field: str = "tag_top50") -> List[str]:
        assert tag_field in ("tag_top50", "tag_top188")
        vocab = set()
        for df in self.frames.values():
            for tags in df[tag_field]:
                vocab.update(tags)
        return sorted(vocab)

    def split_frame(self, split: str, text_field: str = "caption_writing", tag_field: str = "tag_top50",
                     require_caption: bool = True) -> pd.DataFrame:
        assert text_field in TEXT_FIELDS, f"{text_field} not in {TEXT_FIELDS} (caption_attribute_prediction is excluded by design)"
        df = self.frames[split].copy()
        if require_caption:
            df = df[df[text_field].str.len() > 0]
            df = df[df[tag_field].apply(len) > 0]
        return df.reset_index(drop=True)

    def label_matrix(self, df: pd.DataFrame, vocab: List[str], tag_field: str = "tag_top50") -> np.ndarray:
        idx = {v: i for i, v in enumerate(vocab)}
        Y = np.zeros((len(df), len(vocab)), dtype="float32")
        for i, tags in enumerate(df[tag_field]):
            for t in tags:
                if t in idx:
                    Y[i, idx[t]] = 1.0
        return Y

    def audio_path(self, row: pd.Series, audio_root: str) -> str:
        return os.path.join(audio_root, row["path"])


def build_task_frame(ds: LPMusicCapsDataset, split: str, text_field: str = "caption_writing",
                      tag_field: str = "tag_top50", vocab: Optional[List[str]] = None):
    """Convenience wrapper mirroring data.build_task_frame's signature: returns
    (track_ids, text, Y, vocab, paths) for a split, ready for Task 1/3/4 dataloaders."""
    vocab = vocab or ds.vocab(tag_field)
    df = ds.split_frame(split, text_field=text_field, tag_field=tag_field)
    Y = ds.label_matrix(df, vocab, tag_field=tag_field)
    return df["track_id"].tolist(), df[text_field].reset_index(drop=True), Y, vocab, df["path"].tolist()
