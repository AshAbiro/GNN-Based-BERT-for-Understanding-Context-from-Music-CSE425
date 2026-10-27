"""
MTG-Jamendo TSV parsing.

File format (ragged, tab-separated):
    TRACK_ID  ARTIST_ID  ALBUM_ID  PATH  DURATION  TAGS...
    track_0000214  artist_000014  album_000031  14/214.mp3  124.6  genre---punk

Each row can have a variable number of TAGS columns, one per tag, formatted
"<category>---<value>" where category in {genre, instrument, mood/theme}.

NOTE ON DESIGN vs. THE ORIGINAL BRIEF: MTG-Jamendo ships no lyrics/captions
(unlike MusicCaps), only anonymized track/artist/album IDs plus multi-label
tags. To avoid the circularity of "predict tags from a text rendering of the
same tags", the BERT text branch is built from one subset of tag categories
(``text_tag_categories``, e.g. genre + instrument) and the prediction target
is a disjoint category (``target_tag_category``, e.g. mood/theme). This keeps
the text -> label mapping genuinely predictive rather than an identity
function. See report/DECISIONS.md.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


def _read_tags_tsv(path: str) -> pd.DataFrame:
    """Robustly read a ragged MTG-Jamendo TSV into TRACK_ID/ARTIST_ID/ALBUM_ID/PATH/DURATION/tags(list)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()  # discard, columns are fixed
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            parts = line.split("\t")
            track_id, artist_id, album_id, rel_path, duration = parts[:5]
            tags = parts[5:]
            rows.append(
                {
                    "track_id": track_id,
                    "artist_id": artist_id,
                    "album_id": album_id,
                    "path": rel_path,
                    "duration": float(duration),
                    "tags": tags,
                }
            )
    return pd.DataFrame(rows)


def split_tags_by_category(tags: List[str]) -> Dict[str, List[str]]:
    """['genre---rock', 'instrument---guitar'] -> {'genre': ['rock'], 'instrument': ['guitar']}"""
    out: Dict[str, List[str]] = {}
    for t in tags:
        if "---" not in t:
            continue
        cat, val = t.split("---", 1)
        cat = "moodtheme" if cat == "mood/theme" else cat
        out.setdefault(cat, []).append(val)
    return out


@dataclass
class JamendoDataset:
    """Loads the full autotagging.tsv (union of genre/instrument/moodtheme tags) plus official split files."""

    data_dir: str
    split_dir: str

    def __post_init__(self):
        self.df = _read_tags_tsv(os.path.join(self.data_dir, "autotagging.tsv"))
        self.df["by_category"] = self.df["tags"].apply(split_tags_by_category)
        self._splits_cache: Dict[str, List[str]] = {}

    # ---------- vocab / label matrices ----------

    def vocab_for(self, category: str, min_count: int = 1) -> List[str]:
        from collections import Counter

        counts = Counter()
        for row in self.df["by_category"]:
            for v in row.get(category, []):
                counts[v] += 1
        return sorted([v for v, c in counts.items() if c >= min_count])

    def label_matrix(self, category: str, track_ids: List[str], vocab: List[str]):
        """Returns a (n_tracks, n_labels) float32 multi-hot numpy array, aligned to track_ids order."""
        import numpy as np

        idx = {v: i for i, v in enumerate(vocab)}
        sub = self.df.set_index("track_id").loc[track_ids]
        Y = np.zeros((len(track_ids), len(vocab)), dtype="float32")
        for i, cats in enumerate(sub["by_category"]):
            for v in cats.get(category, []):
                if v in idx:
                    Y[i, idx[v]] = 1.0
        return Y

    # ---------- pseudo-text side ----------

    def pseudo_text(self, categories: List[str]) -> pd.Series:
        """
        Render a track's tags from `categories` into a short natural-language-ish
        string for the BERT encoder, e.g.
            "genre: rock, metal. instrument: guitar, drums."
        Tracks with none of these categories get a generic placeholder so the
        BERT branch still has *something* to encode.
        """

        def render(cats: Dict[str, List[str]]) -> str:
            chunks = []
            for c in categories:
                vals = cats.get(c, [])
                if vals:
                    chunks.append(f"{c}: " + ", ".join(sorted(vals)) + ".")
            return " ".join(chunks) if chunks else "context: unknown."

        result = self.df["by_category"].apply(render)
        result.index = self.df["track_id"].values
        return result

    # ---------- splits ----------

    def split_ids(self, split: str, category: str = "autotagging") -> List[str]:
        """split in {train, validation, test}; category matches an official split file, e.g.
        autotagging_moodtheme-train.tsv. Falls back to the plain 'autotagging' split files."""
        fname = f"{category}-{split}.tsv"
        path = os.path.join(self.split_dir, fname)
        if not os.path.exists(path):
            fname = f"autotagging-{split}.tsv"
            path = os.path.join(self.split_dir, fname)
        if path in self._splits_cache:
            return self._splits_cache[path]
        ids = _read_tags_tsv(path)["track_id"].tolist()
        self._splits_cache[path] = ids
        return ids

    def audio_path(self, track_id: str, audio_root: str) -> str:
        rel = self.df.set_index("track_id").loc[track_id, "path"]
        return os.path.join(audio_root, rel)


def build_task_frame(ds: JamendoDataset, text_categories: List[str], target_category: str, split: str):
    """
    Convenience: returns (track_ids, text_series, Y, vocab) for a given split, ready
    for the BERT baseline (Task 1) or as the text side of Task 3/4.
    """
    track_ids = ds.split_ids(split, category=target_category if target_category != "autotagging" else "autotagging")
    vocab = ds.vocab_for(target_category, min_count=5)
    Y = ds.label_matrix(target_category, track_ids, vocab)
    text_all = ds.pseudo_text(text_categories)
    text = text_all.loc[track_ids]
    return track_ids, text.reset_index(drop=True), Y, vocab
