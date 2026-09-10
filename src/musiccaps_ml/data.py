from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperimentData:
    frame: pd.DataFrame
    labels: np.ndarray
    vocabulary: list[str]
    split_method: str


def parse_tags(value: str, source: str) -> list[str]:
    if source == "audioset":
        return [item.strip() for item in str(value).split(",") if item.strip()]
    if source == "aspects":
        parsed = ast.literal_eval(str(value))
        if not isinstance(parsed, list):
            raise ValueError("aspect_list must contain a Python list literal")
        return [str(item).strip().lower() for item in parsed if str(item).strip()]
    raise ValueError(f"Unknown target source: {source}")


def _make_vocabulary(tag_lists: Iterable[list[str]], min_frequency: int) -> list[str]:
    counts: dict[str, int] = {}
    for tags in tag_lists:
        for tag in set(tags):
            counts[tag] = counts.get(tag, 0) + 1
    return sorted(tag for tag, count in counts.items() if count >= min_frequency)


def encode_tags(tag_lists: Iterable[list[str]], vocabulary: list[str]) -> np.ndarray:
    index = {tag: i for i, tag in enumerate(vocabulary)}
    rows = list(tag_lists)
    matrix = np.zeros((len(rows), len(vocabulary)), dtype=np.float32)
    for row_index, tags in enumerate(rows):
        for tag in set(tags):
            if tag in index:
                matrix[row_index, index[tag]] = 1.0
    return matrix


def make_split(
    labels: np.ndarray,
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[np.ndarray, str]:
    """Deterministic iterative multilabel stratification with exact split sizes."""
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be positive and sum to < 1")
    labels = (np.asarray(labels) > 0).astype(np.int8)
    n_rows = len(labels)
    fractions = np.asarray([1.0 - val_fraction - test_fraction, val_fraction, test_fraction])
    raw_sizes = fractions * n_rows
    target_sizes = np.floor(raw_sizes).astype(int)
    for index in np.argsort(-(raw_sizes - target_sizes))[: n_rows - target_sizes.sum()]:
        target_sizes[index] += 1

    remaining_capacity = target_sizes.copy()
    desired_labels = fractions[:, None] * labels.sum(axis=0)[None, :]
    label_remaining = labels.sum(axis=0).astype(int)
    assignments = np.full(n_rows, -1, dtype=int)
    unassigned = np.ones(n_rows, dtype=bool)
    rng = np.random.default_rng(seed)

    while np.any(label_remaining > 0):
        positive_labels = np.flatnonzero(label_remaining > 0)
        rare_count = label_remaining[positive_labels].min()
        rare_labels = positive_labels[label_remaining[positive_labels] == rare_count]
        label = int(rng.choice(rare_labels))
        candidates = np.flatnonzero(unassigned & (labels[:, label] == 1))
        rng.shuffle(candidates)
        for row in candidates:
            available = np.flatnonzero(remaining_capacity > 0)
            if not len(available):
                break
            label_need = desired_labels[available, label]
            best = available[label_need == label_need.max()]
            if len(best) > 1:
                capacity = remaining_capacity[best]
                best = best[capacity == capacity.max()]
            selected = int(rng.choice(best))
            assignments[row] = selected
            unassigned[row] = False
            remaining_capacity[selected] -= 1
            row_labels = labels[row].astype(bool)
            desired_labels[selected, row_labels] -= 1.0
            label_remaining[row_labels] -= 1

    leftovers = np.flatnonzero(unassigned)
    rng.shuffle(leftovers)
    for row in leftovers:
        available = np.flatnonzero(remaining_capacity > 0)
        selected = int(rng.choice(available[remaining_capacity[available] == remaining_capacity[available].max()]))
        assignments[row] = selected
        remaining_capacity[selected] -= 1

    names = np.asarray(["train", "val", "test"], dtype=object)
    return names[assignments], "iterative-multilabel-local"


def prepare_experiment_data(
    csv_path: str | Path,
    audio_dir: str | Path,
    output_dir: str | Path,
    target_source: str = "audioset",
    min_frequency: int = 5,
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    force: bool = False,
) -> ExperimentData:
    csv_path, audio_dir, output_dir = Path(csv_path), Path(audio_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "dataset.csv"
    vocab_path = output_dir / "tag_vocabulary.json"
    meta_path = output_dir / "split_metadata.json"

    if table_path.exists() and vocab_path.exists() and meta_path.exists() and not force:
        frame = pd.read_csv(table_path)
        vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = {"target_source": target_source, "min_frequency": min_frequency, "seed": seed}
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing prepared data uses different settings; pass --force")
        tag_lists = [parse_tags(value, target_source) for value in frame["raw_tags"]]
        return ExperimentData(
            frame, encode_tags(tag_lists, vocabulary), vocabulary, metadata["split_method"]
        )

    raw = pd.read_csv(csv_path)
    tag_column = "audioset_positive_labels" if target_source == "audioset" else "aspect_list"
    required = {"ytid", "caption", tag_column}
    missing_columns = required - set(raw.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

    frame = raw.copy()
    frame["audio_path"] = frame["ytid"].map(lambda item: str(audio_dir / f"{item}.wav"))
    missing_audio = frame.loc[~frame["audio_path"].map(lambda value: Path(value).is_file()), "ytid"]
    if len(missing_audio):
        raise FileNotFoundError(f"Missing audio for {len(missing_audio)} rows; first: {missing_audio.iloc[0]}")

    frame["raw_tags"] = frame[tag_column].astype(str)
    tag_lists = [parse_tags(value, target_source) for value in frame["raw_tags"]]
    vocabulary = _make_vocabulary(tag_lists, min_frequency)
    if not vocabulary:
        raise ValueError("No tags survived min_frequency")
    labels = encode_tags(tag_lists, vocabulary)
    has_target = labels.sum(axis=1) > 0
    frame = frame.loc[has_target].reset_index(drop=True)
    labels = labels[has_target]
    split, split_method = make_split(labels, seed, val_fraction, test_fraction)
    frame["split"] = split

    keep = [
        "ytid", "caption", "audio_path", "raw_tags", "split", "author_id",
        "is_balanced_subset", "is_audioset_eval",
    ]
    for optional in ("valence", "arousal"):
        if optional in frame.columns:
            keep.append(optional)
    frame[keep].to_csv(table_path, index=False)
    vocab_path.write_text(json.dumps(vocabulary, indent=2), encoding="utf-8")
    metadata = {
        "target_source": target_source,
        "min_frequency": min_frequency,
        "seed": seed,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "split_method": split_method,
        "rows": len(frame),
        "labels": len(vocabulary),
        "split_counts": frame["split"].value_counts().to_dict(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return ExperimentData(frame[keep], labels, vocabulary, split_method)


def load_prepared_data(output_dir: str | Path) -> ExperimentData:
    output_dir = Path(output_dir)
    frame = pd.read_csv(output_dir / "dataset.csv")
    vocabulary = json.loads((output_dir / "tag_vocabulary.json").read_text(encoding="utf-8"))
    metadata = json.loads((output_dir / "split_metadata.json").read_text(encoding="utf-8"))
    tag_lists = [parse_tags(value, metadata["target_source"]) for value in frame["raw_tags"]]
    return ExperimentData(
        frame=frame,
        labels=encode_tags(tag_lists, vocabulary),
        vocabulary=vocabulary,
        split_method=metadata["split_method"],
    )
