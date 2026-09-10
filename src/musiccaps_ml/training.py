from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def positive_weights(labels: np.ndarray, maximum: float = 25.0) -> torch.Tensor:
    positives = labels.sum(axis=0)
    negatives = len(labels) - positives
    weights = negatives / np.maximum(positives, 1.0)
    return torch.tensor(np.clip(weights, 1.0, maximum), dtype=torch.float32)


def save_history(history: list[dict], output_dir: str | Path, prefix: str = "learning") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with (output_dir / f"{prefix}_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split in ("train", "val"):
        for axis, metric, title in zip(
            axes, ("macro_f1", "micro_f1"), ("Macro-F1", "Micro-F1")
        ):
            key = f"{split}_{metric}"
            if key in history[0]:
                axis.plot([row["epoch"] for row in history], [row[key] for row in history], label=split)
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylim(0, 1)
            axis.grid(alpha=0.25)
    axes[0].legend()
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"{prefix}_f1_curves.png", dpi=180)
    plt.close(figure)


def save_json(value, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def move_tokens(tokens: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in tokens.items()}
