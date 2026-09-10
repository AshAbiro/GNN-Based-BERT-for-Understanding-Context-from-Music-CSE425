from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
from tqdm import tqdm

from musiccaps_ml.data import load_prepared_data
from musiccaps_ml.datasets import GraphDataset, MelDataset
from musiccaps_ml.metrics import multilabel_f1
from musiccaps_ml.models import GraphTagClassifier, MelCNNClassifier
from musiccaps_ml.training import (
    default_device,
    positive_weights,
    save_history,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2: GraphSAGE versus mel-CNN")
    parser.add_argument("--data-dir", default="artifacts/data")
    parser.add_argument("--output-dir", default="artifacts/task2_gnn")
    parser.add_argument("--cache-dir", default="cache/graphs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-split", type=int, default=None, help="Debug-only row cap")
    parser.add_argument("--no-pos-weight", action="store_true")
    return parser.parse_args()


def subset(data, split, limit=None):
    mask = data.frame["split"].to_numpy() == split
    frame, labels = data.frame.loc[mask].reset_index(drop=True), data.labels[mask]
    return frame.iloc[:limit].reset_index(drop=True), labels[:limit]


def make_loaders(data, kind, args):
    loaders = {}
    for split, shuffle in (("train", True), ("val", False), ("test", False)):
        frame, labels = subset(data, split, args.limit_per_split)
        if kind == "graphsage":
            dataset = GraphDataset(frame, labels, args.cache_dir)
            loaders[split] = GraphDataLoader(
                dataset, batch_size=args.batch_size, shuffle=shuffle
            )
        else:
            mel_cache = Path(args.cache_dir).parent / "mels"
            dataset = MelDataset(frame, labels, mel_cache)
            loaders[split] = TorchDataLoader(
                dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=0
            )
    return loaders


def run_epoch(model, loader, kind, device, optimizer, pos_weight, threshold):
    training = optimizer is not None
    model.train(training)
    losses, targets, probabilities = [], [], []
    for batch in tqdm(loader, leave=False):
        if kind == "graphsage":
            batch = batch.to(device)
            labels = batch.y
            inputs = batch
        else:
            mel, labels, _ = batch
            inputs, labels = mel.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits, _ = model(inputs)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        losses.append(loss.item())
        targets.append(labels.detach().cpu().numpy())
        probabilities.append(logits.detach().sigmoid().cpu().numpy())
    metrics = multilabel_f1(np.concatenate(targets), np.concatenate(probabilities), threshold)
    return float(np.mean(losses)), metrics


def train_model(kind, data, args, device):
    run_dir = Path(args.output_dir) / kind
    run_dir.mkdir(parents=True, exist_ok=True)
    loaders = make_loaders(data, kind, args)
    model = (
        GraphTagClassifier(len(data.vocabulary))
        if kind == "graphsage"
        else MelCNNClassifier(len(data.vocabulary))
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    train_labels = subset(data, "train", args.limit_per_split)[1]
    pos_weight = None if args.no_pos_weight else positive_weights(train_labels).to(device)
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, loaders["train"], kind, device, optimizer, pos_weight, args.threshold
        )
        val_loss, val_metrics = run_epoch(
            model, loaders["val"], kind, device, None, pos_weight, args.threshold
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(kind, json.dumps(row))
        if val_metrics["macro_f1"] > best:
            best = val_metrics["macro_f1"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))
    test_loss, test_metrics = run_epoch(
        model, loaders["test"], kind, device, None, pos_weight, args.threshold
    )
    save_history(history, run_dir)
    save_json({"loss": test_loss, **test_metrics}, run_dir / "test_metrics.json")
    return {"model": kind, "best_val_macro_f1": best, **test_metrics}


def main():
    args = parse_args()
    set_seed(args.seed)
    data = load_prepared_data(args.data_dir)
    device = default_device()
    results = [train_model(kind, data, args, device) for kind in ("graphsage", "mel_cnn")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    save_json(vars(args), output_dir / "config.json")
    save_json(data.vocabulary, output_dir / "tag_vocabulary.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
