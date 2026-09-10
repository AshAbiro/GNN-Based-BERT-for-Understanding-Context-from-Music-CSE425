from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from musiccaps_ml.data import load_prepared_data
from musiccaps_ml.datasets import MultimodalDataset, multimodal_collator
from musiccaps_ml.metrics import multilabel_f1
from musiccaps_ml.models import MultimodalAblationModel, multitask_loss
from musiccaps_ml.training import (
    default_device,
    move_tokens,
    positive_weights,
    save_history,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Task 3: GNN-BERT multimodal ablation")
    parser.add_argument("--data-dir", default="artifacts/data")
    parser.add_argument("--output-dir", default="artifacts/task3_fusion")
    parser.add_argument("--cache-dir", default="cache/graphs")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--variants", nargs="+", choices=("fusion", "bert", "gnn", "early"), default=["fusion", "bert", "gnn", "early"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-split", type=int, default=None, help="Debug-only row cap")
    parser.add_argument("--no-pos-weight", action="store_true")
    return parser.parse_args()


def subset(data, split, limit=None):
    mask = data.frame["split"].to_numpy() == split
    frame, labels = data.frame.loc[mask].reset_index(drop=True), data.labels[mask]
    return frame.iloc[:limit].reset_index(drop=True), labels[:limit]


def make_loaders(data, tokenizer, args):
    loaders = {}
    for split, shuffle in (("train", True), ("val", False), ("test", False)):
        frame, labels = subset(data, split, args.limit_per_split)
        dataset = MultimodalDataset(frame, labels, args.cache_dir)
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=multimodal_collator(tokenizer, args.max_length),
            num_workers=0,
        )
    return loaders


def run_epoch(model, loader, device, optimizer, pos_weight, args):
    training = optimizer is not None
    model.train(training)
    losses, targets, probabilities = [], [], []
    valence_losses, arousal_losses = [], []
    for batch in tqdm(loader, leave=False):
        graph = batch["graph"].to(device)
        tokens = move_tokens(batch["tokens"], device)
        labels = batch["labels"].to(device)
        affect = batch["affect"].to(device)
        with torch.set_grad_enabled(training):
            outputs = model(graph, tokens)
            components = multitask_loss(outputs, labels, affect, args.alpha, args.beta, pos_weight)
            loss = components["total"]
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        losses.append(loss.item())
        if components["valence"] is not None:
            valence_losses.append(components["valence"].item())
        if components["arousal"] is not None:
            arousal_losses.append(components["arousal"].item())
        targets.append(labels.detach().cpu().numpy())
        probabilities.append(outputs["tags"].detach().sigmoid().cpu().numpy())
    metrics = multilabel_f1(np.concatenate(targets), np.concatenate(probabilities), args.threshold)
    metrics["valence_mse"] = float(np.mean(valence_losses)) if valence_losses else None
    metrics["arousal_mse"] = float(np.mean(arousal_losses)) if arousal_losses else None
    return float(np.mean(losses)), metrics


def train_variant(variant, data, loaders, args, device):
    run_dir = Path(args.output_dir) / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    model = MultimodalAblationModel(args.model_name, len(data.vocabulary), variant).to(device)
    text_parameters = list(model.text_encoder.parameters()) if model.text_encoder is not None else []
    text_ids = {id(parameter) for parameter in text_parameters}
    other_parameters = [parameter for parameter in model.parameters() if id(parameter) not in text_ids]
    groups = [{"params": other_parameters, "lr": args.head_learning_rate}]
    if text_parameters:
        groups.append({"params": text_parameters, "lr": args.learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    train_labels = subset(data, "train", args.limit_per_split)[1]
    pos_weight = None if args.no_pos_weight else positive_weights(train_labels).to(device)
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, loaders["train"], device, optimizer, pos_weight, args)
        val_loss, val_metrics = run_epoch(model, loaders["val"], device, None, pos_weight, args)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(variant, json.dumps(row))
        if val_metrics["macro_f1"] > best:
            best = val_metrics["macro_f1"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))
    test_loss, test_metrics = run_epoch(model, loaders["test"], device, None, pos_weight, args)
    save_history(history, run_dir)
    save_json({"loss": test_loss, **test_metrics}, run_dir / "test_metrics.json")
    return {"variant": variant, "best_val_macro_f1": best, **test_metrics}


def main():
    args = parse_args()
    set_seed(args.seed)
    data = load_prepared_data(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    loaders = make_loaders(data, tokenizer, args)
    device = default_device()
    results = [train_variant(variant, data, loaders, args, device) for variant in args.variants]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ablation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    save_json(vars(args), output_dir / "config.json")
    save_json(data.vocabulary, output_dir / "tag_vocabulary.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
