from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from musiccaps_ml.data import load_prepared_data
from musiccaps_ml.datasets import TextDataset, text_collator
from musiccaps_ml.metrics import multilabel_f1
from musiccaps_ml.models import BertTagClassifier
from musiccaps_ml.training import (
    default_device,
    move_tokens,
    positive_weights,
    save_history,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Task 1: BERT multi-label tag classifier")
    parser.add_argument("--data-dir", default="artifacts/data")
    parser.add_argument("--output-dir", default="artifacts/task1_bert")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-split", type=int, default=None, help="Debug-only row cap")
    parser.add_argument("--no-pos-weight", action="store_true")
    return parser.parse_args()


def subset(data, split, limit=None):
    mask = data.frame["split"].to_numpy() == split
    frame, labels = data.frame.loc[mask].reset_index(drop=True), data.labels[mask]
    return frame.iloc[:limit].reset_index(drop=True), labels[:limit]


def run_epoch(model, loader, device, optimizer=None, pos_weight=None, threshold=0.5):
    training = optimizer is not None
    model.train(training)
    losses, targets, probabilities = [], [], []
    for batch in tqdm(loader, leave=False):
        labels = batch.pop("labels").to(device)
        batch.pop("ids")
        batch.pop("texts")
        tokens = move_tokens(batch, device)
        with torch.set_grad_enabled(training):
            logits, _ = model(**tokens)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        losses.append(loss.item())
        targets.append(labels.detach().cpu().numpy())
        probabilities.append(logits.detach().sigmoid().cpu().numpy())
    metrics = multilabel_f1(np.concatenate(targets), np.concatenate(probabilities), threshold)
    return float(np.mean(losses)), metrics


@torch.no_grad()
def visualize_attention(model, dataset, tokenizer, vocabulary, device, output_dir, count=5):
    model.eval()
    output_dir = Path(output_dir)
    attention_dir = output_dir / "attention"
    attention_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for index in range(min(count, len(dataset))):
        item = dataset[index]
        encoded = tokenizer(
            item["text"],
            max_length=128,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        tokens = move_tokens(encoded, device)
        logits, attentions = model(**tokens, output_attentions=True)
        if attentions is None:
            raise RuntimeError("The selected transformer did not return attention tensors")
        # Average heads in the last layer; show attention emitted by the first token.
        weights = attentions[-1][0, :, 0, :].mean(dim=0).detach().cpu().numpy()
        token_names = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        probability = logits.sigmoid()[0].detach().cpu().numpy()
        prediction_index = np.argsort(-probability)[:5]
        truth_index = np.flatnonzero(item["labels"].numpy() > 0.5)
        sample = {
            "id": item["id"],
            "caption": item["text"],
            "true_tags": [vocabulary[i] for i in truth_index],
            "top_predictions": [
                {"tag": vocabulary[i], "probability": float(probability[i])}
                for i in prediction_index
            ],
            "tokens": token_names,
            "cls_attention": [float(value) for value in weights],
        }
        samples.append(sample)
        figure_width = max(9, len(token_names) * 0.42)
        figure, axis = plt.subplots(figsize=(figure_width, 4))
        axis.bar(range(len(weights)), weights)
        axis.set_xticks(range(len(token_names)), token_names, rotation=60, ha="right")
        axis.set_ylabel("Mean last-layer attention")
        axis.set_title(f"{item['id']}: first-token attention")
        figure.tight_layout()
        figure.savefig(attention_dir / f"sample_{index + 1}_{item['id']}.png", dpi=180)
        plt.close(figure)
    save_json(samples, output_dir / "prediction_samples.json")


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_prepared_data(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    loaders, datasets = {}, {}
    for split, shuffle in (("train", True), ("val", False), ("test", False)):
        frame, labels = subset(data, split, args.limit_per_split)
        datasets[split] = TextDataset(frame, labels)
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=text_collator(tokenizer, args.max_length),
        )

    device = default_device()
    model = BertTagClassifier(args.model_name, len(data.vocabulary)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_labels = subset(data, "train", args.limit_per_split)[1]
    pos_weight = None if args.no_pos_weight else positive_weights(train_labels).to(device)
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, loaders["train"], device, optimizer, pos_weight, args.threshold
        )
        val_loss, val_metrics = run_epoch(
            model, loaders["val"], device, None, pos_weight, args.threshold
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row))
        if val_metrics["macro_f1"] > best:
            best = val_metrics["macro_f1"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True))
    test_loss, test_metrics = run_epoch(
        model, loaders["test"], device, None, pos_weight, args.threshold
    )
    save_history(history, output_dir)
    save_json({"loss": test_loss, **test_metrics}, output_dir / "test_metrics.json")
    save_json(vars(args), output_dir / "config.json")
    save_json(data.vocabulary, output_dir / "tag_vocabulary.json")
    visualize_attention(
        model, datasets["test"], tokenizer, data.vocabulary, device, output_dir, count=5
    )
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
