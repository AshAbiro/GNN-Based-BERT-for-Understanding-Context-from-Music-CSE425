from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from musiccaps_ml.data import load_prepared_data
from musiccaps_ml.datasets import MultimodalDataset, multimodal_collator
from musiccaps_ml.metrics import bidirectional_retrieval
from musiccaps_ml.models import GraphTextDualEncoder, symmetric_infonce
from musiccaps_ml.training import default_device, move_tokens, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Task 4: graph-caption contrastive alignment")
    parser.add_argument("--data-dir", default="artifacts/data")
    parser.add_argument("--output-dir", default="artifacts/task4_contrastive")
    parser.add_argument("--cache-dir", default="cache/graphs")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-split", type=int, default=None, help="Debug-only row cap")
    return parser.parse_args()


def subset(data, split, limit=None):
    mask = data.frame["split"].to_numpy() == split
    frame, labels = data.frame.loc[mask].reset_index(drop=True), data.labels[mask]
    return frame.iloc[:limit].reset_index(drop=True), labels[:limit]


def make_loader(data, split, tokenizer, args, shuffle=False):
    frame, labels = subset(data, split, args.limit_per_split)
    dataset = MultimodalDataset(frame, labels, args.cache_dir)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=multimodal_collator(tokenizer, args.max_length),
        num_workers=0,
    )


def train_epoch(model, loader, optimizer, device):
    model.train()
    losses = []
    for batch in tqdm(loader, leave=False):
        graph = batch["graph"].to(device)
        tokens = move_tokens(batch["tokens"], device)
        graph_embedding, text_embedding, temperature = model(graph, tokens)
        loss = symmetric_infonce(graph_embedding, text_embedding, temperature)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    audio, text, ids = [], [], []
    for batch in tqdm(loader, leave=False):
        graph_embedding, text_embedding, _ = model(
            batch["graph"].to(device), move_tokens(batch["tokens"], device)
        )
        audio.append(graph_embedding.cpu().numpy())
        text.append(text_embedding.cpu().numpy())
        ids.extend(batch["ids"])
    metrics = bidirectional_retrieval(np.concatenate(audio), np.concatenate(text))
    return metrics, ids


def mean_recall(metrics):
    return float(np.mean(list(metrics["audio_to_caption"].values()) + list(metrics["caption_to_audio"].values())))


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_prepared_data(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    loaders = {
        split: make_loader(data, split, tokenizer, args, shuffle=(split == "train"))
        for split in ("train", "val", "test")
    }
    device = default_device()
    model = GraphTextDualEncoder(args.model_name, args.embedding_dim).to(device)
    text_parameters = list(model.text_encoder.parameters())
    text_ids = {id(parameter) for parameter in text_parameters}
    other_parameters = [parameter for parameter in model.parameters() if id(parameter) not in text_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": text_parameters, "lr": args.learning_rate},
            {"params": other_parameters, "lr": args.head_learning_rate},
        ],
        weight_decay=1e-4,
    )
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loaders["train"], optimizer, device)
        val_metrics, _ = evaluate(model, loaders["val"], device)
        score = mean_recall(val_metrics)
        row = {"epoch": epoch, "train_loss": loss, "mean_val_recall": score, **{
            f"val_a2t_{key}": value for key, value in val_metrics["audio_to_caption"].items()
        }, **{
            f"val_t2a_{key}": value for key, value in val_metrics["caption_to_audio"].items()
        }}
        history.append(row)
        print(json.dumps(row))
        if score > best:
            best = score
            torch.save(model.state_dict(), output_dir / "best_model.pt")
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True))
    test_metrics, test_ids = evaluate(model, loaders["test"], device)
    save_json(history, output_dir / "learning_metrics.json")
    save_json(test_metrics, output_dir / "test_retrieval_metrics.json")
    save_json(test_ids, output_dir / "test_order.json")
    save_json(vars(args), output_dir / "config.json")
    save_json(data.vocabulary, output_dir / "tag_vocabulary.json")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
