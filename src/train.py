"""End-to-end training entry point for Tasks 1--4."""
import argparse
import json
import os
import warnings

import numpy as np

from config import load_config
from evaluate import multilabel_report
from utils import get_device, set_seed


def _compact_report(y_true, y_prob, epoch, train_loss):
    report = multilabel_report(y_true, y_prob)
    return {
        "epoch": int(epoch), "train_loss": float(train_loss),
        "macro_f1": report["macro_f1"], "micro_f1": report["micro_f1"],
        "mean_average_precision": report["mean_auc_pr"],
        "mean_auc_pr": report["mean_auc_pr"],  # backward-compatible key
    }


def _write_json(value, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def _plot_learning_curves(histories, path, title):
    import matplotlib.pyplot as plt

    if isinstance(histories, list):
        histories = {"model": histories}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, history in histories.items():
        epochs = [row["epoch"] + 1 for row in history]
        axes[0].plot(epochs, [row["macro_f1"] for row in history], marker="o", label=name)
        axes[1].plot(epochs, [row["micro_f1"] for row in history], marker="o", label=name)
    for axis, metric in zip(axes, ("Macro-F1", "Micro-F1")):
        axis.set(xlabel="Epoch", ylabel=metric, title=metric)
        axis.grid(alpha=.25)
        axis.legend()
    fig.suptitle(title); fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _task_frame(ds, cfg, split, vocab=None):
    from captions_data import build_task_frame
    return build_task_frame(ds, split, cfg.data.text_field, cfg.data.tag_field, vocab)


def _available_audio_frame(ds, cfg, split, vocab):
    frame = ds.split_frame(split, cfg.data.text_field, cfg.data.tag_field)
    labels = ds.label_matrix(frame, vocab, cfg.data.tag_field)
    exists = np.asarray([os.path.isfile(os.path.join(cfg.data.audio_dir, p)) for p in frame["path"]])
    if not exists.any():
        raise FileNotFoundError(
            f"No {split} audio found under {cfg.data.audio_dir}. "
            "Run scripts/download_audio.sh first; metadata cannot train Tasks 2-4."
        )
    if not exists.all():
        warnings.warn(f"{split}: using {int(exists.sum())}/{len(exists)} rows with available audio")
    frame, labels = frame.loc[exists].reset_index(drop=True), labels[exists]
    limit = int(getattr(cfg.data, "max_audio_samples", 0) or 0)
    if limit:
        frame, labels = frame.iloc[:limit].reset_index(drop=True), labels[:limit]
    return frame, labels


def _gnn_kwargs(cfg):
    return {key: getattr(cfg.gnn, key) for key in
            ("in_dim", "hidden_dim", "out_dim", "num_layers", "conv_type", "gat_heads", "dropout")}


def _graph_dataset(frame, labels, cfg, with_text=False, va_targets=None):
    from graph_dataset import TrackGraphDataset
    return TrackGraphDataset(
        frame["path"].tolist(), cfg.data.audio_dir, labels,
        frame[cfg.data.text_field].tolist() if with_text else None, va_targets,
        cfg.data.segment_seconds, cfg.data.similarity_threshold, cfg.data.sample_rate,
        cfg.data.n_mels, cfg.data.n_mfcc, cfg.data.node_feature_type,
        os.path.join(cfg.data.processed_dir, f"graphs_{cfg.data.node_feature_type}"),
    )


def _optimizer(model, cfg):
    import torch
    return torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)


def _evaluate_text(model, loader, device, max_length=64):
    import torch
    model.eval(); probs, labels = [], []
    with torch.no_grad():
        for texts, y in loader:
            enc = model.encoder.tokenize(list(texts), max_length, device)
            logits, _ = model(enc["input_ids"], enc["attention_mask"])
            probs.append(torch.sigmoid(logits).cpu().numpy()); labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def _save_attention_examples(model, texts, labels, vocab, output_dir, device, max_length, count=5):
    import matplotlib.pyplot as plt
    import torch
    count = min(count, len(texts)); selected = list(texts[:count])
    enc = model.encoder.tokenize(selected, max_length, device)
    model.eval()
    with torch.no_grad():
        logits, _, layers = model(enc["input_ids"], enc["attention_mask"], return_attentions=True)
        probs = torch.sigmoid(logits).cpu().numpy()
    attention = layers[-1][:, :, 0, :].mean(dim=1).detach().cpu().numpy()
    ids = enc["input_ids"].cpu().numpy(); masks = enc["attention_mask"].cpu().numpy()
    records = []; fig, axes = plt.subplots(count, 1, figsize=(12, max(3, 2.6 * count)), squeeze=False)
    for i in range(count):
        length = int(masks[i].sum())
        tokens = model.encoder.tokenizer.convert_ids_to_tokens(ids[i, :length].tolist())
        weights = attention[i, :length]
        pred = np.flatnonzero(probs[i] >= .5).tolist() or np.argsort(probs[i])[-5:][::-1].tolist()
        truth = np.flatnonzero(labels[i] > 0).tolist()
        records.append({
            "text": selected[i], "true_tags": [vocab[j] for j in truth],
            "predicted_tags": [{"tag": vocab[j], "probability": float(probs[i, j])} for j in pred],
            "tokens": tokens, "cls_attention": weights.tolist(),
        })
        axis = axes[i, 0]; axis.bar(range(length), weights)
        axis.set_xticks(range(length), tokens, rotation=55, ha="right", fontsize=7)
        axis.set_ylabel("attention")
        axis.set_title(f"Sample {i + 1}: true={records[-1]['true_tags'][:4]}, predicted={[p['tag'] for p in records[-1]['predicted_tags'][:4]]}", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(output_dir, "task1_attention_samples.png"), dpi=180, bbox_inches="tight"); plt.close(fig)
    _write_json(records, os.path.join(output_dir, "task1_prediction_samples.json"))


def run_task1(cfg, ds, output_dir=None):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from bert_encoder import BertTagClassifier

    class TextTagDataset(Dataset):
        def __init__(self, texts, labels): self.texts, self.labels = list(texts), labels
        def __len__(self): return len(self.texts)
        def __getitem__(self, index): return self.texts[index], self.labels[index]

    _, train_text, train_y, vocab, _ = _task_frame(ds, cfg, "train")
    _, val_text, val_y, _, _ = _task_frame(ds, cfg, "validation", vocab)
    _, test_text, test_y, _, _ = _task_frame(ds, cfg, "test", vocab)
    limit = int(getattr(cfg.data, "max_audio_samples", 0) or 0)
    if limit:
        train_text, train_y = train_text.iloc[:limit], train_y[:limit]
        val_text, val_y = val_text.iloc[:limit], val_y[:limit]
        test_text, test_y = test_text.iloc[:limit], test_y[:limit]
    train_loader = DataLoader(TextTagDataset(train_text, train_y), batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(TextTagDataset(val_text, val_y), batch_size=cfg.train.batch_size)
    test_loader = DataLoader(TextTagDataset(test_text, test_y), batch_size=cfg.train.batch_size)
    device = get_device(cfg.train.device)
    model = BertTagClassifier(len(vocab), cfg.bert.model_name, cfg.bert.proj_dim, cfg.bert.freeze_backbone).to(device)
    opt = torch.optim.AdamW([
        {"params": model.encoder.backbone.parameters(), "lr": cfg.train.bert_lr},
        {"params": model.head.parameters(), "lr": cfg.train.lr},
    ], weight_decay=cfg.train.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss(); history = []
    for epoch in range(cfg.train.epochs):
        model.train(); losses = []
        for texts, y in train_loader:
            y = y.to(device); enc = model.encoder.tokenize(list(texts), cfg.bert.max_length, device)
            logits, _ = model(enc["input_ids"], enc["attention_mask"]); loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        true, prob = _evaluate_text(model, val_loader, device, cfg.bert.max_length)
        history.append(_compact_report(true, prob, epoch, np.mean(losses)))
        print(f"[task1][{epoch + 1}/{cfg.train.epochs}] macro={history[-1]['macro_f1']:.4f} micro={history[-1]['micro_f1']:.4f}")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        torch.save({"model": model.state_dict(), "vocab": vocab}, os.path.join(output_dir, "task1_bert.pt"))
        _write_json(history, os.path.join(output_dir, "task1_history.json"))
        _plot_learning_curves(history, os.path.join(output_dir, "task1_learning_curves.png"), "Task 1: BERT tag classification")
        _save_attention_examples(model, val_text, val_y, vocab, output_dir, device, cfg.bert.max_length)
        test_true, test_prob = _evaluate_text(model, test_loader, device, cfg.bert.max_length)
        test_metrics = _compact_report(test_true, test_prob, cfg.train.epochs - 1, history[-1]["train_loss"])
        _write_json(test_metrics, os.path.join(output_dir, "task1_test_metrics.json"))
    return history


def _evaluate_graph(model, loader, device, cnn=False):
    import torch
    model.eval(); probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            if cnn:
                x, y = batch[0].to(device), batch[1].to(device); logits = model(x)
            else:
                batch = batch.to(device); y = batch.y.view(batch.num_graphs, -1)
                logits, _ = model(batch.x, batch.edge_index, batch.batch)
            probs.append(torch.sigmoid(logits).cpu().numpy()); labels.append(y.cpu().numpy())
    return np.concatenate(labels), np.concatenate(probs)


def _train_graph_classifier(model, train_loader, val_loader, cfg, device, name):
    import torch
    model.to(device); opt = _optimizer(model, cfg); loss_fn = torch.nn.BCEWithLogitsLoss(); history = []
    for epoch in range(cfg.train.epochs):
        model.train(); losses = []
        for batch in train_loader:
            batch = batch.to(device); y = batch.y.view(batch.num_graphs, -1)
            logits, _ = model(batch.x, batch.edge_index, batch.batch); loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        true, prob = _evaluate_graph(model, val_loader, device)
        history.append(_compact_report(true, prob, epoch, np.mean(losses)))
        print(f"[{name}][{epoch + 1}/{cfg.train.epochs}] micro={history[-1]['micro_f1']:.4f}")
    return history


def _train_cnn(model, train_loader, val_loader, cfg, device):
    import torch
    model.to(device); opt = _optimizer(model, cfg); loss_fn = torch.nn.BCEWithLogitsLoss(); history = []
    for epoch in range(cfg.train.epochs):
        model.train(); losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device); logits = model(x); loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        true, prob = _evaluate_graph(model, val_loader, device, cnn=True)
        history.append(_compact_report(true, prob, epoch, np.mean(losses)))
        print(f"[cnn][{epoch + 1}/{cfg.train.epochs}] micro={history[-1]['micro_f1']:.4f}")
    return history


def run_task2(cfg, ds, output_dir=None):
    import torch
    from torch.utils.data import DataLoader
    from torch_geometric.loader import DataLoader as GraphLoader
    from baselines import build_cnn_baseline
    from gnn_model import GnnTagClassifier
    from graph_dataset import MelSpectrogramDataset
    vocab = ds.vocab(cfg.data.tag_field)
    train_frame, train_y = _available_audio_frame(ds, cfg, "train", vocab)
    val_frame, val_y = _available_audio_frame(ds, cfg, "validation", vocab)
    graph_train = GraphLoader(_graph_dataset(train_frame, train_y, cfg), batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)
    graph_val = GraphLoader(_graph_dataset(val_frame, val_y, cfg), batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
    mel_cache = os.path.join(cfg.data.processed_dir, "mels")
    mel_train_ds = MelSpectrogramDataset(train_frame["path"].tolist(), cfg.data.audio_dir, train_y, cfg.data.sample_rate, cfg.data.n_mels, cache_dir=mel_cache)
    mel_val_ds = MelSpectrogramDataset(val_frame["path"].tolist(), cfg.data.audio_dir, val_y, cfg.data.sample_rate, cfg.data.n_mels, cache_dir=mel_cache)
    mel_train = DataLoader(mel_train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)
    mel_val = DataLoader(mel_val_ds, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
    device = get_device(cfg.train.device); gnn = GnnTagClassifier(len(vocab), **_gnn_kwargs(cfg)); cnn = build_cnn_baseline(len(vocab), cfg.data.n_mels)
    histories = {
        cfg.gnn.conv_type: _train_graph_classifier(gnn, graph_train, graph_val, cfg, device, "gnn"),
        "2d_cnn": _train_cnn(cnn, mel_train, mel_val, cfg, device),
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        torch.save({"model": gnn.state_dict(), "vocab": vocab}, os.path.join(output_dir, "task2_gnn.pt"))
        torch.save({"model": cnn.state_dict(), "vocab": vocab}, os.path.join(output_dir, "task2_cnn.pt"))
        _write_json(histories, os.path.join(output_dir, "task2_history.json"))
        _plot_learning_curves(histories, os.path.join(output_dir, "task2_gnn_vs_cnn.png"), "Task 2: GNN vs 2D-CNN")
    return histories


def _multimodal_loader(dataset, tokenizer, cfg, shuffle=False):
    from torch.utils.data import DataLoader
    from graph_dataset import make_collate_fn
    return DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=shuffle,
                      num_workers=0,  # closure collator is intentionally local/non-picklable on Windows
                      collate_fn=make_collate_fn(tokenizer, cfg.bert.max_length))


def _fusion_logits(model, kind, graph, enc):
    if kind == "bert_only": return model(enc["input_ids"], enc["attention_mask"])[0], None
    if kind == "gnn_only": return model(graph.x, graph.edge_index, graph.batch)[0], None
    output = model(graph.x, graph.edge_index, graph.batch, enc["input_ids"], enc["attention_mask"])
    return output["tag_logits"], output["emotion"]


def _train_ablation(model, kind, train_loader, val_loader, cfg, device, use_emotion=False):
    import torch
    from fusion_model import multi_task_loss
    model.to(device); opt = _optimizer(model, cfg); history = []
    for epoch in range(cfg.train.epochs):
        model.train(); losses = []
        for graph, enc, y in train_loader:
            graph, y = graph.to(device), y.to(device); enc = {k: v.to(device) for k, v in enc.items()}
            logits, emotion = _fusion_logits(model, kind, graph, enc)
            va = graph.va.view(-1, 2) if use_emotion and hasattr(graph, "va") else None
            loss, _ = multi_task_loss(logits, y, emotion, va, cfg.train.emotion_alpha, cfg.train.emotion_beta)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        model.eval(); probs, labels = [], []
        with torch.no_grad():
            for graph, enc, y in val_loader:
                graph = graph.to(device); enc = {k: v.to(device) for k, v in enc.items()}
                logits, _ = _fusion_logits(model, kind, graph, enc)
                probs.append(torch.sigmoid(logits).cpu().numpy()); labels.append(y.numpy())
        history.append(_compact_report(np.concatenate(labels), np.concatenate(probs), epoch, np.mean(losses)))
        print(f"[{kind}][{epoch + 1}/{cfg.train.epochs}] micro={history[-1]['micro_f1']:.4f}")
    return history


def run_task3(cfg, ds, output_dir=None):
    import torch
    from bert_encoder import BertTagClassifier
    from fusion_model import EarlyConcatFusion, GnnBertFusion
    from gnn_model import GnnTagClassifier
    vocab = ds.vocab(cfg.data.tag_field)
    train_frame, train_y = _available_audio_frame(ds, cfg, "train", vocab)
    val_frame, val_y = _available_audio_frame(ds, cfg, "validation", vocab)
    use_emotion = bool(getattr(cfg.train, "use_emotion", False)); va_train = va_val = None
    if use_emotion and {"valence", "arousal"}.issubset(train_frame.columns):
        va_train = train_frame[["valence", "arousal"]].to_numpy("float32"); va_val = val_frame[["valence", "arousal"]].to_numpy("float32")
    elif use_emotion:
        warnings.warn("DEAM valence/arousal targets unavailable; optimizing tag loss only"); use_emotion = False
    train_ds = _graph_dataset(train_frame, train_y, cfg, True, va_train)
    val_ds = _graph_dataset(val_frame, val_y, cfg, True, va_val)
    gkw = _gnn_kwargs(cfg)
    models = {
        "bert_only": BertTagClassifier(len(vocab), cfg.bert.model_name, cfg.bert.proj_dim),
        "gnn_only": GnnTagClassifier(len(vocab), **gkw),
        "early_concat": EarlyConcatFusion(len(vocab), gkw, cfg.bert.model_name, cfg.fusion.d_model, cfg.fusion.dropout, use_emotion),
        "cross_attention": GnnBertFusion(len(vocab), gkw, cfg.bert.model_name, cfg.fusion.d_model, cfg.fusion.n_heads, cfg.fusion.dropout, use_emotion),
    }
    device = get_device(cfg.train.device); histories = {}; fallback = models["cross_attention"].bert.tokenizer
    for name, model in models.items():
        tokenizer = model.encoder.tokenizer if name == "bert_only" else (model.bert.tokenizer if hasattr(model, "bert") else fallback)
        train_loader = _multimodal_loader(train_ds, tokenizer, cfg, True); val_loader = _multimodal_loader(val_ds, tokenizer, cfg)
        histories[name] = _train_ablation(model, name, train_loader, val_loader, cfg, device, use_emotion and name in ("early_concat", "cross_attention"))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for name, model in models.items(): torch.save({"model": model.state_dict(), "vocab": vocab}, os.path.join(output_dir, f"task3_{name}.pt"))
        _write_json(histories, os.path.join(output_dir, "task3_ablation.json"))
        _plot_learning_curves(histories, os.path.join(output_dir, "task3_ablation.png"), "Task 3: fusion ablation")
    return histories


def run_task4(cfg, ds, output_dir=None):
    import torch
    from contrastive import ContrastiveDualEncoder, info_nce_loss, recall_at_k
    vocab = ds.vocab(cfg.data.tag_field)
    train_frame, train_y = _available_audio_frame(ds, cfg, "train", vocab)
    test_frame, test_y = _available_audio_frame(ds, cfg, "test", vocab)
    train_ds = _graph_dataset(train_frame, train_y, cfg, True); test_ds = _graph_dataset(test_frame, test_y, cfg, True)
    model = ContrastiveDualEncoder(_gnn_kwargs(cfg), cfg.bert.model_name, cfg.fusion.d_model, cfg.train.contrastive_temp)
    train_loader = _multimodal_loader(train_ds, model.bert.tokenizer, cfg, True); test_loader = _multimodal_loader(test_ds, model.bert.tokenizer, cfg)
    device = get_device(cfg.train.device); model.to(device); opt = _optimizer(model, cfg); history = []
    for epoch in range(cfg.train.epochs):
        model.train(); losses = []
        for graph, enc, _ in train_loader:
            graph = graph.to(device); enc = {k: v.to(device) for k, v in enc.items()}
            g, t = model(graph.x, graph.edge_index, graph.batch, enc["input_ids"], enc["attention_mask"])
            loss, _ = info_nce_loss(g, t, model.temperature); opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "temperature": float(model.temperature.detach().cpu())})
        print(f"[task4][{epoch + 1}/{cfg.train.epochs}] loss={history[-1]['train_loss']:.4f} tau={history[-1]['temperature']:.4f}")
    model.eval(); graph_embeddings, text_embeddings = [], []
    with torch.no_grad():
        for graph, enc, _ in test_loader:
            graph = graph.to(device); enc = {k: v.to(device) for k, v in enc.items()}
            g, t = model(graph.x, graph.edge_index, graph.batch, enc["input_ids"], enc["attention_mask"])
            graph_embeddings.append(g.cpu()); text_embeddings.append(t.cpu())
    g, t = torch.cat(graph_embeddings), torch.cat(text_embeddings)
    metrics = recall_at_k(g @ t.T / model.temperature.detach().cpu(), (1, 5, 10))
    result = {"history": history, "test_size": len(test_ds), "test_retrieval": metrics}
    if output_dir:
        os.makedirs(output_dir, exist_ok=True); torch.save({"model": model.state_dict()}, os.path.join(output_dir, "task4_contrastive.pt"))
        _write_json(result, os.path.join(output_dir, "task4_retrieval.json"))
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--task", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--config"); parser.add_argument("--epochs", type=int); parser.add_argument("--emotion", action="store_true")
    parser.add_argument("--max-audio-samples", type=int, default=None,
                        help="Explicit pilot limit per split; omit or use 0 for the full corpus")
    args = parser.parse_args(); cfg = load_config(args.config)
    if args.epochs is not None: cfg.train.epochs = args.epochs
    if args.max_audio_samples is not None: cfg.data.max_audio_samples = args.max_audio_samples
    cfg.train.use_emotion = args.emotion; set_seed(cfg.train.seed)
    if cfg.data.dataset != "lp_musiccaps": raise ValueError("Tasks 1-4 require LP-MusicCaps metadata/captions")
    from captions_data import LPMusicCapsDataset
    ds = LPMusicCapsDataset(cfg.data.lp_musiccaps_dir)
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    result = {1: run_task1, 2: run_task2, 3: run_task3, 4: run_task4}[args.task](cfg, ds, output_dir)
    _write_json(result, os.path.join(output_dir, f"task{args.task}_history.json"))


if __name__ == "__main__":
    main()
