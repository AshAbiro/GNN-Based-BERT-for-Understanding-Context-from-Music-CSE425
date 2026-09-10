from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class BertTagClassifier(nn.Module):
    """A linear (W, b) classification head directly over the first token."""

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        from transformers import AutoModel

        self.bert = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

    def encode(self, input_ids, attention_mask, output_attentions: bool = False):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )

    def forward(self, input_ids, attention_mask, output_attentions: bool = False, **_):
        output = self.encode(input_ids, attention_mask, output_attentions)
        logits = self.classifier(output.last_hidden_state[:, 0])
        return logits, output.attentions


class GraphSAGEEncoder(nn.Module):
    """GraphSAGE mean aggregation followed by global mean pooling."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, layers: int = 3):
        super().__init__()
        from torch_geometric.nn import SAGEConv

        dimensions = [input_dim] + [hidden_dim] * layers
        self.input_norm = nn.LayerNorm(input_dim)
        self.convolutions = nn.ModuleList(
            [SAGEConv(dimensions[i], dimensions[i + 1], aggr="mean") for i in range(layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.output_dim = hidden_dim

    def forward(self, graph):
        from torch_geometric.nn import global_mean_pool

        x = self.input_norm(graph.x)
        for convolution, norm in zip(self.convolutions, self.norms):
            x = F.relu(norm(convolution(x, graph.edge_index)))
        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return global_mean_pool(x, batch)


class GraphTagClassifier(nn.Module):
    def __init__(self, num_labels: int, input_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.encoder = GraphSAGEEncoder(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, graph):
        embedding = self.encoder(graph)
        return self.classifier(embedding), embedding


class MelCNNClassifier(nn.Module):
    def __init__(self, num_labels: int, hidden_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, hidden_dim, 3, padding=1), nn.BatchNorm2d(hidden_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, mel):
        embedding = self.features(mel).flatten(1)
        return self.classifier(embedding), embedding


class MultimodalAblationModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        variant: str = "fusion",
        graph_dim: int = 128,
        heads: int = 4,
    ):
        super().__init__()
        from transformers import AutoModel

        if variant not in {"fusion", "early", "bert", "gnn"}:
            raise ValueError(f"Unknown ablation variant: {variant}")
        self.variant = variant
        self.text_encoder = AutoModel.from_pretrained(model_name) if variant != "gnn" else None
        self.graph_encoder = GraphSAGEEncoder(hidden_dim=graph_dim) if variant != "bert" else None
        text_dim = self.text_encoder.config.hidden_size if self.text_encoder is not None else 0

        if variant == "fusion":
            self.query_projection = nn.Linear(graph_dim, text_dim)
            self.cross_attention = nn.MultiheadAttention(text_dim, heads, batch_first=True)
            representation_dim = graph_dim + text_dim
        elif variant == "early":
            representation_dim = graph_dim + text_dim
        elif variant == "bert":
            representation_dim = text_dim
        else:
            representation_dim = graph_dim
        self.fusion = nn.Sequential(
            nn.Linear(representation_dim, 256), nn.ReLU(), nn.Dropout(0.2)
        )
        self.tag_head = nn.Linear(256, num_labels)
        self.affect_head = nn.Linear(256, 2)

    def forward(self, graph, tokens):
        text_sequence = None
        graph_vector = None
        if self.text_encoder is not None:
            text_sequence = self.text_encoder(**tokens, return_dict=True).last_hidden_state
        if self.graph_encoder is not None:
            graph_vector = self.graph_encoder(graph)

        attention = None
        if self.variant == "fusion":
            query = self.query_projection(graph_vector).unsqueeze(1)
            attended, attention = self.cross_attention(
                query,
                text_sequence,
                text_sequence,
                key_padding_mask=~tokens["attention_mask"].bool(),
                need_weights=True,
            )
            representation = torch.cat([graph_vector, attended.squeeze(1)], dim=-1)
        elif self.variant == "early":
            representation = torch.cat([graph_vector, text_sequence[:, 0]], dim=-1)
        elif self.variant == "bert":
            representation = text_sequence[:, 0]
        else:
            representation = graph_vector
        hidden = self.fusion(representation)
        return {"tags": self.tag_head(hidden), "affect": self.affect_head(hidden), "attention": attention}


class GraphTextDualEncoder(nn.Module):
    def __init__(self, model_name: str, embedding_dim: int = 256, graph_dim: int = 128):
        super().__init__()
        from transformers import AutoModel

        self.graph_encoder = GraphSAGEEncoder(hidden_dim=graph_dim)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.graph_projection = nn.Linear(graph_dim, embedding_dim)
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, embedding_dim)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.07)))

    def forward(self, graph, tokens):
        graph_embedding = F.normalize(self.graph_projection(self.graph_encoder(graph)), dim=-1)
        sequence = self.text_encoder(**tokens, return_dict=True).last_hidden_state
        text_embedding = F.normalize(self.text_projection(sequence[:, 0]), dim=-1)
        temperature = self.log_temperature.exp().clamp(0.01, 1.0)
        return graph_embedding, text_embedding, temperature


def symmetric_infonce(graph_embedding, text_embedding, temperature):
    logits = graph_embedding @ text_embedding.T / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def multitask_loss(outputs, labels, affect, alpha: float = 0.2, beta: float = 0.2, pos_weight=None):
    tag_loss = F.binary_cross_entropy_with_logits(outputs["tags"], labels, pos_weight=pos_weight)
    losses = {"tags": tag_loss}
    total = tag_loss
    for index, (name, weight) in enumerate((("valence", alpha), ("arousal", beta))):
        mask = torch.isfinite(affect[:, index])
        if mask.any():
            auxiliary = F.mse_loss(outputs["affect"][mask, index], affect[mask, index])
            total = total + weight * auxiliary
            losses[name] = auxiliary
        else:
            losses[name] = None
    losses["total"] = total
    return losses
