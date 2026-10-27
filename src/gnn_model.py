"""
Task 2: GNN encoder on music structure graphs (PyTorch Geometric).

GraphSAGE update (brief, Sec 4.2):
    h_i^(l+1) = sigma( W^(l) . CONCAT(h_i^(l), MEAN_{j in N(i)} h_j^(l)) )
Readout (mean pooling):
    g = (1/|V|) sum_i h_i^(L)

Requires torch and torch_geometric; covered by forward/backward tests and a
real-audio pilot run.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool


class GnnEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 256, num_layers: int = 3,
                 conv_type: str = "sage", gat_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        assert conv_type in ("sage", "gat")
        self.dropout = dropout
        self.convs = nn.ModuleList()

        def make_conv(cin, cout):
            if conv_type == "sage":
                return SAGEConv(cin, cout)
            return GATConv(cin, cout // gat_heads, heads=gat_heads)

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [hidden_dim]
        for i in range(num_layers):
            self.convs.append(make_conv(dims[i], dims[i + 1]))
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = global_mean_pool(h, batch)          # graph-level readout
        g = self.out_proj(g)
        return g, h                              # g: (num_graphs, out_dim), h: node embeddings (Task-3 cross-attn key)


class GnnTagClassifier(nn.Module):
    """Task 2 standalone head: predict tags directly from the graph readout (no text branch)."""

    def __init__(self, num_tags: int, **gnn_kwargs):
        super().__init__()
        self.encoder = GnnEncoder(**gnn_kwargs)
        out_dim = gnn_kwargs.get("out_dim", 256)
        self.head = nn.Linear(out_dim, num_tags)

    def forward(self, x, edge_index, batch):
        g, _ = self.encoder(x, edge_index, batch)
        return self.head(g), g
