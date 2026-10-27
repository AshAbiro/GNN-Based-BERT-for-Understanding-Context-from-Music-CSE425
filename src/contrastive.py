"""
Task 4: contrastive dual-encoder alignment between audio graphs and text.

InfoNCE (brief, Sec 4.4):
    L_NCE = -(1/N) sum_i log( exp(sim(g_i,t_i)/tau) / sum_j exp(sim(g_i,t_j)/tau) )
    sim(u,v) = u^T v / (||u|| ||v||)

The primary workflow pairs real LP-MusicCaps free-text captions with their
corresponding MagnaTagATune audio. The legacy MTG-Jamendo pseudo-text path is
not used for reported retrieval results.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_encoder import BertTextEncoder
from gnn_model import GnnEncoder


class ContrastiveDualEncoder(nn.Module):
    def __init__(self, gnn_kwargs: dict, bert_model_name: str = "distilbert-base-uncased", d_model: int = 256,
                 initial_temperature: float = 0.07):
        super().__init__()
        gnn_kwargs = dict(gnn_kwargs)
        gnn_kwargs.pop("out_dim", None)
        self.gnn = GnnEncoder(out_dim=d_model, **gnn_kwargs)
        self.bert = BertTextEncoder(bert_model_name, proj_dim=d_model)
        # Optimize log(tau) so the temperature stays strictly positive.
        self.log_temperature = nn.Parameter(torch.tensor(float(initial_temperature)).log())

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(min=1e-4, max=1.0)

    def encode_graph(self, x, edge_index, batch):
        g, _ = self.gnn(x, edge_index, batch)
        return F.normalize(g, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        t = self.bert(input_ids, attention_mask)
        return F.normalize(t, dim=-1)

    def forward(self, x, edge_index, batch, input_ids, attention_mask):
        g = self.encode_graph(x, edge_index, batch)
        t = self.encode_text(input_ids, attention_mask)
        return g, t


def info_nce_loss(g: torch.Tensor, t: torch.Tensor, temperature=0.07):
    """g, t: (N, d) L2-normalized embeddings, positives on the diagonal. Symmetric InfoNCE."""
    logits = g @ t.T / temperature          # (N, N)
    targets = torch.arange(g.shape[0], device=g.device)
    loss_g2t = F.cross_entropy(logits, targets)
    loss_t2g = F.cross_entropy(logits.T, targets)
    return (loss_g2t + loss_t2g) / 2, logits


@torch.no_grad()
def recall_at_k(logits: torch.Tensor, ks=(1, 5, 10)):
    """logits: (N, N) similarity matrix, row i's positive is column i.
    Returns {'g2t_R@k': ..., 't2g_R@k': ...} for each k."""
    N = logits.shape[0]
    targets = torch.arange(N, device=logits.device)
    out = {}
    for direction, mat in [("g2t", logits), ("t2g", logits.T)]:
        ranks = mat.argsort(dim=-1, descending=True)
        for k in ks:
            hit = (ranks[:, :k] == targets[:, None]).any(dim=-1).float().mean().item()
            out[f"{direction}_R@{k}"] = hit
    return out
