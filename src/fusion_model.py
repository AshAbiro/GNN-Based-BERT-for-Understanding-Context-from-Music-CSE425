"""
Task 3: GNN-BERT fusion for multi-context understanding.

Cross-attention (brief, Sec 4.3):
    A = softmax(Q K^T / sqrt(d)),   Q = g W_Q,  K = H_text W_K
    z = CONCAT(g, A H_text)
    yhat = sigma(W z)

Multi-task loss:
    L = L_tags + alpha * ||v - vhat||^2 + beta * ||a - ahat||^2   (DEAM valence/arousal, optional)

Requires torch; cross-attention and early-concatenation variants are tested
and included in the real-audio pilot ablation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_encoder import BertTextEncoder
from gnn_model import GnnEncoder


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, g, H_text, text_mask=None):
        """
        g: (batch, d_model) graph readout (query, one per example)
        H_text: (batch, seq_len, d_model) per-token BERT states (key/value)
        text_mask: (batch, seq_len) 1 for real tokens, 0 for padding
        """
        B, S, D = H_text.shape
        Q = self.q_proj(g).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)          # (B, heads, 1, d_head)
        K = self.k_proj(H_text).view(B, S, self.n_heads, self.d_head).transpose(1, 2)     # (B, heads, S, d_head)
        V = self.v_proj(H_text).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.d_head ** 0.5)                          # (B, heads, 1, S)
        if text_mask is not None:
            mask = text_mask[:, None, None, :].bool()
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = (attn @ V).transpose(1, 2).reshape(B, D)                                  # (B, d_model)
        context = self.out_proj(context)
        return context, attn.squeeze(2)                                                     # attn for case-study visualization


class GnnBertFusion(nn.Module):
    def __init__(self, num_tags: int, gnn_kwargs: dict, bert_model_name: str = "distilbert-base-uncased",
                 d_model: int = 256, n_heads: int = 4, dropout: float = 0.1, predict_emotion: bool = False):
        super().__init__()
        gnn_kwargs = dict(gnn_kwargs)
        gnn_kwargs.pop("out_dim", None)
        self.gnn = GnnEncoder(out_dim=d_model, **gnn_kwargs)
        self.bert = BertTextEncoder(bert_model_name, proj_dim=d_model)
        self.fusion = CrossAttentionFusion(d_model, n_heads, dropout)
        self.tag_head = nn.Linear(d_model * 2, num_tags)
        self.predict_emotion = predict_emotion
        if predict_emotion:
            self.emotion_head = nn.Linear(d_model * 2, 2)   # (valence, arousal)

    def forward(self, x, edge_index, batch, input_ids, attention_mask):
        g, _ = self.gnn(x, edge_index, batch)                         # (B, d_model)
        H_text = self.bert.token_states(input_ids, attention_mask)     # (B, S, hidden) -- projected below
        H_text = self.bert.proj(H_text)                                # reuse the same proj layer token-wise -> (B, S, d_model)
        context, attn = self.fusion(g, H_text, text_mask=attention_mask)
        z = torch.cat([g, context], dim=-1)
        tag_logits = self.tag_head(z)
        emo = self.emotion_head(z) if self.predict_emotion else None
        return {"tag_logits": tag_logits, "emotion": emo, "attn": attn, "z": z}


class EarlyConcatFusion(nn.Module):
    """Ablation: concatenate graph and pooled text vectors without attention."""

    def __init__(self, num_tags: int, gnn_kwargs: dict, bert_model_name: str = "distilbert-base-uncased",
                 d_model: int = 256, dropout: float = 0.1, predict_emotion: bool = False):
        super().__init__()
        gnn_kwargs = dict(gnn_kwargs)
        gnn_kwargs.pop("out_dim", None)
        self.gnn = GnnEncoder(out_dim=d_model, **gnn_kwargs)
        self.bert = BertTextEncoder(bert_model_name, proj_dim=d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2), nn.ReLU(), nn.Dropout(dropout)
        )
        self.tag_head = nn.Linear(d_model * 2, num_tags)
        self.predict_emotion = predict_emotion
        if predict_emotion:
            self.emotion_head = nn.Linear(d_model * 2, 2)

    def forward(self, x, edge_index, batch, input_ids, attention_mask):
        g, _ = self.gnn(x, edge_index, batch)
        t = self.bert(input_ids, attention_mask)
        z = self.fuse(torch.cat([g, t], dim=-1))
        return {
            "tag_logits": self.tag_head(z),
            "emotion": self.emotion_head(z) if self.predict_emotion else None,
            "attn": None,
            "z": z,
        }


def multi_task_loss(tag_logits, y_tags, emotion_pred=None, va_target=None, alpha: float = 0.5, beta: float = 0.5):
    tag_loss = F.binary_cross_entropy_with_logits(tag_logits, y_tags)
    if emotion_pred is None or va_target is None:
        return tag_loss, {"tag_loss": tag_loss.item()}
    v_loss = F.mse_loss(emotion_pred[:, 0], va_target[:, 0])
    a_loss = F.mse_loss(emotion_pred[:, 1], va_target[:, 1])
    total = tag_loss + alpha * v_loss + beta * a_loss
    return total, {"tag_loss": tag_loss.item(), "valence_mse": v_loss.item(), "arousal_mse": a_loss.item()}
