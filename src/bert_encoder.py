"""
BERT text branch (Task 1 baseline, and the text side of Task 3/4 fusion).

Text input is the pseudo-description built in data.py from a subset of tag
categories (see data.py module docstring for why -- MTG-Jamendo has no
captions/lyrics). Swap in real captions (e.g. MusicCaps) by replacing
`JamendoDataset.pseudo_text` with a caption lookup; nothing downstream
changes since this module only ever sees strings.

Requires torch and transformers. Unit tests inject a tiny local transformer
for speed; pilot runs use the genuine pretrained DistilBERT weights.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BertTextEncoder(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", proj_dim: int = 256, freeze_backbone: bool = False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden, proj_dim)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def tokenize(self, texts, max_length: int = 64, device=None):
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        return enc

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token (first position) as the pooled contextual representation `t`.
        cls = out.last_hidden_state[:, 0, :]
        return self.proj(cls)              # (batch, proj_dim)

    def token_states(self, input_ids, attention_mask):
        """Full per-token hidden states H_text, used by fusion_model's cross-attention (Task 3)."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state        # (batch, seq_len, hidden)


class BertTagClassifier(nn.Module):
    """Task 1: yhat_k = sigma(w_k^T t + b_k), trained with per-tag BCE (LBERT in the brief)."""

    def __init__(self, num_tags: int, model_name: str = "distilbert-base-uncased", proj_dim: int = 256, freeze_backbone: bool = False):
        super().__init__()
        self.encoder = BertTextEncoder(model_name, proj_dim, freeze_backbone)
        # Eager attention is required to return token-level attention maps for
        # the five Task-1 explanations. (PyTorch SDPA omits these tensors.)
        if hasattr(self.encoder.backbone, "set_attn_implementation"):
            self.encoder.backbone.set_attn_implementation("eager")
        # Task 1 deliberately places a single linear classifier directly on
        # the transformer's first ([CLS]) token.  The encoder projection is
        # retained for Tasks 3/4, but is not part of this classifier.
        self.head = nn.Linear(self.encoder.backbone.config.hidden_size, num_tags)

    def forward(self, input_ids, attention_mask, return_attentions: bool = False):
        out = self.encoder.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attentions,
        )
        cls = out.last_hidden_state[:, 0, :]
        logits = self.head(cls)              # BCEWithLogitsLoss expects raw logits
        if return_attentions:
            return logits, cls, out.attentions
        return logits, cls
