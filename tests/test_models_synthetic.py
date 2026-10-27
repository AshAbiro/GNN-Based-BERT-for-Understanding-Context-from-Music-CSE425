"""
Real forward/backward-pass smoke tests for bert_encoder.py, gnn_model.py,
fusion_model.py, contrastive.py, and graph_dataset.py's collate logic --
using actual torch / transformers / torch_geometric (installed in this
environment), on synthetic data.

These are NOT training-quality results: `AutoModel.from_pretrained` /
`AutoTokenizer.from_pretrained` are mocked to return a tiny, randomly
initialized DistilBERT (2 layers, dim=32) and a small local offline
tokenizer (`tests/fixtures/tiny_vocab.txt`) to keep the suite fast and
deterministic. What these tests verify for real: every module actually
imports, wires together, and runs a forward + backward pass without shape
errors or NaNs -- the thing that could NOT be checked with `py_compile`
alone. Separate saved pilot runs use genuine pretrained weights and audio.
"""
import os
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import BertTokenizer, DistilBertConfig, DistilBertModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

VOCAB_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "tiny_vocab.txt")


def _tiny_backbone(*args, **kwargs):
    cfg = DistilBertConfig(vocab_size=120, dim=32, n_layers=2, n_heads=2, hidden_dim=64, max_position_embeddings=64)
    return DistilBertModel(cfg)


def _tiny_tokenizer(*args, **kwargs):
    return BertTokenizer(vocab_file=VOCAB_PATH)


@pytest.fixture(autouse=True)
def mock_pretrained():
    with patch("transformers.AutoModel.from_pretrained", side_effect=_tiny_backbone), \
         patch("transformers.AutoTokenizer.from_pretrained", side_effect=_tiny_tokenizer):
        yield


# ---------- Task 1: bert_encoder.py ----------

def test_bert_text_encoder_forward_shape():
    from bert_encoder import BertTextEncoder

    enc = BertTextEncoder(proj_dim=16)
    texts = ["genre: rock, metal. instrument: guitar, drums.", "context: unknown."]
    tokens = enc.tokenize(texts, max_length=16)
    out = enc(tokens["input_ids"], tokens["attention_mask"])
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


def test_bert_tag_classifier_trains_one_step():
    from bert_encoder import BertTagClassifier

    model = BertTagClassifier(num_tags=5, proj_dim=16)
    texts = ["genre: rock, metal.", "genre: classical.", "instrument: guitar, drums."]
    y = torch.randint(0, 2, (3, 5)).float()
    tokens = model.encoder.tokenize(texts, max_length=16)

    logits, t = model(tokens["input_ids"], tokens["attention_mask"])
    assert logits.shape == (3, 5)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    assert torch.isfinite(loss)
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(np.isfinite(g) for g in grad_norms)
    # The task specification requires the classifier directly on [CLS].
    assert model.head.in_features == model.encoder.backbone.config.hidden_size


# ---------- Task 2: gnn_model.py (real torch_geometric, no mocking needed) ----------

def _synthetic_graph_batch(n_graphs=3, n_nodes=6, feat_dim=8, seed=0):
    from torch_geometric.data import Batch, Data

    from graph_builder import build_segment_graph, nx_to_edge_index

    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(n_graphs):
        feats = rng.normal(size=(n_nodes, feat_dim)).astype("float32")
        G = build_segment_graph(feats, similarity_threshold=0.9)
        ei = nx_to_edge_index(G)
        graphs.append(Data(x=torch.tensor(feats), edge_index=torch.tensor(ei, dtype=torch.long)))
    return Batch.from_data_list(graphs)


def test_gnn_encoder_forward_shape():
    from gnn_model import GnnEncoder

    batch = _synthetic_graph_batch()
    enc = GnnEncoder(in_dim=8, hidden_dim=16, out_dim=32, num_layers=2, conv_type="sage")
    g, h = enc(batch.x, batch.edge_index, batch.batch)
    assert g.shape == (3, 32)
    assert torch.isfinite(g).all()


def test_gnn_encoder_gat_variant():
    from gnn_model import GnnEncoder

    batch = _synthetic_graph_batch()
    enc = GnnEncoder(in_dim=8, hidden_dim=16, out_dim=32, num_layers=2, conv_type="gat", gat_heads=4)
    g, h = enc(batch.x, batch.edge_index, batch.batch)
    assert g.shape == (3, 32)
    assert torch.isfinite(g).all()


def test_gnn_tag_classifier_trains_one_step():
    from gnn_model import GnnTagClassifier

    batch = _synthetic_graph_batch()
    model = GnnTagClassifier(num_tags=5, in_dim=8, hidden_dim=16, out_dim=32, num_layers=2)
    logits, g = model(batch.x, batch.edge_index, batch.batch)
    y = torch.randint(0, 2, (3, 5)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


# ---------- Task 3: fusion_model.py ----------

def test_cross_attention_fusion_shapes_and_masking():
    from fusion_model import CrossAttentionFusion

    fusion = CrossAttentionFusion(d_model=16, n_heads=4)
    g = torch.randn(3, 16)
    H_text = torch.randn(3, 7, 16)
    mask = torch.ones(3, 7, dtype=torch.long)
    mask[0, 4:] = 0  # pad the first example
    context, attn = fusion(g, H_text, text_mask=mask)
    assert context.shape == (3, 16)
    assert attn.shape == (3, 4, 7)  # (batch, heads, seq_len)
    # attention on padded positions should be ~0 after softmax masking
    assert attn[0, :, 4:].sum().item() < 1e-4


def test_gnn_bert_fusion_trains_one_step():
    from fusion_model import GnnBertFusion, multi_task_loss

    batch = _synthetic_graph_batch(n_graphs=3, feat_dim=8)
    model = GnnBertFusion(
        num_tags=5,
        gnn_kwargs=dict(in_dim=8, hidden_dim=16, num_layers=2),
        d_model=16, n_heads=4, predict_emotion=True,
    )
    texts = ["genre: rock.", "genre: classical.", "instrument: guitar."]
    tokens = model.bert.tokenize(texts, max_length=12)
    out = model(batch.x, batch.edge_index, batch.batch, tokens["input_ids"], tokens["attention_mask"])

    assert out["tag_logits"].shape == (3, 5)
    assert out["emotion"].shape == (3, 2)

    y = torch.randint(0, 2, (3, 5)).float()
    va = torch.rand(3, 2)
    loss, parts = multi_task_loss(out["tag_logits"], y, out["emotion"], va)
    assert torch.isfinite(loss)
    loss.backward()
    gnn_grads = [p.grad for p in model.gnn.parameters() if p.grad is not None]
    bert_grads = [p.grad for p in model.bert.parameters() if p.grad is not None]
    assert len(gnn_grads) > 0 and len(bert_grads) > 0
    assert all(torch.isfinite(g).all() for g in gnn_grads)


def test_early_concat_ablation_trains_one_step():
    from fusion_model import EarlyConcatFusion

    batch = _synthetic_graph_batch(n_graphs=3, feat_dim=8)
    model = EarlyConcatFusion(num_tags=5, gnn_kwargs=dict(in_dim=8, hidden_dim=16, num_layers=2), d_model=16)
    tokens = model.bert.tokenize(["rock guitar", "classical strings", "calm piano"], max_length=12)
    out = model(batch.x, batch.edge_index, batch.batch, tokens["input_ids"], tokens["attention_mask"])
    assert out["tag_logits"].shape == (3, 5)
    out["tag_logits"].sum().backward()
    assert any(p.grad is not None for p in model.parameters())


# ---------- Task 4: contrastive.py ----------

def test_contrastive_dual_encoder_and_info_nce():
    from contrastive import ContrastiveDualEncoder, info_nce_loss, recall_at_k

    batch = _synthetic_graph_batch(n_graphs=4, feat_dim=8)
    model = ContrastiveDualEncoder(gnn_kwargs=dict(in_dim=8, hidden_dim=16, num_layers=2), d_model=16)
    texts = ["genre: rock.", "genre: classical.", "instrument: guitar.", "mood: calm."]
    tokens = model.bert.tokenize(texts, max_length=12)

    g, t = model(batch.x, batch.edge_index, batch.batch, tokens["input_ids"], tokens["attention_mask"])
    assert g.shape == (4, 16) and t.shape == (4, 16)
    # embeddings should be L2-normalized
    assert torch.allclose(g.norm(dim=-1), torch.ones(4), atol=1e-4)

    loss, logits = info_nce_loss(g, t, temperature=0.07)
    assert torch.isfinite(loss)
    loss.backward()

    metrics = recall_at_k(logits.detach(), ks=(1, 2))
    assert set(metrics.keys()) == {"g2t_R@1", "g2t_R@2", "t2g_R@1", "t2g_R@2"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    loss2, _ = info_nce_loss(g.detach(), t.detach(), model.temperature)
    model.zero_grad(); loss2.backward()
    assert model.log_temperature.grad is not None


# ---------- graph_dataset.py collate logic ----------

def test_collate_fn_batches_graphs_and_text():
    from torch_geometric.data import Data

    from graph_dataset import make_collate_fn

    rng = np.random.default_rng(0)
    examples = []
    texts = ["genre: rock.", "genre: jazz.", "instrument: piano."]
    for i in range(3):
        feats = rng.normal(size=(5, 8)).astype("float32")
        d = Data(x=torch.tensor(feats), edge_index=torch.zeros((2, 0), dtype=torch.long))
        d.y = torch.randint(0, 2, (5,)).float()
        d.text = texts[i]
        examples.append(d)

    tok = _tiny_tokenizer()
    collate = make_collate_fn(tok, max_length=12)
    graph_batch, enc, ys = collate(examples)

    assert graph_batch.x.shape == (15, 8)   # 3 graphs x 5 nodes
    assert enc["input_ids"].shape[0] == 3
    assert ys.shape == (3, 5)


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-v"]))
