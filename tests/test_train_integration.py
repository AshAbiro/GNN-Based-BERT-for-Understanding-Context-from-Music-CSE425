"""
Integration test: src/train.py's real `run_task1` function, run against the
actual LP-MusicCaps-MTT data, exercising the full loop -- DataLoader,
tokenization, forward/backward, optimizer step, eval loop, metrics -- not
just individual module forward passes (see test_models_synthetic.py for
those). AutoModel/AutoTokenizer.from_pretrained are mocked to a tiny local
DistilBERT to keep the integration test fast and deterministic; everything else is the real
production code path on the real, full metadata dataset. It uses large
batches (few steps/epoch) -- this is a wiring smoke test, not a quality
benchmark, so predictions are near-random (single epoch, untrained tiny
model, mostly-[UNK] tokenization against the tiny test vocab).
"""
import os
import sys
from unittest.mock import patch

import pytest
from transformers import BertTokenizer, DistilBertConfig, DistilBertModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

VOCAB_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "tiny_vocab.txt")


def _tiny_backbone(*args, **kwargs):
    cfg = DistilBertConfig(vocab_size=120, dim=32, n_layers=2, n_heads=2, hidden_dim=64, max_position_embeddings=64)
    return DistilBertModel(cfg)


def _tiny_tokenizer(*args, **kwargs):
    return BertTokenizer(vocab_file=VOCAB_PATH)


def test_run_task1_full_real_loop():
    with patch("transformers.AutoModel.from_pretrained", side_effect=_tiny_backbone), \
         patch("transformers.AutoTokenizer.from_pretrained", side_effect=_tiny_tokenizer):
        from config import load_config
        from captions_data import LPMusicCapsDataset
        from train import run_task1

        cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
        cfg.train.epochs = 1
        cfg.train.batch_size = 2048     # few large batches -> fast test, still the real DataLoader/loop
        cfg.bert.proj_dim = 16
        cfg.bert.max_length = 16

        ds = LPMusicCapsDataset(data_dir=os.path.join(os.path.dirname(__file__), "..", "data", "lp_musiccaps"))
        history = run_task1(cfg, ds)

        assert len(history) == 1
        report = history[0]
        for key in ("macro_f1", "micro_f1", "mean_auc_pr", "epoch"):
            assert key in report
        assert 0.0 <= report["macro_f1"] <= 1.0
        assert 0.0 <= report["mean_auc_pr"] <= 1.0


if __name__ == "__main__":
    test_run_task1_full_real_loop()
    print("train.py Task 1 integration test passed")
