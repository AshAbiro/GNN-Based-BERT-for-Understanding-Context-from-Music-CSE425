"""Central configuration. Edit values here or override via config.yaml + --config flag."""
import os
from dataclasses import dataclass, field
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class DataConfig:
    # --- primary dataset: LP-MusicCaps-MTT (real captions over MagnaTagATune audio) ---
    dataset: str = "lp_musiccaps"             # lp_musiccaps (recommended, real captions) | mtg_jamendo (legacy, pseudo-text)
    lp_musiccaps_dir: str = os.path.join(ROOT, "data", "lp_musiccaps")
    text_field: str = "caption_writing"       # caption_writing | caption_summary | caption_paraphrase
                                                # NEVER caption_attribute_prediction -- dataset authors advise against it
    tag_field: str = "tag_top50"              # tag_top50 | tag_top188
    audio_dir: str = os.path.join(ROOT, "data", "raw", "audio")          # populated by scripts/download_audio.sh (MagnaTagATune mp3s)
    processed_dir: str = os.path.join(ROOT, "data", "processed")
    sample_rate: int = 22050
    segment_seconds: float = 5.0
    n_mels: int = 128
    n_chroma: int = 12
    n_mfcc: int = 20
    node_feature_type: str = "mfcc_chroma"  # 20 MFCC + 12 chroma features
    similarity_threshold: float = 0.85        # cosine sim threshold (tau) for segment-similarity edges
    max_audio_samples: int = 0               # 0 = full split; positive = reproducible pilot subset

    # --- legacy dataset: MTG-Jamendo (no captions -- tag-derived pseudo-text, see data.py) ---
    data_dir: str = os.path.join(ROOT, "data")
    split_dir: str = os.path.join(ROOT, "data", "splits", "split-0")
    tag_category: str = "moodtheme"
    text_tag_categories: List[str] = field(default_factory=lambda: ["genre", "instrument"])
    target_tag_category: str = "moodtheme"


@dataclass
class BertConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 64
    freeze_backbone: bool = False
    proj_dim: int = 256


@dataclass
class GnnConfig:
    in_dim: int = 32           # default: 20 MFCC + 12 chroma
    hidden_dim: int = 128
    out_dim: int = 256
    num_layers: int = 3
    conv_type: str = "sage"    # sage | gat
    gat_heads: int = 4
    dropout: float = 0.2


@dataclass
class FusionConfig:
    d_model: int = 256
    n_heads: int = 4
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 20
    lr: float = 2e-4
    bert_lr: float = 2e-5
    weight_decay: float = 1e-4
    emotion_alpha: float = 0.5      # weight for valence in multi-task loss
    emotion_beta: float = 0.5       # weight for arousal
    contrastive_temp: float = 0.07
    seed: int = 42
    device: str = "cuda"            # falls back to cpu automatically if unavailable
    num_workers: int = 4


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    bert: BertConfig = field(default_factory=BertConfig)
    gnn: GnnConfig = field(default_factory=GnnConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: str = None) -> Config:
    cfg = Config()
    if path and os.path.exists(path):
        import yaml
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        for section, values in overrides.items():
            sub = getattr(cfg, section, None)
            if sub is None:
                continue
            for k, v in (values or {}).items():
                if hasattr(sub, k):
                    setattr(sub, k, v)
    return cfg
