"""
Baselines required by the brief (Sec 8): must compare against >= 2.
  B1: majority-class / random tag predictor      -- numpy only, real & runnable now
  B2: CNN on mel-spectrogram (no graph, no text)  -- torch, needs audio
  B3: BERT-only                                    -- see bert_encoder.BertTagClassifier (Task 1)
  B4: PCA + MLP on hand-crafted audio features (optional)  -- sklearn, needs audio
"""
import numpy as np


class MajorityClassBaseline:
    """B1. Predicts each tag's marginal train-set frequency for every example (a real, non-cheating
    lower-bound baseline -- NOT the same as thresholding at 0.5 on training frequency)."""

    def fit(self, Y_train: np.ndarray):
        self.freq_ = Y_train.mean(axis=0)   # (num_tags,) probability per tag
        return self

    def predict_proba(self, n: int) -> np.ndarray:
        return np.tile(self.freq_, (n, 1))


class RandomBaseline:
    def __init__(self, num_tags: int, seed: int = 0):
        self.num_tags = num_tags
        self.rng = np.random.default_rng(seed)

    def predict_proba(self, n: int) -> np.ndarray:
        return self.rng.random((n, self.num_tags)).astype("float32")


# --- B2: CNN on mel-spectrograms (torch, needs audio) ---
def build_cnn_baseline(num_tags: int, n_mels: int = 128):
    """Simple 4-block CNN over a fixed-length log-mel patch, returned uncompiled (torch required)."""
    import torch.nn as nn

    class MelCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(128, num_tags)

        def forward(self, x):     # x: (B, 1, n_mels, T)
            h = self.conv(x).flatten(1)
            return self.head(h)

    return MelCNN()


# --- B4: PCA + MLP on hand-crafted features (sklearn, needs audio-derived features) ---
def pca_mlp_baseline(n_components: int = 50, hidden_layer_sizes=(128,)):
    from sklearn.decomposition import PCA
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.neural_network import MLPClassifier

    pca = PCA(n_components=n_components)
    clf = MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=200))
    return pca, clf
