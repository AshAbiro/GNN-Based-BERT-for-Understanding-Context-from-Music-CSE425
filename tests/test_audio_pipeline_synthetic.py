"""
Real, executed test of the full audio pipeline: audio_features.py ->
graph_builder.py -> graph_dataset.py -> gnn_model.py, using librosa against
a genuine (synthetically generated, but real waveform) audio fixture --
not synthetic feature tensors, an actual wav file on disk.

tests/fixtures/synthetic_track.wav is 15s: a C-major chord (0-5s), a
different chord (5-10s), then C-major again (10-15s) -- built so the
pipeline's *output* is checkable against known ground truth (segments 0
and 2 should end up more similar to each other than to segment 1), not
just checkable for the right tensor shape.
"""
import os
import sys

import numpy as np
import torch
from torch_geometric.data import Batch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from audio_features import load_and_extract, segment_windows, segment_features, chroma_to_chord_labels
from graph_builder import build_segment_graph, build_chord_transition_graph, nx_to_edge_index
from graph_dataset import TrackGraphDataset

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "synthetic_track.wav")


def test_load_and_extract_real_audio():
    feat = load_and_extract(FIXTURE, sample_rate=22050, n_mels=128, n_chroma=12)
    assert feat.mel.shape[0] == 128
    assert feat.chroma.shape[0] == 12
    assert feat.mel.shape[1] == feat.chroma.shape[1] > 0


def test_segmentation_recovers_known_structure():
    """The fixture is C-major / other-chord / C-major in 5s blocks -- segment
    0 and 2 should be more similar to each other than either is to segment 1."""
    feat = load_and_extract(FIXTURE, sample_rate=22050, n_mels=128)
    windows = segment_windows(feat, segment_seconds=5.0)
    seg_feats = segment_features(feat, windows)
    assert seg_feats.shape[0] >= 3

    from graph_builder import cosine_sim_matrix

    sim = cosine_sim_matrix(seg_feats)
    assert sim[0, 2] > sim[0, 1]
    assert sim[0, 2] > sim[1, 2]


def test_mfcc_chroma_node_features():
    feat = load_and_extract(FIXTURE, sample_rate=22050, n_mfcc=20, n_chroma=12)
    windows = segment_windows(feat, segment_seconds=5.0)
    nodes = segment_features(feat, windows, feature_type="mfcc_chroma")
    assert nodes.shape[1] == 32
    assert np.isfinite(nodes).all()


def test_chord_labels_recover_repeat_structure():
    feat = load_and_extract(FIXTURE, sample_rate=22050)
    windows = segment_windows(feat, segment_seconds=5.0)
    labels = chroma_to_chord_labels(feat, windows)
    assert len(labels) >= 3
    assert labels[0] == labels[2]      # segments 0 and 2 are the same chord by construction
    assert labels[1] != labels[0]      # segment 1 is a different chord


def test_full_graph_construction_on_real_audio():
    feat = load_and_extract(FIXTURE, sample_rate=22050)
    windows = segment_windows(feat, segment_seconds=5.0)
    seg_feats = segment_features(feat, windows)
    G = build_segment_graph(seg_feats, similarity_threshold=0.7)
    assert G.number_of_nodes() == seg_feats.shape[0]
    assert G.number_of_edges() > 0

    labels = chroma_to_chord_labels(feat, windows)
    CG = build_chord_transition_graph(labels)
    assert CG.number_of_nodes() >= 2

    ei = nx_to_edge_index(G)
    assert ei.shape[0] == 2


def test_track_graph_dataset_end_to_end_into_gnn():
    """The full glue path: TrackGraphDataset (real audio -> Data) -> GnnEncoder forward pass."""
    from gnn_model import GnnEncoder

    audio_root, fname = os.path.split(FIXTURE)
    ds = TrackGraphDataset(paths=[fname], audio_root=audio_root, segment_seconds=5.0, n_mels=128)
    data = ds[0]
    assert data.x.shape[1] == 128
    assert data.x.shape[0] >= 3

    batch = Batch.from_data_list([data, data])   # batch of 2 (same track twice, cheap)
    enc = GnnEncoder(in_dim=128, hidden_dim=32, out_dim=16, num_layers=2)
    g, h = enc(batch.x, batch.edge_index, batch.batch)
    assert g.shape == (2, 16)
    assert torch.isfinite(g).all()


if __name__ == "__main__":
    test_load_and_extract_real_audio()
    test_segmentation_recovers_known_structure()
    test_chord_labels_recover_repeat_structure()
    test_full_graph_construction_on_real_audio()
    test_track_graph_dataset_end_to_end_into_gnn()
    print("all audio-pipeline tests passed")
