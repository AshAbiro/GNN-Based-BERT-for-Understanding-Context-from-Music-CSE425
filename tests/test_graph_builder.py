import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graph_builder import build_segment_graph, build_chord_transition_graph, nx_to_edge_index, graph_coherence_score


def test_segment_graph_temporal_edges_always_present():
    feats = np.random.default_rng(0).normal(size=(5, 16)).astype("float32")
    G = build_segment_graph(feats, similarity_threshold=2.0)  # threshold above any real cosine sim -> no similarity edges
    # 4 consecutive temporal edges expected for 5 nodes, temporal_window=1
    assert G.number_of_edges() == 4
    for i in range(4):
        assert G.has_edge(i, i + 1)


def test_segment_graph_similarity_edges_appear_for_identical_vectors():
    feats = np.stack([np.ones(16), np.ones(16), -np.ones(16)]).astype("float32")
    G = build_segment_graph(feats, similarity_threshold=0.99)
    assert G.has_edge(0, 1)  # identical vectors -> cosine sim = 1.0
    assert not G.has_edge(0, 2) or G[0][2]["kind"] == "temporal"  # opposite vectors, only temporal if adjacent


def test_chord_transition_graph_counts():
    labels = [0, 1, 0, 1, 2]
    G = build_chord_transition_graph(labels)
    assert G.number_of_nodes() == 3
    assert G[0][1]["weight"] == 2.0
    assert G[1][0]["weight"] == 1.0
    assert G[1][2]["weight"] == 1.0


def test_edge_index_shape_matches_undirected_doubling():
    feats = np.random.default_rng(1).normal(size=(4, 8)).astype("float32")
    G = build_segment_graph(feats, similarity_threshold=2.0)
    ei = nx_to_edge_index(G)
    assert ei.shape[0] == 2
    assert ei.shape[1] == 2 * G.number_of_edges()


def test_coherence_score_bounds():
    feats = np.random.default_rng(2).normal(size=(6, 8)).astype("float32")
    G = build_segment_graph(feats, similarity_threshold=0.0)
    score = graph_coherence_score(G, tau=0.0)
    assert 0.0 <= score <= 1.0


if __name__ == "__main__":
    test_segment_graph_temporal_edges_always_present()
    test_segment_graph_similarity_edges_appear_for_identical_vectors()
    test_chord_transition_graph_counts()
    test_edge_index_shape_matches_undirected_doubling()
    test_coherence_score_bounds()
    print("all graph_builder tests passed")
