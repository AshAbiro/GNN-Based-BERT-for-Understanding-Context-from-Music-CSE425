"""
Music structure graph construction (Task 2/3/4).

Two graph types, matching the project brief (Sec. 3, "Graph construction"):
  - chord-transition graph: nodes = unique chords observed in the track,
    edges = observed transitions weighted by count.
  - segment graph: nodes = time segments, edges = temporal adjacency +
    cosine similarity of segment features above a threshold tau.

Pure numpy/networkx -- no audio or torch needed, so this module is fully
testable in any environment (see tests/test_graph_builder.py).
"""
from collections import Counter
from typing import List, Tuple

import networkx as nx
import numpy as np


def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1e-8
    Xn = X / norm
    return Xn @ Xn.T


def build_segment_graph(features: np.ndarray, similarity_threshold: float = 0.85, temporal_window: int = 1) -> nx.Graph:
    """
    features: (n_segments, feat_dim) mean-pooled per-segment features (e.g. mel).
    Edges: consecutive segments (within `temporal_window`) always connected;
    additionally, any pair with cosine similarity > similarity_threshold.
    """
    n = features.shape[0]
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, feat=features[i])

    for i in range(n):
        for j in range(i + 1, min(i + 1 + temporal_window, n)):
            G.add_edge(i, j, kind="temporal", weight=1.0)

    if n > 1:
        sim = cosine_sim_matrix(features)
        ii, jj = np.where(np.triu(sim, k=1) > similarity_threshold)
        for i, j in zip(ii.tolist(), jj.tolist()):
            if G.has_edge(i, j):
                G[i][j]["kind"] = "temporal+similarity"
            else:
                G.add_edge(i, j, kind="similarity", weight=float(sim[i, j]))
    return G


def build_chord_transition_graph(chord_labels: List[int]) -> nx.DiGraph:
    """
    chord_labels: sequence of chord/pitch-class ids per segment (see
    audio_features.chroma_to_chord_labels). Nodes = unique chords observed;
    directed edges = observed transitions, weight = transition count.
    """
    G = nx.DiGraph()
    unique = sorted(set(chord_labels))
    for c in unique:
        G.add_node(c)
    counts = Counter(zip(chord_labels[:-1], chord_labels[1:]))
    for (a, b), w in counts.items():
        G.add_edge(a, b, weight=float(w))
    return G


def nx_to_edge_index(G: nx.Graph) -> np.ndarray:
    """networkx graph -> (2, num_edges) edge_index array (both directions for undirected G),
    ready to hand to torch.tensor(...) for a torch_geometric.data.Data object."""
    edges = list(G.edges())
    if not edges:
        return np.zeros((2, 0), dtype="int64")
    src, dst = zip(*edges)
    src = list(src) + list(dst)
    dst = list(dst) + list(src[: len(edges)]) if not G.is_directed() else list(dst)
    if not G.is_directed():
        src_full = [e[0] for e in edges] + [e[1] for e in edges]
        dst_full = [e[1] for e in edges] + [e[0] for e in edges]
        return np.array([src_full, dst_full], dtype="int64")
    return np.array([[e[0] for e in edges], [e[1] for e in edges]], dtype="int64")


def graph_coherence_score(G: nx.Graph, tau: float = 0.85) -> float:
    """Sgraph metric from the brief: fraction of edges whose endpoint features
    are similar (cosine sim > tau). Only meaningful for similarity-typed edges."""
    if G.number_of_edges() == 0:
        return 0.0
    feats = nx.get_node_attributes(G, "feat")
    hits = 0
    total = 0
    for u, v in G.edges():
        if u not in feats or v not in feats:
            continue
        total += 1
        fu, fv = feats[u], feats[v]
        denom = (np.linalg.norm(fu) * np.linalg.norm(fv)) or 1e-8
        sim = float(np.dot(fu, fv) / denom)
        if sim > tau:
            hits += 1
    return hits / total if total else 0.0
