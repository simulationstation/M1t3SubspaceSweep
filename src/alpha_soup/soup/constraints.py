"""Emergent interface criteria."""
from __future__ import annotations

import numpy as np
import networkx as nx


def coherence_scores(graph: nx.Graph, theta: np.ndarray) -> dict[int, float]:
    scores: dict[int, float] = {}
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            scores[node] = 0.0
            continue
        diffs = np.exp(1j * (theta[neighbors] - theta[node]))
        scores[node] = float(np.abs(np.mean(diffs)))
    return scores


def loop_density_scores(graph: nx.Graph) -> dict[int, float]:
    clustering = nx.clustering(graph)
    return {node: float(val) for node, val in clustering.items()}
