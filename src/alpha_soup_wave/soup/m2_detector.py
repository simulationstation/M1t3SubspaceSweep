"""Detect emergent M2 subset based on coherence and loopiness."""

from __future__ import annotations

import numpy as np
import networkx as nx


def coherence_values(graph: nx.Graph, theta: np.ndarray) -> dict[int, float]:
    values: dict[int, float] = {}
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            values[node] = 0.0
            continue
        phases = np.exp(1j * (theta[neighbors] - theta[node]))
        values[node] = float(np.abs(phases.mean()))
    return values


def loopiness_values(graph: nx.Graph) -> dict[int, float]:
    clustering = nx.clustering(graph)
    return {node: float(val) for node, val in clustering.items()}


def detect_m2(
    graph: nx.Graph,
    theta: np.ndarray,
    coherence_threshold: float,
    loopiness_threshold: float,
) -> set[int]:
    coherence = coherence_values(graph, theta)
    loopiness = loopiness_values(graph)
    return {
        node
        for node in graph.nodes
        if coherence[node] >= coherence_threshold and loopiness[node] >= loopiness_threshold
    }
