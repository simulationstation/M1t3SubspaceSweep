"""Emergent M2 identification."""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from alpha_soup.config import EmergentCriteria
from alpha_soup.soup.constraints import coherence_scores, loop_density_scores


@dataclass
class M2Detection:
    nodes: list[int]
    coherence: dict[int, float]
    loop_density: dict[int, float]


def detect_m2(graph: nx.Graph, theta: np.ndarray, criteria: EmergentCriteria) -> M2Detection:
    coherence = coherence_scores(graph, theta)
    loop_density = loop_density_scores(graph)
    nodes = [
        node
        for node in graph.nodes
        if coherence.get(node, 0.0) >= criteria.coherence_thresh
        and loop_density.get(node, 0.0) >= criteria.loop_thresh
    ]
    return M2Detection(nodes=nodes, coherence=coherence, loop_density=loop_density)
