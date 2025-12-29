"""Graph generation utilities."""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from alpha_soup.config import DegreeProfile, DecayConfig
from alpha_soup.rng import make_rng


@dataclass
class GraphBundle:
    graph: nx.Graph
    degree_labels: dict[int, int]


def _degree_counts(n_nodes: int, profile: DegreeProfile) -> tuple[int, int, int]:
    n3 = int(round(n_nodes * profile.frac_deg3))
    n2 = int(round(n_nodes * profile.frac_deg2))
    n1 = max(n_nodes - n3 - n2, 0)
    return n1, n2, n3


def generate_degree_graph(n_nodes: int, profile: DegreeProfile, seed: int | None) -> GraphBundle:
    rng = make_rng(seed)
    n1, n2, n3 = _degree_counts(n_nodes, profile)
    degrees = [1] * n1 + [2] * n2 + [3] * n3
    rng.shuffle(degrees)
    if sum(degrees) % 2 == 1:
        degrees[0] += 1
    graph = nx.configuration_model(degrees, seed=rng.integers(0, 1_000_000))
    graph = nx.Graph(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph = nx.convert_node_labels_to_integers(graph)
    degree_labels = {i: deg for i, deg in enumerate(degrees)}
    return GraphBundle(graph=graph, degree_labels=degree_labels)


def apply_decay(bundle: GraphBundle, decay: DecayConfig, seed: int | None) -> GraphBundle:
    if not decay.enabled:
        return bundle
    rng = make_rng(seed)
    labels = bundle.degree_labels.copy()
    for node, deg in labels.items():
        if deg == 3 and rng.random() < decay.rate_deg3_to_deg2:
            labels[node] = 2
        elif deg == 2 and rng.random() < decay.rate_deg2_to_deg1:
            labels[node] = 1
    return GraphBundle(graph=bundle.graph, degree_labels=labels)
