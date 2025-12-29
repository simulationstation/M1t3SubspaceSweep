"""Generate backbone M3 network graphs."""

from __future__ import annotations

import networkx as nx

from alpha_soup_wave.rng import make_rng


def generate_m3_graph(n3: int, seed: int | None, degree2_fraction: float = 0.0) -> nx.Graph:
    rng = make_rng(seed)
    graph = nx.random_regular_graph(3, n3, seed=rng)
    if degree2_fraction <= 0:
        return graph

    n_tail = max(1, int(n3 * degree2_fraction))
    anchor = rng.choice(list(graph.nodes))
    last = anchor
    for idx in range(n_tail):
        node = n3 + idx
        graph.add_node(node)
        graph.add_edge(last, node)
        last = node
    return graph
