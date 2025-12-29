"""M2 interface detection or construction."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from alpha_em_soup.rng import seeded_rng


@dataclass(frozen=True)
class M2Interface:
    nodes: set[int]
    mode: str
    coherence_mean: float
    loopiness_mean: float
    stable: bool


def _node_loopiness(graph: nx.Graph) -> dict[int, float]:
    clustering = nx.clustering(graph)
    loopiness = {}
    cycles = nx.cycle_basis(graph)
    cycle_count = {node: 0 for node in graph.nodes}
    for cycle in cycles:
        for node in cycle:
            cycle_count[node] += 1
    for node in graph.nodes:
        loopiness[node] = clustering[node] + 0.05 * cycle_count[node]
    return loopiness


def detect_m2(graph: nx.Graph, coherence_threshold: float, loopiness_threshold: float,
              min_fraction: float) -> M2Interface:
    clustering = nx.clustering(graph)
    loopiness = _node_loopiness(graph)
    candidates = [
        node
        for node in graph.nodes
        if clustering[node] >= coherence_threshold and loopiness[node] >= loopiness_threshold
    ]
    scores = {node: clustering[node] + loopiness[node] for node in graph.nodes}
    min_size = max(1, int(min_fraction * graph.number_of_nodes()))
    stable = len(candidates) >= min_size
    if len(candidates) < min_size:
        ranked = sorted(scores, key=scores.get, reverse=True)
        candidates = ranked[:min_size]
    coherence_vals = [clustering[node] for node in candidates]
    loop_vals = [loopiness[node] for node in candidates]
    return M2Interface(
        nodes=set(candidates),
        mode="emergent",
        coherence_mean=float(np.mean(coherence_vals)) if coherence_vals else 0.0,
        loopiness_mean=float(np.mean(loop_vals)) if loop_vals else 0.0,
        stable=stable,
    )


def construct_m2(graph: nx.Graph, seed: int, min_fraction: float) -> M2Interface:
    rng = seeded_rng(seed)
    n_target = max(1, int(min_fraction * graph.number_of_nodes()))
    start = int(rng.integers(0, graph.number_of_nodes()))
    nodes = {start}
    frontier = [start]
    while len(nodes) < n_target and frontier:
        current = frontier.pop(0)
        neighbors = list(graph.neighbors(current))
        rng.shuffle(neighbors)
        for nbr in neighbors:
            if nbr not in nodes:
                nodes.add(nbr)
                frontier.append(nbr)
            if len(nodes) >= n_target:
                break
    subgraph = graph.subgraph(nodes)
    clustering = nx.clustering(subgraph)
    coherence_mean = float(np.mean(list(clustering.values()))) if clustering else 0.0
    loopiness_mean = float(np.mean(list(clustering.values()))) if clustering else 0.0
    return M2Interface(nodes=set(nodes), mode="constructed", coherence_mean=coherence_mean,
                       loopiness_mean=loopiness_mean, stable=True)


def build_m2(graph: nx.Graph, seed: int, mode: str, coherence_threshold: float,
             loopiness_threshold: float, min_fraction: float) -> M2Interface:
    if mode == "constructed":
        return construct_m2(graph, seed, min_fraction)
    return detect_m2(graph, coherence_threshold, loopiness_threshold, min_fraction)
