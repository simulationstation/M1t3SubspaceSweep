"""M1 wave paths with lengths and loopbacks."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from alpha_em_soup.rng import seeded_rng


@dataclass(frozen=True)
class M1Path:
    nodes: list[int]
    length: float
    anchor_a: int
    anchor_b: int | None
    loop: bool


@dataclass(frozen=True)
class M1WavePaths:
    graph: nx.Graph
    paths: list[M1Path]
    anchor_nodes: list[int]


def _sample_length(rng: np.random.Generator, mean: float, sigma: float, min_len: float = 1.0) -> float:
    value = rng.normal(mean, sigma)
    return float(max(value, min_len))


def build_m1_wavepaths(anchor_nodes: list[int], seed: int, path_count: int, length_mean: float,
                        length_sigma: float, loop_fraction: float, loopback_fraction: float,
                        max_path_edges: int) -> M1WavePaths:
    rng = seeded_rng(seed)
    graph = nx.Graph()
    paths: list[M1Path] = []
    node_counter = max(anchor_nodes) + 1 if anchor_nodes else 0
    graph.add_nodes_from(anchor_nodes)
    for _ in range(path_count):
        anchor_a = int(rng.choice(anchor_nodes)) if anchor_nodes else 0
        loop = rng.random() < loop_fraction
        loopback = rng.random() < loopback_fraction
        anchor_b = None
        if loopback and anchor_nodes:
            anchor_b = int(rng.choice(anchor_nodes))
        edges = int(rng.integers(2, max_path_edges + 1))
        length = _sample_length(rng, length_mean, length_sigma)
        segment_length = length / edges
        path_nodes = [anchor_a]
        for _ in range(edges - 1):
            node_counter += 1
            path_nodes.append(node_counter)
            graph.add_node(node_counter)
        end_node = None
        if loopback and anchor_b is not None:
            end_node = anchor_b
        elif not loop:
            node_counter += 1
            end_node = node_counter
            graph.add_node(node_counter)
        if end_node is not None:
            path_nodes.append(end_node)
        for idx in range(len(path_nodes) - 1):
            graph.add_edge(path_nodes[idx], path_nodes[idx + 1], length=segment_length)
        if loop:
            graph.add_edge(path_nodes[-1], anchor_a, length=segment_length)
        paths.append(M1Path(nodes=path_nodes, length=length, anchor_a=anchor_a, anchor_b=anchor_b, loop=loop))
    return M1WavePaths(graph=graph, paths=paths, anchor_nodes=anchor_nodes)
