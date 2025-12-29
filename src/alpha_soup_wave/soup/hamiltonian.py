"""Network XY-like energy and sampling."""

from __future__ import annotations

import numpy as np
import networkx as nx

from alpha_soup_wave.rng import make_rng


def network_energy(graph: nx.Graph, theta: np.ndarray, coupling_j: float, loop_penalty: float) -> float:
    energy = 0.0
    for i, j in graph.edges:
        energy -= coupling_j * np.cos(theta[i] - theta[j])
    if loop_penalty > 0:
        triangles = sum(nx.triangles(graph).values()) / 3
        energy += loop_penalty * triangles
    return float(energy)


def sample_phases(
    graph: nx.Graph,
    coupling_j: float,
    temperature: float,
    steps: int,
    seed: int | None,
) -> np.ndarray:
    rng = make_rng(seed)
    n = graph.number_of_nodes()
    theta = rng.uniform(0, 2 * np.pi, size=n)
    for _ in range(steps):
        node = rng.integers(0, n)
        proposal = theta[node] + rng.normal(scale=0.5)
        delta_e = 0.0
        for neighbor in graph.neighbors(node):
            delta_e += -coupling_j * (
                np.cos(proposal - theta[neighbor]) - np.cos(theta[node] - theta[neighbor])
            )
        if delta_e <= 0 or rng.random() < np.exp(-delta_e / temperature):
            theta[node] = proposal
    return theta
