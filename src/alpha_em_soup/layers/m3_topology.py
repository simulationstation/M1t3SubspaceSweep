"""M3 substrate generation."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from alpha_em_soup.rng import seeded_rng


@dataclass(frozen=True)
class M3Result:
    graph: nx.Graph
    short_cycle_ratio: float
    generator: str


def short_cycle_ratio(graph: nx.Graph, max_len: int = 4) -> float:
    cycles = nx.cycle_basis(graph)
    if not cycles:
        return 0.0
    short_cycles = [cycle for cycle in cycles if len(cycle) <= max_len]
    return len(short_cycles) / max(1, graph.number_of_nodes())


def _triangle_boosted(n3: int, seed: int, steps: int) -> nx.Graph:
    rng = seeded_rng(seed)
    graph = nx.random_regular_graph(3, n3, seed=rng)
    best_ratio = short_cycle_ratio(graph)
    for _ in range(steps):
        u, v = rng.choice(list(graph.edges))
        x, y = rng.choice(list(graph.edges))
        if len({u, v, x, y}) < 4:
            continue
        graph.remove_edge(u, v)
        graph.remove_edge(x, y)
        if graph.has_edge(u, x) or graph.has_edge(v, y):
            graph.add_edge(u, v)
            graph.add_edge(x, y)
            continue
        graph.add_edge(u, x)
        graph.add_edge(v, y)
        ratio = short_cycle_ratio(graph)
        if ratio < best_ratio:
            graph.remove_edge(u, x)
            graph.remove_edge(v, y)
            graph.add_edge(u, v)
            graph.add_edge(x, y)
        else:
            best_ratio = ratio
    return graph


def _geometric_cubic(n3: int, seed: int, radius: float) -> nx.Graph:
    rng = seeded_rng(seed)
    points = rng.random((n3, 2))
    graph = nx.random_regular_graph(3, n3, seed=rng)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    for _ in range(n3 * 3):
        (u, v) = rng.choice(list(graph.edges))
        (x, y) = rng.choice(list(graph.edges))
        if len({u, v, x, y}) < 4:
            continue
        current = distances[u, v] + distances[x, y]
        proposed = distances[u, x] + distances[v, y]
        if proposed < current and not graph.has_edge(u, x) and not graph.has_edge(v, y):
            graph.remove_edge(u, v)
            graph.remove_edge(x, y)
            graph.add_edge(u, x)
            graph.add_edge(v, y)
    return graph


def _planted_short_cycle(n3: int, seed: int, cycle_count: int) -> nx.Graph:
    rng = seeded_rng(seed)
    graph = nx.random_regular_graph(3, n3, seed=rng)
    best_ratio = short_cycle_ratio(graph)
    for _ in range(cycle_count * 4):
        (u, v) = rng.choice(list(graph.edges))
        (x, y) = rng.choice(list(graph.edges))
        if len({u, v, x, y}) < 4:
            continue
        graph.remove_edge(u, v)
        graph.remove_edge(x, y)
        if graph.has_edge(u, x) or graph.has_edge(v, y):
            graph.add_edge(u, v)
            graph.add_edge(x, y)
            continue
        graph.add_edge(u, x)
        graph.add_edge(v, y)
        ratio = short_cycle_ratio(graph)
        if ratio < best_ratio:
            graph.remove_edge(u, x)
            graph.remove_edge(v, y)
            graph.add_edge(u, v)
            graph.add_edge(x, y)
        else:
            best_ratio = ratio
    return graph


def generate_m3_graph(n3: int, seed: int, generator: str, min_short_cycle_ratio: float,
                      triangle_boost_steps: int, geometric_radius: float, planted_cycle_count: int) -> M3Result:
    creators = {
        "triangle_boosted": lambda: _triangle_boosted(n3, seed, triangle_boost_steps),
        "geometric": lambda: _geometric_cubic(n3, seed, geometric_radius),
        "planted": lambda: _planted_short_cycle(n3, seed, planted_cycle_count),
    }
    if generator not in creators:
        raise ValueError(f"Unknown M3 generator '{generator}'.")
    graph = creators[generator]()
    ratio = short_cycle_ratio(graph)
    if ratio < min_short_cycle_ratio:
        graph = _triangle_boosted(n3, seed + 11, triangle_boost_steps)
        ratio = short_cycle_ratio(graph)
    return M3Result(graph=graph, short_cycle_ratio=ratio, generator=generator)
