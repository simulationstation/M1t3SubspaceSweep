"""Observable calculations for emergent alpha estimators."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import networkx as nx


@dataclass
class ObservableBundle:
    spectral_gap_m2: float
    spectral_radius_m3: float
    corr_length_m2: float
    stiffness_m2: float
    stiffness_m3: float
    overlap_23: float
    stability_m2: float


def _laplacian_spectrum(graph: nx.Graph) -> np.ndarray:
    lap = nx.laplacian_matrix(graph).toarray()
    eigvals = np.linalg.eigvalsh(lap)
    return np.sort(eigvals)


def _spectral_gap(graph: nx.Graph) -> float:
    eigvals = _laplacian_spectrum(graph)
    if eigvals.size < 2:
        return 0.0
    return float(eigvals[1])


def _spectral_radius(graph: nx.Graph) -> float:
    eigvals = _laplacian_spectrum(graph)
    if eigvals.size == 0:
        return 0.0
    return float(np.max(eigvals))


def _correlation_length(graph: nx.Graph, theta: np.ndarray, max_dist: int = 4) -> float:
    if graph.number_of_nodes() < 2:
        return 0.0
    distances = dict(nx.all_pairs_shortest_path_length(graph, cutoff=max_dist))
    by_dist: dict[int, list[float]] = {}
    for i, dist_map in distances.items():
        for j, dist in dist_map.items():
            if i >= j or dist == 0:
                continue
            corr = np.cos(theta[i] - theta[j])
            by_dist.setdefault(dist, []).append(corr)
    if not by_dist:
        return 0.0
    xs = []
    ys = []
    for dist, vals in by_dist.items():
        mean_corr = float(np.mean(vals))
        if mean_corr <= 0:
            continue
        xs.append(dist)
        ys.append(np.log(mean_corr))
    if len(xs) < 2:
        return 0.0
    slope, _ = np.polyfit(xs, ys, 1)
    if slope >= 0:
        return 0.0
    return float(-1.0 / slope)


def _stiffness(graph: nx.Graph, theta: np.ndarray) -> float:
    if graph.number_of_edges() == 0:
        return 0.0
    diffs = []
    for i, j in graph.edges:
        diffs.append(np.cos(theta[i] - theta[j]))
    return float(np.mean(diffs))


def _overlap(graph_a: nx.Graph, graph_b: nx.Graph) -> float:
    if graph_a.number_of_nodes() < 2 or graph_b.number_of_nodes() < 2:
        return 0.0
    la = nx.laplacian_matrix(graph_a).toarray()
    lb = nx.laplacian_matrix(graph_b).toarray()
    ea = np.linalg.eigh(la)[1][:, 0]
    eb = np.linalg.eigh(lb)[1][:, 0]
    return float(np.abs(np.dot(ea, eb)) / (np.linalg.norm(ea) * np.linalg.norm(eb)))


def compute_observables(graph: nx.Graph, m2_nodes: list[int], degree_labels: dict[int, int], theta: np.ndarray) -> ObservableBundle:
    if m2_nodes:
        sub_m2 = graph.subgraph(m2_nodes).copy()
    else:
        sub_m2 = nx.Graph()
    m3_nodes = [node for node, deg in degree_labels.items() if deg == 3]
    sub_m3 = graph.subgraph(m3_nodes).copy() if m3_nodes else nx.Graph()

    spectral_gap_m2 = _spectral_gap(sub_m2)
    spectral_radius_m3 = _spectral_radius(sub_m3)
    corr_length_m2 = _correlation_length(sub_m2, theta)
    stiffness_m2 = _stiffness(sub_m2, theta)
    stiffness_m3 = _stiffness(sub_m3, theta)
    overlap_23 = _overlap(sub_m2, sub_m3)
    stability_m2 = float(np.var(theta[m2_nodes])) if m2_nodes else 0.0

    return ObservableBundle(
        spectral_gap_m2=spectral_gap_m2,
        spectral_radius_m3=spectral_radius_m3,
        corr_length_m2=corr_length_m2,
        stiffness_m2=stiffness_m2,
        stiffness_m3=stiffness_m3,
        overlap_23=overlap_23,
        stability_m2=stability_m2,
    )
