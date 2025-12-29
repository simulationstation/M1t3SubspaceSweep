"""Wave operator utilities."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx


def laplacian_matrix(graph: nx.Graph, weight_attr: str = "weight") -> sp.csr_matrix:
    return graph_laplacian_from_adj(nx_to_sparse(graph, weight_attr))


def nx_to_sparse(graph, weight_attr: str) -> sp.csr_matrix:
    nodes = list(graph.nodes)
    index = {node: idx for idx, node in enumerate(nodes)}
    rows = []
    cols = []
    data = []
    for u, v, attrs in graph.edges(data=True):
        weight = float(attrs.get(weight_attr, 1.0))
        i = index[u]
        j = index[v]
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([weight, weight])
    size = len(nodes)
    return sp.csr_matrix((data, (rows, cols)), shape=(size, size))


def graph_laplacian_from_adj(adj: sp.csr_matrix) -> sp.csr_matrix:
    degrees = np.array(adj.sum(axis=1)).ravel()
    return sp.diags(degrees) - adj


def helmholtz_solve(laplacian: sp.csr_matrix, damping: np.ndarray, omega: float,
                    drive: np.ndarray) -> np.ndarray:
    size = laplacian.shape[0]
    matrix = laplacian + 1j * omega * sp.diags(damping) - (omega**2) * sp.eye(size, format="csr")
    return spla.spsolve(matrix, drive)
