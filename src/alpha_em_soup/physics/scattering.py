"""Scattering and power computation."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def compute_power(omega: float, damping: np.ndarray, psi: np.ndarray, indices: np.ndarray) -> float:
    if indices.size == 0:
        return 0.0
    return float((omega**2) * np.sum(damping[indices] * np.abs(psi[indices]) ** 2))


def green_diagonal(matrix: sp.csr_matrix, indices: np.ndarray) -> np.ndarray:
    diag = np.zeros(len(indices), dtype=complex)
    size = matrix.shape[0]
    for idx, node in enumerate(indices):
        basis = np.zeros(size, dtype=complex)
        basis[node] = 1.0
        diag[idx] = spla.spsolve(matrix, basis)[node]
    return diag
