"""Alpha hat estimators from internal scales."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from alpha_em_soup.physics.wave_operator import graph_laplacian_from_adj, nx_to_sparse


@dataclass(frozen=True)
class AlphaCandidates:
    alpha_hat_cov: float
    alpha_hat_ov: float
    alpha_hat_z: float


def _spectral_density(eigs: np.ndarray, omega_grid: np.ndarray, width: float = 0.2) -> np.ndarray:
    density = np.zeros_like(omega_grid)
    for eig in eigs:
        density += np.exp(-0.5 * ((omega_grid - np.sqrt(abs(eig))) / width) ** 2)
    if density.max() > 0:
        density /= density.max()
    return density


def estimate_alpha_candidates(m1_paths, m1_graph, m2_graph, omega_grid: np.ndarray,
                              attenuation_length: float, damping_m1: float, damping_m2: float,
                              anchors: list[int]) -> AlphaCandidates:
    cov_vals = []
    for path in m1_paths:
        spacing = np.pi / max(path.length, 1e-6)
        gamma = (1.0 / max(attenuation_length, 1e-6)) + damping_m1
        cov_vals.append(gamma / spacing)
    alpha_hat_cov = float(np.median(cov_vals)) if cov_vals else 0.0

    m1_adj = nx_to_sparse(m1_graph, "weight")
    m2_adj = nx_to_sparse(m2_graph, "weight")
    m1_lap = graph_laplacian_from_adj(m1_adj)
    m2_lap = graph_laplacian_from_adj(m2_adj)
    m1_eigs = _safe_eigs(m1_lap)
    m2_eigs = _safe_eigs(m2_lap)
    rho_m1 = _spectral_density(m1_eigs, omega_grid)
    rho_m2 = _spectral_density(m2_eigs, omega_grid)
    overlap = np.trapz(rho_m1 * rho_m2, omega_grid)
    norm = np.sqrt(np.trapz(rho_m1**2, omega_grid) * np.trapz(rho_m2**2, omega_grid))
    alpha_hat_ov = float(np.sqrt(overlap / norm)) if norm > 0 else 0.0

    alpha_hat_z = 0.0
    if anchors:
        omega = float(np.median(omega_grid))
        m1_impedance = []
        m2_impedance = []
        m1_nodes = list(m1_graph.nodes)
        m2_nodes = list(m2_graph.nodes)
        m1_index = {node: idx for idx, node in enumerate(m1_nodes)}
        m2_index = {node: idx for idx, node in enumerate(m2_nodes)}
        m1_drive = np.zeros(len(m1_nodes), dtype=complex)
        m2_drive = np.zeros(len(m2_nodes), dtype=complex)
        for anchor in anchors:
            if anchor not in m1_index or anchor not in m2_index:
                continue
            m1_drive[:] = 0.0
            m2_drive[:] = 0.0
            m1_drive[m1_index[anchor]] = 1.0
            m2_drive[m2_index[anchor]] = 1.0
            m1_mat = m1_lap + 1j * omega * sp.eye(m1_lap.shape[0]) * damping_m1 - (omega**2) * sp.eye(m1_lap.shape[0])
            m2_mat = m2_lap + 1j * omega * sp.eye(m2_lap.shape[0]) * damping_m2 - (omega**2) * sp.eye(m2_lap.shape[0])
            m1_resp = spla.spsolve(m1_mat, m1_drive)
            m2_resp = spla.spsolve(m2_mat, m2_drive)
            m1_impedance.append(1 / max(abs(m1_resp[m1_index[anchor]]), 1e-6))
            m2_impedance.append(1 / max(abs(m2_resp[m2_index[anchor]]), 1e-6))
        if m1_impedance and m2_impedance:
            ratio = np.median(np.array(m2_impedance) / np.array(m1_impedance))
            alpha_hat_z = float(ratio / (1 + ratio))

    return AlphaCandidates(
        alpha_hat_cov=float(np.clip(alpha_hat_cov, 0.0, 1.0)),
        alpha_hat_ov=float(np.clip(alpha_hat_ov, 0.0, 1.0)),
        alpha_hat_z=float(np.clip(alpha_hat_z, 0.0, 1.0)),
    )


def _safe_eigs(matrix: sp.csr_matrix) -> np.ndarray:
    size = matrix.shape[0]
    if size <= 2:
        return np.array([0.0])
    k = min(8, size - 2)
    return np.real(spla.eigs(matrix, k=k, return_eigenvectors=False))
