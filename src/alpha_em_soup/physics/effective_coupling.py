"""Compute effective couplings across layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from alpha_em_soup.physics.scattering import compute_power
from alpha_em_soup.physics.wave_operator import graph_laplacian_from_adj, helmholtz_solve
from alpha_em_soup.rng import seeded_rng


@dataclass(frozen=True)
class CouplingResult:
    g21_eff: float
    g23_eff: float
    q_ratio: float
    omega_grid: np.ndarray
    g21_vals: np.ndarray
    g23_vals: np.ndarray
    power_in: np.ndarray


def _base_adjacency(nodes: list[int], m3_edges, m2_nodes: set[int], j3: float, j2: float, j23: float) -> sp.csr_matrix:
    index = {node: idx for idx, node in enumerate(nodes)}
    rows = []
    cols = []
    data = []
    for u, v in m3_edges:
        if u in m2_nodes and v in m2_nodes:
            weight = j2
        elif u in m2_nodes or v in m2_nodes:
            weight = j23
        else:
            weight = j3
        i = index[u]
        j = index[v]
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([weight, weight])
    size = len(nodes)
    return sp.csr_matrix((data, (rows, cols)), shape=(size, size))


def _m1_edges_to_adj(nodes: list[int], m1_edges, k_scale: float, omega: float,
                     attenuation_length: float) -> sp.csr_matrix:
    index = {node: idx for idx, node in enumerate(nodes)}
    rows = []
    cols = []
    data = []
    for u, v, length in m1_edges:
        weight = float(np.cos(k_scale * omega * length) * np.exp(-length / max(attenuation_length, 1e-6)))
        i = index[u]
        j = index[v]
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([weight, weight])
    size = len(nodes)
    return sp.csr_matrix((data, (rows, cols)), shape=(size, size))


def _drive_vector(nodes: list[int], m2_nodes: set[int], drive_fraction: float,
                  drive_strength: float, seed: int) -> np.ndarray:
    rng = seeded_rng(seed)
    drive = np.zeros(len(nodes), dtype=complex)
    m2_indices = [idx for idx, node in enumerate(nodes) if node in m2_nodes]
    if not m2_indices:
        return drive
    count = max(1, int(len(m2_indices) * drive_fraction))
    chosen = rng.choice(m2_indices, size=count, replace=False)
    drive[chosen] = drive_strength
    return drive


def compute_effective_coupling(nodes: list[int], m3_edges, m1_edges, m2_nodes: set[int],
                               damping: np.ndarray, omega_grid: np.ndarray, k_scale: float,
                               attenuation_length: float, j3: float, j2: float, j23: float,
                               drive_fraction: float, drive_strength: float, seed: int,
                               m1_indices: np.ndarray, m3_indices: np.ndarray) -> CouplingResult:
    base_adj = _base_adjacency(nodes, m3_edges, m2_nodes, j3, j2, j23)
    g21_vals = []
    g23_vals = []
    power_in = []
    drive = _drive_vector(nodes, m2_nodes, drive_fraction, drive_strength, seed)
    for omega in omega_grid:
        adj = base_adj + _m1_edges_to_adj(nodes, m1_edges, k_scale, omega, attenuation_length)
        laplacian = graph_laplacian_from_adj(adj)
        psi = helmholtz_solve(laplacian, damping, omega, drive)
        p_total = compute_power(omega, damping, psi, np.arange(len(nodes)))
        p_m1 = compute_power(omega, damping, psi, m1_indices)
        p_m3 = compute_power(omega, damping, psi, m3_indices)
        g21 = np.sqrt(p_m1 / p_total) if p_total > 0 else 0.0
        g23 = np.sqrt(p_m3 / p_total) if p_total > 0 else 0.0
        g21_vals.append(g21)
        g23_vals.append(g23)
        power_in.append(p_total)
    g21_vals = np.array(g21_vals)
    g23_vals = np.array(g23_vals)
    g21_eff = float(np.sqrt(np.mean(g21_vals**2))) if g21_vals.size else 0.0
    g23_eff = float(np.sqrt(np.mean(g23_vals**2))) if g23_vals.size else 0.0
    q_ratio = compute_q_ratio(g21_eff, g23_eff)
    return CouplingResult(
        g21_eff=g21_eff,
        g23_eff=g23_eff,
        q_ratio=q_ratio,
        omega_grid=omega_grid,
        g21_vals=g21_vals,
        g23_vals=g23_vals,
        power_in=np.array(power_in),
    )


def compute_q_ratio(g21_eff: float, g23_eff: float) -> float:
    return float(g23_eff / max(g21_eff**2, 1e-12))
