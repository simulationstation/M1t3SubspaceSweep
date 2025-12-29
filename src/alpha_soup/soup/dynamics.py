"""Dynamics for XY-like phases."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alpha_soup.config import Couplings, DynamicsConfig
from alpha_soup.rng import make_rng


@dataclass
class SoupState:
    theta: np.ndarray


def initialize_state(n_nodes: int, seed: int | None) -> SoupState:
    rng = make_rng(seed)
    theta = rng.uniform(0.0, 2 * np.pi, size=n_nodes)
    return SoupState(theta=theta)


def _edge_weight(deg_i: int, deg_j: int, couplings: Couplings) -> float:
    if deg_i == deg_j == 3:
        return couplings.j3
    if deg_i == deg_j == 2:
        return couplings.j2
    if deg_i == deg_j == 1:
        return couplings.j1
    if 3 in (deg_i, deg_j):
        return couplings.g23
    if {deg_i, deg_j} == {1, 2}:
        return couplings.g21
    return 0.0


def _compute_grad(theta: np.ndarray, edges: list[tuple[int, int]], degree_labels: dict[int, int], couplings: Couplings) -> np.ndarray:
    grad = np.zeros_like(theta)
    for i, j in edges:
        w = _edge_weight(degree_labels[i], degree_labels[j], couplings)
        if w == 0.0:
            continue
        diff = theta[i] - theta[j]
        grad[i] += w * np.sin(diff)
        grad[j] -= w * np.sin(diff)
    return grad


def evolve(state: SoupState, edges: list[tuple[int, int]], degree_labels: dict[int, int], couplings: Couplings, cfg: DynamicsConfig, seed: int | None) -> SoupState:
    rng = make_rng(seed)
    theta = state.theta.copy()
    for _ in range(cfg.steps):
        grad = _compute_grad(theta, edges, degree_labels, couplings)
        noise = rng.normal(0.0, 1.0, size=theta.shape)
        theta = theta - cfg.step_size * grad + np.sqrt(2 * cfg.noise * cfg.temperature * cfg.step_size) * noise
        theta %= 2 * np.pi
    return SoupState(theta=theta)
