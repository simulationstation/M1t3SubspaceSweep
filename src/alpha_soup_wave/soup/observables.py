"""Compute internal scales and observables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import networkx as nx

from alpha_soup_wave.soup.m1_strings import StringPopulation
from alpha_soup_wave.soup.wave_dynamics import (
    leakage_rate,
    network_impedance,
    string_impedance,
    string_mode_spacing,
)


@dataclass(frozen=True)
class SoupObservables:
    omega_star: float
    z_net_star: float
    z_str_star: np.ndarray
    transmission_star: np.ndarray
    leak_rate: float
    delta_omega_net: float
    delta_omega_str: float


def _eigenfrequencies(graph: nx.Graph) -> np.ndarray:
    lap = nx.normalized_laplacian_matrix(graph).toarray()
    eigvals = np.linalg.eigvalsh(lap)
    eigvals = eigvals[eigvals > 1e-6]
    return np.sqrt(eigvals)


def compute_observables(graph: nx.Graph, population: StringPopulation) -> SoupObservables:
    eigenfrequencies = _eigenfrequencies(graph)
    omega_star = float(np.median(eigenfrequencies))
    z_net_star = float(network_impedance(omega_star, eigenfrequencies))
    z_str_star = np.array(
        [string_impedance(spec.tension, spec.density, spec.damping, omega_star) for spec in population.specs]
    )
    transmission_star = 4 * z_net_star * z_str_star / ((z_net_star + z_str_star) ** 2)
    leak_rate = leakage_rate(population, eigenfrequencies, omega_star)
    delta_omega_net = float(np.median(np.diff(np.sort(eigenfrequencies))))
    delta_omega_str = float(np.median(string_mode_spacing(population)))
    return SoupObservables(
        omega_star=omega_star,
        z_net_star=z_net_star,
        z_str_star=z_str_star,
        transmission_star=transmission_star,
        leak_rate=leak_rate,
        delta_omega_net=delta_omega_net,
        delta_omega_str=delta_omega_str,
    )
