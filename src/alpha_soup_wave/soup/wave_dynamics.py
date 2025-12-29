"""Wave dynamics and transmission models for network-string coupling."""

from __future__ import annotations

import numpy as np

from alpha_soup_wave.soup.m1_strings import StringPopulation


def string_wave_speed(tension: float, density: float) -> float:
    return float(np.sqrt(tension / density))


def string_impedance(tension: float, density: float, damping: float, omega: float) -> float:
    base = np.sqrt(tension * density)
    return float(base * (1.0 + damping / (omega + 1e-6)))


def network_impedance(omega: float, eigenfrequencies: np.ndarray) -> float:
    omega_star = float(np.median(eigenfrequencies))
    spread = float(np.std(eigenfrequencies) + 1e-6)
    return float(1.0 / (1.0 + abs(omega - omega_star) / spread))


def transmission_coeff(z_net: float | np.ndarray, z_str: float | np.ndarray) -> np.ndarray:
    z_net_arr = np.asarray(z_net)
    z_str_arr = np.asarray(z_str)
    return 4.0 * z_net_arr * z_str_arr / ((z_net_arr + z_str_arr) ** 2)


def leakage_rate(
    population: StringPopulation,
    eigenfrequencies: np.ndarray,
    omega_star: float,
    weight_sigma: float = 0.3,
) -> float:
    omegas = np.linspace(0.5 * omega_star, 1.5 * omega_star, 64)
    weights = np.exp(-0.5 * ((omegas - omega_star) / weight_sigma) ** 2)
    weights /= weights.sum()
    total = 0.0
    for spec in population.specs:
        z_str = np.array([string_impedance(spec.tension, spec.density, spec.damping, o) for o in omegas])
        z_net = np.array([network_impedance(o, eigenfrequencies) for o in omegas])
        transmissions = transmission_coeff(z_net, z_str)
        total += float(np.sum(weights * transmissions))
    return float(total)


def string_mode_spacing(population: StringPopulation) -> np.ndarray:
    spacings = []
    for spec in population.specs:
        v = string_wave_speed(spec.tension, spec.density)
        spacings.append(np.pi * v / spec.length)
    return np.array(spacings)
