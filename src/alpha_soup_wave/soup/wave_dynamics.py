"""Wave dynamics and transmission models for network-string coupling."""

from __future__ import annotations

import numpy as np

from alpha_soup_wave.soup.m1_strings import StringPopulation, StringSpec


def string_wave_speed(tension: float, density: float) -> float:
    return float(np.sqrt(tension / density))


def damping_length(spec: StringSpec) -> float:
    v = string_wave_speed(spec.tension, spec.density)
    return float(v / max(spec.damping_rate, 1e-6))


def string_modes(spec: StringSpec, mode_count: int) -> np.ndarray:
    v = string_wave_speed(spec.tension, spec.density)
    if spec.boundary == "free":
        n = np.arange(1, mode_count + 1) - 0.5
    else:
        n = np.arange(1, mode_count + 1)
    return n * np.pi * v / spec.length


def string_linewidths(spec: StringSpec, modes: np.ndarray) -> np.ndarray:
    return spec.damping_rate * (1.0 + 0.1 * modes / (modes.max() + 1e-6))


def string_mode_spacing(population: StringPopulation, mode_count: int) -> np.ndarray:
    spacings = []
    for spec in population.specs:
        modes = string_modes(spec, mode_count)
        if len(modes) > 1:
            spacings.append(float(np.median(np.diff(modes))))
    return np.array(spacings)


def lorentzian_density(omega_grid: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> np.ndarray:
    density = np.zeros_like(omega_grid)
    for center, width in zip(centers, widths, strict=True):
        gamma = max(width, 1e-6)
        density += gamma**2 / ((omega_grid - center) ** 2 + gamma**2)
    if density.max() > 0:
        density /= density.max()
    return density


