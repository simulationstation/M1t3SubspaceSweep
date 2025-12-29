"""Attach degree-1 strings (M1) to anchor nodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alpha_soup_wave.rng import make_rng


@dataclass(frozen=True)
class StringSpec:
    anchor: int
    length: float
    tension: float
    density: float
    damping_rate: float
    boundary: str


@dataclass(frozen=True)
class StringPopulation:
    specs: list[StringSpec]

    @property
    def anchors(self) -> list[int]:
        return [spec.anchor for spec in self.specs]


def _draw_lognormal(rng: np.random.Generator, mean: float, sigma: float, size: int) -> np.ndarray:
    mu = np.log(mean) - 0.5 * sigma**2
    return rng.lognormal(mu, sigma, size=size)


def attach_strings(
    nodes: list[int],
    anchor_fraction: float,
    length_mean: float,
    length_sigma: float,
    tension_mean: float,
    tension_sigma: float,
    density_mean: float,
    density_sigma: float,
    damping_mean: float,
    damping_sigma: float,
    boundary_choices: list[str],
    seed: int | None,
) -> StringPopulation:
    rng = make_rng(seed)
    n_attach = max(1, int(len(nodes) * anchor_fraction))
    anchors = rng.choice(nodes, size=n_attach, replace=False)
    lengths = _draw_lognormal(rng, length_mean, length_sigma, n_attach)
    tensions = _draw_lognormal(rng, tension_mean, tension_sigma, n_attach)
    densities = _draw_lognormal(rng, density_mean, density_sigma, n_attach)
    dampings = _draw_lognormal(rng, damping_mean, damping_sigma, n_attach)
    boundaries = rng.choice(boundary_choices, size=n_attach, replace=True)
    specs = [
        StringSpec(
            anchor=int(anchor),
            length=float(length),
            tension=float(tension),
            density=float(density),
            damping_rate=float(damping),
            boundary=str(boundary),
        )
        for anchor, length, tension, density, damping, boundary in zip(
            anchors, lengths, tensions, densities, dampings, boundaries, strict=True
        )
    ]
    return StringPopulation(specs=specs)


