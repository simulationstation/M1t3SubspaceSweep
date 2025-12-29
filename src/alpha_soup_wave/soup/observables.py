"""Compute internal scales and observables for v3 mechanisms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import networkx as nx

from alpha_soup_wave.config import GateConfig, M2Config, SelectionConfig, SpectralConfig, default_m2_min
from alpha_soup_wave.rng import make_rng
from alpha_soup_wave.soup.m1_strings import StringPopulation
from alpha_soup_wave.soup.m2_detector import coherence_values, detect_m2
from alpha_soup_wave.soup.wave_dynamics import damping_length, lorentzian_density, string_modes, string_linewidths


@dataclass(frozen=True)
class SoupObservables:
    attenuation: np.ndarray
    overlap: np.ndarray
    selection_eff: float
    gate_pA: float
    gate_pB: float
    gate_pAB: float
    m2_size: int
    m2_coherence_mean: float
    m2_valid: bool


def _eigenfrequencies(graph: nx.Graph) -> np.ndarray:
    if graph.number_of_nodes() <= 1:
        return np.array([])
    lap = nx.normalized_laplacian_matrix(graph).toarray()
    eigvals = np.linalg.eigvalsh(lap)
    eigvals = eigvals[eigvals > 1e-6]
    return np.sqrt(eigvals)


def _spectral_density(freqs: np.ndarray, omega_grid: np.ndarray, sigma: float) -> np.ndarray:
    if freqs.size == 0:
        return np.zeros_like(omega_grid)
    density = np.zeros_like(omega_grid)
    for freq in freqs:
        density += np.exp(-0.5 * ((omega_grid - freq) / sigma) ** 2)
    density /= density.max() if density.max() > 0 else 1.0
    return density


def _overlap_integral(
    omega_grid: np.ndarray,
    rho_a: np.ndarray,
    rho_b: np.ndarray,
    weight_scale: float,
) -> float:
    weight = 1.0 / (1.0 + (omega_grid / max(weight_scale, 1e-6)) ** 2)
    norm = np.trapz(rho_a * rho_a * weight, omega_grid) * np.trapz(rho_b * rho_b * weight, omega_grid)
    if norm <= 0:
        return 0.0
    overlap = np.trapz(rho_a * rho_b * weight, omega_grid)
    return float(np.clip(overlap / np.sqrt(norm), 0.0, 1.0))


def _gate_samples(
    graph: nx.Graph,
    theta: np.ndarray,
    m2_config: M2Config,
    gate_config: GateConfig,
    strings: StringPopulation,
    omega_star: float,
    mode_count: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = make_rng(seed)
    base_m2 = detect_m2(graph, theta, m2_config.coherence_threshold, m2_config.loopiness_threshold)
    base_size = len(base_m2)
    min_size = m2_config.min_size or default_m2_min(graph.number_of_nodes())
    damping_rates = np.array([spec.damping_rate for spec in strings.specs])
    damping_threshold = float(np.quantile(damping_rates, gate_config.low_damping_quantile)) if damping_rates.size else 0.0

    gate_a_hits = 0
    gate_b_hits = 0
    gate_ab_hits = 0
    for _ in range(gate_config.samples):
        theta_sample = theta + rng.normal(0.0, gate_config.theta_noise_scale, size=theta.shape)
        m2_sample = detect_m2(
            graph,
            theta_sample,
            m2_config.coherence_threshold,
            m2_config.loopiness_threshold,
        )
        coherence_sample = coherence_values(graph, theta_sample)
        mean_coherence = float(np.mean([coherence_sample[node] for node in m2_sample]) if m2_sample else 0.0)
        stable_size = (
            base_size > 0
            and abs(len(m2_sample) - base_size) / base_size <= gate_config.size_stability_fraction
        )
        gate_a = len(m2_sample) >= min_size and stable_size and mean_coherence >= gate_config.coherence_gate

        jitter = rng.normal(0.0, gate_config.length_jitter_fraction, size=len(strings.specs))
        resonance_hits = 0
        for spec, jitter_frac in zip(strings.specs, jitter, strict=True):
            length = spec.length * (1.0 + jitter_frac)
            modes = string_modes(spec, mode_count) * (spec.length / max(length, 1e-6))
            if np.any(np.abs(modes - omega_star) <= gate_config.resonance_eps) and spec.damping_rate <= damping_threshold:
                resonance_hits += 1
        gate_b = resonance_hits >= gate_config.resonance_k

        if gate_a:
            gate_a_hits += 1
        if gate_b:
            gate_b_hits += 1
        if gate_a and gate_b:
            gate_ab_hits += 1

    samples = max(gate_config.samples, 1)
    return (
        gate_a_hits / samples,
        gate_b_hits / samples,
        gate_ab_hits / samples,
    )


def compute_observables(
    graph: nx.Graph,
    theta: np.ndarray,
    m2_nodes: set[int],
    strings: StringPopulation,
    m2_config: M2Config,
    spectral_config: SpectralConfig,
    selection_config: SelectionConfig,
    gate_config: GateConfig,
    mode_count: int,
    seed: int,
) -> SoupObservables:
    eigenfrequencies = _eigenfrequencies(graph)
    omega_star = float(np.median(eigenfrequencies)) if eigenfrequencies.size else 1.0
    omega_max = spectral_config.omega_max_factor * omega_star
    omega_grid = np.linspace(0.0, omega_max, spectral_config.grid_size)

    m2_subgraph = graph.subgraph(m2_nodes).copy() if m2_nodes else graph.subgraph([])
    m2_freqs = _eigenfrequencies(m2_subgraph)
    rho_m2 = _spectral_density(m2_freqs, omega_grid, spectral_config.kernel_sigma)
    rho_m3 = _spectral_density(eigenfrequencies, omega_grid, spectral_config.kernel_sigma)

    attenuation = np.array([np.exp(-spec.length / damping_length(spec)) for spec in strings.specs])

    overlaps = []
    overlaps_m3 = []
    selection_sum = 0.0
    coherence = coherence_values(graph, theta)
    for spec in strings.specs:
        modes = string_modes(spec, mode_count)
        widths = string_linewidths(spec, modes)
        rho_str = lorentzian_density(omega_grid, modes, widths)
        overlaps.append(_overlap_integral(omega_grid, rho_m2, rho_str, spectral_config.weight_scale))
        overlaps_m3.append(_overlap_integral(omega_grid, rho_m3, rho_str, spectral_config.weight_scale))
        g_m2_str = coherence.get(spec.anchor, 0.0) if spec.anchor in m2_nodes else 0.0
        g_str_m3 = overlaps_m3[-1]
        delta = np.abs(modes - omega_star) + widths
        selection_sum += float(np.sum(g_m2_str * g_str_m3 / delta))

    selection_eff = 1.0 - float(np.exp(-abs(selection_sum) / max(selection_config.scale, 1e-6)))
    gate_pA, gate_pB, gate_pAB = _gate_samples(
        graph,
        theta,
        m2_config,
        gate_config,
        strings,
        omega_star,
        mode_count,
        seed,
    )
    m2_coherence_mean = float(np.mean([coherence[node] for node in m2_nodes]) if m2_nodes else 0.0)
    min_size = m2_config.min_size or default_m2_min(graph.number_of_nodes())
    m2_valid = len(m2_nodes) >= min_size

    return SoupObservables(
        attenuation=np.array(attenuation),
        overlap=np.array(overlaps),
        selection_eff=selection_eff,
        gate_pA=gate_pA,
        gate_pB=gate_pB,
        gate_pAB=gate_pAB,
        m2_size=len(m2_nodes),
        m2_coherence_mean=m2_coherence_mean,
        m2_valid=m2_valid,
    )


