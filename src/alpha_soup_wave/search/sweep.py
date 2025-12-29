"""Coarse parameter sweep for emergent alpha."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from alpha_soup_wave.config import ModelConfig, NetworkConfig, M2Config, StringConfig, SweepConfig
from alpha_soup_wave.constants import ALPHA_TARGET
from alpha_soup_wave.reporting.report_md import generate_report, write_best_candidates
from alpha_soup_wave.search.alpha_estimators import all_alpha_hats
from alpha_soup_wave.search.filters import compute_log_distance
from alpha_soup_wave.soup.graph_gen import generate_m3_graph
from alpha_soup_wave.soup.hamiltonian import sample_phases
from alpha_soup_wave.soup.m1_strings import attach_strings, StringPopulation
from alpha_soup_wave.soup.m2_detector import detect_m2
from alpha_soup_wave.soup.observables import compute_observables
from alpha_soup_wave.viz.diagnostics import plot_alpha_hist, plot_robustness
from alpha_soup_wave.viz.soup_plot import plot_soup


@dataclass(frozen=True)
class SweepResult:
    config_id: str
    alpha_hat: float
    alpha_std: float
    alpha_log_distance: float
    m2_size_mean: float
    m2_size_std: float
    neighbor_stability: float


def _alpha_hat_from_observables(observables) -> float:
    hats = list(all_alpha_hats(observables).values())
    hats = [val for val in hats if np.isfinite(val) and val > 0]
    if not hats:
        return float("nan")
    return float(np.median(hats))


def _build_soup(model: ModelConfig, seed: int) -> tuple[np.ndarray, set[int], StringPopulation, object]:
    graph = generate_m3_graph(model.network.n3, seed, model.network.degree2_fraction)
    theta = sample_phases(
        graph,
        coupling_j=model.network.coupling_j,
        temperature=model.network.temperature,
        steps=model.network.metropolis_steps,
        seed=seed,
    )
    m2_nodes = detect_m2(
        graph,
        theta,
        coherence_threshold=model.m2.coherence_threshold,
        loopiness_threshold=model.m2.loopiness_threshold,
    )
    anchor_nodes = list(m2_nodes) if m2_nodes else list(graph.nodes)
    strings = attach_strings(
        anchor_nodes,
        model.strings.anchor_fraction,
        model.strings.length_mean,
        model.strings.length_sigma,
        model.strings.tension_mean,
        model.strings.tension_sigma,
        model.strings.density_mean,
        model.strings.density_sigma,
        model.strings.damping_mean,
        model.strings.damping_sigma,
        seed=seed,
    )
    return graph, m2_nodes, strings, theta


def _run_single_seed(model: ModelConfig, seed: int) -> tuple[float, int, dict[str, float]]:
    graph, m2_nodes, strings, _ = _build_soup(model, seed)
    observables = compute_observables(graph, strings)
    alpha_hat = _alpha_hat_from_observables(observables)
    return alpha_hat, len(m2_nodes), all_alpha_hats(observables)


def _perturb_model(model: ModelConfig, factor: float) -> ModelConfig:
    return ModelConfig(
        network=NetworkConfig(
            n3=model.network.n3,
            degree2_fraction=model.network.degree2_fraction,
            coupling_j=model.network.coupling_j * factor,
            temperature=model.network.temperature * factor,
            loop_penalty=model.network.loop_penalty,
            metropolis_steps=model.network.metropolis_steps,
        ),
        m2=M2Config(
            coherence_threshold=model.m2.coherence_threshold,
            loopiness_threshold=model.m2.loopiness_threshold,
        ),
        strings=StringConfig(
            anchor_fraction=model.strings.anchor_fraction,
            length_mean=model.strings.length_mean * factor,
            length_sigma=model.strings.length_sigma,
            tension_mean=model.strings.tension_mean,
            tension_sigma=model.strings.tension_sigma,
            density_mean=model.strings.density_mean,
            density_sigma=model.strings.density_sigma,
            damping_mean=model.strings.damping_mean,
            damping_sigma=model.strings.damping_sigma,
        ),
    )


def _neighbor_stability(model: ModelConfig, seeds: Iterable[int], perturb: float) -> float:
    base_values = [
        _run_single_seed(model, seed)[0]
        for seed in seeds
    ]
    base = float(np.nanmean(base_values))
    values = []
    for factor in (1 - perturb, 1 + perturb):
        pert_model = _perturb_model(model, factor)
        pert_values = [_run_single_seed(pert_model, seed)[0] for seed in seeds]
        values.append(float(np.nanmean(pert_values)))
    distances = [abs(np.log(val) - np.log(base)) for val in values if val > 0 and base > 0]
    return float(max(distances) if distances else float("inf"))


def run_sweep(
    model_grid: list[ModelConfig],
    sweep: SweepConfig,
    output_dir: str,
    label: str,
) -> tuple[pd.DataFrame, list[dict]]:
    output = Path(output_dir)
    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results: list[SweepResult] = []
    best_candidates: list[dict] = []

    for idx, model in enumerate(model_grid):
        config_id = f"{label}_{idx:04d}"
        alpha_vals = []
        m2_sizes = []
        for seed in sweep.seeds:
            alpha_hat, m2_size, _ = _run_single_seed(model, seed)
            alpha_vals.append(alpha_hat)
            m2_sizes.append(m2_size)
        alpha_vals = np.array(alpha_vals)
        alpha_mean = float(np.nanmean(alpha_vals))
        alpha_std = float(np.nanstd(alpha_vals))
        m2_sizes = np.array(m2_sizes)
        m2_mean = float(np.mean(m2_sizes))
        m2_std = float(np.std(m2_sizes))
        log_distance = compute_log_distance(alpha_mean, ALPHA_TARGET) if alpha_mean > 0 else float("inf")
        neighbor = _neighbor_stability(model, sweep.seeds, sweep.perturbation_fraction)
        results.append(
            SweepResult(
                config_id=config_id,
                alpha_hat=alpha_mean,
                alpha_std=alpha_std,
                alpha_log_distance=log_distance,
                m2_size_mean=m2_mean,
                m2_size_std=m2_std,
                neighbor_stability=neighbor,
            )
        )

    results_df = pd.DataFrame([res.__dict__ for res in results])
    results_df = results_df.sort_values("alpha_log_distance")

    stable_mask = (
        (results_df["alpha_log_distance"] <= sweep.tolerance_log)
        & (results_df["alpha_std"] <= results_df["alpha_std"].median() * 1.5)
        & (results_df["m2_size_mean"] > 0)
        & (results_df["neighbor_stability"] <= sweep.tolerance_log)
    )
    candidates = results_df[stable_mask]
    for _, row in candidates.iterrows():
        best_candidates.append(row.to_dict())

    results_df.to_csv(output / "results.csv", index=False)
    write_best_candidates(best_candidates, output / "best_candidates.json")
    generate_report(results_df, best_candidates, output / "report.md")

    if not results_df.empty:
        plot_alpha_hist(results_df, figures_dir / "alpha_hist.png")
        plot_robustness(results_df, figures_dir / "robustness.png")
        best_id = results_df.iloc[0]["config_id"]
        try:
            best_index = int(str(best_id).split("_")[-1])
        except ValueError:
            best_index = 0
        best_model = model_grid[min(best_index, len(model_grid) - 1)]
        graph, m2_nodes, strings, _ = _build_soup(best_model, seed=0)
        ax = plot_soup(graph, m2_nodes, strings)
        ax.figure.savefig(figures_dir / "soup_best.png", bbox_inches="tight")
        ax.figure.clf()

    return results_df, best_candidates


def build_model_grid(sweep: SweepConfig) -> list[ModelConfig]:
    n3_values = [200, 400]
    coupling_values = [0.8, 1.0, 1.2]
    temperature_values = [0.3, 0.4, 0.5]
    length_means = [4.0, 5.0, 6.0]
    grid: list[ModelConfig] = []
    for n3 in n3_values:
        for coupling in coupling_values:
            for temp in temperature_values:
                for length_mean in length_means:
                    grid.append(
                        ModelConfig(
                            network=NetworkConfig(n3=n3, coupling_j=coupling, temperature=temp),
                            m2=M2Config(),
                            strings=StringConfig(length_mean=length_mean),
                        )
                    )
                    if len(grid) >= sweep.max_configs:
                        return grid
    return grid
