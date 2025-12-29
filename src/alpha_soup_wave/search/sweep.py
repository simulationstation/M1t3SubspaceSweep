"""Coarse parameter sweep for emergent alpha (v3)."""

from __future__ import annotations

from dataclasses import dataclass
import os
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from alpha_soup_wave.config import (
    GateConfig,
    M2Config,
    MechanismConfig,
    ModelConfig,
    NetworkConfig,
    SelectionConfig,
    SpectralConfig,
    StringConfig,
    SweepConfig,
    default_m2_min,
    expand_values,
)
from alpha_soup_wave.reporting.report_md import generate_report, write_best_candidates
from alpha_soup_wave.search.alpha_estimators import all_alpha_hats, combined_alpha_hat
from alpha_soup_wave.search.robustness import log_distance, plateau_check, parameter_r2, fine_tune_flag_for_config
from alpha_soup_wave.soup.graph_gen import generate_m3_graph
from alpha_soup_wave.soup.hamiltonian import sample_phases
from alpha_soup_wave.soup.m1_strings import attach_strings, StringPopulation
from alpha_soup_wave.soup.m2_detector import detect_m2
from alpha_soup_wave.soup.observables import compute_observables
from alpha_soup_wave.viz.plots import plot_alpha_histograms, plot_seed_stability, plot_sensitivity
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
    fine_tune_flag: bool
    plateau_pass: bool
    score: float


def _alpha_hat_from_observables(observables, mechanisms: MechanismConfig) -> tuple[float, dict[str, float]]:
    hats = all_alpha_hats(observables, mechanisms)
    return combined_alpha_hat(hats), hats


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
        list(model.strings.boundary_choices),
        seed=seed,
    )
    return graph, m2_nodes, strings, theta


def _run_single_seed(model: ModelConfig, seed: int) -> tuple[float, int, bool, dict[str, float], dict[str, float]]:
    graph, m2_nodes, strings, theta = _build_soup(model, seed)
    observables = compute_observables(
        graph,
        theta,
        m2_nodes,
        strings,
        model.m2,
        model.spectral,
        model.selection,
        model.gate,
        model.strings.mode_count,
        seed,
    )
    alpha_hat, hats = _alpha_hat_from_observables(observables, model.mechanisms)
    extra = {
        "gate_pA": observables.gate_pA,
        "gate_pB": observables.gate_pB,
        "gate_pAB": observables.gate_pAB,
        "m2_coherence_mean": observables.m2_coherence_mean,
    }
    return alpha_hat, observables.m2_size, observables.m2_valid, hats, extra


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
            min_size=model.m2.min_size,
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
            boundary_choices=model.strings.boundary_choices,
            mode_count=model.strings.mode_count,
        ),
        spectral=SpectralConfig(**model.spectral.model_dump()),
        mechanisms=MechanismConfig(**model.mechanisms.model_dump()),
        selection=SelectionConfig(**model.selection.model_dump()),
        gate=GateConfig(**model.gate.model_dump()),
    )


def _neighbor_values(model: ModelConfig, seeds: Iterable[int], perturb: float, samples: int) -> list[float]:
    values = []
    if samples <= 1:
        factors = [1 - perturb, 1 + perturb]
    else:
        factors = np.linspace(1 - perturb, 1 + perturb, samples)
    for factor in factors:
        pert_model = _perturb_model(model, factor)
        pert_values = [_run_single_seed(pert_model, seed)[0] for seed in seeds]
        values.append(float(np.nanmean(pert_values)))
    return values


def _neighbor_stability(alpha_base: float, neighbor_values: list[float]) -> float:
    distances = [log_distance(val) for val in neighbor_values if val > 0]
    return float(max(distances) if distances else float("inf"))


def _flatten_model(model: ModelConfig) -> dict[str, float | int | str]:
    return {
        "n3": model.network.n3,
        "degree2_fraction": model.network.degree2_fraction,
        "coupling_j": model.network.coupling_j,
        "temperature": model.network.temperature,
        "coherence_threshold": model.m2.coherence_threshold,
        "loopiness_threshold": model.m2.loopiness_threshold,
        "m2_min_size": model.m2.min_size or default_m2_min(model.network.n3),
        "anchor_fraction": model.strings.anchor_fraction,
        "length_mean": model.strings.length_mean,
        "length_sigma": model.strings.length_sigma,
        "tension_mean": model.strings.tension_mean,
        "density_mean": model.strings.density_mean,
        "damping_mean": model.strings.damping_mean,
        "boundary_choices": ",".join(model.strings.boundary_choices),
        "mode_count": model.strings.mode_count,
        "attenuation_weight_m2": float(model.mechanisms.attenuation_weight_m2),
        "enable_selection": float(model.mechanisms.enable_selection),
        "enable_gating": float(model.mechanisms.enable_gating),
    }


def _evaluate_config(args: tuple[str, ModelConfig, SweepConfig]) -> tuple[dict, list[dict]]:
    config_id, model, sweep = args
    per_seed_rows: list[dict] = []
    alpha_vals = []
    m2_sizes = []
    valid_flags = []

    for seed in sweep.seeds:
        alpha_hat, m2_size, m2_valid, hats, extra = _run_single_seed(model, seed)
        row = {
            "config_id": config_id,
            "seed": seed,
            "alpha_hat": alpha_hat,
            "m2_size": m2_size,
            "m2_valid": m2_valid,
            **hats,
            **extra,
        }
        row.update(_flatten_model(model))
        per_seed_rows.append(row)
        if m2_valid and np.isfinite(alpha_hat):
            alpha_vals.append(alpha_hat)
        m2_sizes.append(m2_size)
        valid_flags.append(m2_valid)

    alpha_vals = np.array(alpha_vals)
    alpha_mean = float(np.nanmean(alpha_vals)) if alpha_vals.size else float("nan")
    alpha_std = float(np.nanstd(alpha_vals)) if alpha_vals.size else float("inf")
    m2_sizes = np.array(m2_sizes)
    m2_mean = float(np.mean(m2_sizes)) if m2_sizes.size else 0.0
    m2_std = float(np.std(m2_sizes)) if m2_sizes.size else 0.0
    log_dist = log_distance(alpha_mean) if np.isfinite(alpha_mean) else float("inf")
    neighbor_values = _neighbor_values(model, sweep.seeds, sweep.perturbation_fraction, sweep.neighborhood_samples)
    neighbor = _neighbor_stability(alpha_mean, neighbor_values)
    plateau_pass = plateau_check(alpha_mean, neighbor_values, sweep.plateau_rel_tol)

    summary_row = {
        "config_id": config_id,
        "alpha_hat": alpha_mean,
        "alpha_std": alpha_std,
        "alpha_log_distance": log_dist,
        "m2_size_mean": m2_mean,
        "m2_size_std": m2_std,
        "neighbor_stability": neighbor,
        "plateau_pass": plateau_pass,
        "valid_fraction": float(np.mean(valid_flags)),
    }
    summary_row.update(_flatten_model(model))
    return summary_row, per_seed_rows


def build_model_grid(grid_spec: dict, sweep: SweepConfig) -> list[ModelConfig]:
    network_spec = grid_spec.get("network", {})
    m2_spec = grid_spec.get("m2", {})
    string_spec = grid_spec.get("strings", {})
    spectral_spec = grid_spec.get("spectral", {})
    mech_spec = grid_spec.get("mechanisms", {})
    selection_spec = grid_spec.get("selection", {})
    gate_spec = grid_spec.get("gate", {})

    grid: list[ModelConfig] = []
    for n3, coupling_j, temperature, length_mean, damping_mean, coherence, loopiness in product(
        expand_values(network_spec.get("n3", 200)),
        expand_values(network_spec.get("coupling_j", 1.0)),
        expand_values(network_spec.get("temperature", 0.4)),
        expand_values(string_spec.get("length_mean", 5.0)),
        expand_values(string_spec.get("damping_mean", 0.08)),
        expand_values(m2_spec.get("coherence_threshold", 0.7)),
        expand_values(m2_spec.get("loopiness_threshold", 0.1)),
    ):
        model = ModelConfig(
            network=NetworkConfig(
                n3=int(n3),
                degree2_fraction=float(network_spec.get("degree2_fraction", 0.0)),
                coupling_j=float(coupling_j),
                temperature=float(temperature),
                metropolis_steps=int(network_spec.get("metropolis_steps", 3000)),
            ),
            m2=M2Config(
                coherence_threshold=float(coherence),
                loopiness_threshold=float(loopiness),
                min_size=m2_spec.get("min_size"),
            ),
            strings=StringConfig(
                anchor_fraction=float(string_spec.get("anchor_fraction", 0.25)),
                length_mean=float(length_mean),
                length_sigma=float(string_spec.get("length_sigma", 0.6)),
                tension_mean=float(string_spec.get("tension_mean", 1.4)),
                tension_sigma=float(string_spec.get("tension_sigma", 0.3)),
                density_mean=float(string_spec.get("density_mean", 0.9)),
                density_sigma=float(string_spec.get("density_sigma", 0.2)),
                damping_mean=float(damping_mean),
                damping_sigma=float(string_spec.get("damping_sigma", 0.03)),
                boundary_choices=expand_values(string_spec.get("boundary_choices", ["fixed", "free"])),
                mode_count=int(string_spec.get("mode_count", 6)),
            ),
            spectral=SpectralConfig(
                omega_max_factor=float(spectral_spec.get("omega_max_factor", 2.5)),
                grid_size=int(spectral_spec.get("grid_size", 160)),
                kernel_sigma=float(spectral_spec.get("kernel_sigma", 0.2)),
                weight_scale=float(spectral_spec.get("weight_scale", 1.0)),
            ),
            mechanisms=MechanismConfig(
                attenuation_weight_m2=bool(mech_spec.get("attenuation_weight_m2", True)),
                enable_selection=bool(mech_spec.get("enable_selection", True)),
                enable_gating=bool(mech_spec.get("enable_gating", True)),
            ),
            selection=SelectionConfig(scale=float(selection_spec.get("scale", 0.35))),
            gate=GateConfig(
                coherence_gate=float(gate_spec.get("coherence_gate", 0.75)),
                size_stability_fraction=float(gate_spec.get("size_stability_fraction", 0.2)),
                resonance_k=int(gate_spec.get("resonance_k", 2)),
                resonance_eps=float(gate_spec.get("resonance_eps", 0.12)),
                low_damping_quantile=float(gate_spec.get("low_damping_quantile", 0.35)),
                samples=int(gate_spec.get("samples", 8)),
                length_jitter_fraction=float(gate_spec.get("length_jitter_fraction", 0.05)),
                theta_noise_scale=float(gate_spec.get("theta_noise_scale", 0.08)),
            ),
        )
        grid.append(model)
        if len(grid) >= sweep.max_configs:
            break
    return grid


def run_sweep(
    model_grid: list[ModelConfig],
    sweep: SweepConfig,
    output_dir: str,
    label: str,
) -> tuple[pd.DataFrame, list[dict]]:
    output = Path(output_dir)
    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    per_seed_rows: list[dict] = []

    tasks = [(f"{label}_{idx:04d}", model, sweep) for idx, model in enumerate(model_grid)]
    if sweep.max_workers and sweep.max_workers > 1:
        import concurrent.futures

        with concurrent.futures.ProcessPoolExecutor(max_workers=sweep.max_workers) as executor:
            for summary_row, seed_rows in executor.map(_evaluate_config, tasks):
                summary_rows.append(summary_row)
                per_seed_rows.extend(seed_rows)
    else:
        for task in tasks:
            summary_row, seed_rows = _evaluate_config(task)
            summary_rows.append(summary_row)
            per_seed_rows.extend(seed_rows)

    results_df = pd.DataFrame(per_seed_rows)
    summary_df = pd.DataFrame(summary_rows)

    param_columns = [
        "n3",
        "degree2_fraction",
        "coupling_j",
        "temperature",
        "coherence_threshold",
        "loopiness_threshold",
        "anchor_fraction",
        "length_mean",
        "length_sigma",
        "tension_mean",
        "density_mean",
        "damping_mean",
    ]
    r2_values = parameter_r2(summary_df, param_columns, "alpha_hat")
    summary_df["fine_tune_flag"] = summary_df.apply(
        lambda row: fine_tune_flag_for_config(r2_values, bool(row["plateau_pass"])), axis=1
    )

    summary_df["score"] = (
        summary_df["alpha_log_distance"]
        + sweep.seed_std_weight * summary_df["alpha_std"]
        + sweep.neighbor_weight * summary_df["neighbor_stability"]
        + sweep.m2_instability_weight * (1.0 - summary_df["valid_fraction"])
        + sweep.fine_tune_weight * summary_df["fine_tune_flag"].astype(float)
    )
    summary_df = summary_df.sort_values("score")

    best_candidates = summary_df.head(sweep.top_k).to_dict(orient="records")

    results_df.to_csv(output / "results.csv", index=False)
    summary_df.to_csv(output / "summary.csv", index=False)
    write_best_candidates(best_candidates, output / "best_candidates.json")
    generate_report(summary_df, best_candidates, output / "report.md", r2_values)

    if not results_df.empty:
        plot_alpha_histograms(results_df, figures_dir)
        plot_seed_stability(summary_df, figures_dir / "seed_stability.png")
        plot_sensitivity(summary_df, param_columns, figures_dir / "sensitivity.png")

        best_id = summary_df.iloc[0]["config_id"]
        best_index = int(str(best_id).split("_")[-1]) if best_id else 0
        best_model = model_grid[min(best_index, len(model_grid) - 1)]
        graph, m2_nodes, strings, _ = _build_soup(best_model, seed=0)
        ax = plot_soup(graph, m2_nodes, strings)
        ax.figure.savefig(figures_dir / "soup_best.png", bbox_inches="tight")
        ax.figure.clf()

    return summary_df, best_candidates


def load_sweep_config(path: str) -> tuple[list[ModelConfig], SweepConfig]:
    import json

    spec = json.loads(Path(path).read_text())
    sweep = SweepConfig(**spec.get("sweep", {}))
    if sweep.max_workers is None:
        sweep = sweep.model_copy(update={"max_workers": os.cpu_count()})
    grid = build_model_grid(spec.get("grid", {}), sweep)
    return grid, sweep
