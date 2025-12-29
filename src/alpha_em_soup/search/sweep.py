"""Coarse sweep for EM soup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from alpha_em_soup.config import RunConfig, SweepConfig, apply_overrides, expand_grid, load_config
from alpha_em_soup.constants import ALPHA_TARGET
from alpha_em_soup.layers.m1_wavepaths import build_m1_wavepaths
from alpha_em_soup.layers.m2_interface import build_m2
from alpha_em_soup.layers.m3_topology import generate_m3_graph
from alpha_em_soup.physics.effective_coupling import compute_effective_coupling
from alpha_em_soup.physics.estimators import estimate_alpha_candidates
from alpha_em_soup.physics.gates import gate_probabilities
from alpha_em_soup.search.objective import ObjectiveInputs, score_objective
from alpha_em_soup.search.robustness import plateau_check, parameter_r2
from alpha_em_soup.viz.diagnostics import plot_diagnostics
from alpha_em_soup.viz.soup_plot import plot_soup


@dataclass(frozen=True)
class SeedResult:
    seed: int
    g21_eff: float
    g23_eff: float
    q_ratio: float
    m2_size: int
    m2_stable: bool
    gate_pA: float
    gate_pB: float
    gate_pAB: float
    alpha_hat_cov: float
    alpha_hat_ov: float
    alpha_hat_z: float


def _build_soup(model, seed: int):
    m3 = generate_m3_graph(
        model.m3.n3,
        seed,
        model.m3.generator,
        model.m3.min_short_cycle_ratio,
        model.m3.triangle_boost_steps,
        model.m3.geometric_radius,
        model.m3.planted_cycle_count,
    )
    m2 = build_m2(
        m3.graph,
        seed,
        model.m2.mode,
        model.m2.coherence_threshold,
        model.m2.loopiness_threshold,
        model.m2.min_fraction,
    )
    rng = np.random.default_rng(seed)
    anchors = sorted(m2.nodes)
    if anchors:
        count = max(1, int(len(anchors) * model.m1.anchor_fraction))
        anchors = list(rng.choice(anchors, size=count, replace=False))
    m1 = build_m1_wavepaths(
        anchors,
        seed,
        model.m1.path_count,
        model.m1.length_mean,
        model.m1.length_sigma,
        model.m1.loop_fraction,
        model.m1.loopback_fraction,
        model.m1.max_path_edges,
    )
    return m3, m2, m1


def _frequency_grid(model) -> np.ndarray:
    return np.linspace(model.physics.omega_min, model.physics.omega_max, model.physics.omega_points)


def _resonance_fraction(paths, omega_grid: np.ndarray) -> float:
    if not paths:
        return 0.0
    count = 0
    for path in paths:
        if path.length <= 0:
            continue
        modes = np.arange(1, 6) * np.pi / path.length
        if np.any((modes >= omega_grid.min()) & (modes <= omega_grid.max())):
            count += 1
    return count / max(1, len(paths))


def _evaluate_seed(model, seed: int) -> SeedResult:
    m3, m2, m1 = _build_soup(model, seed)
    nodes = sorted(set(m3.graph.nodes) | set(m1.graph.nodes))
    m1_edges = [(u, v, data.get("length", 1.0)) for u, v, data in m1.graph.edges(data=True)]
    m3_edges = list(m3.graph.edges)
    node_index = {node: idx for idx, node in enumerate(nodes)}
    m1_nodes = np.array([node_index[node] for node in m1.graph.nodes if node not in m3.graph.nodes])
    m3_nodes = np.array([node_index[node] for node in m3.graph.nodes if node not in m2.nodes])
    damping = np.zeros(len(nodes))
    for node in nodes:
        idx = node_index[node]
        if node in m1.graph.nodes and node not in m3.graph.nodes:
            damping[idx] = model.physics.damping_m1 + model.m1.edge_loss
        elif node in m2.nodes:
            damping[idx] = model.physics.damping_m2
        else:
            damping[idx] = model.physics.damping_m3
    omega_grid = _frequency_grid(model)
    j2 = model.coupling.j2
    j23 = model.coupling.j23
    if model.m2.mode == "constructed":
        j2 *= model.m2.constructed_strength
        j23 *= model.m2.boundary_strength
    coupling = compute_effective_coupling(
        nodes,
        m3_edges,
        m1_edges,
        m2.nodes,
        damping,
        omega_grid,
        model.physics.k_scale,
        model.m1.attenuation_length,
        model.coupling.j3,
        j2,
        j23,
        model.coupling.drive_fraction,
        model.coupling.drive_strength,
        seed,
        m1_nodes,
        m3_nodes,
    )
    resonance = _resonance_fraction(m1.paths, omega_grid)
    if model.gate.enable:
        gates = gate_probabilities(m2.coherence_mean, resonance, model.gate.coherence_min, model.gate.resonance_min)
    else:
        gates = gate_probabilities(1.0, 1.0, 1.0, 1.0)
    m2_graph = m3.graph.subgraph(m2.nodes).copy()
    alpha_candidates = estimate_alpha_candidates(
        m1.paths,
        m1.graph,
        m2_graph,
        omega_grid,
        model.m1.attenuation_length,
        model.physics.damping_m1,
        model.physics.damping_m2,
        m1.anchor_nodes,
    )
    return SeedResult(
        seed=seed,
        g21_eff=coupling.g21_eff,
        g23_eff=coupling.g23_eff,
        q_ratio=coupling.q_ratio,
        m2_size=len(m2.nodes),
        m2_stable=m2.stable,
        gate_pA=gates.pA,
        gate_pB=gates.pB,
        gate_pAB=gates.pAB,
        alpha_hat_cov=alpha_candidates.alpha_hat_cov,
        alpha_hat_ov=alpha_candidates.alpha_hat_ov,
        alpha_hat_z=alpha_candidates.alpha_hat_z,
    )


def _aggregate_seed_results(results: list[SeedResult], model, objective, neighborhood_penalty: float = 0.0) -> dict[str, Any]:
    g21_vals = np.array([res.g21_eff for res in results])
    g23_vals = np.array([res.g23_eff for res in results])
    q_vals = np.array([res.q_ratio for res in results])
    m2_sizes = np.array([res.m2_size for res in results])
    m2_stability = np.mean([1.0 if res.m2_stable else 0.0 for res in results])
    m2_instability = 1.0 - m2_stability
    seed_std = float(np.std(q_vals))
    q_ratio = float(np.median(q_vals))
    inputs = ObjectiveInputs(
        q_ratio=q_ratio,
        m2_instability=m2_instability,
        seed_std=seed_std,
        neighborhood_penalty=neighborhood_penalty,
        g21_eff=float(np.median(g21_vals)),
    )
    score = score_objective(
        inputs,
        objective.weight_log_q,
        objective.weight_m2_instability,
        objective.weight_seed_std,
        objective.weight_neighborhood,
        objective.weight_alpha_distance,
    )
    return {
        "g21_eff": float(np.median(g21_vals)),
        "g23_eff": float(np.median(g23_vals)),
        "q_ratio": q_ratio,
        "m2_size_mean": float(np.mean(m2_sizes)),
        "m2_stable_fraction": float(m2_stability),
        "q_std": seed_std,
        "alpha_distance": float(abs(np.log(max(inputs.g21_eff, 1e-6) / ALPHA_TARGET))),
        "score": score,
        "gate_pA": float(np.mean([res.gate_pA for res in results])),
        "gate_pB": float(np.mean([res.gate_pB for res in results])),
        "gate_pAB": float(np.mean([res.gate_pAB for res in results])),
        "alpha_hat_cov": float(np.mean([res.alpha_hat_cov for res in results])),
        "alpha_hat_ov": float(np.mean([res.alpha_hat_ov for res in results])),
        "alpha_hat_z": float(np.mean([res.alpha_hat_z for res in results])),
    }


def run_sweep(config_path: str | Path, out_dir: str | Path) -> Path:
    run_config = load_config(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seed_rows = []
    grid = expand_grid(run_config.sweep.param_grid)
    if not grid:
        grid = [{}]
    for idx, overrides in enumerate(grid[: run_config.sweep.config_count]):
        model = apply_overrides(run_config.model, overrides)
        seed_results = [_evaluate_seed(model, seed) for seed in run_config.sweep.seeds]
        summary = _aggregate_seed_results(seed_results, model, run_config.sweep.objective)
        summary.update({"config_id": idx})
        summary.update({"overrides": json.dumps(overrides)})
        rows.append(summary)
        for res in seed_results:
            seed_rows.append({
                "config_id": idx,
                "seed": res.seed,
                "g21_eff": res.g21_eff,
                "g23_eff": res.g23_eff,
                "q_ratio": res.q_ratio,
                "m2_size": res.m2_size,
                "m2_stable": res.m2_stable,
                "gate_pA": res.gate_pA,
                "gate_pB": res.gate_pB,
                "gate_pAB": res.gate_pAB,
                "alpha_hat_cov": res.alpha_hat_cov,
                "alpha_hat_ov": res.alpha_hat_ov,
                "alpha_hat_z": res.alpha_hat_z,
            })
        if idx == 0:
            plot_soup(model, seed_results[0], out_dir / "figures")
    summary_df = pd.DataFrame(rows)
    seeds_df = pd.DataFrame(seed_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    seeds_df.to_csv(out_dir / "results.csv", index=False)
    (out_dir / "config.json").write_text(Path(config_path).read_text())
    plot_diagnostics(summary_df, out_dir / "figures")
    return out_dir
