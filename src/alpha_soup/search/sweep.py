"""Coarse parameter sweep."""
from __future__ import annotations

import json
import itertools
import hashlib
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from alpha_soup.config import (
    SoupConfig,
    DegreeProfile,
    Couplings,
    EmergentCriteria,
    DynamicsConfig,
    SweepConfig,
)
from alpha_soup.rng import make_rng
from alpha_soup.search.objectives import evaluate_objective
from alpha_soup.soup import generate_degree_graph, apply_decay, initialize_state, evolve, detect_m2, compute_observables


def _config_hash(config: SoupConfig) -> str:
    payload = json.dumps(config.model_dump(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _run_single(args: tuple[SoupConfig, int]) -> dict:
    config, seed = args
    rng = make_rng(seed)
    graph_bundle = generate_degree_graph(config.n_nodes, config.profile, seed)
    graph_bundle = apply_decay(graph_bundle, config.decay, seed)
    state = initialize_state(config.n_nodes, seed)
    evolved = evolve(
        state,
        list(graph_bundle.graph.edges),
        graph_bundle.degree_labels,
        config.couplings,
        config.dynamics,
        seed,
    )
    m2 = detect_m2(graph_bundle.graph, evolved.theta, config.emergent)
    obs = compute_observables(graph_bundle.graph, m2.nodes, graph_bundle.degree_labels, evolved.theta)
    objective = evaluate_objective(obs)
    return {
        "config_hash": _config_hash(config),
        "seed": seed,
        "loss": objective.loss,
        "alpha_gap_ratio": objective.alpha_estimates.alpha_gap_ratio,
        "alpha_stiffness_ratio": objective.alpha_estimates.alpha_stiffness_ratio,
        "alpha_overlap_gap": objective.alpha_estimates.alpha_overlap_gap,
        "m2_count": len(m2.nodes),
        "spectral_gap_m2": obs.spectral_gap_m2,
        "spectral_radius_m3": obs.spectral_radius_m3,
        "corr_length_m2": obs.corr_length_m2,
        "stiffness_m2": obs.stiffness_m2,
        "stiffness_m3": obs.stiffness_m3,
        "overlap_23": obs.overlap_23,
        "stability_m2": obs.stability_m2,
    }


def _sweep_configs(cfg: SweepConfig) -> list[SoupConfig]:
    sweep = cfg.sweep
    configs: list[SoupConfig] = []
    for n_nodes, frac3, frac2, j3, j2, j1, g23, g21, temp, c_thresh, l_thresh in itertools.product(
        sweep.n_nodes,
        sweep.frac_deg3,
        sweep.frac_deg2,
        sweep.j3,
        sweep.j2,
        sweep.j1,
        sweep.g23,
        sweep.g21,
        sweep.temperature,
        sweep.coherence_thresh,
        sweep.loop_thresh,
    ):
        frac1 = max(1.0 - frac3 - frac2, 0.05)
        profile = DegreeProfile(frac_deg3=frac3, frac_deg2=frac2, frac_deg1=frac1)
        couplings = Couplings(j3=j3, j2=j2, j1=j1, g23=g23, g21=g21)
        emergent = EmergentCriteria(coherence_thresh=c_thresh, loop_thresh=l_thresh)
        dynamics = DynamicsConfig(temperature=temp, steps=cfg.fast_mode_steps)
        configs.append(
            SoupConfig(
                n_nodes=n_nodes,
                profile=profile,
                couplings=couplings,
                emergent=emergent,
                dynamics=dynamics,
                seed=0,
            )
        )
    return configs


def run_sweep(config: SweepConfig, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = _sweep_configs(config)
    tasks: list[tuple[SoupConfig, int]] = []
    for soup_cfg in configs:
        for seed in range(config.search.seeds_per_point):
            tasks.append((soup_cfg, seed))

    with Pool(processes=config.search.max_workers) as pool:
        rows = list(pool.map(_run_single, tasks))

    df = pd.DataFrame(rows)
    configs_used = {cfg_hash: cfg.model_dump() for cfg_hash, cfg in {_config_hash(c): c for c in configs}.items()}
    (out_dir / "configs_used.json").write_text(json.dumps(configs_used, indent=2))
    df.to_csv(out_dir / "results.csv", index=False)
    return df
