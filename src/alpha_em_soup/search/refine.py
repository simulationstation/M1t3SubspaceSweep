"""Refine top configurations with additional seeds and perturbations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_em_soup.config import RunConfig, apply_overrides, load_config
from alpha_em_soup.search.sweep import _aggregate_seed_results, _evaluate_seed
from alpha_em_soup.search.robustness import plateau_check, parameter_r2
from alpha_em_soup.viz.diagnostics import plot_diagnostics


def _neighbor_models(model, perturb: float) -> list:
    candidates = []
    factors = [1 - perturb, 1 + perturb]
    param_sets = [
        ("m1.length_mean", model.m1.length_mean),
        ("m1.attenuation_length", model.m1.attenuation_length),
        ("coupling.j23", model.coupling.j23),
        ("coupling.j2", model.coupling.j2),
    ]
    for name, base in param_sets:
        for factor in factors:
            overrides = {name: base * factor}
            candidates.append((name, base * factor, apply_overrides(model, overrides)))
    return candidates


def run_refine(input_dir: str | Path, out_dir: str | Path, top_n: int = 50) -> Path:
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(input_dir / "summary.csv")
    run_config = load_config(input_dir / "config.json")
    summary_sorted = summary.sort_values("score").head(top_n)
    rows = []
    for _, row in summary_sorted.iterrows():
        overrides = json.loads(row["overrides"]) if "overrides" in row else {}
        model = apply_overrides(run_config.model, overrides)
        seeds = list(run_config.sweep.seeds) + [max(run_config.sweep.seeds) + i + 1 for i in range(5)]
        seed_results = [_evaluate_seed(model, seed) for seed in seeds]
        neighbor_models = _neighbor_models(model, run_config.sweep.neighborhood_perturb)
        neighbor_q = []
        sensitivity = {}
        for name, value, neighbor in neighbor_models:
            res = [_evaluate_seed(neighbor, seed).q_ratio for seed in run_config.sweep.seeds]
            neighbor_q.append(float(np.median(res)))
            g21_vals = [_evaluate_seed(neighbor, seed).g21_eff for seed in run_config.sweep.seeds]
            sensitivity.setdefault(name, {"x": [], "y": []})
            sensitivity[name]["x"].append(value)
            sensitivity[name]["y"].append(float(np.median(g21_vals)))
        neighborhood_penalty = float(np.std(neighbor_q)) if neighbor_q else 0.0
        max_r2 = 0.0
        for data in sensitivity.values():
            x = np.array(data["x"])
            y = np.array(data["y"])
            max_r2 = max(max_r2, parameter_r2(x, y))
        aggregated = _aggregate_seed_results(
            seed_results,
            model,
            run_config.sweep.objective,
            neighborhood_penalty=neighborhood_penalty,
        )
        aggregated["neighborhood_penalty"] = neighborhood_penalty
        plateau_pass = plateau_check(neighbor_q)
        aggregated["plateau_pass"] = plateau_pass
        aggregated["max_param_r2"] = max_r2
        aggregated["fine_tune_flag"] = bool(max_r2 > 0.9 and not plateau_pass)
        aggregated["config_id"] = row["config_id"]
        aggregated["overrides"] = json.dumps(overrides)
        rows.append(aggregated)
    refined = pd.DataFrame(rows)
    refined.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "config.json").write_text((input_dir / "config.json").read_text())
    plot_diagnostics(refined, out_dir / "figures")
    return out_dir
