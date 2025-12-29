"""Local refinement around promising candidates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alpha_soup_wave.config import ModelConfig, SweepConfig
from alpha_soup_wave.search.sweep import run_sweep


def _perturb_values(value: float, scale: float, rng: np.random.Generator) -> float:
    return float(value * (1.0 + rng.normal(0.0, scale)))


def _config_from_row(row: pd.Series) -> ModelConfig:
    return ModelConfig(
        network={
            "n3": int(row["n3"]),
            "degree2_fraction": float(row["degree2_fraction"]),
            "coupling_j": float(row["coupling_j"]),
            "temperature": float(row["temperature"]),
        },
        m2={
            "coherence_threshold": float(row["coherence_threshold"]),
            "loopiness_threshold": float(row["loopiness_threshold"]),
            "min_size": int(row["m2_min_size"]),
        },
        strings={
            "anchor_fraction": float(row["anchor_fraction"]),
            "length_mean": float(row["length_mean"]),
            "length_sigma": float(row["length_sigma"]),
            "tension_mean": float(row["tension_mean"]),
            "density_mean": float(row["density_mean"]),
            "damping_mean": float(row["damping_mean"]),
            "boundary_choices": str(row["boundary_choices"]).split(","),
            "mode_count": int(row["mode_count"]),
        },
    )


def refine_from_results(
    results_path: str,
    output_dir: str,
    seed: int = 42,
    n_samples: int = 30,
) -> tuple[pd.DataFrame, list[dict]]:
    summary_path = Path(results_path).with_name("summary.csv")
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
    else:
        summary = pd.read_csv(results_path)
    best = summary.nsmallest(30, "score")
    rng = np.random.default_rng(seed)
    configs: list[ModelConfig] = []

    for _, row in best.iterrows():
        base_config = _config_from_row(row)
        for _ in range(n_samples):
            configs.append(
                ModelConfig(
                    network={
                        "n3": base_config.network.n3,
                        "degree2_fraction": base_config.network.degree2_fraction,
                        "coupling_j": _perturb_values(base_config.network.coupling_j, 0.05, rng),
                        "temperature": _perturb_values(base_config.network.temperature, 0.05, rng),
                    },
                    m2={
                        "coherence_threshold": _perturb_values(base_config.m2.coherence_threshold, 0.03, rng),
                        "loopiness_threshold": _perturb_values(base_config.m2.loopiness_threshold, 0.03, rng),
                        "min_size": base_config.m2.min_size,
                    },
                    strings={
                        "anchor_fraction": _perturb_values(base_config.strings.anchor_fraction, 0.05, rng),
                        "length_mean": _perturb_values(base_config.strings.length_mean, 0.05, rng),
                        "length_sigma": base_config.strings.length_sigma,
                        "tension_mean": base_config.strings.tension_mean,
                        "density_mean": base_config.strings.density_mean,
                        "damping_mean": base_config.strings.damping_mean,
                        "boundary_choices": base_config.strings.boundary_choices,
                        "mode_count": base_config.strings.mode_count,
                    },
                    spectral=base_config.spectral,
                    mechanisms=base_config.mechanisms,
                    selection=base_config.selection,
                    gate=base_config.gate,
                )
            )

    sweep = SweepConfig(seeds=range(10), max_configs=len(configs), neighborhood_samples=10)
    return run_sweep(configs, sweep, output_dir, label="refine")


def refine_cli(input_dir: str, output_dir: str) -> None:
    results_path = Path(input_dir) / "results.csv"
    refine_from_results(str(results_path), output_dir)

