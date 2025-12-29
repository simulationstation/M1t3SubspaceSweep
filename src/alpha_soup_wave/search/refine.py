"""Local refinement around promising candidates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alpha_soup_wave.config import ModelConfig, NetworkConfig, M2Config, StringConfig, SweepConfig
from alpha_soup_wave.search.sweep import run_sweep


def _perturb_values(value: float, scale: float, rng: np.random.Generator) -> float:
    return float(value * (1.0 + rng.normal(0.0, scale)))


def refine_from_results(
    results_path: str,
    output_dir: str,
    seed: int = 42,
    n_samples: int = 50,
) -> tuple[pd.DataFrame, list[dict]]:
    results = pd.read_csv(results_path)
    best = results.nsmallest(3, "alpha_log_distance")
    rng = np.random.default_rng(seed)
    configs: list[ModelConfig] = []

    for _, row in best.iterrows():
        base_network = NetworkConfig(
            n3=int(200),
            coupling_j=1.0,
            temperature=0.4,
        )
        base_strings = StringConfig(length_mean=5.0)
        for _ in range(n_samples):
            configs.append(
                ModelConfig(
                    network=NetworkConfig(
                        n3=base_network.n3,
                        coupling_j=_perturb_values(base_network.coupling_j, 0.1, rng),
                        temperature=_perturb_values(base_network.temperature, 0.1, rng),
                    ),
                    m2=M2Config(
                        coherence_threshold=_perturb_values(0.7, 0.05, rng),
                        loopiness_threshold=_perturb_values(0.1, 0.05, rng),
                    ),
                    strings=StringConfig(
                        length_mean=_perturb_values(base_strings.length_mean, 0.1, rng),
                        anchor_fraction=_perturb_values(0.25, 0.1, rng),
                    ),
                )
            )

    sweep = SweepConfig()
    return run_sweep(configs, sweep, output_dir, label="refine")


def refine_cli(input_dir: str, output_dir: str) -> None:
    results_path = Path(input_dir) / "results.csv"
    refine_from_results(str(results_path), output_dir)
