"""Adaptive refinement around top candidates."""
from __future__ import annotations

import json
import hashlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_soup.config import SoupConfig, SearchConfig
from alpha_soup.rng import make_rng
from alpha_soup.search.sweep import _run_single


def _config_hash(config: SoupConfig) -> str:
    payload = json.dumps(config.model_dump(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
from alpha_soup.search.robustness import best_candidates


def _perturb_config(config: SoupConfig, rng: np.random.Generator, scale: float) -> SoupConfig:
    couplings = config.couplings.model_copy()
    couplings.j3 = max(couplings.j3 * (1 + rng.normal(0, scale)), 0.01)
    couplings.j2 = max(couplings.j2 * (1 + rng.normal(0, scale)), 0.01)
    couplings.j1 = max(couplings.j1 * (1 + rng.normal(0, scale)), 0.01)
    couplings.g23 = max(couplings.g23 * (1 + rng.normal(0, scale)), 0.001)
    couplings.g21 = max(couplings.g21 * (1 + rng.normal(0, scale)), 0.001)

    emergent = config.emergent.model_copy()
    emergent.coherence_thresh = float(np.clip(emergent.coherence_thresh + rng.normal(0, scale * 0.5), 0.2, 0.9))
    emergent.loop_thresh = float(np.clip(emergent.loop_thresh + rng.normal(0, scale * 0.5), 0.05, 0.8))

    dynamics = config.dynamics.model_copy()
    dynamics.temperature = max(dynamics.temperature * (1 + rng.normal(0, scale)), 0.05)

    return SoupConfig(
        n_nodes=config.n_nodes,
        profile=config.profile,
        couplings=couplings,
        emergent=emergent,
        dynamics=dynamics,
        decay=config.decay,
        seed=config.seed,
    )


def refine_from_sweep(in_dir: Path, out_dir: Path, search_cfg: SearchConfig) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(in_dir / "results.csv")
    configs_raw = json.loads((in_dir / "configs_used.json").read_text())
    summary = best_candidates(results, search_cfg.top_k)
    rng = make_rng(2024)
    tasks: list[tuple[SoupConfig, int]] = []
    configs_used: dict[str, dict] = {}
    for _, row in summary.iterrows():
        cfg_hash = row["config_hash"]
        config = SoupConfig(**configs_raw[cfg_hash])
        for _ in range(search_cfg.neighborhood_samples):
            perturbed = _perturb_config(config, rng, search_cfg.neighborhood_perturb)
            configs_used[_config_hash(perturbed)] = perturbed.model_dump()
            for seed in range(search_cfg.seeds_per_point):
                tasks.append((perturbed, seed))

    with Pool(processes=search_cfg.max_workers) as pool:
        rows = list(pool.map(_run_single, tasks))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    (out_dir / "configs_used.json").write_text(json.dumps(configs_used, indent=2))
    return df
