"""Configuration objects for alpha soup wave simulations."""

from __future__ import annotations

import os
from typing import Iterable, Sequence

from pydantic import BaseModel, Field


class NetworkConfig(BaseModel):
    n3: int = 200
    degree2_fraction: float = 0.0
    coupling_j: float = 1.0
    temperature: float = 0.4
    loop_penalty: float = 0.0
    metropolis_steps: int = 3000


class M2Config(BaseModel):
    coherence_threshold: float = 0.7
    loopiness_threshold: float = 0.1
    min_size: int | None = None


class StringConfig(BaseModel):
    anchor_fraction: float = 0.25
    length_mean: float = 5.0
    length_sigma: float = 0.6
    tension_mean: float = 1.4
    tension_sigma: float = 0.3
    density_mean: float = 0.9
    density_sigma: float = 0.2
    damping_mean: float = 0.08
    damping_sigma: float = 0.03
    boundary_choices: Sequence[str] = ("fixed", "free")
    mode_count: int = 6


class SpectralConfig(BaseModel):
    omega_max_factor: float = 2.5
    grid_size: int = 160
    kernel_sigma: float = 0.2
    weight_scale: float = 1.0


class MechanismConfig(BaseModel):
    attenuation_weight_m2: bool = True
    enable_selection: bool = True
    enable_gating: bool = True


class SelectionConfig(BaseModel):
    scale: float = 0.35


class GateConfig(BaseModel):
    coherence_gate: float = 0.75
    size_stability_fraction: float = 0.2
    resonance_k: int = 2
    resonance_eps: float = 0.12
    low_damping_quantile: float = 0.35
    samples: int = 8
    length_jitter_fraction: float = 0.05
    theta_noise_scale: float = 0.08


class SweepConfig(BaseModel):
    seeds: Iterable[int] = Field(default_factory=lambda: range(3))
    perturbation_fraction: float = 0.05
    tolerance_log: float = 0.25
    max_configs: int = 300
    seed_std_weight: float = 0.6
    neighbor_weight: float = 0.6
    m2_instability_weight: float = 0.5
    fine_tune_weight: float = 0.8
    top_k: int = 30
    refine_seeds: int = 10
    refine_neighborhood_samples: int = 10
    neighborhood_samples: int = 2
    plateau_rel_tol: float = 0.2
    max_workers: int | None = os.cpu_count()


class ModelConfig(BaseModel):
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    m2: M2Config = Field(default_factory=M2Config)
    strings: StringConfig = Field(default_factory=StringConfig)
    spectral: SpectralConfig = Field(default_factory=SpectralConfig)
    mechanisms: MechanismConfig = Field(default_factory=MechanismConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    gate: GateConfig = Field(default_factory=GateConfig)


class RunConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    sweep: SweepConfig = Field(default_factory=SweepConfig)
    label: str = "default"


def default_m2_min(n3: int) -> int:
    if n3 >= 400:
        return 60
    return 30


def expand_values(value) -> list:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]
