"""Configuration objects for alpha soup wave simulations."""

from __future__ import annotations

from typing import Iterable

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


class SweepConfig(BaseModel):
    seeds: Iterable[int] = Field(default_factory=lambda: range(5))
    perturbation_fraction: float = 0.05
    tolerance_log: float = 0.2
    max_configs: int = 300


class ModelConfig(BaseModel):
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    m2: M2Config = Field(default_factory=M2Config)
    strings: StringConfig = Field(default_factory=StringConfig)


class RunConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    sweep: SweepConfig = Field(default_factory=SweepConfig)
    label: str = "default"
