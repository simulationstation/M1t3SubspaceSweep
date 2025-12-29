"""Configuration models for the EM soup simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class M3Config(BaseModel):
    n3: int = 60
    generator: str = "triangle_boosted"
    min_short_cycle_ratio: float = 0.08
    triangle_boost_steps: int = 200
    geometric_radius: float = 0.35
    planted_cycle_count: int = 6


class M2Config(BaseModel):
    mode: str = "emergent"  # emergent or constructed
    min_fraction: float = 0.1
    coherence_threshold: float = 0.55
    loopiness_threshold: float = 0.05
    constructed_strength: float = 1.4
    boundary_strength: float = 0.3


class M1Config(BaseModel):
    anchor_fraction: float = 0.3
    path_count: int = 24
    length_mean: float = 4.2
    length_sigma: float = 1.2
    loop_fraction: float = 0.4
    loopback_fraction: float = 0.2
    attenuation_length: float = 8.0
    edge_loss: float = 0.08
    max_path_edges: int = 5


class PhysicsConfig(BaseModel):
    omega_min: float = 0.6
    omega_max: float = 3.0
    omega_points: int = 24
    k_scale: float = 1.0
    damping_m1: float = 0.12
    damping_m2: float = 0.08
    damping_m3: float = 0.05


class CouplingConfig(BaseModel):
    j3: float = 1.0
    j2: float = 1.2
    j23: float = 0.35
    drive_strength: float = 1.0
    drive_fraction: float = 0.2


class GateConfig(BaseModel):
    enable: bool = True
    coherence_min: float = 0.6
    resonance_min: float = 0.4


class ObjectiveConfig(BaseModel):
    weight_log_q: float = 1.0
    weight_m2_instability: float = 0.7
    weight_seed_std: float = 0.4
    weight_neighborhood: float = 0.3
    weight_alpha_distance: float = 0.1
    q_tolerance: float = 0.25


class ModelConfig(BaseModel):
    m3: M3Config = Field(default_factory=M3Config)
    m2: M2Config = Field(default_factory=M2Config)
    m1: M1Config = Field(default_factory=M1Config)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    coupling: CouplingConfig = Field(default_factory=CouplingConfig)
    gate: GateConfig = Field(default_factory=GateConfig)


class SweepConfig(BaseModel):
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    config_count: int = 10
    param_grid: dict[str, list[Any]] = Field(default_factory=dict)
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    neighborhood_samples: int = 5
    neighborhood_perturb: float = 0.05


class RunConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    sweep: SweepConfig = Field(default_factory=SweepConfig)


def load_config(path: str | Path) -> RunConfig:
    data = Path(path).read_text()
    return RunConfig.model_validate_json(data)


def expand_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not param_grid:
        return [{}]
    keys = list(param_grid)
    values = [param_grid[key] for key in keys]
    combos: list[dict[str, Any]] = []
    for selection in _product(values):
        combos.append(dict(zip(keys, selection, strict=True)))
    return combos


def _product(values: list[list[Any]]):
    if not values:
        yield []
        return
    first, *rest = values
    for item in first:
        for tail in _product(rest):
            yield [item, *tail]


def apply_overrides(model: ModelConfig, overrides: dict[str, Any]) -> ModelConfig:
    data = model.model_dump()
    for dotted, value in overrides.items():
        target = data
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    return ModelConfig.model_validate(data)
