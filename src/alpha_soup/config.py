"""Configuration models for soup simulations and searches."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DegreeProfile(BaseModel):
    frac_deg3: float = 0.7
    frac_deg2: float = 0.2
    frac_deg1: float = 0.1


class Couplings(BaseModel):
    j3: float = 1.4
    j2: float = 0.8
    j1: float = 0.4
    g23: float = 0.12
    g21: float = 0.08


class EmergentCriteria(BaseModel):
    coherence_thresh: float = 0.65
    loop_thresh: float = 0.25


class DynamicsConfig(BaseModel):
    temperature: float = 0.6
    steps: int = 200
    step_size: float = 0.08
    noise: float = 0.2
    use_langevin: bool = True


class DecayConfig(BaseModel):
    enabled: bool = False
    rate_deg3_to_deg2: float = 0.01
    rate_deg2_to_deg1: float = 0.005


class SoupConfig(BaseModel):
    n_nodes: int = 200
    profile: DegreeProfile = Field(default_factory=DegreeProfile)
    couplings: Couplings = Field(default_factory=Couplings)
    emergent: EmergentCriteria = Field(default_factory=EmergentCriteria)
    dynamics: DynamicsConfig = Field(default_factory=DynamicsConfig)
    decay: DecayConfig = Field(default_factory=DecayConfig)
    seed: int = 1234


class SearchConfig(BaseModel):
    seeds_per_point: int = 3
    top_k: int = 5
    alpha_tolerance: float = 0.12
    neighborhood_perturb: float = 0.08
    neighborhood_samples: int = 6
    require_stable_points: int = 3
    max_workers: Optional[int] = None


class SweepRange(BaseModel):
    n_nodes: list[int] = Field(default_factory=lambda: [200, 400])
    frac_deg3: list[float] = Field(default_factory=lambda: [0.6, 0.7, 0.8])
    frac_deg2: list[float] = Field(default_factory=lambda: [0.1, 0.2])
    j3: list[float] = Field(default_factory=lambda: [1.2, 1.5])
    j2: list[float] = Field(default_factory=lambda: [0.6, 0.9])
    j1: list[float] = Field(default_factory=lambda: [0.3, 0.5])
    g23: list[float] = Field(default_factory=lambda: [0.05, 0.12, 0.2])
    g21: list[float] = Field(default_factory=lambda: [0.05, 0.1])
    temperature: list[float] = Field(default_factory=lambda: [0.4, 0.6])
    coherence_thresh: list[float] = Field(default_factory=lambda: [0.6, 0.7])
    loop_thresh: list[float] = Field(default_factory=lambda: [0.2, 0.3])


class SweepConfig(BaseModel):
    sweep: SweepRange = Field(default_factory=SweepRange)
    search: SearchConfig = Field(default_factory=SearchConfig)
    fast_mode_steps: int = 120
