"""Alpha estimators and loss evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alpha_soup.constants import ALPHA_TARGET
from alpha_soup.soup.observables import ObservableBundle


@dataclass
class AlphaEstimates:
    alpha_gap_ratio: float
    alpha_stiffness_ratio: float
    alpha_overlap_gap: float


@dataclass
class ObjectiveResult:
    loss: float
    alpha_estimates: AlphaEstimates


def estimate_alpha(obs: ObservableBundle) -> AlphaEstimates:
    gap_ratio = obs.spectral_gap_m2 / obs.spectral_radius_m3 if obs.spectral_radius_m3 > 0 else 0.0
    stiffness_ratio = obs.stiffness_m2 / obs.stiffness_m3 if obs.stiffness_m3 > 0 else 0.0
    overlap_gap = obs.overlap_23 * (obs.spectral_gap_m2 / (obs.spectral_radius_m3 + 1e-9))
    return AlphaEstimates(
        alpha_gap_ratio=float(gap_ratio),
        alpha_stiffness_ratio=float(stiffness_ratio),
        alpha_overlap_gap=float(overlap_gap),
    )


def objective_loss(est: AlphaEstimates) -> float:
    values = np.array([est.alpha_gap_ratio, est.alpha_stiffness_ratio, est.alpha_overlap_gap])
    values = np.clip(values, 1e-12, None)
    target = ALPHA_TARGET
    return float(np.min(np.abs(np.log(values) - np.log(target))))


def evaluate_objective(obs: ObservableBundle) -> ObjectiveResult:
    estimates = estimate_alpha(obs)
    loss = objective_loss(estimates)
    return ObjectiveResult(loss=loss, alpha_estimates=estimates)
