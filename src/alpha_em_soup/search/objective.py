"""Objective scoring for configurations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alpha_em_soup.constants import ALPHA_TARGET


@dataclass(frozen=True)
class ObjectiveInputs:
    q_ratio: float
    m2_instability: float
    seed_std: float
    neighborhood_penalty: float
    g21_eff: float


def score_objective(inputs: ObjectiveInputs, weight_log_q: float, weight_m2_instability: float,
                    weight_seed_std: float, weight_neighborhood: float, weight_alpha_distance: float) -> float:
    log_q = abs(np.log(max(inputs.q_ratio, 1e-6)))
    alpha_distance = abs(np.log(max(inputs.g21_eff, 1e-6) / ALPHA_TARGET))
    return float(
        weight_log_q * log_q
        + weight_m2_instability * inputs.m2_instability
        + weight_seed_std * inputs.seed_std
        + weight_neighborhood * inputs.neighborhood_penalty
        + weight_alpha_distance * alpha_distance
    )
