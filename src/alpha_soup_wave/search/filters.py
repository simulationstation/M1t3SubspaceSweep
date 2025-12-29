"""Filter criteria for robustness and stability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StabilityMetrics:
    mean: float
    std: float
    log_distance: float
    m2_mean: float
    m2_std: float


def compute_log_distance(alpha_hat: float, alpha_target: float) -> float:
    return float(abs(np.log(alpha_hat) - np.log(alpha_target)))


def is_stable(
    metrics: StabilityMetrics,
    tolerance_log: float,
    std_threshold: float,
    m2_nonempty: bool,
    m2_std_threshold: float,
) -> bool:
    return (
        metrics.log_distance <= tolerance_log
        and metrics.std <= std_threshold
        and m2_nonempty
        and metrics.m2_std <= m2_std_threshold
    )
