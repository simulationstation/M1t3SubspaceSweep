"""Robustness checks and sensitivity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensitivityResult:
    parameter: str
    r2: float


def log_distance(value: float) -> float:
    return float(abs(np.log(max(value, 1e-6))))


def plateau_check(values: list[float], tolerance: float = 0.2) -> bool:
    if not values:
        return False
    ref = np.median(values)
    return all(abs(val - ref) / max(ref, 1e-6) <= tolerance for val in values)


def parameter_r2(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    coeffs = np.polyfit(x, y, deg=1)
    preds = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
