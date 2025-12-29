"""Robustness checks against fine-tuned hits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_soup_wave.constants import ALPHA_TARGET


@dataclass(frozen=True)
class RobustnessSummary:
    plateau_pass: bool
    fine_tune_flag: bool
    neighborhood_penalty: float
    fine_tune_penalty: float


def log_distance(alpha_hat: float, alpha_target: float = ALPHA_TARGET) -> float:
    if alpha_hat <= 0:
        return float("inf")
    return float(abs(np.log(alpha_hat) - np.log(alpha_target)))


def parameter_r2(results: pd.DataFrame, param_columns: list[str], target_column: str) -> dict[str, float]:
    r2_values = {}
    y = results[target_column].values
    for col in param_columns:
        x = results[col].values
        if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
            r2_values[col] = 0.0
            continue
        corr = np.corrcoef(x, y)[0, 1]
        r2_values[col] = float(corr**2)
    return r2_values


def plateau_check(base: float, neighbors: list[float], rel_tol: float) -> bool:
    if base <= 0:
        return False
    for val in neighbors:
        if val <= 0:
            return False
        if abs(val - base) / base > rel_tol:
            return False
    return True


def fine_tune_flag_for_config(
    r2_values: dict[str, float],
    plateau_pass: bool,
    threshold: float = 0.9,
) -> bool:
    if plateau_pass:
        return False
    return any(value > threshold for value in r2_values.values())


