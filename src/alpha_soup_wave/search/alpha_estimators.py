"""Estimate alpha from internal scales for v3 mechanisms."""

from __future__ import annotations

import numpy as np

from alpha_soup_wave.config import MechanismConfig
from alpha_soup_wave.soup.observables import SoupObservables


def alpha_hat_attenuation(observables: SoupObservables, weighted: bool) -> float:
    if observables.attenuation.size == 0:
        return float("nan")
    values = observables.attenuation
    if weighted:
        values = values * (0.5 + 0.5 * (observables.m2_size > 0))
    return float(np.clip(np.median(values), 0.0, 1.0))


def alpha_hat_overlap(observables: SoupObservables) -> float:
    if observables.overlap.size == 0:
        return float("nan")
    return float(np.clip(np.sqrt(np.median(observables.overlap)), 0.0, 1.0))


def alpha_hat_selection(observables: SoupObservables) -> float:
    return float(np.clip(observables.selection_eff, 0.0, 1.0))


def alpha_hat_gate(observables: SoupObservables) -> float:
    pA = observables.gate_pA
    pB = observables.gate_pB
    if pA <= 0 or pB <= 0:
        return 0.0
    if observables.gate_pAB > 0:
        rho = observables.gate_pAB / (pA * pB)
    else:
        rho = 0.0
    return float(np.clip(np.sqrt(pA * pB) * np.sqrt(max(rho, 0.0)), 0.0, 1.0))


def all_alpha_hats(observables: SoupObservables, mechanisms: MechanismConfig) -> dict[str, float]:
    hats = {
        "alpha_hat_att": alpha_hat_attenuation(observables, mechanisms.attenuation_weight_m2),
        "alpha_hat_overlap": alpha_hat_overlap(observables),
    }
    if mechanisms.enable_selection:
        hats["alpha_hat_sel"] = alpha_hat_selection(observables)
    else:
        hats["alpha_hat_sel"] = float("nan")
    if mechanisms.enable_gating:
        hats["alpha_hat_gate"] = alpha_hat_gate(observables)
    else:
        hats["alpha_hat_gate"] = float("nan")
    return hats


def combined_alpha_hat(hats: dict[str, float]) -> float:
    values = [val for val in hats.values() if np.isfinite(val) and 0 < val < 1]
    if not values:
        return float("nan")
    return float(np.median(values))


