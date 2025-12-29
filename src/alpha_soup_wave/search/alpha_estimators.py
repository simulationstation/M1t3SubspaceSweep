"""Estimate alpha from internal scales."""

from __future__ import annotations

import numpy as np

from alpha_soup_wave.soup.observables import SoupObservables


def alpha_hat_t(observables: SoupObservables) -> float:
    return float(np.sqrt(np.median(observables.transmission_star)))


def alpha_hat_z(observables: SoupObservables) -> float:
    ratio = observables.z_net_star / np.median(observables.z_str_star)
    return float(ratio / (1.0 + ratio))


def alpha_hat_g(observables: SoupObservables) -> float:
    return float(observables.leak_rate / observables.omega_star)


def alpha_hat_res(observables: SoupObservables) -> float:
    return float(observables.delta_omega_str / observables.delta_omega_net)


def all_alpha_hats(observables: SoupObservables) -> dict[str, float]:
    return {
        "alpha_hat_T": alpha_hat_t(observables),
        "alpha_hat_Z": alpha_hat_z(observables),
        "alpha_hat_G": alpha_hat_g(observables),
        "alpha_hat_res": alpha_hat_res(observables),
    }
