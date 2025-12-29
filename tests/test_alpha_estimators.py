import numpy as np

from alpha_soup_wave.search.alpha_estimators import all_alpha_hats
from alpha_soup_wave.soup.observables import SoupObservables


def test_alpha_estimators_finite_and_bounded():
    observables = SoupObservables(
        omega_star=2.0,
        z_net_star=0.8,
        z_str_star=np.array([1.2, 1.4, 1.1]),
        transmission_star=np.array([0.1, 0.2, 0.15]),
        leak_rate=0.05,
        delta_omega_net=0.3,
        delta_omega_str=0.05,
    )
    hats = all_alpha_hats(observables)
    for key, value in hats.items():
        assert np.isfinite(value), f"{key} not finite"
        assert value > 0
    assert 0 < hats["alpha_hat_T"] < 1
    assert 0 < hats["alpha_hat_Z"] < 1
