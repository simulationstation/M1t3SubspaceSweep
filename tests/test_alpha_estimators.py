import numpy as np

from alpha_soup_wave.config import MechanismConfig
from alpha_soup_wave.search.alpha_estimators import all_alpha_hats
from alpha_soup_wave.soup.observables import SoupObservables


def test_alpha_estimators_finite_and_bounded():
    observables = SoupObservables(
        attenuation=np.array([0.1, 0.2, 0.15]),
        overlap=np.array([0.04, 0.09, 0.16]),
        selection_eff=0.3,
        gate_pA=0.25,
        gate_pB=0.4,
        gate_pAB=0.1,
        m2_size=40,
        m2_coherence_mean=0.8,
        m2_valid=True,
    )
    hats = all_alpha_hats(observables, MechanismConfig())
    for key, value in hats.items():
        assert np.isfinite(value), f"{key} not finite"
        assert 0.0 <= value <= 1.0
