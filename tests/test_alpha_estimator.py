import numpy as np

from alpha_soup.search.objectives import estimate_alpha
from alpha_soup.soup.observables import ObservableBundle


def test_alpha_estimator_dimensionless():
    obs = ObservableBundle(
        spectral_gap_m2=0.2,
        spectral_radius_m3=4.0,
        corr_length_m2=1.5,
        stiffness_m2=0.3,
        stiffness_m3=0.9,
        overlap_23=0.4,
        stability_m2=0.1,
    )
    est = estimate_alpha(obs)
    assert np.isfinite(est.alpha_gap_ratio)
    assert np.isfinite(est.alpha_stiffness_ratio)
    assert np.isfinite(est.alpha_overlap_gap)
