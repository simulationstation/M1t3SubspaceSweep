import numpy as np

from alpha_soup_wave.config import ModelConfig
from alpha_soup_wave.search.sweep import _run_single_seed


def test_alpha_hat_bounds():
    model = ModelConfig(
        network={"n3": 50, "coupling_j": 1.0, "temperature": 0.4, "metropolis_steps": 200},
        m2={"coherence_threshold": 0.5, "loopiness_threshold": 0.05, "min_size": 5},
        strings={"anchor_fraction": 0.3, "length_mean": 4.5},
    )
    alpha_hat, _, _, hats, _ = _run_single_seed(model, seed=7)
    assert np.isfinite(alpha_hat)
    for value in hats.values():
        assert 0.0 <= value <= 1.0
