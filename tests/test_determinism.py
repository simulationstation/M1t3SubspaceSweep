import numpy as np

from alpha_soup_wave.config import ModelConfig, M2Config, NetworkConfig, StringConfig
from alpha_soup_wave.search.sweep import _run_single_seed


def test_determinism_fixed_seed():
    model = ModelConfig(
        network=NetworkConfig(n3=60, coupling_j=1.0, temperature=0.4, metropolis_steps=500),
        m2=M2Config(coherence_threshold=0.6, loopiness_threshold=0.05, min_size=5),
        strings=StringConfig(anchor_fraction=0.3, length_mean=4.5),
    )
    alpha_a, m2_a, valid_a, _, _ = _run_single_seed(model, seed=123)
    alpha_b, m2_b, valid_b, _, _ = _run_single_seed(model, seed=123)
    assert np.isclose(alpha_a, alpha_b)
    assert m2_a == m2_b
    assert valid_a == valid_b
