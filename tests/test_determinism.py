import numpy as np

from alpha_em_soup.config import ModelConfig
from alpha_em_soup.search.sweep import _evaluate_seed


def test_determinism_fixed_seed():
    model = ModelConfig()
    result_a = _evaluate_seed(model, seed=123)
    result_b = _evaluate_seed(model, seed=123)
    assert np.isclose(result_a.g21_eff, result_b.g21_eff)
    assert np.isclose(result_a.g23_eff, result_b.g23_eff)
    assert result_a.m2_size == result_b.m2_size
