from pathlib import Path

from alpha_em_soup.config import ModelConfig
from alpha_em_soup.search.sweep import _evaluate_seed
from alpha_em_soup.viz.soup_plot import plot_soup


def test_color_map(tmp_path: Path):
    model = ModelConfig()
    result = _evaluate_seed(model, seed=0)
    output = plot_soup(model, result, tmp_path)
    assert output.exists()
