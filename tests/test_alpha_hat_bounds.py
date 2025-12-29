import numpy as np

from alpha_em_soup.config import ModelConfig
from alpha_em_soup.layers.m1_wavepaths import build_m1_wavepaths
from alpha_em_soup.layers.m2_interface import build_m2
from alpha_em_soup.layers.m3_topology import generate_m3_graph
from alpha_em_soup.physics.estimators import estimate_alpha_candidates


def test_alpha_hat_bounded():
    model = ModelConfig()
    m3 = generate_m3_graph(
        model.m3.n3,
        0,
        model.m3.generator,
        model.m3.min_short_cycle_ratio,
        model.m3.triangle_boost_steps,
        model.m3.geometric_radius,
        model.m3.planted_cycle_count,
    )
    m2 = build_m2(
        m3.graph,
        0,
        model.m2.mode,
        model.m2.coherence_threshold,
        model.m2.loopiness_threshold,
        model.m2.min_fraction,
    )
    m1 = build_m1_wavepaths(
        sorted(m2.nodes),
        0,
        model.m1.path_count,
        model.m1.length_mean,
        model.m1.length_sigma,
        model.m1.loop_fraction,
        model.m1.loopback_fraction,
        model.m1.max_path_edges,
    )
    omega_grid = np.linspace(model.physics.omega_min, model.physics.omega_max, model.physics.omega_points)
    candidates = estimate_alpha_candidates(
        m1.paths,
        m1.graph,
        m3.graph.subgraph(m2.nodes).copy(),
        omega_grid,
        model.m1.attenuation_length,
        model.physics.damping_m1,
        model.physics.damping_m2,
        m1.anchor_nodes,
    )
    for value in [candidates.alpha_hat_cov, candidates.alpha_hat_ov, candidates.alpha_hat_z]:
        assert 0.0 <= value <= 1.0
