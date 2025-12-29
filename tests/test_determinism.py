import numpy as np

from alpha_soup.config import SoupConfig
from alpha_soup.soup import generate_degree_graph, initialize_state, evolve, detect_m2, compute_observables
from alpha_soup.search.objectives import estimate_alpha


def test_determinism_seed():
    cfg = SoupConfig(n_nodes=80)
    cfg.dynamics.steps = 50
    cfg.seed = 42

    graph_bundle = generate_degree_graph(cfg.n_nodes, cfg.profile, cfg.seed)
    state = initialize_state(cfg.n_nodes, cfg.seed)
    evolved = evolve(
        state,
        list(graph_bundle.graph.edges),
        graph_bundle.degree_labels,
        cfg.couplings,
        cfg.dynamics,
        cfg.seed,
    )
    m2 = detect_m2(graph_bundle.graph, evolved.theta, cfg.emergent)
    obs = compute_observables(graph_bundle.graph, m2.nodes, graph_bundle.degree_labels, evolved.theta)
    alpha_1 = estimate_alpha(obs)

    graph_bundle2 = generate_degree_graph(cfg.n_nodes, cfg.profile, cfg.seed)
    state2 = initialize_state(cfg.n_nodes, cfg.seed)
    evolved2 = evolve(
        state2,
        list(graph_bundle2.graph.edges),
        graph_bundle2.degree_labels,
        cfg.couplings,
        cfg.dynamics,
        cfg.seed,
    )
    m2_2 = detect_m2(graph_bundle2.graph, evolved2.theta, cfg.emergent)
    obs2 = compute_observables(graph_bundle2.graph, m2_2.nodes, graph_bundle2.degree_labels, evolved2.theta)
    alpha_2 = estimate_alpha(obs2)

    assert np.isclose(alpha_1.alpha_gap_ratio, alpha_2.alpha_gap_ratio)
    assert np.isclose(alpha_1.alpha_stiffness_ratio, alpha_2.alpha_stiffness_ratio)
    assert np.isclose(alpha_1.alpha_overlap_gap, alpha_2.alpha_overlap_gap)
