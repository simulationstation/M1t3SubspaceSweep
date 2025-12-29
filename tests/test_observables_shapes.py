import numpy as np

from alpha_soup.config import SoupConfig
from alpha_soup.soup import generate_degree_graph, initialize_state, evolve, detect_m2, compute_observables


def test_observables_finite():
    cfg = SoupConfig(n_nodes=60)
    cfg.dynamics.steps = 20
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
    for value in obs.__dict__.values():
        assert np.isfinite(value)
