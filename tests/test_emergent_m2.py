from alpha_soup.config import SoupConfig
from alpha_soup.soup import generate_degree_graph, initialize_state, evolve, detect_m2


def test_emergent_m2_non_empty():
    cfg = SoupConfig(n_nodes=80)
    cfg.emergent.coherence_thresh = 0.3
    cfg.emergent.loop_thresh = 0.1
    cfg.dynamics.steps = 30
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
    assert len(m2.nodes) > 0
