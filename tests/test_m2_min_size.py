import networkx as nx

from alpha_em_soup.layers.m2_interface import detect_m2


def test_m2_min_size_enforced():
    graph = nx.cycle_graph(20)
    m2 = detect_m2(graph, coherence_threshold=0.9, loopiness_threshold=1.0, min_fraction=0.2)
    assert len(m2.nodes) >= int(0.2 * graph.number_of_nodes())
