import networkx as nx
import numpy as np

from alpha_soup_wave.soup.m2_detector import detect_m2


def test_m2_detector_nontrivial():
    graph = nx.cycle_graph(6)
    theta = np.linspace(0, 0.5, 6)
    m2 = detect_m2(graph, theta, coherence_threshold=0.2, loopiness_threshold=0.1)
    assert len(m2) > 0
