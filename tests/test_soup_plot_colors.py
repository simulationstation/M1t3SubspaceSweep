import networkx as nx

from alpha_soup_wave.soup.m1_strings import StringPopulation
from alpha_soup_wave.viz.soup_plot import M1_COLOR, M2_COLOR, M3_COLOR, plot_soup


def test_soup_plot_color_mapping():
    assert M3_COLOR == "#2ca02c"
    assert M2_COLOR == "#1f77b4"
    assert M1_COLOR == "#d62728"

    graph = nx.path_graph(3)
    m2_nodes = {1}
    strings = StringPopulation(specs=[])
    ax = plot_soup(graph, m2_nodes, strings)
    colors = [c.get_facecolor() for c in ax.collections if hasattr(c, "get_facecolor")]
    assert colors
