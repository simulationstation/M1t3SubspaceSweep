"""Plot the soup graph with M3/M2/M1 annotations."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx

from alpha_soup_wave.soup.m1_strings import StringPopulation

M3_COLOR = "#2ca02c"
M2_COLOR = "#1f77b4"
M1_COLOR = "#d62728"


@dataclass(frozen=True)
class SoupPlotSpec:
    m3_color: str = M3_COLOR
    m2_color: str = M2_COLOR
    m1_color: str = M1_COLOR


def plot_soup(
    graph: nx.Graph,
    m2_nodes: set[int],
    strings: StringPopulation,
    ax: plt.Axes | None = None,
    seed: int | None = 0,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    pos = nx.spring_layout(graph, seed=seed)
    node_colors = []
    for node in graph.nodes:
        node_colors.append(M2_COLOR if node in m2_nodes else M3_COLOR)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, ax=ax, node_size=80)
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.6)
    if strings.specs:
        for idx, spec in enumerate(strings.specs):
            anchor_pos = pos[spec.anchor]
            offset = ((idx % 3) - 1) * 0.05
            string_pos = (anchor_pos[0] + 0.1 + offset, anchor_pos[1] + 0.1 + offset)
            ax.plot(
                [anchor_pos[0], string_pos[0]],
                [anchor_pos[1], string_pos[1]],
                color=M1_COLOR,
                linewidth=2,
            )
            ax.scatter([string_pos[0]], [string_pos[1]], color=M1_COLOR, s=30)
    ax.set_axis_off()
    return ax
