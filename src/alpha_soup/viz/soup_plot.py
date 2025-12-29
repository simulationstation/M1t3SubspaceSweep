"""Soup visualization."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def plot_soup(graph: nx.Graph, degree_labels: dict[int, int], m2_nodes: list[int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pos = nx.spring_layout(graph, seed=42)
    colors = []
    for node in graph.nodes:
        if node in m2_nodes:
            colors.append("#1f77b4")  # blue
        else:
            deg = degree_labels[node]
            if deg == 3:
                colors.append("#2ca02c")  # green
            elif deg == 1:
                colors.append("#d62728")  # red
            else:
                colors.append("#7f7f7f")  # gray

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=80, alpha=0.9)
    nx.draw_networkx_edges(graph, pos, alpha=0.2)
    plt.axis("off")
    counts = {
        "M3": sum(1 for d in degree_labels.values() if d == 3),
        "M2": len(m2_nodes),
        "M1": sum(1 for d in degree_labels.values() if d == 1),
        "deg2": sum(1 for d in degree_labels.values() if d == 2),
    }
    plt.title(f"Soup layers: M3={counts['M3']} M2={counts['M2']} M1={counts['M1']} deg2={counts['deg2']}")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="M3 (deg3)", markerfacecolor="#2ca02c", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="M2 (emergent)", markerfacecolor="#1f77b4", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="M1 (deg1)", markerfacecolor="#d62728", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="deg2 tail", markerfacecolor="#7f7f7f", markersize=8),
    ]
    plt.legend(handles=handles, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
