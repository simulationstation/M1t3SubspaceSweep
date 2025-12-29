"""Soup plotting for visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from alpha_em_soup.layers.m1_wavepaths import build_m1_wavepaths
from alpha_em_soup.layers.m2_interface import build_m2
from alpha_em_soup.layers.m3_topology import generate_m3_graph


def plot_soup(model, seed_result, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    m3 = generate_m3_graph(
        model.m3.n3,
        seed_result.seed,
        model.m3.generator,
        model.m3.min_short_cycle_ratio,
        model.m3.triangle_boost_steps,
        model.m3.geometric_radius,
        model.m3.planted_cycle_count,
    )
    m2 = build_m2(
        m3.graph,
        seed_result.seed,
        model.m2.mode,
        model.m2.coherence_threshold,
        model.m2.loopiness_threshold,
        model.m2.min_fraction,
    )
    m1 = build_m1_wavepaths(
        sorted(m2.nodes),
        seed_result.seed,
        model.m1.path_count,
        model.m1.length_mean,
        model.m1.length_sigma,
        model.m1.loop_fraction,
        model.m1.loopback_fraction,
        model.m1.max_path_edges,
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(m3.graph, seed=seed_result.seed)
    rng = np.random.default_rng(seed_result.seed)
    for path in m1.paths:
        anchor_pos = pos.get(path.anchor_a, (0.0, 0.0))
        for node in path.nodes:
            if node in pos:
                continue
            jitter = rng.normal(scale=0.1, size=2)
            pos[node] = (anchor_pos[0] + jitter[0], anchor_pos[1] + jitter[1])
    nx.draw_networkx_nodes(
        m3.graph,
        pos,
        nodelist=[n for n in m3.graph.nodes if n not in m2.nodes],
        node_color="green",
        node_size=80,
        ax=ax,
        label="M3",
    )
    nx.draw_networkx_nodes(
        m3.graph,
        pos,
        nodelist=list(m2.nodes),
        node_color="blue",
        node_size=100,
        ax=ax,
        label="M2",
    )
    nx.draw_networkx_edges(m3.graph, pos, alpha=0.3, ax=ax)
    for path in m1.paths:
        coords = [pos.get(node, (0.0, 0.0)) for node in path.nodes]
        xs, ys = zip(*coords)
        ax.plot(xs, ys, color="red", linewidth=1.5, alpha=0.7)
    ax.set_title("Overlapped EM Soup")
    ax.axis("off")
    ax.legend(loc="upper right")
    text = (
        f"M1 paths: {len(m1.paths)}\n"
        f"M2 nodes: {len(m2.nodes)}\n"
        f"g21_eff={seed_result.g21_eff:.3f} g23_eff={seed_result.g23_eff:.3f} Q={seed_result.q_ratio:.3f}\n"
        f"alpha_cov={seed_result.alpha_hat_cov:.3f} alpha_ov={seed_result.alpha_hat_ov:.3f} alpha_Z={seed_result.alpha_hat_z:.3f}"
    )
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=9, va="bottom")
    output_path = output_dir / "soup_best.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
