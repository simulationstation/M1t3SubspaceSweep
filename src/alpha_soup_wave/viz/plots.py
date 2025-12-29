"""Visualization helpers for v3 diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_alpha_histograms(results: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mechanisms = [
        ("alpha_hat_att", "attenuation"),
        ("alpha_hat_overlap", "overlap"),
        ("alpha_hat_sel", "selection"),
        ("alpha_hat_gate", "gate"),
    ]
    for column, label in mechanisms:
        if column not in results.columns:
            continue
        data = results[column].dropna()
        if data.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(data, bins=30, color="#1f77b4", alpha=0.8)
        ax.set_xlabel(column)
        ax.set_ylabel("count")
        ax.set_title(f"Alpha_hat distribution: {label}")
        fig.tight_layout()
        fig.savefig(output_dir / f"alpha_hist_{label}.png")
        plt.close(fig)


def plot_seed_stability(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(summary["alpha_hat"], summary["alpha_std"], alpha=0.7)
    ax.set_xlabel("alpha_hat mean")
    ax.set_ylabel("alpha_hat std")
    ax.set_title("Seed stability")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_sensitivity(summary: pd.DataFrame, param_columns: list[str], output_path: Path) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(1, min(3, len(param_columns)), figsize=(12, 4))
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    for ax, param in zip(axes, param_columns[: len(axes)], strict=False):
        ax.scatter(summary[param], summary["alpha_hat"], alpha=0.6)
        ax.set_xlabel(param)
        ax.set_ylabel("alpha_hat")
    fig.suptitle("Sensitivity (alpha_hat vs parameters)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


