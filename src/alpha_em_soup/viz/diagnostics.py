"""Diagnostic plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_diagnostics(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not summary.empty:
        _hist(summary, "q_ratio", output_dir / "q_distribution.png", "Q distribution")
        _hist(summary, "g21_eff", output_dir / "g21_hist.png", "g21_eff histogram")
        _scatter(summary, "q_std", output_dir / "seed_stability.png", "Seed stability (Q std)")
        if "neighborhood_penalty" in summary.columns:
            _scatter(summary, "neighborhood_penalty", output_dir / "neighborhood.png", "Neighborhood robustness")


def _hist(df: pd.DataFrame, column: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[column].dropna(), bins=20, color="steelblue", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _scatter(df: pd.DataFrame, column: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(range(len(df)), df[column].fillna(0.0), color="purple", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("config index")
    ax.set_ylabel(column)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
