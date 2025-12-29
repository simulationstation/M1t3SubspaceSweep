"""Summary plotting utilities."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from alpha_soup.constants import ALPHA_TARGET


def plot_alpha_scatter(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    for col, color in [
        ("alpha_gap_ratio", "#1f77b4"),
        ("alpha_stiffness_ratio", "#ff7f0e"),
        ("alpha_overlap_gap", "#2ca02c"),
    ]:
        plt.scatter(df.index, df[col], alpha=0.6, label=col, color=color)
    plt.axhline(ALPHA_TARGET, color="red", linestyle="--", label="alpha target")
    plt.ylabel("alpha estimates")
    plt.xlabel("run index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_robustness(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = df.groupby("config_hash").agg(
        alpha_gap_mean=("alpha_gap_ratio", "mean"),
        alpha_gap_std=("alpha_gap_ratio", "std"),
    )
    summary = summary.sort_values("alpha_gap_mean")
    plt.figure(figsize=(7, 5))
    plt.errorbar(summary.index.astype(str), summary["alpha_gap_mean"], yerr=summary["alpha_gap_std"], fmt="o")
    plt.axhline(ALPHA_TARGET, color="red", linestyle="--", label="alpha target")
    plt.xticks(rotation=90, fontsize=6)
    plt.ylabel("alpha_gap_ratio")
    plt.xlabel("config hash")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
