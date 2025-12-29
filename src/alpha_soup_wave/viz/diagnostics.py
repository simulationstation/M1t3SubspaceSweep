"""Diagnostic plots for alpha search."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_alpha_hist(results: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(results["alpha_hat"], bins=30, color="#1f77b4", alpha=0.8)
    ax.set_xlabel("alpha_hat")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_robustness(results: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(results["alpha_log_distance"], results["alpha_std"], alpha=0.7)
    ax.set_xlabel("log distance to target")
    ax.set_ylabel("alpha std")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
