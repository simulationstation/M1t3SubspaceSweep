"""Robustness scoring utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_soup.constants import ALPHA_TARGET


def summarize_configs(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("config_hash")
    summary = grouped.agg(
        loss_mean=("loss", "mean"),
        loss_std=("loss", "std"),
        alpha_gap_mean=("alpha_gap_ratio", "mean"),
        alpha_stiffness_mean=("alpha_stiffness_ratio", "mean"),
        alpha_overlap_mean=("alpha_overlap_gap", "mean"),
        alpha_gap_std=("alpha_gap_ratio", "std"),
        alpha_stiffness_std=("alpha_stiffness_ratio", "std"),
        alpha_overlap_std=("alpha_overlap_gap", "std"),
        m2_count_mean=("m2_count", "mean"),
        stability_mean=("stability_m2", "mean"),
    )
    summary = summary.fillna(0.0)
    return summary.reset_index()


def best_candidates(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    summary = summarize_configs(df)
    summary = summary.sort_values("loss_mean").head(top_k)
    return summary


def stability_filter(df: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    summary = summarize_configs(df)
    alpha_vals = summary[["alpha_gap_mean", "alpha_stiffness_mean", "alpha_overlap_mean"]].to_numpy()
    targets = np.abs(np.log(np.clip(alpha_vals, 1e-12, None)) - np.log(ALPHA_TARGET))
    summary["best_alpha_deviation"] = np.min(targets, axis=1)
    return summary[summary["best_alpha_deviation"] <= tolerance]
