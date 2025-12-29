"""Report generation for alpha search."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alpha_soup_wave.constants import ALPHA_TARGET


def generate_report(results: pd.DataFrame, best_candidates: list[dict], output_path: str) -> None:
    output = Path(output_path)
    lines = [
        "# Alpha Soup Wave Report",
        "",
        f"Target alpha: {ALPHA_TARGET:.12f}",
        "",
        "## Summary",
        f"Total configs: {len(results)}",
        f"Best candidates: {len(best_candidates)}",
        "",
    ]
    if best_candidates:
        lines.append("## Top Candidate")
        lines.append("")
        top = best_candidates[0]
        lines.append(f"Alpha hat: {top['alpha_hat']:.6f}")
        lines.append(f"Log distance: {top['alpha_log_distance']:.4f}")
        lines.append(f"Alpha std: {top['alpha_std']:.6f}")
        lines.append(f"M2 size mean: {top['m2_size_mean']:.2f}")
        lines.append("")
    output.write_text("\n".join(lines))


def write_best_candidates(best_candidates: list[dict], output_path: str) -> None:
    Path(output_path).write_text(json.dumps(best_candidates, indent=2))
