"""Report generation for alpha search."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alpha_soup_wave.constants import ALPHA_TARGET


def generate_report(
    summary: pd.DataFrame,
    best_candidates: list[dict],
    output_path: str,
    r2_values: dict[str, float] | None = None,
) -> None:
    output = Path(output_path)
    lines = [
        "# Alpha Soup Wave Report",
        "",
        f"Target alpha: {ALPHA_TARGET:.12f}",
        "",
        "## Summary",
        f"Total configs: {len(summary)}",
        f"Best candidates: {len(best_candidates)}",
        "",
    ]

    if r2_values:
        lines.append("## No-Cheating Scan")
        for name, value in sorted(r2_values.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"- R^2(alpha_hat, {name}) = {value:.3f}")
        lines.append("")

    lines.append("## Robustness Filters")
    if not summary.empty:
        lines.append(f"Plateau pass: {(summary['plateau_pass']).sum()} / {len(summary)}")
        lines.append(f"Fine-tune flagged: {(summary['fine_tune_flag']).sum()} / {len(summary)}")
        lines.append("")

    if best_candidates:
        lines.append("## Top Candidate")
        lines.append("")
        top = best_candidates[0]
        mechanisms = [
            "attenuation",
            "overlap",
        ]
        if bool(top.get("enable_selection", True)):
            mechanisms.append("selection rule (second order)")
        if bool(top.get("enable_gating", True)):
            mechanisms.append("duty-cycle gating")
        lines.append(f"Mechanisms enabled: {', '.join(mechanisms)}")
        lines.append(f"Alpha hat: {top['alpha_hat']:.6f}")
        lines.append(f"Score: {top['score']:.4f}")
        lines.append(f"Log distance: {top['alpha_log_distance']:.4f}")
        lines.append(f"Alpha std: {top['alpha_std']:.6f}")
        lines.append(f"M2 size mean: {top['m2_size_mean']:.2f}")
        lines.append(f"Plateau pass: {top['plateau_pass']}")
        lines.append(f"Fine-tune flagged: {top['fine_tune_flag']}")
        lines.append("")
    output.write_text("\n".join(lines))


def write_best_candidates(best_candidates: list[dict], output_path: str) -> None:
    Path(output_path).write_text(json.dumps(best_candidates, indent=2))

