"""Command line interface for alpha soup wave."""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

from alpha_soup_wave.config import SweepConfig
from alpha_soup_wave.reporting.report_md import generate_report
from alpha_soup_wave.search.refine import refine_cli
from alpha_soup_wave.search.sweep import build_model_grid, run_sweep


def _timestamped_dir(prefix: str) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return str(Path(prefix) / f"run_{stamp}")


def demo(out_dir: str) -> None:
    sweep = SweepConfig(max_configs=6, seeds=range(2))
    model_grid = build_model_grid(sweep)
    run_sweep(model_grid, sweep, out_dir, label="demo")


def sweep(out_dir: str) -> None:
    sweep_cfg = SweepConfig()
    model_grid = build_model_grid(sweep_cfg)
    run_sweep(model_grid, sweep_cfg, out_dir, label="sweep")


def report(input_dir: str) -> None:
    from alpha_soup_wave.reporting.report_md import write_best_candidates
    import pandas as pd
    from pathlib import Path

    results = pd.read_csv(Path(input_dir) / "results.csv")
    best_candidates = []
    if (Path(input_dir) / "best_candidates.json").exists():
        import json

        best_candidates = json.loads((Path(input_dir) / "best_candidates.json").read_text())
    generate_report(results, best_candidates, str(Path(input_dir) / "report.md"))
    write_best_candidates(best_candidates, str(Path(input_dir) / "best_candidates.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha Soup Wave simulations")
    sub = parser.add_subparsers(dest="command", required=True)

    demo_cmd = sub.add_parser("demo", help="Run a fast demo sweep")
    demo_cmd.add_argument("--out", default=_timestamped_dir("outputs/demo"))

    sweep_cmd = sub.add_parser("sweep", help="Run a coarse sweep")
    sweep_cmd.add_argument("--out", default=_timestamped_dir("outputs/sweep_small"))

    refine_cmd = sub.add_parser("refine", help="Refine around best configs")
    refine_cmd.add_argument("--in", dest="input_dir", required=True)
    refine_cmd.add_argument("--out", default=_timestamped_dir("outputs/refined"))

    report_cmd = sub.add_parser("report", help="Generate report")
    report_cmd.add_argument("--in", dest="input_dir", required=True)

    args = parser.parse_args()

    if args.command == "demo":
        demo(args.out)
    elif args.command == "sweep":
        sweep(args.out)
    elif args.command == "refine":
        refine_cli(args.input_dir, args.out)
    elif args.command == "report":
        report(args.input_dir)


if __name__ == "__main__":
    main()
