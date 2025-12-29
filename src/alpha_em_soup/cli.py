"""CLI for alpha-em soup."""

from __future__ import annotations

import argparse
from pathlib import Path

from alpha_em_soup.config import RunConfig
from alpha_em_soup.reporting.report_md import generate_report
from alpha_em_soup.search.refine import run_refine
from alpha_em_soup.search.sweep import run_sweep


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="alpha-em")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Run a demo sweep with default config")
    demo.add_argument("--out", type=Path, required=True)

    sweep = sub.add_parser("sweep", help="Run coarse sweep")
    sweep.add_argument("--config", type=Path, required=True)
    sweep.add_argument("--out", type=Path, required=True)

    refine = sub.add_parser("refine", help="Refine top candidates")
    refine.add_argument("--in", dest="input_dir", type=Path, required=True)
    refine.add_argument("--out", type=Path, required=True)

    report = sub.add_parser("report", help="Generate report")
    report.add_argument("--in", dest="input_dir", type=Path, required=True)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "demo":
        config = RunConfig()
        config.sweep.config_count = 1
        config.sweep.seeds = [0]
        config_path = args.out / "demo_config.json"
        args.out.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config.model_dump_json(indent=2))
        run_sweep(config_path, args.out)
        generate_report(args.out, args.out / "report.md")
    elif args.command == "sweep":
        run_sweep(args.config, args.out)
    elif args.command == "refine":
        run_refine(args.input_dir, args.out)
    elif args.command == "report":
        generate_report(args.input_dir, Path(args.input_dir) / "report.md")


if __name__ == "__main__":
    main()
