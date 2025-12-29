"""CLI for alpha soup experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd

from alpha_soup.config import SweepConfig, SoupConfig
from alpha_soup.search.sweep import run_sweep
from alpha_soup.search.refine import refine_from_sweep
from alpha_soup.reporting import write_report, export_best
from alpha_soup.viz import plot_alpha_scatter, plot_robustness, plot_soup
from alpha_soup.soup import generate_degree_graph, apply_decay, initialize_state, evolve, detect_m2


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _ensure_out(out: str | None) -> Path:
    base = Path(out) if out else Path("outputs") / f"run_{_timestamp()}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def demo(out: str | None) -> None:
    out_dir = _ensure_out(out)
    cfg = SoupConfig()
    graph_bundle = generate_degree_graph(cfg.n_nodes, cfg.profile, cfg.seed)
    graph_bundle = apply_decay(graph_bundle, cfg.decay, cfg.seed)
    state = initialize_state(cfg.n_nodes, cfg.seed)
    evolved = evolve(
        state,
        list(graph_bundle.graph.edges),
        graph_bundle.degree_labels,
        cfg.couplings,
        cfg.dynamics,
        cfg.seed,
    )
    m2 = detect_m2(graph_bundle.graph, evolved.theta, cfg.emergent)
    plot_soup(graph_bundle.graph, graph_bundle.degree_labels, m2.nodes, out_dir / "figures" / "soup_best.png")
    (out_dir / "configs_used.json").write_text(json.dumps({"demo": cfg.model_dump()}, indent=2))


def sweep(config_path: str, out: str | None) -> None:
    out_dir = _ensure_out(out)
    cfg = SweepConfig.model_validate_json(Path(config_path).read_text())
    (out_dir / "sweep_config.json").write_text(cfg.model_dump_json(indent=2))
    df = run_sweep(cfg, out_dir)
    plot_alpha_scatter(df, out_dir / "figures" / "alpha_scatter.png")
    plot_robustness(df, out_dir / "figures" / "robustness.png")
    export_best(df, out_dir / "best_candidates.json", top_k=cfg.search.top_k)
    write_report(df, out_dir / "report.md")


def refine(in_dir: str, out: str | None) -> None:
    out_dir = _ensure_out(out)
    cfg_path = Path(in_dir) / "sweep_config.json"
    if cfg_path.exists():
        sweep_cfg = SweepConfig.model_validate_json(cfg_path.read_text())
        search_cfg = sweep_cfg.search
    else:
        search_cfg = SweepConfig().search
    df = refine_from_sweep(Path(in_dir), out_dir, search_cfg)
    plot_alpha_scatter(df, out_dir / "figures" / "alpha_scatter.png")
    plot_robustness(df, out_dir / "figures" / "robustness.png")
    export_best(df, out_dir / "best_candidates.json", top_k=search_cfg.top_k)
    write_report(df, out_dir / "report.md")


def report(in_dir: str, out: str | None) -> None:
    in_dir_path = Path(in_dir)
    out_path = Path(out) if out else in_dir_path / "report.md"
    results = pd.read_csv(in_dir_path / "results.csv")
    write_report(results, out_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Alpha soup simulation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    demo_p = sub.add_parser("demo", help="Run a quick demo")
    demo_p.add_argument("--out", type=str, default=None)

    sweep_p = sub.add_parser("sweep", help="Run a coarse sweep")
    sweep_p.add_argument("--config", type=str, required=True)
    sweep_p.add_argument("--out", type=str, default=None)

    refine_p = sub.add_parser("refine", help="Refine around top candidates")
    refine_p.add_argument("--in", dest="in_dir", type=str, required=True)
    refine_p.add_argument("--out", type=str, default=None)

    report_p = sub.add_parser("report", help="Generate a report")
    report_p.add_argument("--in", dest="in_dir", type=str, required=True)
    report_p.add_argument("--out", type=str, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "demo":
        demo(args.out)
    elif args.command == "sweep":
        sweep(args.config, args.out)
    elif args.command == "refine":
        refine(args.in_dir, args.out)
    elif args.command == "report":
        report(args.in_dir, args.out)


if __name__ == "__main__":
    main()
