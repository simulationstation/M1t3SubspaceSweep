import sys
from pathlib import Path

from alpha_em_soup import cli


def test_cli_smoke(tmp_path: Path, monkeypatch):
    out_dir = tmp_path / "demo"
    monkeypatch.setattr(sys, "argv", ["alpha-em", "demo", "--out", str(out_dir)])
    cli.main()
    assert (out_dir / "summary.csv").exists()
