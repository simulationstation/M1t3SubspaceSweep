from pathlib import Path

from alpha_soup.cli import demo


def test_cli_demo(tmp_path: Path):
    out_dir = tmp_path / "demo"
    demo(str(out_dir))
    assert (out_dir / "configs_used.json").exists()
    assert (out_dir / "figures" / "soup_best.png").exists()
