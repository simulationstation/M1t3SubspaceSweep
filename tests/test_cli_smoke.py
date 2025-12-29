import subprocess
import sys
from pathlib import Path


def test_cli_demo_smoke(tmp_path: Path):
    out_dir = tmp_path / "demo"
    subprocess.run(
        [sys.executable, "-m", "alpha_soup_wave.cli", "demo", "--out", str(out_dir)],
        check=True,
    )
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "figures" / "soup_best.png").exists()
