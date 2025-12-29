from pathlib import Path

from alpha_em_soup.cli import main


if __name__ == "__main__":
    import sys

    output = Path("outputs/demo")
    sys.argv = ["alpha-em", "demo", "--out", str(output)]
    main()
