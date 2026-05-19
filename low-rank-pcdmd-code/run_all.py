from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_script(script_name: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / script_name)],
        check=True,
        cwd=ROOT,
    )


def main() -> None:
    run_script("run_full_space.py")
    run_script("run_low_rank.py")


if __name__ == "__main__":
    main()
