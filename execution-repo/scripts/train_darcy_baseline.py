from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = REPO_ROOT / "scripts" / "train_baseline.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "S1_darcy_baseline.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Darcy baseline training wrapper.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the Darcy baseline config.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name override.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = [
        sys.executable,
        str(BASE_SCRIPT),
        "--config",
        str(args.config.resolve()),
    ]
    if args.run_name:
        command.extend(["--run-name", args.run_name])
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

