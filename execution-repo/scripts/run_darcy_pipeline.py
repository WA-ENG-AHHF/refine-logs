from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = REPO_ROOT / "scripts" / "run_wave1_pipeline.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Darcy-specialized wrapper around the legacy stage pipeline.")
    parser.add_argument("--run-prefix", default="darcy_wave1")
    parser.add_argument(
        "--s0-config",
        type=Path,
        default=REPO_ROOT / "configs" / "S0_darcy.yaml",
    )
    parser.add_argument(
        "--s1-config",
        type=Path,
        default=REPO_ROOT / "configs" / "S1_darcy_baseline.yaml",
    )
    parser.add_argument(
        "--s2-config",
        type=Path,
        default=REPO_ROOT / "configs" / "S2_darcy_main.yaml",
    )
    parser.add_argument("--skip-s2", action="store_true")
    parser.add_argument("--s2-dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = [
        sys.executable,
        str(BASE_SCRIPT),
        "--run-prefix",
        args.run_prefix,
        "--s0-config",
        str(args.s0_config.resolve()),
        "--s1-config",
        str(args.s1_config.resolve()),
        "--s2-config",
        str(args.s2_config.resolve()),
    ]
    if args.skip_s2:
        command.append("--skip-s2")
    if args.s2_dry_run:
        command.append("--s2-dry-run")
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
