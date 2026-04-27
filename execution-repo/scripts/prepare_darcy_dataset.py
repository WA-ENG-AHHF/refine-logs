from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCRIPT = REPO_ROOT / "scripts" / "prepare_coupled_pde_dataset.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "DATASET_darcy2d.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Darcy dataset entrypoint. Currently preserves the scaffold contract while the real generator is implemented."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the Darcy dataset config.",
    )
    parser.add_argument(
        "--legacy-fallback",
        action="store_true",
        help="Call the legacy generator as a temporary scaffold fallback.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Darcy dataset config selected: {args.config.resolve()}")
    if not args.legacy_fallback:
        print("Placeholder entrypoint: implement the true Darcy dataset generator here.")
        print("Use --legacy-fallback only if you explicitly want to exercise the old scaffold path.")
        return 0

    command = [
        sys.executable,
        str(LEGACY_SCRIPT),
        "--config",
        str(args.config.resolve()),
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

