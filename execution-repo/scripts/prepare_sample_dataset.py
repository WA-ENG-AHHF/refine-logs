from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a tiny local dataset so S0 and S1 can run against real files."
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "sample_adr"),
        help="Directory where the sample dataset should be created.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sample data.",
    )
    return parser.parse_args()


def write_split_file(output_dir: Path, split_name: str, num_samples: int, rng: np.random.Generator) -> dict[str, str]:
    inputs = rng.normal(size=(num_samples, 64, 4)).astype(np.float32)
    targets = rng.normal(size=(num_samples, 64, 1)).astype(np.float32)
    path = output_dir / f"{split_name}.npz"
    np.savez(path, inputs=inputs, targets=targets)
    return {"path": path.name}


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    manifest = {
        "dataset_name": "ADR-sample",
        "dataset_version": "2026-04-26",
        "task_type": "operator-learning",
        "format": "npz",
        "grid_size": 64,
        "splits": {
            "train": [write_split_file(output_dir, "train", 12, rng)],
            "val": [write_split_file(output_dir, "val", 4, rng)],
            "test": [write_split_file(output_dir, "test", 4, rng)],
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote sample dataset to {output_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
