from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.coupled_adr_generator import (
    config_from_dict,
    generate_dataset,
    sample_config_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a first research-grade coupled ADR coarse-to-fine dataset."
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "DATASET_coupled_adr.yaml"),
        help="Path to the generator config.",
    )
    return parser.parse_args()


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_simple_yaml(raw_text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in raw_text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line: {raw_line}")

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        current = stack[-1][1]
        if value == "":
            nested: dict[str, Any] = {}
            current[key] = nested
            stack.append((indent, nested))
        else:
            current[key] = parse_scalar(value)

    return root


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        default_payload = sample_config_dict()
        config_path.write_text(json.dumps(default_payload, indent=2), encoding="utf-8")
        return default_payload

    raw_text = config_path.read_text(encoding="utf-8")
    if yaml is not None and config_path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(raw_text) or {}
    if config_path.suffix.lower() == ".json":
        return json.loads(raw_text)
    return parse_simple_yaml(raw_text)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    payload = load_config(config_path)
    config = config_from_dict(payload)
    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        config.output_dir = str((REPO_ROOT / output_dir).resolve())
    result = generate_dataset(config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
