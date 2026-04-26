from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


CORE_METRICS = {
    "rel_l2": [
        "rel_l2",
        "rel-l2",
        "relative_l2",
        "relative-l2",
        "relative_l2_error",
        "relative_error_l2",
        "l2_relative_error",
    ],
    "pde_residual": [
        "pde_residual",
        "pde-residual",
        "pde residual",
        "residual_pde",
        "equation_residual",
    ],
    "bc_violation": [
        "bc_violation",
        "bc-violation",
        "bc violation",
        "boundary_violation",
        "boundary_condition_violation",
    ],
    "conservation_error": [
        "conservation_error",
        "conservation-error",
        "conservation error",
        "mass_conservation_error",
        "energy_conservation_error",
    ],
}

CANONICAL_COLUMNS = [
    "source_file",
    "source_type",
    "record_index",
    "run_id",
    "experiment",
    "case_id",
    "step",
    "epoch",
    "split",
    "tag",
    "rel_l2",
    "pde_residual",
    "bc_violation",
    "conservation_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate metrics from CSV/JSON files into a master summary CSV."
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="Input file or directory. Repeatable.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output master summary CSV.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input directories for .csv/.json files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if no canonical metric columns can be populated for a record.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Run the plotting stub after writing the master summary CSV.",
    )
    return parser.parse_args()


def normalize_key(key: str) -> str:
    normalized = key.strip().lower()
    for token in (" ", "-", ".", "/"):
        normalized = normalized.replace(token, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def build_alias_map() -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for canonical, aliases in CORE_METRICS.items():
        alias_map[normalize_key(canonical)] = canonical
        for alias in aliases:
            alias_map[normalize_key(alias)] = canonical
    return alias_map


ALIAS_MAP = build_alias_map()


def discover_input_files(inputs: list[str], recursive: bool) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    patterns = ("*.csv", "*.json")

    for raw_input in inputs:
        path = Path(raw_input).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        candidates: Iterable[Path]
        if path.is_file():
            candidates = [path]
        else:
            iterator_name = "rglob" if recursive else "glob"
            candidates = []
            for pattern in patterns:
                iterator = getattr(path, iterator_name)(pattern)
                candidates = [*candidates, *iterator]

        for candidate in candidates:
            suffix = candidate.suffix.lower()
            if suffix not in {".csv", ".json"}:
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(candidate)

    return sorted(discovered)


def coerce_scalar(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return ""
        try:
            if any(mark in stripped for mark in (".", "e", "E")):
                return float(stripped)
            return int(stripped)
        except ValueError:
            return stripped
    return value


def flatten_record(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in record.items():
        joined_key = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_record(value, joined_key))
        elif isinstance(value, list):
            if value and all(not isinstance(item, (dict, list)) for item in value):
                flattened[joined_key] = "|".join(str(item) for item in value)
            else:
                flattened[joined_key] = json.dumps(value, ensure_ascii=True)
        else:
            flattened[joined_key] = value
    return flattened


def load_csv_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {key: coerce_scalar(value) for key, value in row.items() if key is not None}


def load_json_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records: list[dict[str, Any]]
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("metrics"), list):
            records = payload["metrics"]
        elif isinstance(payload.get("records"), list):
            records = payload["records"]
        else:
            records = [payload]
    else:
        raise ValueError(f"Unsupported JSON payload in {path}")

    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"JSON record must be an object in {path}")
        yield flatten_record(record)


def load_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        yield from load_csv_records(path)
        return
    if path.suffix.lower() == ".json":
        yield from load_json_records(path)
        return
    raise ValueError(f"Unsupported file type: {path}")


def resolve_canonical_metric(normalized_key: str) -> str | None:
    direct_match = ALIAS_MAP.get(normalized_key)
    if direct_match is not None:
        return direct_match

    parts = normalized_key.split("_")
    for start_index in range(1, len(parts)):
        suffix_match = ALIAS_MAP.get("_".join(parts[start_index:]))
        if suffix_match is not None:
            return suffix_match
    return None


def canonicalize_record(
    raw_record: dict[str, Any],
    source_file: Path,
    record_index: int,
    strict: bool,
) -> dict[str, Any]:
    normalized_items = {
        normalize_key(str(key)): coerce_scalar(value)
        for key, value in raw_record.items()
        if key is not None
    }

    record: dict[str, Any] = {
        "source_file": str(source_file),
        "source_type": source_file.suffix.lower().lstrip("."),
        "record_index": record_index,
        "run_id": normalized_items.get("run_id", ""),
        "experiment": normalized_items.get("experiment", normalized_items.get("experiment_name", "")),
        "case_id": normalized_items.get("case_id", normalized_items.get("sample_id", "")),
        "step": normalized_items.get("step", normalized_items.get("global_step", "")),
        "epoch": normalized_items.get("epoch", ""),
        "split": normalized_items.get("split", normalized_items.get("dataset_split", "")),
        "tag": normalized_items.get("tag", normalized_items.get("phase", "")),
    }

    populated_metrics = 0
    for key, value in normalized_items.items():
        canonical_key = resolve_canonical_metric(key)
        if canonical_key is None:
            continue
        record[canonical_key] = value
        if value not in ("", None):
            populated_metrics += 1

    for canonical_metric in CORE_METRICS:
        record.setdefault(canonical_metric, "")

    if strict and populated_metrics == 0:
        raise ValueError(
            f"Record {record_index} in {source_file} does not contain any canonical metrics."
        )

    return record


def aggregate_records(paths: list[Path], strict: bool) -> list[dict[str, Any]]:
    aggregated: list[dict[str, Any]] = []
    for path in paths:
        for record_index, raw_record in enumerate(load_records(path), start=1):
            aggregated.append(canonicalize_record(raw_record, path, record_index, strict))
    return aggregated


def write_master_summary_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def plot_summary(summary_csv: Path) -> None:
    print(
        "Plotting stub: implement chart generation here using the normalized summary at "
        f"{summary_csv}"
    )


def main() -> None:
    args = parse_args()
    input_files = discover_input_files(args.inputs, recursive=args.recursive)
    if not input_files:
        raise SystemExit("No CSV/JSON metric files found.")

    records = aggregate_records(input_files, strict=args.strict)
    write_master_summary_csv(records, Path(args.output))

    print(f"Discovered files: {len(input_files)}")
    print(f"Aggregated records: {len(records)}")
    print(f"Master summary written to: {Path(args.output)}")

    if args.plot:
        plot_summary(Path(args.output))


if __name__ == "__main__":
    main()
