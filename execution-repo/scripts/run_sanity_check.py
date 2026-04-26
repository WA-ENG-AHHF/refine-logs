from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import inspect_dataset_manifest


REQUIRED_TOP_LEVEL_FIELDS = {
    "stage": str,
    "dataset": dict,
    "checks": dict,
    "output": dict,
}

REQUIRED_DATASET_FIELDS = {
    "name": str,
    "grid_size": int,
    "num_samples": int,
}

DEFAULT_REPO_DIRS = ("configs", "scripts", "models", "reports")


@dataclass
class CheckResult:
    name: str
    status: str
    severity: str
    message: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
        }


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]

    parser = argparse.ArgumentParser(
        description="Run scaffold sanity checks from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "configs" / "S0_sanity.yaml",
        help="Path to the YAML sanity config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional report path override. Falls back to config.output.report_json.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Execution repo root used for directory validation and relative paths.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Build and print the report without writing the JSON file.",
    )
    return parser.parse_args()


def _coerce_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "Null", "none", "None", "~"}:
        return None
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _basic_yaml_load(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(
                f"Expected multiples of two spaces for indentation on line {line_number}."
            )

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid YAML nesting near line {line_number}.")

        current = stack[-1][1]
        if stripped.startswith("- "):
            if not isinstance(current, list):
                raise ValueError(f"List item found outside a list on line {line_number}.")
            current.append(_coerce_scalar(stripped[2:]))
            continue

        if ":" not in raw_line:
            raise ValueError(f"Unsupported YAML syntax on line {line_number}: {raw_line}")

        key, raw_value = raw_line.strip().split(":", 1)
        value = raw_value.strip()
        if value == "":
            next_container: Any = []
            sibling_lines = text.splitlines()[line_number:]
            next_meaningful = next(
                (candidate.strip() for candidate in sibling_lines if candidate.strip() and not candidate.strip().startswith("#")),
                "",
            )
            if not next_meaningful.startswith("- "):
                next_container = {}
            nested = next_container
            current[key] = nested
            stack.append((indent, nested))
        else:
            current[key] = _coerce_scalar(value)

    return root


def load_config(config_path: Path) -> dict[str, Any]:
    text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        loaded = yaml.safe_load(text)
    else:
        loaded = _basic_yaml_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a mapping.")
    return loaded


def _matches_type(value: Any, expected_type: type[Any]) -> bool:
    if expected_type is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected_type)


def validate_required_fields(config: dict[str, Any]) -> list[CheckResult]:
    results: list[CheckResult] = []

    missing_top_level: list[str] = []
    invalid_top_level: list[str] = []
    for field_name, expected_type in REQUIRED_TOP_LEVEL_FIELDS.items():
        if field_name not in config:
            missing_top_level.append(field_name)
            continue
        if not _matches_type(config[field_name], expected_type):
            invalid_top_level.append(
                f"{field_name} expected {expected_type.__name__}, got {type(config[field_name]).__name__}"
            )

    results.append(
        CheckResult(
            name="required_top_level_fields",
            status="passed" if not missing_top_level and not invalid_top_level else "failed",
            severity="error" if missing_top_level or invalid_top_level else "info",
            message="Validated required top-level config fields.",
            details={
                "missing": missing_top_level,
                "invalid": invalid_top_level,
            },
        )
    )

    dataset = config.get("dataset", {})
    missing_dataset: list[str] = []
    invalid_dataset: list[str] = []
    if isinstance(dataset, dict):
        for field_name, expected_type in REQUIRED_DATASET_FIELDS.items():
            if field_name not in dataset:
                missing_dataset.append(field_name)
                continue
            if not _matches_type(dataset[field_name], expected_type):
                invalid_dataset.append(
                    f"dataset.{field_name} expected {expected_type.__name__}, got {type(dataset[field_name]).__name__}"
                )
    else:
        missing_dataset.extend(REQUIRED_DATASET_FIELDS.keys())

    results.append(
        CheckResult(
            name="required_dataset_fields",
            status="passed" if not missing_dataset and not invalid_dataset else "failed",
            severity="error" if missing_dataset or invalid_dataset else "info",
            message="Validated required dataset fields.",
            details={
                "missing": missing_dataset,
                "invalid": invalid_dataset,
            },
        )
    )

    return results


def validate_repo_layout(repo_root: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    for directory_name in DEFAULT_REPO_DIRS:
        directory_path = repo_root / directory_name
        results.append(
            CheckResult(
                name=f"directory:{directory_name}",
                status="passed" if directory_path.is_dir() else "failed",
                severity="error" if not directory_path.is_dir() else "info",
                message=f"Checked repo directory `{directory_name}`.",
                details={"path": str(directory_path), "exists": directory_path.is_dir()},
            )
        )
    return results


def validate_configured_paths(config: dict[str, Any], repo_root: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    dataset = config.get("dataset", {})

    root_dir = dataset.get("root_dir") if isinstance(dataset, dict) else None
    if root_dir:
        dataset_root = resolve_path(root_dir, repo_root)
        results.append(
            CheckResult(
                name="dataset_root_dir",
                status="passed" if dataset_root.exists() else "failed",
                severity="error" if not dataset_root.exists() else "info",
                message="Validated dataset.root_dir.",
                details={"path": str(dataset_root), "exists": dataset_root.exists()},
            )
        )

    required_paths = dataset.get("required_paths") if isinstance(dataset, dict) else None
    if required_paths is not None:
        if not isinstance(required_paths, list):
            results.append(
                CheckResult(
                    name="dataset_required_paths_type",
                    status="failed",
                    severity="error",
                    message="dataset.required_paths must be a list when provided.",
                    details={"value_type": type(required_paths).__name__},
                )
            )
        else:
            missing: list[str] = []
            for item in required_paths:
                resolved = resolve_path(item, repo_root)
                if not resolved.exists():
                    missing.append(str(resolved))
            results.append(
                CheckResult(
                    name="dataset_required_paths",
                    status="passed" if not missing else "failed",
                    severity="error" if missing else "info",
                    message="Validated dataset.required_paths entries.",
                    details={"missing_paths": missing},
                )
            )

    return results


def validate_dataset_manifest(config: dict[str, Any], repo_root: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    dataset = config.get("dataset", {})
    if not isinstance(dataset, dict):
        return results

    manifest_value = dataset.get("manifest_path")
    if not manifest_value:
        return results

    manifest_path = resolve_path(manifest_value, repo_root)
    if not manifest_path.exists():
        results.append(
            CheckResult(
                name="dataset_manifest",
                status="failed",
                severity="error",
                message="Configured dataset manifest does not exist.",
                details={"path": str(manifest_path)},
            )
        )
        return results

    inspection = inspect_dataset_manifest(manifest_path)
    expected_samples = dataset.get("num_samples")
    expected_grid_size = dataset.get("grid_size")

    results.append(
        CheckResult(
            name="dataset_manifest",
            status="passed",
            severity="info",
            message="Loaded dataset manifest successfully.",
            details=inspection.to_dict(),
        )
    )
    results.append(
        CheckResult(
            name="dataset_sample_count",
            status="passed" if expected_samples == inspection.total_samples else "failed",
            severity="error" if expected_samples != inspection.total_samples else "info",
            message="Compared configured sample count with manifest inspection.",
            details={
                "expected_num_samples": expected_samples,
                "observed_num_samples": inspection.total_samples,
            },
        )
    )
    observed_grid_size = inspection.input_shape[1] if len(inspection.input_shape) > 1 else None
    results.append(
        CheckResult(
            name="dataset_grid_size",
            status="passed" if expected_grid_size == observed_grid_size else "failed",
            severity="error" if expected_grid_size != observed_grid_size else "info",
            message="Compared configured grid size with observed sample shape.",
            details={
                "expected_grid_size": expected_grid_size,
                "observed_grid_size": observed_grid_size,
            },
        )
    )
    results.append(
        CheckResult(
            name="dataset_finite_values",
            status="passed" if not inspection.contains_nan and not inspection.contains_inf else "failed",
            severity="error" if inspection.contains_nan or inspection.contains_inf else "info",
            message="Checked dataset sample arrays for NaN/Inf.",
            details={
                "contains_nan": inspection.contains_nan,
                "contains_inf": inspection.contains_inf,
            },
        )
    )
    return results


def build_requested_check_statuses(config: dict[str, Any]) -> list[dict[str, Any]]:
    checks = config.get("checks", {})
    if not isinstance(checks, dict):
        return []

    statuses: list[dict[str, Any]] = []
    for name, enabled in checks.items():
        statuses.append(
            {
                "name": name,
                "enabled": bool(enabled),
                "status": "pending_implementation" if enabled else "disabled",
                "message": (
                    "Hook is enabled in config and reserved for a future concrete check."
                    if enabled
                    else "Hook is disabled in config."
                ),
            }
        )
    return statuses


def resolve_path(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (repo_root / path).resolve()


def determine_report_path(
    config: dict[str, Any], repo_root: Path, explicit_output: Path | None
) -> Path:
    if explicit_output is not None:
        return resolve_path(explicit_output, repo_root)

    output_section = config.get("output", {})
    report_value = None
    if isinstance(output_section, dict):
        report_value = output_section.get("report_json")
    if not report_value:
        report_value = "logs/S0_sanity_report.json"
    return resolve_path(report_value, repo_root)


def summarize_results(checks: list[CheckResult]) -> dict[str, Any]:
    failed = [check for check in checks if check.status == "failed"]
    passed = [check for check in checks if check.status == "passed"]
    return {
        "overall_status": "passed" if not failed else "failed",
        "passed_checks": len(passed),
        "failed_checks": len(failed),
        "total_checks": len(checks),
    }


def build_report(
    config: dict[str, Any], config_path: Path, repo_root: Path, report_path: Path
) -> dict[str, Any]:
    performed_checks = (
        validate_required_fields(config)
        + validate_repo_layout(repo_root)
        + validate_configured_paths(config, repo_root)
        + validate_dataset_manifest(config, repo_root)
    )
    summary = summarize_results(performed_checks)

    return {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "config_path": str(config_path.resolve()),
            "repo_root": str(repo_root.resolve()),
            "report_path": str(report_path),
        },
        "config_snapshot": {
            "stage": config.get("stage"),
            "dataset": config.get("dataset", {}),
            "checks": config.get("checks", {}),
        },
        "summary": summary,
        "performed_checks": [check.to_dict() for check in performed_checks],
        "requested_check_hooks": build_requested_check_statuses(config),
    }


def write_report(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config, args.repo_root)
    repo_root = args.repo_root.resolve()

    config = load_config(config_path)
    report_path = determine_report_path(config, repo_root, args.output)
    report = build_report(config, config_path, repo_root, report_path)

    if not args.no_write:
        write_report(report_path, report)

    print(json.dumps(report["summary"], indent=2))
    if args.no_write:
        print("Skipped writing JSON report because --no-write was provided.")
    else:
        print(f"Wrote sanity report to {report_path}")

    return 0 if report["summary"]["overall_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
