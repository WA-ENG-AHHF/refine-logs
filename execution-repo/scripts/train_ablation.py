from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated S1/S2 experiments and aggregate metrics."
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["S1", "S2"],
        choices=["S1", "S2"],
        help="Stages to run for the repeated experiment sweep.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated runs per stage.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Base seed; each repeat uses seed_base + repeat_index.",
    )
    parser.add_argument(
        "--run-prefix",
        default="formal_repeat",
        help="Prefix used to name generated runs and reports.",
    )
    parser.add_argument(
        "--s1-config",
        default=str(REPO_ROOT / "configs" / "S1_baseline_coupled_adr.yaml"),
        help="Path to the S1 config file.",
    )
    parser.add_argument(
        "--s2-config",
        default=str(REPO_ROOT / "configs" / "S2_main_coupled_adr.yaml"),
        help="Path to the S2 config file.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch stage scripts.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "ablation"),
        help="Directory for sweep reports.",
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
    if value and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
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


def dump_simple_yaml(data: dict[str, Any], indent: int = 0) -> str:
    lines: list[str] = []
    for key, value in data.items():
        prefix = " " * indent
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(dump_simple_yaml(value, indent + 2))
        else:
            scalar = "null" if value is None else str(value).lower() if isinstance(value, bool) else str(value)
            lines.append(f"{prefix}{key}: {scalar}")
    return "\n".join(line for line in lines if line)


def load_config(config_path: Path) -> dict[str, Any]:
    raw_text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        payload = yaml.safe_load(raw_text) or {}
    else:
        payload = parse_simple_yaml(raw_text)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return payload


def save_config(config_path: Path, payload: dict[str, Any]) -> None:
    with config_path.open("w", encoding="utf-8") as handle:
        if yaml is not None:
            yaml.safe_dump(payload, handle, sort_keys=False)
        else:
            handle.write(dump_simple_yaml(payload))
            handle.write("\n")


def stage_script_path(stage: str) -> Path:
    return {
        "S1": SCRIPTS_DIR / "train_baseline.py",
        "S2": SCRIPTS_DIR / "train_pcr_no.py",
    }[stage]


def stage_config_path(stage: str, args: argparse.Namespace) -> Path:
    return {
        "S1": Path(args.s1_config).resolve(),
        "S2": Path(args.s2_config).resolve(),
    }[stage]


def override_seed(payload: dict[str, Any], seed: int) -> dict[str, Any]:
    updated = json.loads(json.dumps(payload))
    train_cfg = updated.setdefault("train", {})
    train_cfg["seed"] = int(seed)
    return updated


def read_summary(summary_path: Path) -> dict[str, Any]:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def run_stage(
    stage: str,
    stage_config: Path,
    run_name: str,
    python_executable: str,
) -> tuple[dict[str, Any], subprocess.CompletedProcess[str]]:
    command = [
        python_executable,
        str(stage_script_path(stage)),
        "--config",
        str(stage_config),
        "--run-name",
        run_name,
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{stage} run failed for {run_name}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    summary_root = REPO_ROOT / "outputs"
    if stage == "S1":
        summary_path = summary_root / "S1_baseline_coupled_adr" / run_name / "reports" / "run_summary.json"
    else:
        summary_path = summary_root / "S2" / run_name / "reports" / "run_summary.json"
    return read_summary(summary_path), result


def extract_record(stage: str, repeat_index: int, seed: int, summary: dict[str, Any]) -> dict[str, Any]:
    training = summary["training_results"]
    train_metrics = training["train_metrics"]
    val_metrics = training["val_metrics"]
    return {
        "stage": stage,
        "repeat_index": repeat_index,
        "seed": seed,
        "run_name": summary["run_name"],
        "backend": training["backend"],
        "device": training.get("device", summary.get("device", "")),
        "epochs": training.get("epochs", ""),
        "best_epoch": training.get("best_epoch", ""),
        "train_rel_l2": train_metrics.get("rel_l2", ""),
        "val_rel_l2": val_metrics.get("rel_l2", ""),
        "train_total": train_metrics.get("total", ""),
        "val_total": val_metrics.get("total", ""),
        "train_pde_residual": train_metrics.get("pde_residual", ""),
        "val_pde_residual": val_metrics.get("pde_residual", ""),
        "train_bc_violation": train_metrics.get("bc_violation", ""),
        "val_bc_violation": val_metrics.get("bc_violation", ""),
    }


def aggregate_stage_records(stage: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    def metric_values(key: str) -> list[float]:
        return [float(record[key]) for record in records]

    aggregate = {"stage": stage, "num_runs": len(records)}
    for key in (
        "train_rel_l2",
        "val_rel_l2",
        "train_total",
        "val_total",
        "train_pde_residual",
        "val_pde_residual",
        "train_bc_violation",
        "val_bc_violation",
    ):
        values = metric_values(key)
        aggregate[f"{key}_mean"] = mean(values)
        aggregate[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return aggregate


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve() / args.run_prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed_records: list[dict[str, Any]] = []
    stage_aggregates: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="pcrno-repeat-configs-") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        for stage in args.stages:
            base_config = load_config(stage_config_path(stage, args))
            stage_records: list[dict[str, Any]] = []

            for repeat_index in range(1, args.repeats + 1):
                seed = int(args.seed_base + repeat_index - 1)
                run_name = f"{args.run_prefix}_{stage.lower()}_r{repeat_index:02d}"
                stage_config = override_seed(base_config, seed)
                temp_config_path = temp_dir / f"{run_name}.yaml"
                save_config(temp_config_path, stage_config)
                summary, _ = run_stage(
                    stage=stage,
                    stage_config=temp_config_path,
                    run_name=run_name,
                    python_executable=args.python,
                )
                record = extract_record(stage, repeat_index, seed, summary)
                detailed_records.append(record)
                stage_records.append(record)

            stage_aggregates.append(aggregate_stage_records(stage, stage_records))

    detailed_csv = output_dir / "repeat_runs.csv"
    aggregate_csv = output_dir / "repeat_summary.csv"
    summary_json = output_dir / "repeat_summary.json"

    write_csv(detailed_csv, detailed_records)
    write_csv(aggregate_csv, stage_aggregates)
    summary_json.write_text(
        json.dumps(
            {
                "run_prefix": args.run_prefix,
                "python": args.python,
                "repeats": args.repeats,
                "seed_base": args.seed_base,
                "stages": args.stages,
                "artifacts": {
                    "repeat_runs_csv": str(detailed_csv),
                    "repeat_summary_csv": str(aggregate_csv),
                    "repeat_summary_json": str(summary_json),
                },
                "records": detailed_records,
                "aggregates": stage_aggregates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Detailed runs: {detailed_csv}")
    print(f"Aggregate summary: {aggregate_csv}")
    print(f"JSON summary: {summary_json}")


if __name__ == "__main__":
    main()
