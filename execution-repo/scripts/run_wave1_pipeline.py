from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end wave-1 scaffold pipeline across S0, S1, S2, and metrics aggregation."
    )
    parser.add_argument(
        "--run-prefix",
        default="wave1",
        help="Prefix used to name stage-specific scaffold runs.",
    )
    parser.add_argument(
        "--skip-s2",
        action="store_true",
        help="Skip the S2 PCR-NO stage entirely.",
    )
    parser.add_argument(
        "--strict-metrics",
        action="store_true",
        help="Pass --strict to plot_metrics.py.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "outputs" / "pipeline" / "wave1_pipeline_summary.json"),
        help="Path to the pipeline summary JSON.",
    )
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    finished = datetime.now(timezone.utc)
    return {
        "command": command,
        "cwd": str(cwd),
        "returncode": result.returncode,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def stage_ok(result: dict[str, Any]) -> bool:
    return int(result["returncode"]) == 0


def main() -> int:
    args = parse_args()
    python_executable = sys.executable
    summary_path = Path(args.output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "results").mkdir(parents=True, exist_ok=True)

    s0_command = [
        python_executable,
        str(SCRIPTS_DIR / "run_sanity_check.py"),
        "--config",
        str(REPO_ROOT / "configs" / "S0_sanity.yaml"),
    ]
    s1_run_name = f"{args.run_prefix}_s1"
    s1_command = [
        python_executable,
        str(SCRIPTS_DIR / "train_baseline.py"),
        "--config",
        str(REPO_ROOT / "configs" / "S1_baseline.yaml"),
        "--run-name",
        s1_run_name,
    ]
    s2_run_name = f"{args.run_prefix}_s2"
    s2_command = [
        python_executable,
        str(SCRIPTS_DIR / "train_pcr_no.py"),
        "--config",
        str(REPO_ROOT / "configs" / "S2_main.yaml"),
        "--run-name",
        s2_run_name,
    ]
    if not args.skip_s2:
        s2_command.append("--dry-run")

    metrics_summary = REPO_ROOT / "results" / "MASTER_METRICS.csv"
    metric_inputs = [REPO_ROOT / "outputs", REPO_ROOT / "results"]
    metrics_command = [
        python_executable,
        str(SCRIPTS_DIR / "plot_metrics.py"),
        "--output",
        str(metrics_summary),
        "--recursive",
    ]
    for metric_input in metric_inputs:
        if metric_input.exists():
            metrics_command.extend(["--input", str(metric_input)])
    if args.strict_metrics:
        metrics_command.append("--strict")

    stage_results: dict[str, dict[str, Any]] = {}
    overall_status = "passed"

    for stage_name, command in (("S0", s0_command), ("S1", s1_command)):
        result = run_command(command, cwd=REPO_ROOT)
        stage_results[stage_name] = result
        if not stage_ok(result):
            overall_status = "failed"
            payload = {
                "overall_status": overall_status,
                "pipeline_stage": stage_name,
                "stage_results": stage_results,
            }
            summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(json.dumps(payload, indent=2))
            return 1

    if args.skip_s2:
        stage_results["S2"] = {
            "command": s2_command,
            "cwd": str(REPO_ROOT),
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "status": "skipped",
        }
    else:
        result = run_command(s2_command, cwd=REPO_ROOT)
        stage_results["S2"] = result
        if not stage_ok(result):
            overall_status = "failed"

    metrics_result = run_command(metrics_command, cwd=REPO_ROOT)
    stage_results["METRICS"] = metrics_result
    if not stage_ok(metrics_result):
        overall_status = "failed"

    payload = {
        "overall_status": overall_status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_prefix": args.run_prefix,
        "artifacts": {
            "summary_json": str(summary_path),
            "master_metrics_csv": str(metrics_summary),
        },
        "stage_results": stage_results,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if overall_status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
