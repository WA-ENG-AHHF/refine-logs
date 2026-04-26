from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml

    YAML_AVAILABLE = True
except ModuleNotFoundError:
    yaml = None
    YAML_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.fno_baseline import FNOBaselineStub, TORCH_AVAILABLE
from datasets import inspect_dataset_manifest, load_split_arrays
from training import compute_direct_metrics_numpy, fit_torch_direct_supervised, set_global_seed


@dataclass
class RunArtifacts:
    run_dir: Path
    summary_path: Path
    checkpoint_path: Path
    metrics_path: Path
    resolved_config_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S1 baseline training scaffold.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "S1_baseline.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional override for the run directory name.",
    )
    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp_slug(dt: datetime) -> str:
    return dt.strftime("%Y%m%d-%H%M%S")


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        raw_text = handle.read()

    if YAML_AVAILABLE:
        config = yaml.safe_load(raw_text) or {}
    else:
        config = parse_simple_yaml(raw_text)

    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping at config root, got {type(config).__name__}.")
    return config


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
    return "\n".join(line for line in lines if line != "")


def build_run_name(config: dict[str, Any], override: str | None, started_at: datetime) -> str:
    if override:
        return override

    configured_name = config.get("output", {}).get("run_name")
    if configured_name:
        return str(configured_name)

    stage = config.get("stage", "stage")
    model_name = config.get("model", {}).get("name", "model")
    safe_model_name = str(model_name).replace(" ", "_").replace("/", "-")
    return f"{stage}_{safe_model_name}_{timestamp_slug(started_at)}"


def prepare_artifacts(config: dict[str, Any], run_name: str) -> RunArtifacts:
    output_cfg = config.get("output", {})
    root_dir = REPO_ROOT / output_cfg.get("root_dir", "outputs")
    run_dir = root_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / output_cfg.get("checkpoint_file", "checkpoints/S1_baseline_placeholder.pt")
    summary_path = run_dir / output_cfg.get("summary_file", "run_summary.json")
    metrics_path = run_dir / output_cfg.get("metrics_file", "results/S1_metrics.csv")
    resolved_config_path = run_dir / output_cfg.get("resolved_config_file", "resolved_config.yaml")

    for path in (checkpoint_path, summary_path, metrics_path, resolved_config_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    return RunArtifacts(
        run_dir=run_dir,
        summary_path=summary_path,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        resolved_config_path=resolved_config_path,
    )


def build_model(config: dict[str, Any]) -> FNOBaselineStub:
    model_cfg = config.get("model", {})
    return FNOBaselineStub(
        input_dim=int(model_cfg.get("input_dim", 4)),
        hidden_dim=int(model_cfg.get("hidden_dim", 32)),
        output_dim=int(model_cfg.get("output_dim", 2)),
        depth=int(model_cfg.get("depth", 2)),
    )


def inspect_configured_dataset(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config.get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        return {"status": "not_configured"}

    manifest_path = dataset_cfg.get("manifest_path")
    if not manifest_path:
        return {"status": "not_configured"}

    resolved_manifest = (REPO_ROOT / str(manifest_path)).resolve()
    inspection = inspect_dataset_manifest(resolved_manifest)
    return {"status": "loaded", **inspection.to_dict()}


def count_parameters(model: Any) -> int:
    if hasattr(model, "parameter_count"):
        return int(model.parameter_count())
    if hasattr(model, "parameters"):
        return sum(parameter.numel() for parameter in model.parameters())
    return 0


def load_training_data(config: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    dataset_cfg = config.get("dataset", {})
    manifest_path = Path(str(dataset_cfg.get("manifest_path", "")))
    if not manifest_path.is_absolute():
        manifest_path = (REPO_ROOT / manifest_path).resolve()
    return {
        "train": load_split_arrays(manifest_path, "train"),
        "val": load_split_arrays(manifest_path, "val"),
    }


def flatten_supervised_arrays(split_payload: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.asarray(split_payload["inputs"], dtype=np.float64)
    targets = np.asarray(split_payload["targets"], dtype=np.float64)
    return inputs.reshape(-1, inputs.shape[-1]), targets.reshape(-1, targets.shape[-1])


def fit_numpy_linear_baseline(
    train_payload: dict[str, np.ndarray], val_payload: dict[str, np.ndarray], config: dict[str, Any]
) -> dict[str, Any]:
    train_x, train_y = flatten_supervised_arrays(train_payload)
    val_x, val_y = flatten_supervised_arrays(val_payload)

    train_aug = np.concatenate([train_x, np.ones((train_x.shape[0], 1))], axis=1)
    val_aug = np.concatenate([val_x, np.ones((val_x.shape[0], 1))], axis=1)
    ridge = 1e-6
    gram = train_aug.T @ train_aug + ridge * np.eye(train_aug.shape[1])
    weights = np.linalg.solve(gram, train_aug.T @ train_y)

    train_pred = train_aug @ weights
    val_pred = val_aug @ weights
    loss_weights = {
        "l2": float(config.get("loss", {}).get("l2", 1.0)),
        "pde": float(config.get("loss", {}).get("pde", 0.0)),
        "bc": float(config.get("loss", {}).get("bc", 0.0)),
        "conservation": float(config.get("loss", {}).get("conservation", 0.0)),
    }
    train_pred_seq = train_pred.reshape(train_payload["targets"].shape)
    val_pred_seq = val_pred.reshape(val_payload["targets"].shape)
    train_metrics = compute_direct_metrics_numpy(train_pred_seq, train_payload["targets"], loss_weights)
    val_metrics = compute_direct_metrics_numpy(val_pred_seq, val_payload["targets"], loss_weights)
    return {
        "backend": "numpy-linear",
        "epochs": 1,
        "best_epoch": 1,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "history": [
            {
                "epoch": 1.0,
                "train_rel_l2": train_metrics["rel_l2"],
                "val_rel_l2": val_metrics["rel_l2"],
                "train_total": train_metrics["total"],
                "val_total": val_metrics["total"],
            }
        ],
        "weights": weights.astype(np.float32).tolist(),
        "train_samples": int(train_payload["inputs"].shape[0]),
        "val_samples": int(val_payload["inputs"].shape[0]),
    }


def write_metrics_rows(metrics_path: Path, rows: list[dict[str, Any]]) -> None:
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "stage",
                "epoch",
                "split",
                "tag",
                "rel_l2",
                "pde_residual",
                "bc_violation",
                "conservation_error",
                "total",
                "status",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_checkpoint_metadata(
    checkpoint_path: Path,
    config: dict[str, Any],
    model: Any,
    started_at: datetime,
    finished_at: datetime,
    training_state: dict[str, Any],
) -> None:
    checkpoint_payload = {
        "kind": "baseline_scaffold_checkpoint",
        "stage": config.get("stage", "S1"),
        "model_name": config.get("model", {}).get("name", "FNO-baseline"),
        "created_at_utc": finished_at.isoformat(),
        "started_at_utc": started_at.isoformat(),
        "parameter_count": count_parameters(model),
        "state_dict": model.state_dict(),
        "training_state": training_state,
    }
    if TORCH_AVAILABLE:
        import torch

        torch.save(checkpoint_payload, checkpoint_path)
        return

    with checkpoint_path.open("w", encoding="utf-8") as handle:
        json.dump(checkpoint_payload, handle, indent=2)


def write_resolved_config(config: dict[str, Any], artifacts: RunArtifacts, run_name: str) -> None:
    resolved = dict(config)
    resolved["runtime"] = {
        "repo_root": str(REPO_ROOT),
        "run_name": run_name,
        "run_dir": str(artifacts.run_dir),
        "summary_path": str(artifacts.summary_path),
        "checkpoint_path": str(artifacts.checkpoint_path),
        "metrics_path": str(artifacts.metrics_path),
    }
    with artifacts.resolved_config_path.open("w", encoding="utf-8") as handle:
        if YAML_AVAILABLE:
            yaml.safe_dump(resolved, handle, sort_keys=False)
        else:
            handle.write(dump_simple_yaml(resolved))
            handle.write("\n")


def write_summary(
    summary_path: Path,
    config_path: Path,
    config: dict[str, Any],
    artifacts: RunArtifacts,
    run_name: str,
    model: Any,
    started_at: datetime,
    finished_at: datetime,
    training_results: dict[str, Any],
) -> None:
    summary = {
        "status": "trained_baseline",
        "stage": config.get("stage", "S1"),
        "run_name": run_name,
        "config_path": str(config_path),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "cwd": os.getcwd(),
        "model": {
            "name": config.get("model", {}).get("name", "FNO-baseline"),
            "parameter_count": count_parameters(model),
        },
        "train": config.get("train", {}),
        "loss": config.get("loss", {}),
        "dataset": inspect_configured_dataset(config),
        "training_results": training_results,
        "artifacts": {key: str(value) for key, value in asdict(artifacts).items()},
        "notes": [
            "This run consumed manifest-backed dataset files.",
            "Torch environments use a real epoch-based supervised loop.",
            "Torch-free environments retain a portable numpy fallback.",
        ],
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    started_at = utc_now()

    config = load_config(config_path)
    run_name = build_run_name(config, args.run_name, started_at)
    artifacts = prepare_artifacts(config, run_name)
    model = build_model(config)
    split_payloads = load_training_data(config)
    train_cfg = config.get("train", {}) or {}
    set_global_seed(int(train_cfg.get("seed", 42)))
    best_state_dict: dict[str, Any] | None = None
    if TORCH_AVAILABLE:
        from models.losses import build_loss_breakdown

        training_results, best_state_dict = fit_torch_direct_supervised(
            model=model,
            train_payload=split_payloads["train"],
            val_payload=split_payloads["val"],
            config=config,
            build_loss_breakdown=build_loss_breakdown,
        )
    else:
        training_results = fit_numpy_linear_baseline(split_payloads["train"], split_payloads["val"], config)

    write_metrics_rows(
        artifacts.metrics_path,
        [
            {
                "run_id": run_name,
                "stage": "S1",
                "epoch": training_results.get("best_epoch", training_results.get("epochs", 1)),
                "split": "train",
                "tag": training_results["backend"],
                "rel_l2": training_results["train_metrics"]["rel_l2"],
                "pde_residual": training_results["train_metrics"].get("pde_residual", ""),
                "bc_violation": training_results["train_metrics"].get("bc_violation", ""),
                "conservation_error": training_results["train_metrics"].get("conservation_error", ""),
                "total": training_results["train_metrics"].get("total", ""),
                "status": "trained",
            },
            {
                "run_id": run_name,
                "stage": "S1",
                "epoch": training_results.get("best_epoch", training_results.get("epochs", 1)),
                "split": "val",
                "tag": training_results["backend"],
                "rel_l2": training_results["val_metrics"]["rel_l2"],
                "pde_residual": training_results["val_metrics"].get("pde_residual", ""),
                "bc_violation": training_results["val_metrics"].get("bc_violation", ""),
                "conservation_error": training_results["val_metrics"].get("conservation_error", ""),
                "total": training_results["val_metrics"].get("total", ""),
                "status": "evaluated",
            },
        ],
    )
    write_resolved_config(config, artifacts, run_name)

    finished_at = utc_now()
    if best_state_dict is not None and hasattr(model, "load_state_dict"):
        model.load_state_dict(best_state_dict)
    save_checkpoint_metadata(
        artifacts.checkpoint_path,
        config,
        model,
        started_at,
        finished_at,
        training_state={
            "epoch": int(training_results.get("best_epoch", training_results.get("epochs", 1))),
            "global_step": int(
                training_results.get("epochs", 1) * split_payloads["train"]["inputs"].shape[0]
            ),
            "status": "trained_baseline",
            "backend": training_results["backend"],
            "train_rel_l2": training_results["train_metrics"]["rel_l2"],
            "val_rel_l2": training_results["val_metrics"]["rel_l2"],
        },
    )
    write_summary(
        summary_path=artifacts.summary_path,
        config_path=config_path,
        config=config,
        artifacts=artifacts,
        run_name=run_name,
        model=model,
        started_at=started_at,
        finished_at=finished_at,
        training_results=training_results,
    )

    print(f"Initialized baseline run: {run_name}")
    print(f"Run directory: {artifacts.run_dir}")
    print(f"Summary: {artifacts.summary_path}")
    print(f"Checkpoint metadata: {artifacts.checkpoint_path}")
    print(f"Metrics: {artifacts.metrics_path}")
    print(f"Train rel-L2: {training_results['train_metrics']['rel_l2']:.6f}")
    print(f"Val rel-L2: {training_results['val_metrics']['rel_l2']:.6f}")


if __name__ == "__main__":
    main()
