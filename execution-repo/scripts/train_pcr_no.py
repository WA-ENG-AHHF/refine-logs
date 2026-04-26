"""Main-stage scaffold for PCR-NO training.

This script intentionally stays lightweight: it loads a YAML config, prepares
stage-specific output locations, instantiates the PCR-NO skeleton, and can run
either a dry-run validation pass or a tiny synthetic training loop.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

try:
    import torch
except ModuleNotFoundError:
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import inspect_dataset_manifest, load_split_arrays

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the scaffold runner."""

    parser = argparse.ArgumentParser(description="Train the PCR-NO main model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "S2_main.yaml",
        help="Path to the YAML config for the main-stage run.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional suffix used to organize logs and summaries for one run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Explicit torch device override, for example 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and model wiring without writing stage artifacts.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config into a mutable dictionary."""

    with path.open("r", encoding="utf-8") as handle:
        raw_text = handle.read()
    if yaml is not None:
        config = yaml.safe_load(raw_text) or {}
    else:
        config = parse_simple_yaml(raw_text)
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping at config root, got: {type(config)!r}")
    return config


def parse_simple_yaml(raw_text: str) -> dict[str, Any]:
    """Parse a minimal indentation-based YAML subset.

    This fallback intentionally supports the small nested mapping style used by
    the execution repo configs, so the scaffold can still dry-run when PyYAML is
    unavailable in the host environment.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML syntax on line {line_number}: {raw_line}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while indent <= stack[-1][0] and len(stack) > 1:
            stack.pop()
        current = stack[-1][1]
        if not value:
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
            continue
        current[key] = parse_scalar(value)
    return root


def parse_scalar(value: str) -> Any:
    """Convert simple YAML scalars into Python primitives."""

    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_training_components() -> tuple[ModuleType, Any, Any]:
    """Import torch-backed training modules only when the environment supports them."""

    if torch is None:
        raise RuntimeError(
            "torch is not installed in this environment. Install execution-repo requirements "
            "to run model construction or optimization."
        )
    from models.losses import build_loss_breakdown
    from models.pcr_no import PCRNO, PCRNOConfig

    return build_loss_breakdown, PCRNO, PCRNOConfig


def rel_l2(predictions: np.ndarray, targets: np.ndarray) -> float:
    numerator = float(np.linalg.norm(predictions - targets))
    denominator = float(np.linalg.norm(targets))
    return numerator / max(denominator, 1e-8)


def load_dataset_payloads(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config.get("dataset", {})
    if not isinstance(dataset_cfg, dict) or "manifest_path" not in dataset_cfg:
        return {"status": "not_configured"}

    manifest_path = Path(str(dataset_cfg["manifest_path"]))
    if not manifest_path.is_absolute():
        manifest_path = (REPO_ROOT / manifest_path).resolve()

    return {
        "status": "loaded",
        "manifest_path": str(manifest_path),
        "inspection": inspect_dataset_manifest(manifest_path).to_dict(),
        "train": load_split_arrays(manifest_path, "train"),
        "val": load_split_arrays(manifest_path, "val"),
    }


def dataset_summary_view(dataset_payload: dict[str, Any]) -> dict[str, Any]:
    if dataset_payload.get("status") != "loaded":
        return dataset_payload
    return {
        "status": dataset_payload["status"],
        "manifest_path": dataset_payload["manifest_path"],
        "inspection": dataset_payload["inspection"],
    }


def flatten_residual_problem(split_payload: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs = np.asarray(split_payload["inputs"], dtype=np.float64)
    residual_targets = np.asarray(split_payload["residual_targets"], dtype=np.float64)
    fine_targets = np.asarray(split_payload["targets"], dtype=np.float64)
    return (
        inputs.reshape(-1, inputs.shape[-1]),
        residual_targets.reshape(-1, residual_targets.shape[-1]),
        fine_targets.reshape(-1, fine_targets.shape[-1]),
    )


def compute_numpy_residual_metrics(
    predicted_residual: np.ndarray,
    split_payload: dict[str, np.ndarray],
) -> dict[str, float]:
    residual_targets = np.asarray(split_payload["residual_targets"], dtype=np.float64)
    fine_targets = np.asarray(split_payload["targets"], dtype=np.float64)
    coarse_interp = np.asarray(split_payload["coarse_interp"], dtype=np.float64)
    pred_residual = predicted_residual.reshape(residual_targets.shape)
    reconstructed = coarse_interp + pred_residual
    boundary_error = float(np.mean(np.abs(reconstructed[:, [0, -1], :] - fine_targets[:, [0, -1], :])))
    conservation_error = float(
        np.mean(
            np.abs(
                reconstructed.sum(axis=1) - fine_targets.sum(axis=1)
            )
        )
    )
    residual_mae = float(np.mean(np.abs(pred_residual - residual_targets)))
    return {
        "rel_l2": rel_l2(reconstructed, fine_targets),
        "pde_residual": residual_mae,
        "bc_violation": boundary_error,
        "conservation_error": conservation_error,
    }


def fit_numpy_residual_model(config: dict[str, Any], dataset_payloads: dict[str, Any]) -> dict[str, Any]:
    train_x, train_residual_y, _ = flatten_residual_problem(dataset_payloads["train"])
    val_x, _, _ = flatten_residual_problem(dataset_payloads["val"])

    train_aug = np.concatenate([train_x, np.ones((train_x.shape[0], 1))], axis=1)
    val_aug = np.concatenate([val_x, np.ones((val_x.shape[0], 1))], axis=1)
    ridge = 1e-6
    gram = train_aug.T @ train_aug + ridge * np.eye(train_aug.shape[1])
    weights = np.linalg.solve(gram, train_aug.T @ train_residual_y)
    train_pred = train_aug @ weights
    val_pred = val_aug @ weights

    train_metrics = compute_numpy_residual_metrics(train_pred, dataset_payloads["train"])
    val_metrics = compute_numpy_residual_metrics(val_pred, dataset_payloads["val"])
    return {
        "backend": "numpy-linear-residual",
        "weights": weights.astype(np.float32).tolist(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }


def choose_device(explicit: str | None) -> Any:
    """Select a torch device with a stable CPU fallback."""

    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def default_run_name(stage: str) -> str:
    """Build a timestamped run identifier for the stage."""

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stage.lower()}_{stamp}"


def resolve_output_plan(
    config: dict[str, Any],
    run_name: str,
) -> dict[str, Path]:
    """Resolve the main output files used by the scaffold.

    The checkpoint and metrics paths remain aligned with the YAML contract, while
    logs and reports are grouped by run name so one main-stage execution is easy
    to inspect.
    """

    stage = str(config.get("stage", "S2"))
    output_cfg = config.get("output", {})
    checkpoint = REPO_ROOT / str(output_cfg.get("checkpoint", f"checkpoints/{stage}_best.pt"))
    metrics_csv = REPO_ROOT / str(
        output_cfg.get("metrics_csv", f"results/{stage}_metric_comparison.csv")
    )
    log_file = REPO_ROOT / "logs" / run_name / "train.log"
    summary_json = REPO_ROOT / "reports" / f"{run_name}_summary.json"
    return {
        "checkpoint": checkpoint,
        "metrics_csv": metrics_csv,
        "log_file": log_file,
        "summary_json": summary_json,
    }


def ensure_output_dirs(output_plan: dict[str, Path]) -> None:
    """Create parent directories for the planned output artifacts."""

    for path in output_plan.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def build_model_config(config: dict[str, Any]) -> PCRNOConfig:
    """Translate YAML model settings into the module config object."""

    _, _, pcr_config_cls = load_training_components()
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    return pcr_config_cls(
        input_dim=int(data_cfg.get("input_dim", 4)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        proj_dim=int(model_cfg.get("proj_dim", 64)),
        output_dim=int(data_cfg.get("output_dim", 1)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        eq_wise_heads=bool(model_cfg.get("eq_wise_heads", True)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        sequence_length=int(data_cfg.get("sequence_length", 64)),
        batch_size=int(train_cfg.get("batch_size", 32)),
    )


def synthetic_batch(
    model_cfg: Any,
    device: Any,
) -> tuple[Any, Any]:
    """Generate a synthetic batch so the scaffold can run before data plumbing exists."""

    features = torch.randn(
        model_cfg.batch_size,
        model_cfg.sequence_length,
        model_cfg.input_dim,
        device=device,
    )
    targets = torch.randn(
        model_cfg.batch_size,
        model_cfg.sequence_length,
        model_cfg.output_dim,
        device=device,
    )
    return features, targets


def run_training_step(
    model: Any,
    optimizer: Any,
    features: Any,
    targets: Any,
    loss_weights: dict[str, float],
) -> dict[str, float]:
    """Execute one synthetic optimization step and return scalar metrics."""

    optimizer.zero_grad(set_to_none=True)
    build_loss_breakdown, _, _ = load_training_components()
    predictions = model(features)
    losses = build_loss_breakdown(predictions=predictions, targets=targets, weights=loss_weights)
    losses["total"].backward()
    optimizer.step()
    return {name: float(value.detach().cpu().item()) for name, value in losses.items()}


def append_metrics_row(metrics_csv: Path, row: dict[str, Any]) -> None:
    """Append a metrics row, creating the CSV header when needed."""

    fieldnames = [
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
    ]
    write_header = not metrics_csv.exists()
    with metrics_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def save_summary(summary_json: Path, payload: dict[str, Any]) -> None:
    """Persist a human-readable run summary for the main stage."""

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_checkpoint(checkpoint_path: Path, model: Any, metadata: dict[str, Any]) -> None:
    """Save the scaffold model state together with simple run metadata."""

    if torch is not None and hasattr(model, "state_dict"):
        torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, checkpoint_path)
        return
    checkpoint_path.write_text(json.dumps({"metadata": metadata}, indent=2), encoding="utf-8")


def main() -> None:
    """Entrypoint used by the S2 main-stage worker."""

    args = parse_args()
    config = load_config(args.config)
    stage = str(config.get("stage", "S2"))
    run_name = args.run_name or default_run_name(stage)
    output_plan = resolve_output_plan(config=config, run_name=run_name)
    summary = {
        "stage": stage,
        "run_name": run_name,
        "config_path": str(args.config.resolve()),
        "outputs": {name: str(path) for name, path in output_plan.items()},
        "torch_available": torch is not None,
        "dataset": {},
    }
    dataset_payloads = load_dataset_payloads(config)
    summary["dataset"] = dataset_summary_view(dataset_payloads)

    if args.dry_run and torch is None:
        summary["device"] = args.device or "unavailable"
        summary["model"] = {
            "input_dim": int(config.get("data", {}).get("input_dim", 4)),
            "hidden_dim": int(config.get("model", {}).get("hidden_dim", 128)),
            "proj_dim": int(config.get("model", {}).get("proj_dim", 64)),
            "output_dim": int(config.get("data", {}).get("output_dim", 1)),
            "num_heads": int(config.get("model", {}).get("num_heads", 4)),
            "eq_wise_heads": bool(config.get("model", {}).get("eq_wise_heads", True)),
            "dropout": float(config.get("model", {}).get("dropout", 0.0)),
            "sequence_length": int(config.get("data", {}).get("sequence_length", 64)),
            "batch_size": int(config.get("train", {}).get("batch_size", 32)),
        }
        summary["dry_run_note"] = (
            "Config and output planning validated without torch; install dependencies "
            "to validate model wiring."
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if torch is None:
        ensure_output_dirs(output_plan)
        dataset_payloads = dataset_payloads
        if dataset_payloads["status"] != "loaded":
            raise RuntimeError("Dataset manifest is required for numpy fallback training.")
        fit_result = fit_numpy_residual_model(config, dataset_payloads)
        summary["device"] = "numpy-cpu"
        summary["model"] = {
            "input_dim": int(config.get("data", {}).get("input_dim", 4)),
            "hidden_dim": int(config.get("model", {}).get("hidden_dim", 128)),
            "proj_dim": int(config.get("model", {}).get("proj_dim", 64)),
            "output_dim": int(config.get("data", {}).get("output_dim", 1)),
            "num_heads": int(config.get("model", {}).get("num_heads", 4)),
            "eq_wise_heads": bool(config.get("model", {}).get("eq_wise_heads", True)),
            "dropout": float(config.get("model", {}).get("dropout", 0.0)),
            "sequence_length": int(config.get("data", {}).get("sequence_length", 64)),
            "batch_size": int(config.get("train", {}).get("batch_size", 32)),
        }
        summary["training_results"] = fit_result
        append_metrics_row(
            output_plan["metrics_csv"],
            {
                "run_id": run_name,
                "stage": stage,
                "epoch": 1,
                "split": "train",
                "tag": fit_result["backend"],
                "rel_l2": fit_result["train_metrics"]["rel_l2"],
                "pde_residual": fit_result["train_metrics"]["pde_residual"],
                "bc_violation": fit_result["train_metrics"]["bc_violation"],
                "conservation_error": fit_result["train_metrics"]["conservation_error"],
                "total": "",
                "status": "trained",
            },
        )
        append_metrics_row(
            output_plan["metrics_csv"],
            {
                "run_id": run_name,
                "stage": stage,
                "epoch": 1,
                "split": "val",
                "tag": fit_result["backend"],
                "rel_l2": fit_result["val_metrics"]["rel_l2"],
                "pde_residual": fit_result["val_metrics"]["pde_residual"],
                "bc_violation": fit_result["val_metrics"]["bc_violation"],
                "conservation_error": fit_result["val_metrics"]["conservation_error"],
                "total": "",
                "status": "evaluated",
            },
        )
        save_summary(output_plan["summary_json"], summary)
        save_checkpoint(output_plan["checkpoint"], model={"backend": "numpy-linear-residual"}, metadata=summary)
        output_plan["log_file"].write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    build_loss_breakdown, pcr_model_cls, _ = load_training_components()
    device = choose_device(args.device)
    model_cfg = build_model_config(config)
    model = pcr_model_cls(model_cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("train", {}).get("learning_rate", 1e-3)),
    )
    loss_weights = {
        name: float(value)
        for name, value in (config.get("loss", {}) or {}).items()
    }
    features, targets = synthetic_batch(model_cfg=model_cfg, device=device)
    summary["device"] = str(device)
    summary["model"] = model_cfg.to_dict()

    if args.dry_run:
        dry_losses = build_loss_breakdown(
            predictions=model(features),
            targets=targets,
            weights=loss_weights,
        )
        summary["dry_run_losses"] = {
            name: float(value.detach().cpu().item())
            for name, value in dry_losses.items()
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    ensure_output_dirs(output_plan)
    metrics = run_training_step(
        model=model,
        optimizer=optimizer,
        features=features,
        targets=targets,
        loss_weights=loss_weights,
    )
    summary["final_metrics"] = metrics
    append_metrics_row(
        output_plan["metrics_csv"],
        {
            "run_id": run_name,
            "stage": stage,
            "epoch": 1,
            "split": "train",
            "tag": "synthetic_step",
            "rel_l2": metrics.get("l2", ""),
            "pde_residual": metrics.get("pde", ""),
            "bc_violation": metrics.get("bc", ""),
            "conservation_error": metrics.get("conservation", ""),
            "total": metrics.get("total", ""),
            "status": "synthetic_train_step",
        },
    )
    save_summary(output_plan["summary_json"], summary)
    save_checkpoint(output_plan["checkpoint"], model, metadata=summary)
    output_plan["log_file"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
