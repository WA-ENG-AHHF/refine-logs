from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False


def set_global_seed(seed: int) -> None:
    """Set seeds across python, numpy, and torch when available."""

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def rel_l2(predictions: np.ndarray, targets: np.ndarray) -> float:
    numerator = float(np.linalg.norm(predictions - targets))
    denominator = float(np.linalg.norm(targets))
    return numerator / max(denominator, 1e-8)


def weighted_total_from_metrics(metrics: dict[str, float], weights: dict[str, float]) -> float:
    return (
        weights.get("l2", 0.0) * metrics.get("rel_l2", 0.0)
        + weights.get("pde", 0.0) * metrics.get("pde_residual", 0.0)
        + weights.get("bc", 0.0) * metrics.get("bc_violation", 0.0)
        + weights.get("conservation", 0.0) * metrics.get("conservation_error", 0.0)
    )


def compute_direct_metrics_numpy(
    predictions: np.ndarray,
    targets: np.ndarray,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute portable direct-regression metrics on dense sequence tensors."""

    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    residual = predictions - targets
    metrics = {
        "rel_l2": rel_l2(predictions, targets),
        "pde_residual": float(np.mean(np.abs(residual))),
        "bc_violation": float(np.mean(np.abs(residual[:, [0, -1], :]))),
        "conservation_error": float(
            np.mean(np.abs(predictions.sum(axis=1) - targets.sum(axis=1)))
        ),
    }
    metrics["total"] = weighted_total_from_metrics(metrics, weights or {})
    return metrics


def compute_residual_metrics_numpy(
    predicted_residual: np.ndarray,
    residual_targets: np.ndarray,
    fine_targets: np.ndarray,
    coarse_interp: np.ndarray,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute coarse-to-fine residual metrics on dense sequence tensors."""

    predicted_residual = np.asarray(predicted_residual, dtype=np.float64)
    residual_targets = np.asarray(residual_targets, dtype=np.float64)
    fine_targets = np.asarray(fine_targets, dtype=np.float64)
    coarse_interp = np.asarray(coarse_interp, dtype=np.float64)
    reconstructed = coarse_interp + predicted_residual
    residual_error = predicted_residual - residual_targets
    metrics = {
        "rel_l2": rel_l2(reconstructed, fine_targets),
        "pde_residual": float(np.mean(np.abs(residual_error))),
        "bc_violation": float(
            np.mean(np.abs(reconstructed[:, [0, -1], :] - fine_targets[:, [0, -1], :]))
        ),
        "conservation_error": float(
            np.mean(np.abs(reconstructed.sum(axis=1) - fine_targets.sum(axis=1)))
        ),
    }
    metrics["total"] = weighted_total_from_metrics(metrics, weights or {})
    return metrics


def _loss_weights_from_config(config: dict[str, Any]) -> dict[str, float]:
    loss_cfg = config.get("loss", {}) or {}
    return {
        "l2": float(loss_cfg.get("l2", 1.0)),
        "pde": float(loss_cfg.get("pde", 0.0)),
        "bc": float(loss_cfg.get("bc", 0.0)),
        "conservation": float(loss_cfg.get("conservation", 0.0)),
    }


def _device_from_config(explicit: str | None) -> Any:
    if torch is None:
        raise RuntimeError("torch is required for direct supervised training.")
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _history_log_epochs(num_epochs: int) -> set[int]:
    log_epochs = {1, num_epochs}
    stride = max(num_epochs // 10, 1)
    log_epochs.update(range(stride, num_epochs + 1, stride))
    return log_epochs


def _evaluate_torch_direct_model(
    model: Any,
    inputs: Any,
    targets: Any,
    build_loss_breakdown: Any,
    loss_weights: dict[str, float],
) -> dict[str, float]:
    with torch.no_grad():
        predictions = model(inputs)
        losses = build_loss_breakdown(
            predictions=predictions,
            targets=targets,
            weights=loss_weights,
        )
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        metrics = compute_direct_metrics_numpy(predictions_np, targets_np, loss_weights)
    metrics["torch_l2"] = float(losses["l2"].detach().cpu().item())
    metrics["torch_total"] = float(losses["total"].detach().cpu().item())
    return metrics


def fit_torch_direct_supervised(
    model: Any,
    train_payload: dict[str, np.ndarray],
    val_payload: dict[str, np.ndarray],
    config: dict[str, Any],
    build_loss_breakdown: Any,
    device: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run a real epoch-based torch training loop for direct baseline regression."""

    if torch is None:
        raise RuntimeError("torch is not installed.")

    train_cfg = config.get("train", {}) or {}
    seed = int(train_cfg.get("seed", 42))
    epochs = int(train_cfg.get("epochs", 100))
    batch_size = int(train_cfg.get("batch_size", 32))
    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    loss_weights = _loss_weights_from_config(config)
    run_device = _device_from_config(device)
    model = model.to(run_device)

    set_global_seed(seed)

    train_inputs = torch.tensor(np.asarray(train_payload["inputs"], dtype=np.float32), device=run_device)
    train_targets = torch.tensor(np.asarray(train_payload["targets"], dtype=np.float32), device=run_device)
    val_inputs = torch.tensor(np.asarray(val_payload["inputs"], dtype=np.float32), device=run_device)
    val_targets = torch.tensor(np.asarray(val_payload["targets"], dtype=np.float32), device=run_device)

    dataset = TensorDataset(train_inputs, train_targets)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    best_val_total = float("inf")
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    log_epochs = _history_log_epochs(epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_inputs, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_inputs)
            losses = build_loss_breakdown(
                predictions=predictions,
                targets=batch_targets,
                weights=loss_weights,
            )
            losses["total"].backward()
            optimizer.step()

        model.eval()
        train_metrics = _evaluate_torch_direct_model(
            model,
            train_inputs,
            train_targets,
            build_loss_breakdown,
            loss_weights,
        )
        val_metrics = _evaluate_torch_direct_model(
            model,
            val_inputs,
            val_targets,
            build_loss_breakdown,
            loss_weights,
        )

        if val_metrics["total"] < best_val_total:
            best_val_total = val_metrics["total"]
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        if epoch in log_epochs:
            history.append(
                {
                    "epoch": float(epoch),
                    "train_rel_l2": train_metrics["rel_l2"],
                    "val_rel_l2": val_metrics["rel_l2"],
                    "train_total": train_metrics["total"],
                    "val_total": val_metrics["total"],
                }
            )

    model.load_state_dict(best_state_dict)
    model.eval()
    best_train_metrics = _evaluate_torch_direct_model(
        model,
        train_inputs,
        train_targets,
        build_loss_breakdown,
        loss_weights,
    )
    best_val_metrics = _evaluate_torch_direct_model(
        model,
        val_inputs,
        val_targets,
        build_loss_breakdown,
        loss_weights,
    )

    summary = {
        "backend": "torch-direct-supervised",
        "device": str(run_device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "best_epoch": best_epoch,
        "history": history,
        "train_metrics": best_train_metrics,
        "val_metrics": best_val_metrics,
        "train_samples": int(train_payload["inputs"].shape[0]),
        "val_samples": int(val_payload["inputs"].shape[0]),
    }
    return summary, best_state_dict


def fit_torch_residual_supervised(
    model: Any,
    train_payload: dict[str, np.ndarray],
    val_payload: dict[str, np.ndarray],
    config: dict[str, Any],
    build_loss_breakdown: Any,
    device: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run a real epoch-based torch training loop for residual correction."""

    if torch is None:
        raise RuntimeError("torch is not installed.")

    train_cfg = config.get("train", {}) or {}
    seed = int(train_cfg.get("seed", 42))
    epochs = int(train_cfg.get("epochs", 100))
    batch_size = int(train_cfg.get("batch_size", 32))
    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    loss_weights = _loss_weights_from_config(config)
    run_device = _device_from_config(device)

    set_global_seed(seed)
    model = model.to(run_device)

    train_inputs = torch.tensor(np.asarray(train_payload["inputs"], dtype=np.float32), device=run_device)
    train_residual_targets = torch.tensor(
        np.asarray(train_payload["residual_targets"], dtype=np.float32),
        device=run_device,
    )
    val_inputs = torch.tensor(np.asarray(val_payload["inputs"], dtype=np.float32), device=run_device)
    val_residual_targets = torch.tensor(
        np.asarray(val_payload["residual_targets"], dtype=np.float32),
        device=run_device,
    )
    train_fine_targets = np.asarray(train_payload["targets"], dtype=np.float64)
    val_fine_targets = np.asarray(val_payload["targets"], dtype=np.float64)
    train_coarse_interp = np.asarray(train_payload["coarse_interp"], dtype=np.float64)
    val_coarse_interp = np.asarray(val_payload["coarse_interp"], dtype=np.float64)

    dataset = TensorDataset(train_inputs, train_residual_targets)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    best_val_total = float("inf")
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    log_epochs = _history_log_epochs(epochs)

    def evaluate_model(
        inputs: Any,
        residual_targets: Any,
        fine_targets: np.ndarray,
        coarse_interp: np.ndarray,
    ) -> dict[str, float]:
        with torch.no_grad():
            predicted_residual = model(inputs)
            losses = build_loss_breakdown(
                predictions=predicted_residual,
                targets=residual_targets,
                weights=loss_weights,
            )
            predicted_residual_np = predicted_residual.detach().cpu().numpy()
            residual_targets_np = residual_targets.detach().cpu().numpy()
            metrics = compute_residual_metrics_numpy(
                predicted_residual_np,
                residual_targets_np,
                fine_targets,
                coarse_interp,
                loss_weights,
            )
        metrics["torch_residual_l2"] = float(losses["l2"].detach().cpu().item())
        metrics["torch_residual_total"] = float(losses["total"].detach().cpu().item())
        return metrics

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_inputs, batch_residual_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_inputs)
            losses = build_loss_breakdown(
                predictions=predictions,
                targets=batch_residual_targets,
                weights=loss_weights,
            )
            losses["total"].backward()
            optimizer.step()

        model.eval()
        train_metrics = evaluate_model(
            train_inputs,
            train_residual_targets,
            train_fine_targets,
            train_coarse_interp,
        )
        val_metrics = evaluate_model(
            val_inputs,
            val_residual_targets,
            val_fine_targets,
            val_coarse_interp,
        )

        if val_metrics["total"] < best_val_total:
            best_val_total = val_metrics["total"]
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        if epoch in log_epochs:
            history.append(
                {
                    "epoch": float(epoch),
                    "train_rel_l2": train_metrics["rel_l2"],
                    "val_rel_l2": val_metrics["rel_l2"],
                    "train_total": train_metrics["total"],
                    "val_total": val_metrics["total"],
                }
            )

    model.load_state_dict(best_state_dict)
    model.eval()
    best_train_metrics = evaluate_model(
        train_inputs,
        train_residual_targets,
        train_fine_targets,
        train_coarse_interp,
    )
    best_val_metrics = evaluate_model(
        val_inputs,
        val_residual_targets,
        val_fine_targets,
        val_coarse_interp,
    )

    summary = {
        "backend": "torch-residual-supervised",
        "device": str(run_device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "best_epoch": best_epoch,
        "history": history,
        "train_metrics": best_train_metrics,
        "val_metrics": best_val_metrics,
        "train_samples": int(train_payload["inputs"].shape[0]),
        "val_samples": int(val_payload["inputs"].shape[0]),
    }
    return summary, best_state_dict
