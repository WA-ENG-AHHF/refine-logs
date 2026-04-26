from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DatasetInspection:
    manifest_path: Path
    dataset_name: str
    dataset_version: str
    total_samples: int
    split_counts: dict[str, int]
    input_shape: list[int]
    target_shape: list[int]
    input_dim: int
    target_dim: int
    contains_nan: bool
    contains_inf: bool
    files: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "total_samples": self.total_samples,
            "split_counts": self.split_counts,
            "input_shape": self.input_shape,
            "target_shape": self.target_shape,
            "input_dim": self.input_dim,
            "target_dim": self.target_dim,
            "contains_nan": self.contains_nan,
            "contains_inf": self.contains_inf,
            "files": self.files,
        }


def load_dataset_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Dataset manifest root must be a JSON object.")
    return payload


def _resolve_split_files(manifest_path: Path, split_entries: list[dict[str, Any]]) -> list[Path]:
    resolved: list[Path] = []
    for entry in split_entries:
        relative_path = entry.get("path")
        if not relative_path:
            raise ValueError(f"Split entry is missing `path` in {manifest_path}")
        resolved.append((manifest_path.parent / str(relative_path)).resolve())
    return resolved


def _inspect_npz_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as payload:
        if "inputs" not in payload or "targets" not in payload:
            raise ValueError(f"{path} must contain `inputs` and `targets` arrays.")
        inputs = np.asarray(payload["inputs"])
        targets = np.asarray(payload["targets"])
    return inputs, targets


def inspect_dataset_manifest(manifest_path: Path) -> DatasetInspection:
    manifest = load_dataset_manifest(manifest_path)
    splits = manifest.get("splits", {})
    if not isinstance(splits, dict) or not splits:
        raise ValueError("Manifest must contain a non-empty `splits` mapping.")

    split_counts: dict[str, int] = {}
    files: list[str] = []
    total_samples = 0
    input_shape: list[int] | None = None
    target_shape: list[int] | None = None
    input_dim: int | None = None
    target_dim: int | None = None
    contains_nan = False
    contains_inf = False

    for split_name, split_entries in splits.items():
        if not isinstance(split_entries, list):
            raise ValueError(f"Split `{split_name}` must be a list.")

        split_total = 0
        for file_path in _resolve_split_files(manifest_path, split_entries):
            if not file_path.exists():
                raise FileNotFoundError(f"Dataset file missing: {file_path}")
            files.append(str(file_path))
            inputs, targets = _inspect_npz_file(file_path)

            if inputs.shape[0] != targets.shape[0]:
                raise ValueError(f"Input/target sample count mismatch in {file_path}")

            split_total += int(inputs.shape[0])
            total_samples += int(inputs.shape[0])

            if input_shape is None:
                input_shape = list(inputs.shape)
                target_shape = list(targets.shape)
                input_dim = int(inputs.shape[-1]) if inputs.ndim >= 1 else 1
                target_dim = int(targets.shape[-1]) if targets.ndim >= 1 else 1

            contains_nan = contains_nan or bool(np.isnan(inputs).any() or np.isnan(targets).any())
            contains_inf = contains_inf or bool(np.isinf(inputs).any() or np.isinf(targets).any())

        split_counts[str(split_name)] = split_total

    if input_shape is None or target_shape is None or input_dim is None or target_dim is None:
        raise ValueError("Manifest inspection found no samples.")

    return DatasetInspection(
        manifest_path=manifest_path,
        dataset_name=str(manifest.get("dataset_name", "unknown")),
        dataset_version=str(manifest.get("dataset_version", "unknown")),
        total_samples=total_samples,
        split_counts=split_counts,
        input_shape=input_shape,
        target_shape=target_shape,
        input_dim=input_dim,
        target_dim=target_dim,
        contains_nan=contains_nan,
        contains_inf=contains_inf,
        files=files,
    )


def load_split_arrays(manifest_path: Path, split_name: str) -> dict[str, np.ndarray]:
    manifest = load_dataset_manifest(manifest_path)
    splits = manifest.get("splits", {})
    if split_name not in splits:
        raise KeyError(f"Split `{split_name}` not found in {manifest_path}")

    merged: dict[str, list[np.ndarray]] = {}
    for file_path in _resolve_split_files(manifest_path, splits[split_name]):
        with np.load(file_path) as payload:
            for key in payload.files:
                merged.setdefault(key, []).append(np.asarray(payload[key]))

    return {key: np.concatenate(value, axis=0) for key, value in merged.items()}
