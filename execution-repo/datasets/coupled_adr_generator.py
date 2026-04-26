from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CoupledADRConfig:
    dataset_name: str
    dataset_version: str
    output_dir: str
    train_samples: int
    val_samples: int
    test_samples: int
    coarse_size: int
    fine_size: int
    total_time: float
    base_num_steps: int
    seed: int
    coupling_min: float
    coupling_max: float
    diffusion_u_min: float
    diffusion_u_max: float
    diffusion_v_min: float
    diffusion_v_max: float
    advection_u_min: float
    advection_u_max: float
    advection_v_min: float
    advection_v_max: float
    reaction_u_min: float
    reaction_u_max: float
    reaction_v_min: float
    reaction_v_max: float
    source_amplitude_min: float
    source_amplitude_max: float
    source_width_min: float
    source_width_max: float


def sample_config_dict() -> dict[str, Any]:
    return {
        "dataset": {
            "name": "coupled-adr-v1",
            "version": "2026-04-26",
            "output_dir": "data/coupled_adr_v1",
            "train_samples": 256,
            "val_samples": 64,
            "test_samples": 64,
            "seed": 42,
        },
        "grid": {
            "coarse_size": 16,
            "fine_size": 64,
        },
        "time": {
            "total_time": 0.12,
            "base_num_steps": 60,
        },
        "physics": {
            "coupling_min": 0.05,
            "coupling_max": 0.35,
            "diffusion_u_min": 0.008,
            "diffusion_u_max": 0.02,
            "diffusion_v_min": 0.01,
            "diffusion_v_max": 0.025,
            "advection_u_min": -0.2,
            "advection_u_max": 0.2,
            "advection_v_min": -0.15,
            "advection_v_max": 0.15,
            "reaction_u_min": -0.08,
            "reaction_u_max": 0.03,
            "reaction_v_min": -0.08,
            "reaction_v_max": 0.03,
            "source_amplitude_min": 0.15,
            "source_amplitude_max": 0.6,
            "source_width_min": 0.05,
            "source_width_max": 0.18,
        },
    }


def config_from_dict(payload: dict[str, Any]) -> CoupledADRConfig:
    dataset = payload["dataset"]
    grid = payload["grid"]
    time_cfg = payload["time"]
    physics = payload["physics"]
    return CoupledADRConfig(
        dataset_name=str(dataset["name"]),
        dataset_version=str(dataset["version"]),
        output_dir=str(dataset["output_dir"]),
        train_samples=int(dataset["train_samples"]),
        val_samples=int(dataset["val_samples"]),
        test_samples=int(dataset["test_samples"]),
        seed=int(dataset["seed"]),
        coarse_size=int(grid["coarse_size"]),
        fine_size=int(grid["fine_size"]),
        total_time=float(time_cfg["total_time"]),
        base_num_steps=int(time_cfg["base_num_steps"]),
        coupling_min=float(physics["coupling_min"]),
        coupling_max=float(physics["coupling_max"]),
        diffusion_u_min=float(physics["diffusion_u_min"]),
        diffusion_u_max=float(physics["diffusion_u_max"]),
        diffusion_v_min=float(physics["diffusion_v_min"]),
        diffusion_v_max=float(physics["diffusion_v_max"]),
        advection_u_min=float(physics["advection_u_min"]),
        advection_u_max=float(physics["advection_u_max"]),
        advection_v_min=float(physics["advection_v_min"]),
        advection_v_max=float(physics["advection_v_max"]),
        reaction_u_min=float(physics["reaction_u_min"]),
        reaction_u_max=float(physics["reaction_u_max"]),
        reaction_v_min=float(physics["reaction_v_min"]),
        reaction_v_max=float(physics["reaction_v_max"]),
        source_amplitude_min=float(physics["source_amplitude_min"]),
        source_amplitude_max=float(physics["source_amplitude_max"]),
        source_width_min=float(physics["source_width_min"]),
        source_width_max=float(physics["source_width_max"]),
    )


def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def sample_case_parameters(config: CoupledADRConfig, rng: np.random.Generator) -> dict[str, float]:
    return {
        "coupling": _uniform(rng, config.coupling_min, config.coupling_max),
        "diffusion_u": _uniform(rng, config.diffusion_u_min, config.diffusion_u_max),
        "diffusion_v": _uniform(rng, config.diffusion_v_min, config.diffusion_v_max),
        "advection_u": _uniform(rng, config.advection_u_min, config.advection_u_max),
        "advection_v": _uniform(rng, config.advection_v_min, config.advection_v_max),
        "reaction_u": _uniform(rng, config.reaction_u_min, config.reaction_u_max),
        "reaction_v": _uniform(rng, config.reaction_v_min, config.reaction_v_max),
        "source_u_amp": _uniform(rng, config.source_amplitude_min, config.source_amplitude_max),
        "source_v_amp": _uniform(rng, config.source_amplitude_min, config.source_amplitude_max),
        "source_u_center": _uniform(rng, 0.15, 0.85),
        "source_v_center": _uniform(rng, 0.15, 0.85),
        "source_u_width": _uniform(rng, config.source_width_min, config.source_width_max),
        "source_v_width": _uniform(rng, config.source_width_min, config.source_width_max),
        "init_u_amp1": _uniform(rng, 0.3, 0.9),
        "init_u_amp2": _uniform(rng, 0.1, 0.45),
        "init_v_amp1": _uniform(rng, 0.3, 0.9),
        "init_v_amp2": _uniform(rng, 0.1, 0.45),
        "phase_u": _uniform(rng, 0.0, math.pi),
        "phase_v": _uniform(rng, 0.0, math.pi),
    }


def _gaussian(x: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-((x - center) ** 2) / max(width * width, 1e-8))


def source_profile(x: np.ndarray, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    source_u = params["source_u_amp"] * _gaussian(x, params["source_u_center"], params["source_u_width"])
    source_v = params["source_v_amp"] * _gaussian(x, params["source_v_center"], params["source_v_width"])
    return source_u.astype(np.float32), source_v.astype(np.float32)


def initial_state(x: np.ndarray, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    u0 = (
        params["init_u_amp1"] * np.sin(math.pi * x + params["phase_u"])
        + params["init_u_amp2"] * np.sin(2.0 * math.pi * x)
    )
    v0 = (
        params["init_v_amp1"] * np.sin(math.pi * x + params["phase_v"])
        + params["init_v_amp2"] * np.sin(3.0 * math.pi * x)
    )
    u0[[0, -1]] = 0.0
    v0[[0, -1]] = 0.0
    return u0.astype(np.float32), v0.astype(np.float32)


def stable_num_steps(config: CoupledADRConfig, params: dict[str, float], dx: float) -> int:
    max_diffusion = max(params["diffusion_u"], params["diffusion_v"])
    max_advection = max(abs(params["advection_u"]), abs(params["advection_v"]), 1e-6)
    reaction_scale = max(
        abs(params["reaction_u"]) + params["coupling"],
        abs(params["reaction_v"]) + params["coupling"],
        1e-6,
    )
    diffusion_dt = 0.45 * dx * dx / max(max_diffusion, 1e-6)
    advection_dt = 0.40 * dx / max_advection
    reaction_dt = 0.20 / reaction_scale
    stable_dt = min(diffusion_dt, advection_dt, reaction_dt)
    return max(config.base_num_steps, int(math.ceil(config.total_time / max(stable_dt * 0.9, 1e-6))))


def solve_case(grid_size: int, config: CoupledADRConfig, params: dict[str, float]) -> dict[str, np.ndarray]:
    x = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    dx = float(x[1] - x[0])
    num_steps = stable_num_steps(config, params, dx)
    dt = config.total_time / float(num_steps)

    source_u, source_v = source_profile(x, params)
    u, v = initial_state(x, params)

    for _ in range(num_steps):
        lap_u = np.zeros_like(u)
        lap_v = np.zeros_like(v)
        adv_u = np.zeros_like(u)
        adv_v = np.zeros_like(v)

        lap_u[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
        lap_v[1:-1] = (v[2:] - 2.0 * v[1:-1] + v[:-2]) / (dx * dx)
        adv_u[1:-1] = (u[2:] - u[:-2]) / (2.0 * dx)
        adv_v[1:-1] = (v[2:] - v[:-2]) / (2.0 * dx)

        du = (
            params["diffusion_u"] * lap_u
            - params["advection_u"] * adv_u
            + params["reaction_u"] * u
            + params["coupling"] * (v - u)
            + source_u
        )
        dv = (
            params["diffusion_v"] * lap_v
            - params["advection_v"] * adv_v
            + params["reaction_v"] * v
            + params["coupling"] * (u - v)
            + source_v
        )
        u = u + dt * du
        v = v + dt * dv
        u[[0, -1]] = 0.0
        v[[0, -1]] = 0.0

    return {
        "x": x.astype(np.float32),
        "u": u.astype(np.float32),
        "v": v.astype(np.float32),
        "source_u": source_u.astype(np.float32),
        "source_v": source_v.astype(np.float32),
    }


def build_sample(config: CoupledADRConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    params = sample_case_parameters(config, rng)
    fine_solution = solve_case(config.fine_size, config, params)
    coarse_solution = solve_case(config.coarse_size, config, params)

    coarse_u_interp = np.interp(fine_solution["x"], coarse_solution["x"], coarse_solution["u"]).astype(np.float32)
    coarse_v_interp = np.interp(fine_solution["x"], coarse_solution["x"], coarse_solution["v"]).astype(np.float32)
    coupling_channel = np.full_like(fine_solution["x"], params["coupling"], dtype=np.float32)

    inputs = np.stack(
        [
            coarse_u_interp,
            coarse_v_interp,
            fine_solution["source_u"],
            fine_solution["source_v"],
            fine_solution["x"],
            coupling_channel,
        ],
        axis=-1,
    ).astype(np.float32)
    targets = np.stack([fine_solution["u"], fine_solution["v"]], axis=-1).astype(np.float32)
    coarse_interp = np.stack([coarse_u_interp, coarse_v_interp], axis=-1).astype(np.float32)
    residual_targets = targets - coarse_interp
    coefficients = np.array(
        [
            params["coupling"],
            params["diffusion_u"],
            params["diffusion_v"],
            params["advection_u"],
            params["advection_v"],
            params["reaction_u"],
            params["reaction_v"],
        ],
        dtype=np.float32,
    )

    return {
        "inputs": inputs,
        "targets": targets,
        "coarse_interp": coarse_interp,
        "residual_targets": residual_targets.astype(np.float32),
        "coefficients": coefficients,
    }


def write_split(
    output_dir: Path,
    split_name: str,
    num_samples: int,
    config: CoupledADRConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    coarse_interp: list[np.ndarray] = []
    residual_targets: list[np.ndarray] = []
    coefficients: list[np.ndarray] = []

    for _ in range(num_samples):
        sample = build_sample(config, rng)
        inputs.append(sample["inputs"])
        targets.append(sample["targets"])
        coarse_interp.append(sample["coarse_interp"])
        residual_targets.append(sample["residual_targets"])
        coefficients.append(sample["coefficients"])

    output_path = output_dir / f"{split_name}.npz"
    np.savez(
        output_path,
        inputs=np.stack(inputs, axis=0).astype(np.float32),
        targets=np.stack(targets, axis=0).astype(np.float32),
        coarse_interp=np.stack(coarse_interp, axis=0).astype(np.float32),
        residual_targets=np.stack(residual_targets, axis=0).astype(np.float32),
        coefficients=np.stack(coefficients, axis=0).astype(np.float32),
    )
    return {"path": output_path.name, "num_samples": num_samples}


def write_manifest(output_dir: Path, config: CoupledADRConfig, split_records: dict[str, list[dict[str, Any]]]) -> Path:
    manifest = {
        "dataset_name": config.dataset_name,
        "dataset_version": config.dataset_version,
        "task_type": "coarse-to-fine operator-learning",
        "format": "npz",
        "pde_family": "coupled-adr-1d",
        "coarse_grid_size": config.coarse_size,
        "grid_size": config.fine_size,
        "input_channels": [
            "coarse_u_interp",
            "coarse_v_interp",
            "source_u",
            "source_v",
            "x",
            "coupling",
        ],
        "target_channels": ["fine_u", "fine_v"],
        "target_type": "fine_state",
        "residual_target_type": "fine_minus_interpolated_coarse",
        "config": asdict(config),
        "splits": split_records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def generate_dataset(config: CoupledADRConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)

    split_records = {
        "train": [write_split(output_dir, "train", config.train_samples, config, rng)],
        "val": [write_split(output_dir, "val", config.val_samples, config, rng)],
        "test": [write_split(output_dir, "test", config.test_samples, config, rng)],
    }
    manifest_path = write_manifest(output_dir, config, split_records)
    return {
        "output_dir": str(output_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "split_records": split_records,
    }
