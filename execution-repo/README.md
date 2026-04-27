# Execution Repo Scaffold

This directory is the execution-side companion to the research control docs in this workspace.

It now contains two tracks:
- **Main track**: 2D Darcy pressure super-resolution with flux-repaired residual reconstruction
- **Legacy track**: 1D coupled ADR scaffold retained for sanity checks and debugging

## Purpose

- hold runnable code, configs, logs, checkpoints, and results
- isolate implementation work from proposal and planning documents
- support multi-agent execution with clean ownership boundaries
- preserve backward compatibility with the original toy scaffold while moving the main story to Darcy

## Stage mapping

| Stage | Goal | Primary script | Primary output |
|---|---|---|---|
| S0 | Darcy sanity and dataset validation | `scripts/run_sanity_check.py` or `scripts/run_darcy_pipeline.py` | `logs/S0_*_sanity_report.json` |
| S1 | direct and residual baseline reproduction | `scripts/train_darcy_baseline.py` | `outputs/S1_darcy_baseline/` |
| S2 | flux-repaired residual main model | `scripts/train_darcy_pcr.py` | `outputs/S2_darcy_main/` |
| S3 | novelty isolation and ablation evidence | legacy placeholder today | `results/S3_darcy_ablation_matrix.csv` |
| S4 | high-contrast and BC robustness evaluation | legacy placeholder today | `results/S4_darcy_ood_analysis.csv` |

## Suggested implementation order

1. fill the Darcy dataset contract
2. implement the Darcy dataset generator and sanity path
3. build direct and residual Darcy baselines with a true 2D backbone
4. add the differentiable local flux repair module
5. add novelty-isolation and robustness runners
6. add Darcy-specific metric aggregation and plotting

## Directory map

- `configs/`: stage configs
- `scripts/`: training and evaluation entrypoints
- `models/`: model definitions
- `metrics/`: Darcy-specific residual and conservation metrics (planned)
- `reports/`: execution-side stage reports
- `results/`: csv outputs
- `logs/`: raw logs
- `checkpoints/`: model snapshots
- `figures/`: generated plots
- `data/`: local data pointers or prepared subsets

## Multi-agent ownership suggestion

- Darcy data path: dataset generator, manifest format, sanity scripts
- baseline path: `scripts/train_darcy_baseline.py`, baseline configs, baseline operator code
- PCR path: `scripts/train_darcy_pcr.py`, repair module, main configs
- eval path: Darcy ablations, robustness runs, plotting

Prefer separate modules over shared edits when two agents are active at the same time.

## Minimal setup

```bash
pip install -r requirements.txt
python scripts/prepare_sample_dataset.py
python scripts/run_sanity_check.py
python scripts/run_wave1_pipeline.py --run-prefix legacy_smoke
```

## Expected evolution

This scaffold should gradually grow into:
- reusable dataset module
- baseline 2D operator implementation
- flux repair layer
- Darcy residual and conservation metrics
- shared metrics and plotting utilities

## Darcy specialization status

- `prepare_sample_dataset.py` and `prepare_coupled_pde_dataset.py` remain as legacy toy data paths
- Darcy-specific config files are now added under `configs/`
- Darcy wrapper scripts are now added under `scripts/`
- The next implementation step is to replace generic placeholders with a true 2D Darcy path while reusing the same scaffold contracts

## Smoke pipeline

Use the integration runner to exercise the current scaffold end to end:

```bash
python scripts/run_wave1_pipeline.py --run-prefix smoke
```

This currently runs:
- `S0` sanity validation
- `S1` baseline scaffold initialization
- `S2` PCR-NO dry-run validation
- metrics aggregation into `results/MASTER_METRICS.csv`

## Legacy sample dataset path

The repo now includes a tiny local-data workflow:

```bash
python scripts/prepare_sample_dataset.py
```

This writes a small manifest-based dataset under `data/sample_adr/`, which is ignored by Git but usable by `S0` and `S1` for local validation.

## Legacy coupled ADR pipeline

Generate the first research-oriented coupled ADR dataset with:

```bash
python scripts/prepare_coupled_pde_dataset.py --config configs/DATASET_coupled_adr.yaml
```

Then validate and run the scaffold pipeline against the generated data:

```bash
python scripts/run_sanity_check.py --config configs/S0_coupled_adr.yaml
python scripts/run_wave1_pipeline.py ^
  --run-prefix coupled_adr ^
  --s0-config configs/S0_coupled_adr.yaml ^
  --s1-config configs/S1_baseline_coupled_adr.yaml ^
  --s2-config configs/S2_main_coupled_adr.yaml
```

The generated formal dataset uses:
- coarse grid size `16`
- fine grid size `64`
- two coupled fields `u` and `v`
- six input channels: interpolated coarse fields, two source channels, coordinate, coupling
- two target channels: fine `u` and fine `v`
- residual targets saved alongside direct fine-state targets

## Darcy target pipeline

The intended main-track file mapping is:

```text
configs/DATASET_darcy2d.yaml
configs/S0_darcy.yaml
configs/S1_darcy_baseline.yaml
configs/S2_darcy_main.yaml
configs/S3_darcy_ablation.yaml
configs/S4_darcy_robustness.yaml

scripts/prepare_darcy_dataset.py
scripts/train_darcy_baseline.py
scripts/train_darcy_pcr.py
scripts/run_darcy_pipeline.py
```

These wrappers preserve the original stage structure while specializing the repo toward the current paper plan.
