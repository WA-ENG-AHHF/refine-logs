# PCR-NO Execution Repo Scaffold

This directory is the execution-side companion to the research control docs in `D:\Download\refine-logs`.

## Purpose

- hold runnable code, configs, logs, checkpoints, and results
- isolate implementation work from proposal and planning documents
- support multi-agent execution with clean ownership boundaries

## Stage mapping

| Stage | Goal | Primary script | Primary output |
|---|---|---|---|
| S0 | sanity and dataset validation | `scripts/run_sanity_check.py` | `logs/S0_sanity_report.json` |
| S1 | baseline reproduction | `scripts/train_baseline.py` | `checkpoints/S1_baseline_best.pt` |
| S2 | PCR-NO main model | `scripts/train_pcr_no.py` | `checkpoints/S2_best.pt` |
| S3 | ablation evidence | `scripts/train_ablation.py` | `results/S3_ablation_matrix.csv` |
| S4 | robustness evaluation | `scripts/eval_robustness.py` | `results/S4_ood_analysis.csv` |

## Suggested implementation order

1. fill dataset contract in `D:\Download\refine-logs\artifacts\DATASET_CONTRACT.md`
2. implement S0 sanity script and smoke tests
3. build dataset loading and baseline training path
4. add PCR projection and equation-wise heads
5. add ablation and robustness runners
6. add metric aggregation and plotting

## Directory map

- `configs/`: stage configs
- `scripts/`: training and evaluation entrypoints
- `models/`: model definitions
- `reports/`: execution-side stage reports
- `results/`: csv outputs
- `logs/`: raw logs
- `checkpoints/`: model snapshots
- `figures/`: generated plots
- `data/`: local data pointers or prepared subsets

## Multi-agent ownership suggestion

- baseline path: `scripts/train_baseline.py`, baseline configs, baseline model code
- PCR path: `scripts/train_pcr_no.py`, PCR modules, main configs
- eval path: `scripts/train_ablation.py`, `scripts/eval_robustness.py`, plotting

Prefer separate modules over shared edits when two agents are active at the same time.

## Minimal setup

```bash
pip install -r requirements.txt
python scripts/run_sanity_check.py
python scripts/run_wave1_pipeline.py --run-prefix smoke
```

## Expected evolution

This scaffold should gradually grow into:
- reusable dataset module
- baseline FNO implementation
- PCR projection layer
- equation-wise residual heads
- shared metrics and plotting utilities

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
