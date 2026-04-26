# PCR-NO Execution Repo Scaffold

This scaffold is the execution-side companion to the research control docs in `D:\Download\refine-logs`.

## Purpose
- hold runnable code, configs, logs, checkpoints, and results
- separate execution artifacts from proposal documents

## Suggested flow
1. fill dataset contract in `D:\Download\refine-logs\artifacts\DATASET_CONTRACT.md`
2. implement S0 in `scripts/run_sanity_check.py`
3. reproduce baseline with `scripts/train_baseline.py`
4. implement main method with `scripts/train_pcr_no.py`
5. run S3 and S4 after freezing `checkpoints/S2_best.pt`

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
