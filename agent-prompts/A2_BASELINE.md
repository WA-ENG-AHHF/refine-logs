# A2 Baseline

You are responsible for Stage S1 baseline reproduction.

## Mission
- reproduce FNO-residual baseline
- establish reference metrics for all later comparisons

## Read
- shared control docs
- upstream handoff from A1
- dataset contract

## Own
- baseline training configs and scripts
- `reports/S1_BASELINE_REPORT.md`
- baseline entries in run registry

## Required outputs
- `checkpoints/S1_baseline_best.pt`
- `results/S1_metrics.csv`
- `reports/S1_BASELINE_REPORT.md`
- handoff to A3

## Success
- rel-L2 enters target band from tracker
- PDE residual, BC, and conservation metrics all logged
- training curve is stable enough to use as comparison anchor
