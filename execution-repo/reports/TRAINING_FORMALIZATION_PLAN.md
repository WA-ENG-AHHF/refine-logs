# Training Formalization Plan

## Current State

- `S1` can run end to end, but its default portable path is still a closed-form `numpy-linear` fit.
- `S2` can run end to end, but its current non-torch path is still a lightweight residual prototype, and its torch path is not yet wired to real manifest-backed batches.
- The repo already has stable external interfaces:
  - manifest-based dataset loading
  - stage configs
  - run summaries
  - stage metrics CSV outputs

## Goal

Turn the current runnable scaffold into a research-capable training stack without breaking the existing CLI and artifact contracts.

## Freeze First

Keep these interfaces stable while implementation evolves:

- dataset contract: `manifest.json + train/val/test.npz`
- stage entrypoints:
  - `scripts/train_baseline.py`
  - `scripts/train_pcr_no.py`
- summary artifact: `reports/run_summary.json`
- metrics artifact: stage CSVs plus `results/MASTER_METRICS.csv`
- canonical comparison metrics:
  - `rel_l2`
  - `pde_residual`
  - `bc_violation`
  - `conservation_error`
  - `total`

## Execution Order

1. Formalize shared training infrastructure.
2. Upgrade `S1` to a real epoch-based supervised trainer.
3. Reuse the same infrastructure in `S2` for real manifest-backed residual training.
4. Replace placeholder physics losses with PDE-aware operators.
5. Run formal `S1 vs S2` comparisons, then ablations.

## Shared Infrastructure Slice

The first reusable slice should own:

- seed control
- mini-batch iteration
- train and validation epoch loops
- best-checkpoint selection
- portable metric computation
- serializable history logging

This is now partially landed in `training/supervised.py`, and `S1` should be the first consumer.

## Baseline Upgrade Target

`S1` should become:

- input: `inputs`
- target: `targets`
- training: real epoch and batch loop
- optimizer: Adam
- checkpoint rule: best validation `total`
- fallback: keep `numpy-linear` for torch-free environments

## PCR Upgrade Target

`S2` should become:

- input: manifest-backed coarse-to-fine features
- target: `residual_targets`
- reconstruction: `coarse_interp + predicted_residual`
- training: same epoch and batch loop pattern as `S1`
- losses:
  - direct residual or reconstructed state error
  - PDE residual term
  - boundary consistency term
  - conservation term

## Validation Checklist

Each step is only considered done when:

- the script still runs in the current torch-free environment
- the script runs with a real torch path when torch is installed
- summary JSON records backend, epoch count, and train/val metrics
- stage CSV rows are readable and append cleanly
- the wave-1 pipeline still completes
