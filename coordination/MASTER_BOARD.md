# MASTER BOARD

## Project
- Topic: PCR-NO for coupled PDE coarse-to-fine correction
- Current phase: pre-execution
- Global owner: A0 Supervisor
- Last updated: 2026-04-26

## Stage Status

| Stage | Owner | Status | Input Ready | Output Ready | Notes |
|---|---|---|---|---|---|
| S0 Sanity | A1 Data & Sanity | pending | yes | no | freeze dataset contract first |
| S1 Baseline | A2 Baseline | blocked | no | no | waits for S0 pass |
| S2 Main | A3 PCR Core | blocked | no | no | waits for S1 checkpoint |
| S3 Ablation | A4 Ablation | blocked | no | no | waits for S2 freeze |
| S4 Robustness | A5 Robustness | blocked | no | no | waits for S2 freeze |
| Metrics/Plots | A6 Metrics & Visualization | pending | partial | no | can scaffold early |
| Writing | A7 Writing | pending | yes | no | can start claim table early |

## Active Decisions
- Proposal scope frozen to PCR-NO + eq-wise decomposition.
- First benchmark remains 2D coupled PDE.
- Evaluation metrics remain rel-L2, PDE residual, BC violation, conservation error.

## Immediate Next Actions
1. A1 produces `artifacts/DATASET_CONTRACT.md`.
2. A1 completes S0 sanity report.
3. A0 unlocks S1 only after S0 success criteria pass.
