# MASTER BOARD

## Project
- Topic: PCR-NO for coupled PDE coarse-to-fine correction
- Current phase: wave-1 execution
- Global owner: A0 Supervisor
- Last updated: 2026-04-26

## Stage Status

| Stage | Owner | Status | Input Ready | Output Ready | Notes |
|---|---|---|---|---|---|
| S0 Sanity | A1 Data & Sanity | active | yes | no | worktree `refine-logs-wt-a1-sanity` |
| S1 Baseline | A2 Baseline | active | partial | no | worktree `refine-logs-wt-a2-baseline`; may scaffold before S0 handoff |
| S2 Main | A3 PCR Core | active | partial | no | worktree `refine-logs-wt-a3-pcr`; module scaffolding only |
| S3 Ablation | A4 Ablation | blocked | no | no | waits for S2 freeze |
| S4 Robustness | A5 Robustness | blocked | no | no | waits for S2 freeze |
| Metrics/Plots | A6 Metrics & Visualization | active | partial | no | worktree `refine-logs-wt-a6-metrics` |
| Writing | A7 Writing | pending | yes | no | can start claim table early |

## Active Decisions
- Proposal scope frozen to PCR-NO + eq-wise decomposition.
- First benchmark remains 2D coupled PDE.
- Evaluation metrics remain rel-L2, PDE residual, BC violation, conservation error.

## Immediate Next Actions
1. A1 produces `artifacts/DATASET_CONTRACT.md`.
2. A1 completes S0 sanity report.
3. A2 prepares baseline runner shell without assuming dataset specifics.
4. A3 prepares PCR module skeleton without touching baseline path.
5. A6 freezes metric field names and aggregation contract.
6. A0 unlocks full S1 execution only after S0 success criteria pass.
