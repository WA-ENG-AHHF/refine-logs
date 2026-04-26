# MASTER BOARD

## Project
- Topic: PCR-NO for coupled PDE coarse-to-fine correction
- Current phase: wave-1 merged / wave-2 preparation
- Global owner: A0 Supervisor
- Last updated: 2026-04-26

## Stage Status

| Stage | Owner | Status | Input Ready | Output Ready | Notes |
|---|---|---|---|---|---|
| S0 Sanity | A1 Data & Sanity | merged | yes | yes | scaffold merged from `feat/a1-s0-sanity` |
| S1 Baseline | A2 Baseline | merged | yes | yes | scaffold merged from `feat/a2-s1-baseline` |
| S2 Main | A3 PCR Core | merged | yes | yes | scaffold merged from `feat/a3-s2-pcr-core` |
| S3 Ablation | A4 Ablation | blocked | no | no | waits for S2 freeze |
| S4 Robustness | A5 Robustness | blocked | no | no | waits for S2 freeze |
| Metrics/Plots | A6 Metrics & Visualization | merged | yes | yes | scaffold merged from `feat/a6-metrics-vis` |
| Writing | A7 Writing | pending | yes | no | can start claim table early |

## Active Decisions
- Proposal scope frozen to PCR-NO + eq-wise decomposition.
- First benchmark remains 2D coupled PDE.
- Evaluation metrics remain rel-L2, PDE residual, BC violation, conservation error.

## Immediate Next Actions
1. Review merged wave-1 scaffolds and run smoke checks on main.
2. Create user-owned worktree for wave-2 integration.
3. Decide whether wave-2 prioritizes real data plumbing, baseline training loop, or metric/report integration.
4. Unlock A4/A5 only after S2 moves beyond scaffold into a real checkpoint path.
