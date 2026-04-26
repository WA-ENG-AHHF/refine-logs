# DELEGATION BOARD

## Purpose

Track exactly which agent owns which slice of work.

## Rules

- one task has one owner
- one file group has one active editor at a time
- downstream work cannot start until upstream handoff is logged
- if scope changes, update `DECISIONS.md` first

## Current delegation

| Agent | Scope | Files/Area | Status | Depends on | Deliverable |
|---|---|---|---|---|---|
| A0 Supervisor | orchestration | `coordination/` | active | none | board, decisions, handoffs |
| A1 Data & Sanity | S0 | dataset contract, sanity scripts | merged | none | S0 scaffold merged |
| A2 Baseline | S1 | baseline configs and script | merged | A1 handoff for full run | S1 scaffold merged |
| A3 PCR Core | S2 | PCR config, model, training | merged | A2 handoff for full run | S2 scaffold merged |
| A4 Ablation | S3 | ablation configs and runner | blocked | A3 | S3 matrix |
| A5 Robustness | S4 | robustness runner and reports | blocked | A3 | S4 analysis |
| A6 Metrics & Visualization | shared eval | plotting and merged metrics | merged | partial | metrics scaffold merged |
| A7 Writing | claim packaging | reports and claim tables | pending | partial | claim status |

## Ready queue

1. User worktree starts wave-2 integration on top of merged scaffolds
2. Next implementation priority: connect real dataset paths and replace placeholder loops
3. A7 can now start claim/evidence alignment from merged outputs

## Active wave-1 worktrees

| Worktree | Branch | Agent | Write scope |
|---|---|---|---|
| `D:\Download\refine-logs-wt-a1-sanity` | `feat/a1-s0-sanity` | A1 | S0 script and dataset contract |
| `D:\Download\refine-logs-wt-a2-baseline` | `feat/a2-s1-baseline` | A2 | baseline trainer and config |
| `D:\Download\refine-logs-wt-a3-pcr` | `feat/a3-s2-pcr-core` | A3 | PCR trainer and model modules |
| `D:\Download\refine-logs-wt-a6-metrics` | `feat/a6-metrics-vis` | A6 | metrics aggregation and contract |
