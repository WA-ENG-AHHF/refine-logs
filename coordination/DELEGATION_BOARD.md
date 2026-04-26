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
| A1 Data & Sanity | S0 | dataset contract, sanity scripts | pending | none | S0 report |
| A2 Baseline | S1 | baseline configs and script | blocked | A1 | S1 checkpoint |
| A3 PCR Core | S2 | PCR config, model, training | blocked | A2 | S2 checkpoint |
| A4 Ablation | S3 | ablation configs and runner | blocked | A3 | S3 matrix |
| A5 Robustness | S4 | robustness runner and reports | blocked | A3 | S4 analysis |
| A6 Metrics & Visualization | shared eval | plotting and merged metrics | pending | partial | figures and master metrics |
| A7 Writing | claim packaging | reports and claim tables | pending | partial | claim status |

## Ready queue

1. A1 fills `artifacts/DATASET_CONTRACT.md`
2. A1 implements `execution-repo/scripts/run_sanity_check.py`
3. A2 scaffolds baseline data path without touching PCR files
4. A6 drafts shared metric schema
