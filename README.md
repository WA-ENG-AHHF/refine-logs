# refine-logs

Research control tower and execution scaffold for two closely related threads:
- a legacy toy scaffold for coupled 1D ADR coarse-to-fine correction
- the current main direction: high-contrast 2D Darcy pressure super-resolution with flux-repaired residual reconstruction

## Project status

- Research direction: re-anchored around 2D Darcy
- Legacy toy path: retained for sanity checks only
- Experiment roadmap: defined for both legacy and Darcy tracks
- Multi-agent workflow: scaffolded
- Execution code: initialized and being specialized

## Repository goals

This repository separates the project into two layers:

1. Research control
   - proposal, experiment claims, stage tracking, decisions, and handoffs
2. Execution scaffold
   - configs, scripts, model placeholders, reports, and result directories

This structure is designed for supervised multi-agent delivery, where planning, implementation, evaluation, and writing can proceed with clear ownership.

## Start here

Read these files in order:

1. `FINAL_PROPOSAL_DARCY.md`
2. `EXPERIMENT_PLAN_DARCY.md`
3. `EXPERIMENT_TRACKER_DARCY.md`
4. `MULTI_AGENT_EXECUTION_BLUEPRINT.md`
5. `agent-prompts/README.md`
6. `execution-repo/README.md`

Legacy proposal files remain in the repo for historical reference, but they no longer define the main paper direction.

## Repository map

### Research control
- `MULTI_AGENT_EXECUTION_BLUEPRINT.md`
- `FINAL_PROPOSAL_DARCY.md`
- `EXPERIMENT_PLAN_DARCY.md`
- `EXPERIMENT_TRACKER_DARCY.md`
- legacy proposal and tracker files for the coupled-ADR scaffold

### Coordination
- `coordination/MASTER_BOARD.md`
- `coordination/DECISIONS.md`
- `coordination/AGENT_HANDOFFS.md`
- `coordination/RUN_REGISTRY.csv`
- `coordination/DELEGATION_BOARD.md`
- `coordination/WORKTREE_PLAN.md`

### Agent launch assets
- `agent-prompts/`

### Execution-side scaffold
- `execution-repo/configs/`
- `execution-repo/scripts/`
- `execution-repo/models/`
- `execution-repo/reports/`
- `execution-repo/results/`
- `execution-repo/checkpoints/`

## Working model

The recommended workflow is:

1. A0 Supervisor controls stage gates.
2. A1 finishes dataset contract and S0 sanity.
3. A2 reproduces the S1 baseline.
4. A3 implements and tunes PCR-NO for S2.
5. A4 and A5 run in parallel only after S2 is frozen.
6. A6 aggregates metrics and figures.
7. A7 keeps claims aligned with evidence.

## Quick start

### Coordination-first
- update `coordination/MASTER_BOARD.md`
- assign owners in `coordination/DELEGATION_BOARD.md`
- create or activate worktrees using `coordination/WORKTREE_PLAN.md`

### Execution-first
```bash
cd execution-repo
python scripts/run_sanity_check.py
python scripts/train_baseline.py
python scripts/train_pcr_no.py
```

## Current next milestone

Specialize the scaffold to the Darcy main line:
- finalize the 2D Darcy dataset contract
- implement Darcy-specific S0 sanity checks
- reproduce direct and residual Darcy baselines
- freeze the first valid flux-repaired residual checkpoint

## Notes

- The repository is intentionally stage-driven rather than framework-heavy.
- The old 1D coupled ADR assets are kept as a toy benchmark and debugging path.
- The new main line should prefer Darcy-specific configs and scripts when both exist.
- Large binary outputs should stay out of Git and be tracked through reports and registries.
