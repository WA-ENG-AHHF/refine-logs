# refine-logs

Research control tower and execution scaffold for the PCR-NO project:
Physics-Consistent Residual Neural Operator for coupled PDE coarse-to-fine correction.

## Project status

- Research direction: frozen
- Experiment roadmap: defined
- Multi-agent workflow: scaffolded
- Execution code: initialized
- GitHub repo: connected and live

## Repository goals

This repository separates the project into two layers:

1. Research control
   - proposal, experiment claims, stage tracking, decisions, and handoffs
2. Execution scaffold
   - configs, scripts, model placeholders, reports, and result directories

This structure is designed for supervised multi-agent delivery, where planning, implementation, evaluation, and writing can proceed with clear ownership.

## Start here

Read these files in order:

1. `FINAL_PROPOSAL.md`
2. `EXPERIMENT_PLAN.md`
3. `EXPERIMENT_TRACKER_DETAILED_20260425_141645.md`
4. `MULTI_AGENT_EXECUTION_BLUEPRINT.md`
5. `agent-prompts/README.md`
6. `execution-repo/README.md`

## Repository map

### Research control
- `MULTI_AGENT_EXECUTION_BLUEPRINT.md`
- `FINAL_PROPOSAL.md`
- `EXPERIMENT_PLAN.md`
- `EXPERIMENT_TRACKER*.md`
- `PIPELINE_SUMMARY*.md`

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

Build the first real execution wave:
- finalize dataset contract
- implement S0 sanity checks
- reproduce S1 baseline
- freeze the first valid S2 checkpoint

## Notes

- This repository is intentionally stage-driven rather than framework-heavy.
- The current execution code is scaffold-level and ready for multi-agent implementation.
- Large binary outputs should stay out of Git and be tracked through reports and registries.
