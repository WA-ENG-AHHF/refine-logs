# Agent Prompt Pack

This folder contains ready-to-use prompts for the PCR-NO multi-agent workflow.

## Recommended launch order
1. `A0_SUPERVISOR.md`
2. `A1_DATA_SANITY.md`
3. `A2_BASELINE.md`
4. `A3_PCR_CORE.md`
5. `A4_ABLATION.md`
6. `A5_ROBUSTNESS.md`
7. `A6_METRICS_VIS.md`
8. `A7_WRITING.md`

## Shared inputs
- `D:\Download\refine-logs\FINAL_PROPOSAL.md`
- `D:\Download\refine-logs\EXPERIMENT_PLAN.md`
- `D:\Download\refine-logs\EXPERIMENT_TRACKER_DETAILED_20260425_141645.md`
- `D:\Download\refine-logs\coordination\MASTER_BOARD.md`
- `D:\Download\refine-logs\coordination\DECISIONS.md`
- `D:\Download\refine-logs\coordination\AGENT_HANDOFFS.md`

## Rule
Each agent reads only its upstream handoff, its owned files, and the shared control docs.
