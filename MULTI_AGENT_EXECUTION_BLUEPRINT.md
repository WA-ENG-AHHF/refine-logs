# Multi-Agent Execution Blueprint for PCR-NO

## 1. Current Project Read

This directory is already strong on **research planning** but weak on **execution coordination**.

Existing artifacts:
- `FINAL_PROPOSAL.md`: method thesis and contribution boundary
- `EXPERIMENT_PLAN.md`: claim-driven roadmap
- `EXPERIMENT_TRACKER.md`: lightweight stage tracker
- `EXPERIMENT_TRACKER_DETAILED_20260425_141645.md`: executable run sheet
- `PIPELINE_SUMMARY*.md`: pipeline history and launch guidance
- `.no/MANIFEST.md`: artifact provenance

What is missing for multi-agent collaboration:
- a single command board for current ownership and status
- explicit handoff rules between planning, coding, training, and analysis
- shared output contract for every stage
- a parallelization plan that prevents agents from duplicating work

This means the right design is not "many equal agents talking freely", but a **supervisor-led research execution swarm**.

## 2. Recommended Agent Topology

Use 1 supervisor + 5 execution agents + 1 optional writing agent.

### A0. Supervisor / PM Agent
- Purpose: global planner and traffic controller
- Reads: `FINAL_PROPOSAL.md`, `EXPERIMENT_PLAN.md`, latest tracker, decision log
- Owns:
  - stage gating
  - task assignment
  - dependency checks
  - merge decisions
  - stop/go calls when metrics fail
- Writes:
  - `coordination/MASTER_BOARD.md`
  - `coordination/DECISIONS.md`
  - `coordination/AGENT_HANDOFFS.md`

### A1. Data & Sanity Agent
- Purpose: execute S0 and freeze dataset contract
- Owns:
  - dataset preparation
  - interpolation checks
  - boundary encoding validation
  - shape and NaN checks
- Inputs:
  - problem definition from `FINAL_PROPOSAL.md`
  - S0 targets from detailed tracker
- Outputs:
  - `reports/S0_SANITY_REPORT.md`
  - `logs/S0_sanity_report.json`
  - `artifacts/DATASET_CONTRACT.md`

### A2. Baseline Agent
- Purpose: reproduce S1 FNO-residual baseline
- Owns:
  - baseline config
  - baseline training loop
  - metric logging for L2, PDE residual, BC violation, conservation
- Inputs:
  - dataset contract from A1
  - S1 success criteria
- Outputs:
  - `reports/S1_BASELINE_REPORT.md`
  - `checkpoints/S1_baseline_best.pt`
  - `results/S1_metrics.csv`

### A3. PCR Core Agent
- Purpose: implement and tune S2 main method
- Owns:
  - residual projection or constraint layer
  - equation-wise residual heads
  - training integration with baseline backbone
- Inputs:
  - baseline reference from A2
  - method constraints from `FINAL_PROPOSAL.md`
- Outputs:
  - `reports/S2_MAIN_REPORT.md`
  - `checkpoints/S2_best.pt`
  - `results/S2_metric_comparison.csv`

### A4. Ablation Agent
- Purpose: isolate novelty for S3
- Owns:
  - eq-wise removal runs
  - physics-loss-only control
  - same-parameter control
- Inputs:
  - frozen best config from A3
  - ablation targets from detailed tracker
- Outputs:
  - `reports/S3_ABLATION_REPORT.md`
  - `results/S3_ablation_matrix.csv`

### A5. Robustness Agent
- Purpose: test OOD coupling and rollout stability for S4
- Owns:
  - strong-coupling evaluation
  - long-horizon rollout evaluation
  - divergence detection
- Inputs:
  - best checkpoint from A3
  - robustness scenarios from detailed tracker
- Outputs:
  - `reports/S4_ROBUSTNESS_REPORT.md`
  - `results/S4_ood_analysis.csv`
  - `results/S4_rollout_metrics.csv`

### A6. Metrics & Visualization Agent
- Purpose: unify evaluation and generate comparable plots
- Owns:
  - metric definitions
  - plot generation
  - summary tables across S1-S4
- Inputs:
  - outputs from A2-A5
- Outputs:
  - `results/MASTER_METRICS.csv`
  - `figures/training_curves/`
  - `figures/paper_ready/`

### A7. Writing Agent (optional but high value)
- Purpose: keep paper-facing narrative aligned with experiments
- Owns:
  - claim-evidence map
  - result interpretation
  - reviewer-risk notes
- Inputs:
  - all stage reports
  - `DECISIONS.md`
- Outputs:
  - `reports/CLAIM_STATUS.md`
  - `reports/PAPER_NOTES.md`

## 3. Serial vs Parallel Work

### Must be serial
1. Freeze proposal scope
2. Complete S0 sanity
3. Reproduce S1 baseline
4. Train S2 main model
5. Freeze best S2 checkpoint

These are hard dependencies. Do not parallelize them away.

### Can run in parallel
- A3 can scaffold PCR modules while A2 is still finishing baseline training
- A6 can prepare metric scripts during S1 and S2
- A7 can draft claim tables from day one
- After S2 checkpoint is frozen:
  - A4 ablation
  - A5 robustness
  - A6 figure generation
  can all run in parallel

### Best execution rhythm
- Wave 1: A1
- Wave 2: A2 + A6 + A7
- Wave 3: A3 + A6
- Wave 4: A4 + A5 + A6 + A7
- Wave 5: A0 synthesizes go/no-go decision for paper packaging

## 4. Ownership Rules

Every file should have one owning agent at a time.

Suggested ownership split:
- `configs/`: A1-A5 by stage, one config file per run group
- `scripts/data_*`: A1
- `scripts/train_baseline*`: A2
- `scripts/train_pcr_no*`: A3
- `scripts/train_ablation*`: A4
- `scripts/eval_robustness*`: A5
- `scripts/metrics*`, `scripts/plot*`: A6
- `reports/`, `figures/`: A6 and A7
- `coordination/`: A0 only

Rule: if two agents need the same file, split by new file creation instead of shared editing whenever possible.

## 5. Handoff Contract

Each handoff should include exactly five items:

1. `Task completed`
2. `Artifacts produced`
3. `Metrics observed`
4. `Open risks`
5. `Recommended next agent`

Recommended template:

```md
## Handoff: A2 -> A3
- Task completed: S1 baseline reproduction
- Artifacts produced:
  - checkpoints/S1_baseline_best.pt
  - results/S1_metrics.csv
  - reports/S1_BASELINE_REPORT.md
- Metrics observed:
  - rel-L2: 1.42%
  - PDE residual: ...
  - BC violation: ...
  - conservation error: ...
- Open risks:
  - BC metric unstable on mixed boundary subset
- Recommended next agent:
  - A3 PCR Core Agent
```

## 6. Minimal Coordination Files to Add

Add these before real execution starts:

- `coordination/MASTER_BOARD.md`
  - one-line status for each stage and agent
- `coordination/DECISIONS.md`
  - frozen choices, parameter changes, failed ideas
- `coordination/AGENT_HANDOFFS.md`
  - append-only stage handoffs
- `coordination/RUN_REGISTRY.csv`
  - run id, owner, config, seed, checkpoint, result path
- `artifacts/DATASET_CONTRACT.md`
  - grid size, variables, BC encoding, train/test split

Without these files, multi-agent work will drift into duplicated experiments and unclear provenance.

## 7. Suggested Directory Upgrade

Current directory is document-heavy. Upgrade it to a project-control layout:

```text
refine-logs/
  coordination/
    MASTER_BOARD.md
    DECISIONS.md
    AGENT_HANDOFFS.md
    RUN_REGISTRY.csv
  artifacts/
    DATASET_CONTRACT.md
  reports/
    S0_SANITY_REPORT.md
    S1_BASELINE_REPORT.md
    S2_MAIN_REPORT.md
    S3_ABLATION_REPORT.md
    S4_ROBUSTNESS_REPORT.md
    CLAIM_STATUS.md
  results/
  figures/
  checkpoints/
  logs/
```

If a real code repository exists elsewhere, keep this folder as the **research control tower**, and let the code repo hold `scripts/`, `models/`, and `configs/`.

## 8. Practical Operating Protocol

### Daily loop
1. A0 updates `MASTER_BOARD.md`
2. Each active agent reads only:
   - the latest board
   - its own upstream handoff
   - the files it owns
3. Each agent writes results to stage-specific locations
4. A0 checks success criteria against `EXPERIMENT_TRACKER_DETAILED_20260425_141645.md`
5. A0 either:
   - advances the stage
   - requests rerun
   - logs a design change in `DECISIONS.md`

### Stop conditions
- S0 fails shape, NaN, BC mapping, or checkpoint reload
- S1 baseline cannot reproduce target band
- S2 improves L2 but worsens BC or conservation badly
- S3 cannot isolate contribution from parameter scaling
- S4 shows OOD or rollout collapse

Stop early, log the failure, and replan. Do not let downstream agents continue on a broken upstream artifact.

## 9. Best Prompting Pattern for the Agents

Each agent prompt should contain:
- scope boundary
- allowed files
- required outputs
- success criteria
- what not to touch

Example:

```text
You are A4 Ablation Agent.
Own only S3 artifacts.
Read:
- FINAL_PROPOSAL.md
- EXPERIMENT_TRACKER_DETAILED_20260425_141645.md
- coordination/AGENT_HANDOFFS.md
Do:
- run S3.1-S3.5 analysis
- write reports/S3_ABLATION_REPORT.md and results/S3_ablation_matrix.csv
Do not:
- modify S2 checkpoint
- redefine evaluation metrics
Success:
- isolate eq-wise and subspace contributions against controls
```

## 10. Recommended First Real Rollout

For this project, start with a lean 4-agent rollout first:

1. A0 Supervisor
2. A1 Data & Sanity
3. A2 Baseline
4. A3 PCR Core

Then expand to A4-A7 only after S2 produces a credible checkpoint.

Reason:
- current workspace contains planning docs, not a mature execution repo
- too many agents too early will create paper plans instead of verified results
- the highest-value milestone is a trustworthy S2 checkpoint

## 11. Bottom-Line Recommendation

For PCR-NO, the best multi-agent design is:
- **centralized orchestration**
- **stage-based ownership**
- **artifact-driven handoff**
- **parallelism only after checkpoint freeze**

If you want, this blueprint can be turned next into:
- a concrete `coordination/` folder with templates
- a per-agent prompt pack
- a runnable supervisor checklist for Codex sub-agents
