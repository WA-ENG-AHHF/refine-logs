# Experiment Plan

**Problem**: High-contrast heterogeneous 2D Darcy coarse-to-fine pressure reconstruction  
**Method Thesis**: Predict only the coarse-to-fine pressure residual, then apply a differentiable local flux repair module so conservation is enforced structurally rather than only encouraged by soft losses.  
**Date**: 2026-04-27

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| C1: Residual reconstruction is the right formulation for Darcy super-resolution | Prevents the paper from looking like a generic direct-prediction baseline comparison | Residual baseline beats direct prediction at matched backbone size and budget on the anchor task | B1 |
| C2: Local flux repair is a real mechanism, not just another regularizer | This is the dominant novelty claim and must survive reviewer scrutiny | Flux-repaired model materially lowers local mass-balance error and flux mismatch versus residual-only and residual+soft-physics baselines, without unacceptable Rel-L2 regression | B1, B2 |

## Paper Storyline
- Main paper must prove:
  - Residual coarse-to-fine reconstruction is a stronger formulation than direct fine prediction for this task.
  - Explicit local flux repair improves physical validity beyond soft PDE or flux penalties.
  - The gain matters most in the intended harder regime: high contrast and varying boundary conditions.
- Appendix can support:
  - Additional seeds
  - Extra backbone variants
  - More boundary-condition combinations
  - Sensitivity to repair strength or iteration count
- Experiments intentionally cut:
  - Arbitrary geometry in the first paper version
  - 3D Darcy
  - Navier-Stokes or shallow-water transfer
  - Large baseline lists beyond the strongest three families

## Experiment Blocks

### Block 1: Main Anchor Result
- Claim tested: C1 and the practical half of C2
- Why this block exists:
  - This is the paper's main table.
  - It must show that our final method improves both reconstruction accuracy and physical validity on the actual target problem.
- Dataset / split / task:
  - 2D Darcy on regular grids
  - Coarse grid `32x32`, fine grid `128x128`
  - Training on moderate-to-high contrast coefficients
  - Validation and test on matched-distribution held-out samples
  - Dirichlet boundary first; source term simple but nontrivial
- Compared systems:
  - `CoarseInterp`: interpolated coarse solution only
  - `DirectNO`: direct fine prediction with true 2D FNO or strongest comparable operator baseline
  - `ResidualNO`: predict `delta_p` only, no physics terms
  - `ResidualNO+SoftPhys`: residual model with Darcy residual and boundary penalties
  - `ResidualNO+FluxRepair`: final method
- Metrics:
  - Primary: local mass-balance error, flux mismatch, boundary violation
  - Secondary: Rel-L2 pressure error
- Setup details:
  - Same backbone width and depth across `DirectNO`, `ResidualNO`, `ResidualNO+SoftPhys`, and `ResidualNO+FluxRepair`
  - `DEFAULT_SEEDS = 3`
  - Fixed training budget per model
  - Report mean and std
- Success criterion:
  - Final method is best or tied-best on primary physical metrics.
  - Rel-L2 does not degrade beyond a small acceptable margin; ideally improves or stays flat.
- Failure interpretation:
  - If repair only improves physical metrics while badly hurting Rel-L2, the mechanism is too destructive.
  - If soft-physics already matches repair, the novelty claim weakens substantially.
- Table / figure target:
  - Main paper Table 1
  - One qualitative figure showing coarse, fine, residual prediction, and repaired result
- Priority: MUST-RUN

### Block 2: Novelty Isolation
- Claim tested: C2
- Why this block exists:
  - Reviewers will ask whether the gain comes from residual learning, extra parameters, or just another physics loss.
- Dataset / split / task:
  - Same as Block 1, plus one high-contrast stress subset
- Compared systems:
  - `ResidualNO`
  - `ResidualNO+SoftPDE`
  - `ResidualNO+SoftPDE+SoftFlux`
  - `ResidualNO+BoundaryClamp`
  - `ResidualNO+FluxRepair` (ours)
- Metrics:
  - Primary: local mass-balance error, per-cell conservation histogram, flux mismatch
  - Secondary: Rel-L2
- Setup details:
  - Match parameter count as closely as possible
  - Keep the backbone frozen across variants; only mechanism changes
  - At least 3 seeds on the decisive comparison
- Success criterion:
  - The repaired model has a clear conservation advantage over soft-loss baselines.
  - The gain is visible not only in averages but also in tail behavior or worst-cell statistics.
- Failure interpretation:
  - If differences are only tiny average improvements, reviewers may call this an engineering tweak.
- Table / figure target:
  - Main paper Table 2
  - Appendix histogram or percentile plot for cell-wise conservation defects
- Priority: MUST-RUN

### Block 3: Hard-Regime Robustness
- Claim tested: C2 under intended use conditions
- Why this block exists:
  - Without a hard regime, the paper risks looking incremental.
- Dataset / split / task:
  - Train on moderate contrast
  - Test on stronger contrast
  - Test on varying boundary conditions not fully seen in training
- Compared systems:
  - `DirectNO`
  - `ResidualNO+SoftPhys`
  - `ResidualNO+FluxRepair`
- Metrics:
  - Primary: conservation and flux metrics under OOD contrast / BC shift
  - Secondary: Rel-L2 under OOD
- Setup details:
  - Contrast sweep, for example low, medium, high, extreme
  - Boundary-condition family sweep, e.g. amplitude or profile changes
- Success criterion:
  - Our method degrades more gracefully on conservation-sensitive metrics than soft-physics residual baselines.
- Failure interpretation:
  - If the method wins only in-distribution, the "high-contrast" positioning becomes weak.
- Table / figure target:
  - Main paper robustness figure
  - Appendix full sweep tables
- Priority: MUST-RUN

### Block 4: Simplicity Check
- Claim tested: the method does not need extra complexity beyond repair
- Why this block exists:
  - We need to show the paper is focused and not under-built only because implementation is incomplete.
- Dataset / split / task:
  - Same anchor setting
- Compared systems:
  - `ResidualNO+FluxRepair`
  - `ResidualNO+FluxRepair+ExtraHead`
  - `ResidualNO+FluxRepair+LargerBackbone`
- Metrics:
  - Same primary metrics, plus parameter count and training cost
- Setup details:
  - One overbuilt variant only; do not let this become a benchmark detour
- Success criterion:
  - Overbuilt variants do not materially justify their added complexity.
- Failure interpretation:
  - If a larger or more fragmented model dominates, the simplicity story collapses.
- Table / figure target:
  - Appendix ablation table
- Priority: NICE-TO-HAVE

### Block 5: Failure Analysis and Diagnostics
- Claim tested: not a contribution claim; this block protects paper honesty
- Why this block exists:
  - It shows where repair still fails and helps shape discussion and future work.
- Dataset / split / task:
  - Collect worst-case examples from Block 3
- Compared systems:
  - `ResidualNO+SoftPhys`
  - `ResidualNO+FluxRepair`
- Metrics:
  - Error maps
  - Local conservation maps
  - Boundary artifacts
- Setup details:
  - Focus on a few representative failures
- Success criterion:
  - Produce interpretable qualitative evidence for what the repair helps and what it does not fix
- Failure interpretation:
  - If no diagnostic difference is visible, the mechanism may be too weak or too opaque
- Table / figure target:
  - Discussion figure or appendix case study
- Priority: NICE-TO-HAVE

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 Sanity | Validate data, metrics, and one-batch overfit | `R001-R004` | Data tensors, Darcy residual metric, and repair op are numerically stable | Low | Metric bugs or repair instability |
| M1 Baselines | Reproduce strongest direct and residual baselines | `R005-R010` | Residual baseline should beat or match direct baseline before investing in repair | Medium | Weak backbone or bad data design masks the story |
| M2 Main Method | Train repaired residual model on anchor setting | `R011-R013` | Must beat `ResidualNO+SoftPhys` on physical metrics | Medium | Repair acts like a no-op or destroys accuracy |
| M3 Novelty Decision | Run decisive ablations for mechanism isolation | `R014-R020` | If repair does not clearly beat soft losses, reconsider paper positioning | Medium | Results too incremental |
| M4 Robustness | Stress-test high contrast and varying BC | `R021-R026` | Need clear hard-regime advantage | Medium to High | OOD gap too small or training too narrow |
| M5 Polish | Figures, extra seeds, simplicity appendix | `R027-R032` | Only after main story is secure | Medium | Nice-to-have runs consume budget |

## Compute and Data Budget
- Total estimated GPU-hours:
  - M0: 2-4 GPU-hours
  - M1: 12-18 GPU-hours
  - M2: 10-15 GPU-hours
  - M3: 12-18 GPU-hours
  - M4: 12-20 GPU-hours
  - M5: 8-12 GPU-hours
  - Total must-run target: roughly 48-75 GPU-hours
- Data preparation needs:
  - New 2D Darcy dataset generator
  - Contrast-controlled train/val/test splits
  - Boundary-condition family generation
- Human evaluation needs:
  - None required
- Biggest bottleneck:
  - Implementing a meaningful differentiable repair module that is stronger than soft penalties but not numerically destructive

## Risks and Mitigations
- Risk: The repaired output can improve conservation while hurting pressure reconstruction too much.
- Mitigation: Treat repair strength as a controlled module, and compare before/after repair explicitly.

- Risk: Reviewers cite residual-connected Darcy reconstruction as too close.
- Mitigation: Make Block 2 and Block 3 the center of the paper, showing the value of explicit local repair over residual reconstruction alone.

- Risk: Soft-physics baselines already achieve similar conservation.
- Mitigation: Use harder high-contrast and varying-BC regimes where structural correction should matter more.

- Risk: The first backbone is too weak, making all conclusions suspect.
- Mitigation: Use one credible true 2D operator backbone before exploring mechanism-heavy variants.

## Final Checklist
- [ ] Main paper tables are covered
- [ ] Novelty is isolated
- [ ] Simplicity is defended
- [ ] Frontier contribution is explicitly not overstated
- [ ] Nice-to-have runs are separated from must-run runs
