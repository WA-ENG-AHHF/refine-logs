# Research Proposal: Flux-Repaired Residual Neural Operator for High-Contrast Darcy Pressure Super-Resolution

## Problem Anchor
- Bottom-line problem: In high-contrast heterogeneous porous media, how can we recover a fine-grid pressure field from a coarse-grid numerical solution without paying full fine-grid solve cost?
- Must-solve bottleneck: Standard neural operator super-resolution methods may reduce pressure error, but they often fail to preserve local mass balance and flux consistency under strong coefficient heterogeneity and varying boundary conditions.
- Application setting: 2D Darcy flow in porous media, with pressure reconstruction relevant to subsurface flow, reservoir pressure estimation, and heterogeneous diffusion media.
- Non-goals: We do not aim to replace full PDE solvers end to end, solve arbitrary geometries first, or claim novelty from "applying FNO to Darcy."
- Constraints: Moderate compute budget, 2D regular-grid first implementation, direct comparability to FNO/PINO-style baselines, and a clear novelty delta against recent residual-connected Darcy reconstruction work.
- Success condition: At matched budget, our method must improve physical fidelity metrics, especially local mass balance and flux mismatch, without sacrificing core reconstruction accuracy.

## Method Thesis

Instead of predicting the full fine-grid pressure directly, we predict only the coarse-to-fine residual and then apply a differentiable local flux repair module that explicitly corrects cell-wise conservation errors. The intended contribution is not "physics-informed loss for Darcy," but a residual reconstruction pipeline in which accuracy is learned and conservation is repaired structurally.

## Contribution Focus
- Dominant contribution: A differentiable cell-wise flux repair mechanism for coarse-to-fine Darcy pressure reconstruction.
- Supporting contribution: A residual-only reconstruction formulation specialized for high-contrast coefficient fields and varying boundary conditions.
- Experimental contribution: A harder evaluation regime centered on high contrast, BC variation, and conservation-sensitive metrics rather than Rel-L2 alone.
- Explicit non-contributions: A fundamentally new operator backbone, arbitrary-geometry generality, or novelty from super-resolution alone.

## Proposed Method

### PDE Setting
We study the 2D Darcy equation:

\[
-\nabla \cdot (k(x,y)\nabla p(x,y)) = f(x,y)
\]

where:
- `k(x,y)` is a heterogeneous permeability or conductivity field
- `p(x,y)` is the pressure field
- `f(x,y)` is a source term

### System Overview
1. Solve the PDE on a coarse grid to obtain `p_H`.
2. Interpolate the coarse solution to the fine grid: `p~_H = I(p_H)`.
3. Encode `k`, `p~_H`, `f`, boundary signals, and coordinates.
4. Use a neural operator backbone to predict a raw residual correction `r_raw`.
5. Form the preliminary prediction `p_raw = p~_H + r_raw`.
6. Apply a differentiable local flux repair module to obtain `p_hat`, reducing local conservation defects while respecting boundary conditions.

### Core Mechanism
- Input channels: `k`, `p~_H`, `f`, boundary mask or boundary values, `x`, `y`
- Output target: residual correction `r = p_fine - p~_H`
- Final prediction: `p_hat = Repair(p~_H + r_raw; k, f, BC)`
- Intended novelty: We do not rely only on soft PDE penalties. We explicitly repair conservation defects after residual prediction.

### Why This Is Not Just Another Physics Loss
- A soft PDE residual loss encourages conservation on average.
- Our intended repair layer targets cell-wise local defects directly.
- This matters most under high-contrast coefficients, where pressure accuracy alone may hide physically invalid local flux behavior.

### Complexity Budget
- Frozen or standard backbone: a true 2D FNO or closely matched operator baseline
- New trainable or structured components:
  1. residual reconstruction head
  2. differentiable local flux repair module
- Not used in the first paper version: complex geometry handling, multi-stage teacher distillation, 3D extension, or large hybrid foundation backbones

## Claim Structure

### C1: Residual reconstruction is a better formulation than direct fine prediction for this task
- Why it matters: The coarse solver already captures low-frequency structure; learning only the correction should improve efficiency and stability.
- Minimum evidence: residual baseline beats direct baseline at matched parameter count and compute.

### C2: Explicit local flux repair improves physical validity beyond soft physics losses
- Why it matters: This is the main mechanism claim.
- Minimum evidence: our repair version significantly reduces local mass-balance error and flux mismatch compared with residual-only and residual-plus-soft-loss baselines.

### C3: The method is especially useful in harder regimes
- Why it matters: Without this, the method risks looking like an incremental easy-benchmark tweak.
- Minimum evidence: stronger gains under high-contrast coefficients and varying boundary conditions than in easy in-distribution settings.

## Experimental Plan Skeleton

### Primary Baselines
- Coarse interpolation only
- Direct FNO or U-Net style predictor
- Residual operator without physics
- Residual operator with soft PDE and BC losses
- Residual operator with local flux repair (ours)

### Primary Metrics
- Relative L2 pressure error
- Cell-wise local mass balance error
- Flux mismatch or divergence residual
- Boundary violation

### Stress Regimes
- Increasing coefficient contrast
- Boundary-condition variation
- OOD contrast range

### Minimum Convincing Table
The first decisive table should compare:
- direct prediction
- residual prediction
- residual + soft physics loss
- residual + flux repair

and should report both reconstruction accuracy and conservation-sensitive metrics.

## Failure Modes and Diagnostics
- Failure 1: Repair layer weakly changes outputs and collapses into a soft regularizer.
  - Diagnostic: conservation metrics stay close to residual-plus-loss baseline.
- Failure 2: Repair improves conservation but damages pressure accuracy too much.
  - Diagnostic: Rel-L2 worsens sharply while conservation improves.
- Failure 3: Gains appear only in easy settings.
  - Diagnostic: advantage disappears under truly high-contrast or varying-BC evaluation.
- Failure 4: Reviewers cite prior residual-connected Darcy reconstruction as overlap.
  - Diagnostic: our method does not clearly distinguish itself from residual-connected or two-stage reconstruction papers.

## Novelty Boundary

This proposal is likely not novel if framed as:
- Darcy super-resolution with neural operators
- physics-informed residual learning for Darcy
- coarse-to-fine reconstruction alone

This proposal is potentially publishable if framed as:
- coarse-to-fine Darcy reconstruction under high contrast
- with a differentiable local flux repair mechanism
- showing clear gains over soft-constraint methods in cell-wise conservation

## Closest Prior Work and Intended Delta
- FNO established Darcy as a neural operator benchmark.
- PINO established coarse data plus high-resolution PDE constraints.
- Boundary-aware elliptic operator methods cover BC-aware modeling.
- Recent residual-connected Darcy reconstruction work covers coarse-to-fine residual reasoning in subsurface flow.
- Local mass-conservative Darcy ROM work shows exact-constraint ideas in a reduced-order setting.

Our intended delta is the combination of:
1. residual-only coarse-to-fine reconstruction
2. explicit differentiable local flux repair at the fine scale
3. evaluation centered on high-contrast and BC-varying regimes

## Working Title Alternatives
- Flux-Repaired Residual Neural Operator for High-Contrast Darcy Pressure Super-Resolution
- Coarse-to-Fine Darcy Reconstruction with Differentiable Local Flux Repair
- Locally Conservative Residual Neural Operator for Heterogeneous Darcy Flow

## Current Recommendation

Proceed, but only with a narrow claim:

> We are not proposing a generic physics-informed Darcy neural operator. We are proposing a residual reconstruction pipeline with explicit local conservative repair, designed for high-contrast heterogeneous Darcy flow where soft losses alone are insufficient.
