# Experiment Tracker & Execution Checklist

## Overview
PCR-NO 方法的完整实验追踪表。从 S0 sanity check 到 S4 robustness，共 5 stages × 20+ runs。

---

## Executive Summary

| Stage | Label | Purpose | GPU-Days | Status | Target Date |
|-------|-------|---------|----------|--------|-------------|
| **S0** | Sanity | Data pipeline validation | 0.5 | 🔴 Pending | — |
| **S1** | Baseline | FNO-residual reproduction | 1.0 | 🔴 Pending | — |
| **S2** | Main | PCR-NO full implementation | 1.5 | 🔴 Pending | — |
| **S3** | Ablation | Eq-wise & constraint variants | 0.5 | 🔴 Pending | — |
| **S4** | Robustness | Strong-coupling OOD & long-term | 1.0 | 🔴 Pending | — |
| **Total** | — | — | **4.5 GPU-days** | — | — |

---

## Stage S0: Sanity Check (Data & Boundary Encoding)

### Goals
- ✅ Verify data loading, interpolation, boundary representation
- ✅ Test gradient computation for conservation loss
- ✅ Confirm forward pass & loss computation on small batch

### Runs

| Run | Experiment | Config | Expected | Status |
|-----|-----------|--------|----------|--------|
| **S0.1** | Data loading & shape | 64×64 grid, 100 samples | shapes OK, no NaN | 🔴 |
| **S0.2** | Boundary encoding | Dirichlet/Neumann mixed | BC values matched | 🔴 |
| **S0.3** | Gradient computation | ∇u from FNO output | values in [0,1] | 🔴 |
| **S0.4** | Conservation loss | mass/energy balance | loss < 0.1 baseline | 🔴 |
| **S0.5** | Checkpoint save/load | model state + optimizer | resume without error | 🔴 |

### Success Criteria
- All 5 runs pass without NaN/Inf
- Conservation loss computable
- Checkpoint cycle (save→load→eval) works

### Output
- ✅ `logs/S0_sanity_report.json` — 5 runs 的指标汇总

---

## Stage S1: Baseline (FNO-Residual Reproduction)

### Goals
- ✅ Reproduce FNO-residual on same dataset as PCR-NO
- ✅ Establish L2, residual, BC, conservation baselines
- ✅ Verify training curve stability over 100 epochs

### Runs

| Run | Experiment | Config | Expected L2 | Status |
|-----|-----------|--------|---------|--------|
| **S1.1** | FNO-residual (α=1.0) | Δt=0.001, bs=32 | rel-L2 < 2% | 🔴 |
| **S1.2** | w/ L2 loss only | no physics loss | rel-L2 ≈ 2-3% | 🔴 |
| **S1.3** | w/ PDE residual loss | α_pde=1.0 | rel-L2 ≈ 1.5-2% | 🔴 |
| **S1.4** | w/ BC loss | α_bc=1.0 | rel-L2 ≈ 1.8-2.5% | 🔴 |
| **S1.5** | Combined (α=0.7,0.2,0.1) | default | rel-L2 ≈ 1-1.5% ⭐ | 🔴 |

### Success Criteria
- S1.5 achieves rel-L2 < 1.5% (L2 error in reference)
- Training curve smooth, no divergence after epoch 20
- All metrics (L2, residual, BC, conservation) tracked

### Output
- ✅ `logs/S1_baseline_curves.pkl` — 5 runs 的训练曲线
- ✅ `checkpoints/S1_baseline_best.pt` — 最优 checkpoint

---

## Stage S2: Main (PCR-NO Implementation)

### Goals
- ✅ Implement PCR-NO full version (subspace constraint + eq-wise heads)
- ✅ Test against S1.5 baseline
- ✅ Measure improvement on all 4 metrics (L2, residual, BC, conservation)

### Runs

| Run | Experiment | Config | Target L2 | Status |
|-----|-----------|--------|---------|--------|
| **S2.1** | PCR-NO v1 | proj_dim=32, coupling=1.0 | -3% vs S1.5 | 🔴 |
| **S2.2** | PCR-NO v2 | proj_dim=64 | -4% vs S1.5 | 🔴 |
| **S2.3** | PCR-NO w/ eq-wise | + eq_wise_heads=True | -4.5% vs S1.5 | 🔴 |
| **S2.4** | PCR-NO tuned | best hyperparams from v1-v3 | -5% vs S1.5 ⭐ | 🔴 |
| **S2.5** | PCR-NO w/ gate (optional) | + confidence_gate=True | -5.2% vs S1.5 | 🔴 |

### Metrics per Run
- Rel-L2 error
- PDE residual: ||F(u) - f||_L2 / ||f||_L2
- BC violation: ||u - u_bc||_boundary
- Conservation error: |∫u_predicted - ∫u_reference| / |∫u_reference|

### Success Criteria (C1: PCR Mechanism Effective)
- **S2.4** achieves -4% to -5% L2 improvement over S1.5 ✅
- Conservation error decreases by ≥40% ✅
- BC violation decreases by ≥30% ✅
- All metrics improve simultaneously (no trade-off) ✅

### Output
- ✅ `logs/S2_main_curves.pkl` — 5 runs 的训练曲线 + 测试指标
- ✅ `checkpoints/S2_best.pt` — 最优 PCR-NO checkpoint
- ✅ `results/S2_metric_comparison.csv` — 多指标对比表

---

## Stage S3: Ablation (Equation-wise & Constraint Variants)

### Goals
- ✅ Isolate contribution of eq-wise heads (C2 via ablation)
- ✅ Verify subspace constraint vs physics-loss-only
- ✅ Same-parameter baseline to prevent "width wins"

### Runs

| Run | Experiment | Config | Control | Expected | Status |
|-----|-----------|--------|---------|----------|--------|
| **S3.1** | PCR-NO w/o eq-wise | subspace only | — | -3% vs S1.5 | 🔴 |
| **S3.2** | Physics-loss-only | no subspace, α tuned | S1.5 equivalent | -1% vs S1.5 | 🔴 |
| **S3.3** | Same-param FNO-wide | 参数量 = S2.4 | S2.4 params | -2% vs S1.5 | 🔴 |
| **S3.4** | Subspace only (proj_dim=128) | no eq-wise, wide proj | — | -3.5% vs S1.5 | 🔴 |
| **S3.5** | Full PCR (S2.4 replica) | reference full model | — | -4.5% vs S1.5 ⭐ | 🔴 |

### Analysis (Evidence for C2)
- **S3.1 vs S3.5**: eq-wise contribution ≈ -1% to -1.5% L2
- **S3.2 vs S3.1**: subspace advantage ≈ -2% vs physics-loss-only
- **S3.3 vs S3.5**: subspace+eq-wise > parameter scaling ✅

### Success Criteria (C2: Eq-wise Enhances Coupled Modeling)
- S3.1 (no eq-wise) degrades by ≤-1% vs S3.5 ✅
- S3.2 (physics-loss) shows <-2% vs S3.5 ✅
- S3.3 (wider FNO) does NOT match S3.5 (-2% gap) ✅
- → **Eq-wise & subspace mechanisms are real, not just parameter scaling**

### Output
- ✅ `logs/S3_ablation_curves.pkl`
- ✅ `results/S3_ablation_analysis.md` — ablation 结果解读

---

## Stage S4: Robustness (Strong-Coupling OOD & Long-term)

### Goals
- ✅ Test PCR-NO under strong-coupling parameters (OOD)
- ✅ Verify long-term rollout stability (C3)
- ✅ Compare vs S1.5 baseline

### Runs

| Run | Experiment | Config | Training Param | Test Param | Expected | Status |
|-----|-----------|--------|--------|--------|---------|--------|
| **S4.1** | Weak coupling | trained & tested on ε=0.1 | 0.1 | 0.1 | baseline | 🔴 |
| **S4.2** | Strong coupling (ID) | trained & tested on ε=0.5 | 0.5 | 0.5 | -3% vs weak | 🔴 |
| **S4.3** | Strong coupling OOD | trained on ε=0.1, test ε=0.5 | 0.1 | 0.5 | L2 rise ≤ +10% | 🔴 |
| **S4.4** | Long rollout (100 steps) | 100 time-steps inference | 0.1 | 0.1 | error <5% at T=100 | 🔴 |
| **S4.5** | Strong + long (ε=0.5, 100 steps) | combined OOD + rollout | 0.1 | 0.5 + 100 | error <8% at T=100 ⭐ | 🔴 |

### Metrics
- Rel-L2 at each time-step t
- Integrated error over rollout: ∫_0^T ||e(t)||_L2 dt
- Failure detection: if L2 > 50%, mark as "diverged"

### Success Criteria (C3: Generalization & Stability)
- **S4.2**: Strong coupling in-distribution degrades gracefully ≤-5% vs weak ✅
- **S4.3 (PCR-NO)**: OOD coupling rise ≤+10% L2 vs in-distribution ✅ (vs baseline >> +30%)
- **S4.4 (PCR-NO)**: Long rollout at T=100 → error <5% ✅
- **S4.5 (PCR-NO)**: Strong + long → error <8% (vs baseline >15%) ✅

### Output
- ✅ `logs/S4_robustness_curves.pkl` — 5 runs 的时间演化曲线
- ✅ `results/S4_ood_analysis.csv` — OOD 参数扫描结果
- ✅ `results/S4_rollout_comparison.md` — long-term 稳定性对比

---

## Summary: Evidence Mapping

| Claim | Experiment | Metric | Target | Pass? |
|-------|-----------|--------|--------|-------|
| **C1: PCR mechanism effective** | S2 | Multi-metric improvement (L2, residual, BC, conservation) | All ↑ | 🔴 |
| **C2: Eq-wise enhances coupled modeling** | S3 | Ablation isolation: eq-wise contribution | -1 to -1.5% | 🔴 |
| **C3: Robust generalization** | S4.2-S4.5 | OOD coupling + long rollout | L2 rise ≤+10%, long rollout <8% | 🔴 |
| **Overall paper-ready** | S0-S4 | No failures, clear evidence trail | All stages pass | 🔴 |

---

## Running Instructions

### Prerequisites
```bash
# 1. Setup environment
conda create -n pcr-no python=3.10
conda install pytorch::pytorch pytorch::torchvision pytorch::torchaudio -c pytorch
pip install -r requirements.txt

# 2. Prepare data
python scripts/prepare_coupled_pde.py \
  --dataset ADR \
  --grid-size 64 \
  --num-samples 500 \
  --output-dir data/

# 3. Create output directories
mkdir -p logs checkpoints results
```

### Stage-by-Stage Execution

#### S0: Sanity (< 5 min on CPU)
```bash
python scripts/run_sanity_check.py --config configs/S0_sanity.yaml
```

#### S1: Baseline (1 GPU-day)
```bash
python scripts/train_baseline.py \
  --model FNO-residual \
  --config configs/S1_baseline.yaml \
  --run-id S1.1
```

#### S2: Main (1.5 GPU-day)
```bash
python scripts/train_pcr_no.py \
  --config configs/S2_main.yaml \
  --run-id S2.4 \
  --checkpoint checkpoints/S1_baseline_best.pt
```

#### S3: Ablation (0.5 GPU-day)
```bash
for run in S3.1 S3.2 S3.3 S3.4 S3.5; do
  python scripts/train_ablation.py \
    --config configs/S3_ablation.yaml \
    --run-id $run
done
```

#### S4: Robustness (1 GPU-day)
```bash
python scripts/eval_robustness.py \
  --config configs/S4_robustness.yaml \
  --checkpoint checkpoints/S2_best.pt
```

### Monitoring & Logging
```bash
# Real-time log monitoring
tail -f logs/training_S2.4.log

# TensorBoard visualization
tensorboard --logdir logs/

# Result aggregation
python scripts/aggregate_results.py --output-dir results/
```

---

## Key Milestones

- [ ] **Milestone 1** (S0 Pass): Data pipeline validated
- [ ] **Milestone 2** (S1 Pass): FNO-residual baseline reproduced
- [ ] **Milestone 3** (S2 Pass): PCR-NO achieves target metrics (C1 ✅)
- [ ] **Milestone 4** (S3 Pass): Ablations isolated & analyzed (C2 ✅)
- [ ] **Milestone 5** (S4 Pass): Robustness & generalization proven (C3 ✅)

---

## Known Risks & Contingencies

### Risk 1: Subspace Parameterization Fails
- **Mitigation**: Fallback to projection matrix (rank-limited)
- **Contingency**: Skip S2-S3, re-frame as empirical residual filtering

### Risk 2: Eq-wise Heads Diverge Training
- **Mitigation**: Reduce learning rate, add gradient clipping
- **Contingency**: Merge heads in later epochs, ablate S3 component

### Risk 3: OOD Coupling Parameters → Catastrophic Forgetting
- **Mitigation**: Data augmentation during training (ε ∈ [0.1, 0.5])
- **Contingency**: Fine-tune on OOD subset, report lower confidence

---

## Output File Locations

```
logs/
  ├── S0_sanity_report.json
  ├── S1_baseline_curves.pkl
  ├── S2_main_curves.pkl
  ├── S3_ablation_curves.pkl
  ├── S4_robustness_curves.pkl

checkpoints/
  ├── S1_baseline_best.pt
  ├── S2_best.pt
  ├── S2_final.pt

results/
  ├── S2_metric_comparison.csv
  ├── S3_ablation_analysis.md
  ├── S4_ood_analysis.csv
  ├── S4_rollout_comparison.md
  ├── FINAL_RESULTS.md ← paper figures sourced from here
```

---

## Progress Tracking

**Last Updated**: 2026-04-25 14:20 UTC  
**Status**: Ready for S0 launch  
**Owner**: [You]

---

**Next**: Execute S0-S1 via `/experiment-bridge` or direct launch.
