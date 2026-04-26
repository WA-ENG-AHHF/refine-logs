# Experiment Plan

**Problem**: 耦合 PDE 粗解到精细解修正  
**Method Thesis**: 物理一致残差子空间约束 + 方程级残差分解 
**Date**: 2026-04-25

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| C1: PCR 机制提升精度且更物理一致 | 主贡献 | 相比残差基线，L2 与 PDE/BC/守恒指标同时提升 | B1, B2 |
| C2: 方程级分块在强耦合下更稳 | 支撑贡献 | 强耦合参数区间中稳定性与泛化优于 unified head | B3 |

## Blocks

### B1 主结果（Anchor）
- Dataset/Task: 2D 耦合 PDE 基准（固定几何 + 参数化边界/源项）。 
- Compare: Coarse-only, FNO-direct, FNO-residual, PCR-NO(ours)。 
- Metrics: Rel-L2, PDE residual, BC violation, conservation error。 
- Success: ours 在至少 3/4 指标领先且无显著 trade-off。 

### B2 贡献隔离（Novelty Isolation）
- Compare: FNO-residual + physics loss vs PCR-NO（同参数量同预算）。 
- Purpose: 证明收益来自“子空间约束机制”，非普通正则。 

### B3 耦合强度与泛化（Robustness）
- Split: 训练弱/中耦合，测试强耦合；并做跨分辨率测试。 
- Success: 强耦合下误差与违约增长斜率更低。 

### B4 简化检查（Simplicity Check）
- Compare: PCR-NO vs PCR-NO+额外复杂模块（如多级蒸馏）。 
- Goal: 证明简洁版本已足够，避免贡献扩散。 

## Run Order

1. **S0 Sanity**: 数据管线、插值正确性、单位/边界检查。 
2. **S1 Baseline**: 复现 FNO-residual。 
3. **S2 Main**: 训练 PCR-NO 主模型。 
4. **S3 Ablation**: B2/B4。 
5. **S4 Robustness**: B3 强耦合与跨分辨率。 

## Compute Budget (initial)

- S0-S1: 0.5-1 GPU-day  
- S2: 1-2 GPU-day  
- S3-S4: 1-2 GPU-day  
- Total target: 3-5 GPU-days（首轮）
