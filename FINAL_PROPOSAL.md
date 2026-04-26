# Research Proposal: Physics-Consistent Residual Neural Operator for Coupled PDE Coarse-to-Fine Correction

## Problem Anchor
- Bottom-line problem: 在耦合 PDE 中，用低成本粗求解器获得可接受精度的精细解替代。 
- Must-solve bottleneck: 纯残差回归虽降误差，但经常破坏守恒与边界一致性，长时/强耦合下不稳定。 
- Non-goals: 不追求替代全部数值求解器；不做超大模型堆叠。 
- Constraints: 中等算力；优先 2D 可复现实验；保持与现有 FNO 基线可比。 
- Success condition: 同等预算下，相比残差基线显著降低误差并降低守恒/边界违约。 

## Method Thesis

将 NO 残差预测限制到“守恒/边界一致子空间”，并进行方程级分块残差建模，以最小机制代价提升耦合 PDE 的精度与稳定性。 

## Contribution Focus
- Dominant contribution: 物理一致残差子空间约束（PCR）。 
- Supporting contribution: 方程级残差分解头（Eq-wise residual heads）。 
- Explicit non-contributions: 新 backbone、大规模多任务统一框架。 

## Proposed Method

### Complexity Budget
- Frozen/reused backbone: 标准 FNO/U-FNO 编码-解码骨干。 
- New trainable components: (1) 子空间投影/约束层；(2) 方程分块输出头。 
- Not used: 额外 teacher 模型、扩散式重建、多阶段蒸馏。 

### System Overview
1. 粗求解器输出 `u_H`。 
2. 插值到细网格 `u~_H = I(u_H)`。 
3. NO 输出初始残差 `r_raw`。 
4. 约束映射 `r = P_cons,bc(r_raw; a,BC)`，投影到守恒/边界一致子空间。 
5. `u_hat = u~_H + r`。 

### Core Mechanism
- Input: `u~_H`, PDE 参数、系数场、源项、坐标与边界编码。 
- Output: 多变量耦合残差 `r`。 
- Training signal: `L = 位1 L_res + 位2 L_sol + 位3 L_pde + 位4 L_bc + 位5 L_cons`。 
- Novelty: 非“学更复杂残差”，而是“学物理可行残差”。 

### Training Plan
- Stage 1: 训练无约束残差头（快速收敛到可用精度）。 
- Stage 2: 引入子空间约束与方程分块，联合微调。 
- Stage 3: 针对强耦合参数区间 curriculum 强化。 

### Failure Modes and Diagnostics
- 失败1：约束过强导致欠拟合。诊断：L2 降不下去但物理指标优。 
- 失败2：分块头耦合不足。诊断：单变量好、耦合量差。 
- 失败3：跨分辨率退化。诊断：训练分辨率外误差激增。 

### Novelty and Elegance Argument

相较 RELift/MgFNO 的 coarse-to-fine 框架，本方案把贡献聚焦在“残差可行域建模”，避免系统堆叠，贡献边界清晰，便于审稿验证。 
